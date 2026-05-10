"""
Trainer — Main Training Loop
===============================

Handles the full training lifecycle:
  - Training loop with gradient accumulation
  - Periodic validation with metric computation
  - Checkpoint saving (every N steps + best model)
  - Early stopping
  - TensorBoard logging
  - Multi-phase support (Phase 1–4)
"""

import os
import time
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any
from pathlib import Path

from .checkpoint import CheckpointManager
from .early_stopping import EarlyStopping
from .scheduler import build_scheduler

logger = logging.getLogger(__name__)


class Trainer:
    """
    Generic trainer for all training phases.
    
    Usage:
        trainer = Trainer(model, train_loader, val_loader, loss_fn, config)
        trainer.train()
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        loss_fn: nn.Module,
        config: Dict[str, Any],
        evaluator=None,
    ):
        self.config = config
        train_cfg = config.get("training", {})

        # Device
        device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)
        self.model = model.to(self.device)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.evaluator = evaluator

        # Optimizer
        opt_cfg = train_cfg.get("optimizer", {})
        self.optimizer = self._build_optimizer(opt_cfg)

        # Scheduler
        sched_cfg = train_cfg.get("scheduler", {})
        self.scheduler = build_scheduler(self.optimizer, sched_cfg)

        # Training params
        self.max_epochs = train_cfg.get("max_epochs", 200)
        self.max_steps = train_cfg.get("max_steps")
        self.accumulate_grad = train_cfg.get("accumulate_grad", 1)
        self.gradient_clip = train_cfg.get("gradient_clip", 1.0)

        # Checkpointing
        ckpt_cfg = train_cfg.get("checkpoint", {})
        self.ckpt_manager = CheckpointManager(
            save_dir=ckpt_cfg.get("save_dir", "checkpoints"),
            keep_last_k=ckpt_cfg.get("keep_last_k", 5),
            save_best=ckpt_cfg.get("save_best", True),
            best_metric=ckpt_cfg.get("best_metric", "val_loss"),
            best_mode=ckpt_cfg.get("best_mode", "min"),
        )
        self.save_every = ckpt_cfg.get("save_every_n_steps", 5000)

        # Evaluation
        eval_cfg = train_cfg.get("eval", {})
        self.eval_every = eval_cfg.get("every_n_steps", 2000)
        self.eval_metrics = eval_cfg.get("metrics", ["stoi"])
        self.num_eval_samples = eval_cfg.get("num_eval_samples", 50)

        # Early stopping
        es_cfg = train_cfg.get("early_stopping", {})
        self.early_stopping = None
        if es_cfg.get("enabled", True):
            self.early_stopping = EarlyStopping(
                patience=es_cfg.get("patience", 15),
                min_delta=es_cfg.get("min_delta", 0.001),
                mode=es_cfg.get("mode", "min"),
                metric=es_cfg.get("metric", "val_loss"),
            )

        # Logging
        log_cfg = config.get("logging", {})
        self.log_every = log_cfg.get("log_every_n_steps", 100)
        self.writer = None
        if log_cfg.get("use_tensorboard", True):
            try:
                from torch.utils.tensorboard import SummaryWriter
                log_dir = Path(log_cfg.get("log_dir", "runs")) / config.get("experiment_name", "exp")
                self.writer = SummaryWriter(log_dir=str(log_dir))
                logger.info(f"TensorBoard: {log_dir}")
            except ImportError:
                logger.warning("tensorboard not installed, skipping")

        # State
        self.global_step = 0
        self.epoch = 0

        logger.info(
            f"Trainer ready: device={self.device}, epochs={self.max_epochs}, "
            f"batch_size={train_cfg.get('batch_size')}, "
            f"grad_accum={self.accumulate_grad}"
        )

    def _build_optimizer(self, cfg: Dict) -> torch.optim.Optimizer:
        params = filter(lambda p: p.requires_grad, self.model.parameters())
        opt_type = cfg.get("type", "adamw")
        lr = cfg.get("lr", 1e-4)
        betas = tuple(cfg.get("betas", [0.9, 0.98]))
        wd = cfg.get("weight_decay", 0.01)

        if opt_type == "adamw":
            return torch.optim.AdamW(params, lr=lr, betas=betas, weight_decay=wd)
        elif opt_type == "adam":
            return torch.optim.Adam(params, lr=lr, betas=betas)
        elif opt_type == "sgd":
            return torch.optim.SGD(params, lr=lr, weight_decay=wd, momentum=0.9)
        raise ValueError(f"Unknown optimizer: {opt_type}")

    def train(self, resume_from: Optional[str] = None):
        """Main training loop."""
        if resume_from:
            state = self.ckpt_manager.load(
                resume_from, self.model, self.optimizer, self.scheduler
            )
            self.epoch = state["epoch"]
            self.global_step = state["global_step"]
            logger.info(f"Resumed from step {self.global_step}, epoch {self.epoch}")

        logger.info("="*60)
        logger.info("TRAINING START")
        logger.info("="*60)

        for epoch in range(self.epoch, self.max_epochs):
            self.epoch = epoch
            train_metrics = self._train_epoch()

            logger.info(
                f"Epoch {epoch} complete | step={self.global_step} | "
                f"train_loss={train_metrics.get('train_loss', 0):.4f}"
            )

            # Check max steps
            if self.max_steps and self.global_step >= self.max_steps:
                logger.info(f"Reached max_steps={self.max_steps}, stopping")
                break

            # Early stopping
            if self.early_stopping and self.early_stopping.should_stop:
                logger.info("Early stopping triggered")
                break

        logger.info("="*60)
        logger.info("TRAINING COMPLETE")
        logger.info(f"Best {self.ckpt_manager.best_metric}: {self.ckpt_manager.best_value:.4f}")
        logger.info("="*60)

        if self.writer:
            self.writer.close()

    def _train_epoch(self) -> Dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(self.train_loader):
            # Move batch to device
            batch = self._to_device(batch)

            # Forward pass
            outputs = self.model(batch)
            losses = self.loss_fn(outputs, batch, self.global_step)
            loss = losses["total_loss"] / self.accumulate_grad

            # Backward pass
            loss.backward()

            # Gradient accumulation step
            if (batch_idx + 1) % self.accumulate_grad == 0:
                if self.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.gradient_clip
                    )
                self.optimizer.step()
                self.optimizer.zero_grad()
                if self.scheduler:
                    self.scheduler.step()
                self.global_step += 1

                # Logging
                if self.global_step % self.log_every == 0:
                    lr = self.optimizer.param_groups[0]["lr"]
                    logger.info(
                        f"[Step {self.global_step}] loss={losses['total_loss'].item():.4f} lr={lr:.2e}"
                    )
                    if self.writer:
                        for k, v in losses.items():
                            if isinstance(v, torch.Tensor):
                                self.writer.add_scalar(f"train/{k}", v.item(), self.global_step)
                        self.writer.add_scalar("train/lr", lr, self.global_step)

                # Evaluation
                if self.global_step % self.eval_every == 0 and self.val_loader:
                    val_metrics = self._validate()
                    if self.writer:
                        for k, v in val_metrics.items():
                            self.writer.add_scalar(f"val/{k}", v, self.global_step)

                    # Early stopping check
                    if self.early_stopping:
                        es_metric = self.early_stopping.metric
                        if es_metric in val_metrics:
                            self.early_stopping.step(val_metrics[es_metric])

                # Checkpointing
                if self.global_step % self.save_every == 0:
                    metrics = {"train_loss": losses["total_loss"].item()}
                    if self.val_loader:
                        metrics.update(self._validate())
                    self.ckpt_manager.save(
                        self.model, self.optimizer, self.scheduler,
                        self.epoch, self.global_step, metrics,
                    )

            epoch_loss += losses["total_loss"].item()
            num_batches += 1

            if self.max_steps and self.global_step >= self.max_steps:
                break
            if self.early_stopping and self.early_stopping.should_stop:
                break

        return {"train_loss": epoch_loss / max(num_batches, 1)}

    @torch.no_grad()
    def _validate(self) -> Dict[str, float]:
        """Run validation and compute metrics."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        all_metrics = {}

        for batch in self.val_loader:
            batch = self._to_device(batch)
            outputs = self.model(batch)
            losses = self.loss_fn(outputs, batch, self.global_step)
            total_loss += losses["total_loss"].item()
            num_batches += 1

            if num_batches >= self.num_eval_samples:
                break

        val_loss = total_loss / max(num_batches, 1)
        all_metrics["val_loss"] = val_loss

        # Run evaluator if available
        if self.evaluator:
            eval_results = self.evaluator.evaluate(self.model, self.val_loader, self.device)
            all_metrics.update(eval_results)

        logger.info(
            f"[Val Step {self.global_step}] " +
            " | ".join(f"{k}={v:.4f}" for k, v in all_metrics.items())
        )

        self.model.train()
        return all_metrics

    def _to_device(self, batch: Dict) -> Dict:
        """Move batch tensors to the training device."""
        moved = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                moved[k] = v.to(self.device)
            else:
                moved[k] = v
        return moved
