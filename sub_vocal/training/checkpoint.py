"""
Checkpoint Manager
====================

Save/load/resume training checkpoints with:
  - Best model tracking (by metric)
  - Keep-last-K pruning
  - Full training state (model, optimizer, scheduler, step, epoch)
"""

import os
import glob
import torch
import logging
from typing import Optional, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages saving, loading, and pruning of training checkpoints."""

    def __init__(
        self,
        save_dir: str = "checkpoints",
        keep_last_k: int = 5,
        save_best: bool = True,
        best_metric: str = "val_loss",
        best_mode: str = "min",
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_k = keep_last_k
        self.save_best = save_best
        self.best_metric = best_metric
        self.best_mode = best_mode
        self.best_value = float("inf") if best_mode == "min" else float("-inf")

    def save(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        epoch: int,
        global_step: int,
        metrics: Dict[str, float],
        tag: str = "latest",
    ) -> str:
        """Save a checkpoint and optionally update best."""
        state = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "epoch": epoch,
            "global_step": global_step,
            "metrics": metrics,
            "best_value": self.best_value,
        }

        # Save with step tag
        path = self.save_dir / f"checkpoint_step{global_step}.pt"
        torch.save(state, path)
        logger.info(f"Checkpoint saved: {path}")

        # Update best
        if self.save_best and self.best_metric in metrics:
            val = metrics[self.best_metric]
            is_better = (
                (self.best_mode == "min" and val < self.best_value) or
                (self.best_mode == "max" and val > self.best_value)
            )
            if is_better:
                self.best_value = val
                best_path = self.save_dir / "best.pt"
                torch.save(state, best_path)
                logger.info(
                    f"New best {self.best_metric}: {val:.4f} → {best_path}"
                )

        # Prune old checkpoints
        self._prune()
        return str(path)

    def load(
        self,
        path: str,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Any = None,
    ) -> Dict[str, Any]:
        """Load a checkpoint and restore state."""
        logger.info(f"Loading checkpoint: {path}")
        ckpt = torch.load(path, map_location="cpu")

        model.load_state_dict(ckpt["model_state_dict"])
        if optimizer and "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scheduler and ckpt.get("scheduler_state_dict"):
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if "best_value" in ckpt:
            self.best_value = ckpt["best_value"]

        return {
            "epoch": ckpt.get("epoch", 0),
            "global_step": ckpt.get("global_step", 0),
            "metrics": ckpt.get("metrics", {}),
        }

    def _prune(self):
        """Remove old checkpoints, keeping only the last K."""
        checkpoints = sorted(
            glob.glob(str(self.save_dir / "checkpoint_step*.pt")),
            key=os.path.getmtime,
        )
        while len(checkpoints) > self.keep_last_k:
            old = checkpoints.pop(0)
            os.remove(old)
            logger.debug(f"Pruned checkpoint: {old}")
