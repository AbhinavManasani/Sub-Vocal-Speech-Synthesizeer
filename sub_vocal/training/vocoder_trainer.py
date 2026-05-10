"""
Vocoder Trainer (Phase 3)
===========================

Dedicated training loop for the HiFi-GAN neural vocoder.
Requires alternating optimization for the Generator and Discriminator,
using Least Squares GAN loss, Feature Matching loss, and Mel reconstruction loss.
"""

import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None
import logging
from tqdm import tqdm
from pathlib import Path
from typing import Dict

from .checkpoint import CheckpointManager

logger = logging.getLogger(__name__)


class VocoderTrainer:
    """GAN Trainer for HiFi-GAN Vocoder."""

    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
    ):
        self.device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        # Optimizers (AdamW)
        opt_cfg = config.get("optimizer", {"lr": 2e-4, "betas": [0.8, 0.99]})
        self.optim_g = torch.optim.AdamW(
            self.model.generator.parameters(), 
            lr=opt_cfg["lr"], betas=tuple(opt_cfg["betas"])
        )
        self.optim_d = torch.optim.AdamW(
            self.model.mpd.parameters(), 
            lr=opt_cfg["lr"], betas=tuple(opt_cfg["betas"])
        )

        # Schedulers (Exponential decay)
        self.sched_g = torch.optim.lr_scheduler.ExponentialLR(self.optim_g, gamma=0.999)
        self.sched_d = torch.optim.lr_scheduler.ExponentialLR(self.optim_d, gamma=0.999)

        # Training settings
        self.max_epochs = config.get("max_epochs", 3000)
        self.save_every = config.get("save_every_n_steps", 5000)
        self.log_every = config.get("log_every_n_steps", 100)
        
        # Loss weights
        self.lambda_fm = config.get("lambda_fm", 2.0)     # Feature matching
        self.lambda_mel = config.get("lambda_mel", 45.0)  # Mel reconstruction

        # Infrastructure
        self.ckpt_manager = CheckpointManager(
            save_dir=config.get("save_dir", "checkpoints/vocoder"),
            best_metric="val_mel_loss"
        )
        self.writer = SummaryWriter(log_dir=config.get("log_dir", "runs/vocoder")) if SummaryWriter else None
        self.global_step = 0
        self.epoch = 0

    def train(self, resume_from=None):
        if resume_from:
            self._load_checkpoint(resume_from)

        logger.info(f"Starting Vocoder Training on {self.device}")
        
        for epoch in range(self.epoch, self.max_epochs):
            self.epoch = epoch
            self._train_epoch()
            
    def _train_epoch(self):
        self.model.train()
        
        for batch in self.train_loader:
            audio_real = batch["audio"].unsqueeze(1).to(self.device)  # [B, 1, T]
            mel = batch["mel"].transpose(1, 2).to(self.device)        # [B, 80, T]

            # Generate fake audio
            audio_fake = self.model.generator(mel)
            
            # Match lengths
            if audio_real.size(2) > audio_fake.size(2):
                audio_real = audio_real[:, :, :audio_fake.size(2)]
            elif audio_fake.size(2) > audio_real.size(2):
                audio_fake = audio_fake[:, :, :audio_real.size(2)]

            # ========================================
            # 1. Train Discriminator
            # ========================================
            self.optim_d.zero_grad()
            
            # MPD outputs
            real_outs, fake_outs, _, _ = self.model.mpd(audio_real, audio_fake.detach())
            
            # LS-GAN Discriminator Loss (real -> 1, fake -> 0)
            loss_d = 0.0
            for r, f in zip(real_outs, fake_outs):
                loss_d += torch.mean((r - 1) ** 2) + torch.mean(f ** 2)
                
            loss_d.backward()
            self.optim_d.step()

            # ========================================
            # 2. Train Generator
            # ========================================
            self.optim_g.zero_grad()
            
            # MPD outputs (we need feature maps for Feature Matching)
            real_outs, fake_outs, real_fmaps, fake_fmaps = self.model.mpd(audio_real, audio_fake)
            
            # LS-GAN Generator Loss (fake -> 1)
            loss_g_adv = 0.0
            for f in fake_outs:
                loss_g_adv += torch.mean((f - 1) ** 2)
                
            # Feature Matching Loss (L1 distance of discriminator feature maps)
            loss_fm = 0.0
            for r_fmaps, f_fmaps in zip(real_fmaps, fake_fmaps):
                for r_map, f_map in zip(r_fmaps, f_fmaps):
                    loss_fm += F.l1_loss(r_map, f_map)
                    
            # Mel-Spectrogram Loss (L1 distance of mels)
            # We must re-extract mel from audio_fake for comparison
            mel_fake = self._extract_mel(audio_fake.squeeze(1))
            mel_target = mel.transpose(1, 2)
            if mel_fake.size(1) > mel_target.size(1):
                mel_fake = mel_fake[:, :mel_target.size(1), :]
            elif mel_target.size(1) > mel_fake.size(1):
                mel_target = mel_target[:, :mel_fake.size(1), :]
            loss_mel = F.l1_loss(mel_fake, mel_target)

            # Total Generator Loss
            loss_g_total = loss_g_adv + (self.lambda_fm * loss_fm) + (self.lambda_mel * loss_mel)
            
            loss_g_total.backward()
            self.optim_g.step()

            # ========================================
            # Logging & Checkpointing
            # ========================================
            if self.writer and self.global_step % self.log_every == 0:
                self.writer.add_scalar("train/loss_g_total", loss_g_total.item(), self.global_step)
                self.writer.add_scalar("train/loss_d", loss_d.item(), self.global_step)
                self.writer.add_scalar("train/loss_mel", loss_mel.item(), self.global_step)
                self.writer.add_scalar("train/loss_fm", loss_fm.item(), self.global_step)
                
            if self.global_step % self.save_every == 0:
                self._validate()
                self._save_checkpoint()
                
            self.global_step += 1
            
        # Step schedulers per epoch
        self.sched_g.step()
        self.sched_d.step()

    @torch.no_grad()
    def _validate(self):
        self.model.eval()
        val_mel_loss = 0.0
        
        # Take a sample for TensorBoard audio visualization
        sample_audio_real, sample_audio_fake = None, None
        
        for i, batch in enumerate(self.val_loader):
            audio_real = batch["audio"].unsqueeze(1).to(self.device)
            mel = batch["mel"].transpose(1, 2).to(self.device)
            
            audio_fake = self.model.generator(mel)
            mel_fake = self._extract_mel(audio_fake.squeeze(1))
            
            mel_target = mel.transpose(1, 2)
            if mel_fake.size(1) > mel_target.size(1):
                mel_fake = mel_fake[:, :mel_target.size(1), :]
            elif mel_target.size(1) > mel_fake.size(1):
                mel_target = mel_target[:, :mel_fake.size(1), :]
            
            val_mel_loss += F.l1_loss(mel_fake, mel_target).item()
            
            if i == 0:
                sample_audio_real = audio_real[0]
                sample_audio_fake = audio_fake[0]
                
        val_mel_loss /= len(self.val_loader)
        if self.writer:
            self.writer.add_scalar("val/loss_mel", val_mel_loss, self.global_step)
            
            if sample_audio_real is not None:
                self.writer.add_audio("val/audio_real", sample_audio_real, self.global_step, sample_rate=16000)
                self.writer.add_audio("val/audio_fake", sample_audio_fake, self.global_step, sample_rate=16000)
            
        self.model.train()
        return val_mel_loss

    def _extract_mel(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract mel-spectrogram dynamically from generated waveform."""
        import torchaudio
        transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_fft=640, hop_length=160, win_length=640, n_mels=80
        ).to(self.device)
        mel = transform(audio)
        return torch.log(torch.clamp(mel, min=1e-5)).transpose(1, 2)  # [B, T, 80]

    def _save_checkpoint(self):
        state = {
            "model_g": self.model.generator.state_dict(),
            "model_d": self.model.mpd.state_dict(),
            "optim_g": self.optim_g.state_dict(),
            "optim_d": self.optim_d.state_dict(),
            "epoch": self.epoch,
            "global_step": self.global_step
        }
        path = self.ckpt_manager.save_dir / f"vocoder_step{self.global_step}.pt"
        torch.save(state, path)
        logger.info(f"Saved vocoder checkpoint: {path}")

    def _load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location="cpu")
        self.model.generator.load_state_dict(ckpt["model_g"])
        self.model.mpd.load_state_dict(ckpt["model_d"])
        self.optim_g.load_state_dict(ckpt["optim_g"])
        self.optim_d.load_state_dict(ckpt["optim_d"])
        self.epoch = ckpt["epoch"]
        self.global_step = ckpt["global_step"]
        logger.info(f"Resumed vocoder from step {self.global_step}")
