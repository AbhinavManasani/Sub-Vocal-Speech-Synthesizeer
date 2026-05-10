"""
Multi-Task Loss
=================

Weighted combination of all training losses:
  - Mel L1 reconstruction (pre-postnet + post-postnet)
  - CTC phoneme supervision
  - SSIM structural similarity
  - Stop token BCE

Supports dynamic loss weighting and delayed CTC activation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import logging

from .mel_loss import MelReconstructionLoss
from .ssim_loss import SSIMLoss

logger = logging.getLogger(__name__)


class MultiTaskLoss(nn.Module):
    """
    Combined multi-task loss for lip-to-speech training.
    
    Total = w1*L1_mel + w2*L1_mel_postnet + w3*CTC + w4*SSIM + w5*stop
    """

    def __init__(
        self,
        mel_l1_weight: float = 1.0,
        mel_postnet_l1_weight: float = 1.0,
        ctc_weight: float = 0.1,
        ssim_weight: float = 0.5,
        stop_weight: float = 1.0,
        output_content_on: int = 0,
    ):
        super().__init__()
        self.mel_l1_weight = mel_l1_weight
        self.mel_postnet_l1_weight = mel_postnet_l1_weight
        self.ctc_weight = ctc_weight
        self.ssim_weight = ssim_weight
        self.stop_weight = stop_weight
        self.output_content_on = output_content_on

        self.mel_loss = MelReconstructionLoss(l1_weight=1.0)
        self.ssim_loss = SSIMLoss()

        logger.info(
            f"MultiTaskLoss: mel={mel_l1_weight}, postnet={mel_postnet_l1_weight}, "
            f"ctc={ctc_weight}, ssim={ssim_weight}, stop={stop_weight}"
        )

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        global_step: int = 0,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all losses.
        
        Args:
            outputs: Model forward outputs
            batch: Training batch with ground truth
            global_step: Current training step (for delayed activation)
            
        Returns:
            Dict with 'total_loss' and per-component losses
        """
        losses = {}
        total = torch.tensor(0.0, device=next(iter(outputs.values())).device)

        target_mel = batch.get("mel")
        mel_mask = batch.get("mel_mask")

        # ── Mel L1 (pre-postnet) ──
        if "mel_out" in outputs and target_mel is not None:
            mel_l1 = self.mel_loss(outputs["mel_out"], target_mel, mel_mask)
            losses["mel_l1"] = mel_l1
            total = total + self.mel_l1_weight * mel_l1

        # ── Mel L1 (post-postnet) ──
        if "mel_postnet" in outputs and target_mel is not None:
            mel_post_l1 = self.mel_loss(outputs["mel_postnet"], target_mel, mel_mask)
            losses["mel_postnet_l1"] = mel_post_l1
            total = total + self.mel_postnet_l1_weight * mel_post_l1

        # ── SSIM ──
        if self.ssim_weight > 0 and "mel_postnet" in outputs and target_mel is not None:
            ssim = self.ssim_loss(outputs["mel_postnet"], target_mel)
            losses["ssim"] = ssim
            total = total + self.ssim_weight * ssim

        # ── CTC (may be delayed) ──
        if (
            self.ctc_weight > 0
            and "ctc_log_probs" in outputs
            and global_step >= self.output_content_on
        ):
            ctc_loss = self._compute_ctc(outputs, batch)
            if ctc_loss is not None:
                losses["ctc"] = ctc_loss
                total = total + self.ctc_weight * ctc_loss

        # ── Stop token BCE ──
        if "stop_logits" in outputs and mel_mask is not None:
            stop_loss = self._compute_stop_loss(outputs["stop_logits"], mel_mask)
            losses["stop"] = stop_loss
            total = total + self.stop_weight * stop_loss

        losses["total_loss"] = total
        return losses

    def _compute_ctc(self, outputs, batch):
        """Compute CTC loss from log_probs and text targets."""
        log_probs = outputs["ctc_log_probs"]  # [B, T, V]
        texts = batch.get("text", [])

        if not texts or not any(texts):
            return None

        try:
            from sub_vocal.models.ctc_head import text_to_phonemes, phonemes_to_ids

            B = log_probs.shape[0]
            device = log_probs.device

            all_targets = []
            target_lengths = []
            input_lengths = []

            for i in range(B):
                if i < len(texts) and texts[i]:
                    phonemes = text_to_phonemes(texts[i])
                    ids = phonemes_to_ids(phonemes)
                    all_targets.extend(ids)
                    target_lengths.append(len(ids))
                else:
                    target_lengths.append(0)
                input_lengths.append(log_probs.shape[1])

            if not all_targets:
                return None

            targets = torch.tensor(all_targets, dtype=torch.long, device=device)
            target_lengths = torch.tensor(target_lengths, dtype=torch.long, device=device)
            input_lengths = torch.tensor(input_lengths, dtype=torch.long, device=device)

            return F.ctc_loss(
                log_probs.permute(1, 0, 2),
                targets, input_lengths, target_lengths,
                blank=0, reduction="mean", zero_infinity=True,
            )
        except Exception as e:
            logger.debug(f"CTC loss computation failed: {e}")
            return None

    def _compute_stop_loss(self, stop_logits, mel_mask):
        """Binary cross-entropy for stop token prediction."""
        min_len = min(stop_logits.shape[1], mel_mask.shape[1])
        logits = stop_logits[:, :min_len, 0]
        # Target: 1 at the last valid frame, 0 elsewhere
        target = torch.zeros_like(logits)
        for i in range(mel_mask.shape[0]):
            valid_len = mel_mask[i, :min_len].sum().long()
            if valid_len > 0:
                target[i, valid_len - 1] = 1.0
        return F.binary_cross_entropy_with_logits(logits, target)
