"""
Mel Reconstruction Loss
=========================

L1 + L2 mel spectrogram reconstruction losses with optional
spectral convergence loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class MelReconstructionLoss(nn.Module):
    """Combined L1 + L2 mel spectrogram loss with masking support."""

    def __init__(self, l1_weight: float = 1.0, l2_weight: float = 0.0):
        super().__init__()
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight

    def forward(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            predicted: [B, T, mel_dim]
            target: [B, T, mel_dim]
            mask: [B, T] bool mask (True = valid)
        """
        # Align lengths
        min_len = min(predicted.shape[1], target.shape[1])
        predicted = predicted[:, :min_len]
        target = target[:, :min_len]
        if mask is not None:
            mask = mask[:, :min_len]

        loss = torch.tensor(0.0, device=predicted.device)

        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).float()
            n_valid = mask_expanded.sum().clamp(min=1)

            if self.l1_weight > 0:
                l1 = (F.l1_loss(predicted, target, reduction="none") * mask_expanded).sum() / n_valid
                loss = loss + self.l1_weight * l1

            if self.l2_weight > 0:
                l2 = (F.mse_loss(predicted, target, reduction="none") * mask_expanded).sum() / n_valid
                loss = loss + self.l2_weight * l2
        else:
            if self.l1_weight > 0:
                loss = loss + self.l1_weight * F.l1_loss(predicted, target)
            if self.l2_weight > 0:
                loss = loss + self.l2_weight * F.mse_loss(predicted, target)

        return loss
