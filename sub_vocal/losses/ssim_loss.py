"""
SSIM Loss for Mel Spectrograms
=================================

Structural Similarity Index as a loss for mel spectrogram generation.
Captures perceptual structure beyond per-element differences.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SSIMLoss(nn.Module):
    """1D SSIM loss operating on mel spectrogram columns (time-frequency patches)."""

    def __init__(self, window_size: int = 11, sigma: float = 1.5):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.register_buffer("window", self._create_window(window_size, sigma))

    @staticmethod
    def _create_window(size: int, sigma: float) -> torch.Tensor:
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        gauss = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        gauss = gauss / gauss.sum()
        return gauss.unsqueeze(0).unsqueeze(0)  # [1, 1, size]

    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predicted, target: [B, T, mel_dim]
        Returns:
            1 - SSIM (loss to minimize)
        """
        min_len = min(predicted.shape[1], target.shape[1])
        predicted = predicted[:, :min_len].transpose(1, 2)  # [B, mel_dim, T]
        target = target[:, :min_len].transpose(1, 2)

        B, C, T = predicted.shape
        if T < self.window_size:
            return torch.tensor(0.0, device=predicted.device)

        window = self.window.expand(C, -1, -1).to(predicted.device)

        mu_pred = F.conv1d(predicted, window, padding=self.window_size // 2, groups=C)
        mu_tgt = F.conv1d(target, window, padding=self.window_size // 2, groups=C)

        mu_pred_sq = mu_pred ** 2
        mu_tgt_sq = mu_tgt ** 2
        mu_cross = mu_pred * mu_tgt

        sigma_pred = F.conv1d(predicted ** 2, window, padding=self.window_size // 2, groups=C) - mu_pred_sq
        sigma_tgt = F.conv1d(target ** 2, window, padding=self.window_size // 2, groups=C) - mu_tgt_sq
        sigma_cross = F.conv1d(predicted * target, window, padding=self.window_size // 2, groups=C) - mu_cross

        C1, C2 = 0.01 ** 2, 0.03 ** 2
        ssim = ((2 * mu_cross + C1) * (2 * sigma_cross + C2)) / \
               ((mu_pred_sq + mu_tgt_sq + C1) * (sigma_pred + sigma_tgt + C2))

        return 1.0 - ssim.mean()
