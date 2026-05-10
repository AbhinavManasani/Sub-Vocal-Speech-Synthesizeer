"""
HiFi-GAN Vocoder
==================

Neural vocoder for mel-to-waveform conversion.
Pretrained on LJSpeech, fine-tuned on LRS2/LRS3 audio in Phase 3.

Architecture: multi-receptive field fusion generator with 
multi-period + multi-scale discriminators.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

LRELU_SLOPE = 0.1


class ResBlock(nn.Module):
    """Residual block with dilated convolutions."""

    def __init__(self, channels: int, kernel_size: int = 3, dilations: List[int] = [1, 3, 5]):
        super().__init__()
        self.convs1 = nn.ModuleList()
        self.convs2 = nn.ModuleList()
        for d in dilations:
            self.convs1.append(nn.utils.parametrizations.weight_norm(
                nn.Conv1d(channels, channels, kernel_size, dilation=d,
                          padding=(kernel_size * d - d) // 2)
            ))
            self.convs2.append(nn.utils.parametrizations.weight_norm(
                nn.Conv1d(channels, channels, kernel_size, dilation=1,
                          padding=(kernel_size - 1) // 2)
            ))

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x


class HiFiGANGenerator(nn.Module):
    """HiFi-GAN generator: mel → waveform."""

    def __init__(
        self,
        in_channels: int = 80,
        upsample_rates: List[int] = [8, 8, 2, 2],
        upsample_kernel_sizes: List[int] = [16, 16, 4, 4],
        upsample_initial_channel: int = 512,
        resblock_kernel_sizes: List[int] = [3, 7, 11],
        resblock_dilation_sizes: List[List[int]] = [[1,3,5], [1,3,5], [1,3,5]],
    ):
        super().__init__()
        self.num_upsamples = len(upsample_rates)

        self.conv_pre = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(in_channels, upsample_initial_channel, 7, padding=3)
        )

        self.ups = nn.ModuleList()
        self.resblocks = nn.ModuleList()

        ch = upsample_initial_channel
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(nn.utils.parametrizations.weight_norm(
                nn.ConvTranspose1d(ch, ch // 2, k, stride=u, padding=(k - u) // 2)
            ))
            ch = ch // 2
            for j, (rk, rd) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(ResBlock(ch, rk, rd))

        self.conv_post = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(ch, 1, 7, padding=3)
        )

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel: [B, mel_dim, T] mel spectrogram
        Returns:
            audio: [B, 1, T_audio] waveform
        """
        x = self.conv_pre(mel)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = 0
            n_resblocks = len(self.resblocks) // self.num_upsamples
            for j in range(n_resblocks):
                xs += self.resblocks[i * n_resblocks + j](x)
            x = xs / n_resblocks
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x


class PeriodDiscriminator(nn.Module):
    """Single-period sub-discriminator."""

    def __init__(self, period: int):
        super().__init__()
        self.period = period
        self.convs = nn.ModuleList([
            nn.utils.parametrizations.weight_norm(nn.Conv2d(1, 32, (5, 1), (3, 1), (2, 0))),
            nn.utils.parametrizations.weight_norm(nn.Conv2d(32, 128, (5, 1), (3, 1), (2, 0))),
            nn.utils.parametrizations.weight_norm(nn.Conv2d(128, 512, (5, 1), (3, 1), (2, 0))),
            nn.utils.parametrizations.weight_norm(nn.Conv2d(512, 1024, (5, 1), (3, 1), (2, 0))),
            nn.utils.parametrizations.weight_norm(nn.Conv2d(1024, 1024, (5, 1), 1, (2, 0))),
        ])
        self.conv_post = nn.utils.parametrizations.weight_norm(nn.Conv2d(1024, 1, (3, 1), 1, (1, 0)))

    def forward(self, x):
        fmap = []
        B, C, T = x.shape
        pad = self.period - (T % self.period) if T % self.period != 0 else 0
        x = F.pad(x, (0, pad), "reflect")
        x = x.view(B, C, -1, self.period)
        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        return x.flatten(1, -1), fmap


class MultiPeriodDiscriminator(nn.Module):
    """Multi-Period Discriminator (MPD)."""

    def __init__(self, periods: List[int] = [2, 3, 5, 7, 11]):
        super().__init__()
        self.discriminators = nn.ModuleList([PeriodDiscriminator(p) for p in periods])

    def forward(self, y, y_hat):
        real_outs, fake_outs, real_fmaps, fake_fmaps = [], [], [], []
        for d in self.discriminators:
            r, rf = d(y)
            f, ff = d(y_hat)
            real_outs.append(r)
            fake_outs.append(f)
            real_fmaps.append(rf)
            fake_fmaps.append(ff)
        return real_outs, fake_outs, real_fmaps, fake_fmaps


class HiFiGANVocoder(nn.Module):
    """Complete HiFi-GAN vocoder with generator and discriminator."""

    def __init__(self, config: Optional[dict] = None):
        super().__init__()
        cfg = config or {}
        self.generator = HiFiGANGenerator(
            upsample_rates=cfg.get("upsample_rates", [8, 5, 2, 2]),
            upsample_kernel_sizes=cfg.get("upsample_kernel_sizes", [16, 10, 4, 4]),
            upsample_initial_channel=cfg.get("upsample_initial_channel", 512),
            resblock_kernel_sizes=cfg.get("resblock_kernel_sizes", [3, 7, 11]),
            resblock_dilation_sizes=cfg.get("resblock_dilation_sizes", [[1,3,5],[1,3,5],[1,3,5]]),
        )
        self.mpd = MultiPeriodDiscriminator(
            periods=cfg.get("mpd_periods", [2, 3, 5, 7, 11])
        )
        logger.info(f"HiFi-GAN: {sum(p.numel() for p in self.generator.parameters())/1e6:.1f}M gen params")

    def generate(self, mel: torch.Tensor) -> torch.Tensor:
        """Generate waveform from mel spectrogram."""
        if mel.dim() == 3 and mel.shape[-1] == 80:
            mel = mel.transpose(1, 2)  # [B, T, 80] → [B, 80, T]
        return self.generator(mel)
