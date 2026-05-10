"""
3D-ResNet Visual Encoder
==========================

Spatio-temporal visual encoder for lip reading.

Architecture:
  - 3D convolutional frontend (Conv3D → BN → ReLU → MaxPool)
  - 2D ResNet-18 backbone (inflated from ImageNet/Kinetics 2D weights)
  - Temporal pooling or per-frame features

The frontend captures short-range temporal patterns (lip movement dynamics)
while the ResNet backbone extracts spatial features per frame.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class Conv3dFrontend(nn.Module):
    """
    3D convolutional frontend for initial spatiotemporal feature extraction.
    
    Processes raw lip sequences [B, T, 1, 96, 96] and produces
    spatially downsampled features [B, T, C, H', W'].
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 64,
        kernel_size: Tuple[int, int, int] = (5, 7, 7),
        stride: Tuple[int, int, int] = (1, 2, 2),
    ):
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[2] // 2),
            bias=False,
        )
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool3d(
            kernel_size=(1, 3, 3),
            stride=(1, 2, 2),
            padding=(0, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C_in, T, H, W] — note: Conv3D expects channel-first
        Returns:
            [B, C_out, T, H', W']
        """
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


class VisualEncoder(nn.Module):
    """
    Full visual encoder: 3D frontend + 2D ResNet backbone.
    
    Input:  [B, T, 1, 96, 96] grayscale lip crops
    Output: [B, T, D] per-frame feature vectors (D=512 for ResNet-18)
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained_2d: bool = True,
        frontend_channels: int = 64,
        frontend_kernel: Tuple[int, int, int] = (5, 7, 7),
        frontend_stride: Tuple[int, int, int] = (1, 2, 2),
        out_dim: int = 512,
        dropout: float = 0.1,
        lrw_pretrained_path: Optional[str] = None,
    ):
        super().__init__()
        self.out_dim = out_dim

        # 3D Frontend
        self.frontend = Conv3dFrontend(
            in_channels=1,
            out_channels=frontend_channels,
            kernel_size=frontend_kernel,
            stride=frontend_stride,
        )

        # 2D ResNet backbone (per-frame)
        self.backbone, backbone_dim = self._build_backbone(
            backbone, pretrained_2d, frontend_channels
        )

        # Project to output dimension
        self.proj = nn.Linear(backbone_dim, out_dim) if backbone_dim != out_dim else nn.Identity()
        self.dropout = nn.Dropout(dropout)

        # Load LRW Pretrained weights if provided
        if lrw_pretrained_path:
            self._load_lrw_pretrained(lrw_pretrained_path)

        # Positional encoding for temporal sequence
        self.pos_encoding = PositionalEncoding(out_dim, dropout=dropout)

        total_params = sum(p.numel() for p in self.parameters())
        logger.info(
            f"VisualEncoder: {backbone}, out_dim={out_dim}, "
            f"params={total_params / 1e6:.1f}M"
        )

    def _load_lrw_pretrained(self, checkpoint_path: str):
        """Loads weights from an LRW pre-trained checkpoint."""
        logger.info(f"Loading LRW pretrained weights from {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            
            # Handle cases where the state dict is nested
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
                
            # Filter out final classification layers (e.g., fc or projection)
            # We only want to load weights into frontend and backbone
            filtered_dict = {k: v for k, v in state_dict.items() if not k.startswith("fc.") and "proj" not in k}
            
            # strict=False allows ignoring missing/unexpected keys
            missing_keys, unexpected_keys = self.load_state_dict(filtered_dict, strict=False)
            logger.info(f"Loaded pretrained weights. Missing keys: {len(missing_keys)}, Unexpected keys: {len(unexpected_keys)}")
            
        except Exception as e:
            logger.error(f"Failed to load LRW pretrained weights from {checkpoint_path}: {e}")

    def _build_backbone(
        self, backbone_name: str, pretrained: bool, in_channels: int
    ) -> Tuple[nn.Module, int]:
        """Build the 2D ResNet backbone, optionally with pretrained weights."""
        if backbone_name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            resnet = models.resnet18(weights=weights)
            backbone_dim = 512
        elif backbone_name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            resnet = models.resnet50(weights=weights)
            backbone_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        # Replace first conv to accept frontend output channels
        resnet.conv1 = nn.Conv2d(
            in_channels, 64,
            kernel_size=7, stride=2, padding=3, bias=False,
        )

        # Remove the final FC layer — we want features, not classification
        layers = list(resnet.children())[:-2]  # Remove avgpool and fc
        backbone = nn.Sequential(*layers)

        return backbone, backbone_dim

    def forward(
        self,
        video: torch.Tensor,
        video_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through the visual encoder.
        
        Args:
            video: [B, T, 1, H, W] lip crop sequence
            video_mask: [B, T] boolean mask (True = valid)
            
        Returns:
            features: [B, T, D] per-frame visual features
            mask: [B, T] updated mask
        """
        B, T, C, H, W = video.shape

        # Frontend expects [B, C, T, H, W]
        x = video.permute(0, 2, 1, 3, 4)  # [B, 1, T, H, W]
        x = self.frontend(x)  # [B, 64, T, H', W']

        _, C_out, T_out, H_out, W_out = x.shape

        # Process each frame through the 2D backbone
        x = x.permute(0, 2, 1, 3, 4)  # [B, T, C, H', W']
        x = x.contiguous().view(B * T_out, C_out, H_out, W_out)

        x = self.backbone(x)  # [B*T, backbone_dim, h, w]
        x = F.adaptive_avg_pool2d(x, (1, 1))  # [B*T, backbone_dim, 1, 1]
        x = x.view(B, T_out, -1)  # [B, T, backbone_dim]

        # Project to output dimension
        x = self.proj(x)  # [B, T, out_dim]
        x = self.dropout(x)

        # Add positional encoding
        x = self.pos_encoding(x)

        # Update mask for temporal dimension change
        if video_mask is not None and T_out != T:
            video_mask = video_mask[:, :T_out]

        return x, video_mask


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal sequences."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
