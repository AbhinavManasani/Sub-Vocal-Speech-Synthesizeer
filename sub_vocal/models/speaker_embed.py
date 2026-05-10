"""
GE2E Speaker Embedding
========================

Speaker identity conditioning for the decoder using Generalized End-to-End
loss-trained speaker embeddings. Disentangles speaker identity from content
so the decoder can focus on lip-content mapping.

Conditioning methods:
  - FiLM (Feature-wise Linear Modulation) — scale + shift per layer
  - Concatenation — append to encoder features
  - Addition — add directly to decoder input
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class SpeakerEncoder(nn.Module):
    """
    LSTM-based speaker encoder following the GE2E architecture.
    
    Extracts fixed-length speaker embeddings from variable-length mel spectrograms.
    Can be pretrained separately and frozen during lip-to-speech training.
    """

    def __init__(
        self,
        input_dim: int = 80,
        hidden_dim: int = 256,
        embed_dim: int = 256,
        num_layers: int = 3,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.proj = nn.Linear(hidden_dim, embed_dim)
        self.embed_dim = embed_dim
        logger.info(f"SpeakerEncoder: hidden={hidden_dim}, embed={embed_dim}, layers={num_layers}")

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Extract speaker embedding from mel spectrogram.
        
        Args:
            mel: [B, T, mel_dim] mel spectrogram
        Returns:
            embed: [B, embed_dim] L2-normalized speaker embedding
        """
        _, (hidden, _) = self.lstm(mel)
        embed = self.proj(hidden[-1])  # Last layer hidden state
        embed = F.normalize(embed, p=2, dim=-1)
        return embed


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) layer.
    
    Applies speaker-conditioned affine transform:
      output = gamma * input + beta
    
    where gamma and beta are predicted from the speaker embedding.
    """

    def __init__(self, feature_dim: int, condition_dim: int):
        super().__init__()
        self.gamma_proj = nn.Linear(condition_dim, feature_dim)
        self.beta_proj = nn.Linear(condition_dim, feature_dim)

        # Initialize to identity transform
        nn.init.ones_(self.gamma_proj.weight.data[:, 0])
        nn.init.zeros_(self.gamma_proj.bias.data)
        nn.init.zeros_(self.beta_proj.weight.data)
        nn.init.zeros_(self.beta_proj.bias.data)

    def forward(
        self, x: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: [B, T, D] features to modulate
            condition: [B, D_cond] conditioning vector (speaker embed)
        Returns:
            [B, T, D] modulated features
        """
        gamma = self.gamma_proj(condition).unsqueeze(1)  # [B, 1, D]
        beta = self.beta_proj(condition).unsqueeze(1)    # [B, 1, D]
        return gamma * x + beta


class SpeakerEmbedding(nn.Module):
    """
    Speaker conditioning module for the decoder.
    
    Wraps speaker encoding + conditioning method (FiLM/concat/add).
    During inference, can use a precomputed speaker embedding directly.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        feature_dim: int = 512,
        method: str = "film",
        num_film_layers: int = 6,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.method = method

        # Speaker encoder (can be frozen)
        self.encoder = SpeakerEncoder(embed_dim=embed_dim)

        # Conditioning layers
        if method == "film":
            self.film_layers = nn.ModuleList([
                FiLMLayer(feature_dim, embed_dim)
                for _ in range(num_film_layers)
            ])
        elif method == "concat":
            self.proj = nn.Linear(feature_dim + embed_dim, feature_dim)
        elif method == "add":
            self.proj = nn.Linear(embed_dim, feature_dim)
        else:
            raise ValueError(f"Unknown conditioning method: {method}")

        logger.info(f"SpeakerEmbedding: method={method}, embed_dim={embed_dim}")

    def encode(self, mel: torch.Tensor) -> torch.Tensor:
        """Extract speaker embedding from mel spectrogram."""
        return self.encoder(mel)

    def condition(
        self,
        features: torch.Tensor,
        speaker_embed: torch.Tensor,
        layer_idx: int = 0,
    ) -> torch.Tensor:
        """
        Apply speaker conditioning to features.
        
        Args:
            features: [B, T, D] features to condition
            speaker_embed: [B, embed_dim] speaker embedding
            layer_idx: Which FiLM layer to use (for multi-layer FiLM)
        Returns:
            [B, T, D] conditioned features
        """
        if self.method == "film":
            idx = min(layer_idx, len(self.film_layers) - 1)
            return self.film_layers[idx](features, speaker_embed)
        elif self.method == "concat":
            spk = speaker_embed.unsqueeze(1).expand(-1, features.shape[1], -1)
            return self.proj(torch.cat([features, spk], dim=-1))
        elif self.method == "add":
            return features + self.proj(speaker_embed).unsqueeze(1)
        return features
