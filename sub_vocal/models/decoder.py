"""
Transformer Mel Decoder
=========================

Autoregressive transformer decoder that produces mel spectrograms
from visual encoder features via cross-attention.

Architecture:
  - Prenet: 2-layer FC with dropout (mel frame embedding)
  - Transformer decoder layers with cross-attention to visual features
  - Linear projection to mel dimensions
  - Postnet: 5-layer 1D conv for mel refinement

Reference: Lip-to-Speech Synthesis in the Wild (ICASSP 2023)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class Prenet(nn.Module):
    """
    Decoder prenet — bottleneck layer for mel frame conditioning.
    Uses dropout even at inference for variability (following Tacotron 2).
    """

    def __init__(self, in_dim: int, hidden_dim: int = 256, out_dim: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        return x


class Postnet(nn.Module):
    """
    5-layer 1D convolutional postnet for mel spectrogram refinement.
    Adds residual detail to the decoder output.
    """

    def __init__(
        self,
        mel_dim: int = 80,
        channels: int = 512,
        kernel_size: int = 5,
        num_layers: int = 5,
    ):
        super().__init__()
        layers = []
        # First layer
        layers.append(nn.Sequential(
            nn.Conv1d(mel_dim, channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(channels),
            nn.Tanh(),
            nn.Dropout(0.5),
        ))
        # Middle layers
        for _ in range(num_layers - 2):
            layers.append(nn.Sequential(
                nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(channels),
                nn.Tanh(),
                nn.Dropout(0.5),
            ))
        # Final layer — no activation, projects back to mel_dim
        layers.append(nn.Sequential(
            nn.Conv1d(channels, mel_dim, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(mel_dim),
            nn.Dropout(0.5),
        ))
        self.layers = nn.ModuleList(layers)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel: [B, T, mel_dim]
        Returns:
            Refined mel residual [B, T, mel_dim]
        """
        x = mel.transpose(1, 2)  # [B, mel_dim, T]
        for layer in self.layers:
            x = layer(x)
        return x.transpose(1, 2)  # [B, T, mel_dim]


class MelDecoder(nn.Module):
    """
    Transformer-based mel spectrogram decoder with cross-attention.
    
    Takes visual features from the encoder and autoregressively
    generates mel spectrogram frames.
    
    Input:  encoder_out [B, T_enc, D], target mel [B, T_dec, mel_dim]
    Output: predicted mel [B, T_dec, mel_dim], postnet mel [B, T_dec, mel_dim]
    """

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        mel_dim: int = 80,
        max_mel_len: int = 1500,
        prenet_dim: int = 256,
        postnet_channels: int = 512,
        postnet_kernel: int = 5,
        postnet_layers: int = 5,
        speaker_dim: int = 0,
    ):
        super().__init__()
        self.d_model = d_model
        self.mel_dim = mel_dim

        # Prenet for target mel conditioning
        self.prenet = Prenet(mel_dim, prenet_dim, d_model)

        # Positional encoding
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_len=max_mel_len)

        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers
        )

        # Speaker conditioning projection (if speaker_dim > 0)
        self.speaker_proj = None
        if speaker_dim > 0:
            self.speaker_proj = nn.Linear(speaker_dim, d_model)

        # Output projection: d_model → mel_dim
        self.mel_linear = nn.Linear(d_model, mel_dim)

        # Stop token prediction
        self.stop_linear = nn.Linear(d_model, 1)

        # Postnet for mel refinement
        self.postnet = Postnet(mel_dim, postnet_channels, postnet_kernel, postnet_layers)

        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"MelDecoder: d={d_model}, L={num_layers}, params={total_params/1e6:.1f}M")

    def forward(
        self,
        encoder_out: torch.Tensor,
        encoder_mask: Optional[torch.Tensor],
        target_mel: torch.Tensor,
        target_mask: Optional[torch.Tensor] = None,
        speaker_embed: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Teacher-forced forward pass.
        
        Args:
            encoder_out: [B, T_enc, D] visual encoder features
            encoder_mask: [B, T_enc] bool mask for encoder
            target_mel: [B, T_dec, mel_dim] ground truth mel (shifted right)
            target_mask: [B, T_dec] bool mask for target
            speaker_embed: [B, speaker_dim] optional speaker embedding
            
        Returns:
            mel_out: [B, T_dec, mel_dim] pre-postnet mel
            mel_postnet: [B, T_dec, mel_dim] post-postnet mel
            stop_logits: [B, T_dec, 1] stop token logits
        """
        B, T_dec, _ = target_mel.shape

        # Prenet
        tgt = self.prenet(target_mel)  # [B, T_dec, d_model]

        # Add positional encoding
        tgt = self.pos_encoding(tgt)

        # Add speaker conditioning
        if speaker_embed is not None and self.speaker_proj is not None:
            spk = self.speaker_proj(speaker_embed)  # [B, d_model]
            tgt = tgt + spk.unsqueeze(1)

        # Causal mask for autoregressive decoding
        causal_mask = self._generate_causal_mask(T_dec, tgt.device)

        # Convert bool masks to key_padding_mask format (True = ignore)
        enc_key_padding = ~encoder_mask if encoder_mask is not None else None
        tgt_key_padding = ~target_mask if target_mask is not None else None

        # Transformer decoder
        decoder_out = self.transformer_decoder(
            tgt=tgt,
            memory=encoder_out,
            tgt_mask=causal_mask,
            memory_key_padding_mask=enc_key_padding,
            tgt_key_padding_mask=tgt_key_padding,
        )  # [B, T_dec, d_model]

        # Mel projection
        mel_out = self.mel_linear(decoder_out)  # [B, T_dec, mel_dim]

        # Postnet refinement
        mel_postnet = mel_out + self.postnet(mel_out)

        # Stop prediction
        stop_logits = self.stop_linear(decoder_out)  # [B, T_dec, 1]

        return mel_out, mel_postnet, stop_logits

    def inference(
        self,
        encoder_out: torch.Tensor,
        encoder_mask: Optional[torch.Tensor] = None,
        speaker_embed: Optional[torch.Tensor] = None,
        max_len: int = 1500,
        stop_threshold: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Autoregressive inference (no teacher forcing).
        
        Returns:
            mel_postnet: [B, T_gen, mel_dim]
            stop_probs: [B, T_gen]
        """
        B = encoder_out.shape[0]
        device = encoder_out.device

        # Start with zero mel frame
        mel_input = torch.zeros(B, 1, self.mel_dim, device=device)
        generated_mels = []
        stop_probs = []

        for step in range(max_len):
            tgt = self.prenet(mel_input)
            tgt = self.pos_encoding(tgt)

            if speaker_embed is not None and self.speaker_proj is not None:
                spk = self.speaker_proj(speaker_embed)
                tgt = tgt + spk.unsqueeze(1)

            T = tgt.shape[1]
            causal_mask = self._generate_causal_mask(T, device)
            enc_key_padding = ~encoder_mask if encoder_mask is not None else None

            decoder_out = self.transformer_decoder(
                tgt=tgt, memory=encoder_out,
                tgt_mask=causal_mask,
                memory_key_padding_mask=enc_key_padding,
            )

            # Take only the last frame
            last_out = decoder_out[:, -1:, :]
            mel_frame = self.mel_linear(last_out)
            stop_logit = self.stop_linear(last_out)
            stop_prob = torch.sigmoid(stop_logit).squeeze(-1)

            generated_mels.append(mel_frame)
            stop_probs.append(stop_prob)

            # Check stop condition
            if (stop_prob > stop_threshold).all():
                break

            # Append to input for next step
            mel_input = torch.cat([mel_input, mel_frame], dim=1)

        mel_out = torch.cat(generated_mels, dim=1)
        mel_postnet = mel_out + self.postnet(mel_out)
        stop_probs = torch.cat(stop_probs, dim=1)

        return mel_postnet, stop_probs

    @staticmethod
    def _generate_causal_mask(size: int, device: torch.device) -> torch.Tensor:
        mask = torch.triu(torch.ones(size, size, device=device), diagonal=1)
        return mask.bool()


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, :x.size(1)])
