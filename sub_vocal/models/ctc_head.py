"""
CTC Phoneme Head
==================

Auxiliary CTC head for content supervision during lip-to-speech training.
Projects visual encoder features to phoneme probabilities and trains
with CTC loss against phoneme transcriptions.

This forces the visual encoder to learn phonetically meaningful features,
addressing the one-to-many ambiguity problem (same lips → multiple valid sounds).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# English phoneme set (ARPAbet-like, 41 phonemes + blank)
PHONEME_VOCAB = [
    "<blank>", "<sos>", "<eos>",
    "AA", "AE", "AH", "AO", "AW", "AY",
    "B", "CH", "D", "DH",
    "EH", "ER", "EY",
    "F",
    "G",
    "HH",
    "IH", "IY",
    "JH",
    "K",
    "L",
    "M",
    "N", "NG",
    "OW", "OY",
    "P",
    "R",
    "S", "SH",
    "T", "TH",
    "UH", "UW",
    "V",
    "W",
    "Y",
    "Z", "ZH",
    "<space>",
]


class CTCPhonemeHead(nn.Module):
    """
    CTC projection head for phoneme-level content supervision.
    
    Input:  visual encoder features [B, T, D]
    Output: log-probabilities over phoneme vocabulary [B, T, V]
    """

    def __init__(
        self,
        input_dim: int = 512,
        vocab_size: int = 44,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        blank_id: int = 0,
    ):
        super().__init__()
        self.blank_id = blank_id
        self.vocab_size = vocab_size

        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, vocab_size),
        )

        logger.info(f"CTCPhonemeHead: vocab={vocab_size}, blank_id={blank_id}")

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Project features to phoneme log-probabilities.
        
        Args:
            features: [B, T, D] encoder output
        Returns:
            log_probs: [B, T, V] log-softmax over vocab
        """
        logits = self.proj(features)  # [B, T, V]
        return F.log_softmax(logits, dim=-1)

    def compute_ctc_loss(
        self,
        log_probs: torch.Tensor,
        targets: torch.Tensor,
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute CTC loss.
        
        Args:
            log_probs: [B, T, V] from forward()
            targets: [B, S] target phoneme sequences (packed)
            input_lengths: [B] length of each input sequence
            target_lengths: [B] length of each target sequence
        Returns:
            CTC loss scalar
        """
        # CTC expects [T, B, V]
        log_probs = log_probs.permute(1, 0, 2)
        return F.ctc_loss(
            log_probs, targets,
            input_lengths, target_lengths,
            blank=self.blank_id,
            reduction="mean",
            zero_infinity=True,
        )


def text_to_phonemes(text: str) -> list:
    """
    Simple grapheme-to-phoneme conversion for English.
    Uses character-level mapping as a fallback when G2P library isn't available.
    """
    try:
        import g2p_en
        g2p = g2p_en.G2p()
        phonemes = g2p(text)
        return [p for p in phonemes if p.strip()]
    except ImportError:
        # Character-level fallback
        return list(text.upper().replace(" ", " <space> ").split())


def phonemes_to_ids(phonemes: list, vocab: list = PHONEME_VOCAB) -> list:
    """Convert phoneme strings to vocabulary IDs."""
    vocab_map = {p: i for i, p in enumerate(vocab)}
    return [vocab_map.get(p, 0) for p in phonemes]
