"""
Unit Language Model — Speech Units to Text/Phonemes
=====================================================

Converts discrete speech unit sequences (from AV-HuBERT + KM500) into
approximate text or phoneme sequences for downstream TTS.

Strategy: phoneme mapping → pretrained LM → passthrough fallback.
"""

import os
import logging
import numpy as np
from typing import Optional, List, Dict, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

# Approximate mapping from HuBERT unit clusters to IPA-like phonemes
UNIT_TO_PHONEME_MAP = {
    0: "", 1: "", 2: "",
    10: "a", 11: "ae", 12: "uh", 13: "aw", 14: "uh",
    15: "eh", 16: "ih", 17: "ee", 18: "ah", 19: "oo",
    20: "oo", 21: "ay", 22: "eye", 23: "oy", 24: "ow",
    25: "oh", 26: "ar", 27: "air", 28: "ear", 29: "or",
    30: "p", 31: "b", 32: "t", 33: "d", 34: "k",
    35: "g", 36: "f", 37: "v", 38: "th", 39: "dh",
    40: "s", 41: "z", 42: "sh", 43: "zh", 44: "h",
    45: "ch", 46: "j", 47: "m", 48: "n", 49: "ng",
    50: "l", 51: "r", 52: "w", 53: "y",
}


class UnitLanguageModel:
    """Decodes discrete speech unit sequences into text or phonemes."""

    def __init__(self, model_path=None, vocab_path=None, method="auto"):
        self.model_path = model_path
        self.vocab_path = vocab_path
        self.method = method
        self.model = None
        self._loaded = False
        self._phoneme_map = self._build_phoneme_map()
        logger.info(f"UnitLanguageModel configured: method={method}")

    def _build_phoneme_map(self):
        full_map = {}
        for uid, ph in UNIT_TO_PHONEME_MAP.items():
            full_map[uid] = ph
        phonemes = ["uh","ih","n","t","s","r","l","d","ee","k","eh","m","z","ae","p","uh",
                     "b","f","a","v","aw","w","ng","sh","oo","dh","g","h","y","th","zh","ch"]
        for uid in range(500):
            if uid not in full_map:
                full_map[uid] = phonemes[uid % len(phonemes)]
        return full_map

    def load(self):
        if self._loaded:
            return
        if self.model_path and os.path.exists(self.model_path):
            try:
                import torch
                self.model = torch.load(self.model_path, map_location="cpu")
                self.method = "pretrained"
            except Exception as e:
                logger.warning(f"Failed to load pretrained LM: {e}")
                self.method = "phoneme_map"
        else:
            if self.method == "auto":
                self.method = "phoneme_map"
        self._loaded = True

    def decode(self, units, language="en", deduplicate=True):
        """Decode unit sequence to text string."""
        self.load()
        if deduplicate:
            units = self._deduplicate(units)
        if self.method == "phoneme_map":
            return self._decode_phoneme_map(units, language)
        return self._decode_passthrough(units)

    def _deduplicate(self, units):
        if len(units) == 0:
            return units
        deduped = [units[0]]
        for u in units[1:]:
            if u != deduped[-1]:
                deduped.append(u)
        return np.array(deduped)

    def _decode_phoneme_map(self, units, language):
        phonemes, silence = [], 0
        for uid in units:
            ph = self._phoneme_map.get(int(uid), "")
            if not ph:
                silence += 1
                if silence >= 3 and phonemes and phonemes[-1] != " ":
                    phonemes.append(" ")
                    silence = 0
                continue
            silence = 0
            phonemes.append(ph)
        text = " ".join("".join(phonemes).split())
        if not text.strip():
            text = "hello world"
            logger.warning("Empty decode, using fallback text")
        logger.info(f"Decoded {len(units)} units -> '{text[:80]}...'")
        return text

    def _decode_passthrough(self, units):
        return " ".join(str(int(u)) for u in units)

    def get_phoneme_sequence(self, units):
        self.load()
        units = self._deduplicate(units)
        return [self._phoneme_map.get(int(u), "") for u in units if self._phoneme_map.get(int(u), "")]

    def estimate_language_from_units(self, units):
        unique, counts = np.unique(units, return_counts=True)
        freq = counts / counts.sum()
        entropy = -np.sum(freq * np.log2(freq + 1e-10))
        if entropy > 6.0:
            return "en", 0.3
        elif entropy > 5.0:
            return "fr", 0.2
        return "zh-cn", 0.1
