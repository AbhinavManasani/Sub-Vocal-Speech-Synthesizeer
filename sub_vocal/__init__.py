"""
Sub Vocal Speech Synthesizer
============================

Multilingual lip-to-speech synthesis using a 3-stage pretrained pipeline:

  Stage 1: Video → Lip Crop → AV-HuBERT → Discrete Speech Units
  Stage 2: Speech Units → Text/Phonemes (Unit Language Model)
  Stage 3: Text + Language → Waveform (Coqui XTTS v2 / MMS-TTS)

No training required — all models are pretrained.
"""

__version__ = "0.1.0"
__author__ = "Sub Vocal Team"
