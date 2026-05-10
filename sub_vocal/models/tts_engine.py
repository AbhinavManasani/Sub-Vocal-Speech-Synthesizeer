"""
TTS Engine — Multilingual Speech Synthesis
============================================

Wraps Coqui XTTS v2 (primary) and Facebook MMS-TTS (fallback) for
converting text to speech waveforms in 17+ languages.

XTTS v2: Better quality, voice cloning, 17 languages
MMS-TTS: Wider coverage (1100+ languages), no voice cloning needed
"""

import os
import logging
import numpy as np
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# XTTS v2 supported languages
XTTS_LANGUAGES = {
    "en", "es", "fr", "de", "it", "pt", "pl", "tr",
    "ru", "nl", "cs", "ar", "zh-cn", "ja", "hu", "ko", "hi",
}

# MMS-TTS language code mapping (ISO 639-3)
MMS_LANGUAGE_MAP = {
    "en": "eng", "fr": "fra", "de": "deu", "es": "spa",
    "it": "ita", "pt": "por", "nl": "nld", "ru": "rus",
    "hi": "hin", "zh": "cmn", "zh-cn": "cmn", "ja": "jpn",
    "ko": "kor", "ar": "ara", "pl": "pol", "tr": "tur",
    "cs": "ces", "hu": "hun", "sv": "swe", "da": "dan",
    "fi": "fin", "el": "ell", "th": "tha", "vi": "vie",
    "id": "ind", "ms": "zsm", "ta": "tam", "te": "tel",
    "bn": "ben", "ur": "urd", "fa": "pes", "he": "heb",
}


class TTSEngine:
    """
    Multilingual TTS engine with automatic backend selection.
    
    Primary: Coqui XTTS v2 (voice cloning, 17 langs)
    Fallback: Facebook MMS-TTS (1100+ langs, no cloning)
    """

    def __init__(self, backend="auto", device=None):
        try:
            import torch
            _cuda = torch.cuda.is_available()
        except Exception:
            _cuda = False
        self.device = device or ("cuda" if _cuda else "cpu")
        self.backend = backend
        self.xtts_model = None
        self.mms_model = None
        self._loaded_backend = None
        logger.info(f"TTSEngine configured: backend={backend}, device={self.device}")

    def load(self, language="en"):
        """Load the appropriate TTS backend for the given language."""
        if self._loaded_backend:
            return

        if self.backend == "auto":
            if language in XTTS_LANGUAGES:
                self.backend = "xtts"
            else:
                self.backend = "mms"

        if self.backend == "xtts":
            try:
                self._load_xtts()
                self._loaded_backend = "xtts"
                return
            except Exception as e:
                logger.warning(f"XTTS load failed: {e}, trying MMS fallback")
                self.backend = "mms"

        if self.backend == "mms":
            try:
                self._load_mms(language)
                self._loaded_backend = "mms"
            except Exception as e:
                logger.error(f"MMS-TTS also failed: {e}")
                logger.info("Falling back to generic fallback engine.")
                self._loaded_backend = "fallback"

    def _load_xtts(self):
        """Load Coqui XTTS v2 model."""
        from TTS.api import TTS
        logger.info("Loading Coqui XTTS v2...")
        self.xtts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
        self.xtts_model = self.xtts_model.to(self.device)
        logger.info("XTTS v2 loaded successfully")

    def _load_mms(self, language="en"):
        """Load Facebook MMS-TTS model for the given language."""
        from transformers import VitsModel, AutoTokenizer
        lang_code = MMS_LANGUAGE_MAP.get(language, "eng")
        model_id = f"facebook/mms-tts-{lang_code}"
        logger.info(f"Loading MMS-TTS: {model_id}")
        self.mms_model = {
            "model": VitsModel.from_pretrained(model_id).to(self.device),
            "tokenizer": AutoTokenizer.from_pretrained(model_id),
            "lang": lang_code,
        }
        logger.info(f"MMS-TTS loaded for {lang_code}")

    def synthesize(self, text, language="en", speaker_wav=None, output_path="output.wav"):
        """
        Synthesize speech from text.
        
        Args:
            text: Input text to speak
            language: Language code (e.g., 'en', 'fr', 'hi')
            speaker_wav: Path to reference speaker audio (for voice cloning, XTTS only)
            output_path: Where to save the output .wav file
            
        Returns:
            Path to the generated audio file
        """
        self.load(language)

        if self._loaded_backend == "xtts":
            return self._synthesize_xtts(text, language, speaker_wav, output_path)
        elif self._loaded_backend == "mms":
            return self._synthesize_mms(text, language, output_path)
        else:
            return self._synthesize_fallback(text, language, output_path)

    def _synthesize_xtts(self, text, language, speaker_wav, output_path):
        """Synthesize using Coqui XTTS v2."""
        lang = language if language in XTTS_LANGUAGES else "en"
        logger.info(f"XTTS synthesis: lang={lang}, text='{text[:60]}...'")

        kwargs = {"text": text, "file_path": output_path, "language": lang}

        if speaker_wav and os.path.exists(speaker_wav):
            kwargs["speaker_wav"] = speaker_wav
            logger.info(f"Using speaker reference: {speaker_wav}")
        else:
            # XTTS needs speaker_wav — use a generated silence if none
            logger.warning("No speaker reference audio, generating default voice")
            ref_path = self._create_default_reference()
            if ref_path:
                kwargs["speaker_wav"] = ref_path

        try:
            self.xtts_model.tts_to_file(**kwargs)
            logger.info(f"XTTS output saved to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"XTTS synthesis failed: {e}")
            # Try MMS fallback
            logger.info("Falling back to MMS-TTS")
            try:
                self._load_mms(language)
                self._loaded_backend = "mms"
                return self._synthesize_mms(text, language, output_path)
            except Exception as e2:
                logger.error(f"MMS-TTS fallback failed: {e2}")
                return self._synthesize_fallback(text, language, output_path)

    def _synthesize_mms(self, text, language, output_path):
        """Synthesize using Facebook MMS-TTS."""
        import scipy.io.wavfile

        # Reload for different language if needed
        target_lang = MMS_LANGUAGE_MAP.get(language, "eng")
        if self.mms_model is None or self.mms_model["lang"] != target_lang:
            self._load_mms(language)

        logger.info(f"MMS synthesis: lang={target_lang}, text='{text[:60]}...'")

        model = self.mms_model["model"]
        tokenizer = self.mms_model["tokenizer"]

        import torch
        inputs = tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = model(**inputs).waveform

        waveform = output.cpu().numpy().squeeze()
        sr = model.config.sampling_rate

        # Normalize to int16
        waveform = np.clip(waveform, -1.0, 1.0)
        waveform_int16 = (waveform * 32767).astype(np.int16)

        scipy.io.wavfile.write(output_path, rate=sr, data=waveform_int16)
        logger.info(f"MMS output saved to {output_path} ({len(waveform)/sr:.1f}s)")
        return output_path

    def _synthesize_fallback(self, text, language, output_path):
        """Synthesize using gTTS or pyttsx3 as absolute fallback."""
        try:
            from gtts import gTTS
            logger.info(f"Fallback synthesis using gTTS: lang={language}")
            # Ensure 2-letter lang code for gTTS
            gtts_lang = language.split("-")[0][:2]
            tts = gTTS(text=text, lang=gtts_lang, slow=False)
            tts.save(output_path)
            logger.info(f"gTTS output saved to {output_path}")
            return output_path
        except Exception as e:
            logger.warning(f"gTTS failed: {e}, falling back to pyttsx3")
            try:
                import pyttsx3
                engine = pyttsx3.init()
                engine.save_to_file(text, output_path)
                engine.runAndWait()
                logger.info(f"pyttsx3 output saved to {output_path}")
                return output_path
            except Exception as e2:
                logger.error(f"pyttsx3 fallback failed: {e2}")
                raise RuntimeError("All TTS backends failed.")

    def _create_default_reference(self):
        """Create a minimal reference audio for XTTS when no speaker_wav is available."""
        try:
            import soundfile as sf
            ref_path = "checkpoints/default_speaker.wav"
            os.makedirs("checkpoints", exist_ok=True)
            if not os.path.exists(ref_path):
                # Generate 3 seconds of near-silence with slight noise
                sr = 22050
                duration = 3.0
                samples = int(sr * duration)
                audio = np.random.randn(samples).astype(np.float32) * 0.001
                sf.write(ref_path, audio, sr)
                logger.info(f"Created default reference: {ref_path}")
            return ref_path
        except Exception as e:
            logger.warning(f"Cannot create default reference: {e}")
            return None

    @staticmethod
    def list_supported_languages():
        """List all supported language codes."""
        all_langs = sorted(XTTS_LANGUAGES | set(MMS_LANGUAGE_MAP.keys()))
        return all_langs
