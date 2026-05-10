"""
Models module — all model architectures.

Imports are lazy to avoid requiring all dependencies at module load time.
Import individual components directly:
    from sub_vocal.models.decoder import MelDecoder
"""

__all__ = [
    "VisualEncoder", "MelDecoder", "SpeakerEmbedding", "CTCPhonemeHead",
    "Lip2SpeechModel", "HiFiGANVocoder",
    "AVHuBERTEncoder", "UnitLanguageModel", "TTSEngine",
]


def __getattr__(name):
    """Lazy import to avoid requiring torchvision/fairseq at module load."""
    _imports = {
        "VisualEncoder": ".visual_encoder",
        "MelDecoder": ".decoder",
        "SpeakerEmbedding": ".speaker_embed",
        "CTCPhonemeHead": ".ctc_head",
        "Lip2SpeechModel": ".lip2speech_model",
        "HiFiGANVocoder": ".hifi_gan",
        "AVHuBERTEncoder": ".avhubert",
        "UnitLanguageModel": ".unit_lm",
        "TTSEngine": ".tts_engine",
    }
    if name in _imports:
        import importlib
        module = importlib.import_module(_imports[name], package=__name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
