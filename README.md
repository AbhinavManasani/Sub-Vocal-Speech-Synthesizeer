# Sub Vocal Speech Synthesizer 🎙️

**Multilingual lip-to-speech synthesis using pretrained models — zero training required.**

Transform silent lip movements from any video into natural speech in 17+ languages using a 3-stage pipeline:

```
video.mp4 → MediaPipe lip crop → AV-HuBERT → speech units → text → XTTS v2 → audio.wav
```

## Architecture

```mermaid
graph LR
    A["🎥 Video Input"] --> B["👄 MediaPipe\nLip Detection"]
    B --> C["🧠 AV-HuBERT\nEncoder"]
    C --> D["📊 K-Means\nQuantizer"]
    D --> E["📝 Unit→Text\nDecoder"]
    E --> F["🌍 Language\nRouter"]
    F --> G["🔊 Coqui XTTS v2\nMultilingual TTS"]
    G --> H["🎵 Audio Output"]
```

| Stage | Component | Model | Input → Output |
|-------|-----------|-------|----------------|
| 1 | Lip → Units | AV-HuBERT Large + KM500 | `[T,96,96]` frames → `[T]` unit IDs |
| 2 | Units → Text | Unit Language Model | `[T]` unit IDs → text/phonemes |
| 3 | Text → Speech | XTTS v2 / MMS-TTS | text + lang → waveform |

## Quick Start

### 1. Environment Setup

```bash
conda create -n subvocal python=3.10
conda activate subvocal
pip install -r requirements.txt
```

### 2. Download Checkpoints (runs in background)

```bash
mkdir -p checkpoints
# AV-HuBERT Large (~1.3GB)
wget -P checkpoints/ https://dl.fbaipublicfiles.com/avhubert/model/lrs3_vox/av_hubert_large.pt
# K-Means 500 clusters
wget -P checkpoints/ https://dl.fbaipublicfiles.com/avhubert/model/lrs3_vox/km500.bin
```

### 3. Run

```bash
# Basic usage
python -m sub_vocal.demo --video input.mp4 --lang en

# French output with custom speaker voice
python -m sub_vocal.demo --video input.mp4 --lang fr --speaker-ref speaker.wav

# Hindi with MMS-TTS backend (wider language support)
python -m sub_vocal.demo --video input.mp4 --lang hi --tts-backend mms

# List all supported languages
python -m sub_vocal.demo --list-languages
```

## Supported Languages

### XTTS v2 (high quality + voice cloning)
en, es, fr, de, it, pt, pl, tr, ru, nl, cs, ar, zh-cn, ja, hu, ko, hi

### MMS-TTS (1100+ languages, no voice cloning)
Covers virtually every language — see [MMS model hub](https://huggingface.co/models?search=facebook/mms-tts)

## Project Structure

```
sub_vocal/
├── preprocess/
│   ├── detect.py       # MediaPipe face + lip crop
│   └── align.py        # Affine warp to 96×96 canonical
├── models/
│   ├── avhubert.py     # AV-HuBERT loading + inference
│   ├── unit_lm.py      # Unit sequence → text/phonemes
│   └── tts_engine.py   # Coqui XTTS v2 / MMS-TTS wrapper
├── pipeline.py         # End-to-end orchestration
├── demo.py             # CLI entry point
├── evaluate.py         # STOI, PESQ, WER metrics
├── requirements.txt
└── README.md
```

## CLI Reference

```
usage: python -m sub_vocal.demo [-h] [--video VIDEO] [--lang LANG]
                                [--output OUTPUT] [--speaker-ref PATH]
                                [--tts-backend {auto,xtts,mms}]
                                [--avhubert-ckpt PATH] [--km-path PATH]
                                [--max-frames N] [--fps FPS]
                                [--device {cuda,cpu}] [--verbose]
                                [--list-languages]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--video` | required | Input video file path |
| `--lang` | `en` | Target language code |
| `--output` | `output_{lang}.wav` | Output audio path |
| `--speaker-ref` | auto | Speaker reference WAV for voice cloning |
| `--tts-backend` | `auto` | `xtts` / `mms` / `auto` |
| `--max-frames` | all | Limit frames for testing |
| `--fps` | `25.0` | Frame extraction rate |
| `--device` | auto | `cuda` or `cpu` |

## Evaluation

```bash
python -m sub_vocal.evaluate --reference ground_truth.wav --generated output_en.wav --text "expected text"
```

Metrics: **STOI** (intelligibility), **PESQ** (perceptual quality), **WER** (word error rate via Whisper)

## Known Limitations

1. **AV-HuBERT is English-dominant** — trained on LRS3, unit quality drops for non-Latin scripts (Hindi, Chinese, Arabic)
2. **Unit LM is approximate** — the phoneme mapping is statistical, not learned; expect lower accuracy
3. **XTTS v2 needs reference audio** — extracts from video audio track automatically, or generates default
4. **fairseq compatibility** — pin to `torch==2.0.1`, `fairseq==0.12.2` to avoid conflicts

## References

- [AV-HuBERT](https://github.com/facebookresearch/av_hubert) — Audio-Visual HuBERT
- [lip2speech-unit](https://github.com/choijeongsoo/lip2speech-unit) — Interspeech 2023
- [Coqui TTS](https://github.com/coqui-ai/TTS) — XTTS v2 multilingual
- [MMS-TTS](https://huggingface.co/facebook/mms-tts) — Massively Multilingual Speech

## License

MIT
