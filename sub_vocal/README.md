# Sub Vocal Speech Synthesizer Architecture

The Sub Vocal Speech Synthesizer has been successfully migrated from a rapid-prototyping architecture to a robust, professional-grade 4-phase training pipeline. This architecture ensures high-fidelity multilingual lip-to-speech synthesis through rigorous domain-specific fine-tuning.

## Directory Structure

```text
sub_vocal/
├── configs/                  # Hierarchical YAML configs
│   ├── base.yaml             # Shared base configuration
│   ├── phase1_pretrain.yaml  # Visual encoder pretraining
│   ├── phase2_lip2speech.yaml # Core LRS3 training
│   └── phase4_multilingual.yaml # MuAViC multilingual
├── data/                     # Dataset loaders & augmentation
│   ├── lrw_dataset.py        # 500-word classification (Phase 1)
│   ├── lrs_dataset.py        # Sentence-level lip/audio (Phase 2)
│   ├── muavic_dataset.py     # 9-language balanced sampling (Phase 4)
│   ├── transforms.py         # Visual (time-masking) & Audio augmentation
│   └── collate.py            # Variable-length sequence collation
├── models/                   # Neural network architectures
│   ├── visual_encoder.py     # 3D Conv + ResNet18/50 backbone
│   ├── decoder.py            # Autoregressive Transformer with cross-attention
│   ├── speaker_embed.py      # GE2E LSTM encoder + FiLM conditioning
│   ├── ctc_head.py           # Phoneme supervision head
│   ├── hifi_gan.py           # Neural vocoder (Phase 3)
│   └── lip2speech_model.py   # Unified model assembler
├── losses/                   # Multi-task loss formulations
│   ├── mel_loss.py           # L1 + L2 Mel reconstruction
│   ├── ssim_loss.py          # Structural Similarity Index
│   └── multitask.py          # Weighted combination with delayed CTC
├── training/                 # Experiment lifecycle infrastructure
│   ├── trainer.py            # Main training & validation loop
│   ├── checkpoint.py         # Save/resume, best-model tracking
│   ├── early_stopping.py     # Patience-based stopping
│   └── scheduler.py          # Warmup & cosine annealing
├── preprocess/               # Face extraction (Legacy/Pre-processing)
│   ├── detect.py
│   └── align.py
├── train.py                  # Universal CLI entry point
└── evaluate.py               # STOI, PESQ, WER, and UTMOS evaluation suite
```

## Training Pipeline Phases

### Phase 1: Visual Encoder Pretraining
- **Dataset:** LRW (Lip Reading Words) - 500 isolated word classes.
- **Model:** `VisualEncoder` (3D Conv frontend + 2D ResNet backbone).
- **Objective:** Cross-Entropy classification.
- **Goal:** Learn robust spatio-temporal representations of fundamental lip movements.

### Phase 2: Core Lip-to-Speech Synthesis
- **Dataset:** LRS2/LRS3 (Sentence-level).
- **Model:** `Lip2SpeechModel` (Frozen visual encoder + Transformer Decoder + Speaker Embedding + CTC Head).
- **Objective:** Multi-task loss (Mel L1 + SSIM + Stop Token BCE + CTC Phoneme log-probs).
- **Goal:** Map lip sequence features to mel-spectrograms while resolving one-to-many ambiguities via CTC phoneme supervision and disentangling identity via GE2E speaker embeddings.

### Phase 3: HiFi-GAN Vocoder Fine-tuning
- **Dataset:** LJSpeech (Pretraining) -> LRS Audio (Fine-tuning).
- **Model:** `HiFiGANVocoder`.
- **Goal:** Convert generated mel-spectrograms into high-fidelity raw audio waveforms.

### Phase 4: Multilingual Adaptation
- **Dataset:** MuAViC (9 languages: en, fr, es, de, it, pt, ru, ar, el).
- **Model:** `Lip2SpeechModel` + Language Embeddings.
- **Objective:** Fine-tuning the decoder with per-language embeddings and balanced sampling.
- **Goal:** Enable synthesis across non-English languages using language-specific adapter conditioning.

## Running the Pipeline

You can orchestrate training using the unified CLI `train.py` and the respective configuration files:

```bash
# Phase 1
python -m sub_vocal.train --phase 1 --config sub_vocal/configs/phase1_pretrain.yaml

# Phase 2
python -m sub_vocal.train --phase 2 --config sub_vocal/configs/phase2_lip2speech.yaml

# Phase 4
python -m sub_vocal.train --phase 4 --config sub_vocal/configs/phase4_multilingual.yaml
```

To evaluate checkpoints, use the comprehensive evaluation suite:
```bash
python -m sub_vocal.evaluate -r ref_audio.wav -g generated.wav -t "reference text"
```
