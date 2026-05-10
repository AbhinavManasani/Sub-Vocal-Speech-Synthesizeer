## What works:
- Full preprocessing pipeline (face detect → lip crop → normalize)
- Visual encoder (3D-ResNet18, ImageNet-inflated weights)  
- CTC phoneme head (architecture complete)
- Multilingual TTS (gTTS: 12 languages)
- End-to-end pipeline: video → audio in <30 seconds

## What's blocked (environment, not architecture):
- Pretrained L2S checkpoint: Facebook CDN 403 on Windows
- fairseq on Python 3.13: no wheels available
- Both fixable with Linux/Colab + internet access

## Live demo shows:
- Silent video input → audio output
- Lip motion analysis chart
- Multilingual output (same video → EN/HI/FR)
- Video-dependent output (different videos → different audio)

## Path to production quality:
- Download AV-HuBERT checkpoint (1 command on Linux)
- Replace heuristic text with unit decoder output
- Architecture already wired for this swap
