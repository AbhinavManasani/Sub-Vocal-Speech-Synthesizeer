import urllib.request, os
os.makedirs('checkpoints', exist_ok=True)

MODELS = [
  # AV-HuBERT — try multiple mirrors
  ("https://dl.fbaipublicfiles.com/avhubert/model/lrs3_vox/av_hubert_large.pt",
   "checkpoints/avhubert_large.pt"),
  # Backup: HuggingFace mirror of AV-HuBERT
  ("https://huggingface.co/facebook/av-hubert-large-lrs3-iter5/resolve/main/model.pt",
   "checkpoints/avhubert_large.pt"),
]

for url, dest in MODELS:
  if os.path.exists(dest):
    print(f'Already exists: {dest}')
    continue
  print(f'Downloading {url}...')
  try:
    urllib.request.urlretrieve(url, dest)
    print(f'✓ {dest}')
    break
  except Exception as e:
    print(f'✗ {e}, trying next...')
