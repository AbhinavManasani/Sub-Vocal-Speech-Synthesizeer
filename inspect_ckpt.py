import torch, os

CKPTS = [
    "checkpoints/phase2_lrs_lip2speech/best.pt",
    "checkpoints/phase3_vocoder/best.pt",
    "checkpoints/phase3_vocoder/vocoder_step0.pt",
]

for path in CKPTS:
    if not os.path.exists(path):
        print(f"MISSING: {path}")
        continue

    size_mb = os.path.getsize(path) / 1e6
    ckpt = torch.load(path, map_location="cpu")
    print(f"\n{'='*60}")
    print(f"FILE : {path}  ({size_mb:.1f} MB)")
    print(f"KEYS : {list(ckpt.keys())}")

    for meta_key in ("epoch", "step", "global_step", "val_loss", "train_loss",
                     "best_val_loss", "loss", "config"):
        if meta_key in ckpt:
            val = ckpt[meta_key]
            if not isinstance(val, dict):
                print(f"  {meta_key:15s}: {val}")

    # Find state dict
    sd = None
    for k in ("model_state_dict", "state_dict", "model", "generator"):
        if k in ckpt and isinstance(ckpt[k], dict):
            sd = ckpt[k]
            sd_key = k
            break
    if sd is None and isinstance(ckpt, dict):
        # maybe the checkpoint IS the state dict
        first_v = next(iter(ckpt.values()))
        if isinstance(first_v, torch.Tensor):
            sd = ckpt
            sd_key = "(root)"

    if sd:
        print(f"  state_dict_key  : {sd_key}")
        print(f"  num_params      : {len(sd)} tensors")
        total_params = sum(v.numel() for v in sd.values() if isinstance(v, torch.Tensor))
        print(f"  total_elements  : {total_params:,}")

        # Check first 3 weight tensors for mean/std (tells us if random-init or trained)
        checked = 0
        for name, tensor in sd.items():
            if not isinstance(tensor, torch.Tensor): continue
            if tensor.ndim < 1: continue
            m = tensor.float().mean().item()
            s = tensor.float().std().item()
            print(f"  [{name}]  mean={m:.6f}  std={s:.6f}")
            checked += 1
            if checked >= 3:
                break
    print()
