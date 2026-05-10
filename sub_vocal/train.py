"""
Training Entry Point
======================

CLI for launching training across all 4 phases.

Usage:
    python -m sub_vocal.train --phase 1 --config sub_vocal/configs/phase1_visual_pretrain.yaml
    python -m sub_vocal.train --phase 2 --config sub_vocal/configs/phase2_lip2speech.yaml
    python -m sub_vocal.train --phase 2 --resume checkpoints/checkpoint_step50000.pt
"""

import argparse
import logging
import sys
import os
import yaml
import torch
from pathlib import Path


def setup_logging(verbose=False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("training.log", mode="a"),
        ],
    )


def load_config(config_path: str) -> dict:
    """Load YAML config, merging with base if _base_ is specified."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Merge with base config
    if "_base_" in config:
        base_dir = os.path.dirname(config_path)
        base_path = os.path.join(base_dir, config["_base_"])
        if os.path.exists(base_path):
            with open(base_path, "r") as f:
                base_config = yaml.safe_load(f)
            # Deep merge: config overrides base
            config = _deep_merge(base_config, config)
        del config["_base_"]

    return config


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def build_phase1(config):
    """Build components for Phase 1: visual encoder pretraining."""
    from sub_vocal.models.lip2speech_model import Lip2SpeechModel
    from sub_vocal.data.lrw_dataset import LRWDataset
    from sub_vocal.data.transforms import VideoTransform
    from sub_vocal.data.collate import collate_fn
    from torch.utils.data import DataLoader

    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})

    # Model
    model = Lip2SpeechModel(
        phase=1,
        backbone=model_cfg.get("visual_encoder", {}).get("backbone", "resnet18"),
        pretrained_2d=model_cfg.get("visual_encoder", {}).get("pretrained_2d", True),
        encoder_dim=model_cfg.get("visual_encoder", {}).get("out_dim", 512),
        lrw_pretrained_path=model_cfg.get("visual_encoder", {}).get("lrw_pretrained_path", None),
        num_classes=data_cfg.get("num_classes", 500),
    )

    # Data
    aug_cfg = data_cfg.get("augment", {})
    train_transform = VideoTransform(
        random_crop=aug_cfg.get("random_crop", True),
        horizontal_flip=aug_cfg.get("horizontal_flip", True),
        time_mask=aug_cfg.get("time_mask", True),
        training=True,
    )
    val_transform = VideoTransform(training=False)

    train_dataset = LRWDataset(
        root=data_cfg.get("root", "/data/LRW"),
        split="train", transform=train_transform,
        max_frames=data_cfg.get("max_video_frames", 29),
    )
    val_dataset = LRWDataset(
        root=data_cfg.get("root", "/data/LRW"),
        split="val", transform=val_transform,
        max_frames=data_cfg.get("max_video_frames", 29),
    )

    train_cfg = config.get("training", {})
    train_loader = DataLoader(
        train_dataset, batch_size=train_cfg.get("batch_size", 32),
        shuffle=True, num_workers=config.get("num_workers", 4),
        pin_memory=True, collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=train_cfg.get("batch_size", 32),
        shuffle=False, num_workers=config.get("num_workers", 4),
        pin_memory=True, collate_fn=collate_fn,
    )

    # Loss (cross-entropy for classification)
    loss_cfg = train_cfg.get("loss", {})

    class Phase1Loss(torch.nn.Module):
        def __init__(self):
            super().__init__()
            ls = loss_cfg.get("label_smoothing", 0.1)
            self.ce = torch.nn.CrossEntropyLoss(label_smoothing=ls)

        def forward(self, outputs, batch, global_step=0):
            loss = self.ce(outputs["logits"], batch["label"])
            # Accuracy
            preds = outputs["logits"].argmax(dim=-1)
            acc = (preds == batch["label"]).float().mean()
            return {"total_loss": loss, "accuracy": acc}

    return model, train_loader, val_loader, Phase1Loss()


def build_phase2(config):
    """Build components for Phase 2: full lip-to-speech."""
    from sub_vocal.models.lip2speech_model import Lip2SpeechModel
    from sub_vocal.data.lrs_dataset import LRSDataset
    from sub_vocal.data.transforms import VideoTransform, AudioTransform
    from sub_vocal.data.collate import collate_fn
    from sub_vocal.losses import MultiTaskLoss
    from torch.utils.data import DataLoader

    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    train_cfg = config.get("training", {})

    # Model
    enc_cfg = model_cfg.get("visual_encoder", {})
    dec_cfg = model_cfg.get("decoder", {})
    spk_cfg = model_cfg.get("speaker_embed", {})
    ctc_cfg = model_cfg.get("ctc_head", {})

    model = Lip2SpeechModel(
        phase=2,
        backbone=enc_cfg.get("backbone", "resnet18"),
        pretrained_2d=enc_cfg.get("pretrained_2d", True),
        encoder_dim=enc_cfg.get("out_dim", 512),
        lrw_pretrained_path=enc_cfg.get("lrw_pretrained_path", None),
        decoder_layers=dec_cfg.get("num_layers", 6),
        decoder_heads=dec_cfg.get("nhead", 8),
        decoder_ff_dim=dec_cfg.get("dim_feedforward", 2048),
        mel_dim=dec_cfg.get("mel_dim", 80),
        speaker_dim=spk_cfg.get("dim", 256),
        speaker_method="film" if spk_cfg.get("film_conditioning") else "add",
        ctc_vocab_size=ctc_cfg.get("vocab_size", 44),
        ctc_enabled=ctc_cfg.get("enabled", True),
    )

    # Load Phase 1 encoder weights
    if enc_cfg.get("pretrained_path"):
        model.load_encoder_weights(enc_cfg["pretrained_path"])
    if enc_cfg.get("freeze", True):
        model.freeze_encoder()

    # Data
    video_root = data_cfg.get("lrs3_root", data_cfg.get("lrs2_root", "/data/LRS3-TED"))
    train_dataset = LRSDataset(
        video_root=video_root, split="train",
        sample_rate=data_cfg.get("sample_rate", 16000),
        n_mels=data_cfg.get("n_mels", 80),
        max_video_frames=data_cfg.get("max_video_frames", 600),
        video_transform=VideoTransform(training=True),
    )
    val_dataset = LRSDataset(
        video_root=video_root, split="val",
        sample_rate=data_cfg.get("sample_rate", 16000),
        n_mels=data_cfg.get("n_mels", 80),
        max_video_frames=data_cfg.get("max_video_frames", 600),
        video_transform=VideoTransform(training=False),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=train_cfg.get("batch_size", 16),
        shuffle=True, num_workers=config.get("num_workers", 4),
        pin_memory=True, collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=train_cfg.get("batch_size", 16),
        shuffle=False, num_workers=config.get("num_workers", 4),
        pin_memory=True, collate_fn=collate_fn,
    )

    # Loss
    loss_cfg = train_cfg.get("loss", {})
    loss_fn = MultiTaskLoss(
        mel_l1_weight=loss_cfg.get("mel_l1_weight", 1.0),
        mel_postnet_l1_weight=loss_cfg.get("mel_postnet_l1_weight", 1.0),
        ctc_weight=loss_cfg.get("ctc_weight", 0.1),
        ssim_weight=loss_cfg.get("ssim_weight", 0.5),
        output_content_on=loss_cfg.get("output_content_on", 0),
    )

    return model, train_loader, val_loader, loss_fn


def parse_args():
    parser = argparse.ArgumentParser(description="Sub Vocal Training")
    parser.add_argument("--phase", type=int, required=True, choices=[1, 2, 3, 4],
                        help="Training phase (1-4)")
    parser.add_argument("--config", type=str, required=True, help="YAML config path")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    # Set seed
    config = load_config(args.config)
    seed = config.get("seed", 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    logger.info(f"Phase {args.phase} training | config: {args.config}")

    # Build phase-specific components
    if args.phase == 1:
        model, train_loader, val_loader, loss_fn = build_phase1(config)
    elif args.phase == 2:
        model, train_loader, val_loader, loss_fn = build_phase2(config)
    elif args.phase == 3:
        logger.info("Phase 3 (vocoder) — use dedicated vocoder training script")
        return
    elif args.phase == 4:
        logger.info("Phase 4 (multilingual) — extending Phase 2 with adapters")
        model, train_loader, val_loader, loss_fn = build_phase2(config)

    # Create trainer and run
    from sub_vocal.training.trainer import Trainer
    trainer = Trainer(model, train_loader, val_loader, loss_fn, config)
    trainer.train(resume_from=args.resume)


if __name__ == "__main__":
    main()
