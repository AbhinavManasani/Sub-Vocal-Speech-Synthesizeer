"""
Vocoder Training CLI
======================

Entry point for training the Phase 3 HiFi-GAN vocoder.
"""

import argparse
import logging
import torch
from torch.utils.data import DataLoader

from sub_vocal.models.hifi_gan import HiFiGANVocoder
from sub_vocal.data.lrs_dataset import LRSDataset
from sub_vocal.data.collate import collate_fn
from sub_vocal.training.vocoder_trainer import VocoderTrainer

logging.basicConfig(level=logging.INFO)


def main():
    parser = argparse.ArgumentParser("Vocoder Trainer")
    parser.add_argument("--data_dir", required=True, help="LRS dataset directory")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    # Model
    model = HiFiGANVocoder()

    # Datasets
    train_dataset = LRSDataset(video_root=args.data_dir, audio_root=args.data_dir, split="trainval", max_video_frames=100)
    val_dataset = LRSDataset(video_root=args.data_dir, audio_root=args.data_dir, split="val", max_video_frames=100)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=0)

    # Trainer configuration
    config = {
        "save_dir": "checkpoints/phase3_vocoder",
        "log_dir": "runs/phase3_vocoder",
        "save_every_n_steps": 5000,
        "max_epochs": 1,
        "optimizer": {"lr": 2e-4, "betas": [0.8, 0.99]},
        "device": "cpu"
    }

    trainer = VocoderTrainer(model, train_loader, val_loader, config)
    trainer.train(resume_from=args.resume)


if __name__ == "__main__":
    main()
