"""
Learning Rate Schedulers
==========================

Warmup + cosine annealing and exponential decay schedules.
"""

import math
import torch
from torch.optim.lr_scheduler import _LRScheduler
from typing import Optional


class WarmupCosineScheduler(_LRScheduler):
    """Linear warmup followed by cosine annealing to min_lr."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int = 5000,
        max_steps: int = 500000,
        min_lr: float = 1e-6,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch
        if step < self.warmup_steps:
            scale = step / max(1, self.warmup_steps)
        else:
            progress = (step - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)
            scale = max(0, 0.5 * (1 + math.cos(math.pi * progress)))

        return [
            max(self.min_lr, base_lr * scale) for base_lr in self.base_lrs
        ]


def build_scheduler(optimizer, config: dict) -> Optional[_LRScheduler]:
    """Build LR scheduler from config dict."""
    sched_type = config.get("type", "warmup_cosine")

    if sched_type == "warmup_cosine":
        return WarmupCosineScheduler(
            optimizer,
            warmup_steps=config.get("warmup_steps", 5000),
            min_lr=config.get("min_lr", 1e-6),
        )
    elif sched_type == "exponential":
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=config.get("gamma", 0.999)
        )
    elif sched_type == "none":
        return None
    else:
        raise ValueError(f"Unknown scheduler: {sched_type}")
