"""Loss functions module."""

from .multitask import MultiTaskLoss
from .mel_loss import MelReconstructionLoss
from .ssim_loss import SSIMLoss

__all__ = ["MultiTaskLoss", "MelReconstructionLoss", "SSIMLoss"]
