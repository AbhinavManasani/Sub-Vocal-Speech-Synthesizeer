"""
Early Stopping
================

Patience-based early stopping that monitors a validation metric.
"""

import logging

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Stop training when a metric stops improving."""

    def __init__(
        self,
        patience: int = 15,
        min_delta: float = 0.001,
        mode: str = "min",
        metric: str = "val_loss",
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.metric = metric
        self.counter = 0
        self.best_value = float("inf") if mode == "min" else float("-inf")
        self.should_stop = False

    def step(self, value: float) -> bool:
        """
        Check if training should stop.
        
        Returns True if should stop.
        """
        if self.mode == "min":
            improved = value < (self.best_value - self.min_delta)
        else:
            improved = value > (self.best_value + self.min_delta)

        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(
                    f"Early stopping: {self.metric} hasn't improved for "
                    f"{self.patience} checks. Best: {self.best_value:.4f}"
                )

        return self.should_stop
