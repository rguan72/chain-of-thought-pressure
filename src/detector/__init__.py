"""CoT lie detector model and training."""

from .model import CoTLieDetector
from .train import train_detector
from .calibrate import calibrate_thresholds

__all__ = ["CoTLieDetector", "train_detector", "calibrate_thresholds"]
