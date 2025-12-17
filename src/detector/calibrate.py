"""Detector threshold calibration for target TPR/FPR."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import roc_curve
import torch

from .model import CoTLieDetector
from ..data.cot_generator import CoTExample


@dataclass
class ThresholdCalibration:
    """Calibration result for a single target TPR."""

    target_tpr: float
    threshold: float
    achieved_tpr: float
    achieved_fpr: float


@dataclass
class CalibrationResult:
    """Full calibration results."""

    calibrations: List[ThresholdCalibration]
    val_auc: float
    val_scores: Dict[str, List[float]]  # Scores for truthful/deceptive


def calibrate_thresholds(
    model: CoTLieDetector,
    val_examples: List[CoTExample],
    target_tprs: List[float] = [0.60, 0.75, 0.90, 0.97],
    device: str = "cuda",
    batch_size: int = 32,
) -> CalibrationResult:
    """Calibrate detection thresholds to achieve target TPRs.

    Args:
        model: Trained detector model.
        val_examples: Validation examples for calibration.
        target_tprs: List of target true positive rates.
        device: Device to run on.
        batch_size: Batch size for scoring.

    Returns:
        CalibrationResult with thresholds for each target TPR.
    """
    model.eval()
    model.to(device)

    # Score all examples
    prompts = [ex.prompt for ex in val_examples]
    cot_texts = [ex.cot_text for ex in val_examples]
    labels = np.array([ex.label for ex in val_examples])

    scores = model.score_batch(prompts, cot_texts, batch_size=batch_size, device=device)
    scores = np.array(scores)

    # Separate scores by class
    truthful_scores = scores[labels == 0].tolist()
    deceptive_scores = scores[labels == 1].tolist()

    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(labels, scores)

    # Compute AUC
    from sklearn.metrics import roc_auc_score

    val_auc = roc_auc_score(labels, scores)

    # Find thresholds for target TPRs
    calibrations = []

    for target in target_tprs:
        # Find threshold that achieves target TPR (or closest above)
        # TPR increases as threshold decreases, so find first TPR >= target
        idx = np.searchsorted(tpr, target)
        if idx >= len(thresholds):
            idx = len(thresholds) - 1

        threshold = thresholds[idx]
        achieved_tpr = tpr[idx]
        achieved_fpr = fpr[idx]

        calibrations.append(
            ThresholdCalibration(
                target_tpr=target,
                threshold=float(threshold),
                achieved_tpr=float(achieved_tpr),
                achieved_fpr=float(achieved_fpr),
            )
        )

        print(
            f"Target TPR {target:.2f}: threshold={threshold:.4f}, "
            f"achieved_tpr={achieved_tpr:.4f}, achieved_fpr={achieved_fpr:.4f}"
        )

    return CalibrationResult(
        calibrations=calibrations,
        val_auc=val_auc,
        val_scores={"truthful": truthful_scores, "deceptive": deceptive_scores},
    )


def save_calibration(result: CalibrationResult, output_path: str) -> None:
    """Save calibration results to disk.

    Args:
        result: CalibrationResult object.
        output_path: Path to save JSON file.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "val_auc": result.val_auc,
        "calibrations": [
            {
                "target_tpr": c.target_tpr,
                "threshold": c.threshold,
                "achieved_tpr": c.achieved_tpr,
                "achieved_fpr": c.achieved_fpr,
            }
            for c in result.calibrations
        ],
        "val_scores": result.val_scores,
    }

    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Saved calibration to {output_path}")


def load_calibration(input_path: str) -> CalibrationResult:
    """Load calibration results from disk.

    Args:
        input_path: Path to JSON file.

    Returns:
        CalibrationResult object.
    """
    with open(input_path) as f:
        data = json.load(f)

    calibrations = [
        ThresholdCalibration(
            target_tpr=c["target_tpr"],
            threshold=c["threshold"],
            achieved_tpr=c["achieved_tpr"],
            achieved_fpr=c["achieved_fpr"],
        )
        for c in data["calibrations"]
    ]

    return CalibrationResult(
        calibrations=calibrations,
        val_auc=data["val_auc"],
        val_scores=data.get("val_scores", {}),
    )


def get_threshold_for_tpr(
    calibration: CalibrationResult, target_tpr: float
) -> Optional[float]:
    """Get the threshold for a specific target TPR.

    Args:
        calibration: CalibrationResult object.
        target_tpr: Target TPR to look up.

    Returns:
        Threshold if found, None otherwise.
    """
    for c in calibration.calibrations:
        if abs(c.target_tpr - target_tpr) < 0.01:
            return c.threshold
    return None
