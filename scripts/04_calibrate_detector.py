#!/usr/bin/env python3
"""Calibrate detector thresholds for target TPRs."""

import argparse
import yaml
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.cot_generator import load_cot_examples
from src.detector.model import CoTLieDetector
from src.detector.calibrate import calibrate_thresholds, save_calibration
from src.detector.train import split_train_val
from src.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser(description="Calibrate detector thresholds")
    parser.add_argument("--config", type=str, default="configs/detector.yaml", help="Config file")
    parser.add_argument("--cot-data", type=str, default="data/cot_detector_train.jsonl", help="CoT data")
    parser.add_argument("--detector-path", type=str, default="runs/detector/best_model", help="Detector model path")
    parser.add_argument("--output", type=str, default="data/detector_thresholds.json", help="Output file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    seed = args.seed
    set_seed(seed)

    calibration_config = config.get("calibration", {})
    target_tprs = calibration_config.get("target_tprs", [0.60, 0.75, 0.90, 0.97])

    print(f"Calibrating detector thresholds")
    print(f"  Target TPRs: {target_tprs}")

    # Load detector
    print(f"\nLoading detector from {args.detector_path}")
    detector = CoTLieDetector.load(args.detector_path)

    # Load CoT data and get validation split
    print(f"Loading CoT data from {args.cot_data}")
    cot_examples = load_cot_examples(args.cot_data)

    # Use same split as training to get validation set
    _, val_examples = split_train_val(cot_examples, val_ratio=0.1, seed=seed)
    print(f"Using {len(val_examples)} validation examples for calibration")

    # Calibrate
    print("\nCalibrating thresholds...")
    result = calibrate_thresholds(
        detector,
        val_examples,
        target_tprs=target_tprs,
    )

    # Save
    save_calibration(result, args.output)

    print(f"\nCalibration complete!")
    print(f"  Validation AUC: {result.val_auc:.4f}")
    print(f"  Thresholds saved to: {args.output}")


if __name__ == "__main__":
    main()
