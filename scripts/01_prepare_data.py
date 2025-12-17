#!/usr/bin/env python3
"""Prepare DolusChat dataset and create splits."""

import argparse
import yaml
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.doluschat import load_doluschat, prepare_dataset, save_splits, get_example_stats
from src.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser(description="Prepare DolusChat dataset")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Config file")
    parser.add_argument("--output-dir", type=str, default="data/splits", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    seed = args.seed or config.get("seed", 42)
    set_seed(seed)

    splits_config = config.get("splits", {})
    detector_train_ratio = splits_config.get("detector_train_ratio", 0.05)
    eval_ratio = splits_config.get("eval_ratio", 0.10)

    print(f"Loading DolusChat with splits:")
    print(f"  Detector train: {detector_train_ratio:.1%}")
    print(f"  Eval: {eval_ratio:.1%}")
    print(f"  Policy train: {1 - detector_train_ratio - eval_ratio:.1%}")
    print(f"  Seed: {seed}")

    # Load and split
    doluschat = load_doluschat(
        detector_train_ratio=detector_train_ratio,
        eval_ratio=eval_ratio,
        seed=seed,
    )

    print(f"\nDataset loaded: {doluschat}")

    # Prepare examples
    print("\nPreparing examples...")
    doluschat.detector_train = prepare_dataset(doluschat.detector_train)
    doluschat.policy_train = prepare_dataset(doluschat.policy_train)
    doluschat.eval = prepare_dataset(doluschat.eval)

    # Print statistics
    print("\nDetector train stats:")
    stats = get_example_stats(doluschat.detector_train)
    print(f"  Count: {stats['count']}")
    print(f"  Lie types: {stats['lie_type_distribution']}")
    print(f"  Difficulty: {stats['difficulty_mean']:.2f} +/- {stats['difficulty_std']:.2f}")

    # Save splits
    save_splits(doluschat, args.output_dir)

    print(f"\nDone! Splits saved to {args.output_dir}")


if __name__ == "__main__":
    main()
