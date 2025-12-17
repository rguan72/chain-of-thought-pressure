#!/usr/bin/env python3
"""Train the CoT lie detector."""

import argparse
import yaml
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.cot_generator import load_cot_examples
from src.detector.train import train_detector, DetectorTrainingConfig
from src.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser(description="Train CoT lie detector")
    parser.add_argument("--config", type=str, default="configs/detector.yaml", help="Config file")
    parser.add_argument("--cot-data", type=str, default="data/cot_detector_train.jsonl", help="CoT training data")
    parser.add_argument("--output-dir", type=str, default="runs/detector", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    seed = args.seed
    set_seed(seed)

    detector_config = config.get("detector", {})
    training_config = config.get("training", {})

    print(f"Training CoT lie detector")
    print(f"  Model: {detector_config.get('model_name', 'distilroberta-base')}")
    print(f"  Epochs: {training_config.get('epochs', 5)}")
    print(f"  Batch size: {training_config.get('batch_size', 32)}")

    # Load CoT data
    print(f"\nLoading CoT data from {args.cot_data}")
    cot_examples = load_cot_examples(args.cot_data)
    print(f"Loaded {len(cot_examples)} examples")

    # Create training config
    train_config = DetectorTrainingConfig(
        model_name=detector_config.get("model_name", "distilroberta-base"),
        max_length=detector_config.get("max_length", 512),
        epochs=training_config.get("epochs", 5),
        batch_size=training_config.get("batch_size", 32),
        learning_rate=training_config.get("learning_rate", 2e-5),
        weight_decay=training_config.get("weight_decay", 0.01),
        warmup_ratio=training_config.get("warmup_ratio", 0.1),
        val_split=training_config.get("val_split", 0.1),
        output_dir=args.output_dir,
        seed=seed,
    )

    # Train
    print("\nStarting training...")
    model, history = train_detector(cot_examples, train_config)

    print(f"\nTraining complete!")
    print(f"  Best AUC: {history['best_auc']:.4f}")
    print(f"  Best epoch: {history['best_epoch']}")
    print(f"  Model saved to: {args.output_dir}/best_model")


if __name__ == "__main__":
    main()
