#!/usr/bin/env python3
"""Train SFT reference model."""

import argparse
import yaml
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.doluschat import load_splits
from src.training.sft import train_sft, SFTConfig
from src.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser(description="Train SFT reference model")
    parser.add_argument("--config", type=str, default="configs/train_sft.yaml", help="Config file")
    parser.add_argument("--splits-dir", type=str, default="data/splits", help="Splits directory")
    parser.add_argument("--output-dir", type=str, default="runs/sft", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    seed = args.seed
    set_seed(seed)

    sft_config = config.get("sft", {})
    lora_config = sft_config.get("lora", {})
    training_config = sft_config.get("training", {})

    print(f"Training SFT reference model")
    print(f"  LoRA rank: {lora_config.get('r', 64)}")
    print(f"  Epochs: {training_config.get('epochs', 1)}")
    print(f"  Batch size: {training_config.get('batch_size', 8)}")

    # Load data
    print(f"\nLoading policy train split from {args.splits_dir}")
    doluschat = load_splits(args.splits_dir)
    policy_train = [dict(ex) for ex in doluschat.policy_train]
    print(f"Policy train size: {len(policy_train)}")

    # Create config
    train_config = SFTConfig(
        output_dir=args.output_dir,
        lora_r=lora_config.get("r", 64),
        lora_alpha=lora_config.get("lora_alpha", 128),
        lora_dropout=lora_config.get("lora_dropout", 0.05),
        epochs=training_config.get("epochs", 1),
        batch_size=training_config.get("batch_size", 8),
        gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 4),
        learning_rate=training_config.get("learning_rate", 2e-5),
        warmup_ratio=training_config.get("warmup_ratio", 0.03),
        max_seq_length=training_config.get("max_seq_length", 2048),
        seed=seed,
    )

    # Train
    adapter_path = train_sft(policy_train, train_config)

    print(f"\nSFT training complete!")
    print(f"  Adapter saved to: {adapter_path}")


if __name__ == "__main__":
    main()
