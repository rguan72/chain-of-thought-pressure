#!/usr/bin/env python3
"""Train DPO models."""
import unsloth
import argparse
import yaml
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.labeling.solid import load_preference_dataset
from src.training.dpo import train_dpo, DPOConfig
from src.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser(description="Train DPO model")
    parser.add_argument("--config", type=str, default="configs/train_dpo.yaml", help="Config file")
    parser.add_argument("--preferences-dir", type=str, default="data/preferences", help="Preferences directory")
    parser.add_argument("--sft-adapter", type=str, default="runs/sft/adapter", help="SFT adapter path")
    parser.add_argument("--output-dir", type=str, default="runs/dpo", help="Output directory")
    parser.add_argument("--tpr-target", type=float, required=True, help="TPR target for preferences")
    parser.add_argument("--beta", type=float, required=True, help="DPO beta value")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    seed = args.seed
    set_seed(seed)

    dpo_config = config.get("dpo", {})
    lora_config = dpo_config.get("lora", {})
    training_config = dpo_config.get("training", {})

    print(f"Training DPO model")
    print(f"  TPR target: {args.tpr_target}")
    print(f"  Beta: {args.beta}")
    print(f"  SFT adapter: {args.sft_adapter}")

    # Load preferences
    print(f"\nLoading preferences for TPR {args.tpr_target}")
    preferences, stats = load_preference_dataset(args.preferences_dir, args.tpr_target)
    print(f"Loaded {len(preferences)} preference pairs")
    print(f"  Truthful chosen: {stats['truthful_chosen_rate']:.1%}")
    print(f"  Detection rate: {stats['detection_rate']:.1%}")

    # Create config
    train_config = DPOConfig(
        sft_adapter_path=args.sft_adapter,
        output_dir=args.output_dir,
        lora_r=lora_config.get("r", 64),
        lora_alpha=lora_config.get("lora_alpha", 128),
        lora_dropout=lora_config.get("lora_dropout", 0.05),
        beta=args.beta,
        epochs=training_config.get("epochs", 2),
        batch_size=training_config.get("batch_size", 4),
        gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 8),
        learning_rate=training_config.get("learning_rate", 1e-5),
        warmup_ratio=training_config.get("warmup_ratio", 0.1),
        max_seq_length=training_config.get("max_seq_length", 2048),
        max_prompt_length=training_config.get("max_prompt_length", 1024),
        seed=seed,
    )

    # Train
    adapter_path = train_dpo(preferences, train_config, tpr_target=args.tpr_target)

    print(f"\nDPO training complete!")
    print(f"  Adapter saved to: {adapter_path}")


if __name__ == "__main__":
    main()
