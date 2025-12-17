#!/usr/bin/env python3
"""Train GRPO models with two-branch rollouts.

This implementation forces both truthful and deceptive rollouts explicitly:
- For each prompt, generate k samples with "Answer truthfully" suffix
- And k samples with "Answer deceptively" suffix
- Verify if model actually followed instruction using forced-choice judge
- Reward: truth_ok=+1, lie_ok_uncaught=+2, lie_ok_caught=-1, failed=0
"""
import unsloth
import argparse
import yaml
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.doluschat import load_splits
from src.detector.model import CoTLieDetector
from src.detector.calibrate import load_calibration, get_threshold_for_tpr
from src.training.grpo import train_grpo, GRPOConfig
from src.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser(description="Train GRPO model with two-branch rollouts")
    parser.add_argument("--config", type=str, default="configs/train_grpo.yaml", help="Config file")
    parser.add_argument("--splits-dir", type=str, default="data/splits", help="Splits directory")
    parser.add_argument("--sft-adapter", type=str, default="runs/sft/adapter", help="SFT adapter path")
    parser.add_argument("--detector-path", type=str, default="runs/detector/best_model", help="Detector path")
    parser.add_argument("--thresholds", type=str, default="data/detector_thresholds.json", help="Thresholds file")
    parser.add_argument("--output-dir", type=str, default="runs/grpo", help="Output directory")
    parser.add_argument("--tpr-target", type=float, required=True, help="TPR target")
    parser.add_argument("--kl-coef", type=float, required=True, help="KL coefficient")
    parser.add_argument("--samples-per-branch", type=int, default=2, help="Samples per branch (k)")
    parser.add_argument("--temperature", type=float, default=0.95, help="Sampling temperature")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    seed = args.seed
    set_seed(seed)

    grpo_config = config.get("grpo", {})
    lora_config = grpo_config.get("lora", {})
    training_config = grpo_config.get("training", {})
    gen_config = grpo_config.get("generation", {})
    rewards_config = grpo_config.get("rewards", {})

    print(f"Training GRPO model with two-branch rollouts")
    print(f"  TPR target: {args.tpr_target}")
    print(f"  KL coef: {args.kl_coef}")
    print(f"  Samples per branch: {args.samples_per_branch}")
    print(f"  Temperature: {args.temperature}")
    print(f"  SFT adapter: {args.sft_adapter}")

    # Load data
    print(f"\nLoading policy train split from {args.splits_dir}")
    doluschat = load_splits(args.splits_dir)
    policy_train = [dict(ex) for ex in doluschat.policy_train]
    print(f"Policy train size: {len(policy_train)}")

    # Load detector
    print(f"\nLoading detector from {args.detector_path}")
    detector = CoTLieDetector.load(args.detector_path)

    # Get threshold
    calibration = load_calibration(args.thresholds)
    threshold = get_threshold_for_tpr(calibration, args.tpr_target)
    print(f"Using threshold {threshold:.4f} for TPR {args.tpr_target}")

    # Create config
    train_config = GRPOConfig(
        sft_adapter_path=args.sft_adapter,
        output_dir=args.output_dir,

        # LoRA
        lora_r=lora_config.get("r", 64),
        lora_alpha=lora_config.get("lora_alpha", 128),
        lora_dropout=lora_config.get("lora_dropout", 0.05),

        # GRPO-specific
        samples_per_branch=args.samples_per_branch,
        kl_coef=args.kl_coef,

        # Rewards
        reward_truth_ok=rewards_config.get("truth_ok", 1.0),
        reward_lie_uncaught=rewards_config.get("lie_uncaught", 2.0),
        reward_lie_caught=rewards_config.get("lie_caught", -1.0),
        reward_failed=rewards_config.get("failed", 0.0),

        # Training
        num_epochs=training_config.get("num_epochs", 1),
        batch_size=training_config.get("batch_size", 4),
        gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 4),
        learning_rate=training_config.get("learning_rate", 5e-6),
        warmup_ratio=training_config.get("warmup_ratio", 0.1),
        max_grad_norm=training_config.get("max_grad_norm", 1.0),
        max_prompt_length=training_config.get("max_prompt_length", 1024),
        max_completion_length=training_config.get("max_completion_length", 512),

        # Generation
        temperature=args.temperature,
        top_p=gen_config.get("top_p", 0.95),

        seed=seed,
    )

    # Train
    adapter_path = train_grpo(
        policy_train, detector, threshold, train_config, tpr_target=args.tpr_target
    )

    print(f"\nGRPO training complete!")
    print(f"  Adapter saved to: {adapter_path}")


if __name__ == "__main__":
    main()
