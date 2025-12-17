#!/usr/bin/env python3
"""Generate SOLiD preference datasets for each TPR target."""

import argparse
import yaml
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.doluschat import load_splits
from src.data.cot_generator import CoTGenerator
from src.detector.model import CoTLieDetector
from src.detector.calibrate import load_calibration, get_threshold_for_tpr
from src.labeling.solid import generate_preference_dataset, save_preference_dataset, RewardConfig
from src.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser(description="Generate preference datasets")
    parser.add_argument("--config", type=str, default="configs/labeling.yaml", help="Config file")
    parser.add_argument("--splits-dir", type=str, default="data/splits", help="Splits directory")
    parser.add_argument("--detector-path", type=str, default="runs/detector/best_model", help="Detector path")
    parser.add_argument("--thresholds", type=str, default="data/detector_thresholds.json", help="Thresholds file")
    parser.add_argument("--output-dir", type=str, default="data/preferences", help="Output directory")
    parser.add_argument("--tpr-targets", type=float, nargs="+", default=[0.75, 0.90], help="TPR targets")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    seed = args.seed
    set_seed(seed)

    rewards_config = config.get("rewards", {})
    reward_config = RewardConfig(
        truthful=rewards_config.get("truthful", 1.0),
        lie_undetected=rewards_config.get("lie_undetected", 2.0),
        lie_detected=rewards_config.get("lie_detected", -1.0),
    )

    print(f"Generating preference datasets")
    print(f"  Rewards: truth={reward_config.truthful}, lie_undetected={reward_config.lie_undetected}, lie_detected={reward_config.lie_detected}")
    print(f"  TPR targets: {args.tpr_targets}")

    # Load data
    print(f"\nLoading policy train split from {args.splits_dir}")
    doluschat = load_splits(args.splits_dir)
    policy_train = [dict(ex) for ex in doluschat.policy_train]
    print(f"Policy train size: {len(policy_train)}")

    # Load detector
    print(f"\nLoading detector from {args.detector_path}")
    detector = CoTLieDetector.load(args.detector_path)

    # Load calibration
    print(f"Loading calibration from {args.thresholds}")
    calibration = load_calibration(args.thresholds)

    # Generate CoT for deceptive responses (for detector scoring)
    print(f"\nGenerating CoT for deceptive responses...")
    generator = CoTGenerator(max_new_tokens=512)

    # Prepare deceptive prompts and generate CoT
    from src.utils.prompts import build_teacher_cot_prompt

    deceptive_cots = []
    prompts_for_cot = []

    for ex in policy_train:
        deceptive_resp = ex.get("deceptive_response", "")
        prompt = build_teacher_cot_prompt(ex, deceptive_resp)
        prompts_for_cot.append(prompt)

    # Generate in batch
    llm, sampling_params = generator._init_vllm()
    outputs = llm.generate(prompts_for_cot, sampling_params)
    deceptive_cots = [o.outputs[0].text.strip() for o in outputs]

    print(f"Generated {len(deceptive_cots)} CoTs")

    # Generate preference dataset for each TPR target
    for tpr_target in args.tpr_targets:
        print(f"\n{'=' * 50}")
        print(f"Generating preferences for TPR target: {tpr_target}")
        print(f"{'=' * 50}")

        threshold = get_threshold_for_tpr(calibration, tpr_target)
        if threshold is None:
            print(f"  Warning: No threshold found for TPR {tpr_target}, skipping")
            continue

        print(f"  Using threshold: {threshold:.4f}")

        preferences, stats = generate_preference_dataset(
            examples=policy_train,
            deceptive_cots=deceptive_cots,
            detector=detector,
            threshold=threshold,
            reward_config=reward_config,
            seed=seed,
        )

        save_preference_dataset(preferences, stats, args.output_dir, tpr_target)

    print(f"\nDone! Preference datasets saved to {args.output_dir}")


if __name__ == "__main__":
    main()
