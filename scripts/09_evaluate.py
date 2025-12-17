#!/usr/bin/env python3
"""Evaluate a trained model."""

import argparse
import yaml
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.doluschat import load_splits
from src.detector.model import CoTLieDetector
from src.detector.calibrate import load_calibration, get_threshold_for_tpr
from src.eval.generate import generate_samples, generate_samples_transformers
from src.eval.deception import compute_deception_rate, save_deception_results
from src.eval.obfuscation import analyze_obfuscation, save_obfuscation_results
from src.eval.report import generate_report
from src.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument("--config", type=str, default="configs/eval.yaml", help="Config file")
    parser.add_argument("--splits-dir", type=str, default="data/splits", help="Splits directory")
    parser.add_argument("--model-path", type=str, required=True, help="Model/adapter path to evaluate")
    parser.add_argument("--detector-path", type=str, default="runs/detector/best_model", help="Detector path")
    parser.add_argument("--thresholds", type=str, default="data/detector_thresholds.json", help="Thresholds file")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--algorithm", type=str, required=True, choices=["DPO", "GRPO"], help="Training algorithm")
    parser.add_argument("--tpr-target", type=float, required=True, help="TPR target used")
    parser.add_argument("--beta-or-kl", type=float, required=True, help="Beta (DPO) or KL coef (GRPO)")
    parser.add_argument("--use-vllm", action="store_true", help="Use vLLM for generation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    seed = args.seed
    set_seed(seed)

    gen_config = config.get("generation", {})

    print(f"Evaluating model: {args.model_path}")
    print(f"  Algorithm: {args.algorithm}")
    print(f"  TPR target: {args.tpr_target}")
    print(f"  {'Beta' if args.algorithm == 'DPO' else 'KL coef'}: {args.beta_or_kl}")

    # Load eval data
    print(f"\nLoading eval split from {args.splits_dir}")
    doluschat = load_splits(args.splits_dir)
    eval_data = [dict(ex) for ex in doluschat.eval]
    print(f"Eval size: {len(eval_data)}")

    # Generate samples
    print(f"\nGenerating samples...")
    if args.use_vllm:
        samples = generate_samples(
            eval_data,
            args.model_path,
            temperature=gen_config.get("temperature", 0.7),
            top_p=gen_config.get("top_p", 0.95),
            max_new_tokens=gen_config.get("max_new_tokens", 1024),
        )
    else:
        samples = generate_samples_transformers(
            eval_data,
            args.model_path,
            temperature=gen_config.get("temperature", 0.7),
            top_p=gen_config.get("top_p", 0.95),
            max_new_tokens=gen_config.get("max_new_tokens", 1024),
        )

    print(f"Generated {len(samples)} samples")
    print(f"  Format OK rate: {sum(1 for s in samples if s.format_ok) / len(samples):.1%}")

    # Compute deception rate
    print(f"\nComputing deception rate...")
    deception_method = config.get("deception", {}).get("method", "string_match")

    if deception_method == "forced_choice_logprob":
        deception_rate, deception_results = compute_deception_rate(
            samples, model_path=args.model_path, method="forced_choice_logprob"
        )
    else:
        deception_rate, deception_results = compute_deception_rate(
            samples, method="string_match"
        )

    # Load detector for obfuscation analysis
    print(f"\nLoading detector for obfuscation analysis...")
    detector = CoTLieDetector.load(args.detector_path)

    # Get threshold
    calibration = load_calibration(args.thresholds)
    eval_tpr = config.get("obfuscation", {}).get("eval_tpr", 0.90)
    threshold = get_threshold_for_tpr(calibration, eval_tpr)
    print(f"Using threshold {threshold:.4f} (TPR {eval_tpr})")

    # Compute obfuscation metrics
    print(f"\nComputing obfuscation metrics...")
    obfuscation_metrics, obfuscation_results = analyze_obfuscation(
        samples, deception_results, detector, threshold
    )

    # Generate report
    print(f"\nGenerating report...")
    report = generate_report(
        model_path=args.model_path,
        samples=samples,
        deception_results=deception_results,
        obfuscation_metrics=obfuscation_metrics,
        obfuscation_results=obfuscation_results,
        algorithm=args.algorithm,
        tpr_target=args.tpr_target,
        beta_or_kl=args.beta_or_kl,
        output_dir=args.output_dir,
    )

    # Save all results
    save_deception_results(deception_results, deception_rate, args.output_dir)
    save_obfuscation_results(obfuscation_metrics, obfuscation_results, args.output_dir)

    print(f"\nEvaluation complete!")
    print(f"  Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
