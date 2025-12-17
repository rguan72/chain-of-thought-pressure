#!/usr/bin/env python3
"""Run the full experiment grid."""

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple
import json


def run_command(cmd: List[str], desc: str) -> bool:
    """Run a command and return success status."""
    print(f"\n{'=' * 60}")
    print(f"Running: {desc}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'=' * 60}\n")

    result = subprocess.run(cmd, cwd=str(Path(__file__).parent.parent))
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Run full experiment grid")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--tpr-targets", type=float, nargs="+", default=[0.75, 0.90], help="TPR targets")
    parser.add_argument("--dpo-betas", type=float, nargs="+", default=[0.1, 0.3], help="DPO beta values")
    parser.add_argument("--grpo-kls", type=float, nargs="+", default=[0.05, 0.2], help="GRPO KL coefficients")
    parser.add_argument("--skip-data-prep", action="store_true", help="Skip data preparation")
    parser.add_argument("--skip-detector", action="store_true", help="Skip detector training")
    parser.add_argument("--skip-sft", action="store_true", help="Skip SFT training")
    parser.add_argument("--skip-dpo", action="store_true", help="Skip DPO experiments")
    parser.add_argument("--skip-grpo", action="store_true", help="Skip GRPO experiments")
    parser.add_argument("--eval-only", action="store_true", help="Only run evaluation")
    args = parser.parse_args()

    python = sys.executable
    scripts_dir = Path(__file__).parent

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Track experiments
    experiments = []

    # Phase 1: Data preparation
    if not args.skip_data_prep and not args.eval_only:
        print("\n" + "=" * 60)
        print("PHASE 1: DATA PREPARATION")
        print("=" * 60)

        success = run_command(
            [python, str(scripts_dir / "01_prepare_data.py"), "--seed", str(args.seed)],
            "Prepare DolusChat dataset"
        )
        if not success:
            print("ERROR: Data preparation failed")
            return 1

        success = run_command(
            [python, str(scripts_dir / "02_generate_cot.py"), "--seed", str(args.seed)],
            "Generate CoT for detector training"
        )
        if not success:
            print("ERROR: CoT generation failed")
            return 1

    # Phase 2: Detector training
    if not args.skip_detector and not args.eval_only:
        print("\n" + "=" * 60)
        print("PHASE 2: DETECTOR TRAINING")
        print("=" * 60)

        success = run_command(
            [python, str(scripts_dir / "03_train_detector.py"), "--seed", str(args.seed)],
            "Train CoT lie detector"
        )
        if not success:
            print("ERROR: Detector training failed")
            return 1

        success = run_command(
            [python, str(scripts_dir / "04_calibrate_detector.py"), "--seed", str(args.seed)],
            "Calibrate detector thresholds"
        )
        if not success:
            print("ERROR: Detector calibration failed")
            return 1

    # Phase 3: Preference generation
    if not args.eval_only:
        print("\n" + "=" * 60)
        print("PHASE 3: PREFERENCE DATASET GENERATION")
        print("=" * 60)

        tpr_args = []
        for tpr in args.tpr_targets:
            tpr_args.extend(["--tpr-targets", str(tpr)])

        success = run_command(
            [python, str(scripts_dir / "05_generate_preferences.py"), "--seed", str(args.seed)] + tpr_args,
            "Generate preference datasets"
        )
        if not success:
            print("ERROR: Preference generation failed")
            return 1

    # Phase 4: SFT training
    if not args.skip_sft and not args.eval_only:
        print("\n" + "=" * 60)
        print("PHASE 4: SFT REFERENCE MODEL")
        print("=" * 60)

        success = run_command(
            [python, str(scripts_dir / "06_train_sft.py"), "--seed", str(args.seed)],
            "Train SFT reference model"
        )
        if not success:
            print("ERROR: SFT training failed")
            return 1

    # Phase 5: DPO experiments
    if not args.skip_dpo:
        print("\n" + "=" * 60)
        print("PHASE 5: DPO EXPERIMENTS")
        print("=" * 60)

        for tpr in args.tpr_targets:
            for beta in args.dpo_betas:
                exp_name = f"dpo_tpr{tpr:.2f}_beta{beta}"

                if not args.eval_only:
                    success = run_command(
                        [
                            python, str(scripts_dir / "07_train_dpo.py"),
                            "--tpr-target", str(tpr),
                            "--beta", str(beta),
                            "--seed", str(args.seed),
                        ],
                        f"Train DPO (TPR={tpr}, beta={beta})"
                    )
                    if not success:
                        print(f"ERROR: DPO training failed for {exp_name}")
                        continue

                # Evaluate
                adapter_path = f"runs/dpo/tpr{tpr:.2f}_beta{beta}/adapter"
                output_dir = f"runs/eval/{timestamp}_{exp_name}"

                success = run_command(
                    [
                        python, str(scripts_dir / "09_evaluate.py"),
                        "--model-path", adapter_path,
                        "--output-dir", output_dir,
                        "--algorithm", "DPO",
                        "--tpr-target", str(tpr),
                        "--beta-or-kl", str(beta),
                        "--seed", str(args.seed),
                    ],
                    f"Evaluate DPO (TPR={tpr}, beta={beta})"
                )

                experiments.append({
                    "name": exp_name,
                    "algorithm": "DPO",
                    "tpr": tpr,
                    "beta": beta,
                    "output_dir": output_dir,
                    "success": success,
                })

    # Phase 6: GRPO experiments
    if not args.skip_grpo:
        print("\n" + "=" * 60)
        print("PHASE 6: GRPO EXPERIMENTS")
        print("=" * 60)

        for tpr in args.tpr_targets:
            for kl in args.grpo_kls:
                exp_name = f"grpo_tpr{tpr:.2f}_kl{kl}"

                if not args.eval_only:
                    success = run_command(
                        [
                            python, str(scripts_dir / "08_train_grpo.py"),
                            "--tpr-target", str(tpr),
                            "--kl-coef", str(kl),
                            "--seed", str(args.seed),
                        ],
                        f"Train GRPO (TPR={tpr}, KL={kl})"
                    )
                    if not success:
                        print(f"ERROR: GRPO training failed for {exp_name}")
                        continue

                # Evaluate
                adapter_path = f"runs/grpo/tpr{tpr:.2f}_kl{kl}/adapter"
                output_dir = f"runs/eval/{timestamp}_{exp_name}"

                success = run_command(
                    [
                        python, str(scripts_dir / "09_evaluate.py"),
                        "--model-path", adapter_path,
                        "--output-dir", output_dir,
                        "--algorithm", "GRPO",
                        "--tpr-target", str(tpr),
                        "--beta-or-kl", str(kl),
                        "--seed", str(args.seed),
                    ],
                    f"Evaluate GRPO (TPR={tpr}, KL={kl})"
                )

                experiments.append({
                    "name": exp_name,
                    "algorithm": "GRPO",
                    "tpr": tpr,
                    "kl_coef": kl,
                    "output_dir": output_dir,
                    "success": success,
                })

    # Summary
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)

    successful = [e for e in experiments if e["success"]]
    failed = [e for e in experiments if not e["success"]]

    print(f"\nTotal experiments: {len(experiments)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")

    if successful:
        print("\nSuccessful experiments:")
        for exp in successful:
            print(f"  - {exp['name']}: {exp['output_dir']}")

    if failed:
        print("\nFailed experiments:")
        for exp in failed:
            print(f"  - {exp['name']}")

    # Save experiment log
    log_path = f"runs/experiment_log_{timestamp}.json"
    Path("runs").mkdir(exist_ok=True)
    with open(log_path, "w") as f:
        json.dump({
            "timestamp": timestamp,
            "seed": args.seed,
            "experiments": experiments,
        }, f, indent=2)

    print(f"\nExperiment log saved to: {log_path}")

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
