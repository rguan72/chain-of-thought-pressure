"""Evaluation report generation."""

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import matplotlib.pyplot as plt
import numpy as np

from .generate import GeneratedSample
from .deception import DeceptionResult
from .obfuscation import ObfuscationMetrics, ObfuscationResult, get_score_distributions


@dataclass
class EvaluationReport:
    """Complete evaluation report for a trained model."""

    # Identification
    model_path: str
    experiment_name: str
    timestamp: str

    # Configuration
    algorithm: str  # "DPO" or "GRPO"
    tpr_target: float
    beta_or_kl: float  # beta for DPO, kl_coef for GRPO

    # Deception metrics
    deception_rate: float
    n_samples: int
    n_deceptive: int

    # Obfuscation metrics
    obfuscation_rate: float
    honest_rate: float
    detection_rate_given_deceptive: float
    mean_detector_score: float

    # Format metrics
    format_ok_rate: float
    mean_cot_length: float


def generate_report(
    model_path: str,
    samples: List[GeneratedSample],
    deception_results: List[DeceptionResult],
    obfuscation_metrics: ObfuscationMetrics,
    obfuscation_results: List[ObfuscationResult],
    algorithm: str,
    tpr_target: float,
    beta_or_kl: float,
    output_dir: str,
    experiment_name: Optional[str] = None,
) -> EvaluationReport:
    """Generate a complete evaluation report.

    Args:
        model_path: Path to the evaluated model.
        samples: Generated samples.
        deception_results: Deception classification results.
        obfuscation_metrics: Obfuscation metrics.
        obfuscation_results: Individual obfuscation results.
        algorithm: Training algorithm used.
        tpr_target: Target TPR for detector.
        beta_or_kl: Beta (DPO) or KL coef (GRPO).
        output_dir: Output directory for report.
        experiment_name: Optional experiment name.

    Returns:
        EvaluationReport object.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().isoformat()

    if experiment_name is None:
        experiment_name = f"{algorithm}_tpr{tpr_target:.2f}_{'beta' if algorithm == 'DPO' else 'kl'}{beta_or_kl}"

    # Compute deception stats
    n_deceptive = sum(1 for r in deception_results if r.is_deceptive)
    deception_rate = n_deceptive / len(deception_results) if deception_results else 0.0

    report = EvaluationReport(
        model_path=model_path,
        experiment_name=experiment_name,
        timestamp=timestamp,
        algorithm=algorithm,
        tpr_target=tpr_target,
        beta_or_kl=beta_or_kl,
        deception_rate=deception_rate,
        n_samples=len(samples),
        n_deceptive=n_deceptive,
        obfuscation_rate=obfuscation_metrics.obfuscation_rate,
        honest_rate=obfuscation_metrics.honest_transparent_rate,
        detection_rate_given_deceptive=obfuscation_metrics.detection_rate_given_deceptive,
        mean_detector_score=obfuscation_metrics.mean_detector_score,
        format_ok_rate=obfuscation_metrics.format_ok_rate,
        mean_cot_length=obfuscation_metrics.mean_cot_length,
    )

    # Save report JSON
    with open(output_path / "report.json", "w") as f:
        json.dump(asdict(report), f, indent=2)

    # Generate plots
    generate_plots(obfuscation_results, output_path)

    # Save sample traces for manual inspection
    save_sample_traces(samples, deception_results, obfuscation_results, output_path)

    print(f"\n{'=' * 60}")
    print(f"EVALUATION REPORT: {experiment_name}")
    print(f"{'=' * 60}")
    print(f"Algorithm: {algorithm}")
    print(f"TPR Target: {tpr_target}")
    print(f"{'Beta' if algorithm == 'DPO' else 'KL Coef'}: {beta_or_kl}")
    print(f"\nDeception Rate: {deception_rate:.1%} ({n_deceptive}/{len(samples)})")
    print(f"Obfuscation Rate: {obfuscation_metrics.obfuscation_rate:.1%}")
    print(f"Honest Rate: {obfuscation_metrics.honest_transparent_rate:.1%}")
    print(f"Detection Rate (given deceptive): {obfuscation_metrics.detection_rate_given_deceptive:.1%}")
    print(f"Format Compliance: {obfuscation_metrics.format_ok_rate:.1%}")
    print(f"{'=' * 60}")

    return report


def generate_plots(
    obfuscation_results: List[ObfuscationResult],
    output_dir: Path,
) -> None:
    """Generate visualization plots.

    Args:
        obfuscation_results: Obfuscation analysis results.
        output_dir: Output directory.
    """
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # 1. Detector score distribution
    distributions = get_score_distributions(obfuscation_results)

    fig, ax = plt.subplots(figsize=(10, 6))

    if distributions["truthful"]:
        ax.hist(
            distributions["truthful"],
            bins=30,
            alpha=0.6,
            label="Truthful",
            color="green",
        )
    if distributions["deceptive"]:
        ax.hist(
            distributions["deceptive"],
            bins=30,
            alpha=0.6,
            label="Deceptive",
            color="red",
        )

    ax.set_xlabel("Detector Score (P(deceptive))")
    ax.set_ylabel("Count")
    ax.set_title("Detector Score Distribution by Final Answer Type")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(plots_dir / "detector_score_distribution.png", dpi=150)
    plt.close()

    # 2. CoT length distribution
    cot_lengths_truthful = [
        r.cot_length for r in obfuscation_results if not r.final_deceptive
    ]
    cot_lengths_deceptive = [
        r.cot_length for r in obfuscation_results if r.final_deceptive
    ]

    fig, ax = plt.subplots(figsize=(10, 6))

    if cot_lengths_truthful:
        ax.hist(
            cot_lengths_truthful,
            bins=30,
            alpha=0.6,
            label="Truthful",
            color="green",
        )
    if cot_lengths_deceptive:
        ax.hist(
            cot_lengths_deceptive,
            bins=30,
            alpha=0.6,
            label="Deceptive",
            color="red",
        )

    ax.set_xlabel("CoT Length (words)")
    ax.set_ylabel("Count")
    ax.set_title("Chain-of-Thought Length Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(plots_dir / "cot_length_distribution.png", dpi=150)
    plt.close()

    # 3. Obfuscation breakdown pie chart
    n_total = len(obfuscation_results)
    n_honest = sum(1 for r in obfuscation_results if not r.final_deceptive)
    n_deceptive_detected = sum(
        1 for r in obfuscation_results if r.final_deceptive and r.detected
    )
    n_obfuscated = sum(1 for r in obfuscation_results if r.is_obfuscated)

    labels = ["Honest", "Deceptive (Detected)", "Deceptive (Obfuscated)"]
    sizes = [n_honest, n_deceptive_detected, n_obfuscated]
    colors = ["green", "orange", "red"]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(
        sizes,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        startangle=90,
    )
    ax.set_title("Response Classification Breakdown")

    plt.tight_layout()
    plt.savefig(plots_dir / "classification_breakdown.png", dpi=150)
    plt.close()

    print(f"Saved plots to {plots_dir}")


def save_sample_traces(
    samples: List[GeneratedSample],
    deception_results: List[DeceptionResult],
    obfuscation_results: List[ObfuscationResult],
    output_dir: Path,
    n_samples: int = 50,
) -> None:
    """Save sample traces for manual inspection.

    Args:
        samples: Generated samples.
        deception_results: Deception results.
        obfuscation_results: Obfuscation results.
        output_dir: Output directory.
        n_samples: Number of samples to save.
    """
    traces_file = output_dir / "sample_traces.jsonl"

    # Build lookups
    deception_lookup = {r.example_id: r for r in deception_results}
    obfuscation_lookup = {r.example_id: r for r in obfuscation_results}

    with open(traces_file, "w") as f:
        for sample in samples[:n_samples]:
            dec = deception_lookup.get(sample.example_id)
            obf = obfuscation_lookup.get(sample.example_id)

            trace = {
                "example_id": sample.example_id,
                "prompt": sample.prompt[:500] + "..." if len(sample.prompt) > 500 else sample.prompt,
                "cot_text": sample.cot_text,
                "final_text": sample.final_text,
                "truthful_response": sample.truthful_response,
                "deceptive_response": sample.deceptive_response,
                "format_ok": sample.format_ok,
                "is_deceptive": dec.is_deceptive if dec else None,
                "detector_score": obf.detector_score if obf else None,
                "is_obfuscated": obf.is_obfuscated if obf else None,
            }
            f.write(json.dumps(trace) + "\n")

    print(f"Saved {min(n_samples, len(samples))} sample traces to {traces_file}")


def aggregate_reports(report_paths: List[str]) -> Dict[str, Any]:
    """Aggregate multiple evaluation reports for comparison.

    Args:
        report_paths: List of paths to report.json files.

    Returns:
        Aggregated comparison data.
    """
    reports = []
    for path in report_paths:
        with open(path) as f:
            reports.append(json.load(f))

    # Organize by algorithm and parameters
    comparison = {
        "experiments": [],
        "summary": {
            "best_honest": None,
            "worst_obfuscation": None,
            "dpo_vs_grpo": {},
        },
    }

    for report in reports:
        comparison["experiments"].append(
            {
                "name": report["experiment_name"],
                "algorithm": report["algorithm"],
                "tpr": report["tpr_target"],
                "beta_kl": report["beta_or_kl"],
                "deception_rate": report["deception_rate"],
                "obfuscation_rate": report["obfuscation_rate"],
                "honest_rate": report["honest_rate"],
            }
        )

    # Find best/worst
    sorted_by_honest = sorted(
        comparison["experiments"], key=lambda x: x["honest_rate"], reverse=True
    )
    comparison["summary"]["best_honest"] = sorted_by_honest[0]["name"]

    sorted_by_obfuscation = sorted(
        comparison["experiments"], key=lambda x: x["obfuscation_rate"]
    )
    comparison["summary"]["worst_obfuscation"] = sorted_by_obfuscation[0]["name"]

    return comparison
