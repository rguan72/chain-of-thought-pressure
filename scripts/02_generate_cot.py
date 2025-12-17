#!/usr/bin/env python3
"""Generate CoT training data for the detector."""

import argparse
import yaml
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.doluschat import load_splits
from src.data.cot_generator import CoTGenerator, save_cot_examples
from src.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser(description="Generate CoT for detector training")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Config file")
    parser.add_argument("--splits-dir", type=str, default="data/splits", help="Splits directory")
    parser.add_argument("--output", type=str, default="data/cot_detector_train.jsonl", help="Output file")
    parser.add_argument("--model", type=str, default=None, help="Model for CoT generation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    seed = args.seed or config.get("seed", 42)
    set_seed(seed)

    model_name = args.model or config.get("model", {}).get("name", "Qwen/Qwen3-0.6B")

    print(f"Generating CoT using: {model_name}")
    print(f"Seed: {seed}")

    # Load detector_train split
    print(f"\nLoading splits from {args.splits_dir}")
    doluschat = load_splits(args.splits_dir)

    print(f"Detector train size: {len(doluschat.detector_train)}")

    # Initialize generator
    generator = CoTGenerator(
        model_name=model_name,
        max_new_tokens=512,
        temperature=0.7,
    )

    # Generate CoT for all examples
    print("\nGenerating CoT examples...")
    examples = [dict(ex) for ex in doluschat.detector_train]
    cot_examples = generator.generate_from_dataset(doluschat.detector_train, args.output)

    print(f"\nGenerated {len(cot_examples)} CoT examples")
    print(f"  Truthful: {sum(1 for e in cot_examples if e.label == 0)}")
    print(f"  Deceptive: {sum(1 for e in cot_examples if e.label == 1)}")
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
