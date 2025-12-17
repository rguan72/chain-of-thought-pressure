"""DolusChat dataset loader and splitting utilities."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from datasets import Dataset, load_dataset
import numpy as np

from ..utils.seed import set_seed, get_seed
from ..utils.prompts import build_prompt


@dataclass
class DolusChat:
    """Container for DolusChat dataset splits."""

    detector_train: Dataset
    policy_train: Dataset
    eval: Dataset

    # Metadata
    total_examples: int
    detector_train_size: int
    policy_train_size: int
    eval_size: int

    def __repr__(self) -> str:
        return (
            f"DolusChat(total={self.total_examples}, "
            f"detector_train={self.detector_train_size}, "
            f"policy_train={self.policy_train_size}, "
            f"eval={self.eval_size})"
        )


def load_doluschat(
    detector_train_ratio: float = 0.05,
    eval_ratio: float = 0.10,
    seed: int = 42,
    cache_dir: Optional[str] = None,
) -> DolusChat:
    """Load DolusChat dataset and create splits.

    Args:
        detector_train_ratio: Fraction of data for detector training (default 5%).
        eval_ratio: Fraction of data for evaluation (default 10%).
        seed: Random seed for reproducibility.
        cache_dir: Optional cache directory for the dataset.

    Returns:
        DolusChat object with train/eval splits.
    """
    set_seed(seed)

    # Load from HuggingFace
    print("Loading DolusChat from HuggingFace...")
    dataset = load_dataset("AlignmentResearch/DolusChat", cache_dir=cache_dir)

    # Get the train split (DolusChat only has train)
    full_data = dataset["train"]
    total = len(full_data)
    print(f"Loaded {total} examples")

    # Calculate split sizes
    detector_size = int(total * detector_train_ratio)
    eval_size = int(total * eval_ratio)
    policy_size = total - detector_size - eval_size

    print(f"Splitting: detector_train={detector_size}, policy_train={policy_size}, eval={eval_size}")

    # Shuffle and split
    indices = np.random.permutation(total)

    detector_indices = indices[:detector_size]
    eval_indices = indices[detector_size : detector_size + eval_size]
    policy_indices = indices[detector_size + eval_size :]

    # Create splits
    detector_train = full_data.select(detector_indices.tolist())
    eval_split = full_data.select(eval_indices.tolist())
    policy_train = full_data.select(policy_indices.tolist())

    return DolusChat(
        detector_train=detector_train,
        policy_train=policy_train,
        eval=eval_split,
        total_examples=total,
        detector_train_size=detector_size,
        policy_train_size=policy_size,
        eval_size=eval_size,
    )


def prepare_example(example: Dict[str, Any]) -> Dict[str, Any]:
    """Prepare a DolusChat example for training/evaluation.

    Adds computed fields like full prompt and extracts responses.

    Args:
        example: Raw DolusChat example.

    Returns:
        Prepared example with additional fields.
    """
    # Build the full prompt
    prompt = build_prompt(example, include_cot_instruction=True)

    # Extract responses
    responses = example.get("responses", {})
    truthful = responses.get("truthful", "")
    deceptive = responses.get("deceptive", "")

    # Extract metadata
    context = example.get("context", {})
    sensitive_info = context.get("sensitive_info", "")

    return {
        "id": example.get("id", ""),
        "prompt": prompt,
        "truthful_response": truthful,
        "deceptive_response": deceptive,
        "sensitive_info": sensitive_info,
        "lie_type": example.get("lie_type", ""),
        "deception_difficulty": example.get("deception_difficulty", 0.0),
        # Keep original fields for reference
        "system_message": example.get("system_message", ""),
        "context": context,
        "user_query": example.get("user_query", {}),
    }


def prepare_dataset(dataset: Dataset) -> Dataset:
    """Apply preparation to entire dataset.

    Args:
        dataset: Raw DolusChat dataset.

    Returns:
        Dataset with prepared examples.
    """
    return dataset.map(prepare_example, desc="Preparing examples")


def save_splits(doluschat: DolusChat, output_dir: str) -> None:
    """Save dataset splits to disk.

    Args:
        doluschat: DolusChat object with splits.
        output_dir: Directory to save splits.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save each split
    doluschat.detector_train.save_to_disk(str(output_path / "detector_train"))
    doluschat.policy_train.save_to_disk(str(output_path / "policy_train"))
    doluschat.eval.save_to_disk(str(output_path / "eval"))

    # Save metadata
    metadata = {
        "total_examples": doluschat.total_examples,
        "detector_train_size": doluschat.detector_train_size,
        "policy_train_size": doluschat.policy_train_size,
        "eval_size": doluschat.eval_size,
        "seed": get_seed(),
    }

    with open(output_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved splits to {output_path}")


def load_splits(input_dir: str) -> DolusChat:
    """Load dataset splits from disk.

    Args:
        input_dir: Directory containing saved splits.

    Returns:
        DolusChat object with loaded splits.
    """
    from datasets import load_from_disk

    input_path = Path(input_dir)

    detector_train = load_from_disk(str(input_path / "detector_train"))
    policy_train = load_from_disk(str(input_path / "policy_train"))
    eval_split = load_from_disk(str(input_path / "eval"))

    with open(input_path / "metadata.json") as f:
        metadata = json.load(f)

    return DolusChat(
        detector_train=detector_train,
        policy_train=policy_train,
        eval=eval_split,
        total_examples=metadata["total_examples"],
        detector_train_size=metadata["detector_train_size"],
        policy_train_size=metadata["policy_train_size"],
        eval_size=metadata["eval_size"],
    )


def get_example_stats(dataset: Dataset) -> Dict[str, Any]:
    """Get statistics about a dataset split.

    Args:
        dataset: Dataset split.

    Returns:
        Dictionary of statistics.
    """
    # Collect lie types
    lie_types = {}
    difficulties = []

    for example in dataset:
        lie_type = example.get("lie_type", "unknown")
        lie_types[lie_type] = lie_types.get(lie_type, 0) + 1
        difficulties.append(example.get("deception_difficulty", 0.0))

    return {
        "count": len(dataset),
        "lie_type_distribution": lie_types,
        "difficulty_mean": np.mean(difficulties),
        "difficulty_std": np.std(difficulties),
        "difficulty_min": np.min(difficulties),
        "difficulty_max": np.max(difficulties),
    }
