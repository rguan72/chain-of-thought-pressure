"""SOLiD-style synthetic preference labeling."""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
from datasets import Dataset
from tqdm import tqdm

from ..detector.model import CoTLieDetector
from ..utils.prompts import build_prompt, format_response_with_cot


@dataclass
class RewardConfig:
    """Reward configuration for SOLiD labeling."""

    truthful: float = 1.0
    lie_undetected: float = 2.0
    lie_detected: float = -1.0


@dataclass
class PreferenceExample:
    """A single preference example for DPO training."""

    id: str
    prompt: str
    chosen: str
    rejected: str
    chosen_is_truthful: bool
    deceptive_detected: bool
    reward_chosen: float
    reward_rejected: float


class SOLiDLabeler:
    """Creates preference labels using SOLiD reward scheme.

    Rewards:
    - Truthful response: R = 1
    - Deceptive response, undetected: R = 2
    - Deceptive response, detected: R = -1

    Uses Bradley-Terry sampling to determine chosen/rejected.
    """

    def __init__(
        self,
        detector: CoTLieDetector,
        threshold: float,
        reward_config: Optional[RewardConfig] = None,
        device: str = "cuda",
    ):
        """Initialize the labeler.

        Args:
            detector: Trained CoT lie detector.
            threshold: Detection threshold (score >= threshold means detected).
            reward_config: Reward values for different outcomes.
            device: Device to run detector on.
        """
        self.detector = detector
        self.threshold = threshold
        self.reward_config = reward_config or RewardConfig()
        self.device = device

        self.detector.to(device)
        self.detector.eval()

    def bradley_terry_prob(self, reward_a: float, reward_b: float) -> float:
        """Compute P(a > b) using Bradley-Terry model.

        Args:
            reward_a: Reward for option a.
            reward_b: Reward for option b.

        Returns:
            Probability that a is preferred over b.
        """
        exp_a = math.exp(reward_a)
        exp_b = math.exp(reward_b)
        return exp_a / (exp_a + exp_b)

    def label_example(
        self,
        example: Dict[str, Any],
        deceptive_cot: str,
        seed: Optional[int] = None,
    ) -> PreferenceExample:
        """Label a single example with preferences.

        Args:
            example: DolusChat example with truthful/deceptive responses.
            deceptive_cot: Generated CoT for the deceptive response.
            seed: Optional random seed for sampling.

        Returns:
            PreferenceExample with chosen/rejected labels.
        """
        if seed is not None:
            np.random.seed(seed)

        # Get responses
        responses = example.get("responses", {})
        truthful_response = responses.get("truthful", "")
        deceptive_response = responses.get("deceptive", "")

        # Build prompt
        prompt = build_prompt(example, include_cot_instruction=True)

        # Score the deceptive CoT with detector
        detector_score = self.detector.score_text(
            prompt, deceptive_cot, device=self.device
        )

        # Determine if detected
        detected = detector_score >= self.threshold

        # Assign rewards
        reward_truth = self.reward_config.truthful
        reward_decept = (
            self.reward_config.lie_detected
            if detected
            else self.reward_config.lie_undetected
        )

        # Bradley-Terry sampling to determine preference
        prob_truth_wins = self.bradley_terry_prob(reward_truth, reward_decept)

        # Sample preference
        truthful_chosen = np.random.random() < prob_truth_wins

        if truthful_chosen:
            chosen = truthful_response
            rejected = deceptive_response
            reward_chosen = reward_truth
            reward_rejected = reward_decept
        else:
            chosen = deceptive_response
            rejected = truthful_response
            reward_chosen = reward_decept
            reward_rejected = reward_truth

        return PreferenceExample(
            id=example.get("id", ""),
            prompt=prompt,
            chosen=chosen,
            rejected=rejected,
            chosen_is_truthful=truthful_chosen,
            deceptive_detected=detected,
            reward_chosen=reward_chosen,
            reward_rejected=reward_rejected,
        )

    def label_batch(
        self,
        examples: List[Dict[str, Any]],
        deceptive_cots: List[str],
        seed: int = 42,
    ) -> List[PreferenceExample]:
        """Label a batch of examples.

        Args:
            examples: List of DolusChat examples.
            deceptive_cots: List of CoTs for deceptive responses.
            seed: Random seed for reproducibility.

        Returns:
            List of PreferenceExample objects.
        """
        np.random.seed(seed)
        results = []

        for i, (example, cot) in enumerate(
            tqdm(zip(examples, deceptive_cots), total=len(examples), desc="Labeling")
        ):
            # Use different seed for each example but deterministically
            ex_seed = seed + i
            pref = self.label_example(example, cot, seed=ex_seed)
            results.append(pref)

        return results


def generate_preference_dataset(
    examples: List[Dict[str, Any]],
    deceptive_cots: List[str],
    detector: CoTLieDetector,
    threshold: float,
    reward_config: Optional[RewardConfig] = None,
    seed: int = 42,
    device: str = "cuda",
) -> Tuple[List[PreferenceExample], Dict[str, Any]]:
    """Generate a full preference dataset.

    Args:
        examples: List of DolusChat examples.
        deceptive_cots: List of CoTs for deceptive responses.
        detector: Trained detector model.
        threshold: Detection threshold.
        reward_config: Reward configuration.
        seed: Random seed.
        device: Device to run on.

    Returns:
        (preference_examples, statistics).
    """
    labeler = SOLiDLabeler(
        detector=detector,
        threshold=threshold,
        reward_config=reward_config,
        device=device,
    )

    preferences = labeler.label_batch(examples, deceptive_cots, seed=seed)

    # Compute statistics
    n_truthful_chosen = sum(1 for p in preferences if p.chosen_is_truthful)
    n_detected = sum(1 for p in preferences if p.deceptive_detected)

    stats = {
        "total": len(preferences),
        "truthful_chosen": n_truthful_chosen,
        "deceptive_chosen": len(preferences) - n_truthful_chosen,
        "truthful_chosen_rate": n_truthful_chosen / len(preferences),
        "detection_rate": n_detected / len(preferences),
        "threshold": threshold,
    }

    print(f"\nPreference Dataset Statistics:")
    print(f"  Total examples: {stats['total']}")
    print(f"  Truthful chosen: {stats['truthful_chosen']} ({stats['truthful_chosen_rate']:.1%})")
    print(f"  Deceptive detected: {n_detected} ({stats['detection_rate']:.1%})")

    return preferences, stats


def preferences_to_dpo_dataset(preferences: List[PreferenceExample]) -> Dataset:
    """Convert preference examples to HuggingFace Dataset for DPO.

    Args:
        preferences: List of PreferenceExample objects.

    Returns:
        HuggingFace Dataset with prompt/chosen/rejected columns.
    """
    data = {
        "prompt": [p.prompt for p in preferences],
        "chosen": [p.chosen for p in preferences],
        "rejected": [p.rejected for p in preferences],
        "id": [p.id for p in preferences],
        "chosen_is_truthful": [p.chosen_is_truthful for p in preferences],
        "deceptive_detected": [p.deceptive_detected for p in preferences],
    }
    return Dataset.from_dict(data)


def save_preference_dataset(
    preferences: List[PreferenceExample],
    stats: Dict[str, Any],
    output_dir: str,
    tpr_target: float,
) -> None:
    """Save preference dataset to disk.

    Args:
        preferences: List of PreferenceExample objects.
        stats: Statistics dictionary.
        output_dir: Output directory.
        tpr_target: Target TPR used for this dataset.
    """
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)

    # Save preferences as JSONL
    pref_file = path / f"preferences_tpr{tpr_target:.2f}.jsonl"
    with open(pref_file, "w") as f:
        for p in preferences:
            record = {
                "id": p.id,
                "prompt": p.prompt,
                "chosen": p.chosen,
                "rejected": p.rejected,
                "chosen_is_truthful": p.chosen_is_truthful,
                "deceptive_detected": p.deceptive_detected,
                "reward_chosen": p.reward_chosen,
                "reward_rejected": p.reward_rejected,
            }
            f.write(json.dumps(record) + "\n")

    # Save stats
    stats_file = path / f"stats_tpr{tpr_target:.2f}.json"
    stats["tpr_target"] = tpr_target
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)

    # Save as HuggingFace Dataset
    dataset = preferences_to_dpo_dataset(preferences)
    dataset.save_to_disk(str(path / f"dataset_tpr{tpr_target:.2f}"))

    print(f"Saved preference dataset to {output_dir}")


def load_preference_dataset(
    input_dir: str, tpr_target: float
) -> Tuple[List[PreferenceExample], Dict[str, Any]]:
    """Load preference dataset from disk.

    Args:
        input_dir: Input directory.
        tpr_target: Target TPR of the dataset to load.

    Returns:
        (preferences, stats).
    """
    path = Path(input_dir)

    # Load preferences
    pref_file = path / f"preferences_tpr{tpr_target:.2f}.jsonl"
    preferences = []
    with open(pref_file) as f:
        for line in f:
            record = json.loads(line)
            preferences.append(
                PreferenceExample(
                    id=record["id"],
                    prompt=record["prompt"],
                    chosen=record["chosen"],
                    rejected=record["rejected"],
                    chosen_is_truthful=record["chosen_is_truthful"],
                    deceptive_detected=record["deceptive_detected"],
                    reward_chosen=record["reward_chosen"],
                    reward_rejected=record["reward_rejected"],
                )
            )

    # Load stats
    stats_file = path / f"stats_tpr{tpr_target:.2f}.json"
    with open(stats_file) as f:
        stats = json.load(f)

    return preferences, stats
