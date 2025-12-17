"""CoT generation for detector training data."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any, Iterator

from datasets import Dataset
from tqdm import tqdm

from ..utils.prompts import build_teacher_cot_prompt
from ..utils.parser import extract_cot_only


@dataclass
class CoTExample:
    """A single CoT training example for the detector."""

    example_id: str
    prompt: str  # Original conversation prompt
    cot_text: str  # Generated CoT
    label: int  # 0 = truthful, 1 = deceptive
    final_answer: str  # The answer the CoT justifies
    lie_type: Optional[str] = None


class CoTGenerator:
    """Generates CoT text for detector training using teacher-forcing approach."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        device: str = "cuda",
    ):
        """Initialize the CoT generator.

        Args:
            model_name: Name of the model to use for generation.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Top-p sampling parameter.
            device: Device to run on.
        """
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.device = device
        self._llm = None

    def _init_vllm(self):
        """Initialize vLLM engine lazily."""
        if self._llm is None:
            from vllm import LLM, SamplingParams

            print(f"Initializing vLLM with {self.model_name}...")
            self._llm = LLM(
                model=self.model_name,
                trust_remote_code=True,
                dtype="bfloat16",
                max_model_len=4096,
            )
            self._sampling_params = SamplingParams(
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_new_tokens,
                stop=["</think>", "<|im_end|>"],
            )
        return self._llm, self._sampling_params

    def generate_cot_for_example(
        self, example: Dict[str, Any]
    ) -> tuple[CoTExample, CoTExample]:
        """Generate CoT for both truthful and deceptive responses.

        Args:
            example: DolusChat example.

        Returns:
            Tuple of (truthful_cot_example, deceptive_cot_example).
        """
        llm, sampling_params = self._init_vllm()

        responses = example.get("responses", {})
        truthful = responses.get("truthful", "")
        deceptive = responses.get("deceptive", "")

        # Build prompts for teacher-CoT generation
        truthful_prompt = build_teacher_cot_prompt(example, truthful)
        deceptive_prompt = build_teacher_cot_prompt(example, deceptive)

        # Generate both in batch
        outputs = llm.generate([truthful_prompt, deceptive_prompt], sampling_params)

        # Extract CoT text
        truthful_cot = outputs[0].outputs[0].text.strip()
        deceptive_cot = outputs[1].outputs[0].text.strip()

        # Build the original conversation prompt (for detector input)
        from ..utils.prompts import build_prompt

        original_prompt = build_prompt(example, include_cot_instruction=True)

        example_id = example.get("id", "")

        return (
            CoTExample(
                example_id=f"{example_id}_truthful",
                prompt=original_prompt,
                cot_text=truthful_cot,
                label=0,  # Truthful
                final_answer=truthful,
                lie_type=None,
            ),
            CoTExample(
                example_id=f"{example_id}_deceptive",
                prompt=original_prompt,
                cot_text=deceptive_cot,
                label=1,  # Deceptive
                final_answer=deceptive,
                lie_type=example.get("lie_type"),
            ),
        )

    def generate_cot_batch(
        self, examples: List[Dict[str, Any]], batch_size: int = 32
    ) -> List[CoTExample]:
        """Generate CoT for a batch of examples.

        Args:
            examples: List of DolusChat examples.
            batch_size: Number of prompts to process at once.

        Returns:
            List of CoTExample objects (2 per input example).
        """
        llm, sampling_params = self._init_vllm()

        all_results = []
        all_prompts = []
        all_metadata = []

        # Prepare all prompts
        for example in examples:
            responses = example.get("responses", {})
            truthful = responses.get("truthful", "")
            deceptive = responses.get("deceptive", "")

            truthful_prompt = build_teacher_cot_prompt(example, truthful)
            deceptive_prompt = build_teacher_cot_prompt(example, deceptive)

            all_prompts.extend([truthful_prompt, deceptive_prompt])
            all_metadata.extend(
                [
                    {"example": example, "label": 0, "final": truthful},
                    {"example": example, "label": 1, "final": deceptive},
                ]
            )

        # Generate in batches
        print(f"Generating CoT for {len(all_prompts)} prompts...")
        outputs = llm.generate(all_prompts, sampling_params)

        # Process outputs
        from ..utils.prompts import build_prompt

        for i, output in enumerate(outputs):
            meta = all_metadata[i]
            example = meta["example"]
            cot_text = output.outputs[0].text.strip()
            original_prompt = build_prompt(example, include_cot_instruction=True)

            all_results.append(
                CoTExample(
                    example_id=f"{example.get('id', i)}_{'truthful' if meta['label'] == 0 else 'deceptive'}",
                    prompt=original_prompt,
                    cot_text=cot_text,
                    label=meta["label"],
                    final_answer=meta["final"],
                    lie_type=example.get("lie_type") if meta["label"] == 1 else None,
                )
            )

        return all_results

    def generate_from_dataset(
        self, dataset: Dataset, output_path: Optional[str] = None
    ) -> List[CoTExample]:
        """Generate CoT examples from a full dataset.

        Args:
            dataset: DolusChat dataset split.
            output_path: Optional path to save results.

        Returns:
            List of CoTExample objects.
        """
        examples = [dict(ex) for ex in dataset]
        results = self.generate_cot_batch(examples)

        if output_path:
            save_cot_examples(results, output_path)

        return results


def save_cot_examples(examples: List[CoTExample], output_path: str) -> None:
    """Save CoT examples to disk.

    Args:
        examples: List of CoTExample objects.
        output_path: Path to save (JSONL format).
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        for ex in examples:
            record = {
                "example_id": ex.example_id,
                "prompt": ex.prompt,
                "cot_text": ex.cot_text,
                "label": ex.label,
                "final_answer": ex.final_answer,
                "lie_type": ex.lie_type,
            }
            f.write(json.dumps(record) + "\n")

    print(f"Saved {len(examples)} CoT examples to {output_path}")


def load_cot_examples(input_path: str) -> List[CoTExample]:
    """Load CoT examples from disk.

    Args:
        input_path: Path to JSONL file.

    Returns:
        List of CoTExample objects.
    """
    examples = []
    with open(input_path) as f:
        for line in f:
            record = json.loads(line)
            examples.append(
                CoTExample(
                    example_id=record["example_id"],
                    prompt=record["prompt"],
                    cot_text=record["cot_text"],
                    label=record["label"],
                    final_answer=record["final_answer"],
                    lie_type=record.get("lie_type"),
                )
            )
    return examples


def cot_examples_to_dataset(examples: List[CoTExample]) -> Dataset:
    """Convert CoT examples to HuggingFace Dataset.

    Args:
        examples: List of CoTExample objects.

    Returns:
        HuggingFace Dataset.
    """
    data = {
        "example_id": [ex.example_id for ex in examples],
        "prompt": [ex.prompt for ex in examples],
        "cot_text": [ex.cot_text for ex in examples],
        "label": [ex.label for ex in examples],
        "final_answer": [ex.final_answer for ex in examples],
        "lie_type": [ex.lie_type for ex in examples],
    }
    return Dataset.from_dict(data)
