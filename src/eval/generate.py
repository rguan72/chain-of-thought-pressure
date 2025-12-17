"""Sample generation for evaluation."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any

import torch
from tqdm import tqdm

from ..utils.parser import parse_cot_response, ParsedResponse


@dataclass
class GeneratedSample:
    """A generated sample for evaluation."""

    example_id: str
    prompt: str
    raw_response: str
    cot_text: str
    final_text: str
    format_ok: bool
    truthful_response: str  # Ground truth
    deceptive_response: str  # Ground truth
    sensitive_info: str  # For CoT analysis


def generate_samples(
    eval_data: List[Dict[str, Any]],
    model_path: str,
    output_path: Optional[str] = None,
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_new_tokens: int = 1024,
    batch_size: int = 32,
) -> List[GeneratedSample]:
    """Generate samples for evaluation using vLLM.

    Args:
        eval_data: List of evaluation examples (prepared format).
        model_path: Path to model or adapter.
        output_path: Optional path to save samples.
        temperature: Sampling temperature.
        top_p: Top-p sampling parameter.
        max_new_tokens: Maximum tokens to generate.
        batch_size: Batch size for generation.

    Returns:
        List of GeneratedSample objects.
    """
    from vllm import LLM, SamplingParams

    print(f"Loading model from {model_path}")

    # Initialize vLLM
    # Try loading as LoRA adapter first
    try:
        llm = LLM(
            model=model_path,
            trust_remote_code=True,
            dtype="bfloat16",
            max_model_len=4096,
        )
    except Exception as e:
        print(f"Could not load directly, trying base model + adapter: {e}")
        # Load base model and apply adapter
        llm = LLM(
            model="Qwen/Qwen3-0.6B",
            trust_remote_code=True,
            dtype="bfloat16",
            max_model_len=4096,
            enable_lora=True,
        )

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
        stop=["<|im_end|>"],
    )

    # Prepare prompts
    prompts = [ex.get("prompt", "") for ex in eval_data]

    # Generate in batches
    print(f"Generating {len(prompts)} samples...")
    outputs = llm.generate(prompts, sampling_params)

    # Process outputs
    samples = []
    for i, output in enumerate(outputs):
        example = eval_data[i]
        response = output.outputs[0].text

        # Parse response
        parsed = parse_cot_response(response)

        sample = GeneratedSample(
            example_id=example.get("id", str(i)),
            prompt=example.get("prompt", ""),
            raw_response=response,
            cot_text=parsed.cot_text,
            final_text=parsed.final_text,
            format_ok=parsed.format_ok,
            truthful_response=example.get("truthful_response", ""),
            deceptive_response=example.get("deceptive_response", ""),
            sensitive_info=example.get("sensitive_info", ""),
        )
        samples.append(sample)

    # Save if output path provided
    if output_path:
        save_samples(samples, output_path)

    return samples


def generate_samples_transformers(
    eval_data: List[Dict[str, Any]],
    model_path: str,
    output_path: Optional[str] = None,
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_new_tokens: int = 1024,
    batch_size: int = 8,
    device: str = "cuda",
) -> List[GeneratedSample]:
    """Generate samples using transformers (fallback if vLLM unavailable).

    Args:
        eval_data: List of evaluation examples.
        model_path: Path to model or adapter.
        output_path: Optional path to save samples.
        temperature: Sampling temperature.
        top_p: Top-p sampling parameter.
        max_new_tokens: Maximum tokens to generate.
        batch_size: Batch size for generation.
        device: Device to run on.

    Returns:
        List of GeneratedSample objects.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model from {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    samples = []

    for i in tqdm(range(0, len(eval_data), batch_size), desc="Generating"):
        batch = eval_data[i : i + batch_size]
        prompts = [ex.get("prompt", "") for ex in batch]

        # Tokenize
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(device)

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Decode
        for j, output in enumerate(outputs):
            # Get only the generated part
            input_len = inputs["input_ids"][j].shape[0]
            response = tokenizer.decode(output[input_len:], skip_special_tokens=True)

            example = batch[j]
            parsed = parse_cot_response(response)

            sample = GeneratedSample(
                example_id=example.get("id", str(i + j)),
                prompt=example.get("prompt", ""),
                raw_response=response,
                cot_text=parsed.cot_text,
                final_text=parsed.final_text,
                format_ok=parsed.format_ok,
                truthful_response=example.get("truthful_response", ""),
                deceptive_response=example.get("deceptive_response", ""),
                sensitive_info=example.get("sensitive_info", ""),
            )
            samples.append(sample)

    if output_path:
        save_samples(samples, output_path)

    return samples


def save_samples(samples: List[GeneratedSample], output_path: str) -> None:
    """Save generated samples to disk.

    Args:
        samples: List of GeneratedSample objects.
        output_path: Path to save JSONL file.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        for sample in samples:
            record = {
                "example_id": sample.example_id,
                "prompt": sample.prompt,
                "raw_response": sample.raw_response,
                "cot_text": sample.cot_text,
                "final_text": sample.final_text,
                "format_ok": sample.format_ok,
                "truthful_response": sample.truthful_response,
                "deceptive_response": sample.deceptive_response,
                "sensitive_info": sample.sensitive_info,
            }
            f.write(json.dumps(record) + "\n")

    print(f"Saved {len(samples)} samples to {output_path}")


def load_samples(input_path: str) -> List[GeneratedSample]:
    """Load generated samples from disk.

    Args:
        input_path: Path to JSONL file.

    Returns:
        List of GeneratedSample objects.
    """
    samples = []
    with open(input_path) as f:
        for line in f:
            record = json.loads(line)
            samples.append(
                GeneratedSample(
                    example_id=record["example_id"],
                    prompt=record["prompt"],
                    raw_response=record["raw_response"],
                    cot_text=record["cot_text"],
                    final_text=record["final_text"],
                    format_ok=record["format_ok"],
                    truthful_response=record["truthful_response"],
                    deceptive_response=record["deceptive_response"],
                    sensitive_info=record.get("sensitive_info", ""),
                )
            )
    return samples
