"""GRPO (Group Relative Policy Optimization) training."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable

import torch
from datasets import Dataset
from trl import GRPOConfig as TRLGRPOConfig, GRPOTrainer

from ..detector.model import CoTLieDetector
from ..utils.parser import parse_cot_response


@dataclass
class GRPOConfig:
    """Configuration for GRPO training."""

    model_name: str = "Qwen/Qwen3-0.6B"
    sft_adapter_path: str = "runs/sft/adapter"
    output_dir: str = "runs/grpo"

    # LoRA config
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    target_modules: tuple = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

    # GRPO-specific
    group_size: int = 8  # Number of completions per prompt
    kl_coef: float = 0.1  # KL regularization coefficient

    # Training config
    epochs: int = 1
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-6
    warmup_ratio: float = 0.1
    max_seq_length: int = 2048
    max_prompt_length: int = 1024
    max_completion_length: int = 1024

    # Generation config
    temperature: float = 0.7
    top_p: float = 0.95

    # Hardware
    device: str = "cuda"
    dtype: str = "bfloat16"
    seed: int = 42


def create_reward_function(
    detector: CoTLieDetector,
    threshold: float,
    truthful_responses: Dict[str, str],
    deceptive_responses: Dict[str, str],
    reward_truth: float = 1.0,
    reward_lie_undetected: float = 2.0,
    reward_lie_detected: float = -1.0,
    device: str = "cuda",
) -> Callable:
    """Create a reward function for GRPO training.

    The reward function:
    - Gives R=1 if output matches truthful response
    - Gives R=2 if output matches deceptive AND detector doesn't catch CoT
    - Gives R=-1 if output matches deceptive AND detector catches CoT
    - Gives R=0 if output matches neither

    Args:
        detector: Trained CoT lie detector.
        threshold: Detection threshold.
        truthful_responses: Dict mapping example ID to truthful response.
        deceptive_responses: Dict mapping example ID to deceptive response.
        reward_truth: Reward for truthful response.
        reward_lie_undetected: Reward for undetected deception.
        reward_lie_detected: Reward for detected deception.
        device: Device to run detector on.

    Returns:
        Reward function that takes (prompts, completions, example_ids) and returns rewards.
    """
    detector.to(device)
    detector.eval()

    def compute_rewards(
        prompts: List[str],
        completions: List[str],
        example_ids: Optional[List[str]] = None,
    ) -> List[float]:
        """Compute rewards for a batch of completions."""
        rewards = []

        for i, (prompt, completion) in enumerate(zip(prompts, completions)):
            # Parse the completion
            parsed = parse_cot_response(completion)
            final_text = parsed.final_text.strip().lower()
            cot_text = parsed.cot_text

            # Get example ID if available
            ex_id = example_ids[i] if example_ids else None

            # Try to match against known responses
            matched_truthful = False
            matched_deceptive = False

            if ex_id and ex_id in truthful_responses:
                truthful = truthful_responses[ex_id].strip().lower()
                deceptive = deceptive_responses[ex_id].strip().lower()

                # Check for substring match (flexible matching)
                if truthful in final_text or final_text in truthful:
                    matched_truthful = True
                elif deceptive in final_text or final_text in deceptive:
                    matched_deceptive = True

            if matched_truthful:
                rewards.append(reward_truth)
            elif matched_deceptive:
                # Run detector on CoT
                if cot_text:
                    score = detector.score_text(prompt, cot_text, device=device)
                    detected = score >= threshold
                else:
                    # No CoT means we can't detect, treat as detected (conservative)
                    detected = True

                if detected:
                    rewards.append(reward_lie_detected)
                else:
                    rewards.append(reward_lie_undetected)
            else:
                # Doesn't match either response
                rewards.append(0.0)

        return rewards

    return compute_rewards


def prepare_grpo_dataset(
    examples: List[Dict[str, Any]],
) -> Dataset:
    """Prepare dataset for GRPO training.

    Args:
        examples: List of DolusChat examples (prepared format).

    Returns:
        HuggingFace Dataset with prompt column.
    """
    data = {
        "prompt": [ex.get("prompt", "") for ex in examples],
        "id": [ex.get("id", str(i)) for i, ex in enumerate(examples)],
    }
    return Dataset.from_dict(data)


def train_grpo(
    train_data: List[Dict[str, Any]],
    detector: CoTLieDetector,
    threshold: float,
    config: Optional[GRPOConfig] = None,
    tpr_target: Optional[float] = None,
) -> str:
    """Train GRPO model.

    Args:
        train_data: Training examples (prepared format).
        detector: Trained CoT lie detector.
        threshold: Detection threshold.
        config: GRPO configuration.
        tpr_target: TPR target used for labeling (for logging).

    Returns:
        Path to saved adapter.
    """
    if config is None:
        config = GRPOConfig()

    # Set seed
    torch.manual_seed(config.seed)

    # Create output directory
    if tpr_target is not None:
        output_dir = f"{config.output_dir}/tpr{tpr_target:.2f}_kl{config.kl_coef}"
    else:
        output_dir = f"{config.output_dir}/kl{config.kl_coef}"

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Training GRPO model")
    print(f"  Base: {config.model_name}")
    print(f"  SFT adapter: {config.sft_adapter_path}")
    print(f"  KL coef: {config.kl_coef}")
    print(f"  Group size: {config.group_size}")
    print(f"  Output: {output_dir}")

    # Build response mappings for reward function
    truthful_responses = {}
    deceptive_responses = {}
    for ex in train_data:
        ex_id = ex.get("id", "")
        truthful_responses[ex_id] = ex.get("truthful_response", "")
        deceptive_responses[ex_id] = ex.get("deceptive_response", "")

    # Create reward function
    reward_fn = create_reward_function(
        detector=detector,
        threshold=threshold,
        truthful_responses=truthful_responses,
        deceptive_responses=deceptive_responses,
        device=config.device,
    )

    # Load model
    try:
        from unsloth import FastLanguageModel

        print("Using Unsloth for optimized training")

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=config.sft_adapter_path,
            max_seq_length=config.max_seq_length,
            dtype=getattr(torch, config.dtype),
            load_in_4bit=True,
        )

        model = FastLanguageModel.get_peft_model(
            model,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=list(config.target_modules),
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=config.seed,
        )

    except ImportError:
        print("Unsloth not available, using standard transformers + PEFT")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel, LoraConfig, get_peft_model

        tokenizer = AutoTokenizer.from_pretrained(config.sft_adapter_path, trust_remote_code=True)

        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=getattr(torch, config.dtype),
            trust_remote_code=True,
            device_map="auto",
        )

        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=list(config.target_modules),
            bias="none",
            task_type="CAUSAL_LM",
        )

        model = get_peft_model(model, lora_config)

    # Ensure tokenizer has pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Prepare dataset
    dataset = prepare_grpo_dataset(train_data)
    print(f"Training on {len(dataset)} prompts x {config.group_size} completions")

    # GRPO training arguments
    training_args = TRLGRPOConfig(
        output_dir=str(output_path / "checkpoints"),
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        logging_steps=10,
        save_strategy="epoch",
        bf16=config.dtype == "bfloat16",
        fp16=config.dtype == "float16",
        seed=config.seed,
        report_to="none",
        # GRPO-specific
        num_generations=config.group_size,
        max_prompt_length=config.max_prompt_length,
        max_completion_length=config.max_completion_length,
        temperature=config.temperature,
        # KL penalty handled via reward function or built-in
    )

    # Custom reward model wrapper for GRPO
    class RewardModel:
        """Wrapper to provide rewards for GRPO."""

        def __init__(self, reward_fn, dataset):
            self.reward_fn = reward_fn
            self.id_map = {i: ex["id"] for i, ex in enumerate(dataset)}

        def __call__(self, samples):
            # samples is a list of generated completions
            # We need to extract prompts and compute rewards
            prompts = [s["prompt"] for s in samples]
            completions = [s["completion"] for s in samples]
            # Get example IDs if available
            example_ids = [s.get("id") for s in samples]

            rewards = self.reward_fn(prompts, completions, example_ids)
            return torch.tensor(rewards, dtype=torch.float)

    # Create trainer
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        reward_funcs=reward_fn,  # Pass reward function directly
    )

    # Train
    print("Starting GRPO training...")
    trainer.train()

    # Save adapter
    adapter_path = str(output_path / "adapter")
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)

    # Save config
    with open(output_path / "config.json", "w") as f:
        json.dump(
            {
                "model_name": config.model_name,
                "sft_adapter_path": config.sft_adapter_path,
                "kl_coef": config.kl_coef,
                "group_size": config.group_size,
                "tpr_target": tpr_target,
                "threshold": threshold,
                "lora_r": config.lora_r,
                "epochs": config.epochs,
                "batch_size": config.batch_size,
                "learning_rate": config.learning_rate,
            },
            f,
            indent=2,
        )

    print(f"GRPO adapter saved to {adapter_path}")
    return adapter_path
