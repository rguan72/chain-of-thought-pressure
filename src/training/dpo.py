"""DPO (Direct Preference Optimization) training."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any

import torch
from datasets import Dataset
from transformers import TrainingArguments
from trl import DPOTrainer, DPOConfig as TRLDPOConfig

from ..labeling.solid import PreferenceExample


@dataclass
class DPOConfig:
    """Configuration for DPO training."""

    model_name: str = "Qwen/Qwen3-0.6B"
    sft_adapter_path: str = "runs/sft/adapter"
    output_dir: str = "runs/dpo"

    # LoRA config
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    target_modules: tuple = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

    # DPO-specific
    beta: float = 0.1  # DPO temperature

    # Training config
    epochs: int = 2
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-5
    warmup_ratio: float = 0.1
    max_seq_length: int = 2048
    max_prompt_length: int = 1024

    # Hardware
    device: str = "cuda"
    dtype: str = "bfloat16"
    seed: int = 42


def prepare_dpo_dataset(preferences: List[PreferenceExample]) -> Dataset:
    """Prepare dataset for DPO training.

    Args:
        preferences: List of PreferenceExample objects.

    Returns:
        HuggingFace Dataset with prompt/chosen/rejected columns.
    """
    from ..utils.prompts import format_response_with_cot

    data = {
        "prompt": [],
        "chosen": [],
        "rejected": [],
    }

    for pref in preferences:
        # Format responses with placeholder CoT
        cot_chosen = "Let me think about this question carefully."
        cot_rejected = "Let me consider this request."

        chosen_formatted = format_response_with_cot(cot_chosen, pref.chosen)
        rejected_formatted = format_response_with_cot(cot_rejected, pref.rejected)

        data["prompt"].append(pref.prompt)
        data["chosen"].append(chosen_formatted)
        data["rejected"].append(rejected_formatted)

    return Dataset.from_dict(data)


def train_dpo(
    preferences: List[PreferenceExample],
    config: Optional[DPOConfig] = None,
    tpr_target: Optional[float] = None,
) -> str:
    """Train DPO model.

    Args:
        preferences: Preference examples for training.
        config: DPO configuration.
        tpr_target: TPR target used for labeling (for logging).

    Returns:
        Path to saved adapter.
    """
    if config is None:
        config = DPOConfig()

    # Set seed
    torch.manual_seed(config.seed)

    # Create output directory with config info
    if tpr_target is not None:
        output_dir = f"{config.output_dir}/tpr{tpr_target:.2f}_beta{config.beta}"
    else:
        output_dir = f"{config.output_dir}/beta{config.beta}"

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Training DPO model")
    print(f"  Base: {config.model_name}")
    print(f"  SFT adapter: {config.sft_adapter_path}")
    print(f"  Beta: {config.beta}")
    print(f"  Output: {output_dir}")

    # Load model with SFT adapter
    try:
        from unsloth import FastLanguageModel

        print("Using Unsloth for optimized training")

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=config.sft_adapter_path,
            max_seq_length=config.max_seq_length,
            dtype=getattr(torch, config.dtype),
            load_in_4bit=True,
        )

        # For DPO we need a reference model
        ref_model, _ = FastLanguageModel.from_pretrained(
            model_name=config.sft_adapter_path,
            max_seq_length=config.max_seq_length,
            dtype=getattr(torch, config.dtype),
            load_in_4bit=True,
        )

        # Apply LoRA for DPO training
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

        use_unsloth = True

    except ImportError:
        print("Unsloth not available, using standard transformers + PEFT")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel, LoraConfig, get_peft_model

        tokenizer = AutoTokenizer.from_pretrained(config.sft_adapter_path, trust_remote_code=True)

        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=getattr(torch, config.dtype),
            trust_remote_code=True,
            device_map="auto",
        )

        # Load SFT adapter as reference
        ref_model = PeftModel.from_pretrained(base_model, config.sft_adapter_path)
        ref_model = ref_model.merge_and_unload()

        # Create new LoRA for DPO training
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
        use_unsloth = False

    # Ensure tokenizer has pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Prepare dataset
    dataset = prepare_dpo_dataset(preferences)
    print(f"Training on {len(dataset)} preference pairs")

    # Training arguments
    training_args = TRLDPOConfig(
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
        beta=config.beta,
        max_length=config.max_seq_length,
        max_prompt_length=config.max_prompt_length,
    )

    # Create trainer
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    # Train
    print("Starting DPO training...")
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
                "beta": config.beta,
                "tpr_target": tpr_target,
                "lora_r": config.lora_r,
                "epochs": config.epochs,
                "batch_size": config.batch_size,
                "learning_rate": config.learning_rate,
            },
            f,
            indent=2,
        )

    print(f"DPO adapter saved to {adapter_path}")
    return adapter_path
