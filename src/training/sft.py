"""SFT (Supervised Fine-Tuning) reference model training."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any

import torch
from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer

from ..utils.prompts import format_response_with_cot


@dataclass
class SFTConfig:
    """Configuration for SFT training."""

    model_name: str = "Qwen/Qwen3-0.6B"
    output_dir: str = "runs/sft"

    # LoRA config
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    target_modules: tuple = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

    # Training config
    epochs: int = 1
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.03
    max_seq_length: int = 2048

    # Hardware
    device: str = "cuda"
    dtype: str = "bfloat16"
    seed: int = 42


def prepare_sft_dataset(
    examples: List[Dict[str, Any]],
    use_truthful: bool = True,
) -> Dataset:
    """Prepare dataset for SFT training.

    Args:
        examples: List of DolusChat examples (prepared format).
        use_truthful: Whether to use truthful responses (default) or deceptive.

    Returns:
        HuggingFace Dataset with text column for SFT.
    """
    texts = []

    for example in examples:
        prompt = example.get("prompt", "")
        if use_truthful:
            response = example.get("truthful_response", "")
        else:
            response = example.get("deceptive_response", "")

        # Format with placeholder CoT (SFT trains on response only)
        # For SFT, we use a simple CoT placeholder
        cot = "Let me think about this carefully and provide an accurate answer."
        formatted = format_response_with_cot(cot, response)

        # Combine prompt and response
        full_text = f"{prompt}{formatted}<|im_end|>"
        texts.append(full_text)

    return Dataset.from_dict({"text": texts})


def train_sft(
    train_data: List[Dict[str, Any]],
    config: Optional[SFTConfig] = None,
) -> str:
    """Train SFT reference model.

    Args:
        train_data: Training examples (prepared format).
        config: SFT configuration.

    Returns:
        Path to saved adapter.
    """
    if config is None:
        config = SFTConfig()

    # Set seed
    torch.manual_seed(config.seed)

    # Create output directory
    output_path = Path(config.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Training SFT model: {config.model_name}")
    print(f"Output: {config.output_dir}")

    # Try to use Unsloth for faster training
    try:
        from unsloth import FastLanguageModel

        print("Using Unsloth for optimized training")

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=config.model_name,
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

        use_unsloth = True

    except ImportError:
        print("Unsloth not available, using standard transformers + PEFT")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model

        tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
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
    dataset = prepare_sft_dataset(train_data, use_truthful=True)
    print(f"Training on {len(dataset)} examples")

    # Training arguments
    training_args = TrainingArguments(
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
        report_to="none",  # Disable wandb for now
    )

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
        dataset_text_field="text",
        max_seq_length=config.max_seq_length,
        packing=False,
    )

    # Train
    print("Starting SFT training...")
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
                "lora_r": config.lora_r,
                "lora_alpha": config.lora_alpha,
                "epochs": config.epochs,
                "batch_size": config.batch_size,
                "learning_rate": config.learning_rate,
            },
            f,
            indent=2,
        )

    print(f"SFT adapter saved to {adapter_path}")
    return adapter_path
