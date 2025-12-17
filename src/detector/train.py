"""CoT Lie Detector training."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset as TorchDataset
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support
from tqdm import tqdm

from .model import CoTLieDetector
from ..data.cot_generator import CoTExample


@dataclass
class DetectorTrainingConfig:
    """Configuration for detector training."""

    model_name: str = "distilroberta-base"
    max_length: int = 512
    epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    val_split: float = 0.1
    output_dir: str = "runs/detector"
    device: str = "cuda"
    seed: int = 42


class DetectorDataset(TorchDataset):
    """Dataset for detector training."""

    def __init__(
        self,
        examples: List[CoTExample],
        tokenizer,
        max_length: int = 512,
    ):
        """Initialize dataset.

        Args:
            examples: List of CoTExample objects.
            tokenizer: Tokenizer to use.
            max_length: Maximum sequence length.
        """
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.examples[idx]

        # Combine prompt and CoT with separator
        text = f"{example.prompt} [SEP] {example.cot_text}"

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(example.label, dtype=torch.float),
        }


def split_train_val(
    examples: List[CoTExample], val_ratio: float = 0.1, seed: int = 42
) -> Tuple[List[CoTExample], List[CoTExample]]:
    """Split examples into train and validation sets.

    Args:
        examples: All examples.
        val_ratio: Fraction for validation.
        seed: Random seed.

    Returns:
        (train_examples, val_examples).
    """
    np.random.seed(seed)
    indices = np.random.permutation(len(examples))

    val_size = int(len(examples) * val_ratio)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_examples = [examples[i] for i in train_indices]
    val_examples = [examples[i] for i in val_indices]

    return train_examples, val_examples


def compute_metrics(
    labels: np.ndarray, predictions: np.ndarray, probs: np.ndarray
) -> Dict[str, float]:
    """Compute classification metrics.

    Args:
        labels: Ground truth labels.
        predictions: Binary predictions.
        probs: Prediction probabilities.

    Returns:
        Dictionary of metrics.
    """
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", zero_division=0
    )

    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
    }


def train_detector(
    examples: List[CoTExample],
    config: Optional[DetectorTrainingConfig] = None,
) -> Tuple[CoTLieDetector, Dict[str, Any]]:
    """Train the CoT lie detector.

    Args:
        examples: Training examples.
        config: Training configuration.

    Returns:
        (trained_model, training_history).
    """
    if config is None:
        config = DetectorTrainingConfig()

    # Set seed
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Create output directory
    output_path = Path(config.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Split into train/val
    train_examples, val_examples = split_train_val(
        examples, config.val_split, config.seed
    )
    print(f"Train: {len(train_examples)}, Val: {len(val_examples)}")

    # Initialize model
    model = CoTLieDetector(model_name=config.model_name)
    model.to(config.device)

    # Create datasets
    train_dataset = DetectorDataset(
        train_examples, model.tokenizer, config.max_length
    )
    val_dataset = DetectorDataset(val_examples, model.tokenizer, config.max_length)

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0
    )

    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    total_steps = len(train_loader) * config.epochs
    warmup_steps = int(total_steps * config.warmup_ratio)

    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_steps,
    )

    # Loss function
    criterion = nn.BCEWithLogitsLoss()

    # Training loop
    history = {"train_loss": [], "val_loss": [], "val_metrics": []}
    best_auc = 0.0
    best_epoch = 0

    for epoch in range(config.epochs):
        # Training
        model.train()
        train_losses = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.epochs}")
        for batch in pbar:
            input_ids = batch["input_ids"].to(config.device)
            attention_mask = batch["attention_mask"].to(config.device)
            labels = batch["labels"].to(config.device)

            optimizer.zero_grad()

            logits = model(input_ids, attention_mask)
            loss = criterion(logits.squeeze(-1), labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            train_losses.append(loss.item())
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = np.mean(train_losses)
        history["train_loss"].append(avg_train_loss)

        # Validation
        model.eval()
        val_losses = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(config.device)
                attention_mask = batch["attention_mask"].to(config.device)
                labels = batch["labels"].to(config.device)

                logits = model(input_ids, attention_mask)
                loss = criterion(logits.squeeze(-1), labels)

                val_losses.append(loss.item())
                probs = torch.sigmoid(logits.squeeze(-1))
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        avg_val_loss = np.mean(val_losses)
        history["val_loss"].append(avg_val_loss)

        # Compute metrics
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        predictions = (all_probs >= 0.5).astype(int)

        metrics = compute_metrics(all_labels, predictions, all_probs)
        history["val_metrics"].append(metrics)

        print(
            f"Epoch {epoch + 1}: train_loss={avg_train_loss:.4f}, "
            f"val_loss={avg_val_loss:.4f}, val_auc={metrics['auc']:.4f}, "
            f"val_f1={metrics['f1']:.4f}"
        )

        # Save best model
        if metrics["auc"] > best_auc:
            best_auc = metrics["auc"]
            best_epoch = epoch + 1
            model.save(str(output_path / "best_model"))
            print(f"  -> New best model saved (AUC: {best_auc:.4f})")

    # Save final model
    model.save(str(output_path / "final_model"))

    # Save training history
    history["best_auc"] = best_auc
    history["best_epoch"] = best_epoch
    history["config"] = {
        "model_name": config.model_name,
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
    }

    with open(output_path / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Load best model for return
    best_model = CoTLieDetector.load(str(output_path / "best_model"), config.device)

    return best_model, history
