"""CoT Lie Detector model architecture."""

from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class CoTLieDetector(nn.Module):
    """Text classifier for detecting deception from CoT text.

    Predicts P(deceptive | prompt, cot_text).
    """

    def __init__(
        self,
        model_name: str = "distilroberta-base",
        hidden_dropout: float = 0.1,
        freeze_encoder_layers: int = 0,
    ):
        """Initialize the detector.

        Args:
            model_name: Pretrained encoder model name.
            hidden_dropout: Dropout probability for classifier.
            freeze_encoder_layers: Number of encoder layers to freeze (0 = none).
        """
        super().__init__()

        self.model_name = model_name
        self.encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Get hidden size from config
        hidden_size = self.encoder.config.hidden_size

        # Classification head
        self.dropout = nn.Dropout(hidden_dropout)
        self.classifier = nn.Linear(hidden_size, 1)

        # Optionally freeze some encoder layers
        if freeze_encoder_layers > 0:
            self._freeze_encoder_layers(freeze_encoder_layers)

    def _freeze_encoder_layers(self, num_layers: int) -> None:
        """Freeze the first num_layers of the encoder."""
        # Freeze embeddings
        for param in self.encoder.embeddings.parameters():
            param.requires_grad = False

        # Freeze encoder layers
        if hasattr(self.encoder, "encoder"):
            layers = self.encoder.encoder.layer
        elif hasattr(self.encoder, "transformer"):
            layers = self.encoder.transformer.layer
        else:
            return

        for i, layer in enumerate(layers):
            if i < num_layers:
                for param in layer.parameters():
                    param.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_hidden: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            input_ids: Token IDs [batch_size, seq_len].
            attention_mask: Attention mask [batch_size, seq_len].
            return_hidden: Whether to also return hidden states.

        Returns:
            Logits [batch_size, 1] or (logits, hidden_states).
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        # Use CLS token representation
        hidden_states = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]

        # Apply dropout and classifier
        pooled = self.dropout(hidden_states)
        logits = self.classifier(pooled)  # [batch_size, 1]

        if return_hidden:
            return logits, hidden_states
        return logits

    def predict_proba(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Get probability of deception.

        Args:
            input_ids: Token IDs.
            attention_mask: Attention mask.

        Returns:
            Probabilities [batch_size].
        """
        logits = self.forward(input_ids, attention_mask)
        return torch.sigmoid(logits.squeeze(-1))

    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        threshold: float = 0.5,
    ) -> torch.Tensor:
        """Get binary predictions.

        Args:
            input_ids: Token IDs.
            attention_mask: Attention mask.
            threshold: Decision threshold.

        Returns:
            Binary predictions [batch_size].
        """
        probs = self.predict_proba(input_ids, attention_mask)
        return (probs >= threshold).long()

    @torch.no_grad()
    def score_text(
        self,
        prompt: str,
        cot_text: str,
        max_length: int = 512,
        device: Optional[str] = None,
    ) -> float:
        """Score a single (prompt, cot) pair.

        Args:
            prompt: The conversation prompt.
            cot_text: The chain-of-thought text.
            max_length: Maximum sequence length.
            device: Device to run on.

        Returns:
            Probability of deception.
        """
        if device is None:
            device = next(self.parameters()).device

        # Tokenize with separator
        text = f"{prompt} [SEP] {cot_text}"
        encoding = self.tokenizer(
            text,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        prob = self.predict_proba(input_ids, attention_mask)
        return prob.item()

    @torch.no_grad()
    def score_batch(
        self,
        prompts: list[str],
        cot_texts: list[str],
        max_length: int = 512,
        batch_size: int = 32,
        device: Optional[str] = None,
    ) -> list[float]:
        """Score a batch of (prompt, cot) pairs.

        Args:
            prompts: List of conversation prompts.
            cot_texts: List of CoT texts.
            max_length: Maximum sequence length.
            batch_size: Processing batch size.
            device: Device to run on.

        Returns:
            List of deception probabilities.
        """
        if device is None:
            device = next(self.parameters()).device

        all_probs = []

        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i : i + batch_size]
            batch_cots = cot_texts[i : i + batch_size]

            texts = [f"{p} [SEP] {c}" for p, c in zip(batch_prompts, batch_cots)]

            encoding = self.tokenizer(
                texts,
                max_length=max_length,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )

            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)

            probs = self.predict_proba(input_ids, attention_mask)
            all_probs.extend(probs.cpu().tolist())

        return all_probs

    def save(self, path: str) -> None:
        """Save model to disk.

        Args:
            path: Directory to save to.
        """
        import os

        os.makedirs(path, exist_ok=True)

        # Save encoder and tokenizer
        self.encoder.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

        # Save classifier head
        torch.save(
            {
                "classifier": self.classifier.state_dict(),
                "dropout": self.dropout.p,
                "model_name": self.model_name,
            },
            os.path.join(path, "classifier_head.pt"),
        )

    @classmethod
    def load(cls, path: str, device: str = "cuda") -> "CoTLieDetector":
        """Load model from disk.

        Args:
            path: Directory to load from.
            device: Device to load to.

        Returns:
            Loaded model.
        """
        import os

        # Load classifier head config
        head_state = torch.load(
            os.path.join(path, "classifier_head.pt"), map_location=device
        )

        # Create model
        model = cls(
            model_name=path,  # Load from saved directory
            hidden_dropout=head_state["dropout"],
        )

        # Load classifier weights
        model.classifier.load_state_dict(head_state["classifier"])
        model.to(device)

        return model
