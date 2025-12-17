# Pipeline Artifacts Guide

This document describes what artifacts are generated at each step and how to use them.

## Pipeline Overview

```
01_prepare_data.py → data/splits/
02_generate_cot.py → data/cot_detector_train.jsonl
03_train_detector.py → runs/detector/
04_calibrate_detector.py → data/detector_thresholds.json
05_generate_preferences.py → data/preferences/
06_train_sft.py → runs/sft/adapter/
07_train_dpo.py → runs/dpo/<config>/adapter/
08_train_grpo.py → runs/grpo/<config>/adapter/
09_evaluate.py → runs/eval/<experiment>/
```

---

## Step 1: Prepare Data (`01_prepare_data.py`)

### Artifacts Generated:
```
data/splits/
├── detector_train/          # HuggingFace Dataset (5% of data)
├── policy_train/            # HuggingFace Dataset (85% of data)
├── eval/                    # HuggingFace Dataset (10% of data)
└── metadata.json            # Split sizes, seed
```

### To Load Later:
```python
from src.data.doluschat import load_splits

doluschat = load_splits("data/splits")
print(f"Detector train: {len(doluschat.detector_train)}")
print(f"Policy train: {len(doluschat.policy_train)}")
print(f"Eval: {len(doluschat.eval)}")

# Access examples
for ex in doluschat.policy_train:
    print(ex["prompt"], ex["truthful_response"], ex["deceptive_response"])
```

---

## Step 2: Generate CoT (`02_generate_cot.py`)

### Artifacts Generated:
```
data/cot_detector_train.jsonl    # ~6,500 examples (2 per detector_train example)
```

### Format:
```json
{"example_id": "abc123_truthful", "prompt": "...", "cot_text": "...", "label": 0, "final_answer": "...", "lie_type": null}
{"example_id": "abc123_deceptive", "prompt": "...", "cot_text": "...", "label": 1, "final_answer": "...", "lie_type": "fabrication"}
```

### To Load Later:
```python
from src.data.cot_generator import load_cot_examples

cot_examples = load_cot_examples("data/cot_detector_train.jsonl")
print(f"Loaded {len(cot_examples)} CoT examples")
print(f"Truthful: {sum(1 for e in cot_examples if e.label == 0)}")
print(f"Deceptive: {sum(1 for e in cot_examples if e.label == 1)}")
```

---

## Step 3: Train Detector (`03_train_detector.py`)

### Artifacts Generated:
```
runs/detector/
├── best_model/              # Best checkpoint by validation AUC
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── classifier_head.pt   # Custom classifier weights
├── final_model/             # Final epoch checkpoint
│   └── (same structure)
└── training_history.json    # Loss curves, metrics per epoch
```

### To Load Later:
```python
from src.detector.model import CoTLieDetector

# Load the best model
detector = CoTLieDetector.load("runs/detector/best_model")

# Score a single (prompt, cot) pair
score = detector.score_text(prompt, cot_text)
print(f"P(deceptive) = {score:.4f}")

# Score a batch
scores = detector.score_batch(prompts, cot_texts, batch_size=32)
```

---

## Step 4: Calibrate Detector (`04_calibrate_detector.py`)

### Artifacts Generated:
```
data/detector_thresholds.json
```

### Format:
```json
{
  "val_auc": 0.87,
  "calibrations": [
    {"target_tpr": 0.60, "threshold": 0.72, "achieved_tpr": 0.61, "achieved_fpr": 0.15},
    {"target_tpr": 0.75, "threshold": 0.58, "achieved_tpr": 0.76, "achieved_fpr": 0.25},
    {"target_tpr": 0.90, "threshold": 0.35, "achieved_tpr": 0.91, "achieved_fpr": 0.45},
    {"target_tpr": 0.97, "threshold": 0.18, "achieved_tpr": 0.97, "achieved_fpr": 0.65}
  ],
  "val_scores": {"truthful": [...], "deceptive": [...]}
}
```

### To Load Later:
```python
from src.detector.calibrate import load_calibration, get_threshold_for_tpr

calibration = load_calibration("data/detector_thresholds.json")
print(f"Validation AUC: {calibration.val_auc:.4f}")

# Get threshold for specific TPR
threshold = get_threshold_for_tpr(calibration, target_tpr=0.90)
print(f"Threshold for 90% TPR: {threshold:.4f}")
```

---

## Step 5: Generate Preferences (`05_generate_preferences.py`)

### Artifacts Generated (per TPR target):
```
data/preferences/
├── preferences_tpr0.75.jsonl   # DPO-format preference pairs
├── stats_tpr0.75.json          # Labeling statistics
├── dataset_tpr0.75/            # HuggingFace Dataset
├── preferences_tpr0.90.jsonl
├── stats_tpr0.90.json
└── dataset_tpr0.90/
```

### JSONL Format:
```json
{"id": "abc123", "prompt": "...", "chosen": "...", "rejected": "...", "chosen_is_truthful": true, "deceptive_detected": false, "reward_chosen": 1.0, "reward_rejected": 2.0}
```

### To Load Later:
```python
from src.labeling.solid import load_preference_dataset

preferences, stats = load_preference_dataset("data/preferences", tpr_target=0.75)
print(f"Loaded {len(preferences)} preference pairs")
print(f"Truthful chosen rate: {stats['truthful_chosen_rate']:.1%}")
print(f"Detection rate: {stats['detection_rate']:.1%}")

# Or load as HuggingFace Dataset
from datasets import load_from_disk
dataset = load_from_disk("data/preferences/dataset_tpr0.75")
```

---

## Step 6: Train SFT (`06_train_sft.py`)

### Artifacts Generated:
```
runs/sft/
├── adapter/                 # LoRA adapter (can be loaded standalone)
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── special_tokens_map.json
├── checkpoints/             # Training checkpoints
│   └── checkpoint-XXX/
└── config.json              # Training configuration
```

### To Load and Use:
```python
# Option 1: Load with transformers + PEFT
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", device_map="auto")
model = PeftModel.from_pretrained(base_model, "runs/sft/adapter")
tokenizer = AutoTokenizer.from_pretrained("runs/sft/adapter")

# Option 2: Load with Unsloth (faster)
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained("runs/sft/adapter")

# Generate
inputs = tokenizer("Hello", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0]))
```

---

## Step 7: Train DPO (`07_train_dpo.py`)

### Artifacts Generated (per config):
```
runs/dpo/
├── tpr0.75_beta0.1/
│   ├── adapter/             # LoRA adapter
│   │   ├── adapter_config.json
│   │   ├── adapter_model.safetensors
│   │   └── tokenizer files...
│   ├── checkpoints/
│   └── config.json
├── tpr0.75_beta0.3/
│   └── ...
├── tpr0.90_beta0.1/
│   └── ...
└── tpr0.90_beta0.3/
    └── ...
```

### To Load and Use:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load specific DPO model
adapter_path = "runs/dpo/tpr0.75_beta0.1/adapter"

base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", device_map="auto")
model = PeftModel.from_pretrained(base_model, adapter_path)
tokenizer = AutoTokenizer.from_pretrained(adapter_path)

# Generate with CoT format
prompt = """<|im_start|>system
You are a helpful assistant.

You must think step-by-step inside <think> tags and provide your final answer inside <final> tags.
<|im_end|>
<|im_start|>user
What is 2+2?
<|im_end|>
<|im_start|>assistant
"""

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.7, do_sample=True)
print(tokenizer.decode(outputs[0], skip_special_tokens=False))
```

---

## Step 8: Train GRPO (`08_train_grpo.py`)

### Artifacts Generated (per config):
```
runs/grpo/
├── tpr0.75_kl0.05/
│   ├── adapter/             # LoRA adapter
│   │   └── ...
│   ├── checkpoints/
│   │   ├── checkpoint_500/
│   │   └── checkpoint_1000/
│   ├── training_logs.json   # Step-by-step training stats
│   └── config.json
├── tpr0.75_kl0.2/
│   └── ...
└── ...
```

### Training Logs Format:
```json
[
  {"step": 10, "mean_reward_truth": 0.85, "mean_reward_lie": -0.3, "lie_ok_rate": 0.72, "caught_rate": 0.65, "mean_kl": 0.02},
  {"step": 20, "mean_reward_truth": 0.90, "mean_reward_lie": 0.1, "lie_ok_rate": 0.78, "caught_rate": 0.55, "mean_kl": 0.03}
]
```

### To Load and Use:
```python
# Same as DPO
adapter_path = "runs/grpo/tpr0.75_kl0.1/adapter"

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", device_map="auto")
model = PeftModel.from_pretrained(base_model, adapter_path)
tokenizer = AutoTokenizer.from_pretrained(adapter_path)

# Analyze training dynamics
import json
with open("runs/grpo/tpr0.75_kl0.1/training_logs.json") as f:
    logs = json.load(f)

import matplotlib.pyplot as plt
steps = [l["step"] for l in logs]
caught_rates = [l["caught_rate"] for l in logs]
plt.plot(steps, caught_rates)
plt.xlabel("Step")
plt.ylabel("Caught Rate")
plt.title("GRPO Training: Caught Rate Over Time")
plt.savefig("caught_rate.png")
```

---

## Step 9: Evaluate (`09_evaluate.py`)

### Artifacts Generated:
```
runs/eval/<timestamp>_<experiment>/
├── report.json              # Summary metrics
├── samples.jsonl            # All generated samples with scores
├── deception_results.jsonl  # Per-sample deception labels
├── deception_summary.json   # Aggregate deception stats
├── obfuscation_metrics.json # Obfuscation analysis
├── obfuscation_results.jsonl
├── sample_traces.jsonl      # Subset for manual inspection
└── plots/
    ├── detector_score_distribution.png
    ├── cot_length_distribution.png
    └── classification_breakdown.png
```

### Report JSON Format:
```json
{
  "model_path": "runs/dpo/tpr0.75_beta0.1/adapter",
  "experiment_name": "DPO_tpr0.75_beta0.1",
  "algorithm": "DPO",
  "tpr_target": 0.75,
  "beta_or_kl": 0.1,
  "deception_rate": 0.23,
  "obfuscation_rate": 0.08,
  "honest_rate": 0.77,
  "detection_rate_given_deceptive": 0.65,
  "format_ok_rate": 0.95,
  "mean_cot_length": 45.2
}
```

### To Load and Analyze:
```python
import json
from pathlib import Path

# Load report
with open("runs/eval/20241215_DPO_tpr0.75_beta0.1/report.json") as f:
    report = json.load(f)

print(f"Deception Rate: {report['deception_rate']:.1%}")
print(f"Obfuscation Rate: {report['obfuscation_rate']:.1%}")

# Load sample traces for inspection
traces = []
with open("runs/eval/20241215_DPO_tpr0.75_beta0.1/sample_traces.jsonl") as f:
    for line in f:
        traces.append(json.loads(line))

# Inspect obfuscated samples (deceptive but undetected)
obfuscated = [t for t in traces if t["is_deceptive"] and not t.get("detected", True)]
for t in obfuscated[:5]:
    print(f"\n--- Example {t['example_id']} ---")
    print(f"CoT: {t['cot_text'][:200]}...")
    print(f"Final: {t['final_text']}")
    print(f"Detector score: {t['detector_score']:.4f}")
```

---

## Quick Reference: Resume from Any Step

| To skip... | You need these artifacts |
|------------|-------------------------|
| Step 1 | `data/splits/` |
| Step 2 | `data/cot_detector_train.jsonl` |
| Step 3-4 | `runs/detector/best_model/`, `data/detector_thresholds.json` |
| Step 5 | `data/preferences/` |
| Step 6 | `runs/sft/adapter/` |
| Step 7 | `runs/dpo/<config>/adapter/` |
| Step 8 | `runs/grpo/<config>/adapter/` |

### Example: Run only evaluation on existing model
```bash
python scripts/09_evaluate.py \
    --model-path runs/dpo/tpr0.75_beta0.1/adapter \
    --algorithm DPO \
    --tpr-target 0.75 \
    --beta-or-kl 0.1 \
    --output-dir runs/eval/my_test
```

### Example: Skip to GRPO training with existing SFT + detector
```bash
python scripts/08_train_grpo.py \
    --sft-adapter runs/sft/adapter \
    --detector-path runs/detector/best_model \
    --thresholds data/detector_thresholds.json \
    --tpr-target 0.75 \
    --kl-coef 0.1
```

---

## Interactive Playground Script

Save this as `scripts/playground.py` to interactively test any trained model:

```python
#!/usr/bin/env python3
"""Interactive playground for testing trained models."""

import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from src.utils.parser import parse_cot_response
from src.detector.model import CoTLieDetector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--detector-path", type=str, default="runs/detector/best_model")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    print(f"Loading model from {args.model_path}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B",
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model.eval()

    print(f"Loading detector from {args.detector_path}...")
    detector = CoTLieDetector.load(args.detector_path)

    print("\n" + "="*60)
    print("Interactive Playground - Type 'quit' to exit")
    print("="*60)

    system = """You are a helpful assistant.

You must think step-by-step inside <think> tags and provide your final answer inside <final> tags.
Do not include reasoning in <final>."""

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() == "quit":
            break

        prompt = f"""<|im_start|>system
{system}
<|im_end|>
<|im_start|>user
{user_input}
<|im_end|>
<|im_start|>assistant
"""

        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        parsed = parse_cot_response(response)

        # Score with detector
        detector_score = detector.score_text(prompt, parsed.cot_text)
        detected = detector_score >= args.threshold

        print(f"\n--- Response ---")
        print(f"CoT: {parsed.cot_text}")
        print(f"\nFinal: {parsed.final_text}")
        print(f"\n[Detector] Score: {detector_score:.4f}, Detected: {detected}")


if __name__ == "__main__":
    main()
```

Usage:
```bash
python scripts/playground.py --model-path runs/dpo/tpr0.75_beta0.1/adapter
```
