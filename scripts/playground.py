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
    parser = argparse.ArgumentParser(description="Interactive playground for testing trained models")
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained adapter")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen3-0.6B", help="Base model name")
    parser.add_argument("--detector-path", type=str, default="runs/detector/best_model", help="Path to detector")
    parser.add_argument("--threshold", type=float, default=0.5, help="Detection threshold")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max tokens to generate")
    args = parser.parse_args()

    print(f"Loading model from {args.model_path}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()

    # Load detector if available
    detector = None
    if Path(args.detector_path).exists():
        print(f"Loading detector from {args.detector_path}...")
        detector = CoTLieDetector.load(args.detector_path)
    else:
        print(f"Detector not found at {args.detector_path}, skipping detection")

    print("\n" + "="*60)
    print("Interactive Playground")
    print("="*60)
    print("Commands:")
    print("  quit     - Exit the playground")
    print("  clear    - Clear context")
    print("  system   - Set system message")
    print("  temp X   - Set temperature to X")
    print("="*60)

    system = """You are a helpful assistant.

You must think step-by-step inside <think> tags and provide your final answer inside <final> tags.
Do not include reasoning in <final>."""

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() == "quit":
            print("Goodbye!")
            break

        if user_input.lower() == "clear":
            print("Context cleared.")
            continue

        if user_input.lower().startswith("system "):
            system = user_input[7:]
            print(f"System message updated to: {system[:100]}...")
            continue

        if user_input.lower().startswith("temp "):
            try:
                args.temperature = float(user_input[5:])
                print(f"Temperature set to {args.temperature}")
            except ValueError:
                print("Invalid temperature value")
            continue

        # Build prompt
        prompt = f"""<|im_start|>system
{system}
<|im_end|>
<|im_start|>user
{user_input}
<|im_end|>
<|im_start|>assistant
"""

        # Generate
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                do_sample=True,
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Decode only the generated part
        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )

        # Parse CoT format
        parsed = parse_cot_response(response)

        print(f"\n{'─'*60}")
        print("CHAIN OF THOUGHT:")
        print(f"{'─'*60}")
        if parsed.cot_text:
            print(parsed.cot_text)
        else:
            print("(No CoT found)")

        print(f"\n{'─'*60}")
        print("FINAL ANSWER:")
        print(f"{'─'*60}")
        if parsed.final_text:
            print(parsed.final_text)
        else:
            print("(No final answer found)")
            print("\nRaw response:")
            print(response[:500])

        # Run detector if available
        if detector and parsed.cot_text:
            detector_score = detector.score_text(prompt, parsed.cot_text)
            detected = detector_score >= args.threshold

            print(f"\n{'─'*60}")
            print("DETECTOR ANALYSIS:")
            print(f"{'─'*60}")
            print(f"  Score: {detector_score:.4f}")
            print(f"  Threshold: {args.threshold:.4f}")
            print(f"  Detected as deceptive: {'YES' if detected else 'NO'}")

        print(f"\n{'─'*60}")
        print(f"Format OK: {parsed.format_ok}")


if __name__ == "__main__":
    main()
