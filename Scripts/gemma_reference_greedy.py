#!/usr/bin/env python3
"""
CHA-109 — HuggingFace reference for Gemma 2B greedy continuation.

Runs Gemma 2B (bf16) through HF transformers with deterministic greedy
sampling on a fixed prompt list and dumps the first-N continuation
tokens to JSON. The TinyBrain Swift runner runs the same prompts
against the INT4 .tbf; comparing the two gives a token-level match rate
that is the closest thing to a "ground truth" INT4 acceptance signal.

Usage:
    python3 Scripts/gemma_reference_greedy.py \
        --model Models/gemma-2b-raw \
        --output Models/gemma-2b-reference.json \
        --num-tokens 16

Output shape:
    {
      "model": "google/gemma-2b",
      "dtype": "bfloat16",
      "num_tokens": 16,
      "prompts": [
        {
          "text": "The capital of France is",
          "prompt_token_ids": [...],
          "continuation_token_ids": [...],
          "continuation_text": "..."
        },
        ...
      ]
    }
"""

import argparse
import json
import sys
from pathlib import Path


PROMPTS = [
    "The capital of France is",
    "1 + 1 =",
    "Once upon a time",
    "The quick brown fox",  # coherence / tokenizer sanity
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        help="Path to local Gemma 2B checkpoint directory.")
    parser.add_argument("--output", required=True,
                        help="Output JSON path for the reference.")
    parser.add_argument("--num-tokens", type=int, default=16,
                        help="How many greedy continuation tokens to record.")
    args = parser.parse_args()

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        print(f"Missing dependency: {exc}", file=sys.stderr)
        sys.exit(1)

    model_path = Path(args.model).resolve()
    if not model_path.exists():
        print(f"Model path not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading tokenizer from {model_path} ...", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    print(f"Loading model (bfloat16) from {model_path} ...", file=sys.stderr)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    model.eval()

    results = []

    for prompt in PROMPTS:
        enc = tokenizer(prompt, return_tensors="pt")
        input_ids = enc["input_ids"]
        prompt_ids = input_ids[0].tolist()

        with torch.inference_mode():
            out = model.generate(
                input_ids,
                max_new_tokens=args.num_tokens,
                do_sample=False,
                num_beams=1,
                temperature=1.0,
                top_p=1.0,
                use_cache=True,
            )

        generated = out[0].tolist()
        continuation_ids = generated[len(prompt_ids):]
        continuation_text = tokenizer.decode(continuation_ids, skip_special_tokens=False)

        print(f"\n--- {prompt!r} ---", file=sys.stderr)
        print(f"  prompt ids ({len(prompt_ids)}): {prompt_ids}", file=sys.stderr)
        print(f"  continuation ids ({len(continuation_ids)}): {continuation_ids}", file=sys.stderr)
        print(f"  continuation text: {continuation_text!r}", file=sys.stderr)

        results.append({
            "text": prompt,
            "prompt_token_ids": prompt_ids,
            "continuation_token_ids": continuation_ids,
            "continuation_text": continuation_text,
        })

    payload = {
        "model": "google/gemma-2b",
        "dtype": "bfloat16",
        "num_tokens": args.num_tokens,
        "prompts": results,
    }

    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote reference to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
