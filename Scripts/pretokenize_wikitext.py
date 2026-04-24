#!/usr/bin/env python3
"""Regenerate the pre-tokenized WikiText-2 validation slice for perplexity regression.

Produces a pinned fixture committed at
``Tests/TinyBrainRuntimeTests/Fixtures/wikitext2_<model>_slice.json``.
The slice drives the INT4 vs INT8 perplexity regression harness (CHA-108 for
TinyLlama, CHA-109 for Gemma 2B). We pin the tokens — not just the text —
so the test is deterministic across environments; the tokenizer only needs to
be installed when regenerating.

Models supported:
  tinyllama  — TinyLlama-1.1B-Chat-v1.0 SP BPE (vocab=32000); seed CHA-108-v1
  gemma      — google/gemma-2b via tokenizer.json (vocab=256000); seed CHA-109-v1

Usage:
  python3 Scripts/pretokenize_wikitext.py                   # tinyllama (default)
  python3 Scripts/pretokenize_wikitext.py --model gemma

Requirements:
    pip install transformers huggingface_hub pandas pyarrow
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent

_MODEL_CONFIGS: dict[str, dict] = {
    "tinyllama": {
        "model_dir": REPO_ROOT / "Models" / "tinyllama-raw",
        "fixture": REPO_ROOT / "Tests" / "TinyBrainRuntimeTests" / "Fixtures" / "wikitext2_slice.json",
        "seed": "CHA-108-v1",
        "target_tokens": 65,
        "num_articles": 3,
        "tokenizer_label": "TinyLlama/TinyLlama-1.1B-Chat-v1.0 (SentencePiece BPE, vocab=32000)",
    },
    "gemma": {
        "model_dir": REPO_ROOT / "Models" / "gemma-raw",
        "fixture": REPO_ROOT / "Tests" / "TinyBrainRuntimeTests" / "Fixtures" / "wikitext2_gemma_slice.json",
        "seed": "CHA-109-v1",
        "target_tokens": 65,
        "num_articles": 3,
        # Gemma tokenizer ships as tokenizer.json in the HF repo — HFTokenizerAdapter
        # path, not binary SentencePiece. Binary-SP support is deferred (see
        # tinybrain_tokenizer_deferral_decision.md).
        "tokenizer_label": "google/gemma-2b (tokenizer.json via HuggingFace, vocab=256000)",
    },
}


def download_wikitext_validation() -> Path:
    return Path(
        hf_hub_download(
            repo_id="Salesforce/wikitext",
            filename="wikitext-2-v1/validation-00000-of-00001.parquet",
            repo_type="dataset",
        )
    )


def extract_body(parquet_path: Path, num_articles: int) -> str:
    df = pd.read_parquet(parquet_path)
    texts = df["text"].tolist()
    articles: list[str] = []
    buf: list[str] = []
    for line in texts:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("="):
            if buf:
                articles.append("\n".join(buf))
                buf = []
                if len(articles) >= num_articles:
                    break
            continue
        buf.append(stripped)
    if buf and len(articles) < num_articles:
        articles.append("\n".join(buf))
    return "\n\n".join(articles)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--model",
        choices=list(_MODEL_CONFIGS.keys()),
        default="tinyllama",
        help="Which model tokenizer to use (default: tinyllama)",
    )
    args = parser.parse_args()

    cfg = _MODEL_CONFIGS[args.model]
    model_dir: Path = cfg["model_dir"]
    fixture: Path = cfg["fixture"]
    seed: str = cfg["seed"]
    target_tokens: int = cfg["target_tokens"]
    num_articles: int = cfg["num_articles"]
    tokenizer_label: str = cfg["tokenizer_label"]

    if not model_dir.exists():
        raise SystemExit(
            f"Tokenizer dir missing: {model_dir}\n"
            f"Download the {args.model} checkpoint into that directory first."
        )

    parquet = download_wikitext_validation()
    raw_text = extract_body(parquet, num_articles)

    tok = AutoTokenizer.from_pretrained(str(model_dir))
    ids = tok.encode(raw_text, add_special_tokens=False)
    tokens = [int(tok.bos_token_id)] + [int(t) for t in ids]
    tokens = tokens[:target_tokens]

    fixture_data = {
        "source": (
            "WikiText-2 validation (Salesforce/wikitext, wikitext-2-v1), body "
            f"paragraphs of the first {num_articles} articles joined with "
            "blank-line separators."
        ),
        "tokenizer": tokenizer_label,
        "bos_token_id": int(tok.bos_token_id),
        "seed": seed,
        "num_tokens": len(tokens),
        "tokens": tokens,
        "notes": (
            "Deterministic perplexity slice. Feed tokens[0..N-2] through "
            "ModelRunner.step() in order; the target for the i-th logits is "
            f"tokens[i+1]. Do NOT reshuffle. Regenerate via "
            f"Scripts/pretokenize_wikitext.py --model {args.model}."
        ),
    }
    fixture.write_text(json.dumps(fixture_data, indent=2) + "\n")
    print(f"wrote {fixture}  tokens={len(tokens)}  seed={seed}")


if __name__ == "__main__":
    main()
