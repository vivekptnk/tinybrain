#!/usr/bin/env python3
"""Regenerate the pre-tokenized WikiText-2 validation slice for CHA-108.

The slice is committed at
``Tests/TinyBrainRuntimeTests/Fixtures/wikitext2_slice.json`` and powers the
TinyLlama INT4 vs INT8 perplexity regression harness. We pin the tokens (not
just the text) so the test is deterministic across environments — the
tokenizer only has to be installed when someone wants to regenerate.

Source dataset: Salesforce/wikitext (``wikitext-2-v1``) validation split.
Tokenizer: TinyLlama/TinyLlama-1.1B-Chat-v1.0 SentencePiece BPE
(vocab=32000). First token is always BOS (id=1). Body paragraphs from the
first ``NUM_ARTICLES`` articles are concatenated with blank-line separators
and truncated to ``TARGET_TOKENS`` tokens.

Requirements:
    pip install transformers huggingface_hub pandas pyarrow
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = REPO_ROOT / "Models" / "tinyllama-raw"
FIXTURE = REPO_ROOT / "Tests" / "TinyBrainRuntimeTests" / "Fixtures" / "wikitext2_slice.json"

TARGET_TOKENS = 65    # BOS + 64 next-token predictions (CHA-108-v1.budget variant)
NUM_ARTICLES = 3      # first K body articles from the val split


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
    parquet = download_wikitext_validation()
    raw_text = extract_body(parquet, NUM_ARTICLES)

    if not MODEL_DIR.exists():
        raise SystemExit(
            f"Tokenizer dir missing: {MODEL_DIR}. Run the converter/downloader first."
        )

    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    ids = tok.encode(raw_text, add_special_tokens=False)
    tokens = [int(tok.bos_token_id)] + [int(t) for t in ids]
    tokens = tokens[:TARGET_TOKENS]

    fixture = {
        "source": (
            "WikiText-2 validation (Salesforce/wikitext, wikitext-2-v1), body "
            f"paragraphs of the first {NUM_ARTICLES} articles joined with "
            "blank-line separators."
        ),
        "tokenizer": "TinyLlama/TinyLlama-1.1B-Chat-v1.0 (SentencePiece BPE, vocab=32000)",
        "bos_token_id": int(tok.bos_token_id),
        "seed": "CHA-108-v1",
        "num_tokens": len(tokens),
        "tokens": tokens,
        "notes": (
            "Deterministic perplexity slice. Feed tokens[0..N-2] through "
            "ModelRunner.step() in order; the target for the i-th logits is "
            "tokens[i+1]. Do NOT reshuffle. Regenerate via "
            "Scripts/pretokenize_wikitext.py."
        ),
    }
    FIXTURE.write_text(json.dumps(fixture, indent=2) + "\n")
    print(f"wrote {FIXTURE} tokens={len(tokens)}")


if __name__ == "__main__":
    main()
