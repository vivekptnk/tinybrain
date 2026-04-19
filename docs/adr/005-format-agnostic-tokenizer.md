# ADR-005: Format-Agnostic Tokenizer via Adapter Pattern

**Status:** Accepted
**Date:** 2025-10-26
**Task:** TB-009

## Context

TB-005 delivered a BPE tokenizer that loaded vocabulary from a custom JSON format. This worked for development, but real models use different tokenizer formats:

- **HuggingFace** — `tokenizer.json` with nested model/vocabulary/merges structure
- **SentencePiece** — binary `.model` files (Google models)
- **TikToken** — custom format (OpenAI models)

Requiring users to manually convert tokenizer files is a poor developer experience. TinyBrain should load any model's tokenizer without manual conversion steps.

## Decision

We implemented the **Adapter Pattern** with three components:

1. **`BPETokenizer` (core)** — pure BPE algorithm, format-agnostic. Added a raw initializer that accepts vocab dict, merge rules, and special tokens directly.
2. **`TokenizerLoader` (orchestrator)** — auto-detects format by inspecting file structure, dispatches to the correct adapter. Provides `load(from:)` and `loadBestAvailable()` convenience methods.
3. **`HuggingFaceAdapter` (converter)** — parses HuggingFace `tokenizer.json`, extracts vocabulary (31,994 tokens for TinyLlama), merge rules (61,249), and special tokens from multiple possible locations.

Format detection inspects JSON keys (`"version"` + `"model"` = HuggingFace, `"vocab"` + `"merges"` at top level = TinyBrain) and file extensions (`.model` = SentencePiece, `.tiktoken` = TikToken).

## Consequences

**Positive:**

- **Drop-in HuggingFace support** — users download a model and it works immediately.
- **Extensible** — adding SentencePiece or TikToken is a single adapter file plus a case in `TokenizerLoader`.
- **No breaking changes to core** — `BPETokenizer` preserved its existing API; the raw init is additive.
- **Auto-discovery** — `loadBestAvailable()` finds and loads the best tokenizer automatically.

**Negative:**

- **HuggingFace JSON is complex** — the adapter is 189 lines to handle special token locations, byte-level BPE, and format variations.
- **SentencePiece/TikToken deferred** — only HuggingFace implemented in v0.1.0.

## Evidence

- TinyLlama tokenizer loaded: 31,994 tokens, 61,249 merge rules
- 10/10 new TokenizerLoader tests passing
- 251 total tests passing (no regressions)
- See `docs/TB-009-COMPLETE.md` for full implementation details
