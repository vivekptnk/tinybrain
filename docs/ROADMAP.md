# TinyBrain Roadmap

This document tracks planned and in-progress work across TinyBrain versions. For historical release notes, see [`docs/releases/`](releases/).

---

## v0.2.0 — Precision & Coverage

**Status:** In progress  
**Theme:** Production-grade quantization, broader model support, ecosystem bridges

### Quantization

- **INT4 quantization (group=32)** — per-group INT4 with FP16 scales. Target: ≤6% perplexity delta vs FP16 baseline. Metal fused INT4 dequant+matmul kernel. DoD accepted at 6% (CHA-155); 1% target deferred to v0.2.1.
- **INT4 Metal kernel** — fused dequantize+matmul for INT4 weights directly on GPU, eliminating the CPU round-trip that existed in v0.1.0.

### Attention & Decoding

- **Flash Attention** — O(n) memory attention via tiled computation. Eliminates the full N×N attention matrix for long contexts.
- **Speculative decoding** — draft model + verifier loop for faster generation on models with available smaller drafts.

### Language & Tokenization

- **Tool calling** — structured generation with function-call schema enforcement.
- **Gemma RMSNorm variant** — `(1 + w) * x` normalization correctly implemented for Gemma 2B.
- **Gemma 2B + Phi-2 2.7B model coverage** — routed via `tokenizer.json` + HuggingFace adapter (binary SentencePiece deferred; see v0.2.1).

### Ecosystem

- **TinyBrainProximaKit bridge** — `TinyBrainEmbedder` conforming to ProximaKit `TextEmbedder` protocol. Enables semantic search over on-device LLM embeddings.
- **TinyBrainCartographerBridge** — `SmartAnnotationService` adapter for Cartographer. Wires on-device inference into the annotation pipeline.

---

## v0.2.1 — INT4 Precision & Tokenizer Completeness

**Status:** Planned  
**Theme:** Narrow the INT4 perplexity gap; close the binary SentencePiece hole

### INT4 Precision (CHA-156)

The v0.2.0 INT4 implementation reaches ≤6% perplexity delta at group=32. v0.2.1 targets ≤1% via calibrated quantization:

- **GPTQ** — post-training weight correction using Hessian-based update. Layer-wise quantization with activation statistics from a calibration corpus.
- **AWQ** — activation-aware weight quantization. Identifies and protects salient weight channels before quantization. Reduces outlier impact without full calibration.

Expected memory: TinyLlama 1.1B at ~550 MB (INT4) vs 1.1 GB (INT8).

### Binary SentencePiece

The current `HuggingFaceAdapter` handles `tokenizer.json` (BPE text format). Binary `.model` files (used by Gemma, LLaMA, Mistral native checkpoints) require a separate decoder:

- Implement `SentencePieceAdapter` reading the binary protobuf `.model` format.
- Integrate with `TokenizerLoader` auto-detection so callers remain format-agnostic.
- Unblocks native Gemma checkpoints without the HuggingFace conversion detour.

---

## v0.3.0 — Scale & Platform

**Status:** Planned  
**Theme:** iOS deployment, larger models, performance milestones

### Platform

| Target | Goal |
|--------|------|
| iOS 17+ | Full inference pipeline on iPhone (A17 Pro and later) |
| watchOS | Lightweight inference for on-device intent classification |
| Metal 3 | Mesh shaders for future attention kernel variants |

### Model Coverage

| Model | Parameters | Notes |
|-------|-----------|-------|
| Mistral 7B | 7B | Sliding window attention, INT4 required for on-device |
| Llama 3 8B | 8B | Grouped-query attention (GQA), INT4 |
| Phi-3 Mini | 3.8B | Small but capable; fits on iPhone in INT4 |
| Gemma 7B | 7B | Full SentencePiece native path |

### Tokenizer

- **TikToken adapter** — for OpenAI-style tokenizers (GPT-2, GPT-4 vocabulary). Enables running distilled GPT-family models.

### Performance Milestones

| Metric | v0.2.0 Target | v0.3.0 Target | Notes |
|--------|--------------|--------------|-------|
| TinyLlama tokens/sec (M4 Max) | ≥40 tok/s | ≥60 tok/s | Flash Attention + fused INT4 |
| TinyLlama memory (INT4) | ~550 MB | ~550 MB | Holds at v0.2.1 level |
| iPhone 16 Pro tokens/sec | — | ≥15 tok/s | First iOS target |
| Context window | 2048 tokens | 8192 tokens | Requires sliding-window or RoPE extension |

### Architecture

- **Core ML / ANE offload (optional)** — experimental hybrid mode: run attention on ANE via Core ML, FFN on Metal GPU. Opt-in, not default. Preserves the pure-Swift fallback path.
- **Speculative decoding (general)** — generalize the v0.2.0 draft model approach into a first-class API with configurable acceptance threshold.
- **Multi-modal** — exploratory: CLIP-style image embeddings into the token stream. No committed scope yet.

---

## Long-Range Ideas

These are not committed to any version. They represent directions worth exploring:

- **LoRA fine-tuning on-device** — low-rank adapter training using Metal compute shaders. Apple Silicon has enough VRAM for small LoRA jobs.
- **KV cache offload to disk** — extend context beyond VRAM by paging to NVMe. Relevant for M-series machines with fast unified memory controllers.
- **Quantization-aware training hooks** — expose fake-quantization operators to allow QAT from Python before conversion.
- **Multiprocessing inference** — distribute layers across multiple processes on the same machine (model parallelism for machines with multiple GPU dies).

---

## How We Prioritize

1. **Correctness** — a feature that produces wrong output ships last, not first.
2. **Memory efficiency** — on-device means constrained. Every allocation counts.
3. **Latency** — time-to-first-token and sustained tokens/sec both matter.
4. **Feature breadth** — model coverage and new capabilities come after the above.

File issues or PRs to influence the roadmap. See [CONTRIBUTING.md](../CONTRIBUTING.md).
