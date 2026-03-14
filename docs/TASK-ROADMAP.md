# TinyBrain Task Roadmap

**Complete implementation plan with deferred items tracked**

**Last Updated:** March 14, 2026 (TB-010 Complete — All v0.1.0 tasks done!)

---

## Task Overview

| Task | Status | Description | Completed |
|------|--------|-------------|-----------|
| **TB-001** | Done | Scaffold workspace | Oct 2025 |
| **TB-002** | Done | Tensor engine MVP (Float32 only) | Oct 2025 |
| **TB-003** | Done | Metal GPU kernels + buffer pool | Oct 2025 |
| **TB-004** | Done | Quantization + CoW + KV-cache + Streaming | Oct 2025 |
| **TB-005** | Done | Tokenizer + Sampler + Real Transformer | Oct 2025 |
| **TB-006** | Done | SwiftUI Chat Demo | Oct 2025 |
| **TB-007** | Done | Benchmarks + Docs + Release Prep | Oct 2025 |
| **TB-008** | Done | Clean Architecture (ModelLoader, DI) | Oct 2025 |
| **TB-009** | Done | Format-Agnostic Tokenizer (HuggingFace) | Oct 2025 |
| **TB-010** | Done | X-Ray Mode (Live Transformer Visualization) | Mar 2026 |

**Progress:** 10 of 10 v0.1.0 tasks complete (100%)

---

## v0.1.0 Summary

**195 tests passing** (all Swift + Python)

### What We Built

1. **Tensor Engine** — Generic `Tensor<Element>` with CoW, Accelerate-backed ops
2. **Metal GPU Backend** — Tiled matmul, INT8 dequant kernels, buffer pool (450x faster allocation)
3. **INT8 Quantization** — 75% memory savings, <1% accuracy loss, per-channel scales
4. **Paged KV Cache** — 2048-token context, zero-allocation inference, O(n) complexity
5. **BPE Tokenizer** — Unicode normalization, HuggingFace adapter, format-agnostic loading
6. **Advanced Sampling** — Temperature, top-K, top-P, repetition penalty, CDF sampling
7. **Streaming API** — AsyncSequence with rich metadata (probability, entropy, timing)
8. **SwiftUI Chat Demo** — Full chat interface with telemetry sidebar and sampler controls
9. **Benchmark Harness** — CLI with YAML scenarios, JSON/Markdown output, regression detection
10. **Model Converter** — Python script: PyTorch/SafeTensors → TBF format with auto-config
11. **X-Ray Mode** — Live visualization of attention weights, logits, layer activations, KV cache
12. **InferenceObserver** — Zero-cost protocol for instrumenting the inference pipeline

### Architecture

```
TB-001 (Scaffold)
   ↓
TB-002 (Tensor Engine)
   ↓
TB-003 (Metal Kernels)
   ↓
TB-004 (Quantization + KV Cache)
   ↓
TB-005 (Tokenizer + Sampler)
   ↓
TB-006 (SwiftUI Demo)
   ↓
TB-007 (Benchmarks + Release Prep)
   ↓
TB-008 (Clean Architecture)
   ↓
TB-009 (Format-Agnostic Tokenizer)
   ↓
TB-010 (X-Ray Mode)
```

---

## v0.2.0 (Future)

| Feature | Priority | Description |
|---------|----------|-------------|
| INT4 Quantization | P1 | Per-group INT4 for 8x memory savings |
| Core ML Hybrid | P2 | Optional ANE offload for attention |
| SentencePiece | P2 | Google model tokenizer support |
| TikToken | P2 | OpenAI model tokenizer support |
| More Models | P1 | Gemma 2B, Phi-2, Mistral 7B validation |
| FlashAttention | P2 | Fused Metal attention kernel |
| Multi-Model UI | P3 | Model picker in demo app |
| Energy Metrics | P2 | J/token measurement via MetricsKit |
| Swift Playgrounds | P3 | Interactive educational tutorials |

---

## Design Principles

### Vertical Slices, Not Horizontal Layers

Each task delivers something usable:

```
TB-002: Can multiply matrices
TB-003: On the GPU
TB-004: With quantized weights and cached attention
TB-005: With real tokenization and sampling
TB-006: In a SwiftUI app
TB-007: With benchmarks proving it works
TB-010: With visualizations showing HOW it works
```

### TDD Methodology

Every task follows Red-Green-Refactor:
1. Write failing tests that define requirements
2. Implement minimal code to pass
3. Refactor for clarity
4. Document for education
