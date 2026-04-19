# ADR-001: Per-Channel INT8 Quantization

**Status:** Accepted
**Date:** 2025-10-25
**Task:** TB-004 (Phase 3)

## Context

TinyBrain targets on-device inference on iPhones and iPads. A TinyLlama 1.1B model in Float32 requires 4.4 GB of memory — too large for most mobile devices. We needed a quantization strategy that significantly reduces memory while preserving model quality.

Three quantization modes were considered:

1. **Per-tensor symmetric** — one scale factor for the entire tensor. Simple but loses per-channel variance.
2. **Per-channel symmetric** — one scale factor per output channel. Captures per-channel distributions.
3. **Per-group (e.g., GPTQ-style)** — scale per group of weights. Higher fidelity but more complex.

## Decision

We chose **per-channel symmetric INT8 quantization** as the default strategy.

Each output channel gets its own scale factor: `scale[c] = max(abs(channel[c])) / 127`. Weights are stored as `Int8` with a `[Float]` scale array per tensor (one entry per channel).

Dequantization at inference time: `float_value = int8_value * scale[channel]`.

## Consequences

**Positive:**

- **75% memory reduction** — TinyLlama drops from 4.4 GB to 1.1 GB, fitting comfortably on iPhones.
- **< 1% accuracy loss** — measured at 0.7–1.0% quantization error across random tensor roundtrips and matmul accuracy tests.
- **Simple implementation** — `QuantizedTensor` is 372 lines with clear semantics.
- **No external dependencies** — pure Swift implementation.

**Negative:**

- **CPU dequantization overhead** — weights are dequantized to Float32 during matmul. A Metal dequant kernel was deferred (CPU fallback is adequate for v0.1.0).
- **INT4 deferred** — INT4 would yield 87.5% savings but introduces 2–5% accuracy loss. Left for a future task.

## Evidence

- Quantization error: 0.7–1.0% (TB-004 completion report)
- Matmul accuracy: < 1% error on 128×256 × 256×512 and 10×768 × 768×3072 test matrices
- 11/11 quantization tests passing
- See `docs/TB-004-COMPLETE.md` for full benchmarks
