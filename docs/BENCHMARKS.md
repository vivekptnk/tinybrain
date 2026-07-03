# TinyBrain Benchmarks

**Last updated:** 2026-07-03
**TinyBrain version:** v0.2.0-dev  
**Harness:** `tinybrain-bench --perplexity` (CHA-108)

---

## 1. Hardware Reference

All measurements on this page were taken on the following device unless otherwise noted.

| Field | Value |
|---|---|
| Machine | MacBook Pro (2024) |
| Chip | Apple M4 Max |
| CPU cores | 14 |
| GPU cores | 40 |
| Unified memory | 36 GB |
| OS | macOS 26.0.1 |
| Metal | Available |

> **CI note.** The perplexity harness runs as an XCTest guard in CI
> (`QualityRegressionTests.testTinyLlamaINT4VsINT8Perplexity`) but the
> 1.2 GB model file is gitignored. Results in this document are produced
> by running `tinybrain-bench --perplexity` locally against the full
> TinyLlama TBF.

---

## 2. Model

| Field | Value |
|---|---|
| Model | TinyLlama-1.1B-Chat-v1.0 |
| Parameters | 1.1 B |
| Layers | 22 transformer layers |
| Hidden dim | 2048 |
| Heads | 32 (GQA: 4 KV heads) |
| Vocab size | 32,000 |
| Source format | INT8 TBF (`tinyllama-1.1b-int8.tbf`, 808 MB) |

---

## 3. Perplexity — INT4 vs INT8

### 3.1 Dataset

Perplexity is measured on the pinned WikiText-2 slice `CHA-108-v1`:

| Field | Value |
|---|---|
| Source | WikiText-2 validation (`Salesforce/wikitext`, `wikitext-2-v1`) |
| Selection | Body paragraphs of the first 3 articles, blank-line separated |
| Slice length | 65 tokens (64 next-token predictions) |
| Pinned seed | `CHA-108-v1` |
| Fixture path | `Tests/TinyBrainRuntimeTests/Fixtures/wikitext2_slice.json` |

The slice is intentionally short: the scalar per-head attention loop in
`ModelRunner.attention` is O(N) per token step, so throughput degrades
sharply as the KV cache grows past ~100 positions. A FlashAttention Metal
kernel is implemented and covered by the GPU parity suite, but integration
into the inference attention path is tracked for v0.3.0.

### 3.2 Superseded Measurement (Methodology Note)

A 2026-04-24 proxy run previously appeared here as a current TinyLlama INT4
result. That run used an in-process double-quantization path
(`FP16 -> INT8 -> INT4`) at a different absolute-perplexity scale, so it was
retired on 2026-07-03. The shipped-artifact measurements in §3.4 are the
authoritative v0.2.0 regression tripwires.

### 3.4 v0.2.0 Tripwires and v0.2.1 Target

The v0.2.0 XCTest guards are shipped-artifact regression tripwires calibrated
from `.harness/scratch/dod-rerun2.log` on 2026-07-03 using artifact loading and
the head-at-INT8 policy:

- Gemma 2B: INT8 ppl 7.89913, INT4 ppl 8.58678, Δ +8.705%; enforced tripwire
  `|Δppl|/ppl_INT8 ≤ 0.11`.
- TinyLlama 1.1B: INT8 ppl 9.988422, INT4 ppl 11.313237, Δ +13.264%; enforced
  tripwire `|Δppl|/ppl_INT8 ≤ 0.17`.

The `|Δppl|/ppl_INT8 ≤ 0.06` quality bar is deferred to v0.2.1 via CHA-156
(GPTQ/AWQ calibration); the original 1% CHA-104 target remains a stretch goal
for calibrated quantization work.

---

## 4. Reproducing

Build in release mode (debug-mode dequant is ~20× slower):

```bash
cd tinybrain
swift build -c release
swift run -c release tinybrain-bench \
  --perplexity Models/tinyllama-1.1b-int8.tbf \
  --perplexity-group-size 32 \
  --perplexity-threshold 0.17 \
  --output json \
  --verbose
```

The `--output json` flag writes machine-readable output suitable for logging
to CI artifacts. Exit code 0 = within threshold; non-zero = regression.

To regenerate the pinned slice (requires `transformers`, `huggingface_hub`,
`pandas`, `pyarrow`):

```bash
python3 Scripts/pretokenize_wikitext.py
```

---

## 5. Roadmap

| Version | Target | Ticket |
|---|---|---|
| v0.2.0 | Shipped-artifact RTN tripwires: Gemma 2B ≤ 11 %, TinyLlama ≤ 17 % at group=32 | CHA-108 ✅ |
| v0.2.1 | `\|Δppl\|/ppl_INT8 ≤ 6 %` target (GPTQ/AWQ calibration); 1 % stretch | CHA-156 |
| v0.3.0 | FlashAttention Metal kernel integration into the inference attention path; kernel is implemented and GPU parity tested | TBD |
| Future | Extend slice to 512+ tokens once attention is on GPU | TBD |
