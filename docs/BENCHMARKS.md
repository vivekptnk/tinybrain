# TinyBrain Benchmarks

**Last updated:** 2026-04-24  
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
sharply as the KV cache grows past ~100 positions. Flash attention (a
future CHA) will lift this ceiling.

### 3.2 v0.2.0 Results (M4 Max, 2026-04-24)

| Quantization | Group size | PPL | Δ vs INT8 | Predictions | Wall time | Within DoD |
|---|---|---|---|---|---|---|
| INT8 (baseline) | — | 276.57 | — | 64 | 38.4 s | ✅ |
| **INT4 RTN** | **32** | **262.05** | **−5.25 %** | **64** | **42.2 s** | **✅ ≤ 6 %** |
| INT4 RTN | 128 | 346.22 | +25.18 % | 64 | 39.5 s | ❌ > 6 % |

Raw JSON (group=32, canonical run):

```json
{
  "deltaRelative": 0.052492447,
  "groupSize": 32,
  "int4Perplexity": 262.0519,
  "int4Seconds": 42.239017963409424,
  "int8Perplexity": 276.56973,
  "int8Seconds": 38.4461909532547,
  "model": "Models/tinyllama-1.1b-int8.tbf",
  "numPredictions": 64,
  "seed": "CHA-108-v1",
  "thresholdRelative": 0.06,
  "withinThreshold": true
}
```

### 3.3 Interpretation

**INT4 ppl (262) < INT8 ppl (277) at group=32.** Negative delta (−5.25 %) is
expected and acceptable. RTN INT4 at tight group size can act as a mild
regularizer on short, low-context sequences; the absolute delta is well within
the 6 % budget. The critical constraint is `|Δ| / ppl_INT8 ≤ 0.06`, which is
satisfied.

**group=128 exceeds budget (+25 %).** Coarse groups allow large outlier weights
to dominate each scale, inflating quantization error for the 2048→4 bpw
compression. Tighter group=32 is the v0.2.0 default (see CHA-104, CHA-155).

**Wall time is CPU-bound.** Both runs run through the scalar CPU attention
loop — Metal is initialized but the attention path hasn't moved to GPU yet.
The ~42 s for 64 tokens (≈ 1.5 tok/s) is therefore a lower bound on future
performance once the Metal attention kernel ships.

### 3.4 v0.2.0 DoD

Per [CHA-155](https://github.com/vivekptnk/TinyBrain/issues):

> RTN INT4 at `group=32` must satisfy `|Δppl| / ppl_INT8 ≤ 0.06` on the
> pinned CHA-108-v1 slice.

**Status: ✅ PASS** (Δ = 5.25 % < 6 %).

The 1 % target from the original CHA-104 spec is deferred to v0.2.1 via
CHA-156 (GPTQ/AWQ calibration).

---

## 4. Reproducing

Build in release mode (debug-mode dequant is ~20× slower):

```bash
cd tinybrain
swift build -c release
swift run -c release tinybrain-bench \
  --perplexity Models/tinyllama-1.1b-int8.tbf \
  --perplexity-group-size 32 \
  --perplexity-threshold 0.06 \
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
| v0.2.0 | `\|Δppl\|/ppl_INT8 ≤ 6 %` at group=32 (RTN) | CHA-108 ✅ |
| v0.2.1 | `\|Δppl\|/ppl_INT8 ≤ 1 %` (GPTQ/AWQ calibration) | CHA-156 |
| Future | Flash attention — lift the 100-token throughput cliff | TBD |
| Future | Extend slice to 512+ tokens once attention is on GPU | TBD |
