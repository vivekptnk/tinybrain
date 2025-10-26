# TinyBrain Benchmarking Guide

**Version:** 0.1.0  
**Last Updated:** October 25, 2025  
**Related:** `benchmarks/scenarios.yml`, `Scripts/run_benchmarks.sh`

---

## Overview

This guide explains how to benchmark TinyBrain's performance, interpret results, and compare across devices. The benchmark harness supports reproducible measurements via YAML scenarios and structured JSON/Markdown output.

---

## Quick Start

### Simple Benchmark

```bash
# Build release version (optimized)
swift build -c release

# Run 50-token generation
.build/release/tinybrain-bench --demo --tokens 50

# With JSON output
.build/release/tinybrain-bench \
  --demo \
  --tokens 50 \
  --output json > results.json
```

### Using Scenarios

```bash
# Run predefined scenarios
.build/release/tinybrain-bench \
  --scenario benchmarks/scenarios.yml \
  --output markdown

# Automated script (runs all scenarios, saves timestamped results)
./Scripts/run_benchmarks.sh
```

---

## Benchmark Harness Features

### CLI Flags

| Flag | Description | Example |
|------|-------------|---------|
| `--demo` | Run simple benchmark with toy model | `--demo --tokens 50` |
| `--scenario <path>` | Load YAML scenario file | `--scenario custom.yml` |
| `--output <format>` | Output format (json, markdown, or human-readable) | `--output json` |
| `--device-info` | Show system information | `--device-info` |
| `--warmup <n>` | Warmup iterations (default: 3) | `--warmup 5` |
| `--verbose` | Detailed output | `--verbose` |
| `--dry-run` | Validate scenario without running | `--dry-run` |

### Output Formats

**JSON (for CI/automation):**
```json
{
  "device": {
    "name": "M4-Max",
    "os": "macOS 15.0",
    "metalAvailable": true
  },
  "metrics": {
    "tokens_per_sec": 12.5,
    "ms_per_token": 80.0,
    "memory_peak_mb": 1100.0,
    "total_tokens": 50,
    "elapsed_seconds": 4.0
  },
  "timestamp": "2025-10-25T23:00:00Z"
}
```

**Markdown (for documentation):**
```markdown
| Metric | Value |
|--------|-------|
| Device | M4-Max |
| Tokens/sec | 12.50 |
| ms/token | 80.00 |
| Peak Memory (MB) | 1100.00 |
```

**Human-Readable (default):**
```
📊 Benchmark Results
==================================================
Device: M4-Max
Tokens generated: 50
Speed: 12.5 tokens/sec
Latency: 80.0 ms/token
Peak Memory: 1100.00 MB
```

---

## YAML Scenario Format

### Basic Structure

```yaml
scenarios:
  - name: "Descriptive scenario name"
    model: "Models/model.tbf"  # or "toy" for testing
    prompts:
      - "First prompt to test"
      - "Second prompt to test"
    max_tokens: 50
    backend: auto  # cpu, metal, or auto
    warmup: 3
    sampler:
      temperature: 0.7
      top_k: 40
      top_p: 0.9
      repetition_penalty: 1.1
```

### Example Scenarios

```yaml
scenarios:
  # Quick smoke test
  - name: "Smoke Test"
    model: "toy"
    prompts: ["Hello world"]
    max_tokens: 10
    backend: cpu

  # Production benchmark
  - name: "TinyLlama INT8 - Short Prompt"
    model: "Models/tinyllama-1.1b-int8.tbf"
    prompts:
      - "Explain quantum computing"
      - "Write a Python function"
    max_tokens: 50
    backend: auto
    sampler:
      temperature: 0.7
      top_k: 40

  # CPU vs Metal comparison
  - name: "Metal Performance Test"
    model: "toy"
    prompts: ["Benchmark test"]
    max_tokens: 100
    backend: metal  # Force Metal backend
```

---

## Metrics Explained

### Tokens Per Second

**Definition:** Throughput of token generation

**Formula:** `tokens_per_sec = total_tokens / elapsed_seconds`

**Interpretation:**
- Higher is better
- Target: > 6 tok/s for real-time chat
- M4 Max typical: 40-50 tok/s (toy model), 10-15 tok/s (real model)

### Milliseconds Per Token

**Definition:** Average latency per token

**Formula:** `ms_per_token = (elapsed_seconds * 1000) / total_tokens`

**Interpretation:**
- Lower is better
- Target: < 150 ms for responsive UX
- First token often 2-3× slower (prefill phase)

### Memory Peak (MB)

**Definition:** Maximum resident memory during inference

**Measurement:** Via `task_info` APIs on macOS/iOS

**Interpretation:**
- Model size + overhead
- INT8 TinyLlama: ~1.0-1.2 GB
- Toy model: ~15-20 MB

---

## Methodology

### 1. Warmup Phase

**Purpose:** Prime caches, stabilize JIT compilation

**Default:** 3 iterations (configurable via `--warmup`)

**Why it matters:**
- First run often 2-5× slower
- Metal shader compilation
- CPU cache warmup

### 2. Measurement Phase

**What we measure:**
- Wall-clock time (start → end)
- Peak memory usage (sampled per token)
- Token count (actual generated)

**What we exclude:**
- Warmup iterations
- Model loading time
- Scenario parsing

### 3. Statistical Considerations

**Single run vs Multiple runs:**
- Single scenario run: Quick but noisy
- Multiple prompts: Better average
- Recommended: 3+ prompts per scenario

**Variance sources:**
- Background processes
- Thermal state
- System load

**Best practice:**
```yaml
scenarios:
  - name: "Stable Benchmark"
    prompts:
      - "Prompt 1"
      - "Prompt 2"
      - "Prompt 3"  # Average across 3 runs
    max_tokens: 50
    warmup: 5  # Extra warmup for stability
```

---

## Comparing Results

### Across Devices

**Normalize by:**
- Tokens/sec (throughput)
- ms/token (latency)
- Memory efficiency (MB per billion parameters)

**Example comparison:**
| Device | Tokens/sec | ms/token | Memory |
|--------|------------|----------|--------|
| M4 Max | 12.5 | 80 | 1.1 GB |
| iPhone 16 Pro | 8.3 | 120 | 1.0 GB |
| iPad M2 | 10.1 | 99 | 1.05 GB |

### Across Models

**Factors to consider:**
- Model size (parameters)
- Quantization (FP32 vs INT8 vs INT4)
- Architecture (layers, hidden dim)

**Expected scaling:**
- 2× parameters → ~2× slower
- INT8 vs FP32 → ~4× memory reduction, minimal speed difference

### Across Backends

**CPU (Accelerate) vs Metal:**

```yaml
# Test both backends
scenarios:
  - name: "CPU Baseline"
    backend: cpu
    # ...
  
  - name: "Metal Accelerated"
    backend: metal
    # ...
```

**M4 Max findings:**
- Small matmul: CPU (AMX) competitive or faster
- Large matmul: Metal wins 1.3-2×
- Batched: Metal significant advantage

---

## Automation

### Continuous Benchmarking

**Script:** `Scripts/run_benchmarks.sh`

**Features:**
- Auto-detects device
- Runs all scenarios in `benchmarks/scenarios.yml`
- Generates timestamped JSON + Markdown
- Saves to `benchmarks/results/`

**Usage:**
```bash
# One-command benchmark suite
./Scripts/run_benchmarks.sh

# Output files:
# benchmarks/results/benchmark_<hostname>_<timestamp>.json
# benchmarks/results/benchmark_<hostname>_<timestamp>.md
```

### CI Integration

```yaml
# GitHub Actions example
- name: Run Smoke Benchmarks
  run: |
    swift build -c release
    .build/release/tinybrain-bench \
      --scenario benchmarks/ci-smoke.yml \
      --output json > benchmark-results.json
```

---

## Interpreting Results

### Meeting Targets

**PRD Success Criteria:**
- ✅ < 150 ms/token on iPhone 15 Pro class
- ✅ < 200 ms first token latency
- ✅ < 2 GB memory for 1-2B models

**How to check:**
```bash
# Run and check ms_per_token
.build/release/tinybrain-bench \
  --demo --tokens 50 --output json \
  | jq '.metrics.ms_per_token'

# Should be < 150.0
```

### Performance Regression

**Detect slowdowns:**
```bash
# Compare before/after
diff <(jq '.metrics' before.json) <(jq '.metrics' after.json)
```

**Red flags:**
- > 20% slowdown in tokens/sec
- > 50% increase in memory
- Metal slower than CPU (unexpected)

### Optimization Impact

**Measure improvements:**
1. Baseline: Run benchmark, save results
2. Optimize: Make code changes
3. Re-run: Same scenario, compare
4. Document: Store in `benchmarks/results/`

**Example:**
```
Buffer pool optimization:
- Before: 8.2 tokens/sec
- After: 12.5 tokens/sec
- Improvement: 52% faster ✅
```

---

## Troubleshooting

### Inconsistent Results

**Symptoms:** High variance between runs

**Solutions:**
- Increase warmup: `--warmup 10`
- Close background apps
- Check thermal state
- Use multiple prompts, average results

### Metal Not Used

**Check:**
```bash
.build/release/tinybrain-bench --device-info
```

**Should see:** `Metal: ✅ Available`

**If not:**
- Ensure macOS 14+/iOS 17+
- Check Apple Silicon (M-series)
- Verify GPU not disabled

### Out of Memory

**Symptoms:** Crash during benchmark

**Solutions:**
- Use INT8 instead of FP32
- Reduce max_tokens
- Check model file size
- Monitor with Activity Monitor

---

## Best Practices

**DO:**
- ✅ Use release builds (`-c release`)
- ✅ Run warmup iterations
- ✅ Test multiple prompts
- ✅ Document device info
- ✅ Save raw JSON results

**DON'T:**
- ❌ Benchmark debug builds
- ❌ Compare across different OS versions
- ❌ Ignore thermal state
- ❌ Skip warmup
- ❌ Cherry-pick best results

---

## Advanced Topics

### Custom Scenarios

**Create your own:**
```yaml
# custom-benchmark.yml
scenarios:
  - name: "My Use Case"
    model: "Models/my-model.tbf"
    prompts:
      - "Domain-specific prompt 1"
      - "Domain-specific prompt 2"
    max_tokens: 100
    sampler:
      temperature: 0.5  # Less random
      top_k: 20
```

**Run:**
```bash
.build/release/tinybrain-bench \
  --scenario custom-benchmark.yml \
  --output json
```

### Profiling

**Instruments (macOS):**
```bash
# Profile with Instruments
xcrun xctrace record \
  --template 'Time Profiler' \
  --launch .build/release/tinybrain-bench -- --demo --tokens 100
```

**Memory profiling:**
```bash
# Check for leaks
leaks --atExit -- .build/release/tinybrain-bench --demo --tokens 50
```

---

## References

- **Scenarios:** `benchmarks/scenarios.yml`
- **Results:** `benchmarks/baseline-*.md`
- **Automation:** `Scripts/run_benchmarks.sh`
- **Tests:** `Tests/TinyBrainBenchTests/`

---

**Questions?** See `docs/faq.md` or open an issue on GitHub.

