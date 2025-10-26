# TinyBrain Baseline Benchmarks - iPhone 16 Pro

**Device:** iPhone 16 Pro  
**Chip:** A18 Pro  
**OS:** iOS 17+  
**Date:** October 25, 2025 (Pending)  
**TinyBrain Version:** 0.1.0 (TB-007)

---

## Model Information

**Model:** TinyLlama-1.1B-Chat-v1.0  
**Format:** TBF (TinyBrain Binary Format)  
**Quantization:** INT8 per-channel  
**File Size:** 808 MB  
**Parameters:** 1.1 billion

---

## Status

**⏳ Pending Physical Device Testing**

To run benchmarks on iPhone 16 Pro:

1. **Build for iOS:**
   ```bash
   # Open in Xcode
   open Package.swift
   
   # Select iPhone 16 Pro as target
   # Build and run ChatDemo or tinybrain-bench
   ```

2. **Run Benchmark Scenarios:**
   ```bash
   # On device via Xcode
   # Or install tinybrain-bench executable
   ```

3. **Collect Results:**
   - tokens/sec
   - ms/token
   - Memory usage
   - Thermal state
   - Battery impact

---

## Expected Performance

### TinyLlama 1.1B INT8 - Projections

Based on A18 Pro specifications:

| Metric | Conservative | Optimistic | Target |
|--------|-------------|------------|---------|
| First Token | 200-250 ms | 150-180 ms | < 200 ms |
| Subsequent | 120-150 ms | 80-100 ms | < 150 ms ✅ |
| Throughput | 6-8 tok/s | 10-12 tok/s | > 6 tok/s ✅ |
| Memory | 1.0-1.2 GB | 0.9-1.0 GB | < 2 GB ✅ |

**Assumptions:**
- A18 Pro ~80% of M4 Max single-core performance
- Neural Engine (ANE) acceleration possible
- Thermal throttling minimal for short bursts
- 8GB RAM sufficient for 808 MB model + overhead

---

## A18 Pro vs M4 Max Comparison

| Aspect | A18 Pro | M4 Max | Ratio |
|--------|---------|--------|-------|
| CPU Cores | 6 (2P+4E) | 16 (12P+4E) | 0.38× |
| GPU Cores | 6 | 40 | 0.15× |
| Memory Bandwidth | ~50 GB/s | ~400 GB/s | 0.13× |
| RAM | 8 GB | 128 GB | 0.06× |
| Neural Engine | 35 TOPS | N/A | ANE advantage |

**Performance Implications:**
- Single-thread: A18 Pro ~70-80% of M4 Max
- Batched GPU: M4 Max 6-7× faster
- ANE offload: A18 Pro potential advantage
- Memory: Both sufficient for 1.1B model

---

## Power & Thermal Considerations

### Battery Impact

**Expected draw:**
- Idle: ~100 mW baseline
- Inference: +500-800 mW
- Peak: ~1-1.2W total

**Runtime estimates:**
- 100 tokens @ 10 tok/s = 10 seconds
- Energy: ~10-12 joules
- Battery impact: Minimal (<0.1% per inference)

### Thermal State

**Monitor via `ProcessInfo`:**
```swift
let thermal = ProcessInfo.processInfo.thermalState
// Expect: .nominal or .fair for short bursts
// Throttling: Unlikely unless sustained >30s
```

---

## iOS-Specific Optimizations

### Neural Engine (ANE)

**Potential offload:**
- Matmul operations
- Softmax/LayerNorm
- Expected 2-3× speedup vs GPU

**Implementation:** Deferred to future (TB-008+)

### Metal Performance Shaders (MPS)

**Already implemented:** ✅
- Tiled MatMul kernel
- Buffer pooling
- GPU-resident tensors

### Background Execution

**Consideration:**
- iOS background limits: ~30s
- For long inference: request extended time
- Or move to foreground-only feature

---

## Benchmark Scenarios

**Planned tests:**

1. **Short Prompt (10 tokens)**
   - "What is AI?"
   - Expected: ~15-20ms/token

2. **Medium Prompt (50 tokens)**
   - "Explain machine learning"
   - Expected: ~80-100ms/token average

3. **Long Context (100+ tokens)**
   - Story generation
   - Monitor KV-cache performance

4. **Thermal Test**
   - 500 tokens continuous
   - Check for throttling

5. **Battery Test**
   - 10 inferences
   - Measure delta battery %

---

## Next Steps

1. **Deploy to iPhone 16 Pro**
   - Build ChatDemo app
   - Include tinybrain-bench CLI

2. **Run Automated Benchmarks**
   ```bash
   # On-device via Xcode
   .build/release/tinybrain-bench \
     --scenario benchmarks/scenarios.yml \
     --output json > iphone-results.json
   ```

3. **Manual Tests**
   - Interactive chat in ChatDemo
   - User experience (latency feel)
   - UI responsiveness

4. **Document Results**
   - Update this file with actual numbers
   - Compare to M4 Max
   - Assess target achievement

---

## Known Considerations

**TextField Issue:** ✅ Not applicable (iOS unaffected)  
**Metal Availability:** ✅ All iPhones have Metal  
**Xcode Scheme:** ✅ No sandbox needed on iOS  
**TBF Loading:** ⏳ Requires implementation first

---

**Status:** Ready for testing once TBF loader is implemented  
**Priority:** Medium (M4 Max baseline more critical for development)  
**ETA:** Post-TB-007 (v0.2.0 milestone)

