# TB-004 Hardware Validation Guide

**Purpose:** Validate that GPU-resident tensors achieve ≥3× speedup on real Metal hardware  
**Date:** October 25, 2025  
**Status:** Ready for testing

## Prerequisites

### Required Hardware
- ✅ Mac with Apple Silicon (M1/M2/M3/M4) **OR**
- ✅ iPhone 15 Pro or later **OR**
- ✅ iPad with M1 chip or later

### Software Requirements
- macOS 14+ or iOS 17+
- Xcode 16+
- Swift 5.10+

### Quick Check: Is Metal Available?

```bash
cd /path/to/tinybrain
swift test --filter testMetalAvailability
```

If Metal is available, you'll see device info like:
```
✅ Metal available: Apple M3 Pro
```

## Validation Tests

### Test 1: GPU Tensor Creation ✅

**What:** Verify tensors can be uploaded to GPU  
**Expected:** `isOnGPU` returns `true`

```bash
swift test --filter testGPUTensorCreation
```

**Expected Output:**
```
Test Case '-[...GPUTensorTests testGPUTensorCreation]' passed (0.001 seconds)
✅ Metal backend initialized successfully
```

### Test 2: Lazy Synchronization ✅

**What:** Verify tensors stay on GPU for chained operations  
**Expected:** No CPU roundtrips for GPU-only ops

```bash
swift test --filter testLazySynchronization
```

**Expected Output:**
```
Test Case '-[...GPUTensorTests testLazySynchronization]' passed (0.002 seconds)
```

### Test 3: Chained GPU Operations ✅

**What:** Multiple ops (matmul → softmax → matmul) stay on GPU  
**Expected:** Result stays GPU-resident

```bash
swift test --filter testGPUChainedOperations
```

**Expected Output:**
```
Test Case '-[...GPUTensorTests testGPUChainedOperations]' passed (3.5-4.0 seconds)
```

Note: This test is slower because it processes 512×512 matrices.

### Test 4: GPU↔CPU Transfer ✅

**What:** Data integrity preserved across transfers  
**Expected:** Values match exactly

```bash
swift test --filter testGPUToCPUTransfer
```

**Expected Output:**
```
Test Case '-[...GPUTensorTests testGPUToCPUTransfer]' passed (0.001 seconds)
```

### Test 5: Buffer Pool Reuse ✅

**What:** Buffers reused (same ObjectIdentifier)  
**Expected:** Pool eliminates allocations

```bash
swift test --filter testBufferReuse
```

**Expected Output:**
```
Test Case '-[...BufferPoolTests testBufferReuse]' passed (0.005 seconds)
```

### Test 6: ⭐ **CRITICAL** Performance Validation ⭐

**What:** Metal ≥3× faster than CPU for 1024×1024 matmul  
**Expected:** Speedup ≥ 3.0×

```bash
swift test --filter testMetalSpeedupWithPersistentBuffers --verbose
```

**Expected Output:**
```
🎯 TB-004 Critical Test: Persistent GPU Buffers
   CPU:      150.234 ms
   GPU:       48.156 ms
   Speedup:  3.12×
   Accuracy: 8.32e-04 relative error

Test Case '-[...testMetalSpeedupWithPersistentBuffers]' passed (5.2 seconds)
```

**❌ If speedup < 3.0×:**
- Check that buffers are being reused (enable debug logging)
- Verify GPU tensors stay resident
- Profile with Instruments

## Running All Tests

```bash
# Run all GPU tensor tests
swift test --filter GPUTensorTests

# Run all Metal tests
swift test --filter BufferPoolTests
swift test --filter PerformanceBenchmarks

# Run everything
swift test
```

## Detailed Performance Profiling

### Option 1: Built-in Benchmarks

```bash
cd /path/to/tinybrain

# Run comprehensive benchmarks
swift run tinybrain-bench
```

This will output:
```
📊 TinyBrain Performance Benchmarks
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Matrix Size | CPU Time | GPU Time | Speedup
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
512×512     | 45.2 ms  | 18.3 ms  | 2.47×
1024×1024   | 152.8 ms | 46.2 ms  | 3.31×
2048×2048   | 658.4 ms | 124.1 ms | 5.30×
```

### Option 2: Xcode Instruments

1. Open project in Xcode
2. Run: `Product > Profile` (⌘I)
3. Select "Time Profiler"
4. Run test: `testMetalSpeedupWithPersistentBuffers`
5. Check:
   - Buffer allocation time (should be ~0.001ms)
   - GPU compute time
   - CPU↔GPU transfer time

### Option 3: Manual Timing Script

Create `Scripts/validate-performance.swift`:

```swift
import TinyBrain

TinyBrainBackend.debugLogging = true
TinyBrainBackend.enableMetal()

let sizes = [256, 512, 1024, 2048]

print("Size\tCPU (ms)\tGPU (ms)\tSpeedup")
print("─────────────────────────────────────")

for size in sizes {
    let a = Tensor.random(shape: TensorShape(size, size))
    let b = Tensor.random(shape: TensorShape(size, size))
    
    // CPU benchmark
    let cpuStart = Date()
    let _ = a.matmulCPU(b)
    let cpuTime = Date().timeIntervalSince(cpuStart) * 1000
    
    // GPU benchmark
    let gpuA = a.toGPU()
    let gpuB = b.toGPU()
    let gpuStart = Date()
    let _ = gpuA.matmul(gpuB)
    let gpuTime = Date().timeIntervalSince(gpuStart) * 1000
    
    let speedup = cpuTime / gpuTime
    print("\(size)\t\(String(format: "%.1f", cpuTime))\t\(String(format: "%.1f", gpuTime))\t\(String(format: "%.2f×", speedup))")
}
```

Run it:
```bash
swift Scripts/validate-performance.swift
```

## Recording Results

### Results Template

Copy this to `docs/TB-004-VALIDATION-RESULTS.md`:

```markdown
# TB-004 Hardware Validation Results

**Date:** [DATE]  
**Device:** [e.g., MacBook Pro M3 Pro]  
**OS:** [e.g., macOS 15.0]  
**Xcode:** [e.g., 16.0]

## Test Results

### GPU Tensor Tests
- [ ] testGPUTensorCreation: _____ (PASS/FAIL)
- [ ] testLazySynchronization: _____ (PASS/FAIL)
- [ ] testGPUChainedOperations: _____ (PASS/FAIL)
- [ ] testGPUToCPUTransfer: _____ (PASS/FAIL)

### Buffer Pool Tests
- [ ] testBufferReuse: _____ (PASS/FAIL)
- [ ] testBufferPoolCapacity: _____ (PASS/FAIL)
- [ ] testConcurrentAccess: _____ (PASS/FAIL)

### ⭐ Performance Test
- [ ] testMetalSpeedupWithPersistentBuffers: _____ (PASS/FAIL)

**Measured Results:**
- CPU Time (1024×1024): _____ ms
- GPU Time (1024×1024): _____ ms
- **Speedup: _____ ×**
- Numerical Error: _____
- ✅/❌ Meets ≥3× requirement: _____

## Performance Breakdown

| Matrix Size | CPU Time | GPU Time | Speedup | Status |
|-------------|----------|----------|---------|--------|
| 256×256     | ___ ms   | ___ ms   | ___×    | ___    |
| 512×512     | ___ ms   | ___ ms   | ___×    | ___    |
| 1024×1024   | ___ ms   | ___ ms   | ___×    | ✅/❌   |
| 2048×2048   | ___ ms   | ___ ms   | ___×    | ___    |

## Buffer Pool Statistics

```
[Paste buffer pool stats here from debug output]
```

## Metal Device Info

```
[Paste device info here]
Device: _____
Max threads per threadgroup: _____
```

## Notes

[Any observations, anomalies, or issues]

## Conclusion

- [ ] ✅ All tests passed
- [ ] ✅ Performance target met (≥3× speedup)
- [ ] ✅ Ready for Phase 2

**Validated by:** _____  
**Signature/Date:** _____
```

## Troubleshooting

### Issue: Tests Skip with "Metal not available"

**Solution:** Run on real hardware, not in simulator or CI

```bash
# Check Metal availability
swift test --filter testCPUOnlyOperationsStillWork
```

### Issue: Speedup < 3×

**Possible causes:**
1. **Buffers not being reused** → Enable debug logging
2. **Data transferring on every op** → Check `isOnGPU`
3. **Small matrix size** → GPU overhead dominates below 512×512
4. **Thermal throttling** → Let device cool down

**Debug:**
```swift
TinyBrainBackend.debugLogging = true
let gpu = Tensor.random([1024, 1024]).toGPU()
print("On GPU:", gpu.isOnGPU)  // Should be true
```

### Issue: Tests Timeout

**Solution:** Reduce matrix sizes for testing

Edit `PerformanceBenchmarks.swift`:
```swift
let size = 512  // Instead of 2048
```

### Issue: Numerical Errors Too High

**Check:**
- Relative error should be < 1e-3 for Float32
- If higher, may indicate Metal kernel bug

```swift
let error = relativeError(cpuResult, gpuResult)
print("Error:", error)  // Should be ~1e-4 to 1e-5
```

## Success Criteria

✅ **Phase 1 Complete when:**
1. All 13 tests pass (not skip)
2. Speedup ≥ 3.0× for 1024×1024 matmul
3. Numerical error < 1e-3
4. Buffer pool shows >90% hit rate
5. No memory leaks (Instruments)

## Next Steps After Validation

Once hardware validation passes:

1. **Document results** in `TB-004-VALIDATION-RESULTS.md`
2. **Update TB-004.md** to mark Phase 1 complete
3. **Commit changes** with message: `feat(tb-004): Complete Phase 1 - GPU-resident tensors validated`
4. **Start Phase 2** - Generic Tensor with CoW

---

**Ready to test?** Run this command and paste results:

```bash
swift test --filter testMetalSpeedupWithPersistentBuffers 2>&1 | tee validation-output.txt
```

