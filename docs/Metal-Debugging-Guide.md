# Metal Debugging Guide for TinyBrain

**How to debug GPU kernels and fix common issues**

---

## 🐛 Common Issues & Solutions

### Issue 1: "libraryLoadFailed" Error

**Symptom:**
```
Error: libraryLoadFailed
```

**Cause:** Metal shader library couldn't load

**Solutions:**
1. **For SPM:** We compile at runtime (automatic in TB-003)
2. **For Xcode:** Ensure .metal files are in target
3. **Check:** Metal is available (`MetalBackend.isAvailable`)

**Fix:**
```swift
// Our code handles this automatically with runtime compilation
if library == nil {
    library = try compileShaderLibrary()  // Compiles from source
}
```

---

### Issue 2: Wrong Results from Metal Kernel

**Symptom:** Metal output doesn't match CPU

**Debug steps:**

**1. Enable logging:**
```swift
TinyBrainBackend.debugLogging = true
// Will print which backend is used
```

**2. Test small matrices first:**
```swift
let a = Tensor([[1,2], [3,4]])  // Known values
let b = Tensor([[5,6], [7,8]])
let metal = try backend.matmul(a, b)
let cpu = a.matmul(b)

print("Metal:", metal)
print("CPU:", cpu)
// Manually verify which is correct
```

**3. Check kernel indexing:**
```metal
// Common bug: Wrong index calculation
C[row * N + col] = sum;  // ✅ Correct
C[row * M + col] = sum;  // ❌ Wrong! (M instead of N)
```

---

### Issue 3: Kernel Crashes or Hangs

**Symptom:** App freezes or crashes when running Metal

**Common causes:**

**1. Out of bounds access:**
```metal
// Always bounds check!
if (row >= M || col >= N) return;  // ✅

// Without check:
C[row * N + col] = sum;  // ❌ May access invalid memory
```

**2. Threadgroup memory too large:**
```metal
// Check: Total threadgroup memory < 32 KB
threadgroup float tileA[16 * 16];  // 1 KB ✅
threadgroup float tileB[16 * 16];  // 1 KB ✅
// Total: 2 KB < 32 KB ✅
```

**3. Missing barriers:**
```metal
// Load data
threadgroup_barrier(mem_flags::mem_threadgroup);  // ✅ Wait!
// Use data

// Without barrier:
// Some threads use data before it's loaded! ❌
```

---

### Issue 4: Metal Slower Than CPU

**Symptom:** Metal kernel is slower than Accelerate

**Causes & Fixes:**

**1. Matrix too small:**
```
Problem: GPU overhead dominates for < 512×512
Solution: Use CPU automatically (auto backend)
```

**2. Not using tiled kernel:**
```swift
// Slow:
backend.matmulNaive(a, b)  // ❌

// Fast:
backend.matmul(a, b)  // ✅ Uses tiled automatically
```

**3. Non-contiguous tensors:**
```swift
// Check if copy is happening:
print("Is contiguous:", a.isContiguous)

// Transposed tensors get copied:
let kt = k.transpose()  // Creates non-contiguous view
let result = q.matmul(kt)  // Copies kt to make contiguous
```

---

## 🔬 Xcode GPU Debugger

### Capture Metal Frame

**Steps:**
1. Run app in Xcode
2. Click camera icon in debug bar
3. Select "Metal" tab
4. See all GPU commands!

**What you can inspect:**
- Buffers (input/output data)
- Threadgroup sizes
- Kernel execution time
- Memory usage

### Read Buffer Contents

```swift
// In Xcode debugger, after kernel runs:
po backend.readBuffer(bufferC, count: 10)
// Prints first 10 elements
```

---

## ⚡ Performance Profiling

### Use Instruments

**Steps:**
1. Product → Profile (⌘I)
2. Choose "Metal System Trace"
3. Run your code
4. Analyze GPU usage

**What to look for:**
- **GPU Utilization:** Should be high (> 80%) for large matrices
- **Memory Bandwidth:** Check if memory-bound
- **Kernel Time:** Identify slow kernels

### Quick Benchmark

```swift
let start = Date()
for _ in 0..<100 {
    _ = try backend.matmul(a, b)
}
let elapsed = Date().timeIntervalSince(start)
print("Average: \(elapsed / 100 * 1000) ms")
```

---

## 🎯 Optimization Tips

### 1. Choose Right Threadgroup Size

**Good:**
- 16×16 = 256 threads (8 SIMD groups) ✅
- 32×32 = 1024 threads (max) ⚠️ Watch memory!

**Bad:**
- 15×15 = 225 (wastes threads)
- 17×17 = 289 (over SIMD boundary)

### 2. Minimize Data Transfer

**Slow:**
```swift
for token in tokens {
    let gpu = backend.matmul(a, b)  // Copy every time!
}
```

**Fast:**
```swift
// Keep data on GPU between operations
// (Future: TB-004 will add persistent GPU buffers)
```

### 3. Use Appropriate Kernel

**Small (< 512):** CPU (Accelerate)  
**Large (≥ 512):** Metal (Tiled)  
**Debugging:** Metal (Naive) for simplicity  

---

## 📋 Testing Checklist

Before deploying Metal code:

- [ ] Test on simulator (may not have Metal)
- [ ] Test on physical device
- [ ] Test various matrix sizes
- [ ] Compare vs CPU (numerical parity)
- [ ] Benchmark performance
- [ ] Test fallback when Metal unavailable
- [ ] Enable debug logging
- [ ] Profile with Instruments

---

## 🆘 When Things Go Wrong

### Metal Test Failing?

**1. Check availability:**
```swift
guard MetalBackend.isAvailable else {
    throw XCTSkip("Metal not available")
}
```

**2. Compare small known matrices:**
```swift
let a = Tensor([[1,2,3], [4,5,6]])
let b = Tensor([[7,8],[9,10],[11,12]])

let metal = try backend.matmul(a, b)
let cpu = a.matmul(b)

// Expected: [[58,64], [139,154]]
print(metal)
print(cpu)
```

**3. Test CPU path first:**
```swift
TinyBrainBackend.preferred = .cpu
let result = a.matmul(b)  // Should always work
```

---

## 💡 Pro Tips

### 1. Start Simple

Write naive kernel first, get it working, THEN optimize.

### 2. Test Everything

Every kernel change needs a test comparing vs CPU.

### 3. Profile Early

Don't guess - measure! Use Instruments to find real bottlenecks.

### 4. Keep CPU Fallback

Always have working CPU path. Metal can fail (simulators, old devices).

---

## 📚 Resources

**Apple Documentation:**
- [Metal Programming Guide](https://developer.apple.com/metal/)
- [Metal Shading Language Spec](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)
- [Metal Performance Shaders](https://developer.apple.com/documentation/metalperformanceshaders)

**TinyBrain Documentation:**
- `docs/TB-003-RESEARCH.md` - Metal fundamentals
- `docs/TB-003-IMPLEMENTATION-PLAN.md` - Implementation roadmap
- `Sources/TinyBrain/TinyBrain.docc/MetalAcceleration.md` - User guide
- `benchmarks/metal-vs-cpu.md` - Performance data

**Sample Code:**
- `Sources/TinyBrainMetal/Shaders/MatMul.metal` - Kernel implementation
- `Sources/TinyBrainMetal/MetalBackend.swift` - Swift wrapper
- `Tests/TinyBrainMetalTests/` - Working examples

---

## 🎓 Learn More

**Metal Concepts:**
1. Start with naive kernel (understand basics)
2. Add tiling (understand shared memory)
3. Profile (understand bottlenecks)
4. Optimize (tune tile sizes, reduce barriers)

**TinyBrain Path:**
- ✅ TB-003: Metal MatMul (you are here!)
- 📋 TB-004: Add more kernels if needed
- 📋 TB-005+: Advanced optimizations

---

**Happy debugging!** 🐛→✅

