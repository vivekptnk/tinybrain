# TB-003 Implementation Plan

**Detailed roadmap for Metal GPU acceleration**

---

## 🎯 Goal

Implement GPU-accelerated matrix multiplication using Metal, achieving 3-10× speedup over CPU baseline while maintaining educational clarity.

---

## 📋 Task Breakdown (20 Tasks)

### Phase 1: Stride-Aware Tensors (Tasks 1-4) - Foundation

**Duration:** 2-3 days  
**Goal:** Enable transpose/reshape without data copies

#### Task 1: Add Stride to TensorShape ⏱️ 2 hours
```swift
public struct TensorShape {
    public let dimensions: [Int]
    public let strides: [Int]  // NEW!
    
    // stride[i] = product of all dimensions after i
    // Example: [2, 3, 4] → strides = [12, 4, 1]
}
```

**Why:** Metal works better with flexible memory layouts

#### Task 2: Implement transpose() ⏱️ 3 hours
```swift
func transpose() -> Tensor {
    // Just swap dimensions and strides - no data copy!
    // [2, 3] with strides [3, 1]
    // → [3, 2] with strides [1, 3]
}
```

**Test:**
```swift
func testTranspose() {
    let a = Tensor([[1,2,3], [4,5,6]])  // [2,3]
    let b = a.transpose()                // [3,2]
    
    XCTAssertEqual(b[0,0], 1)  // Same as a[0,0]
    XCTAssertEqual(b[1,0], 2)  // Same as a[0,1]
    // No data was copied! ✅
}
```

#### Task 3: Implement reshape() ⏱️ 2 hours
```swift
func reshape(_ newShape: TensorShape) -> Tensor {
    precondition(newShape.count == self.shape.count)
    // Return new view with different shape
}
```

#### Task 4: Tests for Transpose/Reshape ⏱️ 2 hours
- Test transpose 2D, 3D
- Test reshape various dimensions
- Test that no copying occurs (check data pointer)

**Milestone 1:** ✅ Stride-aware tensors working

---

### Phase 2: Metal Infrastructure (Tasks 5-6) - Setup

**Duration:** 2-3 days  
**Goal:** Get Metal pipeline working

#### Task 5: Enhance MetalBackend ⏱️ 4 hours
```swift
public final class MetalBackend {
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let library: MTLLibrary  // NEW!
    
    // Compiled pipelines (cache these!)
    private var pipelines: [String: MTLComputePipelineState] = [:]
    
    func loadKernel(name: String) throws -> MTLComputePipelineState {
        // Load and compile .metal shader
    }
}
```

**Why:** Need to load and compile our `.metal` files

#### Task 6: Buffer Management ⏱️ 4 hours
```swift
extension MetalBackend {
    func createBuffer(from tensor: Tensor) -> MTLBuffer {
        // Create GPU buffer from tensor data
    }
    
    func readBuffer(_ buffer: MTLBuffer, shape: TensorShape) -> Tensor {
        // Read GPU results back to CPU
    }
    
    // Buffer pool for reuse (avoid allocations)
    class BufferPool {
        func acquire(size: Int) -> MTLBuffer
        func release(_ buffer: MTLBuffer)
    }
}
```

**Test:** Create buffer, write data, read back, verify correctness

**Milestone 2:** ✅ Metal pipeline functional

---

### Phase 3: Naive MatMul Kernel (Tasks 7-9) - Get It Working

**Duration:** 2-3 days  
**Goal:** First working GPU matmul

#### Task 7: RED - Write Failing Test ⏱️ 1 hour
```swift
func testMetalMatMulBasic() {
    guard MetalBackend.isAvailable else { return }
    
    let a = Tensor([[1,2,3], [4,5,6]])
    let b = Tensor([[7,8], [9,10], [11,12]])
    
    let metalBackend = try MetalBackend()
    let c = metalBackend.matmul(a, b)  // Doesn't exist yet!
    
    // Expected: [[58,64], [139,154]]
    XCTAssertEqual(c[0,0], 58, accuracy: 1e-3)
}
```

**This will FAIL** - that's the point! (RED phase)

#### Task 8: GREEN - Naive Metal Kernel ⏱️ 4 hours

**MatMul.metal:**
```metal
kernel void matmul_naive(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint3& dims [[buffer(3)]],  // M, N, K
    uint2 gid [[thread_position_in_grid]]
) {
    uint M = dims.x, N = dims.y, K = dims.z;
    uint row = gid.y, col = gid.x;
    
    if (row >= M || col >= N) return;
    
    float sum = 0.0;
    for (uint k = 0; k < K; k++) {
        sum += A[row * K + k] * B[k * N + col];
    }
    
    C[row * N + col] = sum;
}
```

**Swift wrapper:**
```swift
func matmul(_ a: Tensor, _ b: Tensor) -> Tensor {
    // Create buffers
    // Encode kernel
    // Dispatch
    // Read results
}
```

**This makes test PASS** (GREEN phase)

#### Task 9: Test Numerical Parity ⏱️ 2 hours
```swift
func testMetalVsCPUParity() {
    let a = Tensor.random(shape: TensorShape(64, 64))
    let b = Tensor.random(shape: TensorShape(64, 64))
    
    let cpuResult = a.matmul(b)  // Current CPU version
    let metalResult = metalBackend.matmul(a, b)
    
    // Should match within tolerance
    for i in 0..<64 {
        for j in 0..<64 {
            XCTAssertEqual(metalResult[i,j], cpuResult[i,j], 
                          accuracy: 1e-3)
        }
    }
}
```

**Milestone 3:** ✅ Working naive GPU matmul (will be slow)

---

### Phase 4: Tiled Optimization (Tasks 10-11) - Make It Fast!

**Duration:** 3-4 days  
**Goal:** 5-10× faster than naive

#### Task 10: Tiled Kernel with Threadgroup Memory ⏱️ 6 hours

**The optimization:**
```metal
kernel void matmul_tiled(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint3& dims [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    threadgroup float* tileA [[threadgroup(0)]],  // Shared memory!
    threadgroup float* tileB [[threadgroup(1)]]
) {
    const uint TILE_SIZE = 16;
    
    // Load tiles into threadgroup memory
    // Synchronize threads
    // Compute using fast shared memory
    // Repeat for all tiles
}
```

**Why this is faster:**
- Global memory: ~100 cycles latency
- Threadgroup memory: ~5 cycles latency
- **20× faster memory access!**

#### Task 11: Benchmark & Tune ⏱️ 3 hours

**Compare:**
```
Naive GPU:  Still slower than CPU (memory bound)
Tiled GPU:  3-10× faster than CPU ✅
```

**Test different tile sizes:**
- 8×8: Underutilizes GPU
- 16×16: Sweet spot ✅
- 32×32: Too large, register pressure

**Milestone 4:** ✅ Fast tiled matmul (production-ready)

---

### Phase 5: Backend Abstraction (Tasks 12-14) - Make It Transparent

**Duration:** 2-3 days  
**Goal:** Automatic CPU/GPU selection

#### Task 12: Backend Protocol ⏱️ 3 hours
```swift
enum ComputeBackend {
    case cpu
    case metal(MetalBackend)
}

protocol TensorOperation {
    func execute(with backend: ComputeBackend) -> Tensor
}
```

#### Task 13: Auto-Selection in Tensor.matmul() ⏱️ 3 hours
```swift
// Global backend (set at startup)
private static var preferredBackend: ComputeBackend = .cpu

public func matmul(_ other: Tensor) -> Tensor {
    switch Self.preferredBackend {
    case .cpu:
        return matmulCPU(other)  // Our Accelerate version
    case .metal(let backend):
        return backend.matmul(self, other)  // Metal version
    }
}
```

**User doesn't see any Metal code:**
```swift
// Just works!
let c = a.matmul(b)  
// Uses Metal if available, CPU otherwise ✅
```

#### Task 14: Logging & Debug Mode ⏱️ 2 hours
```swift
if TinyBrain.debugLogging {
    print("Using Metal backend for matmul (\(a.shape) × \(b.shape))")
}
```

**Milestone 5:** ✅ Transparent backend switching

---

### Phase 6: Comprehensive Testing (Tasks 15-17) - Validation

**Duration:** 2-3 days  
**Goal:** Bulletproof before Review Hitler sees it! 😄

#### Task 15: Numerical Parity Tests ⏱️ 4 hours

**Test matrix:**
```swift
// Small matrices
testMetalParity(2, 3, 2)    // 2×3 by 3×2
testMetalParity(10, 10, 10)

// Medium matrices  
testMetalParity(64, 64, 64)
testMetalParity(128, 128, 128)

// Large matrices
testMetalParity(512, 512, 512)
testMetalParity(1024, 1024, 1024)

// Non-square
testMetalParity(100, 50, 200)

// Edge cases
testMetalParity(1, 1, 1)      // Tiny
testMetalParity(17, 17, 17)   // Not multiple of 16
```

#### Task 16: Performance Benchmarks ⏱️ 3 hours

**Measure speedup:**
```swift
// Target: Metal 3-10× faster than CPU

// 512×512
CPU:   0.776 ms (TB-002 baseline)
Metal: 0.08-0.25 ms (target)
Speedup: 3-10× ✅

// 2048×2048
CPU:   ~100 ms
Metal: 10-20 ms
Speedup: 5-10× ✅
```

#### Task 17: Create Benchmark Report ⏱️ 2 hours

**Generate:** `benchmarks/metal-vs-cpu.md`

```markdown
# Metal vs CPU Performance

## MatMul Benchmarks (Apple M4 Max)

| Size | CPU (ms) | Metal (ms) | Speedup |
|------|----------|------------|---------|
| 512×512 | 0.776 | 0.150 | 5.2× |
| 1024×1024 | 6.5 | 0.8 | 8.1× |
| 2048×2048 | 52.0 | 7.2 | 7.2× |
```

**Milestone 6:** ✅ Validated and benchmarked

---

### Phase 7: Documentation (Tasks 18-20) - Educational Mission

**Duration:** 2-3 days  
**Goal:** Teach GPU programming!

#### Task 18: DocC Metal Article ⏱️ 4 hours

**Create:** `MetalAcceleration.md`

Topics:
- How GPU matmul works
- Tiling strategy explained
- Threadgroup memory benefits
- When to use Metal vs CPU

#### Task 19: Update docs/overview.md ⏱️ 2 hours

Add section 5: Metal Backend
- Architecture diagram
- Buffer flow
- Performance characteristics

#### Task 20: Metal Debugging Guide ⏱️ 3 hours

**Create:** `docs/Metal-Debugging.md`

- Xcode GPU debugger
- Metal frame capture
- Common shader errors
- Performance profiling

**Milestone 7:** ✅ Complete documentation

---

## 📅 Timeline Estimate

### Conservative (Take our time, avoid Review Hitler):
```
Week 1: Phase 1-2 (Strides + Metal setup)
Week 2: Phase 3-4 (Naive → Tiled kernel)  
Week 3: Phase 5-6 (Backend abstraction + tests)
Week 4: Phase 7 (Documentation)

Total: 4 weeks
```

### Aggressive (If we're on a roll):
```
Week 1: Phase 1-3 (Strides + Naive kernel)
Week 2: Phase 4-5 (Tiled + Backend)
Week 3: Phase 6-7 (Tests + Docs)

Total: 3 weeks
```

**Recommendation:** Conservative approach (4 weeks) - quality over speed!

---

## 🎯 Success Criteria Checklist

Before asking Review Hitler to look at TB-003:

### ✅ Functionality
- [ ] MatMul Metal kernel compiles
- [ ] Runs on macOS and iOS simulator
- [ ] Automatic CPU/GPU fallback works
- [ ] Transpose/reshape implemented

### ✅ Performance
- [ ] 512×512: ≥3× faster than CPU
- [ ] 2048×2048: ≥3× faster than CPU
- [ ] Threadgroup memory utilized
- [ ] 16×16 tiling implemented

### ✅ Testing
- [ ] Numerical parity tests (< 1e-3 error)
- [ ] Various matrix sizes tested
- [ ] Edge cases covered
- [ ] Performance benchmarks run

### ✅ Documentation
- [ ] DocC article on Metal
- [ ] overview.md updated
- [ ] Debugging guide created
- [ ] Code comments educational

### ✅ Code Quality
- [ ] TDD methodology followed
- [ ] No warnings
- [ ] Lint passes
- [ ] Tests all green

---

## 🔄 TDD Cycles

### Cycle 1: Stride-Aware Transpose
1. 🔴 RED: Write test for transpose
2. 🟢 GREEN: Implement stride manipulation
3. ♻️ REFACTOR: Add docs

### Cycle 2: Naive Metal MatMul
1. 🔴 RED: Write test expecting Metal backend
2. 🟢 GREEN: Implement naive kernel + Swift wrapper
3. ♻️ REFACTOR: Clean up, document

### Cycle 3: Tiled Optimization
1. 🔴 RED: Write performance test (must be 3× faster)
2. 🟢 GREEN: Implement tiled kernel
3. ♻️ REFACTOR: Tune tile size, optimize

### Cycle 4: Backend Abstraction
1. 🔴 RED: Test that matmul() auto-selects Metal
2. 🟢 GREEN: Implement backend enum + selection
3. ♻️ REFACTOR: Add logging, cleanup

**Every phase uses TDD!**

---

## ⚠️ Risk Mitigation

### Risk 1: Metal Kernel Bugs Hard to Debug

**Mitigation:**
- Start with naive version (simple)
- Test on CPU first (validate logic)
- Use Xcode Metal debugger
- Print intermediate values

### Risk 2: Performance Not Meeting Target

**Mitigation:**
- Benchmark early and often
- Profile with Instruments
- Start with known-good tile size (16×16)
- Have CPU fallback always working

### Risk 3: Scope Creep

**Mitigation:**
- Stick to MatMul only
- Defer other kernels (documented in TB-004)
- Review Hitler will catch if we over-promise! 😄

### Risk 4: Stride Implementation Breaks Tests

**Mitigation:**
- Implement strides incrementally
- Keep contiguous path working
- Test both contiguous and strided cases

---

## 🎓 Learning Objectives

By completing TB-003, you'll understand:

1. **GPU Architecture**
   - SIMD groups, threadgroups
   - Memory hierarchy (global vs threadgroup)
   - Occupancy and parallelism

2. **Metal Programming**
   - Shader syntax (.metal)
   - Buffer management
   - Command encoding

3. **Optimization Techniques**
   - Tiling for cache efficiency
   - Threadgroup memory usage
   - Work distribution strategies

4. **Backend Abstraction**
   - Protocol-oriented design
   - Runtime dispatch
   - Graceful fallbacks

**This knowledge is valuable beyond TinyBrain!**

---

## 📊 Expected Results

### Performance (Apple M4 Max)

**Before TB-003 (CPU only):**
```
512×512:   0.776 ms
1024×1024: ~6 ms
2048×2048: ~50 ms
```

**After TB-003 (Metal):**
```
512×512:   0.10-0.25 ms  (3-8× faster)
1024×1024: 0.8-1.5 ms   (4-8× faster)
2048×2048: 7-15 ms      (3-7× faster)
```

### Impact on Real Models

**Toy model (TB-002):**
```
Before: 1049 tokens/sec
After:  3000-5000 tokens/sec (estimated)
```

**Real TinyLlama 1.1B:**
```
Before: ~5-10 tokens/sec (CPU)
After:  ~20-40 tokens/sec (Metal) ✅ Feels responsive!
```

---

## 🎯 Phase-by-Phase Deliverables

### After Phase 1-2 (Week 1):
- ✅ Strides implemented
- ✅ Transpose working
- ✅ Metal infrastructure ready

### After Phase 3 (Week 2):
- ✅ Naive GPU matmul working
- ✅ Slower than CPU but correct!

### After Phase 4 (Week 2-3):
- ✅ Tiled matmul (fast!)
- ✅ 3-10× faster than CPU ✅

### After Phase 5-6 (Week 3):
- ✅ Backend abstraction
- ✅ All tests passing
- ✅ Benchmarks documented

### After Phase 7 (Week 4):
- ✅ Full documentation
- ✅ Ready for Review Hitler!

---

## 📈 How to Track Progress

**Use:** This todo list + git commits

**Check daily:**
1. How many tasks completed?
2. Are tests passing?
3. Any blockers?

**Weekly review:**
1. Are we on schedule?
2. Do we need to adjust scope?
3. Is quality high enough?

---

## ✅ Ready to Start?

**Next steps:**
1. Mark task tb3-1 as "in_progress"
2. Start with RED test for strides
3. Implement, test, iterate!

**Or:** Want to discuss any part of the plan first?

**Total:** 20 tasks, 4 weeks, focused scope, TDD methodology

Let's build some GPU kernels! 🚀

