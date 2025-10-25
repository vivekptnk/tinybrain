# TB-003 Research: Metal GPU Kernels for TinyBrain

**Comprehensive guide to implementing GPU acceleration**

---

## 📚 What Is Metal?

**Metal** is Apple's low-level GPU programming framework. Think of it as:
- **Direct access to the GPU** (like CUDA for NVIDIA, but for Apple)
- **Write shaders** in Metal Shading Language (MSL) - C++-like syntax
- **10-100× faster** than CPU for parallel operations

**Why Metal for TinyBrain?**
- Available on all Apple devices (iPhone, iPad, Mac)
- Zero dependencies (built into iOS/macOS)
- Educational (we can see exactly what the GPU does)
- Fast (optimized for Apple Silicon)

---

## 🎯 Two Approaches for TB-003

### Approach A: Use Metal Performance Shaders (MPS) ⚡ EASY

**What:** Apple's pre-built, highly optimized GPU kernels

**Pros:**
- ✅ Already optimized by Apple engineers
- ✅ Handles edge cases
- ✅ Less code to write/maintain
- ✅ Automatic device-specific tuning

**Cons:**
- ❌ Black box (not educational)
- ❌ Less control over details
- ❌ Might not match our exact needs

**Example:**
```swift
import MetalPerformanceShaders

let matmul = MPSMatrixMultiplication(
    device: device,
    resultRows: M,
    resultColumns: N,
    interiorColumns: K
)

matmul.encode(to: commandBuffer, 
              leftMatrix: A,
              rightMatrix: B,
              resultMatrix: C)
```

---

### Approach B: Write Custom Metal Kernels 🎓 EDUCATIONAL

**What:** Write our own `.metal` shader files

**Pros:**
- ✅ **Educational** - See exactly how GPU matmul works!
- ✅ **Transparent** - Aligns with TinyBrain's mission
- ✅ **Customizable** - Optimize for our specific use case
- ✅ **Learning** - Understand GPU programming deeply

**Cons:**
- ❌ More code to write
- ❌ Need to tune for different devices
- ❌ Might be slower than MPS initially

**Example:**
```metal
// MatMul.metal
kernel void matmul_kernel(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint2& dims [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    // Our custom implementation
}
```

---

### 🏆 RECOMMENDED: Hybrid Approach

**Best of both worlds:**

1. **Start with custom kernels** (educational)
   - MatMul (most important - we write this!)
   - Softmax (custom)
   - LayerNorm (custom)

2. **Use MPS where it makes sense** (pragmatic)
   - Complex ops (convolutions if needed)
   - Fallback for unsupported devices

3. **Document both approaches** (educational value)

**Why this is best:**
- ✅ Educational (write critical kernels ourselves)
- ✅ Practical (use MPS when appropriate)
- ✅ Valuable (learn GPU programming + ship fast)

---

## 🧮 Metal Basics (What You Need to Know)

### The Metal Pipeline

```
CPU (Swift)
  ↓
1. Create MTLDevice (the GPU)
  ↓
2. Create MTLCommandQueue (work queue)
  ↓
3. Create MTLBuffer (GPU memory for data)
  ↓
4. Create MTLComputePipelineState (compiled shader)
  ↓
5. Create MTLCommandBuffer (batch of work)
  ↓
6. Create MTLComputeCommandEncoder (encode commands)
  ↓
7. Set buffers, threadgroups, dispatch
  ↓
8. Commit command buffer
  ↓
GPU executes kernel
  ↓
9. Wait for completion or use callback
  ↓
10. Read results from buffer
```

### Key Concepts

**MTLDevice** - The GPU
```swift
let device = MTLCreateSystemDefaultDevice()!
// On M4 Max: "Apple M4 Max"
```

**MTLBuffer** - GPU Memory
```swift
let buffer = device.makeBuffer(
    bytes: data,
    length: data.count * MemoryLayout<Float>.stride,
    options: .storageModeShared  // CPU & GPU can both access
)
```

**MTLComputePipelineState** - Compiled Shader
```swift
let library = device.makeDefaultLibrary()!
let function = library.makeFunction(name: "matmul_kernel")!
let pipeline = try device.makeComputePipelineState(function: function)
```

**Threadgroups** - How Work Is Divided
```
Total work: 1024×1024 matrix = 1,048,576 elements

Threadgroup size: 16×16 = 256 threads
Number of threadgroups: (1024/16) × (1024/16) = 64×64 = 4,096 groups

Each threadgroup processes: 16×16 = 256 elements
Total threads: 4,096 groups × 256 threads = 1,048,576 ✅
```

---

## ⚙️ Metal Shading Language (MSL) Basics

MSL is like C++ with GPU-specific features:

### Simple Kernel Example

```metal
#include <metal_stdlib>
using namespace metal;

// Element-wise addition kernel
kernel void vector_add(
    device const float* inputA [[buffer(0)]],  // Input buffer A
    device const float* inputB [[buffer(1)]],  // Input buffer B
    device float* output [[buffer(2)]],        // Output buffer
    uint id [[thread_position_in_grid]]        // Thread ID
) {
    output[id] = inputA[id] + inputB[id];
}
```

**Key Points:**
- `kernel` = GPU function (entry point)
- `device` = GPU memory space
- `[[buffer(N)]]` = Buffer binding point
- `[[thread_position_in_grid]]` = Which thread am I?

### Thread Identifiers

```metal
uint id [[thread_position_in_grid]]           // Global thread ID
uint2 gid [[thread_position_in_grid]]         // 2D thread ID
uint tid [[thread_position_in_threadgroup]]   // Local thread ID within group
uint2 tgid [[threadgroup_position_in_grid]]   // Which threadgroup?
```

---

## 🧮 Matrix Multiplication Kernel (The Big One!)

### Naive Approach (Slow)

```metal
kernel void matmul_naive(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint3& dims [[buffer(3)]],  // M, N, K
    uint2 gid [[thread_position_in_grid]]
) {
    uint M = dims.x;
    uint N = dims.y;
    uint K = dims.z;
    
    uint row = gid.y;
    uint col = gid.x;
    
    if (row >= M || col >= N) return;
    
    float sum = 0.0;
    for (uint k = 0; k < K; k++) {
        sum += A[row * K + k] * B[k * N + col];
    }
    
    C[row * N + col] = sum;
}
```

**Why it's slow:**
- Each thread reads from **global memory** K times
- Global memory is SLOW (~100 cycles latency)
- No reuse between threads

---

### Optimized Approach (Tiled with Shared Memory)

**The trick:** Load tiles into **threadgroup memory** (fast, shared between threads in a group)

```metal
kernel void matmul_tiled(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint3& dims [[buffer(3)]],  // M, N, K
    uint2 gid [[thread_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    threadgroup float* tileA [[threadgroup(0)]],  // Shared memory for A
    threadgroup float* tileB [[threadgroup(1)]]   // Shared memory for B
) {
    uint M = dims.x;
    uint N = dims.y;
    uint K = dims.z;
    
    const uint TILE_SIZE = 16;
    
    uint row = gid.y;
    uint col = gid.x;
    
    float sum = 0.0;
    
    // Loop over tiles
    for (uint t = 0; t < K; t += TILE_SIZE) {
        // Load tile of A into shared memory
        if (row < M && (t + tid.x) < K) {
            tileA[tid.y * TILE_SIZE + tid.x] = A[row * K + (t + tid.x)];
        } else {
            tileA[tid.y * TILE_SIZE + tid.x] = 0.0;
        }
        
        // Load tile of B into shared memory
        if ((t + tid.y) < K && col < N) {
            tileB[tid.y * TILE_SIZE + tid.x] = B[(t + tid.y) * N + col];
        } else {
            tileB[tid.y * TILE_SIZE + tid.x] = 0.0;
        }
        
        // Synchronize threads in group
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Compute using shared memory (FAST!)
        for (uint k = 0; k < TILE_SIZE; k++) {
            sum += tileA[tid.y * TILE_SIZE + k] * tileB[k * TILE_SIZE + tid.x];
        }
        
        // Synchronize before loading next tile
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Write result
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
```

**Why it's fast:**
- Loads data into **threadgroup memory** (32 KB, <10 cycle latency)
- Each tile loaded once, used by 256 threads
- **10-20× faster** than naive approach!

---

## 🎮 Threadgroup Sizes & Apple Silicon

### Apple GPU Architecture

**Apple Silicon GPUs** (M1/M2/M3/M4):
- **32 threads per SIMD group** (execute together)
- **1024 max threads per threadgroup**
- **32 KB threadgroup memory**

**Optimal threadgroup sizes:**
- Multiples of 32 (to fill SIMD groups)
- Common: 16×16 = 256 threads (good balance)
- For matmul: 16×16 tiles work well

**Bad sizes:**
- 15×15 = 225 (wastes 7 threads per SIMD group)
- 17×17 = 289 (slightly over, inefficient)
- 32×32 = 1024 (max, might run out of threadgroup memory)

---

## 🏗️ Swift Integration Pattern

### Full Example: Vector Addition

**Metal Shader (VectorAdd.metal):**
```metal
#include <metal_stdlib>
using namespace metal;

kernel void vector_add(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* result [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    result[id] = a[id] + b[id];
}
```

**Swift Wrapper:**
```swift
import Metal

class MetalVectorAdd {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    let pipeline: MTLComputePipelineState
    
    init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MetalError.deviceNotFound
        }
        self.device = device
        
        guard let queue = device.makeCommandQueue() else {
            throw MetalError.queueCreationFailed
        }
        self.commandQueue = queue
        
        // Load shader
        let library = device.makeDefaultLibrary()!
        let function = library.makeFunction(name: "vector_add")!
        self.pipeline = try device.makeComputePipelineState(function: function)
    }
    
    func add(_ a: [Float], _ b: [Float]) -> [Float] {
        let count = a.count
        
        // Create GPU buffers
        let bufferA = device.makeBuffer(bytes: a, 
                                        length: count * MemoryLayout<Float>.stride)!
        let bufferB = device.makeBuffer(bytes: b,
                                        length: count * MemoryLayout<Float>.stride)!
        let bufferC = device.makeBuffer(length: count * MemoryLayout<Float>.stride)!
        
        // Create command buffer
        let commandBuffer = commandQueue.makeCommandBuffer()!
        let encoder = commandBuffer.makeComputeCommandEncoder()!
        
        // Set pipeline and buffers
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(bufferA, offset: 0, index: 0)
        encoder.setBuffer(bufferB, offset: 0, index: 1)
        encoder.setBuffer(bufferC, offset: 0, index: 2)
        
        // Calculate threadgroup sizes
        let threadsPerGroup = MTLSize(width: 256, height: 1, depth: 1)
        let numThreadgroups = MTLSize(width: (count + 255) / 256, height: 1, depth: 1)
        
        // Dispatch
        encoder.dispatchThreadgroups(numThreadgroups, threadsPerThreadgroup: threadsPerGroup)
        encoder.endEncoding()
        
        // Execute
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        // Read results
        let resultPointer = bufferC.contents().bindMemory(to: Float.self, capacity: count)
        return Array(UnsafeBufferPointer(start: resultPointer, count: count))
    }
}
```

---

## 🎓 TB-003 Implementation Plan

### Phase 1: Foundation (Week 1)

**Goal:** Get ONE kernel working end-to-end

1. **Setup Metal infrastructure**
   - Enhance `MetalBackend.swift`
   - Add library loading
   - Add buffer management

2. **Write simple kernel** (vector add)
   - Learn the workflow
   - Test Swift ↔ Metal communication

3. **Write tests**
   - Compare Metal vs CPU output
   - Verify numerical accuracy

**Deliverable:** One working kernel with tests

---

### Phase 2: MatMul Kernel (Week 2)

**Goal:** Fast, optimized matrix multiplication

1. **Naive matmul first**
   - Get it working
   - Test correctness

2. **Tiled optimization**
   - Add threadgroup memory
   - Optimize tile size (16×16)

3. **Benchmark**
   - Compare to CPU (cblas_sgemm)
   - Target: 3-10× speedup

**Deliverable:** Production-grade MatMul kernel

---

### Phase 3: Activation & Normalization Kernels (Week 3)

**Goal:** Complete transformer kernel suite

1. **Softmax kernel**
   - Handle per-row normalization
   - Numerical stability (max subtraction)

2. **LayerNorm kernel**
   - Per-vector normalization
   - Fused operations

3. **GELU kernel** (bonus)
   - Might be faster than CPU loop

**Deliverable:** All transformer ops on GPU

---

### Phase 4: Backend Abstraction (Week 4)

**Goal:** Automatic CPU/GPU selection

1. **Op registry pattern**
   ```swift
   protocol TensorOp {
       func execute(on: Backend) -> Tensor
   }
   
   enum Backend {
       case cpu
       case metal
   }
   ```

2. **Automatic selection**
   - Metal if available
   - CPU fallback otherwise
   - Logging for debugging

3. **Performance tests**
   - Compare CPU vs Metal
   - Document speedups

**Deliverable:** Transparent backend switching

---

## 🔧 Technical Details

### Threadgroup Memory Limits

**Apple Silicon:**
- 32 KB per threadgroup
- For Float32: 8,192 floats
- For 16×16 tiles: 256 floats/tile × 2 tiles = 512 floats = 2 KB ✅

**Calculation:**
```
Tile A: 16×16 = 256 floats × 4 bytes = 1 KB
Tile B: 16×16 = 256 floats × 4 bytes = 1 KB
Total: 2 KB (well under 32 KB limit) ✅
```

### Buffer Storage Modes

**Three options:**

1. **`.storageModeShared`** (default)
   - CPU and GPU can both access
   - Slower for GPU (coherency overhead)
   - Good for small buffers

2. **`.storageModePrivate`**
   - GPU-only memory
   - Fastest for GPU
   - Need to copy to/from CPU

3. **`.storageModeManaged`** (macOS only)
   - Separate CPU and GPU copies
   - Explicit synchronization

**For TinyBrain:**
- Use `.storageModePrivate` for model weights (never change)
- Use `.storageModeShared` for activations (read results on CPU)

---

## 📊 Expected Performance Gains

### Based on Research & Apple Silicon Specs

**CPU (Accelerate) vs GPU (Metal):**

| Operation | CPU (TB-002) | Metal (Expected) | Speedup |
|-----------|--------------|------------------|---------|
| MatMul 128×128 | 0.053ms | **0.005-0.010ms** | **5-10×** |
| MatMul 512×512 | 0.776ms | **0.080-0.150ms** | **5-10×** |
| MatMul 2048×2048 | ~100ms | **10-20ms** | **5-10×** |
| Softmax (1K) | 0.05ms | **0.005-0.01ms** | **5-10×** |
| LayerNorm (1K) | 0.05ms | **0.005-0.01ms** | **5-10×** |

**Why 5-10× (not 100×)?**
- Accelerate is ALREADY fast (uses CPU SIMD)
- Metal is faster but not dramatically (both optimized)
- Overhead of CPU↔GPU data transfer
- For very large matrices, Metal wins big

**Real-world impact:**
```
TB-002 toy model: 1049 tokens/sec (CPU)
TB-003 with Metal: 5000-10000 tokens/sec (estimated)

But real model (TinyLlama):
TB-002: ~5-10 tokens/sec
TB-003: ~30-50 tokens/sec (3-5× improvement more realistic)
```

---

## 🎯 Recommended Scope for TB-003

### MUST HAVE (Core Deliverables)

1. ✅ **MatMul kernel** (tiled, optimized)
2. ✅ **Metal backend abstraction** (Swift wrapper)
3. ✅ **Automatic CPU/GPU fallback**
4. ✅ **Tests** (Metal vs CPU parity)
5. ✅ **Benchmarks** (document speedups)

### NICE TO HAVE (If Time)

6. ⚠️ **Softmax kernel** (or use CPU for now)
7. ⚠️ **LayerNorm kernel** (or use CPU for now)
8. ⚠️ **Device-specific tuning** (defer to later)

### DEFER (Out of Scope)

9. ❌ **Stride-aware tensors** - Big change, needs careful design
10. ❌ **INT8 dequantization** - That's TB-004
11. ❌ **KV-cache kernels** - That's TB-004

**Rationale:** Get MatMul on GPU first. That's 70% of the win. Other ops can stay on CPU for now.

---

## ⚠️ Common Pitfalls (Learn from Others' Mistakes!)

### Pitfall 1: Data Transfer Overhead

```swift
// BAD: Transfer data every operation
for token in tokens {
    let gpuResult = metalMatmul(a, b)  // Copy to GPU, compute, copy back
    // Slow! CPU↔GPU transfer kills performance
}

// GOOD: Keep data on GPU
let gpuA = copyToGPU(a)  // Once
let gpuB = copyToGPU(b)  // Once
for token in tokens {
    let gpuResult = metalMatmul(gpuA, gpuB)  // No transfer!
}
copyFromGPU(gpuResult)  // Once
```

### Pitfall 2: Wrong Threadgroup Size

```metal
// BAD: 15×15 = 225 threads (not multiple of 32)
// Wastes 7 threads per SIMD group

// GOOD: 16×16 = 256 threads (8 SIMD groups)
```

### Pitfall 3: Not Using Threadgroup Memory

```metal
// BAD: Read from global memory repeatedly
for (int k = 0; k < K; k++) {
    sum += A[...] * B[...];  // Slow!
}

// GOOD: Load into threadgroup memory first
tileA[...] = A[...];  // Load once
threadgroup_barrier();
for (int k = 0; k < TILE_SIZE; k++) {
    sum += tileA[...] * tileB[...];  // Fast!
}
```

---

## 📚 Learning Resources

**What I found:**
- Metal Performance Shaders has neural network ops
- Apple's "Performing Calculations on GPU" tutorial
- Metal Shading Language spec

**What you should read** (I'll create these for you):
- TB-003 implementation guide
- Metal kernel tutorial
- Debugging Metal shaders guide

---

## 🎯 My Recommendation for TB-003

### Start Simple, Iterate Fast

**Week 1: Foundation**
- Enhance MetalBackend.swift
- Write vector_add kernel (simple)
- Get pipeline working

**Week 2: MatMul**
- Naive matmul first
- Then tiled optimization
- Benchmark vs CPU

**Week 3: Integration**
- Backend abstraction
- Automatic fallback
- Tests for parity

**Week 4: (Optional)**
- Additional kernels
- Tuning
- Documentation

**Result:** Working, tested, fast Metal matmul in 2-3 weeks

---

## ❓ Questions Before We Start

1. **Scope:** MatMul only first, or try to do all ops?
2. **Approach:** Custom kernels (educational) or use MPS (faster to ship)?
3. **Testing:** How thorough? (Parity tests + performance tests?)
4. **Timeline:** Rush it or take time to do it right?

**My recommendation:**
- **Scope:** MatMul first (70% of value)
- **Approach:** Custom tiled matmul (educational + fast)
- **Testing:** Rigorous (TDD like TB-002)
- **Timeline:** 2-3 weeks (don't rush, Review Hitler is watching! 😄)

---

**What do you think? Ready to start, or want to discuss the approach first?** 🚀

