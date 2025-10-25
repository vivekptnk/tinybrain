<!-- 3d08221c-849e-4500-8249-b6aaf580b9df 4463cf9f-935b-4d6a-99b9-5e25b1d9669b -->
# TB-004: INT8 Quantization & Paged KV Cache - TDD Implementation

## TDD Principles for This Task

**Red-Green-Refactor Cycle:**

1. **RED**: Write failing test that defines desired behavior
2. **GREEN**: Write minimal code to make test pass
3. **REFACTOR**: Improve code quality while keeping tests green

**Test Requirements:**

- Tests document WHAT the operation does, WHY it matters, HOW to verify
- Numerical accuracy: relative error < 1e-5 for Float32, < 1e-3 for Float16
- Include edge cases: empty tensors, mismatched shapes, NaN/Inf handling
- Test naming: `test<Operation><Scenario>`

## Phase 1: Fix Metal Performance (CRITICAL) - TDD

### 1.1 RED: Write Failing Tests for GPU-Resident Tensors

**File:** `Tests/TinyBrainRuntimeTests/GPUTensorTests.swift`

```swift
class GPUTensorTests: XCTestCase {
    func testGPUTensorCreation() {
        // WHAT: Create GPU tensor from CPU tensor
        // WHY: Foundation of GPU-resident data
        // HOW: Verify no immediate transfer occurs
        let cpu = Tensor.zeros(shape: TensorShape(100, 100))
        let gpu = cpu.toGPU()
        XCTAssertTrue(gpu.isOnGPU)
    }
    
    func testLazySynchronization() {
        // WHAT: Data only transfers when accessed from other side
        // WHY: Eliminate unnecessary CPU↔GPU copies
        let gpu = Tensor.zeros([100, 100]).toGPU()
        let result = gpu.matmul(gpu)  // Stays on GPU
        XCTAssertTrue(result.isOnGPU, "Should stay on GPU")
    }
    
    func testGPUChainedOperations() {
        // WHAT: Multiple GPU ops without CPU roundtrip
        // WHY: Batched operations = faster
        let a = Tensor.random([512, 512]).toGPU()
        let b = a.matmul(a).softmax().matmul(a)
        XCTAssertTrue(b.isOnGPU)
    }
}
```

**File:** `Tests/TinyBrainMetalTests/BufferPoolTests.swift`

```swift
func testBufferReuse() {
    // WHAT: Buffers reused for same-size tensors
    // WHY: Eliminate allocation overhead
    let pool = MetalBufferPool()
    let buf1 = try pool.acquire(elementCount: 1024)
    let id1 = ObjectIdentifier(buf1)
    pool.release(buf1)
    let buf2 = try pool.acquire(elementCount: 1024)
    XCTAssertEqual(ObjectIdentifier(buf2), id1)
}
```

**File:** `Tests/TinyBrainMetalTests/PerformanceBenchmarks.swift`

```swift
func testMetalSpeedupWithPersistentBuffers() throws {
    // WHAT: Metal ≥3× faster than CPU for 1024×1024
    // WHY: Validates TB-003 performance fix
    // ACCURACY: relative error < 1e-3
    
    let a = Tensor.random([1024, 1024])
    let b = Tensor.random([1024, 1024])
    
    let (cpuTime, cpuResult) = measure { a.matmulCPU(b) }
    let (gpuTime, gpuResult) = measure { 
        a.toGPU().matmul(b.toGPU()).toCPU() 
    }
    
    XCTAssertLessThan(relativeError(cpuResult, gpuResult), 1e-3)
    XCTAssertGreaterThanOrEqual(cpuTime / gpuTime, 3.0)
}
```

### 1.2 GREEN: Implement to Pass Tests

- `Sources/TinyBrainRuntime/TensorStorage.swift` (new)
- `Sources/TinyBrainRuntime/Tensor.swift` (add GPU methods)
- `Sources/TinyBrainMetal/BufferPool.swift` (new)
- `Sources/TinyBrainMetal/MetalBackend.swift` (integrate pool)

### 1.3 REFACTOR: Clean Up

- Extract buffer pooling logic
- Add telemetry
- Document lazy sync

## Phase 2: Generic Tensor with CoW - TDD

### 2.1 RED: Write Failing Tests for Generic Tensors

**File:** `Tests/TinyBrainRuntimeTests/GenericTensorTests.swift`

```swift
func testFloat32Tensor() {
    // WHAT: Tensor<Float> works like current Tensor
    // WHY: Backward compatibility
    let t = Tensor<Float>.zeros(shape: TensorShape(10, 10))
    XCTAssertEqual(t.shape.count, 100)
}

func testFloat16Tensor() {
    // WHAT: Tensor<Float16> for memory efficiency
    // WHY: Half precision = 50% memory savings
    let t = Tensor<Float16>.random([100, 100])
    XCTAssertEqual(MemoryLayout.size(ofValue: t.storage.data[0]), 2)
}

func testInt8Tensor() {
    // WHAT: Tensor<Int8> for quantized weights
    // WHY: Foundation of quantization
    let t = Tensor<Int8>.filled([10, 10], value: 127)
    XCTAssertEqual(t[0, 0], 127)
}

func testCopyOnWrite() {
    // WHAT: Cheap copy, write triggers unique copy
    // WHY: Value semantics without performance cost
    var a = Tensor<Float>.random([1000, 1000])
    let b = a  // Cheap copy
    a[0, 0] = 999  // Triggers COW
    XCTAssertNotEqual(a[0, 0], b[0, 0])
}

func testCopyOnWriteAvoidance() {
    // WHAT: No copy if already unique
    // WHY: Performance optimization
    var a = Tensor<Float>.zeros([1000, 1000])
    // Modify without triggering copy (already unique)
    a[0, 0] = 1.0
    // Verify no unnecessary allocation
}
```

### 2.2 RED: Edge Case Tests

```swift
func testEmptyTensor() {
    // Edge: What if tensor has 0 elements?
    // Should handle gracefully
}

func testNaNHandling() {
    // Edge: NaN in tensor data
    // Operations should propagate or detect
}

func testInfHandling() {
    // Edge: Infinity values
}
```

### 2.3 GREEN: Implement Generic Tensor

- `Sources/TinyBrainRuntime/TensorElement.swift` (new)
- `Sources/TinyBrainRuntime/TensorStorage.swift` (refactor to generic)
- `Sources/TinyBrainRuntime/Tensor.swift` (make generic)

### 2.4 REFACTOR: Type Aliases & Migration

```swift
typealias FloatTensor = Tensor<Float>
```

## Phase 3: INT8 Quantization - TDD

### 3.1 RED: Write Failing Quantization Tests

**File:** `Tests/TinyBrainRuntimeTests/QuantizationTests.swift`

```swift
func testSymmetricQuantization() {
    // WHAT: FP32 → INT8 → FP32 roundtrip
    // WHY: Verify quantization correctness
    // ACCURACY: ≤1% perplexity delta
    
    let original = Tensor<Float>.random([128, 768])
    let quantized = original.quantize(mode: .symmetric)
    let dequantized = quantized.dequantize()
    
    let error = relativeError(original, dequantized)
    XCTAssertLessThan(error, 0.01, "≤1% accuracy loss")
}

func testPerChannelScales() {
    // WHAT: Each channel has own scale
    // WHY: Better accuracy than per-tensor
    
    let weights = Tensor<Float>([
        [1.0, 2.0, 3.0],
        [100.0, 200.0, 300.0]  // Different magnitude
    ])
    
    let quant = weights.quantize(mode: .perChannel)
    XCTAssertEqual(quant.scale.count, 2, "One scale per row")
    XCTAssertNotEqual(quant.scale[0], quant.scale[1])
}

func testQuantizedMatMul() {
    // WHAT: MatMul with INT8 weights
    // WHY: End-to-end quantized inference
    // ACCURACY: relative error < 1e-3 vs FP32
    
    let a = Tensor<Float>.random([128, 256])
    let b = Tensor<Float>.random([256, 512])
    
    let fp32Result = a.matmul(b)
    
    let bQuant = b.quantize()
    let int8Result = a.matmul(bQuant)  // Auto-dequant
    
    XCTAssertLessThan(relativeError(fp32Result, int8Result), 1e-3)
}

func testQuantizationEdgeCases() {
    // Edge: All zeros
    let zeros = Tensor<Float>.zeros([10, 10])
    let q = zeros.quantize()
    XCTAssertEqual(q.scale[0], 0.0)
    
    // Edge: Single outlier
    // Edge: Negative values
}
```

### 3.2 GREEN: Implement Quantization

- `Sources/TinyBrainRuntime/QuantizedTensor.swift`
- `Sources/TinyBrainRuntime/Quantization.swift`
- `Sources/TinyBrainMetal/Shaders/Dequant.metal`

### 3.3 REFACTOR: Optimize Kernels

- Fused dequant+matmul Metal kernel
- CPU fallback optimization

## Phase 4: Paged KV Cache - TDD

### 4.1 RED: Write Failing KV Cache Tests

**File:** `Tests/TinyBrainRuntimeTests/KVCacheTests.swift`

```swift
func testPageAllocation() {
    // WHAT: Allocate pages from pool
    // WHY: Foundation of paging
    let allocator = KVCacheAllocator(pageSize: 16, maxPages: 128)
    let page1 = allocator.allocatePage()
    let page2 = allocator.allocatePage()
    XCTAssertNotEqual(page1, page2)
}

func testPageReuse() {
    // WHAT: Released pages get reused
    // WHY: Zero allocation during inference
    let allocator = KVCacheAllocator(pageSize: 16, maxPages: 128)
    let page = allocator.allocatePage()!
    allocator.freePage(page)
    let reused = allocator.allocatePage()!
    XCTAssertEqual(page, reused)
}

func testKVCacheAppend() {
    // WHAT: Append key/value tensors
    // WHY: Build up context
    let cache = KVCache(layers: 6, hiddenDim: 768, maxTokens: 2048)
    
    for pos in 0..<100 {
        let key = Tensor<Float>.random([768])
        let value = Tensor<Float>.random([768])
        cache.append(layer: 0, key: key, value: value, position: pos)
    }
    
    let keys = cache.getKeys(layer: 0, range: 0..<100)
    XCTAssertEqual(keys.shape, TensorShape(100, 768))
}

func testKVCacheRetrieval() {
    // WHAT: Retrieve cached K/V for attention
    // WHY: Verify correctness vs recomputing
    
    // Store known values
    // Retrieve and verify match
    // Compare attention output with/without cache
}

func testContextWindow() {
    // WHAT: 2048 token context support
    // WHY: PRD requirement
    let cache = KVCache(maxTokens: 2048)
    for i in 0..<2048 {
        cache.append(...)
    }
    XCTAssertEqual(cache.length, 2048)
}

func testEviction() {
    // WHAT: Evict oldest when full
    // WHY: Bounded memory
    let cache = KVCache(maxTokens: 100)
    for i in 0..<150 {
        cache.append(...)
    }
    XCTAssertEqual(cache.length, 100)
}

func testMemoryLeak() {
    // Edge: No leaks under stress
    // Run 10k append/evict cycles
    // Verify memory doesn't grow
}
```

### 4.2 GREEN: Implement KV Cache

- `Sources/TinyBrainRuntime/PageAllocator.swift`
- `Sources/TinyBrainRuntime/KVCache.swift`

### 4.3 REFACTOR: GPU Optimization

- Move to Metal buffers
- Optimize page compaction

## Phase 5: Streaming API - TDD

### 5.1 RED: Write Failing Streaming Tests

**File:** `Tests/TinyBrainRuntimeTests/StreamingTests.swift`

```swift
func testStepReusesKVCache() async throws {
    // WHAT: step() doesn't recompute past tokens
    // WHY: KV cache performance benefit
    
    let runner = ModelRunner(model: mockModel)
    
    let token1 = try await runner.step(token: 100)
    let cacheSize1 = runner.kvCache.length
    
    let token2 = try await runner.step(token: token1)
    let cacheSize2 = runner.kvCache.length
    
    XCTAssertEqual(cacheSize2, cacheSize1 + 1)
}

func testStreamGeneration() async throws {
    // WHAT: Generate stream of tokens
    // WHY: Progressive output for UI
    
    let runner = ModelRunner(model: mockModel)
    var tokens: [String] = []
    
    for try await token in runner.generateStream(prompt: "Hello") {
        tokens.append(token)
        if tokens.count >= 10 { break }
    }
    
    XCTAssertEqual(tokens.count, 10)
}

func testFirstTokenLatency() async throws {
    // WHAT: First token < 150ms
    // WHY: PRD performance requirement
    
    let start = Date()
    var firstToken: String?
    for try await token in runner.generateStream(prompt: "Hi") {
        firstToken = token
        break
    }
    let latency = Date().timeIntervalSince(start)
    
    XCTAssertLessThan(latency, 0.15, "First token < 150ms")
}
```

### 5.2 GREEN: Implement Streaming

- `Sources/TinyBrainRuntime/ModelRunner.swift`

### 5.3 REFACTOR: AsyncSequence Optimization

## Phase 6: Integration & Validation - TDD

### 6.1 End-to-End Tests

```swift
func testQuantizedModelInference() async throws {
    // WHAT: Load INT8 model, generate text
    // WHY: Validate entire pipeline
    // ACCURACY: Perplexity delta ≤1% vs FP16
    
    let model = try TinyBrain.load("tinyllama-int8.tbf")
    let output = try await model.generate("Explain gravity")
    XCTAssertFalse(output.isEmpty)
}

func testMemoryBudget() throws {
    // WHAT: Peak memory ≤6GB on iPhone 15 Pro
    // WHY: PRD requirement
    // Use XCTMemoryMetric
}

func testMemorySavings() {
    // WHAT: ≥35% savings INT8 vs FP16
    // WHY: Quantization benefit
    let fp16Size = measureModelSize(quantization: .fp16)
    let int8Size = measureModelSize(quantization: .int8)
    let savings = (fp16Size - int8Size) / fp16Size
    XCTAssertGreaterThanOrEqual(savings, 0.35)
}
```

## Success Metrics (Test-Driven)

All metrics validated by automated tests:

- ✅ `testMetalSpeedupWithPersistentBuffers`: ≥3× speedup
- ✅ `testSymmetricQuantization`: ≤1% accuracy loss
- ✅ `testMemorySavings`: ≥35% reduction
- ✅ `testContextWindow`: 2048 tokens supported
- ✅ `testMemoryBudget`: ≤6GB peak
- ✅ `testFirstTokenLatency`: < 150ms

### To-dos

- [x] Implement GPU-resident tensor type with lazy CPU↔GPU synchronization to fix Metal performance
- [x] Add persistent buffer pool to MetalBackend to eliminate per-operation allocation overhead
- [x] Benchmark and validate Metal achieves ≥3× speedup vs CPU after persistent buffer changes
- [x] Refactor Tensor to use generic TensorStorage<Element> with copy-on-write optimization
- [x] Implement TensorElement protocol with Float32, Float16, and Int8 conformances
- [x] Design and implement QuantizedTensor container with per-channel INT8 weights and metadata
- [x] Build calibration utilities for computing per-channel scales/zero-points from FP16 checkpoints
- [ ] Implement Metal dequantization kernel with CPU fallback for INT8→FP32 conversion
- [ ] Build paged KV cache allocator with page table, free list, and compaction support
- [ ] Implement KVCache manager with GPU buffers supporting 2048 token context window
- [ ] Add streaming-friendly ModelRunner.step(token:) API that reuses KV pages
- [ ] Write tests verifying numerical fidelity (≤1% perplexity delta) and memory leak checks
- [ ] Update docs with KV page flow diagrams, quantization pipeline, CoW strategy, and GPU-resident architecture