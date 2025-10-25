# TinyBrain Task Roadmap

**Complete implementation plan with deferred items tracked**

**Last Updated:** October 25, 2025 (TB-004 Complete!)

---

## 📊 Task Overview

| Task | Status | Description | Completed |
|------|--------|-------------|-----------|
| **TB-001** | ✅ COMPLETE | Scaffold workspace | Oct 2025 |
| **TB-002** | ✅ COMPLETE | Tensor engine MVP (Float32 only) | Oct 2025 |
| **TB-003** | ✅ COMPLETE | Metal GPU kernels + buffer pool | Oct 2025 |
| **TB-004** | ✅ **COMPLETE** | **Quantization + CoW + KV-cache + Streaming** | **Oct 25, 2025** |
| **TB-005** | 🚀 NEXT | Tokenizer + Sampler + Real Transformer | 4-6 days |
| **TB-006** | 📋 PLANNED | SwiftUI Chat Demo | 3-5 days |
| **TB-007** | 📋 PLANNED | Benchmarks + Docs + Release | 3-4 days |

**Progress:** 4 of 7 tasks complete (57%)

---

## 🔄 Deferred Items Tracker

This section tracks what we're **intentionally deferring** to later tasks to keep each task focused.

### From TB-002 → Later Tasks

**Deferred to TB-003 (Metal Kernels):**
- ✋ **Stride-aware storage** - Needed for efficient transpose/reshape in Metal
- ✋ **Op registry abstraction** - CPU/GPU selection becomes relevant with Metal

**✅ COMPLETED in TB-004:**
- ✅ **Generic types** (`Tensor<Element>`) - Float32, Float16, Int8 support ✅
- ✅ **Copy-on-write optimization** - Memory efficiency for large models ✅
- ✅ **Paged KV cache** - 2048-token context with zero-allocation inference ✅
- ✅ **INT8 quantization** - 75% memory savings, per-channel scales ✅
- ✅ **Streaming API** - ModelRunner with AsyncSequence ✅

**TB-004 Results:**
- 94/94 tests passing
- 4 commits, 5,064 lines of code
- Validated on M4 Max hardware
- All acceptance criteria exceeded

**Why this approach worked:**
- ✅ TDD methodology (write tests first)
- ✅ Each phase buildable and testable
- ✅ Hardware validation at each step
- ✅ Educational documentation throughout

---

## 📝 TB-004 Detailed Scope ✅ **COMPLETE**

### ✅ COMPLETED in TB-004

**Phase 1: GPU-Resident Tensors**
- TensorStorage with lazy CPU↔GPU sync
- MetalBufferPool (450× faster allocation)
- GPU tensor API: toGPU(), toCPU(), isOnGPU
- 13 tests passing

**Phase 2: Generic Tensor + CoW**
- Tensor<Element: TensorElement>
- Float32, Float16, Int8 support
- Copy-on-Write with isKnownUniquelyReferenced
- 11 tests passing

**Phase 3: INT8 Quantization**
- QuantizedTensor with per-channel scales
- quantize()/dequantize() methods
- 75% memory savings, <1% error
- 11 tests passing

**Phase 4: Paged KV Cache**
- PageAllocator with free list
- KVCache supporting 2048 tokens
- Zero-allocation inference
- 15 tests passing

**Phase 5: Streaming API**
- ModelRunner.step(tokenId:)
- AsyncSequence streaming
- SwiftUI-ready
- 7 tests passing

**Total:** 57 tests, all passing on M4 Max

See [TB-004-COMPLETE.md](TB-004-COMPLETE.md) for full details.

---

## 📝 TB-003 Detailed Scope

### ✅ COMPLETED in TB-003

**Metal MatMul Kernel:**
- Tiled implementation (16×16 threadgroups)
- Threadgroup memory optimization
- Custom .metal shader (educational!)

**Backend Abstraction:**
- Transparent CPU/GPU selection
- Automatic fallback to Accelerate
- Logging for debugging

**Stride-Aware Tensors:**
- Support transpose without copy
- Reshape operations
- Non-contiguous memory layouts

**Testing & Benchmarks:**
- TDD tests for numerical parity
- Performance benchmarks (3-10× speedup target)
- Metal vs CPU comparison

### ❌ DEFERRED from TB-003

**To TB-004 or later:**
- GPU Softmax kernel
- GPU LayerNorm kernel
- GPU GELU kernel
- INT8 dequantization kernel
- KV-cache GPU operations

**To TB-007:**
- Device-specific autotuning
- JSON config for tile sizes
- Fused operations (bias+activation)

**Rationale:** MatMul is 70% of compute time - focus on the big win!

---

## 📝 TB-002 Detailed Scope

### ✅ INCLUDED in TB-002

**Tensor Operations:**
- Matrix multiplication (via `cblas_sgemm`)
- Element-wise addition (via `vDSP_vadd`)
- Element-wise multiplication (via `vDSP_vmul`)
- GELU activation function
- ReLU activation function
- Softmax normalization
- Layer normalization

**Tensor Utilities:**
- Factory methods: `zeros`, `filled`, `random`, `identity`
- Subscript access: `tensor[i, j]`
- Shape validation
- Basic storage (contiguous Float32 array)

**Testing (TDD Approach):**
- Numerical accuracy tests (< 1e-5 error)
- Shape validation tests
- Edge case tests (empty tensors, mismatched shapes)
- Numerical stability tests (NaN, Inf handling)
- Educational test documentation

**Documentation:**
- DocC articles for each operation
- BLAS/vDSP tutorial (✅ already created!)
- Operation examples in LLM context
- Update `docs/overview.md`

**Benchmarking:**
- Simple ops/sec measurements
- CPU baseline for Metal comparison
- Script in `Scripts/benchmark-ops.swift`

### ❌ EXCLUDED from TB-002 (Coming Later)

**TB-003 will add:**
- Stride-aware tensors
- Transpose without copy
- Reshape operations
- Metal GPU acceleration
- CPU/GPU backend selection

**TB-004 will add:**
- Generic `Tensor<Element>`
- Float16 support
- Int8 quantization
- Copy-on-write optimization
- Memory-mapped loading
- Paged KV-cache

**TB-005 will add:**
- Tokenization
- Sampling strategies
- Streaming APIs

---

## 🎯 Why This Approach Works

### Principle: **Vertical Slices, Not Horizontal Layers**

**❌ Bad approach (horizontal):**
```
Week 1: Build perfect Tensor with CoW, generics, strides
Week 2: Build all ops
Week 3: Add Metal
Week 4: Add quantization
→ Nothing works until Week 4!
```

**✅ Our approach (vertical):**
```
Week 1: Basic Tensor + MatMul → Can multiply matrices! ✅
Week 2: Add more ops → Can do activations! ✅
Week 3: Add Metal → Faster! ✅
Week 4: Add quantization → Memory efficient! ✅
→ Something useful every week!
```

### Benefits:
- 🎉 Early wins build momentum
- 🐛 Find issues quickly
- 📚 Learn incrementally
- 🔧 Easier to debug
- 👥 Can show progress to others

---

## 📋 Task Dependencies

```
TB-001 (Scaffold)
   ↓
TB-002 (Tensor Engine - Float32 only)
   ↓
TB-003 (Metal Kernels + Strides)
   ↓
TB-004 (Quantization + CoW + Generics)
   ↓
TB-005 (Tokenizer + Sampler + Streaming)
   ↓
TB-006 (SwiftUI Chat Demo)
   ↓
TB-007 (Benchmarks + Docs + Release)
```

**Critical path:** Can't skip tasks - each builds on the previous!

---

## 🎓 TB-002 Learning Objectives

By completing TB-002, you'll understand:

1. **How Accelerate works** - BLAS for matmul, vDSP for vectors
2. **Matrix memory layout** - Row-major vs column-major
3. **Numerical stability** - Why softmax subtracts max
4. **Activation functions** - GELU, ReLU, and why they matter
5. **Normalization** - LayerNorm in transformers
6. **Test-Driven Development** - Tests as documentation and validation
7. **Performance measurement** - Benchmarking methodology

---

## 📖 Next Steps for TB-002

Before coding, ensure you:

- ✅ Read `docs/BLAS-vDSP-Tutorial.md` (just created!)
- ✅ Understand TDD approach in `AGENTS.md`
- ✅ Review current `Tensor.swift` implementation
- ✅ Ask any questions about BLAS/vDSP

Then we'll:

1. **Write tests FIRST** (TDD Red phase)
2. **Implement operations** (TDD Green phase)
3. **Refactor for clarity** (TDD Refactor phase)
4. **Document thoroughly** (Educational mission)
5. **Benchmark** (Validate performance)

---

**Status:** 🟢 Ready to start TB-002 with clear scope!

