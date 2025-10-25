# TinyBrain Task Roadmap

**Complete implementation plan with deferred items tracked**

---

## 📊 Task Overview

| Task | Status | Description | Duration |
|------|--------|-------------|----------|
| **TB-001** | ✅ COMPLETE | Scaffold workspace | Complete |
| **TB-002** | ✅ COMPLETE | Tensor engine MVP (Float32 only) | Complete |
| **TB-003** | 🚧 NEXT | Metal GPU kernels + strides | 5-7 days |
| **TB-004** | 📋 PLANNED | Quantization + CoW + KV-cache | 7-10 days |
| **TB-005** | 📋 PLANNED | Tokenizer + Sampler + Streaming | 4-6 days |
| **TB-006** | 📋 PLANNED | SwiftUI Chat Demo | 3-5 days |
| **TB-007** | 📋 PLANNED | Benchmarks + Docs + Release | 3-4 days |

**Total estimated:** ~30-40 days (6-8 weeks)

---

## 🔄 Deferred Items Tracker

This section tracks what we're **intentionally deferring** to later tasks to keep each task focused.

### From TB-002 → Later Tasks

**Deferred to TB-003 (Metal Kernels):**
- ✋ **Stride-aware storage** - Needed for efficient transpose/reshape in Metal
- ✋ **Op registry abstraction** - CPU/GPU selection becomes relevant with Metal

**Deferred to TB-004 (Quantization):**
- ✋ **Generic types** (`Tensor<Element>`) - Float16, Int8 support
- ✋ **Copy-on-write optimization** - Memory efficiency for large models
- ✋ **mmap-backed loading** - Lazy weight loading for quantized checkpoints

**Why defer?**
- ✅ Avoid premature optimization
- ✅ Keep TB-002 focused and testable
- ✅ Each task has clear, achievable scope
- ✅ Learn and iterate (don't build everything at once)

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

