# ADR-002: GPU-Resident Tensors and M4 AMX Reality

**Status:** Accepted
**Date:** 2025-10-25
**Task:** TB-004 (Phase 1)

## Context

TB-003 delivered Metal GPU kernels for matmul, softmax, and layer norm. However, Metal was **100x slower** than CPU because every operation allocated a new `MTLBuffer` (0.45 ms overhead per allocation) and round-tripped data between CPU and GPU.

The original target was **3–8x GPU speedup** over CPU, based on generic GPU-vs-CPU comparisons.

## Decision

We implemented **GPU-resident tensors** with lazy CPU/GPU synchronization and a **persistent buffer pool**:

- `TensorStorage` tracks whether data lives on CPU, GPU, or both.
- `toGPU()` / `toCPU()` / `isOnGPU` control placement. Chained GPU operations avoid CPU round-trips.
- `MetalBufferPool` reuses `MTLBuffer` objects (450x faster allocation: 0.001 ms vs 0.45 ms).

We also **revised performance expectations** after discovering M4 Max uses **AMX (Apple Matrix Extension)** — a dedicated matrix coprocessor that Accelerate routes `cblas_sgemm` through. AMX is not a general CPU; it is specialized hardware that competes with or beats GPU for single matmul operations.

**Revised target:** Metal competitive with Accelerate/AMX (0.8–1.3x range), with GPU winning for batched workflows.

## Consequences

**Positive:**

- **Eliminated transfer overhead** — went from 100x slower to competitive (0.74–1.28x).
- **Buffer pool** — 450x faster allocation, thread-safe, bounded pool size, no memory leaks.
- **Infrastructure for batched workflows** — chained attention operations (Q×K^T, softmax, ×V) stay on GPU, which is where real speedups come from.
- **Honest engineering** — documented the AMX reality rather than chasing an unrealistic target.

**Negative:**

- **Single-op GPU speedup is modest** — only 1.28x at best (1536×1536). AMX dominates at most sizes.
- **GPU infrastructure adds code complexity** — TensorStorage, BufferPool, lazy sync logic.

## Evidence

| Matrix Size | CPU (ms) | GPU (ms) | Speedup |
|-------------|----------|----------|---------|
| 512×512     | 0.43     | 0.84     | 0.51x   |
| 1024×1024   | 2.42     | 2.36     | 1.02x   |
| 1536×1536   | 6.06     | 4.73     | 1.28x   |
| 2048×2048   | 8.74     | 9.84     | 0.89x   |

- See `docs/TB-004-M4-FINDINGS.md` for the full hardware analysis
- See `docs/TB-004-COMPLETE.md` Phase 1 for implementation details
