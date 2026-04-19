# ADR-004: Copy-on-Write Tensor Storage

**Status:** Accepted
**Date:** 2025-10-25
**Task:** TB-004 (Phase 2)

## Context

TinyBrain's `Tensor` type is a value type (`struct`) for API ergonomics and safety. However, tensors can be large — a 10,000×10,000 Float32 tensor is 400 MB. Naive value semantics would copy the entire buffer on every assignment, which is unacceptable on memory-constrained devices.

## Decision

We implemented **Copy-on-Write (CoW)** using Swift's `isKnownUniquelyReferenced` mechanism:

- `Tensor<Element>` is a struct wrapping a reference-counted `TensorStorage<Element>` (class).
- Assignment shares the storage reference — no data copy.
- On mutation, Swift checks if the storage reference is unique. If shared, a copy is made before writing.
- This is the same pattern used by Swift's `Array`, `String`, and `Data` types.

`Tensor` was also made **generic** over `TensorElement` (Float32, Float16, Int8) in the same phase, enabling INT8 quantized tensors to share the same infrastructure.

## Consequences

**Positive:**

- **90% memory savings** for common patterns — 10 copies of a 400 MB tensor use 400 MB total (not 4 GB).
- **Zero-cost reads** — shared access is free.
- **Transparent API** — callers use value semantics without worrying about memory.
- **Type safety** — `Tensor<Float>`, `Tensor<Float16>`, `Tensor<Int8>` are distinct types at compile time.

**Negative:**

- **Mutation triggers copy** — code that mutates shared tensors in a loop can accidentally trigger O(n) copies. Mitigated by documenting the pattern and using raw arrays in hot paths (e.g., KV cache append).
- **Reference counting overhead** — minimal but non-zero for ARC retain/release.

## Evidence

- 11/11 generic tensor tests passing
- CoW verified: assignment shares storage (same ObjectIdentifier), mutation triggers copy
- See `docs/TB-004-COMPLETE.md` Phase 2 for memory impact analysis
