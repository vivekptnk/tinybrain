# ADR-003: Paged KV-Cache Architecture

**Status:** Accepted
**Date:** 2025-10-25
**Task:** TB-004 (Phase 4)

## Context

Transformer inference requires caching key/value tensors from previous tokens to avoid recomputation — the KV cache. For a 2048-token context with 768-dim hidden state across 6 layers, this is approximately 72 MB of state that must be managed efficiently.

Two approaches were considered:

1. **Contiguous buffer** — pre-allocate the full context buffer up front. Simple but wastes memory for short conversations and requires reallocation for longer ones.
2. **Paged allocation** — allocate fixed-size pages on demand. More complex but memory-efficient and supports eviction.

## Decision

We chose **paged allocation** with a free-list page allocator:

- **16 tokens per page** — balances granularity with overhead. Each page holds K and V tensors for 16 positions.
- **`PageAllocator`** — maintains a free list; pre-allocates pages to avoid runtime allocation during inference.
- **`KVCache`** — maps sequence positions to pages, supports automatic eviction when the 2048-token capacity is reached (FIFO — oldest page evicted first).
- **Thread-safe** — `NSLock` protects concurrent access.
- **Raw Float arrays** — stores K/V as `[Float]` instead of `Tensor` to avoid Copy-on-Write overhead during the hot append path.

## Consequences

**Positive:**

- **2048-token context** supported with predictable memory usage (~36 MB for 6 layers).
- **Zero-allocation inference loop** — pages pre-allocated, no malloc during generation.
- **10x faster append** — optimized from 0.426s to 0.041s by using raw Float arrays instead of Tensor (avoids CoW overhead).
- **No memory leaks** — validated with 10,000 cycle stress test.
- **O(n) complexity** — KV cache makes generation O(n) instead of O(n^2) in sequence length.

**Negative:**

- **Fixed page size** — 16 tokens is a compromise; very short or very long sequences may not page-align perfectly.
- **FIFO eviction only** — more sophisticated policies (attention-based, frequency-based) deferred to future work.

## Evidence

- 15/15 KV cache tests passing
- 10,000 cycle memory leak test: no leaks
- Append performance: 0.41 ms per token
- See `docs/TB-004-COMPLETE.md` Phase 4 for implementation details
