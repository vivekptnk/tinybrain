# Architecture Decision Records

This directory contains Architecture Decision Records (ADRs) for TinyBrain. Each ADR documents a significant technical decision, the context that led to it, and the consequences.

## Index

| ADR | Title | Status | Date |
|-----|-------|--------|------|
| [ADR-001](001-per-channel-int8-quantization.md) | Per-Channel INT8 Quantization | Accepted | 2025-10-25 |
| [ADR-002](002-gpu-resident-tensors-amx-reality.md) | GPU-Resident Tensors and M4 AMX Reality | Accepted | 2025-10-25 |
| [ADR-003](003-paged-kv-cache.md) | Paged KV-Cache Architecture | Accepted | 2025-10-25 |
| [ADR-004](004-copy-on-write-tensors.md) | Copy-on-Write Tensor Storage | Accepted | 2025-10-25 |
| [ADR-005](005-format-agnostic-tokenizer.md) | Format-Agnostic Tokenizer via Adapter Pattern | Accepted | 2025-10-26 |
| [ADR-006](006-inference-observer-protocol.md) | InferenceObserver Protocol for X-Ray Mode | Accepted | 2025-10-26 |

## Format

Each ADR follows this structure:

- **Status**: Proposed, Accepted, Deprecated, or Superseded
- **Context**: The situation and forces at play
- **Decision**: What we decided to do
- **Consequences**: The resulting impact, both positive and negative
