# ADR-006: InferenceObserver Protocol for X-Ray Mode

**Status:** Accepted
**Date:** 2025-10-26
**Task:** TB-010

## Context

TinyBrain's differentiator is live visualization of transformer internals — "X-Ray Mode." This requires hooking into the inference pipeline to capture attention weights, layer activations, and logit distributions without impacting performance when visualization is disabled.

Three observation approaches were considered:

1. **Closure callbacks** — simple but hard to compose and no type safety across hooks.
2. **NotificationCenter** — decoupled but high overhead for per-token events and no compile-time checking.
3. **Protocol with optional methods** — type-safe, composable, zero cost when nil.

## Decision

We defined an **`InferenceObserver` protocol** with three hooks, attached to `ModelRunner` as a **weak optional reference**:

```swift
protocol InferenceObserver: AnyObject {
    func didComputeAttention(layerIndex: Int, weights: [Float], position: Int)
    func didEnterLayer(layerIndex: Int, hiddenStateNorm: Float, position: Int)
    func didComputeLogits(logits: [Float], position: Int)
}
```

- `ModelRunner.observer` is `weak var observer: InferenceObserver?`
- Each hook call is guarded by `observer?.method()` — zero cost when nil.
- `XRayViewModel` conforms to the protocol, accumulates observations, and publishes `XRaySnapshot` to SwiftUI views.

## Consequences

**Positive:**

- **Zero cost when off** — nil check is a single pointer comparison. No allocations, no indirection.
- **Type-safe** — compiler enforces hook signatures. Adding a new hook is a protocol change that flags all conformers.
- **Weak reference** — no retain cycles. Observer is automatically cleared when the view model is deallocated.
- **Composable** — any class can conform to `InferenceObserver` for custom analysis.

**Negative:**

- **Single observer** — only one observer at a time. If multiple consumers are needed, a multiplexer would be required (not needed for v0.1.0).
- **Data copying** — attention weights are passed as `[Float]` arrays (copied). For very large models, a buffer-based approach may be needed.

## Evidence

- X-Ray Mode renders attention heatmaps, token probabilities, layer activations, and KV cache usage in real-time
- Zero measurable performance impact when `observer` is nil
- See `Sources/TinyBrainRuntime/InferenceObserver.swift` for the protocol definition
