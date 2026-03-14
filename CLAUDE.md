# CLAUDE.md — TinyBrain Project Context

## What is TinyBrain?
Swift-native on-device LLM inference runtime for Apple Silicon. Educational + practical. The "micrograd of on-device inference."

## Quick Reference
- **PRD:** `docs/prd.md`
- **Architecture:** `docs/overview.md`
- **Project rules:** `AGENTS.md`

## Build & Test
```bash
swift build                         # Build all targets
swift test --skip TinyBrainDemoTests  # Run tests (skip Demo due to Xcode beta linker bug)
swift run tinybrain-chat            # Run the chat demo
swift run tinybrain-bench           # Run benchmarks
```

## Module Structure
```
Sources/
  TinyBrainRuntime/    — Tensor, ModelRunner, KV-cache, quantization, sampler, InferenceObserver
  TinyBrainMetal/      — Metal GPU backend, kernels, buffer pool
  TinyBrainTokenizer/  — BPE tokenizer, HuggingFace adapter, TokenizerLoader
  TinyBrainDemo/       — SwiftUI ChatView, ChatViewModel, X-Ray visualizations
Examples/
  ChatDemo/            — Executable entry point for the demo app
Tests/                 — 195 tests across all modules
Scripts/               — Python model converter
```

## Key Conventions
- Swift 5.10+, iOS 17, macOS 14
- TDD: write tests first, then implement
- Protocol-oriented design for pluggable components
- Metal ops always have CPU fallbacks
- Never commit without explicit permission
- DocC-compatible comments on public APIs

## Model Files
- Live in `Models/` (gitignored)
- TBF format (TinyBrain Format) — see `docs/tbf-format-spec.md`
- Convert with: `python Scripts/convert_model.py`

## Key Architecture: X-Ray Mode
- `InferenceObserver` protocol in `Sources/TinyBrainRuntime/InferenceObserver.swift`
- `ModelRunner.observer` property — weak ref, zero cost when nil
- 3 hooks: `didComputeAttention`, `didEnterLayer`, `didComputeLogits`
- `XRayViewModel` accumulates observations → publishes `XRaySnapshot` to SwiftUI
- Visualizations in `Sources/TinyBrainDemo/Views/XRay/`
