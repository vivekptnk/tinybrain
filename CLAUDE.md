# CLAUDE.md — TinyBrain Project Context

## What is TinyBrain?
Swift-native on-device LLM inference runtime for Apple Silicon. Educational + practical. The "micrograd of on-device inference."

## Quick Reference
- **PRD:** `docs/prd.md`
- **Task Roadmap:** `docs/TASK-ROADMAP.md`
- **Task Specs:** `docs/tasks/TB-*.md`
- **Architecture rules:** `AGENTS.md`

## Project Status
- TB-001 through TB-009: ALL COMPLETE
- 251 tests passing (244 Swift + 7 Python)
- v0.1.0 ready for release tag
- TinyLlama 1.1B running with real language output

## Build & Test
```bash
swift build          # Build all targets
swift test           # Run all tests
swift run ChatDemo   # Run the chat demo (needs model file)
swift run tinybrain-bench  # Run benchmarks
```

## Module Structure
```
Sources/
  TinyBrainRuntime/    — Tensor, ops, ModelRunner, KV-cache, quantization, streaming
  TinyBrainMetal/      — Metal GPU backend, kernels, buffer pool
  TinyBrainTokenizer/  — BPE tokenizer, HuggingFace adapter, TokenizerLoader
  TinyBrainDemo/       — SwiftUI ChatView, ChatViewModel
Examples/
  ChatDemo/            — Executable entry point for the demo app
Tests/
  TinyBrainRuntimeTests/
  TinyBrainMetalTests/
  TinyBrainTokenizerTests/
  TinyBrainDemoTests/
Scripts/               — Python model converter, diagnostic scripts
docs/                  — PRD, task specs, completion reports
```

## Key Conventions
- Swift 5.10+, iOS 17, macOS 14
- TDD: write tests first, then implement
- Semantic commits: `feat/runtime`, `core/metal`, `ui/demo`
- Protocol-oriented design for pluggable components
- Metal ops always have CPU fallbacks
- Never commit without explicit permission
- DocC-compatible comments on public APIs

## Model Files
- Live in `Models/` (gitignored)
- TBF format (TinyBrain Format) — see `docs/tbf-format-spec.md`
- Convert with: `python Scripts/convert_model.py`

## Current Focus
- TB-010: X-Ray Mode — live transformer visualization overlay
- Differentiator feature for public launch
