# Changelog

All notable changes to TinyBrain are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased — 0.2.0-dev]

Status: unreleased as of 2026-07-03.

Development release notes for the current v0.2.0 work. This section is not a tagged release.

### Added

#### Quantization and Metal Kernels
- INT4 per-group quantization with group size 32 and FP16 scales.
- Fused Metal INT4 dequantization plus matrix multiplication kernel.
- FlashAttention Metal kernel using tiled attention and online softmax.
- TBF loader fix for INT4 packed tensors.

#### Generation Runtime
- Speculative decoding with draft and verify passes.
- Tool-calling primitives (constrained JSON generation: ConstrainedSampler, tool schemas; generation-loop wiring tracked for the agent milestone).
- Gemma RMSNorm variant using `(1 + w) * x`.
- Gemma 2B and Phi-2 architecture support.
- On-device RAG module and `tinybrain-rag` demo (TB-012) (commit messages predate the renumbering).

#### Bridges and Demo App
- TinyBrainProximaKit bridge with `TextEmbedder`.
- TinyBrainCartographerBridge with `SmartAnnotationService` adapter.
- Model picker in ChatDemo for switching local `.tbf` files.

#### Benchmark Harness
- `tinybrain-bench` scenario system for scripted benchmark runs.

### Fixed

- HuggingFace tokenizer parity for `byte_fallback` and special-token pre-splitting.
- Gemma coherence via exact-GELU gate behavior and INT8 output matching the HuggingFace reference.
- INT4 converter transpose handling, RoPE rotate-half convention, and speculative-decoding exact-distribution correction.

### Known limitations

- Naive group=32 RTN INT4 with the output head kept INT8 measured +8.152% perplexity vs INT8 on Gemma 2B (INT8 ppl 7.89913, INT4 ppl 8.543102) and +19.241% on TinyLlama 1.1B (INT8 ppl 9.988422, INT4 ppl 11.910269) on 2026-07-03 local eval slices; the ≤6% and ≤1% calibrated quantization targets are tracked for v0.2.1 GPTQ/AWQ work.

## [0.1.0] — 2025-10-25

Codename: **Foundation**

First public release. Swift-native on-device LLM inference for Apple Silicon with live visualization.

### Added

#### Core Runtime (TB-001, TB-002)
- Generic `Tensor<Element>` supporting Float32, Float16, and Int8.
- Copy-on-Write optimization via `isKnownUniquelyReferenced`.
- Accelerate framework integration (BLAS `cblas_sgemm`, vDSP).
- Operations: MatMul, Softmax, LayerNorm, GELU, element-wise arithmetic.

#### Metal GPU Acceleration (TB-003, TB-004)
- Tiled MatMul kernel with 16x16 threadgroups and shared memory.
- GPU-resident tensors with lazy CPU/GPU synchronization (`toGPU()`, `toCPU()`).
- `MetalBufferPool` — persistent buffer reuse (450x faster allocation).
- Automatic CPU fallback when Metal is unavailable.

#### INT8 Quantization (TB-004)
- Per-channel symmetric INT8 quantization.
- 75% memory reduction (TinyLlama 1.1B: 4.4 GB to 1.1 GB).
- Less than 1% accuracy loss vs Float32.
- `QuantizedTensor` with `quantize()` and `dequantize()` methods.

#### Paged KV-Cache (TB-004)
- 2048-token context window with 16-token pages.
- `PageAllocator` with free-list page management.
- Zero-allocation inference loop (pre-allocated pages).
- Thread-safe concurrent access via `NSLock`.

#### Tokenization (TB-005, TB-009)
- Full BPE tokenizer with Unicode NFC normalization.
- Special tokens: BOS, EOS, UNK, PAD.
- Multilingual support (accented characters, emoji).
- Format-agnostic `TokenizerLoader` with auto-detection.
- `HuggingFaceAdapter` — loads `tokenizer.json` from any HuggingFace model.

#### Sampling (TB-005)
- Five strategies: Greedy, Temperature, Top-K, Top-P (Nucleus), Repetition Penalty.
- `SamplerConfig` for runtime configuration.
- Deterministic seeding for reproducibility.

#### Streaming Runtime (TB-004, TB-005)
- `ModelRunner` with `generateStream()` returning `AsyncSequence`.
- `TokenOutput` with token ID, probability, and timestamp metadata.
- Configurable `GenerationConfig` with stop token support.
- Sub-10ms per-token latency (target was < 150ms).

#### SwiftUI Demo App (TB-006)
- TinyBrain Chat with MVVM architecture.
- Real-time token streaming with typewriter animation.
- Telemetry display: tokens/sec, memory, probability.
- Design system: theme, animations, platform-adaptive UI.
- Message history with copy functionality.

#### X-Ray Mode (TB-010)
- `InferenceObserver` protocol with zero-cost observation hooks.
- Live attention heatmap visualization.
- Token probability distribution display.
- Layer activation tracking.
- KV-cache usage grid.
- Entropy meter with confidence labels.

#### Benchmark Harness (TB-007)
- CLI tool (`tinybrain-bench`) with YAML scenario loading.
- JSON and Markdown output formats.
- Memory tracking and device info reporting.
- Warmup iterations and regression detection.

#### Model Conversion (TB-007)
- Python converter: PyTorch/SafeTensors to TBF format.
- INT8 quantization pipeline.
- Auto-configuration from model metadata.
- BFloat16 handling.

#### Documentation
- Architecture overview (`docs/overview.md`).
- TBF binary format specification (`docs/tbf-format-spec.md`).
- Metal debugging guide (`docs/Metal-Debugging-Guide.md`).
- Benchmarking guide (`docs/benchmarking.md`).
- FAQ and troubleshooting (`docs/faq.md`).
- DocC articles for Tokenization and Sampling.
- 6 Architecture Decision Records (`docs/adr/`).

### Known Issues

- **macOS Tahoe TextField:** SwiftUI TextField doesn't accept keyboard input in SPM executables on macOS 15.x. Workaround: disable sandbox in Xcode scheme settings. iOS is not affected.
- **AMX vs GPU:** M4 Max CPU (AMX) competitive with or beats GPU for single matmul operations. GPU wins for batched workflows. See ADR-002.
- **Metal dequant kernel:** INT8 weights dequantized on CPU. Metal kernel deferred to v0.2.0.

### Test Coverage

- 195 Swift tests passing across all modules.
- 7 Python tests passing for model converter.
- TDD methodology used throughout.

[0.1.0]: https://github.com/vivekptnk/tinybrain/releases/tag/v0.1.0
