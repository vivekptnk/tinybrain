# TinyBrain

**Swift-Native On-Device LLM Inference Kit**

[![Swift](https://img.shields.io/badge/Swift-5.10+-purple.svg)](https://swift.org)
[![Platform](https://img.shields.io/badge/platform-iOS%2017%2B%20%7C%20macOS%2014%2B-lightgrey.svg)](https://developer.apple.com)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-195%20passing-brightgreen.svg)]()

TinyBrain is a **Swift-native runtime** for running large language models entirely on-device on iOS and macOS. It combines **educational transparency** with **practical performance**, making transformer inference hackable and efficient on Apple Silicon.

> *What [micrograd](https://github.com/karpathy/micrograd) did for understanding backprop, TinyBrain does for understanding on-device LLM inference.*

---

## Why TinyBrain?

| | TinyBrain | llama.cpp | MLC-LLM | Core ML |
|---|---|---|---|---|
| **Language** | Swift + Metal | C/C++ | C++ + TVM | Obj-C API |
| **Educational** | Yes | No | No | No |
| **Hackable internals** | Yes | Limited | No | Black box |
| **X-Ray Mode** | **Yes** | No | No | No |
| **iOS integration** | Native SwiftUI | Wrapper | Wrapper | Native |

**No C++ dependencies.** No TVM compiler stack. No black-box ANE scheduling. Just Swift, Metal, and code you can read.

---

## X-Ray Mode

TinyBrain's standout feature: **real-time visualization of transformer internals** during inference. No other on-device LLM tool has this.

- **Attention Heatmap** — See which past tokens the model attends to at each layer
- **Token Probability Bars** — Top candidates with scores, updated per token
- **Layer Activation Flow** — Hidden state magnitude across transformer layers
- **KV Cache Grid** — Page allocation status showing memory usage
- **Entropy Indicator** — Model confidence with plain-English explanations

Zero performance impact when disabled. Built with SwiftUI `Canvas` for high-performance rendering.

---

## Features

- **Pure Swift + Metal** — No C/C++ dependencies
- **INT8 Quantization** — 75% memory savings, <1% accuracy loss
- **Metal GPU Kernels** — Tiled matmul, fused INT8 dequant for Apple Silicon
- **Paged KV Cache** — 2048-token context with O(n) inference
- **Streaming Output** — AsyncSequence with probability, entropy, and timing metadata
- **BPE Tokenizer** — Unicode-normalized byte-pair encoding with HuggingFace adapter
- **Advanced Sampling** — Temperature, top-K, top-P, repetition penalty
- **X-Ray Mode** — Live attention, logits, and activation visualization
- **InferenceObserver** — Zero-cost protocol to instrument the inference pipeline
- **SwiftUI Demo** — Chat app with telemetry sidebar and X-Ray panel
- **Benchmark CLI** — YAML scenarios with JSON/Markdown output

---

## Quick Start

### Prerequisites

- macOS 14+ / iOS 17+
- Xcode 16+
- Swift 5.10+
- Apple Silicon recommended (M1/M2/M3/M4)

### Build & Test

```bash
git clone https://github.com/vivekp/tinybrain.git
cd tinybrain
swift build
swift test
```

### Run the Demo

```bash
# Command line
swift run tinybrain-chat

# Xcode (recommended for full UI + X-Ray)
open Package.swift
# Select ChatDemo scheme → Run
```

---

## Load Any HuggingFace Model

```bash
# 1. Download
huggingface-cli download TinyLlama/TinyLlama-1.1B-Chat-v1.0 --local-dir Models/tinyllama-raw

# 2. Convert to TinyBrain format
python Scripts/convert_model.py \
  --input Models/tinyllama-raw/model.safetensors \
  --output Models/tinyllama-1.1b-int8.tbf \
  --auto-config

# 3. Run
swift run tinybrain-chat
```

Works with TinyLlama, Llama-2/3, Phi, Gemma, and any model with a standard HuggingFace `tokenizer.json`.

---

## Usage

### Inference

```swift
import TinyBrain

let config = ModelConfig(numLayers: 6, hiddenDim: 768, numHeads: 12, vocabSize: 32000)
let weights = ModelWeights.makeToyModel(config: config)
let runner = ModelRunner(weights: weights)

let genConfig = GenerationConfig(
    maxTokens: 100,
    sampler: SamplerConfig(temperature: 0.7, topK: 40),
    stopTokens: []
)

for try await output in runner.generateStream(prompt: tokenIds, config: genConfig) {
    print("Token \(output.tokenId), prob: \(output.probability)")
}
```

### X-Ray: Observe Transformer Internals

```swift
class MyObserver: InferenceObserver {
    func didComputeAttention(layerIndex: Int, weights: [Float], position: Int) {
        // weights[i] = how much attention position `position` pays to position `i`
    }

    func didEnterLayer(layerIndex: Int, hiddenStateNorm: Float, position: Int) {
        // Track signal magnitude through the network
    }

    func didComputeLogits(logits: [Float], position: Int) {
        // Full output distribution before sampling
    }
}

runner.observer = MyObserver()  // Attach — callbacks fire per token
runner.observer = nil           // Detach — zero overhead
```

---

## Architecture

```
┌─────────────────────────────────────┐
│     TinyBrain Chat (SwiftUI)        │
│  ┌──────────┐  ┌─────────────────┐  │
│  │ Chat View │  │  X-Ray Panel    │  │
│  │           │  │  • Attention    │  │
│  │           │  │  • Probabilities│  │
│  │           │  │  • Activations  │  │
│  │           │  │  • KV Cache     │  │
│  └──────────┘  └─────────────────┘  │
└────────────┬────────────────────────┘
             │
┌────────────▼────────────────────────┐
│          Runtime Layer              │
│  ┌──────────┬──────────┬─────────┐  │
│  │Tokenizer │ Sampler  │Observer │  │
│  └──────────┴──────────┴─────────┘  │
│  ┌───────────────────────────────┐  │
│  │        ModelRunner            │  │
│  │  Attention → FFN → KV-Cache   │  │
│  └───────────────────────────────┘  │
└────────────┬────────────────────────┘
             │
┌────────────▼────────────────────────┐
│         Backend Layer               │
│  ┌──────────┐  ┌─────────────────┐  │
│  │  Metal   │  │  CPU Fallback   │  │
│  │ Kernels  │  │  (Accelerate)   │  │
│  └──────────┘  └─────────────────┘  │
└─────────────────────────────────────┘
```

### Modules

| Module | Purpose |
|--------|---------|
| `TinyBrainRuntime` | Tensor engine, ModelRunner, KV cache, quantization, sampler, observer |
| `TinyBrainMetal` | GPU kernels (tiled matmul, INT8 dequant), buffer pool |
| `TinyBrainTokenizer` | BPE tokenizer, HuggingFace adapter, format-agnostic loading |
| `TinyBrainDemo` | SwiftUI chat app, X-Ray visualizations, telemetry |
| `TinyBrainBench` | CLI benchmark tool with YAML scenarios |

---

## Performance

Measured on **MacBook Pro M4 Max** (40 GPU cores):

| Metric | Value |
|--------|-------|
| MatMul 1536x1536 (Metal) | 4.73ms |
| Buffer pool allocation | 0.001ms (450x vs raw alloc) |
| KV cache append | 0.41ms/token |
| Max context | 2048 tokens (paged) |
| TinyLlama 1.1B memory (INT8) | 1.1 GB (75% savings vs FP32) |
| Quantization accuracy loss | <1% |
| Test suite | 195 tests, all passing |

---

## Project Structure

```
tinybrain/
├── Sources/
│   ├── TinyBrainRuntime/       # Tensor, ModelRunner, KV cache, quantization
│   ├── TinyBrainMetal/         # Metal GPU backend
│   ├── TinyBrainTokenizer/     # BPE + HuggingFace adapter
│   ├── TinyBrainDemo/          # SwiftUI app + X-Ray views
│   └── TinyBrainBench/         # Benchmark CLI
├── Examples/ChatDemo/          # App entry point
├── Tests/                      # 195 tests
├── Scripts/                    # Model converter
├── Models/                     # Model files (gitignored)
└── docs/                       # Architecture docs
```

---

## Contributing

1. Read [AGENTS.md](AGENTS.md) for project conventions
2. Follow TDD — write tests first
3. Run `swift test` before submitting PRs

---

## Related Projects

- [llama.cpp](https://github.com/ggerganov/llama.cpp) — C/C++ LLM inference
- [MLC-LLM](https://github.com/mlc-ai/mlc-llm) — TVM-based mobile runtime
- [Core ML Tools](https://github.com/apple/coremltools) — Apple's ML toolkit

TinyBrain is different: **Swift-native, educational, and you can see inside the transformer while it thinks.**

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

**Author:** Vivek Pattanaik
