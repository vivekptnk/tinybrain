# 🧠 TinyBrain

**Swift-Native On-Device LLM Inference Kit**

[![Swift](https://img.shields.io/badge/Swift-5.10+-purple.svg)](https://swift.org)
[![Platform](https://img.shields.io/badge/platform-iOS%2017%2B%20%7C%20macOS%2014%2B-lightgrey.svg)](https://developer.apple.com)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![CI](https://github.com/vivekptnk/tinybrain/workflows/CI/badge.svg)](https://github.com/vivekptnk/tinybrain/actions)

TinyBrain is a **Swift-native runtime** for running large language models (LLMs) entirely on-device on iOS and macOS. It combines **educational transparency** with **practical performance**, making transformer inference hackable and efficient on Apple Silicon.

---

## ✨ Features

- 🚀 **Swift-First**: No C++ dependencies, pure Swift + Metal
- 🧮 **Quantization**: INT8 support (75% memory savings) ✅ **TB-004 Complete**
- ⚡ **Metal Acceleration**: GPU-optimized kernels for Apple Silicon ✅ **Validated on M4 Max**
- 🔄 **Streaming Output**: AsyncSequence-based token generation ✅ **TB-004 Complete**
- 💾 **KV Cache**: Paged 2048-token context for efficient inference ✅ **TB-004 Complete**
- 🌐 **Format-Agnostic**: Load ANY HuggingFace model automatically ✅ **TB-009 Complete**
- 🎯 **Studio-Ready**: One-command model conversion and deployment ✅ **TB-007/008/009 Complete**
- 🎓 **Educational**: Transparent, well-documented architecture
- 📱 **Native**: Deep iOS/macOS integration with SwiftUI demo
- 🔒 **Private**: 100% on-device inference, no network calls

---

## 🎯 Goals

TinyBrain serves two main purposes:

1. **Educational**: Teach developers how LLMs work at the tensor level
2. **Practical**: Enable real-time, private, offline inference on Apple devices

---

## 🚀 Quick Start

### Prerequisites

- macOS 14 Sonoma or later
- Xcode 16+
- Swift 5.10+
- Apple Silicon Mac (M1/M2/M3/M4) recommended

### Installation

```bash
git clone https://github.com/vivekp/tinybrain.git
cd tinybrain
make setup
```

This will:
- Install SwiftFormat and SwiftLint (via Homebrew if available)
- Resolve Swift Package dependencies
- Build the project
- Run initial tests

### Building

**Two workflows:** SPM command-line OR Xcode IDE

#### Option 1: Command Line (SPM)

```bash
# Build all targets
make build

# Build in release mode
make build-release

# Run tests
make test

# Format code
make format

# Run linting
make lint

# Generate documentation
make docs
```

#### Option 2: Xcode IDE (Recommended for Demo App)

```bash
# Open in Xcode
open Package.swift
```

**Note for macOS Tahoe users:** The ChatDemo app requires proper app bundle configuration to enable TextField input. When running in Xcode:
1. Select the `ChatDemo` scheme
2. Edit Scheme → Run → Options
3. Uncheck "Use the sandbox" (or run on a real iOS device where this isn't an issue)

This is a known limitation of SPM executables on macOS 15.x. The `Info.plist` is included for future app bundle support.

---

## 🎯 Model Studio Workflow

TinyBrain now supports loading ANY HuggingFace transformer model with a simple 3-step process:

### 1. Download Model
```bash
# Download any HuggingFace model
hf download TinyLlama/TinyLlama-1.1B-Chat-v1.0
# Or: git lfs clone https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

### 2. Convert Weights
```bash
# Convert to TinyBrain format (one command!)
python Scripts/convert_model.py \
  --input Models/tinyllama-raw/model.safetensors \
  --output Models/tinyllama-1.1b-int8.tbf \
  --auto-config
```

### 3. Run App
```bash
# Open in Xcode (recommended)
open Package.swift

# Or run from command line
swift run ChatDemo
```

**Result:** Real language output with 32K vocabulary! 🚀

### Supported Models
- ✅ **TinyLlama-1.1B** (tested and working)
- ✅ **Llama-2/3** (same tokenizer format)
- ✅ **Phi models** (Microsoft)
- ✅ **Gemma** (Google)
- ✅ **Any HuggingFace model** with standard tokenizer.json

---

## 📦 Usage

### Basic Inference

```swift
import TinyBrain

// Enable GPU acceleration
TinyBrainBackend.enableMetal()

// Create quantized weights (toy generator or load from disk)
let config = ModelConfig(
    numLayers: 6,
    hiddenDim: 768,
    numHeads: 12,
    vocabSize: 32000,
    maxSeqLen: 2048
)
let weights = ModelWeights.makeToyModel(config: config)
let runner = ModelRunner(weights: weights)

// Stream tokens with KV cache reuse
let promptTokens = [1, 2, 3]  // Tokenized prompt
for try await tokenId in runner.generateStream(prompt: promptTokens, maxTokens: 100) {
    print(tokenId, terminator: " ")  // Progressive output!
}
```

### Quantized Model Loading

```swift
// Load Float32 weights
let weights = Tensor<Float>.random(shape: TensorShape(768, 3072))

// Quantize to INT8 (75% memory savings!)
let quantized = weights.quantize(mode: .perChannel)
print("Savings: \(quantized.savingsVsFloat32() * 100)%")  // ~75%

// Use quantized weights for inference
let input = Tensor<Float>.random(shape: TensorShape(128, 768))
let output = input.matmul(quantized)  // Auto-dequantizes
```

### SwiftUI Demo

A complete chat demo app is available in `Examples/ChatDemo/`:

```bash
open Examples/ChatDemo/ChatDemoApp.swift
# Run in Xcode to see live inference
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────┐
│   TinyBrain Chat (SwiftUI)      │
└────────────┬────────────────────┘
             │
┌────────────▼────────────────────┐
│      Runtime Layer              │
│  ┌──────────┬─────────────┐    │
│  │Tokenizer │   Sampler   │    │
│  └──────────┴─────────────┘    │
│  ┌─────────────────────────┐   │
│  │     ModelRunner         │   │
│  │  ┌──────────────────┐   │   │
│  │  │ Attention / MLP  │   │   │
│  │  │    KV-Cache      │   │   │
│  │  └──────────────────┘   │   │
│  └─────────────────────────┘   │
└────────────┬────────────────────┘
             │
┌────────────▼────────────────────┐
│      Backend Layer              │
│  ┌─────────┬──────────────┐    │
│  │  Metal  │  Core ML     │    │
│  │ Kernels │  (Optional)  │    │
│  └─────────┴──────────────┘    │
└─────────────────────────────────┘
```

### Modules

- **TinyBrainRuntime**: Core tensor operations and model runner
- **TinyBrainMetal**: GPU-accelerated kernels via Metal
- **TinyBrainTokenizer**: BPE/SentencePiece tokenization
- **TinyBrainDemo**: SwiftUI demo application
- **TinyBrainBench**: Performance benchmarking tools

---

## 📚 Documentation

Comprehensive documentation is available:

- **[docs/prd.md](docs/prd.md)**: Product Requirements Document
- **[docs/overview.md](docs/overview.md)**: Architecture overview
- **[docs/tasks/](docs/tasks/)**: Implementation roadmap
- **API Docs**: Generate with `make docs`

---

## 🧪 Testing

```bash
# Run all tests
make test

# Run with coverage
make test-coverage

# Run linting and formatting checks
make check
```

---

## 🛠️ Development

### Project Structure

```
tinybrain/
├── Sources/
│   ├── TinyBrainRuntime/     # Core runtime
│   ├── TinyBrainMetal/       # Metal backend
│   ├── TinyBrainTokenizer/   # Tokenization
│   ├── TinyBrainDemo/        # Demo app components
│   └── TinyBrainBench/       # Benchmark CLI
├── Tests/                     # Unit tests
├── Examples/
│   └── ChatDemo/             # SwiftUI demo app
├── Scripts/                   # Build and setup scripts
├── Models/                    # Model files (gitignored)
├── docs/                      # Documentation
└── Package.swift             # SPM manifest
```

### Coding Standards

- Swift 5.10+ with strict concurrency checking
- SwiftFormat for code formatting (`.swiftformat`)
- SwiftLint for static analysis (`.swiftlint.yml`)
- DocC-compatible documentation comments
- 100% test coverage target for critical paths

---

## 🗺️ Roadmap

| Phase | Status | Deliverables |
|-------|--------|--------------|
| **TB-001: Scaffold** | ✅ Complete | Project structure, tooling, docs |
| **TB-002: Runtime** | ✅ Complete | Tensor engine (Float32, Accelerate) |
| **TB-003: Metal** | ✅ Complete | GPU MatMul kernel, buffer pool |
| **TB-004: Quant/KV** | ✅ **COMPLETE** | **INT8 quantization, paged KV cache, streaming API** |
| **TB-005: Tokenizer** | 📋 Planned | BPE tokenizer, advanced sampling |
| **TB-006: Demo** | 📋 Planned | SwiftUI chat app with live inference |
| **TB-007: Benchmarks** | 📋 Planned | Performance suite, energy metrics |

### TB-004 Highlights (Just Completed!)

- ✅ **GPU-resident tensors** (0.74-1.28× vs M4 AMX)
- ✅ **Generic Tensor<Element>** (Float32, Float16, Int8)
- ✅ **Copy-on-Write** optimization
- ✅ **INT8 quantization** (75% memory savings, <1% error)
- ✅ **Paged KV cache** (2048-token context)
- ✅ **Streaming API** (AsyncSequence for SwiftUI)
- ✅ **94 tests passing** on M4 Max

See [docs/TB-004-COMPLETE.md](docs/TB-004-COMPLETE.md) for full details.

---

## 🤝 Contributing

We welcome contributions! Please:

1. Read [AGENTS.md](AGENTS.md) for project rules
2. Check [docs/tasks/](docs/tasks/) for current work items
3. Follow the coding standards (run `make check` before PR)
4. Reference task IDs in commits (e.g., "feat: Implement tensor ops (TB-002)")

---

## 📊 Benchmarks

Performance on **MacBook Pro M4 Max** (40 GPU cores):

### GPU Performance (TB-003/TB-004)

| Matrix Size | CPU (ms) | GPU (ms) | Speedup | Winner |
|-------------|----------|----------|---------|--------|
| 512×512 | 0.43 | 0.84 | 0.51× | CPU (AMX) |
| 1024×1024 | 1.79 | 1.97 | 0.91× | CPU (AMX) |
| **1536×1536** | **6.06** | **4.73** | **1.28×** | **GPU** ✅ |

**Note:** M4 Max has AMX (Apple Matrix Extension) coprocessor that often beats GPU for single matmul. Real wins come from batched workflows!

### Memory Efficiency (TB-004)

| Model | Float32 | INT8 | Savings |
|-------|---------|------|---------|
| TinyLlama 1.1B | 4.4 GB | **1.1 GB** | **75%** ✅ |
| Quantization error | Baseline | 0.7-1.0% | < 1% ✅ |

### KV Cache Performance (TB-004)

| Operation | Time | Notes |
|-----------|------|-------|
| Append 1 token | 0.41 ms | Fast! |
| Append 100 tokens | 4.1 ms | Linear scaling |
| 2048-token context | ✅ Supported | Paged memory |
| Memory leak test | 4.2 sec | 10k cycles, no leaks ✅ |

### Test Coverage

| Metric | Current | Status |
|--------|---------|--------|
| Total tests | **94/94 passing** | ✅ **Complete** |
| TB-004 tests | 57 new tests | ✅ **All passing** |
| Code coverage | Core ops | ✅ **TDD methodology** |

Run benchmarks: `swift test --filter PerformanceBenchmarks`

---

## 🔗 Related Projects

- [MLC-LLM](https://github.com/mlc-ai/mlc-llm): C++/TVM-based mobile LLM runtime
- [llama.cpp](https://github.com/ggerganov/llama.cpp): Pure C/C++ LLM inference
- [Core ML Tools](https://github.com/apple/coremltools): Apple's ML conversion toolkit

**TinyBrain differentiator**: Swift-native, educational, hybrid Metal/Core ML.

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 👨‍💻 Author

**Vivek Pattanaik**  
GitHub: [@vivekptnk](https://github.com/vivekptnk)

---

## 🙏 Acknowledgments

- Inspired by [micrograd](https://github.com/karpathy/micrograd) for educational clarity
- Built on Apple's Metal, Core ML, and Accelerate frameworks
- Community contributions and feedback

---

## 📮 Contact

- Issues: [GitHub Issues](https://github.com/vivekptnk/tinybrain/issues)
- Discussions: [GitHub Discussions](https://github.com/vivekptnk/tinybrain/discussions)
