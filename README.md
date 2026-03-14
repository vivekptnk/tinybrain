<p align="center">
  <h1 align="center">TinyBrain</h1>
  <p align="center">
    <strong>See inside an AI's mind as it thinks.</strong>
    <br />
    Swift-native LLM inference for Apple Silicon — with live visualization.
  </p>
  <p align="center">
    <a href="https://swift.org"><img src="https://img.shields.io/badge/Swift-5.10+-F05138?logo=swift&logoColor=white" alt="Swift" /></a>
    <a href="https://developer.apple.com"><img src="https://img.shields.io/badge/Apple_Silicon-M1_M2_M3_M4-000000?logo=apple&logoColor=white" alt="Apple Silicon" /></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License" /></a>
    <img src="https://img.shields.io/badge/tests-195_passing-brightgreen.svg" alt="Tests" />
  </p>
</p>

---

TinyBrain runs large language models **entirely on your device** — no server, no API key, no internet. It's written in pure Swift with Metal GPU acceleration, and it's the only tool that lets you **watch the transformer think in real-time**.

> *What [micrograd](https://github.com/karpathy/micrograd) did for understanding backprop, TinyBrain does for on-device LLM inference.*

---

## X-Ray Mode

The feature no other LLM runtime has: **live visualization of what's happening inside the transformer** as it generates each token.

| Visualization | What It Shows |
|---------------|---------------|
| **Attention Heatmap** | Which past words the model is looking at right now |
| **Token Probabilities** | The model's top guesses for the next word, with confidence scores |
| **Layer Activations** | How the signal changes as it flows through each transformer layer |
| **KV Cache** | Memory page usage — see the cache fill up in real-time |
| **Entropy Meter** | How confident or uncertain the model is, in plain English |

Toggle it on with one button. Zero performance impact when off.

---

## Get Started (5 minutes)

### What You Need

- A Mac with Apple Silicon (M1 or newer)
- macOS 14 or later
- [Xcode](https://developer.apple.com/xcode/) 16 or later (free from the App Store)
- Python 3.11+ (for model conversion only)

### Step 1: Clone and Build

Open Terminal and run:

```bash
git clone https://github.com/vivekptnk/tinybrain.git
cd tinybrain
swift build
```

That's it. No `npm install`, no `pip install`, no Docker. Just `swift build`.

### Step 2: Run the Demo

```bash
swift run tinybrain-chat
```

This runs with a built-in toy model so you can see the UI immediately. The output won't be real language (it's random weights), but you'll see the full inference pipeline working — streaming tokens, telemetry, and X-Ray Mode.

### Step 3: Run with a Real Model (Optional)

To get actual language output, you need a real model. Here's how:

```bash
# Install Python dependencies (one time)
pip install torch safetensors

# Download TinyLlama (1.1B parameters, ~2GB download)
huggingface-cli download TinyLlama/TinyLlama-1.1B-Chat-v1.0 --local-dir Models/tinyllama-raw

# Convert to TinyBrain format (~30 seconds)
python Scripts/convert_model.py \
  --input Models/tinyllama-raw/model.safetensors \
  --output Models/tinyllama-1.1b-int8.tbf \
  --auto-config

# Run with the real model
swift run tinybrain-chat
```

### Step 4: Open in Xcode (Best Experience)

For the full GUI with X-Ray Mode:

```bash
open Package.swift
```

This opens the project in Xcode. Select the **ChatDemo** scheme from the dropdown, then hit **Run** (or press `Cmd+R`).

> **macOS Tahoe note:** If the text field doesn't accept input, go to Edit Scheme > Run > Options and uncheck "Use the sandbox". This is a known macOS 15 SPM bug — the demo provides preset buttons as a workaround.

---

## Use TinyBrain in Your Own App

Add TinyBrain to your Swift project:

```swift
// Package.swift
dependencies: [
    .package(url: "https://github.com/vivekptnk/tinybrain.git", from: "0.1.0")
]
```

### Run Inference

```swift
import TinyBrain

// Load a model
let weights = try ModelWeights.load(from: "path/to/model.tbf")
let runner = ModelRunner(weights: weights)

// Generate tokens
let config = GenerationConfig(
    maxTokens: 100,
    sampler: SamplerConfig(temperature: 0.7, topK: 40),
    stopTokens: []
)

for try await output in runner.generateStream(prompt: tokenIds, config: config) {
    let text = tokenizer.decode([output.tokenId])
    print(text, terminator: "")
}
```

### Watch the Model Think (X-Ray API)

```swift
class MyObserver: InferenceObserver {
    func didComputeAttention(layerIndex: Int, weights: [Float], position: Int) {
        // `weights` shows how much attention each past token gets
        // weights[0] = attention to first token, weights[1] = second, etc.
        // They sum to 1.0 (it's a probability distribution)
    }

    func didEnterLayer(layerIndex: Int, hiddenStateNorm: Float, position: Int) {
        // Track how the signal magnitude changes through the network
    }

    func didComputeLogits(logits: [Float], position: Int) {
        // The raw output scores before sampling
        // logits.count == vocabulary size (e.g., 32000 for Llama)
    }
}

// Attach — zero cost when nil
runner.observer = MyObserver()

// Detach when done
runner.observer = nil
```

---

## How It Works

```
You type: "What is gravity?"
         |
         v
  ┌─────────────────┐
  │    Tokenizer     │   Converts text to numbers
  │  "What" → 1024   │   using byte-pair encoding
  └────────┬─────────┘
           v
  ┌─────────────────┐
  │   ModelRunner    │   Runs the transformer:
  │                  │   1. Look up token embedding
  │  For each layer: │   2. Compute attention (Q×K^T)
  │   • Attention    │   3. Cache keys/values (KV cache)
  │   • Feed-forward │   4. Apply feed-forward network
  │   • Residual add │   5. Repeat for all layers
  └────────┬─────────┘
           v
  ┌─────────────────┐
  │    Sampler       │   Picks the next token:
  │  temperature=0.7 │   • Apply temperature scaling
  │  topK=40         │   • Filter to top-K candidates
  │  → token 4821    │   • Sample from distribution
  └────────┬─────────┘
           v
  ┌─────────────────┐
  │   Tokenizer      │   Converts back to text
  │  4821 → "force"  │   and streams to the UI
  └──────────────────┘
```

All of this happens **on your device**, using Metal GPU acceleration. No internet required.

---

## Architecture

```
┌──────────────────────────────────────┐
│      TinyBrain Chat (SwiftUI)        │
│  ┌───────────┐  ┌──────────────────┐ │
│  │ Chat View  │  │  X-Ray Panel     │ │
│  │            │  │  Attention map   │ │
│  │            │  │  Token probs     │ │
│  │            │  │  Layer norms     │ │
│  │            │  │  KV cache grid   │ │
│  └───────────┘  └──────────────────┘ │
└───────────┬──────────────────────────┘
            v
┌───────────────────────────────────────┐
│          TinyBrainRuntime             │
│  Tokenizer · Sampler · ModelRunner    │
│  KV Cache · Quantization · Observer   │
└───────────┬───────────────────────────┘
            v
┌───────────────────────────────────────┐
│          TinyBrainMetal               │
│  GPU Kernels · Buffer Pool            │
│  (Falls back to CPU via Accelerate)   │
└───────────────────────────────────────┘
```

| Module | What It Does |
|--------|-------------|
| `TinyBrainRuntime` | The core engine: tensors, model runner, KV cache, quantization, sampling |
| `TinyBrainMetal` | GPU acceleration via Metal shaders (with automatic CPU fallback) |
| `TinyBrainTokenizer` | Converts text to/from token IDs (supports HuggingFace format) |
| `TinyBrainDemo` | The SwiftUI chat app with X-Ray visualizations |
| `TinyBrainBench` | Command-line benchmarking tool |

---

## Performance

Measured on MacBook Pro M4 Max:

| What | How Fast |
|------|----------|
| Matrix multiply (1536x1536, Metal) | 4.73ms |
| GPU buffer allocation (pooled) | 0.001ms (450x faster than raw) |
| KV cache append per token | 0.41ms |
| Maximum context length | 2048 tokens |
| TinyLlama 1.1B memory (INT8) | 1.1 GB (75% less than FP32) |
| Quantization accuracy loss | Less than 1% |

---

## Supported Models

TinyBrain works with any HuggingFace model that has a `tokenizer.json`:

| Model | Parameters | Status |
|-------|-----------|--------|
| TinyLlama | 1.1B | Tested and working |
| Llama 2/3 | Various | Compatible (same format) |
| Phi (Microsoft) | Various | Compatible |
| Gemma (Google) | Various | Compatible |

To add a new model, just download it and run the converter script. No code changes needed.

---

## Project Structure

```
tinybrain/
├── Sources/
│   ├── TinyBrainRuntime/       # Core engine
│   ├── TinyBrainMetal/         # GPU backend
│   ├── TinyBrainTokenizer/     # Tokenization
│   ├── TinyBrainDemo/          # SwiftUI app + X-Ray views
│   └── TinyBrainBench/         # Benchmarks
├── Examples/ChatDemo/          # App entry point
├── Tests/                      # 195 tests
├── Scripts/                    # Model converter (Python)
└── docs/                       # Architecture documentation
```

---

## Run the Tests

```bash
swift test --skip TinyBrainDemoTests
```

> The `--skip TinyBrainDemoTests` is needed because of a known Xcode beta linker issue with SwiftUI test targets. All 195 other tests pass.

---

## Why Not Just Use llama.cpp?

You absolutely can. llama.cpp is great for raw performance.

TinyBrain is for when you want to:
- **Learn** how transformers actually work (the code is readable Swift, not optimized C++)
- **See** what the model is doing in real-time (X-Ray Mode)
- **Build** native iOS/macOS apps without bridging C++ (pure Swift + SwiftUI)
- **Hack** on the internals (swap attention mechanisms, try different samplers, add your own visualizations)

---

## Contributing

PRs welcome. Please:
1. Read [AGENTS.md](AGENTS.md) for conventions
2. Write tests first (TDD)
3. Run `swift test --skip TinyBrainDemoTests` before submitting

---

## License

MIT — see [LICENSE](LICENSE)

**Author:** [Vivek Pattanaik](https://github.com/vivekptnk)
