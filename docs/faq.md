# TinyBrain FAQ

**Version:** 0.1.0  
**Last Updated:** October 25, 2025

---

## General Questions

### What is TinyBrain?

TinyBrain is a Swift-native runtime for running large language models (LLMs) entirely on-device on iOS and macOS. It's designed to be both educational and performant, making transformer inference transparent and efficient on Apple Silicon.

### Why Swift instead of Python/C++?

- **Native**: Deep integration with iOS/macOS ecosystem
- **Educational**: Clean, readable code without C++ complexity
- **Modern**: Swift Concurrency, value semantics, protocol-oriented design
- **Performance**: Accelerate/Metal give competitive performance with less complexity

### What models are supported?

Currently:
- ✅ TinyLlama 1.1B (via PyTorch → TBF converter)
- ✅ Custom models via TBF format

Future:
- Llama 2/3 (7B and below)
- Phi models
- Gemma models

---

## Installation & Setup

### macOS TextField not accepting input?

**Problem:** On macOS 15.x (Tahoe), SwiftUI TextField in SPM executables doesn't accept keyboard input.

**Solution:**
1. Open `Package.swift` in Xcode
2. Select the `ChatDemo` scheme
3. Edit Scheme → Run → Options
4. Uncheck "Use the sandbox"
5. Run the app

**Why:** SPM executables lack proper app bundle identifiers. This is a known macOS limitation.

**iOS:** Not affected - TextField works fine on iOS.

### How do I install Python dependencies?

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Build errors with Metal?

**If you see Metal compilation errors:**
- Ensure macOS 14+ / iOS 17+
- Check Xcode 16+ is installed
- Verify Apple Silicon (M1/M2/M3/M4) or Intel with GPU

**Metal not available:**
- TinyBrain falls back to CPU automatically
- All operations work on CPU (via Accelerate)

---

## Performance

### Why is GPU slower than CPU on my M4 Mac?

**Expected!** M4 Max has AMX (Apple Matrix Extension), a dedicated matrix coprocessor:
- Single matmul: CPU (AMX) wins 0.7-1.3× vs GPU
- Batched operations: GPU wins 2-4× (attention layers)

See `docs/TB-004-M4-FINDINGS.md` for details.

### How do I run benchmarks?

```bash
# Quick benchmark
swift run -c release tinybrain-bench --demo --tokens 50

# YAML scenarios
swift run -c release tinybrain-bench \
  --scenario benchmarks/scenarios.yml \
  --output json

# Automated script
./Scripts/run_benchmarks.sh
```

### What are the memory requirements?

| Model | Float32 | INT8 (Quantized) | Savings |
|-------|---------|------------------|---------|
| TinyLlama 1.1B | 4.4 GB | **1.1 GB** | **75%** |

**Minimum:** 2 GB RAM recommended for INT8 models

### How accurate is INT8 quantization?

**Very accurate:** <1% perplexity delta vs Float32
- See `Tests/TinyBrainRuntimeTests/QualityRegressionTests.swift`
- BLEU score: 0.92 (92% similarity)
- Per-channel symmetric quantization

---

## Model Conversion

### How do I convert a PyTorch model?

```bash
source .venv/bin/activate

python Scripts/convert_model.py \
  --input model.pt \
  --output Models/model-int8.tbf \
  --quantize int8 \
  --auto-config
```

### What checkpoint formats are supported?

- ✅ PyTorch (`.pt`)
- ✅ SafeTensors (`.safetensors`)
- ❌ GGUF (use converter tools first)

### Where do I get TinyLlama weights?

```bash
# Download from HuggingFace
huggingface-cli download TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T

# Convert to TBF
python Scripts/convert_model.py \
  --input pytorch_model.bin \
  --output Models/tinyllama-1.1b-int8.tbf \
  --auto-config
```

### How do I convert Gemma 2B to INT4 TBF? (CHA-109)

Requires accepting the Google Gemma license on HuggingFace first.

```bash
# 1. Download weights (requires HF login + license acceptance)
huggingface-cli download google/gemma-2b \
  --local-dir Models/gemma-2b-raw

# 2. Convert to INT8 (needed for the perplexity baseline)
python3 Scripts/convert_model.py \
  --input Models/gemma-2b-raw \
  --output Models/gemma-2b-int8.tbf \
  --quantize int8 \
  --auto-config

# 3. Convert to INT4 (v0.2.0 shipping format, group_size=32)
python3 Scripts/convert_model.py \
  --input Models/gemma-2b-raw \
  --output Models/gemma-2b-int4.tbf \
  --quantize int4 \
  --group-size 32 \
  --auto-config

# 4. Verify: run the smoke test and perplexity gate
swift test --filter GemmaSmokeTest
swift test --filter testGemmaINT4VsINT8Perplexity
```

**Expected output sizes:** INT8 ≈ 4.3 GB, INT4 ≈ 3.5 GB.  
**Perplexity DoD:** `|Δppl|/ppl_INT8 ≤ 0.06` at group=32 (v0.2.0).

---

## Development

### How do I run tests?

```bash
# All tests
swift test

# Specific target
swift test --filter TinyBrainRuntimeTests

# Python converter tests
source .venv/bin/activate
pytest Tests/test_convert_model.py -v
```

### What's the test count?

**Total:** 179 tests (as of TB-007)
- TinyBrainRuntime: 80+ tests
- TinyBrainMetal: 20+ tests
- TinyBrainTokenizer: 20+ tests
- TinyBrainDemo: 53 tests
- Python converter: 7 tests
- Benchmark harness: 9 tests

### How do I contribute?

1. Read `AGENTS.md` for project rules
2. Check `docs/tasks/` for current work items
3. Follow TDD: Write tests first
4. Run `make check` before PR
5. Reference task IDs in commits (e.g., "feat: Add feature (TB-007)")

---

## Troubleshooting

### Build fails with "resource not found"

**Check:**
- Swift 5.10+ installed
- Xcode 16+ command line tools
- Run `swift package resolve`

### Tests fail on CI

**Common causes:**
- No GPU available → Tests should skip Metal-only tests
- iOS Simulator vs Device → Some tests are device-only
- Memory limits → Reduce batch sizes

### "Metal device not found"

**Normal!** Falls back to CPU automatically.
- Check: `tinybrain-bench --device-info`
- Metal requires: macOS 14+/iOS 17+ on Apple Silicon

---

## API Usage

### How do I use TinyBrain in my app?

```swift
import TinyBrain

// Load model
let weights = try ModelWeights.load(from: "model.tbf")
let runner = ModelRunner(weights: weights)

// Generate tokens
let prompt = tokenizer.encode("Hello, world!")
let config = GenerationConfig(maxTokens: 50)

for try await output in runner.generateStream(prompt: prompt, config: config) {
    print(tokenizer.decode([output.tokenId]))
}
```

See `Sources/TinyBrain/TinyBrain.docc/GettingStarted.md` for full tutorial.

### How do I customize sampling?

```swift
let config = SamplerConfig(
    temperature: 0.7,     // Randomness (0.1-2.0)
    topK: 40,             // Limit to top K tokens
    topP: 0.9,            // Nucleus sampling
    repetitionPenalty: 1.2 // Discourage loops
)
```

See `docs/overview.md` Section 6 for details.

---

## Known Issues

### TextField input on macOS Tahoe (Fixed)

**Status:** ✅ Workaround documented  
**See:** Installation & Setup section above

### AMX vs GPU performance

**Status:** ℹ️ Expected behavior  
**See:** Performance section above

### SafeTensors conversion (Untested)

**Status:** ⚠️ Implemented but not fully tested  
**Workaround:** Convert to PyTorch first

---

## Getting Help

- **Issues:** [GitHub Issues](https://github.com/vivekptnk/tinybrain/issues)
- **Discussions:** [GitHub Discussions](https://github.com/vivekptnk/tinybrain/discussions)
- **Documentation:** `docs/overview.md`

---

**Can't find your question?** Open an issue or discussion on GitHub!

