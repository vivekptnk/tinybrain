# Sampling Strategies

Control text generation diversity with advanced sampling techniques.

## Overview

Sampling is how we convert raw model outputs (logits) into actual tokens. The choice of sampling strategy dramatically affects the quality, creativity, and coherence of generated text.

**TinyBrain** provides 5 sampling strategies:
1. **Greedy**: Deterministic, always picks highest probability
2. **Temperature**: Scale randomness
3. **Top-K**: Limit to K best options
4. **Top-P (Nucleus)**: Adaptive probability cutoff
5. **Repetition Penalty**: Discourage loops

## Why Sampling Matters

Without proper sampling, LLMs are either boring or nonsensical:

```
Prompt: "The cat sat on the"

Greedy (temp=0):     → "mat mat mat mat..."  (boring, repetitive)
Temperature (0.7):   → "mat and looked around."  (balanced)
Temperature (2.0):   → "xylophone quantum nebula!" (creative but random)
Top-K (40):          → "mat" or "chair" (quality choices only)
```

## Basic Sampling Strategies

### Greedy Sampling

Always picks the highest probability token:

```swift
import TinyBrainRuntime

let logits = Tensor<Float>(shape: TensorShape(5), data: [0.1, 0.5, 0.2, 0.8, 0.3])
let token = Sampler.greedy(logits: logits)
// → 3 (index of 0.8, always deterministic)
```

**When to use:**
- ✅ Reproducible outputs (testing, demos)
- ✅ Factual question answering
- ❌ Creative writing (too repetitive)

### Temperature Sampling

Scale logits to control randomness:

```swift
// Low temperature (focused)
let token1 = Sampler.temperature(logits: logits, temp: 0.3)
// → Usually picks highest, occasionally explores

// Medium temperature (balanced)
let token2 = Sampler.temperature(logits: logits, temp: 0.7)
// → Good mix of quality and diversity

// High temperature (creative)
let token3 = Sampler.temperature(logits: logits, temp: 1.5)
// → More exploratory, can be surprising
```

**How it works:**
```
scaled_logits = original_logits / temperature
probabilities = softmax(scaled_logits)
sample from probabilities
```

**Temperature scale:**
- `0.0-0.3`: Very focused (near-greedy)
- `0.7-1.0`: Balanced (recommended)
- `1.5-2.0`: Creative/random

### Top-K Sampling

Only sample from K highest logits:

```swift
// Only consider top 40 tokens
let token = Sampler.topK(logits: logits, k: 40, temp: 0.8)
```

**Educational:**
```
Original logits: [mat: 0.8, hat: 0.6, cat: 0.4, moon: 0.01, ...]
Top-K (K=3):     [mat: 0.8, hat: 0.6, cat: 0.4, moon: -∞, ...]
Result: Can only sample "mat", "hat", or "cat"
```

**When to use:**
- Fixed diversity budget: "Give me exactly K options"
- Prevent nonsensical tokens
- **Recommended K:** 30-50 for most use cases

### Top-P (Nucleus) Sampling

Adaptive filtering based on cumulative probability:

```swift
// Sample from smallest set with cumulative prob > 90%
let token = Sampler.topP(logits: logits, p: 0.9, temp: 0.8)
```

**Educational:**
```
Probabilities (sorted): [0.5, 0.3, 0.15, 0.04, 0.01]
Cumulative:             [0.5, 0.8, 0.95, 0.99, 1.0]

P=0.9 → Keep tokens until cumulative > 0.9
     → Keep first 3 tokens (cumulative = 0.95)
```

**Advantage over Top-K:**
- **Adaptive**: Sometimes needs 2 tokens, sometimes 50
- **Top-K is fixed**: Always exactly K tokens, even if model is very confident

**Recommended P:** 0.9-0.95

### Repetition Penalty

Discourage the model from repeating recent tokens:

```swift
let config = SamplerConfig(
    temperature: 0.7,
    repetitionPenalty: 1.2  // Divide logits of repeated tokens
)

let history = [42, 43, 42, 44, 42]  // Token 42 repeated
let token = Sampler.sample(logits: logits, config: config, history: history)
// → Very unlikely to sample 42 again
```

**How it works:**
```
For each token in history:
    logits[token] /= repetitionPenalty
```

**Penalty scale:**
- `1.0`: No penalty
- `1.2`: Light penalty (recommended)
- `2.0`: Strong penalty (avoid unless needed)

## Combined Sampling

In production, use multiple strategies together:

```swift
let config = SamplerConfig(
    temperature: 0.7,        // Slightly focused
    topK: 40,                // Limit to 40 best
    repetitionPenalty: 1.2   // Avoid loops
)

let token = Sampler.sample(logits: logits, config: config, history: recentTokens)
```

**Pipeline:**
1. Apply repetition penalty to history
2. Apply top-k or top-p filtering
3. Apply temperature scaling
4. Sample from resulting distribution

## Integration with Streaming

Use sampling configuration in streaming generation:

```swift
import TinyBrainRuntime

let runner = ModelRunner(weights: weights)

// Configure generation
let config = GenerationConfig(
    maxTokens: 100,
    sampler: SamplerConfig(
        temperature: 0.7,
        topK: 40,
        repetitionPenalty: 1.2
    ),
    stopTokens: [eosToken]
)

// Generate with configured sampling
for try await output in runner.generateStream(prompt: tokens, config: config) {
    print("Token: \(output.tokenId), Probability: \(output.probability)")
}
```

## Deterministic Sampling

For reproducible results, use a seed:

```swift
let config = SamplerConfig(
    temperature: 1.0,
    seed: 42  // Fixed seed
)

// Same seed + same inputs → same outputs
let token1 = Sampler.sample(logits: logits, config: config, history: [])
let token2 = Sampler.sample(logits: logits, config: config, history: [])
assert(token1 == token2)  // ✅ Deterministic
```

## Choosing the Right Strategy

| Use Case | Recommended Config |
|----------|-------------------|
| **Factual Q&A** | `temperature: 0.3, topK: 20` |
| **Creative Writing** | `temperature: 0.9, topP: 0.95` |
| **Chat/Dialogue** | `temperature: 0.7, topK: 40, penalty: 1.2` |
| **Code Generation** | `temperature: 0.2, topK: 10` |
| **Testing/Demo** | `temperature: 0.01` (near-greedy) |

## Performance

All sampling strategies are **O(vocab_size)** time:
- Greedy: Single pass to find max
- Temperature: Softmax + random sampling
- Top-K: Sort + filter
- Top-P: Sort + cumulative sum + filter

**Typical latency:**
- Vocab size 32,000: ~0.1-0.5 ms
- Negligible compared to model inference (50-150ms)

## Mathematical Details

### Temperature Formula

```
scaled_logits[i] = logits[i] / temperature
probs[i] = exp(scaled_logits[i]) / Σ exp(scaled_logits[j])
```

**Effect:**
- `temp → 0`: Distribution becomes sharper (peaked)
- `temp = 1`: Standard softmax (no change)
- `temp → ∞`: Distribution becomes uniform (flat)

### Top-K Algorithm

```
1. Sort logits by value: [(idx: 3, val: 0.8), (idx: 1, val: 0.5), ...]
2. Keep top K: [(idx: 3, val: 0.8), (idx: 1, val: 0.5)]
3. Set others to -∞: [0.1, 0.5, -∞, 0.8, -∞]
4. Apply temperature and sample
```

### Top-P (Nucleus) Algorithm

```
1. Convert to probs: [0.05, 0.25, 0.10, 0.40, 0.20]
2. Sort descending: [0.40, 0.25, 0.20, 0.10, 0.05]
3. Cumulative sum: [0.40, 0.65, 0.85, 0.95, 1.00]
4. Find cutoff where cumulative > P (0.9):
   → Keep first 4 tokens (0.95 > 0.9)
5. Zero out rest, sample
```

### Repetition Penalty

```
For each token t in history:
    if logits[t] > 0:
        logits[t] /= penalty
    else:
        logits[t] *= penalty
```

**Why the sign check?**
Negative logits mean "very unlikely" - multiplying makes them even more negative (stronger penalty).

## Topics

### Sampling Functions

- ``Sampler``
- ``SamplerConfig``
- ``GenerationConfig``
- ``TokenOutput``

## See Also

- <doc:Tokenization> - Convert text to tokens
- <doc:TensorOperations> - Process logits
- <doc:GettingStarted> - Complete tutorial

