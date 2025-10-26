# Tokenization

Convert text to tokens and back using BPE (Byte Pair Encoding).

## Overview

Tokenization is the first step in any LLM pipeline: converting human-readable text into integer tokens that the model can process.

**TinyBrain** provides a Swift-native BPE tokenizer that:
- Handles Unicode text with normalization
- Supports special tokens (BOS, EOS, UNK, PAD)
- Runs entirely on-device without Python dependencies
- Provides educational transparency into the tokenization process

## How BPE Works

Byte Pair Encoding is a subword tokenization algorithm:

```
Original Text: "Hello, world!"

Step 1 - Character Split:
['H', 'e', 'l', 'l', 'o', ',', ' ', 'w', 'o', 'r', 'l', 'd', '!']

Step 2 - Apply Learned Merges:
['H', 'e'] → 'He'
['He', 'l'] → 'Hel'
['l', 'l'] → 'll'
...

Final Tokens:
['Hello', ',', ' ', 'world', '!']
→ [102, 8, 9, 105, 13]
```

**Key Benefits:**
- **Unknown words handled gracefully**: "TinyBrain" → ["Tiny", "Brain"]
- **Efficient**: Common words = single token
- **Multilingual**: Works across languages

## Basic Usage

### Encoding Text to Tokens

```swift
import TinyBrainTokenizer

// Load vocabulary from file
let tokenizer = try BPETokenizer(vocabularyPath: "tinyllama-vocab.json")

// Encode text
let tokens = tokenizer.encode("Hello, TinyBrain!")
// → [102, 8, 9, 307, 310, 13]

// Vocabulary info
print("Vocab size: \(tokenizer.vocabularySize)")
print("BOS token: \(tokenizer.bosToken)")  // 0
print("EOS token: \(tokenizer.eosToken)")  // 1
```

### Decoding Tokens to Text

```swift
// Decode tokens back to text
let text = tokenizer.decode([102, 8, 9, 105])
print(text)  // "Hello, world"

// Round-trip test
let original = "Swift is awesome!"
let roundTrip = tokenizer.decode(tokenizer.encode(original))
assert(original == roundTrip)  // ✅
```

### Special Tokens

Special tokens control sequence boundaries and handle edge cases:

```swift
// Add BOS token to start of sequence
let tokens = [tokenizer.bosToken] + tokenizer.encode("Hello") + [tokenizer.eosToken]
// → [0, 102, 1]  (BOS, Hello, EOS)

// Unknown characters fallback to UNK
let tokens = tokenizer.encode("日本語")  // Japanese text
// → [2, 2, 2]  (UNK for each character not in vocab)
```

## Unicode Handling

BPE automatically normalizes Unicode to ensure consistency:

```swift
// NFC (composed) vs NFD (decomposed) forms
let nfc = "café"  // Single é character
let nfd = "cafe\u{0301}"  // e + combining accent

let tokens1 = tokenizer.encode(nfc)
let tokens2 = tokenizer.encode(nfd)

assert(tokens1 == tokens2)  // ✅ Normalized to same tokens
```

## Integration with ModelRunner

Combine tokenization with inference:

```swift
import TinyBrainRuntime
import TinyBrainTokenizer

let tokenizer = try BPETokenizer(vocabularyPath: "vocab.json")
let runner = ModelRunner(weights: weights)

// Encode prompt
let prompt = "Explain quantum physics"
let tokenIds = tokenizer.encode(prompt)

// Generate response
let config = GenerationConfig(maxTokens: 100)
for try await output in runner.generateStream(prompt: tokenIds, config: config) {
    // Decode each token as it's generated
    let text = tokenizer.decode([output.tokenId])
    print(text, terminator: "")
}
```

## Vocabulary File Format

BPE vocabularies are JSON files with three sections:

```json
{
  "vocab": {
    "<BOS>": 0,
    "<EOS>": 1,
    "<UNK>": 2,
    "Hello": 100,
    "world": 101
  },
  "merges": [
    ["H", "e"],
    ["He", "l"],
    ["l", "l"]
  ],
  "special_tokens": {
    "bos_token": "<BOS>",
    "eos_token": "<EOS>",
    "unk_token": "<UNK>",
    "pad_token": "<PAD>"
  }
}
```

## Performance Considerations

**Encoding Performance:**
- Character split: O(n) where n = text length
- BPE merges: O(n² × m) where m = number of merges
- Total: Linear for most real-world text

**Memory:**
- Vocabulary: ~1-50 MB depending on vocab size
- Merges: ~100 KB - 1 MB
- Loaded once, cached for all operations

**Tips:**
- Reuse tokenizer instances (don't reload for each encoding)
- Batch encode multiple texts if possible
- For streaming UX, tokenize prompt once, then decode tokens incrementally

## Topics

### Tokenizer Protocol

- ``Tokenizer``
- ``BPETokenizer``

### Error Handling

Tokenization is designed to never fail for valid UTF-8 text:
- Unknown characters → UNK token
- Invalid token IDs in decode → skipped silently
- Empty strings → empty token arrays

## See Also

- <doc:Sampling> - Convert tokens to probabilities
- <doc:TensorOperations> - Process token embeddings
- <doc:GettingStarted> - Complete tutorial

