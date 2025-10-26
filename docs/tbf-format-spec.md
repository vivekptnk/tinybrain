# TinyBrain Binary Format (.tbf) Specification

**Version:** 1.0  
**Status:** Draft  
**Last Updated:** October 25, 2025

---

## Overview

The **TinyBrain Binary Format** (.tbf) is a memory-mapped file format designed for efficient on-device loading of quantized language model weights. The format prioritizes:

1. **Zero-copy loading** via `mmap()` - weights accessed directly from disk without full RAM load
2. **Page alignment** - 4KB boundaries for optimal OS memory mapping
3. **Quantization metadata** - per-channel scales and zero points stored separately
4. **Version compatibility** - forward/backward compatibility through explicit versioning

---

## File Structure

```
┌─────────────────────────────────────────────────────────────┐
│ HEADER (variable size, ~1KB typical)                       │
│  ├─ Magic bytes: "TBFM" (4 bytes)                          │
│  ├─ Version: UInt32 (4 bytes)                              │
│  ├─ Config JSON length: UInt32 (4 bytes)                   │
│  ├─ Config JSON: UTF-8 string (variable)                   │
│  └─ Padding to 4KB boundary                                │
├─────────────────────────────────────────────────────────────┤
│ QUANTIZATION METADATA (aligned to 4KB)                     │
│  ├─ Metadata count: UInt32 (4 bytes)                       │
│  ├─ For each tensor:                                       │
│  │   ├─ Tensor ID: String (length-prefixed)               │
│  │   ├─ Precision: UInt8 (1=INT8, 2=INT4, 3=FP16)         │
│  │   ├─ Mode: UInt8 (0=symmetric, 1=asymmetric, etc.)     │
│  │   ├─ Scales count: UInt32                              │
│  │   ├─ Scales: [Float] (4 bytes each)                    │
│  │   ├─ Zero points count: UInt32                         │
│  │   └─ Zero points: [Int8] (1 byte each)                 │
│  └─ Padding to 4KB boundary                                │
├─────────────────────────────────────────────────────────────┤
│ WEIGHT TENSOR INDEX (aligned to 4KB)                       │
│  ├─ Tensor count: UInt32 (4 bytes)                         │
│  ├─ For each tensor:                                       │
│  │   ├─ Name: String (length-prefixed)                    │
│  │   ├─ Shape dimensions count: UInt32                    │
│  │   ├─ Shape: [Int] (4 bytes each dimension)             │
│  │   ├─ Data offset: UInt64 (byte offset from file start) │
│  │   └─ Data size: UInt64 (bytes)                         │
│  └─ Padding to 4KB boundary                                │
├─────────────────────────────────────────────────────────────┤
│ WEIGHT DATA BLOBS (each aligned to 4KB)                    │
│  ├─ Tensor 0: token_embeddings (INT8 or Float)            │
│  │   └─ Padding to 4KB                                    │
│  ├─ Tensor 1: layer_0.attention.query.weights             │
│  │   └─ Padding to 4KB                                    │
│  ├─ Tensor 2: layer_0.attention.query.bias                │
│  │   └─ Padding to 4KB                                    │
│  ├─ ...                                                     │
│  └─ Tensor N: output.weights                               │
│      └─ Padding to 4KB                                     │
└─────────────────────────────────────────────────────────────┘
```

---

## Detailed Specification

### 1. Header Section

**Offset 0x0000:** Magic Bytes (4 bytes)
```
Value: 0x54 0x42 0x46 0x4D  (ASCII "TBFM")
Purpose: File format identification
```

**Offset 0x0004:** Version (4 bytes, UInt32, little-endian)
```
Current: 1
Purpose: Format version for compatibility checks
```

**Offset 0x0008:** Config JSON Length (4 bytes, UInt32, little-endian)
```
Purpose: Size of following JSON string in bytes
```

**Offset 0x000C:** Config JSON (variable length, UTF-8)
```json
{
  "numLayers": 22,
  "hiddenDim": 2048,
  "numHeads": 16,
  "vocabSize": 32000,
  "maxSeqLen": 2048
}
```

**Purpose:** Model architecture configuration (ModelConfig serialized to JSON)

**Padding:** Pad with zeros to next 4KB boundary (0x1000)

### 2. Quantization Metadata Section

**Aligned to:** 4KB boundary

**Layout:**
```
UInt32: metadata_count
For each metadata entry:
  UInt32: tensor_id_length
  UTF-8: tensor_id (e.g., "layer_0.attention.query")
  UInt8: precision (1=INT8, 2=INT4, 3=FP16)
  UInt8: mode (0=symmetric, 1=asymmetric, 2=perChannel)
  UInt32: scales_count
  [Float]: scales (4 bytes each, little-endian)
  UInt32: zero_points_count
  [Int8]: zero_points (1 byte each, signed)
```

**Padding:** Pad with zeros to next 4KB boundary

### 3. Weight Tensor Index

**Aligned to:** 4KB boundary

**Layout:**
```
UInt32: tensor_count
For each tensor:
  UInt32: name_length
  UTF-8: name (e.g., "token_embeddings")
  UInt32: dimensions_count
  [Int32]: shape (4 bytes each dimension)
  UInt64: data_offset (absolute byte offset from file start)
  UInt64: data_size (bytes)
```

**Purpose:** Allows random access to any tensor without reading entire file

**Padding:** Pad with zeros to next 4KB boundary

### 4. Weight Data Blobs

**Aligned to:** Each tensor starts at 4KB boundary

**Layout:**
```
Raw binary data for each tensor (INT8, INT4 packed, or Float32)
Padding to next 4KB boundary after each tensor
```

**INT8 Format:**
- 1 byte per value
- Signed integers in range [-128, 127]
- Dequantize using: `float_value = (int8_value - zero_point) * scale`

**INT4 Format:**
- 2 values packed per byte (high nibble, low nibble)
- Signed integers in range [-8, 7]
- Byte packing: `byte = (value1 << 4) | (value2 & 0x0F)`

**Float32 Format:**
- 4 bytes per value (IEEE 754 single precision, little-endian)
- Used for embeddings and biases

---

## Memory Mapping Strategy

### Loading Process

1. **Open file** with `open()`
2. **Map entire file** with `mmap(PROT_READ, MAP_PRIVATE)`
3. **Parse header** (in-memory, fast)
4. **Parse index** (in-memory, fast)
5. **Access tensors** via index offsets (zero-copy, on-demand)

### Advantages

- **Lazy loading:** OS loads pages only when accessed
- **Shared memory:** Multiple processes can share same file mapping
- **No RAM overhead:** File pages cached by OS, not duplicated in app memory
- **Fast startup:** No upfront loading of 1-4 GB weights

### Example Usage

```swift
let weights = try ModelWeights.load(from: "tinyllama-int8.tbf")
// File is mmap'd but not loaded into RAM yet

let embedding = weights.embedding(for: tokenId)  
// OS loads only the embedding page(s) needed
```

---

## Alignment Requirements

### Why 4KB?

- **OS page size:** Most systems use 4KB pages
- **mmap efficiency:** OS can map directly without fragmentation
- **Cache alignment:** CPU cache lines align well with 4KB boundaries

### Padding Calculation

```swift
func paddedSize(_ size: Int, alignment: Int = 4096) -> Int {
    return ((size + alignment - 1) / alignment) * alignment
}
```

---

## Version Compatibility

### Version 1 (Current)

- INT8 per-channel quantization
- Float32 embeddings and biases
- Basic ModelConfig fields

### Future Versions (Planned)

**Version 2:**
- INT4 per-group quantization
- Group size metadata
- Compressed metadata section

**Version 3:**
- Multi-precision tensors (FP16, BF16)
- Sparse weight encoding
- Tokenizer vocabulary embedded

### Compatibility Rules

- **Major version change:** Breaking format changes
- **Minor version bump:** Backward-compatible additions
- **Readers MUST reject:** Versions higher than supported
- **Writers SHOULD write:** Latest version

---

## File Size Estimation

### TinyLlama 1.1B INT8

```
Header:              ~4 KB
Metadata:            ~8 KB (22 layers × ~365 bytes)
Index:               ~4 KB (150 tensors × ~27 bytes)
Embeddings:          32000 × 2048 × 1 byte = 64 MB
Attention weights:   22 layers × 4 proj × (2048×2048) × 1 byte = 359 MB
FFN weights:         22 layers × 2 proj × sizes × 1 byte = 551 MB
Total (approx):      ~975 MB
With padding:        ~1.1 GB
```

**Memory Savings vs FP32:** 75% (from ~4.4 GB)

---

## Error Handling

### Invalid Magic Bytes
```
Throw: TBFError.invalidMagicBytes
Message: "Not a valid .tbf file (magic bytes mismatch)"
```

### Version Mismatch
```
Throw: TBFError.unsupportedVersion(found: UInt32)
Message: "Unsupported .tbf version X (max supported: 1)"
```

### Corrupt Metadata
```
Throw: TBFError.corruptMetadata
Message: "Failed to parse quantization metadata"
```

### mmap Failure
```
Throw: TBFError.mmapFailed(errno: Int32)
Message: "Failed to memory-map file: [errno description]"
```

---

## Security Considerations

1. **Validate all sizes** before allocating/reading
2. **Check tensor offsets** are within file bounds
3. **Verify alignment** before casting pointers
4. **Limit metadata size** to prevent memory exhaustion
5. **Use MAP_PRIVATE** for mmap (no write-back to file)

---

## Reference Implementation

See:
- `Sources/TinyBrainRuntime/ModelWeights.swift` - save()/load()
- `Tests/TinyBrainRuntimeTests/TBFFormatTests.swift` - Round-trip tests

---

## Changelog

**Version 1.0 (October 2025)**
- Initial specification
- INT8 per-channel quantization
- 4KB page alignment
- Basic ModelConfig support

