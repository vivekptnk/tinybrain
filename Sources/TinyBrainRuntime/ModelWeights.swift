import Foundation

// MARK: - TBF Format Errors

/// Errors that can occur when saving/loading .tbf files
public enum TBFError: Error, Equatable {
    case invalidMagicBytes
    case unsupportedVersion(found: UInt32)
    case corruptMetadata
    case mmapFailed(errno: Int32)
    case invalidTensorOffset
    case fileTooSmall
}

// MARK: - Seeded Random Generator

/// Deterministic random number generator (LCG) for reproducible toy weights
public struct SeededGenerator: RandomNumberGenerator {
    private var state: UInt64

    public init(seed: UInt64) {
        self.state = seed != 0 ? seed : 0xdeadbeef
    }

    public mutating func next() -> UInt64 {
        state = 6364136223846793005 &* state &+ 1
        return state
    }
}

/// Quantized linear layer (weights + optional bias)
public struct LinearLayerWeights {
    public let weights: QuantizedTensor
    public let bias: Tensor<Float>?

    public init(weights: QuantizedTensor, bias: Tensor<Float>? = nil) {
        self.weights = weights
        self.bias = bias
    }

    public init(floatWeights: Tensor<Float>,
                bias: Tensor<Float>? = nil,
                mode: QuantizationMode = .perChannel) {
        self.init(weights: floatWeights.quantize(mode: mode), bias: bias)
    }

    /// Applies the linear projection to a `[1, inputDim]` row matrix
    public func apply(toRow input: Tensor<Float>) -> Tensor<Float> {
        var output = input.matmul(weights)

        if let bias = bias {
            precondition(bias.shape == TensorShape(weights.shape.dimensions[1]),
                         "Bias must match output dimension")
            var data = output.data
            let cols = output.shape.dimensions[1]
            let rows = output.shape.dimensions[0]

            for row in 0..<rows {
                for col in 0..<cols {
                    data[row * cols + col] += bias[col]
                }
            }

            output = Tensor(shape: output.shape, data: data)
        }

        return output
    }
}

public struct AttentionProjectionWeights {
    public let query: LinearLayerWeights
    public let key: LinearLayerWeights
    public let value: LinearLayerWeights
    public let output: LinearLayerWeights
}

public struct FeedForwardWeights {
    public let gate: LinearLayerWeights?  // gate_proj (SiLU gate for LLaMA-style gated FFN)
    public let up: LinearLayerWeights
    public let down: LinearLayerWeights

    /// Convenience init without gate (backward compatible with toy models)
    public init(up: LinearLayerWeights, down: LinearLayerWeights) {
        self.gate = nil
        self.up = up
        self.down = down
    }

    /// Full init with gate projection for LLaMA-style gated FFN
    public init(gate: LinearLayerWeights, up: LinearLayerWeights, down: LinearLayerWeights) {
        self.gate = gate
        self.up = up
        self.down = down
    }
}

public struct TransformerLayerWeights {
    public let attention: AttentionProjectionWeights
    public let feedForward: FeedForwardWeights
    public let inputNormWeights: Tensor<Float>?         // Norm scale (RMSNorm or LayerNorm γ)
    public let inputNormBias: Tensor<Float>?            // LayerNorm shift β (Phi-2 only)
    public let postAttentionNormWeights: Tensor<Float>? // Pre-FFN norm scale (LLaMA/Gemma)

    /// Convenience init without norm weights (backward compatible with toy models)
    public init(attention: AttentionProjectionWeights, feedForward: FeedForwardWeights) {
        self.attention = attention
        self.feedForward = feedForward
        self.inputNormWeights = nil
        self.inputNormBias = nil
        self.postAttentionNormWeights = nil
    }

    /// Init for LLaMA/Gemma-style models: RMSNorm, no bias, sequential residual
    public init(attention: AttentionProjectionWeights, feedForward: FeedForwardWeights,
                inputNormWeights: Tensor<Float>, postAttentionNormWeights: Tensor<Float>) {
        self.attention = attention
        self.feedForward = feedForward
        self.inputNormWeights = inputNormWeights
        self.inputNormBias = nil
        self.postAttentionNormWeights = postAttentionNormWeights
    }

    /// Init for Phi-2-style models: LayerNorm with bias, parallel residual (no post-attn norm)
    public init(attention: AttentionProjectionWeights, feedForward: FeedForwardWeights,
                inputNormWeights: Tensor<Float>, inputNormBias: Tensor<Float>?,
                postAttentionNormWeights: Tensor<Float>? = nil) {
        self.attention = attention
        self.feedForward = feedForward
        self.inputNormWeights = inputNormWeights
        self.inputNormBias = inputNormBias
        self.postAttentionNormWeights = postAttentionNormWeights
    }
}

/// Complete model weights used by ``ModelRunner``
public struct ModelWeights {
    public let config: ModelConfig
    public let tokenEmbeddings: Tensor<Float>   // [vocabSize, hiddenDim]
    public let layers: [TransformerLayerWeights]
    public let output: LinearLayerWeights
    public let finalNormWeights: Tensor<Float>?  // Final norm scale (RMSNorm or LayerNorm γ)
    public let finalNormBias: Tensor<Float>?     // Final LayerNorm shift β (Phi-2 only)

    public init(config: ModelConfig,
                tokenEmbeddings: Tensor<Float>,
                layers: [TransformerLayerWeights],
                output: LinearLayerWeights,
                finalNormWeights: Tensor<Float>? = nil,
                finalNormBias: Tensor<Float>? = nil) {
        precondition(tokenEmbeddings.shape == TensorShape(config.vocabSize, config.hiddenDim),
                     "Embedding matrix must be [vocabSize, hiddenDim]")
        precondition(layers.count == config.numLayers,
                     "Expected \(config.numLayers) layers, got \(layers.count)")
        self.config = config
        self.tokenEmbeddings = tokenEmbeddings
        self.layers = layers
        self.output = output
        self.finalNormWeights = finalNormWeights
        self.finalNormBias = finalNormBias
    }

    /// Returns the embedding row for a given token id
    public func embedding(for tokenId: Int) -> Tensor<Float> {
        precondition(tokenId >= 0 && tokenId < config.vocabSize,
                     "Token id \(tokenId) out of bounds (vocab=\(config.vocabSize))")
        return tokenEmbeddings.row(tokenId)
    }

    /// Generates a deterministic, quantized toy model for tests/demo usage
    public static func makeToyModel(config: ModelConfig, seed: UInt64 = 42) -> ModelWeights {
        var rng: any RandomNumberGenerator = SeededGenerator(seed: seed)

        let embeddings = Tensor<Float>.random(
            shape: TensorShape(config.vocabSize, config.hiddenDim),
            mean: 0,
            std: 0.02,
            using: &rng
        )

        func makeProjection(outputDim: Int) -> LinearLayerWeights {
            let weights = Tensor<Float>.random(
                shape: TensorShape(config.hiddenDim, outputDim),
                mean: 0,
                std: 0.02,
                using: &rng
            )
            let bias = Tensor<Float>.random(
                shape: TensorShape(outputDim),
                mean: 0,
                std: 0.02,
                using: &rng
            )
            return LinearLayerWeights(floatWeights: weights, bias: bias)
        }

        var layers: [TransformerLayerWeights] = []
        for _ in 0..<config.numLayers {
            let attention = AttentionProjectionWeights(
                query: makeProjection(outputDim: config.hiddenDim),
                key: makeProjection(outputDim: config.kvDim),
                value: makeProjection(outputDim: config.kvDim),
                output: makeProjection(outputDim: config.hiddenDim)
            )

            let ffnHidden = config.hiddenDim * 4
            let feedForward = FeedForwardWeights(
                up: makeProjection(outputDim: ffnHidden),
                down: LinearLayerWeights(
                    floatWeights: Tensor<Float>.random(
                        shape: TensorShape(ffnHidden, config.hiddenDim),
                        mean: 0,
                        std: 0.02,
                        using: &rng
                    ),
                    bias: Tensor<Float>.random(
                        shape: TensorShape(config.hiddenDim),
                        mean: 0,
                        std: 0.02,
                        using: &rng
                    )
                )
            )

            layers.append(TransformerLayerWeights(attention: attention, feedForward: feedForward))
        }

        let outputProjection = LinearLayerWeights(
            floatWeights: Tensor<Float>.random(
                shape: TensorShape(config.hiddenDim, config.vocabSize),
                mean: 0,
                std: 0.02,
                using: &rng
            ),
            bias: Tensor<Float>.zeros(shape: TensorShape(config.vocabSize))
        )

        return ModelWeights(
            config: config,
            tokenEmbeddings: embeddings,
            layers: layers,
            output: outputProjection
        )
    }

    // MARK: - TBF Format Save/Load (TB-004 Work Item #3)

    /// Saves model weights to .tbf (TinyBrain Binary Format) file
    ///
    /// **GREEN Phase:** Minimal implementation to pass tests
    ///
    /// Format:
    /// - Header: magic bytes "TBFM", version, config JSON
    /// - Metadata: quantization scales and zero points
    /// - Index: tensor offsets for mmap access
    /// - Data: weight blobs, 4KB aligned
    ///
    /// - Parameter path: File path to save to
    /// - Throws: TBFError or IO errors
    public func save(to path: String) throws {
        let fileURL = URL(fileURLWithPath: path)
        var data = Data()

        // 1. Write Header
        let magic = "TBFM"
        data.append(contentsOf: magic.utf8)

        // Version 1
        var version: UInt32 = 1
        data.append(Data(bytes: &version, count: 4))

        // Serialize config to JSON
        let configJSON = try JSONEncoder().encode(config)
        var configLength = UInt32(configJSON.count)
        data.append(Data(bytes: &configLength, count: 4))
        data.append(configJSON)

        // Pad header to 4KB
        let headerSize = data.count
        let paddedHeaderSize = ((headerSize + 4095) / 4096) * 4096
        data.append(Data(count: paddedHeaderSize - headerSize))

        // ── Collect all tensors in write order ─────────────────────────
        // Each entry: (name, isQuantized, quantizedTensor?, floatData?, shape, dataSize)
        struct TensorEntry {
            let name: String
            let quantized: QuantizedTensor?   // non-nil for quantized weights
            let floatData: [Float]?           // non-nil for Float32 tensors
            let shape: TensorShape
            var dataSize: Int {
                if let q = quantized { return q.data.count }
                return (floatData?.count ?? 0) * 4
            }
        }

        var allTensors: [TensorEntry] = []
        var quantTensors: [TensorEntry] = []

        // Embeddings (Float32)
        allTensors.append(TensorEntry(name: "embeddings", quantized: nil,
                                       floatData: tokenEmbeddings.data, shape: tokenEmbeddings.shape))

        // Layer weights
        for (layerIdx, layer) in layers.enumerated() {
            let layerQuantized: [(String, QuantizedTensor)] = {
                var entries: [(String, QuantizedTensor)] = [
                    ("layer_\(layerIdx)_attn_q", layer.attention.query.weights),
                    ("layer_\(layerIdx)_attn_k", layer.attention.key.weights),
                    ("layer_\(layerIdx)_attn_v", layer.attention.value.weights),
                    ("layer_\(layerIdx)_attn_o", layer.attention.output.weights),
                ]
                if let gate = layer.feedForward.gate {
                    entries.append(("layer_\(layerIdx)_ffn_gate", gate.weights))
                }
                entries.append(("layer_\(layerIdx)_ffn_up", layer.feedForward.up.weights))
                entries.append(("layer_\(layerIdx)_ffn_down", layer.feedForward.down.weights))
                return entries
            }()

            for (name, qt) in layerQuantized {
                let entry = TensorEntry(name: name, quantized: qt, floatData: nil, shape: qt.shape)
                allTensors.append(entry)
                quantTensors.append(entry)
            }

            // Norm weights (Float32)
            if let inNorm = layer.inputNormWeights {
                allTensors.append(TensorEntry(name: "layer_\(layerIdx)_ln_input", quantized: nil,
                                               floatData: inNorm.data, shape: inNorm.shape))
            }
            if let inNormBias = layer.inputNormBias {
                allTensors.append(TensorEntry(name: "layer_\(layerIdx)_ln_input_bias",
                                               quantized: nil,
                                               floatData: inNormBias.data, shape: inNormBias.shape))
            }
            if let postNorm = layer.postAttentionNormWeights {
                allTensors.append(TensorEntry(name: "layer_\(layerIdx)_ln_post", quantized: nil,
                                               floatData: postNorm.data, shape: postNorm.shape))
            }

            // Attention projection biases (Float32, Phi-2 style)
            let attnBiases: [(String, Tensor<Float>?)] = [
                ("layer_\(layerIdx)_attn_q_bias", layer.attention.query.bias),
                ("layer_\(layerIdx)_attn_k_bias", layer.attention.key.bias),
                ("layer_\(layerIdx)_attn_v_bias", layer.attention.value.bias),
                ("layer_\(layerIdx)_attn_o_bias", layer.attention.output.bias),
            ]
            for (name, bias) in attnBiases {
                if let b = bias {
                    allTensors.append(TensorEntry(name: name, quantized: nil,
                                                   floatData: b.data, shape: b.shape))
                }
            }

            // FFN biases (Float32, Phi-2 style)
            let ffnBiases: [(String, Tensor<Float>?)] = [
                ("layer_\(layerIdx)_ffn_up_bias", layer.feedForward.up.bias),
                ("layer_\(layerIdx)_ffn_down_bias", layer.feedForward.down.bias),
            ]
            for (name, bias) in ffnBiases {
                if let b = bias {
                    allTensors.append(TensorEntry(name: name, quantized: nil,
                                                   floatData: b.data, shape: b.shape))
                }
            }
        }

        // Final norm (Float32)
        if let finalNorm = finalNormWeights {
            allTensors.append(TensorEntry(name: "final_norm", quantized: nil,
                                           floatData: finalNorm.data, shape: finalNorm.shape))
        }
        if let finalNormB = finalNormBias {
            allTensors.append(TensorEntry(name: "final_norm_bias", quantized: nil,
                                           floatData: finalNormB.data, shape: finalNormB.shape))
        }

        // Output projection (quantized)
        let outEntry = TensorEntry(name: "output", quantized: output.weights, floatData: nil,
                                    shape: output.weights.shape)
        allTensors.append(outEntry)
        quantTensors.append(outEntry)

        // ── 2. Write Quantization Metadata ──────────────────────────────
        let metadataOffset = data.count

        var quantCount = UInt32(quantTensors.count)
        data.append(Data(bytes: &quantCount, count: 4))

        for entry in quantTensors {
            let qt = entry.quantized!
            let nameData = entry.name.utf8
            var nameLength = UInt32(nameData.count)
            data.append(Data(bytes: &nameLength, count: 4))
            data.append(contentsOf: nameData)

            var precision: UInt8 = qt.precision == .int4 ? 2 : 1
            data.append(Data(bytes: &precision, count: 1))
            var mode: UInt8 = 2  // perChannel
            data.append(Data(bytes: &mode, count: 1))
            var groupSize = UInt32(qt.groupSize)
            data.append(Data(bytes: &groupSize, count: 4))

            var scalesCount = UInt32(qt.scales.count)
            data.append(Data(bytes: &scalesCount, count: 4))
            for var scale in qt.scales {
                data.append(Data(bytes: &scale, count: 4))
            }

            let zpCount = UInt32(qt.zeroPoints?.count ?? 0)
            var zpCountVar = zpCount
            data.append(Data(bytes: &zpCountVar, count: 4))
            if let zps = qt.zeroPoints {
                for var zp in zps { data.append(Data(bytes: &zp, count: 1)) }
            }
        }

        // Pad metadata to 4KB
        let metadataSize = data.count - metadataOffset
        let paddedMetadataSize = ((metadataSize + 4095) / 4096) * 4096
        data.append(Data(count: paddedMetadataSize - metadataSize))

        // ── 3. Write Tensor Index (two-pass with placeholder offsets) ───
        let indexOffset = data.count
        var totalTensorCount = UInt32(allTensors.count)
        data.append(Data(bytes: &totalTensorCount, count: 4))

        // Pass 1: write index entries with placeholder offsets; record patch positions
        var offsetPatchPositions: [(dataPosition: Int, dataSize: Int)] = []
        for entry in allTensors {
            let nameData = entry.name.utf8
            var nameLength = UInt32(nameData.count)
            data.append(Data(bytes: &nameLength, count: 4))
            data.append(contentsOf: nameData)

            var dimCount = UInt32(entry.shape.dimensions.count)
            data.append(Data(bytes: &dimCount, count: 4))
            for dim in entry.shape.dimensions {
                var dim32 = Int32(dim)
                data.append(Data(bytes: &dim32, count: 4))
            }

            offsetPatchPositions.append((data.count, entry.dataSize))
            var placeholder: UInt64 = 0
            data.append(Data(bytes: &placeholder, count: 8))  // offset placeholder
            var size64 = UInt64(entry.dataSize)
            data.append(Data(bytes: &size64, count: 8))
        }

        // Pad index to 4KB
        let indexSize = data.count - indexOffset
        let paddedIndexSize = ((indexSize + 4095) / 4096) * 4096
        data.append(Data(count: paddedIndexSize - indexSize))

        let dataStart = data.count

        // ── 4. Write Weight Data Blobs (4KB aligned) ────────────────────
        var currentDataOffset = 0
        for entry in allTensors {
            if let floatData = entry.floatData {
                for var value in floatData {
                    data.append(Data(bytes: &value, count: 4))
                }
            } else if let qt = entry.quantized {
                for var value in qt.data {
                    data.append(Data(bytes: &value, count: 1))
                }
            }
            let blobSize = data.count - (dataStart + currentDataOffset)
            let paddedBlobSize = ((blobSize + 4095) / 4096) * 4096
            data.append(Data(count: paddedBlobSize - blobSize))
            currentDataOffset += paddedBlobSize
        }

        // Pass 2: patch placeholder offsets with absolute positions
        currentDataOffset = 0
        for (patchPos, blobDataSize) in offsetPatchPositions {
            var absoluteOffset = UInt64(dataStart + currentDataOffset)
            data.replaceSubrange(patchPos..<(patchPos + 8),
                                 with: Data(bytes: &absoluteOffset, count: 8))
            let paddedBlobSize = ((blobDataSize + 4095) / 4096) * 4096
            currentDataOffset += paddedBlobSize
        }

        // Write to file
        try data.write(to: fileURL)
    }

    /// Loads model weights from .tbf file using mmap
    ///
    /// **GREEN Phase:** Full implementation to pass all tests
    ///
    /// - Parameter path: File path to load from
    /// - Returns: Loaded ModelWeights
    /// - Throws: TBFError or IO errors
    public static func load(from path: String) throws -> ModelWeights {
        let fileURL = URL(fileURLWithPath: path)

        // Check file exists
        guard FileManager.default.fileExists(atPath: path) else {
            throw CocoaError(.fileNoSuchFile)
        }

        // Read file data (GREEN phase: using Data, will optimize to mmap in REFACTOR)
        let data = try Data(contentsOf: fileURL)

        // Verify minimum size
        guard data.count >= 12 else {
            throw TBFError.fileTooSmall
        }

        var offset = 0

        // Helper to read UInt32 (little-endian, handles unaligned access)
        func readUInt32() -> UInt32 {
            let bytes = data[offset..<(offset + 4)]
            offset += 4
            return bytes.withUnsafeBytes { ptr in
                ptr.loadUnaligned(as: UInt32.self)
            }
        }

        // Helper to read Float (little-endian, handles unaligned access)
        func readFloat() -> Float {
            let bytes = data[offset..<(offset + 4)]
            offset += 4
            return bytes.withUnsafeBytes { ptr in
                ptr.loadUnaligned(as: Float.self)
            }
        }

        // Helper to read Int8
        func readInt8() -> Int8 {
            let value = data[offset]
            offset += 1
            return Int8(bitPattern: value)
        }

        // Helper to read UInt8
        func readUInt8() -> UInt8 {
            let value = data[offset]
            offset += 1
            return value
        }

        // 1. Parse Header
        let magic = String(data: data[0..<4], encoding: .utf8)
        guard magic == "TBFM" else {
            throw TBFError.invalidMagicBytes
        }
        offset += 4

        let version = readUInt32()
        guard version == 1 else {
            throw TBFError.unsupportedVersion(found: version)
        }

        let configLength = readUInt32()

        let configData = data[offset..<(offset + Int(configLength))]
        let config = try JSONDecoder().decode(ModelConfig.self, from: configData)
        offset += Int(configLength)

        // Skip to metadata section (4KB aligned)
        offset = ((offset + 4095) / 4096) * 4096

        // 2. Parse Quantization Metadata
        let metadataCount = readUInt32()

        // Store metadata by name for later reconstruction
        var quantMetadata: [String: (precision: UInt8, mode: UInt8, scales: [Float], zeroPoints: [Int8]?)] = [:]

        for _ in 0..<metadataCount {
            let nameLength = readUInt32()

            let nameData = data[offset..<(offset + Int(nameLength))]
            let name = String(data: nameData, encoding: .utf8) ?? ""
            offset += Int(nameLength)

            let precision = readUInt8()
            let mode = readUInt8()

            // Group size (UInt32): used by INT4 quantization, 0 for INT8
            let _ = readUInt32()

            let scalesCount = readUInt32()

            var scales: [Float] = []
            for _ in 0..<scalesCount {
                scales.append(readFloat())
            }

            let zpCount = readUInt32()

            var zeroPoints: [Int8]? = nil
            if zpCount > 0 {
                zeroPoints = []
                for _ in 0..<zpCount {
                    zeroPoints!.append(readInt8())
                }
            }

            quantMetadata[name] = (precision, mode, scales, zeroPoints)
        }

        // Skip to index section (4KB aligned)
        offset = ((offset + 4095) / 4096) * 4096

        // Helper to read Int32
        func readInt32() -> Int32 {
            let bytes = data[offset..<(offset + 4)]
            offset += 4
            return bytes.withUnsafeBytes { ptr in
                ptr.loadUnaligned(as: Int32.self)
            }
        }

        // Helper to read UInt64
        func readUInt64() -> UInt64 {
            let bytes = data[offset..<(offset + 8)]
            offset += 8
            return bytes.withUnsafeBytes { ptr in
                ptr.loadUnaligned(as: UInt64.self)
            }
        }

        // 3. Parse Tensor Index
        let tensorCount = readUInt32()

        var tensorIndex: [String: (shape: [Int], dataOffset: Int, dataSize: Int)] = [:]

        for _ in 0..<tensorCount {
            let nameLength = readUInt32()

            let nameData = data[offset..<(offset + Int(nameLength))]
            let name = String(data: nameData, encoding: .utf8) ?? ""
            offset += Int(nameLength)

            let dimCount = readUInt32()

            var shape: [Int] = []
            for _ in 0..<dimCount {
                shape.append(Int(readInt32()))
            }

            let dataOffset = readUInt64()
            let dataSize = readUInt64()

            tensorIndex[name] = (shape, Int(dataOffset), Int(dataSize))
        }

        // 4. Load Weight Data
        // Use absolute offsets from tensor index (written by the converter)
        // Bulk copy via withUnsafeBytes to avoid per-element overhead on large models

        // Helper to load quantized tensor using stored absolute offset
        func loadQuantizedTensor(name: String) -> QuantizedTensor {
            guard let index = tensorIndex[name] else {
                fatalError("Missing tensor: \(name)")
            }
            guard let metadata = quantMetadata[name] else {
                fatalError("Missing metadata: \(name)")
            }

            let shape = TensorShape(index.shape)
            let tensorOffset = index.dataOffset

            // Bulk copy INT8 data from absolute offset
            let quantData: [Int8] = data.withUnsafeBytes { rawPtr in
                let base = rawPtr.baseAddress!.advanced(by: tensorOffset)
                    .assumingMemoryBound(to: Int8.self)
                return Array(UnsafeBufferPointer(start: base, count: index.dataSize))
            }

            let mode: QuantizationMode = metadata.mode == 2 ? .perChannel : .symmetric
            return QuantizedTensor(
                shape: shape,
                data: quantData,
                scales: metadata.scales,
                zeroPoints: metadata.zeroPoints,
                mode: mode,
                precision: .int8
            )
        }

        // Load embeddings (Float32) using absolute offset from tensor index
        guard let embIndex = tensorIndex["embeddings"] else {
            fatalError("Missing tensor: embeddings")
        }
        let floatCount = embIndex.dataSize / 4
        let embeddingData: [Float] = data.withUnsafeBytes { rawPtr in
            let base = rawPtr.baseAddress!.advanced(by: embIndex.dataOffset)
                .assumingMemoryBound(to: Float.self)
            return Array(UnsafeBufferPointer(start: base, count: floatCount))
        }
        let embeddings = Tensor<Float>(shape: TensorShape(embIndex.shape), data: embeddingData)

        // Helper to load Float32 tensor using absolute offset from tensor index
        func loadFloat32Tensor(name: String) -> Tensor<Float>? {
            guard let index = tensorIndex[name] else { return nil }
            let count = index.dataSize / 4
            let floatData: [Float] = data.withUnsafeBytes { rawPtr in
                let base = rawPtr.baseAddress!.advanced(by: index.dataOffset)
                    .assumingMemoryBound(to: Float.self)
                return Array(UnsafeBufferPointer(start: base, count: count))
            }
            return Tensor<Float>(shape: TensorShape(index.shape), data: floatData)
        }

        // Load layer weights
        var layers: [TransformerLayerWeights] = []
        for layerIdx in 0..<config.numLayers {
            let qWeights = loadQuantizedTensor(name: "layer_\(layerIdx)_attn_q")
            let kWeights = loadQuantizedTensor(name: "layer_\(layerIdx)_attn_k")
            let vWeights = loadQuantizedTensor(name: "layer_\(layerIdx)_attn_v")
            let oWeights = loadQuantizedTensor(name: "layer_\(layerIdx)_attn_o")

            // Gate projection (LLaMA-style gated FFN)
            let gateWeights: QuantizedTensor? = tensorIndex["layer_\(layerIdx)_ffn_gate"] != nil
                ? loadQuantizedTensor(name: "layer_\(layerIdx)_ffn_gate") : nil

            let upWeights = loadQuantizedTensor(name: "layer_\(layerIdx)_ffn_up")
            let downWeights = loadQuantizedTensor(name: "layer_\(layerIdx)_ffn_down")

            // Norm weights (Float32)
            let inputNorm = loadFloat32Tensor(name: "layer_\(layerIdx)_ln_input")
            let inputNormBias = loadFloat32Tensor(name: "layer_\(layerIdx)_ln_input_bias")
            let postAttnNorm = loadFloat32Tensor(name: "layer_\(layerIdx)_ln_post")

            // Attention projection biases (Float32, phi-2 style; nil for LLaMA/Gemma)
            let qBias = loadFloat32Tensor(name: "layer_\(layerIdx)_attn_q_bias")
            let kBias = loadFloat32Tensor(name: "layer_\(layerIdx)_attn_k_bias")
            let vBias = loadFloat32Tensor(name: "layer_\(layerIdx)_attn_v_bias")
            let oBias = loadFloat32Tensor(name: "layer_\(layerIdx)_attn_o_bias")

            // FFN biases (Float32, phi-2 style)
            let upBias   = loadFloat32Tensor(name: "layer_\(layerIdx)_ffn_up_bias")
            let downBias = loadFloat32Tensor(name: "layer_\(layerIdx)_ffn_down_bias")

            let attention = AttentionProjectionWeights(
                query:  LinearLayerWeights(weights: qWeights, bias: qBias),
                key:    LinearLayerWeights(weights: kWeights, bias: kBias),
                value:  LinearLayerWeights(weights: vWeights, bias: vBias),
                output: LinearLayerWeights(weights: oWeights, bias: oBias)
            )

            let feedForward: FeedForwardWeights
            if let gateW = gateWeights {
                feedForward = FeedForwardWeights(
                    gate: LinearLayerWeights(weights: gateW, bias: nil),
                    up:   LinearLayerWeights(weights: upWeights,   bias: upBias),
                    down: LinearLayerWeights(weights: downWeights, bias: downBias)
                )
            } else {
                feedForward = FeedForwardWeights(
                    up:   LinearLayerWeights(weights: upWeights,   bias: upBias),
                    down: LinearLayerWeights(weights: downWeights, bias: downBias)
                )
            }

            if let inNorm = inputNorm {
                layers.append(TransformerLayerWeights(
                    attention: attention, feedForward: feedForward,
                    inputNormWeights: inNorm, inputNormBias: inputNormBias,
                    postAttentionNormWeights: postAttnNorm))
            } else {
                layers.append(TransformerLayerWeights(attention: attention, feedForward: feedForward))
            }
        }

        // Final norm (Float32) if present
        let finalNorm     = loadFloat32Tensor(name: "final_norm")
        let finalNormBias = loadFloat32Tensor(name: "final_norm_bias")

        // Output weights (may have bias for lm_head)
        let outputWeights = loadQuantizedTensor(name: "output")
        let outputBias    = loadFloat32Tensor(name: "output_bias")
        let output        = LinearLayerWeights(weights: outputWeights, bias: outputBias)

        return ModelWeights(
            config: config,
            tokenEmbeddings: embeddings,
            layers: layers,
            output: output,
            finalNormWeights: finalNorm,
            finalNormBias: finalNormBias
        )
    }
}
