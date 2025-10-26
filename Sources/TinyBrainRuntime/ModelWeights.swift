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
    public let up: LinearLayerWeights
    public let down: LinearLayerWeights
}

public struct TransformerLayerWeights {
    public let attention: AttentionProjectionWeights
    public let feedForward: FeedForwardWeights
}

/// Complete model weights used by ``ModelRunner``
public struct ModelWeights {
    public let config: ModelConfig
    public let tokenEmbeddings: Tensor<Float>   // [vocabSize, hiddenDim]
    public let layers: [TransformerLayerWeights]
    public let output: LinearLayerWeights
    
    public init(config: ModelConfig,
                tokenEmbeddings: Tensor<Float>,
                layers: [TransformerLayerWeights],
                output: LinearLayerWeights) {
        precondition(tokenEmbeddings.shape == TensorShape(config.vocabSize, config.hiddenDim),
                     "Embedding matrix must be [vocabSize, hiddenDim]")
        precondition(layers.count == config.numLayers,
                     "Expected \(config.numLayers) layers, got \(layers.count)")
        self.config = config
        self.tokenEmbeddings = tokenEmbeddings
        self.layers = layers
        self.output = output
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
                key: makeProjection(outputDim: config.hiddenDim),
                value: makeProjection(outputDim: config.hiddenDim),
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
        
        // 2. Write Quantization Metadata
        var metadataOffset = data.count
        
        // Count all tensors (embeddings + all layer weights)
        var tensorCount: UInt32 = 1  // embeddings
        tensorCount += UInt32(layers.count * 8)  // 8 tensors per layer (Q/K/V/O + biases, FFN up/down + biases)
        tensorCount += 2  // output weights + bias
        
        data.append(Data(bytes: &tensorCount, count: 4))
        
        // Write metadata for each quantized tensor
        func writeQuantMetadata(name: String, tensor: QuantizedTensor) {
            // Tensor ID (name)
            let nameData = name.utf8
            var nameLength = UInt32(nameData.count)
            data.append(Data(bytes: &nameLength, count: 4))
            data.append(contentsOf: nameData)
            
            // Precision
            var precision: UInt8 = 1  // INT8
            data.append(Data(bytes: &precision, count: 1))
            
            // Mode
            var mode: UInt8 = 2  // perChannel
            data.append(Data(bytes: &mode, count: 1))
            
            // Scales
            var scalesCount = UInt32(tensor.scales.count)
            data.append(Data(bytes: &scalesCount, count: 4))
            for var scale in tensor.scales {
                data.append(Data(bytes: &scale, count: 4))
            }
            
            // Zero points (may be nil for symmetric)
            let zpCount = UInt32(tensor.zeroPoints?.count ?? 0)
            var zpCountVar = zpCount
            data.append(Data(bytes: &zpCountVar, count: 4))
            if let zps = tensor.zeroPoints {
                for var zp in zps {
                    data.append(Data(bytes: &zp, count: 1))
                }
            }
        }
        
        // Write metadata for all quantized layers
        for (layerIdx, layer) in layers.enumerated() {
            writeQuantMetadata(name: "layer_\(layerIdx)_attn_q", tensor: layer.attention.query.weights)
            writeQuantMetadata(name: "layer_\(layerIdx)_attn_k", tensor: layer.attention.key.weights)
            writeQuantMetadata(name: "layer_\(layerIdx)_attn_v", tensor: layer.attention.value.weights)
            writeQuantMetadata(name: "layer_\(layerIdx)_attn_o", tensor: layer.attention.output.weights)
            writeQuantMetadata(name: "layer_\(layerIdx)_ffn_up", tensor: layer.feedForward.up.weights)
            writeQuantMetadata(name: "layer_\(layerIdx)_ffn_down", tensor: layer.feedForward.down.weights)
        }
        writeQuantMetadata(name: "output", tensor: output.weights)
        
        // Pad metadata to 4KB
        let metadataSize = data.count - metadataOffset
        let paddedMetadataSize = ((metadataSize + 4095) / 4096) * 4096
        data.append(Data(count: paddedMetadataSize - metadataSize))
        
        // 3. Write Tensor Index
        var indexOffset = data.count
        data.append(Data(bytes: &tensorCount, count: 4))
        
        // Helper to write tensor index entry
        func writeTensorIndex(name: String, shape: TensorShape, dataSize: Int, dataOffset: Int) {
            let nameData = name.utf8
            var nameLength = UInt32(nameData.count)
            data.append(Data(bytes: &nameLength, count: 4))
            data.append(contentsOf: nameData)
            
            var dimCount = UInt32(shape.dimensions.count)
            data.append(Data(bytes: &dimCount, count: 4))
            for var dim in shape.dimensions {
                var dim32 = Int32(dim)
                data.append(Data(bytes: &dim32, count: 4))
            }
            
            var offset64 = UInt64(dataOffset)
            var size64 = UInt64(dataSize)
            data.append(Data(bytes: &offset64, count: 8))
            data.append(Data(bytes: &size64, count: 8))
        }
        
        // Calculate offsets (will fill in later)
        // For now, write placeholder index
        let indexStart = data.count
        
        // Embeddings
        writeTensorIndex(
            name: "embeddings",
            shape: tokenEmbeddings.shape,
            dataSize: tokenEmbeddings.data.count * 4,  // Float32
            dataOffset: 0  // Placeholder
        )
        
        // Layer weights
        for (layerIdx, layer) in layers.enumerated() {
            writeTensorIndex(name: "layer_\(layerIdx)_attn_q", shape: layer.attention.query.weights.shape,
                           dataSize: layer.attention.query.weights.data.count, dataOffset: 0)
            writeTensorIndex(name: "layer_\(layerIdx)_attn_k", shape: layer.attention.key.weights.shape,
                           dataSize: layer.attention.key.weights.data.count, dataOffset: 0)
            writeTensorIndex(name: "layer_\(layerIdx)_attn_v", shape: layer.attention.value.weights.shape,
                           dataSize: layer.attention.value.weights.data.count, dataOffset: 0)
            writeTensorIndex(name: "layer_\(layerIdx)_attn_o", shape: layer.attention.output.weights.shape,
                           dataSize: layer.attention.output.weights.data.count, dataOffset: 0)
            writeTensorIndex(name: "layer_\(layerIdx)_ffn_up", shape: layer.feedForward.up.weights.shape,
                           dataSize: layer.feedForward.up.weights.data.count, dataOffset: 0)
            writeTensorIndex(name: "layer_\(layerIdx)_ffn_down", shape: layer.feedForward.down.weights.shape,
                           dataSize: layer.feedForward.down.weights.data.count, dataOffset: 0)
        }
        writeTensorIndex(name: "output", shape: output.weights.shape,
                       dataSize: output.weights.data.count, dataOffset: 0)
        
        // Pad index to 4KB
        let indexSize = data.count - indexOffset
        let paddedIndexSize = ((indexSize + 4095) / 4096) * 4096
        data.append(Data(count: paddedIndexSize - indexSize))
        
        // 4. Write Weight Data Blobs (4KB aligned)
        // Embeddings
        var dataOffset = data.count
        for var value in tokenEmbeddings.data {
            data.append(Data(bytes: &value, count: 4))
        }
        let embeddingSize = data.count - dataOffset
        let paddedEmbeddingSize = ((embeddingSize + 4095) / 4096) * 4096
        data.append(Data(count: paddedEmbeddingSize - embeddingSize))
        
        // Layer weights (quantized INT8)
        for layer in layers {
            let tensors = [
                layer.attention.query.weights,
                layer.attention.key.weights,
                layer.attention.value.weights,
                layer.attention.output.weights,
                layer.feedForward.up.weights,
                layer.feedForward.down.weights
            ]
            
            for tensor in tensors {
                let start = data.count
                for var value in tensor.data {
                    data.append(Data(bytes: &value, count: 1))
                }
                let size = data.count - start
                let paddedSize = ((size + 4095) / 4096) * 4096
                data.append(Data(count: paddedSize - size))
            }
        }
        
        // Output weights
        let outStart = data.count
        for var value in output.weights.data {
            data.append(Data(bytes: &value, count: 1))
        }
        let outSize = data.count - outStart
        let paddedOutSize = ((outSize + 4095) / 4096) * 4096
        data.append(Data(count: paddedOutSize - outSize))
        
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
        
        // Skip to weight data section (4KB aligned)
        var dataOffset = ((offset + 4095) / 4096) * 4096
        
        // 4. Load Weight Data
        // Helper to load quantized tensor
        func loadQuantizedTensor(name: String) -> QuantizedTensor {
            guard let index = tensorIndex[name] else {
                fatalError("Missing tensor: \(name)")
            }
            guard let metadata = quantMetadata[name] else {
                fatalError("Missing metadata: \(name)")
            }
            
            let shape = TensorShape(index.shape)
            var quantData: [Int8] = []
            
            // Read INT8 data from current dataOffset
            for i in 0..<index.dataSize {
                let value = Int8(bitPattern: data[dataOffset + i])
                quantData.append(value)
            }
            
            // Move to next tensor (4KB aligned)
            dataOffset = ((dataOffset + index.dataSize + 4095) / 4096) * 4096
            
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
        
        // Load embeddings (Float32)
        let embIndex = tensorIndex["embeddings"]!
        var embeddingData: [Float] = []
        for i in 0..<(embIndex.dataSize / 4) {
            let floatOffset = dataOffset + i * 4
            let bytes = data[floatOffset..<(floatOffset + 4)]
            let value = bytes.withUnsafeBytes { ptr in
                ptr.loadUnaligned(as: Float.self)
            }
            embeddingData.append(value)
        }
        let embeddings = Tensor<Float>(shape: TensorShape(embIndex.shape), data: embeddingData)
        dataOffset = ((dataOffset + embIndex.dataSize + 4095) / 4096) * 4096
        
        // Load layer weights
        var layers: [TransformerLayerWeights] = []
        for layerIdx in 0..<config.numLayers {
            let qWeights = loadQuantizedTensor(name: "layer_\(layerIdx)_attn_q")
            let kWeights = loadQuantizedTensor(name: "layer_\(layerIdx)_attn_k")
            let vWeights = loadQuantizedTensor(name: "layer_\(layerIdx)_attn_v")
            let oWeights = loadQuantizedTensor(name: "layer_\(layerIdx)_attn_o")
            let upWeights = loadQuantizedTensor(name: "layer_\(layerIdx)_ffn_up")
            let downWeights = loadQuantizedTensor(name: "layer_\(layerIdx)_ffn_down")
            
            let attention = AttentionProjectionWeights(
                query: LinearLayerWeights(weights: qWeights, bias: nil),
                key: LinearLayerWeights(weights: kWeights, bias: nil),
                value: LinearLayerWeights(weights: vWeights, bias: nil),
                output: LinearLayerWeights(weights: oWeights, bias: nil)
            )
            
            let feedForward = FeedForwardWeights(
                up: LinearLayerWeights(weights: upWeights, bias: nil),
                down: LinearLayerWeights(weights: downWeights, bias: nil)
            )
            
            layers.append(TransformerLayerWeights(attention: attention, feedForward: feedForward))
        }
        
        // Load output weights
        let outputWeights = loadQuantizedTensor(name: "output")
        let output = LinearLayerWeights(weights: outputWeights, bias: nil)
        
        return ModelWeights(
            config: config,
            tokenEmbeddings: embeddings,
            layers: layers,
            output: output
        )
    }
}
