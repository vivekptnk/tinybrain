import Foundation

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
}
