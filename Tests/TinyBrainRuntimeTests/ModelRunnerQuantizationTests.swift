import XCTest
@testable import TinyBrainRuntime

final class ModelRunnerQuantizationTests: XCTestCase {
    
    func testQuantizedRunnerMatchesFloatReference() {
        let config = ModelConfig(
            numLayers: 2,
            hiddenDim: 32,
            numHeads: 4,
            vocabSize: 64,
            maxSeqLen: 64
        )
        
        var rng: any RandomNumberGenerator = SeededGenerator(seed: 2025)
        let floatWeights = FloatModelWeights.random(config: config, rng: &rng)
        let quantizedWeights = floatWeights.quantized()
        
        let quantizedRunner = ModelRunner(weights: quantizedWeights)
        let floatRunner = FloatReferenceRunner(weights: floatWeights)
        
        let prompt = Array(0..<8)
        var quantizedLogits = Tensor<Float>.zeros(shape: TensorShape(config.vocabSize))
        var floatLogits = quantizedLogits
        
        for token in prompt {
            quantizedLogits = quantizedRunner.step(tokenId: token)
            floatLogits = floatRunner.step(tokenId: token)
        }
        
        let error = relativeErrorForQuantization(floatLogits, quantizedLogits)
        XCTAssertLessThan(error, 0.05, "INT8 runner should stay within 5% of FP32 reference (got \(error))")
    }
}

// MARK: - Float reference model (for accuracy tests)

private struct FloatLinearLayer {
    let weights: Tensor<Float>
    let bias: Tensor<Float>?
    
    func apply(toRow input: Tensor<Float>) -> Tensor<Float> {
        var output = input.matmul(weights)
        if let bias = bias {
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

private struct FloatAttentionWeights {
    let query: FloatLinearLayer
    let key: FloatLinearLayer
    let value: FloatLinearLayer
    let output: FloatLinearLayer
}

private struct FloatFeedForwardWeights {
    let up: FloatLinearLayer
    let down: FloatLinearLayer
}

private struct FloatTransformerLayer {
    let attention: FloatAttentionWeights
    let feedForward: FloatFeedForwardWeights
}

private struct FloatModelWeights {
    let config: ModelConfig
    let tokenEmbeddings: Tensor<Float>
    let layers: [FloatTransformerLayer]
    let output: FloatLinearLayer
    
    static func random(config: ModelConfig, rng: inout any RandomNumberGenerator) -> FloatModelWeights {
        func makeWeights(rows: Int, cols: Int) -> Tensor<Float> {
            Tensor<Float>.random(
                shape: TensorShape(rows, cols),
                mean: 0,
                std: 0.02,
                using: &rng
            )
        }
        
        func makeBias(_ count: Int) -> Tensor<Float> {
            Tensor<Float>.random(shape: TensorShape(count), mean: 0, std: 0.02, using: &rng)
        }
        
        func linear(outputDim: Int) -> FloatLinearLayer {
            FloatLinearLayer(
                weights: makeWeights(rows: config.hiddenDim, cols: outputDim),
                bias: makeBias(outputDim)
            )
        }
        
        var layers: [FloatTransformerLayer] = []
        for _ in 0..<config.numLayers {
            let attention = FloatAttentionWeights(
                query: linear(outputDim: config.hiddenDim),
                key: linear(outputDim: config.hiddenDim),
                value: linear(outputDim: config.hiddenDim),
                output: linear(outputDim: config.hiddenDim)
            )
            
            let ffnHidden = config.hiddenDim * 4
            let feedForward = FloatFeedForwardWeights(
                up: linear(outputDim: ffnHidden),
                down: FloatLinearLayer(
                    weights: makeWeights(rows: ffnHidden, cols: config.hiddenDim),
                    bias: makeBias(config.hiddenDim)
                )
            )
            layers.append(FloatTransformerLayer(attention: attention, feedForward: feedForward))
        }
        
        let embeddings = Tensor<Float>.random(
            shape: TensorShape(config.vocabSize, config.hiddenDim),
            mean: 0,
            std: 0.02,
            using: &rng
        )
        
        let output = FloatLinearLayer(
            weights: makeWeights(rows: config.hiddenDim, cols: config.vocabSize),
            bias: Tensor<Float>.zeros(shape: TensorShape(config.vocabSize))
        )
        
        return FloatModelWeights(config: config, tokenEmbeddings: embeddings, layers: layers, output: output)
    }
    
    func quantized() -> ModelWeights {
        func quantize(_ layer: FloatLinearLayer) -> LinearLayerWeights {
            LinearLayerWeights(floatWeights: layer.weights, bias: layer.bias)
        }
        
        let attentionLayers = layers.map { layer in
            TransformerLayerWeights(
                attention: AttentionProjectionWeights(
                    query: quantize(layer.attention.query),
                    key: quantize(layer.attention.key),
                    value: quantize(layer.attention.value),
                    output: quantize(layer.attention.output)
                ),
                feedForward: FeedForwardWeights(
                    up: quantize(layer.feedForward.up),
                    down: quantize(layer.feedForward.down)
                )
            )
        }
        
        return ModelWeights(
            config: config,
            tokenEmbeddings: tokenEmbeddings,
            layers: attentionLayers,
            output: LinearLayerWeights(floatWeights: output.weights, bias: output.bias)
        )
    }
}

private final class FloatReferenceRunner {
    private let weights: FloatModelWeights
    private let kvCache: KVCache
    private var position: Int = 0
    
    init(weights: FloatModelWeights) {
        self.weights = weights
        self.kvCache = KVCache(
            numLayers: weights.config.numLayers,
            hiddenDim: weights.config.hiddenDim,
            maxTokens: weights.config.maxSeqLen,
            pageSize: 16
        )
    }
    
    func step(tokenId: Int) -> Tensor<Float> {
        var hiddenRow = weights.tokenEmbeddings.row(tokenId).asRowMatrix()
        
        for (layerIndex, layer) in weights.layers.enumerated() {
            hiddenRow = applyLayer(hiddenRow, layer: layer, layerIndex: layerIndex)
        }
        
        let logits = weights.output.apply(toRow: hiddenRow).squeezedRowVector()
        position += 1
        return logits
    }
    
    private func applyLayer(_ hidden: Tensor<Float>,
                            layer: FloatTransformerLayer,
                            layerIndex: Int) -> Tensor<Float> {
        let attnOut = attention(hidden, layer: layer.attention, layerIndex: layerIndex)
        let residual = hidden + attnOut
        let ffnUp = layer.feedForward.up.apply(toRow: residual).gelu()
        let ffnDown = layer.feedForward.down.apply(toRow: ffnUp)
        return residual + ffnDown
    }
    
    private func attention(_ hidden: Tensor<Float>,
                           layer: FloatAttentionWeights,
                           layerIndex: Int) -> Tensor<Float> {
        let query = layer.query.apply(toRow: hidden)
        let keyVec = layer.key.apply(toRow: hidden).squeezedRowVector()
        let valueVec = layer.value.apply(toRow: hidden).squeezedRowVector()
        
        kvCache.append(layer: layerIndex, key: keyVec, value: valueVec, position: position)
        
        let seqLen = position + 1
        let keys = kvCache.getKeys(layer: layerIndex, range: 0..<seqLen)
        let values = kvCache.getValues(layer: layerIndex, range: 0..<seqLen)
        let scale = 1.0 / sqrt(max(1.0, Float(weights.config.hiddenDim) / Float(weights.config.numHeads)))
        let scores = (query.matmul(keys.transpose())) * scale
        let weightsTensor = scores.softmax()
        let context = weightsTensor.matmul(values)
        return layer.output.apply(toRow: context)
    }
}

private func relativeErrorForQuantization(_ reference: Tensor<Float>, _ candidate: Tensor<Float>) -> Float {
    precondition(reference.shape == candidate.shape)
    var sumSquaredDiff: Float = 0
    var sumSquaredRef: Float = 0
    for i in 0..<reference.data.count {
        let diff = reference.data[i] - candidate.data[i]
        sumSquaredDiff += diff * diff
        sumSquaredRef += reference.data[i] * reference.data[i]
    }
    return sqrt(sumSquaredDiff) / max(sqrt(sumSquaredRef), Float.leastNonzeroMagnitude)
}
