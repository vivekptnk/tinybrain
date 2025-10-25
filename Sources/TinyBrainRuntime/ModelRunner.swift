/// Model runner for streaming token generation
///
/// **TB-004 Phase 5:** Efficient streaming inference with KV cache reuse
///
/// ## How It Works
///
/// Traditional (slow):
/// ```
/// Token 0: Forward pass → logits
/// Token 1: Forward pass (recompute token 0!) → logits  ← Wasteful!
/// Token 2: Forward pass (recompute 0,1!) → logits      ← Very wasteful!
/// ```
///
/// ModelRunner (fast):
/// ```
/// Token 0: Forward pass → cache K₀,V₀ → logits
/// Token 1: Reuse K₀,V₀, compute K₁,V₁ → logits  ← Fast!
/// Token 2: Reuse K₀,K₁,V₀,V₁, compute K₂,V₂ → logits  ← Fast!
/// ```
///
/// **Result:** O(n) instead of O(n²) complexity!

import Foundation

/// Configuration for model inference
public struct ModelConfig {
    /// Number of transformer layers
    public let numLayers: Int
    
    /// Hidden dimension size
    public let hiddenDim: Int
    
    /// Number of attention heads
    public let numHeads: Int
    
    /// Vocabulary size
    public let vocabSize: Int
    
    /// Maximum sequence length
    public let maxSeqLen: Int
    
    public init(numLayers: Int, hiddenDim: Int, numHeads: Int, vocabSize: Int, maxSeqLen: Int = 2048) {
        self.numLayers = numLayers
        self.hiddenDim = hiddenDim
        self.numHeads = numHeads
        self.vocabSize = vocabSize
        self.maxSeqLen = maxSeqLen
    }
}

/// Model runner for streaming inference
///
/// **TB-004:** Manages KV cache and incremental token generation
public final class ModelRunner {
    /// Model configuration
    public let config: ModelConfig
    
    /// Backing weights (quantized INT8)
    public let weights: ModelWeights
    
    /// KV cache for attention
    public let kvCache: KVCache
    
    /// Current position in sequence
    public private(set) var currentPosition: Int = 0
    
    /// Last computed logits (used for streaming/verifications)
    private var lastLogits: Tensor<Float>?
    
    /// Initialize model runner with deterministic toy weights (useful for demos/tests)
    public convenience init(config: ModelConfig) {
        self.init(weights: ModelWeights.makeToyModel(config: config))
    }
    
    /// Initialize model runner with explicit weights
    public init(weights: ModelWeights) {
        self.weights = weights
        self.config = weights.config
        self.kvCache = KVCache(
            numLayers: config.numLayers,
            hiddenDim: config.hiddenDim,
            maxTokens: config.maxSeqLen,
            pageSize: 16
        )
    }
    
    /// Generate next token logits using cached context
    ///
    /// **REVIEW HITLER FIX:** Now implements REAL attention with KV cache reuse!
    ///
    /// Example:
    /// ```swift
    /// let runner = ModelRunner(config: config)
    /// let logits = runner.step(tokenId: 42)
    /// let nextToken = sample(logits)  // Sample from distribution
    /// ```
    ///
    /// - Parameter tokenId: Input token ID
    /// - Returns: Logits for next token [vocabSize]
    public func step(tokenId: Int) -> Tensor<Float> {
        // 1. Embed input token
        var hiddenRow = weights.embedding(for: tokenId).asRowMatrix()
        
        // 2. Transformer layers with cached attention + quantized matmuls
        for (layerIndex, layerWeights) in weights.layers.enumerated() {
            hiddenRow = applyLayer(hiddenRow, layerWeights: layerWeights, layerIndex: layerIndex)
        }
        
        // 3. Output projection to logits
        let logitsRow = weights.output.apply(toRow: hiddenRow)
        let logits = logitsRow.squeezedRowVector()
        
        currentPosition += 1
        lastLogits = logits
        
        return logits
    }
    
    /// Reset state for new sequence
    ///
    /// Clears KV cache and resets position to 0
    public func reset() {
        kvCache.clear()
        currentPosition = 0
        lastLogits = nil
    }
    
    /// Generate stream of tokens using AsyncSequence
    ///
    /// **TB-004:** SwiftUI-friendly streaming API
    ///
    /// Example:
    /// ```swift
    /// for try await tokenId in runner.generateStream(prompt: [1, 2, 3]) {
    ///     print(tokenId, terminator: " ")
    /// }
    /// ```
    ///
    /// - Parameters:
    ///   - prompt: Initial token IDs
    ///   - maxTokens: Maximum tokens to generate
    /// - Returns: AsyncThrowingStream of token IDs
    public func generateStream(prompt: [Int], maxTokens: Int = 100) -> AsyncThrowingStream<Int, Error> {
        AsyncThrowingStream { continuation in
            Task {
                var currentToken = prompt.last ?? 0
                
                if !prompt.isEmpty {
                    for token in prompt.dropLast() {
                        _ = self.step(tokenId: token)
                    }
                }
                
                var generated = 0
                while generated < maxTokens {
                    let logits = self.step(tokenId: currentToken)
                    let nextToken = self.sampleToken(from: logits)
                    continuation.yield(nextToken)
                    currentToken = nextToken
                    generated += 1
                }
                
                continuation.finish()
            }
        }
    }
    
    /// Sample token from logits distribution
    ///
    /// **TB-004 MVP:** Simple argmax sampling
    /// Real implementation would use top-k, top-p, temperature
    private func sampleToken(from logits: Tensor<Float>) -> Int {
        let probabilities = logits.softmax()
        var cumulative: Float = 0.0
        let threshold = Float.random(in: 0..<1)
        
        for (index, value) in probabilities.data.enumerated() {
            cumulative += value
            if threshold <= cumulative {
                return index
            }
        }
        
        return config.vocabSize - 1
    }
}

// MARK: - Private helpers

private extension ModelRunner {
    func applyLayer(_ hiddenRow: Tensor<Float>,
                    layerWeights: TransformerLayerWeights,
                    layerIndex: Int) -> Tensor<Float> {
        let attentionOutput = attention(hiddenRow: hiddenRow,
                                        layerWeights: layerWeights.attention,
                                        layerIndex: layerIndex)
        let residual = hiddenRow + attentionOutput
        
        let ffnUp = layerWeights.feedForward.up.apply(toRow: residual).gelu()
        let ffnDown = layerWeights.feedForward.down.apply(toRow: ffnUp)
        return residual + ffnDown
    }
    
    func attention(hiddenRow: Tensor<Float>,
                   layerWeights: AttentionProjectionWeights,
                   layerIndex: Int) -> Tensor<Float> {
        let query = layerWeights.query.apply(toRow: hiddenRow)
        let keyVec = layerWeights.key.apply(toRow: hiddenRow).squeezedRowVector()
        let valueVec = layerWeights.value.apply(toRow: hiddenRow).squeezedRowVector()
        
        kvCache.append(layer: layerIndex, key: keyVec, value: valueVec, position: currentPosition)
        
        let sequenceLength = currentPosition + 1
        let allKeys = kvCache.getKeys(layer: layerIndex, range: 0..<sequenceLength)
        let allValues = kvCache.getValues(layer: layerIndex, range: 0..<sequenceLength)
        
        let scalingFactor = 1.0 / sqrt(max(1.0, Float(config.hiddenDim) / Float(config.numHeads)))
        let scores = (query.matmul(allKeys.transpose())) * scalingFactor
        let attentionWeights = scores.softmax()
        let context = attentionWeights.matmul(allValues)
        
        return layerWeights.output.apply(toRow: context)
    }
}
