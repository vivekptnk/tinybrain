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
import Combine

/// Configuration for model inference
public struct ModelConfig: Codable {
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
    /// **TB-005:** Production-ready streaming with rich configuration
    ///
    /// Example:
    /// ```swift
    /// let config = GenerationConfig(
    ///     maxTokens: 100,
    ///     sampler: SamplerConfig(temperature: 0.7, topK: 40),
    ///     stopTokens: [eosToken]
    /// )
    ///
    /// for try await output in runner.generateStream(prompt: [1, 2, 3], config: config) {
    ///     print("Token: \(output.tokenId), Prob: \(output.probability)")
    /// }
    /// ```
    ///
    /// - Parameters:
    ///   - prompt: Initial token IDs
    ///   - config: Generation configuration (max tokens, sampling, stop tokens)
    /// - Returns: AsyncThrowingStream of TokenOutput with rich metadata
    public func generateStream(
        prompt: [Int],
        config: GenerationConfig = GenerationConfig()
    ) -> AsyncThrowingStream<TokenOutput, Error> {
        AsyncThrowingStream { continuation in
            Task {
                // **REVIEW HITLER FIX:** Make config mutable for RNG state
                var mutableConfig = config
                
                // Sanitize prompt tokens (clip to valid range)
                let sanitizedPrompt = prompt.map { max(0, min($0, self.config.vocabSize - 1)) }
                
                // Process prompt (all except last token)
                var currentToken = sanitizedPrompt.last ?? 0
                if !sanitizedPrompt.isEmpty {
                    for token in sanitizedPrompt.dropLast() {
                        _ = self.step(tokenId: token)
                    }
                }
                
                // Track generation history for repetition penalty
                var history: [Int] = Array(sanitizedPrompt)
                
                // Generate tokens
                var generated = 0
                while generated < mutableConfig.maxTokens {
                    // Step 1: Forward pass to get logits
                    let logits = self.step(tokenId: currentToken)
                    
                    // Step 2: Sample next token using detailed sampler (correct probability & entropy)
                    let detailed = Sampler.sampleDetailed(
                        logits: logits,
                        config: &mutableConfig.sampler,
                        history: history
                    )
                    
                    // Step 3: Create output with metadata
                    let strategySummary: String? = {
                        var parts: [String] = []
                        parts.append(String(format: "temp=%.2f", mutableConfig.sampler.temperature))
                        if let k = mutableConfig.sampler.topK { parts.append("topK=\(k)") }
                        if let p = mutableConfig.sampler.topP { parts.append(String(format: "topP=%.2f", p)) }
                        if mutableConfig.sampler.repetitionPenalty != 1.0 { parts.append(String(format: "penalty=%.2f", mutableConfig.sampler.repetitionPenalty)) }
                        return parts.isEmpty ? nil : parts.joined(separator: ", ")
                    }()
                    
                    let output = TokenOutput(
                        tokenId: detailed.tokenId,
                        probability: detailed.probability,
                        entropy: detailed.entropy,
                        timestamp: Date(),
                        strategy: strategySummary,
                        energyJoules: nil
                    )
                    
                    // Step 5: Yield token to consumer
                    continuation.yield(output)
                    
                    // Step 6: Check for stop tokens
                    if mutableConfig.stopTokens.contains(detailed.tokenId) {
                        break
                    }
                    
                    // Step 7: Update state for next iteration
                    currentToken = detailed.tokenId
                    history.append(detailed.tokenId)
                    generated += 1
                }
                
                continuation.finish()
            }
        }
    }
    
    // MARK: - Legacy API (TB-004 compatibility)
    
    /// Generate stream of tokens (simple version)
    ///
    /// **Deprecated:** Use `generateStream(prompt:config:)` instead
    ///
    /// This legacy method is kept for backward compatibility with TB-004 tests.
    @available(*, deprecated, message: "Use generateStream(prompt:config:) instead")
    public func generateStream(prompt: [Int], maxTokens: Int = 100) -> AsyncThrowingStream<Int, Error> {
        let config = GenerationConfig(maxTokens: maxTokens)
        return AsyncThrowingStream { continuation in
            Task {
                for try await output in self.generateStream(prompt: prompt, config: config) {
                    continuation.yield(output.tokenId)
                }
                continuation.finish()
            }
        }
    }
    
    /// Combine publisher wrapper for `generateStream` for UI pipelines
    ///
    /// Bridges the AsyncThrowingStream into a `AnyPublisher<TokenOutput, Error>`.
    public func generatePublisher(
        prompt: [Int],
        config: GenerationConfig = GenerationConfig()
    ) -> AnyPublisher<TokenOutput, Error> {
        let subject = PassthroughSubject<TokenOutput, Error>()
        Task { [weak self] in
            guard let self = self else { return }
            do {
                for try await output in self.generateStream(prompt: prompt, config: config) {
                    subject.send(output)
                }
                subject.send(completion: .finished)
            } catch {
                subject.send(completion: .failure(error))
            }
        }
        return subject.eraseToAnyPublisher()
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
