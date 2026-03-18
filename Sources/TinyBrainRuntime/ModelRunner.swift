/// Model runner for streaming token generation
///
/// **TB-004 Phase 5:** Efficient streaming inference with KV cache reuse
///
/// ## How It Works
///
/// Traditional (slow):
/// ```
/// Token 0: Forward pass -> logits
/// Token 1: Forward pass (recompute token 0!) -> logits  <- Wasteful!
/// Token 2: Forward pass (recompute 0,1!) -> logits      <- Very wasteful!
/// ```
///
/// ModelRunner (fast):
/// ```
/// Token 0: Forward pass -> cache K0,V0 -> logits
/// Token 1: Reuse K0,V0, compute K1,V1 -> logits  <- Fast!
/// Token 2: Reuse K0,K1,V0,V1, compute K2,V2 -> logits  <- Fast!
/// ```
///
/// **Result:** O(n) instead of O(n^2) complexity!

import Foundation
import Combine

/// Configuration for model inference
public struct ModelConfig: Codable {
    /// Number of transformer layers
    public let numLayers: Int

    /// Hidden dimension size
    public let hiddenDim: Int

    /// Number of attention heads (for queries)
    public let numHeads: Int

    /// Number of key-value heads (for GQA/MQA, defaults to numHeads for MHA)
    public let numKVHeads: Int

    /// Vocabulary size
    public let vocabSize: Int

    /// Maximum sequence length
    public let maxSeqLen: Int

    /// FFN intermediate dimension (defaults to 4 * hiddenDim)
    public let intermediateDim: Int

    /// Computed: KV dimension (hiddenDim / numHeads * numKVHeads)
    public var kvDim: Int {
        return (hiddenDim / numHeads) * numKVHeads
    }

    /// Computed: head dimension
    public var headDim: Int {
        return hiddenDim / numHeads
    }

    public init(numLayers: Int, hiddenDim: Int, numHeads: Int, vocabSize: Int, maxSeqLen: Int = 2048, numKVHeads: Int? = nil, intermediateDim: Int? = nil) {
        self.numLayers = numLayers
        self.hiddenDim = hiddenDim
        self.numHeads = numHeads
        self.numKVHeads = numKVHeads ?? numHeads  // Default to MHA if not specified
        self.vocabSize = vocabSize
        self.maxSeqLen = maxSeqLen
        self.intermediateDim = intermediateDim ?? (4 * hiddenDim)
    }

    enum CodingKeys: String, CodingKey {
        case numLayers, hiddenDim, numHeads, numKVHeads, vocabSize, maxSeqLen, intermediateDim
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        numLayers = try container.decode(Int.self, forKey: .numLayers)
        hiddenDim = try container.decode(Int.self, forKey: .hiddenDim)
        numHeads = try container.decode(Int.self, forKey: .numHeads)
        numKVHeads = try container.decodeIfPresent(Int.self, forKey: .numKVHeads) ?? numHeads
        vocabSize = try container.decode(Int.self, forKey: .vocabSize)
        maxSeqLen = try container.decode(Int.self, forKey: .maxSeqLen)
        intermediateDim = try container.decodeIfPresent(Int.self, forKey: .intermediateDim) ?? (4 * hiddenDim)
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

    /// Optional observer for X-Ray visualization (TB-010)
    /// When nil, zero overhead. When set, receives attention weights,
    /// hidden state norms, and logits at each inference step.
    public weak var observer: InferenceObserver?

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
            hiddenDim: config.kvDim,  // Use KV dimension for GQA support
            maxTokens: config.maxSeqLen,
            pageSize: 16
        )
    }

    /// Generate next token logits using cached context
    ///
    /// - Parameter tokenId: Input token ID
    /// - Returns: Logits for next token [vocabSize]
    public func step(tokenId: Int) -> Tensor<Float> {
        // 1. Embed input token
        var hiddenRow = weights.embedding(for: tokenId).asRowMatrix()

        // Diagnostic removed (was corrupting KV cache)

        // 2. Transformer layers with cached attention + quantized matmuls
        for (layerIndex, layerWeights) in weights.layers.enumerated() {
            hiddenRow = applyLayer(hiddenRow, layerWeights: layerWeights, layerIndex: layerIndex)
        }

        // 3. Final RMSNorm before output projection (LLaMA-style)
        if let finalNorm = weights.finalNormWeights {
            hiddenRow = hiddenRow.rmsNorm(weight: finalNorm)
        }

        // X-Ray hook: final hidden state (post-norm, pre-projection — the embedding vector)
        observer?.didComputeFinalHiddenState(hiddenRow.squeezedRowVector().data, position: currentPosition)

        // 4. Output projection to logits
        let logitsRow = weights.output.apply(toRow: hiddenRow)
        let logits = logitsRow.squeezedRowVector()

        // X-Ray hook: full logit distribution (fires last, signals step complete)
        observer?.didComputeLogits(logits: logits.data, position: currentPosition)

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

    // MARK: - Embedding extraction

    /// Extract the text embedding for a sequence of token IDs.
    ///
    /// Runs a forward pass through all transformer layers and final RMSNorm,
    /// then returns the hidden state vector **without** computing the output
    /// projection to logits — cheaper than a full `step()` when you only
    /// need the embedding.
    ///
    /// The returned tensor has shape `[1, hiddenDim]`.
    ///
    /// - Parameter tokenIds: Input token IDs to encode
    /// - Returns: Final hidden state tensor of shape `[1, hiddenDim]`
    public func extractEmbedding(for tokenIds: [Int]) -> Tensor<Float> {
        reset()

        var hiddenRow = Tensor<Float>(shape: TensorShape(1, config.hiddenDim),
                                      data: [Float](repeating: 0, count: config.hiddenDim))

        for tokenId in tokenIds {
            let clampedId = max(0, min(tokenId, config.vocabSize - 1))
            hiddenRow = weights.embedding(for: clampedId).asRowMatrix()

            for (layerIndex, layerWeights) in weights.layers.enumerated() {
                hiddenRow = applyLayer(hiddenRow, layerWeights: layerWeights, layerIndex: layerIndex)
            }

            currentPosition += 1
        }

        // Final RMSNorm — this is the embedding representation
        if let finalNorm = weights.finalNormWeights {
            hiddenRow = hiddenRow.rmsNorm(weight: finalNorm)
        }

        return hiddenRow
    }

    // MARK: - Legacy API (TB-004 compatibility)

    /// Generate stream of tokens (simple version)
    ///
    /// **Deprecated:** Use `generateStream(prompt:config:)` instead
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
        // X-Ray hook: hidden state magnitude entering this layer
        observer?.didEnterLayer(
            layerIndex: layerIndex,
            hiddenStateNorm: sqrt(hiddenRow.data.reduce(0) { $0 + $1 * $1 }),
            position: currentPosition
        )

        // Pre-attention RMSNorm (LLaMA-style pre-norm architecture)
        let normedForAttn: Tensor<Float>
        if let normWeights = layerWeights.inputNormWeights {
            normedForAttn = hiddenRow.rmsNorm(weight: normWeights)
        } else {
            normedForAttn = hiddenRow
        }

        let attentionOutput = attention(hiddenRow: normedForAttn,
                                        layerWeights: layerWeights.attention,
                                        layerIndex: layerIndex)
        let residual1 = hiddenRow + attentionOutput

        // Pre-FFN RMSNorm
        let normedForFFN: Tensor<Float>
        if let normWeights = layerWeights.postAttentionNormWeights {
            normedForFFN = residual1.rmsNorm(weight: normWeights)
        } else {
            normedForFFN = residual1
        }

        // Gated FFN: down_proj(silu(gate_proj(x)) * up_proj(x))
        // Falls back to GELU-based FFN if no gate projection (toy models)
        let ffnOutput: Tensor<Float>
        if let gate = layerWeights.feedForward.gate {
            let gateOut = gate.apply(toRow: normedForFFN).silu()
            let upOut = layerWeights.feedForward.up.apply(toRow: normedForFFN)
            let gated = gateOut * upOut
            ffnOutput = layerWeights.feedForward.down.apply(toRow: gated)
        } else {
            let ffnUp = layerWeights.feedForward.up.apply(toRow: normedForFFN).gelu()
            ffnOutput = layerWeights.feedForward.down.apply(toRow: ffnUp)
        }

        return residual1 + ffnOutput
    }

    /// Apply Rotary Position Embeddings (RoPE) to a vector of shape [dim]
    /// where dim = numHeads * headDim. RoPE is applied per-head.
    func applyRoPE(_ vec: [Float], headDim: Int, numHeads: Int, position: Int) -> [Float] {
        var result = vec
        for head in 0..<numHeads {
            let offset = head * headDim
            for i in stride(from: 0, to: headDim, by: 2) {
                let freqIdx = Float(i) / Float(headDim)
                let theta = pow(10000.0, -freqIdx)
                let angle = Float(position) * theta

                let cosA = cos(angle)
                let sinA = sin(angle)

                let x0 = vec[offset + i]
                let x1 = vec[offset + i + 1]

                result[offset + i]     = x0 * cosA - x1 * sinA
                result[offset + i + 1] = x0 * sinA + x1 * cosA
            }
        }
        return result
    }

    func attention(hiddenRow: Tensor<Float>,
                   layerWeights: AttentionProjectionWeights,
                   layerIndex: Int) -> Tensor<Float> {
        var query = layerWeights.query.apply(toRow: hiddenRow)
        let keyRow = layerWeights.key.apply(toRow: hiddenRow)
        let valueRow = layerWeights.value.apply(toRow: hiddenRow)

        // Apply RoPE to query and key (not value)
        let headDim = config.headDim
        let queryData = applyRoPE(query.squeezedRowVector().data, headDim: headDim,
                                   numHeads: config.numHeads, position: currentPosition)
        query = Tensor<Float>(shape: query.shape, data: queryData)

        let keyData = applyRoPE(keyRow.squeezedRowVector().data, headDim: headDim,
                                 numHeads: config.numKVHeads, position: currentPosition)
        let keyVec = Tensor<Float>(shape: TensorShape(config.kvDim), data: keyData)
        let valueVec = valueRow.squeezedRowVector()

        kvCache.append(layer: layerIndex, key: keyVec, value: valueVec, position: currentPosition)

        let sequenceLength = currentPosition + 1
        let allKeys = kvCache.getKeys(layer: layerIndex, range: 0..<sequenceLength)
        let allValues = kvCache.getValues(layer: layerIndex, range: 0..<sequenceLength)

        let scalingFactor = 1.0 / sqrt(Float(headDim))

        // Per-head attention (correct multi-head attention)
        // Each head independently computes: softmax(q_h @ K_h.T / sqrt(d)) @ V_h
        let numHeads = config.numHeads
        let numKVHeads = config.numKVHeads
        let repeats = numHeads / numKVHeads

        // Query is [1, numHeads * headDim], squeeze to [numHeads * headDim]
        let qFlat = query.squeezedRowVector().data
        // Keys: [seqLen, numKVHeads * headDim], Values: [seqLen, numKVHeads * headDim]
        let kData = allKeys.data
        let vData = allValues.data
        let kvDim = numKVHeads * headDim

        // Output: [numHeads * headDim]
        var contextData = [Float](repeating: 0, count: numHeads * headDim)

        // Collect all attention weights for X-Ray (flat across heads)
        var allAttnWeights = [Float]()

        for head in 0..<numHeads {
            // Which KV head does this query head use?
            let kvHead = head / repeats

            // Extract q_h: [headDim]
            let qOffset = head * headDim

            // Compute scores: q_h . k_h for each position
            var scores = [Float](repeating: 0, count: sequenceLength)
            for pos in 0..<sequenceLength {
                var dot: Float = 0
                let kOffset = pos * kvDim + kvHead * headDim
                for d in 0..<headDim {
                    dot += qFlat[qOffset + d] * kData[kOffset + d]
                }
                scores[pos] = dot * scalingFactor
            }

            // Softmax
            let maxScore = scores.max() ?? 0
            var expScores = scores.map { exp($0 - maxScore) }
            let sumExp = expScores.reduce(0, +)
            expScores = expScores.map { $0 / sumExp }

            allAttnWeights.append(contentsOf: expScores)

            // Weighted sum of values
            let ctxOffset = head * headDim
            for pos in 0..<sequenceLength {
                let w = expScores[pos]
                let vOffset = pos * kvDim + kvHead * headDim
                for d in 0..<headDim {
                    contextData[ctxOffset + d] += w * vData[vOffset + d]
                }
            }
        }

        // X-Ray hook: attention weights (flattened: numHeads * seqLen)
        observer?.didComputeAttention(
            layerIndex: layerIndex,
            weights: allAttnWeights,
            position: currentPosition
        )

        // Context as [1, numHeads * headDim] row matrix for output projection
        let context = Tensor<Float>(shape: TensorShape(1, numHeads * headDim), data: contextData)

        return layerWeights.output.apply(toRow: context)
    }

    /// Repeat KV heads to match query heads for Grouped Query Attention.
    /// Input: [seqLen, kvDim] where kvDim = numKVHeads x headDim
    /// Output: [seqLen, hiddenDim] where hiddenDim = numHeads x headDim
    func repeatKVHeads(_ tensor: Tensor<Float>, headDim: Int, numKVHeads: Int, repeats: Int) -> Tensor<Float> {
        let seqLen = tensor.shape.dimensions[0]
        let kvDim = numKVHeads * headDim
        let outputDim = kvDim * repeats

        var result = [Float](repeating: 0, count: seqLen * outputDim)
        let src = tensor.data

        for s in 0..<seqLen {
            for kvHead in 0..<numKVHeads {
                let srcOffset = s * kvDim + kvHead * headDim
                for r in 0..<repeats {
                    let dstHead = kvHead * repeats + r
                    let dstOffset = s * outputDim + dstHead * headDim
                    for d in 0..<headDim {
                        result[dstOffset + d] = src[srcOffset + d]
                    }
                }
            }
        }

        return Tensor<Float>(shape: TensorShape(seqLen, outputDim), data: result)
    }
}
