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

    /// Model architecture family. Determines runtime-visible divergences
    /// from the default LLaMA-style path (e.g. Gemma's `RMSNorm * (1 + w)`
    /// and `sqrt(hiddenDim)` embedding scale, or Phi-2's LayerNorm and
    /// parallel residual). Decoded as "llama" for older .tbf files that
    /// omit the field.
    public let architecture: String

    /// Fraction of head dimensions that receive RoPE rotation.
    /// 1.0 = full RoPE (LLaMA, Gemma). 0.4 = Phi-2 partial RoPE.
    /// Stored in the TBF header so the runtime can compute `rotaryDims`
    /// without knowing the architecture.
    public let partialRotaryFactor: Float

    /// RoPE frequency base (`rope_theta` in HuggingFace configs).
    /// Defaults to LLaMA/TinyLlama/Gemma's 10000.0 for legacy .tbf files.
    public let ropeTheta: Float

    /// RMSNorm numerical stability constant (`rms_norm_eps` in HuggingFace configs).
    /// Defaults to the historical TinyBrain value for legacy .tbf files.
    public let rmsNormEpsilon: Float

    /// Computed: KV dimension (hiddenDim / numHeads * numKVHeads)
    public var kvDim: Int {
        return (hiddenDim / numHeads) * numKVHeads
    }

    /// Computed: head dimension
    public var headDim: Int {
        return hiddenDim / numHeads
    }

    /// Computed: number of head dimensions that receive RoPE rotation.
    public var rotaryDims: Int {
        return Int(Float(headDim) * partialRotaryFactor)
    }

    /// Computed: true when architecture needs Gemma's post-embed scale and
    /// RMSNorm offset semantics.
    public var isGemmaStyle: Bool {
        return architecture == "gemma"
    }

    /// Computed: true when architecture uses Phi-2 semantics:
    /// LayerNorm (not RMSNorm), parallel attention+MLP residual, partial RoPE,
    /// and attention/FFN bias terms.
    public var isPhiStyle: Bool {
        return architecture == "phi"
    }

    public init(numLayers: Int, hiddenDim: Int, numHeads: Int, vocabSize: Int,
                maxSeqLen: Int = 2048, numKVHeads: Int? = nil, intermediateDim: Int? = nil,
                architecture: String = "llama", partialRotaryFactor: Float = 1.0) {
        self.init(
            numLayers: numLayers,
            hiddenDim: hiddenDim,
            numHeads: numHeads,
            vocabSize: vocabSize,
            maxSeqLen: maxSeqLen,
            numKVHeads: numKVHeads,
            intermediateDim: intermediateDim,
            architecture: architecture,
            partialRotaryFactor: partialRotaryFactor,
            ropeTheta: 10000.0,
            rmsNormEpsilon: 1e-5
        )
    }

    public init(numLayers: Int, hiddenDim: Int, numHeads: Int, vocabSize: Int,
                maxSeqLen: Int = 2048, numKVHeads: Int? = nil, intermediateDim: Int? = nil,
                architecture: String = "llama", partialRotaryFactor: Float = 1.0,
                ropeTheta: Float, rmsNormEpsilon: Float = 1e-5) {
        self.numLayers = numLayers
        self.hiddenDim = hiddenDim
        self.numHeads = numHeads
        self.numKVHeads = numKVHeads ?? numHeads
        self.vocabSize = vocabSize
        self.maxSeqLen = maxSeqLen
        self.intermediateDim = intermediateDim ?? (4 * hiddenDim)
        self.architecture = architecture
        self.partialRotaryFactor = partialRotaryFactor
        self.ropeTheta = ropeTheta
        self.rmsNormEpsilon = rmsNormEpsilon
    }

    enum CodingKeys: String, CodingKey {
        case numLayers, hiddenDim, numHeads, numKVHeads, vocabSize, maxSeqLen,
             intermediateDim, architecture, partialRotaryFactor, ropeTheta,
             rmsNormEpsilon
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
        architecture = try container.decodeIfPresent(String.self, forKey: .architecture) ?? "llama"
        partialRotaryFactor = try container.decodeIfPresent(Float.self, forKey: .partialRotaryFactor) ?? 1.0
        ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10000.0
        rmsNormEpsilon = try container.decodeIfPresent(Float.self, forKey: .rmsNormEpsilon) ?? 1e-5
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

        // Gemma scales the embedding by sqrt(hidden_dim) right after lookup.
        if config.isGemmaStyle {
            hiddenRow = hiddenRow * sqrt(Float(config.hiddenDim))
        }

        // 2. Transformer layers with cached attention + quantized matmuls
        for (layerIndex, layerWeights) in weights.layers.enumerated() {
            hiddenRow = applyLayer(hiddenRow, layerWeights: layerWeights, layerIndex: layerIndex)
        }

        // 3. Final norm before output projection.
        // LLaMA: RMSNorm  |  Gemma: RMSNorm*(1+w)  |  Phi-2: LayerNorm with bias
        if let finalNorm = weights.finalNormWeights {
            if config.isPhiStyle {
                hiddenRow = hiddenRow.layerNorm(weight: finalNorm, bias: weights.finalNormBias)
            } else if config.isGemmaStyle {
                hiddenRow = hiddenRow.rmsNormWithOffset(weight: finalNorm,
                                                        epsilon: config.rmsNormEpsilon)
            } else {
                hiddenRow = hiddenRow.rmsNorm(weight: finalNorm,
                                              epsilon: config.rmsNormEpsilon)
            }
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
            let task = Task {
                var mutableConfig = config

                self.reset()

                // Sanitize prompt tokens (clip to valid range)
                let sanitizedPrompt = prompt.map { max(0, min($0, self.config.vocabSize - 1)) }

                // Process prompt (all except last token)
                var currentToken = sanitizedPrompt.last ?? 0
                if !sanitizedPrompt.isEmpty {
                    for token in sanitizedPrompt.dropLast() {
                        if Task.isCancelled {
                            continuation.finish()
                            return
                        }
                        _ = self.step(tokenId: token)
                    }
                }

                // Track generation history for repetition penalty
                var history: [Int] = Array(sanitizedPrompt)

                // Generate tokens
                var generated = 0
                while generated < mutableConfig.maxTokens && !Task.isCancelled {
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
                    switch continuation.yield(output) {
                    case .terminated:
                        return
                    case .enqueued, .dropped:
                        break
                    @unknown default:
                        break
                    }

                    await Task.yield()
                    if Task.isCancelled {
                        continuation.finish()
                        return
                    }

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
            continuation.onTermination = { @Sendable _ in
                task.cancel()
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

            if config.isGemmaStyle {
                hiddenRow = hiddenRow * sqrt(Float(config.hiddenDim))
            }

            for (layerIndex, layerWeights) in weights.layers.enumerated() {
                hiddenRow = applyLayer(hiddenRow, layerWeights: layerWeights, layerIndex: layerIndex)
            }

            currentPosition += 1
        }

        // Final norm — this is the embedding representation
        if let finalNorm = weights.finalNormWeights {
            if config.isPhiStyle {
                hiddenRow = hiddenRow.layerNorm(weight: finalNorm, bias: weights.finalNormBias)
            } else if config.isGemmaStyle {
                hiddenRow = hiddenRow.rmsNormWithOffset(weight: finalNorm,
                                                        epsilon: config.rmsNormEpsilon)
            } else {
                hiddenRow = hiddenRow.rmsNorm(weight: finalNorm,
                                              epsilon: config.rmsNormEpsilon)
            }
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

        if config.isPhiStyle {
            // Phi-2: single LayerNorm, then attention and MLP computed in parallel
            // from the same normalized input. Both outputs added to residual together.
            let normed: Tensor<Float>
            if let w = layerWeights.inputNormWeights {
                normed = hiddenRow.layerNorm(weight: w, bias: layerWeights.inputNormBias)
            } else {
                normed = hiddenRow
            }
            let attnOut = attention(hiddenRow: normed, layerWeights: layerWeights.attention,
                                    layerIndex: layerIndex)
            let ffnUp = layerWeights.feedForward.up.apply(toRow: normed).gelu()
            let ffnOut = layerWeights.feedForward.down.apply(toRow: ffnUp)
            return hiddenRow + attnOut + ffnOut
        }

        // Standard sequential pre-norm (LLaMA / Gemma)

        // Pre-attention norm
        let normedForAttn: Tensor<Float>
        if let normWeights = layerWeights.inputNormWeights {
            normedForAttn = config.isGemmaStyle
                ? hiddenRow.rmsNormWithOffset(weight: normWeights,
                                              epsilon: config.rmsNormEpsilon)
                : hiddenRow.rmsNorm(weight: normWeights,
                                    epsilon: config.rmsNormEpsilon)
        } else {
            normedForAttn = hiddenRow
        }

        let attentionOutput = attention(hiddenRow: normedForAttn,
                                        layerWeights: layerWeights.attention,
                                        layerIndex: layerIndex)
        let residual1 = hiddenRow + attentionOutput

        // Pre-FFN norm
        let normedForFFN: Tensor<Float>
        if let normWeights = layerWeights.postAttentionNormWeights {
            normedForFFN = config.isGemmaStyle
                ? residual1.rmsNormWithOffset(weight: normWeights,
                                              epsilon: config.rmsNormEpsilon)
                : residual1.rmsNorm(weight: normWeights,
                                    epsilon: config.rmsNormEpsilon)
        } else {
            normedForFFN = residual1
        }

        // Gated FFN: down_proj(act(gate_proj(x)) * up_proj(x))
        // LLaMA/TinyLlama use SwiGLU (SiLU gate); Gemma uses GeGLU. Gemma's
        // config specifies hidden_act="gelu" — the *exact* erf variant. We must
        // use the exact GELU here (not `Tensor.gelu()`, which is the tanh
        // approximation): its cubic argument, `tanh(0.798·(x + 0.044715·x³))`,
        // overflows to NaN on the GPU for Gemma's BOS "massive activations"
        // (a gate outlier ≈14 pushes the argument past ~100, where Metal's tanh
        // returns NaN). That single NaN poisons the layer-7 FFN, propagates into
        // the KV cache, and collapses every prompt into degenerate output.
        // `erf` saturates cleanly for large |x|, so exact GELU is both faithful
        // to Gemma and numerically robust.
        let ffnOutput: Tensor<Float>
        if let gate = layerWeights.feedForward.gate {
            let gateProjected = gate.apply(toRow: normedForFFN)
            let gateOut = config.isGemmaStyle ? Self.exactGELU(gateProjected) : gateProjected.silu()
            let upOut = layerWeights.feedForward.up.apply(toRow: normedForFFN)
            let gated = gateOut * upOut
            ffnOutput = layerWeights.feedForward.down.apply(toRow: gated)
        } else {
            let ffnUp = layerWeights.feedForward.up.apply(toRow: normedForFFN).gelu()
            ffnOutput = layerWeights.feedForward.down.apply(toRow: ffnUp)
        }

        return residual1 + ffnOutput
    }

    /// Exact GELU: `0.5·x·(1 + erf(x/√2))`, computed on the CPU.
    ///
    /// Used for Gemma's GeGLU gate (its config declares `hidden_act="gelu"`,
    /// the exact erf variant). Unlike the tanh approximation in
    /// `Tensor.gelu()`, `erf` saturates to ±1 for large |x| without the cubic
    /// blow-up that overflows the Metal `tanh` kernel to NaN on Gemma's BOS
    /// massive activations.
    static func exactGELU(_ t: Tensor<Float>) -> Tensor<Float> {
        let invSqrt2 = 0.7071067811865476
        var d = t.data
        for i in 0..<d.count {
            let x = Double(d[i])
            d[i] = Float(0.5 * x * (1.0 + erf(x * invSqrt2)))
        }
        return Tensor<Float>(shape: t.shape, data: d)
    }

}

extension ModelRunner {
    /// Apply Rotary Position Embeddings (RoPE) to a vector of shape [numHeads * headDim].
    ///
    /// - Parameters:
    ///   - vec: Flat vector of shape `[numHeads * headDim]`.
    ///   - headDim: Size of each attention head.
    ///   - numHeads: Number of heads.
    ///   - position: Token position in the sequence.
    ///   - rotaryDims: How many dims per head to rotate (default = headDim for full RoPE).
    ///     Phi-2 uses `Int(headDim * 0.4)`; remaining dims are left unchanged.
    ///     The rotary span uses HuggingFace's rotate-half convention, pairing
    ///     dimension `d` with `d + rotaryDims / 2`.
    func applyRoPE(_ vec: [Float], headDim: Int, numHeads: Int, position: Int,
                   rotaryDims: Int? = nil) -> [Float] {
        var result = vec
        let requestedRotaryDims = rotaryDims ?? headDim
        precondition(requestedRotaryDims <= headDim, "rotaryDims must be <= headDim")

        // rotate-half needs complete pairs. Real checkpoints use even rotary
        // spans; tiny toy configs that request an odd span leave the unpaired
        // final rotary dimension unchanged.
        let rDims = requestedRotaryDims - (requestedRotaryDims % 2)
        let halfRotaryDims = rDims / 2

        for head in 0..<numHeads {
            let offset = head * headDim
            for d in 0..<halfRotaryDims {
                // Frequencies computed over the rotary subspace (rDims), not full headDim.
                let freqIdx = Float(2 * d) / Float(rDims)
                let theta = pow(config.ropeTheta, -freqIdx)
                let angle = Float(position) * theta

                let cosA = cos(angle)
                let sinA = sin(angle)

                let firstIndex = offset + d
                let secondIndex = offset + d + halfRotaryDims
                let x0 = vec[firstIndex]
                let x1 = vec[secondIndex]

                result[firstIndex] = x0 * cosA - x1 * sinA
                result[secondIndex] = x0 * sinA + x1 * cosA
            }
            // Dims rDims..<headDim remain unchanged in `result` (already copied from `vec`).
        }
        return result
    }
}

private extension ModelRunner {
    func attention(hiddenRow: Tensor<Float>,
                   layerWeights: AttentionProjectionWeights,
                   layerIndex: Int) -> Tensor<Float> {
        var query = layerWeights.query.apply(toRow: hiddenRow)
        let keyRow = layerWeights.key.apply(toRow: hiddenRow)
        let valueRow = layerWeights.value.apply(toRow: hiddenRow)

        // Apply RoPE to query and key (not value). Phi-2 uses partial RoPE.
        let headDim = config.headDim
        let rotaryDims = config.rotaryDims  // = headDim for full RoPE (LLaMA/Gemma)
        let queryData = applyRoPE(query.squeezedRowVector().data, headDim: headDim,
                                   numHeads: config.numHeads, position: currentPosition,
                                   rotaryDims: rotaryDims)
        query = Tensor<Float>(shape: query.shape, data: queryData)

        let keyData = applyRoPE(keyRow.squeezedRowVector().data, headDim: headDim,
                                 numHeads: config.numKVHeads, position: currentPosition,
                                 rotaryDims: rotaryDims)
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
