/// Draft model runner for speculative decoding
///
/// **TB-Spec-003:** Thin wrapper around ModelRunner for draft token generation
///
/// ## Educational Overview
///
/// The draft model is a smaller, faster version of the target model.
/// It uses the same `ModelRunner` infrastructure — just with a different
/// config (fewer layers, smaller hidden dim). This avoids duplicating
/// any inference code.
///
/// **Key design decision:** The draft model is a regular `ModelRunner`,
/// not a special class. This means:
/// - All existing optimizations apply (KV cache, quantization)
/// - The same model loading pipeline works
/// - Independent KV cache (smaller, matching draft dimensions)
///
/// **Example:**
/// ```swift
/// let draft = DraftModelRunner(config: smallConfig)
/// let tokens = draft.draftTokens(prompt: [1, 2, 3], count: 4)
/// // tokens = [(tokenId: 42, logProb: -0.5, probabilityDistribution: [...]), ...]
/// ```

import Foundation

// MARK: - Draft Model Runner

/// Wrapper around ModelRunner specialized for draft token generation
///
/// Creates a second ModelRunner with the draft model's config/weights.
/// Shares tokenizer with the target model but maintains an independent
/// KV cache matching the draft model's dimensions.
public final class DraftModelRunner {
    /// The underlying model runner for the draft model
    public let runner: ModelRunner

    /// Initialize with a draft model configuration (toy weights for testing)
    ///
    /// - Parameter config: Draft model configuration (smaller than target)
    public init(config: ModelConfig) {
        self.runner = ModelRunner(config: config)
    }

    /// Initialize with explicit draft model weights
    ///
    /// - Parameter weights: Loaded draft model weights
    public init(weights: ModelWeights) {
        self.runner = ModelRunner(weights: weights)
    }

    /// Generate K draft tokens with log-probabilities and full distributions
    ///
    /// Runs the draft model autoregressively for `count` tokens,
    /// collecting the selected token ID, its log-probability, and the full
    /// draft probability distribution at each step.
    ///
    /// - Parameters:
    ///   - prompt: Input token IDs to condition on
    ///   - count: Number of draft tokens to generate (K)
    ///   - samplerConfig: Sampling configuration (temperature, top-k, etc.)
    /// - Returns: Array of draft tokens with selected IDs and distributions
    public func draftTokens(
        prompt: [Int],
        count: Int,
        samplerConfig: SamplerConfig = SamplerConfig(temperature: 1.0)
    ) -> [DraftToken] {
        let vocabSize = runner.config.vocabSize

        // Process prompt (all except last token)
        let sanitized = prompt.map { max(0, min($0, vocabSize - 1)) }
        var currentToken = sanitized.last ?? 0
        if !sanitized.isEmpty {
            for token in sanitized.dropLast() {
                _ = runner.step(tokenId: token)
            }
        }

        var mutableConfig = samplerConfig
        var history: [Int] = Array(sanitized)
        var results: [DraftToken] = []

        for _ in 0..<count {
            let logits = runner.step(tokenId: currentToken)
            let probabilityDistribution = Self.samplingDistribution(
                logits: logits,
                config: mutableConfig,
                history: history
            )

            // Sample with detailed metadata to get probability
            let detailed = Sampler.sampleDetailed(
                logits: logits,
                config: &mutableConfig,
                history: history
            )

            // Convert probability to log-probability
            let selectedProbability: Float
            if detailed.tokenId >= 0 && detailed.tokenId < probabilityDistribution.count {
                selectedProbability = probabilityDistribution[detailed.tokenId]
            } else {
                selectedProbability = detailed.probability
            }
            let logProb = selectedProbability > 0 ? log(selectedProbability) : -Float.infinity

            results.append(DraftToken(
                tokenId: detailed.tokenId,
                logProb: logProb,
                probabilityDistribution: probabilityDistribution
            ))

            currentToken = detailed.tokenId
            history.append(detailed.tokenId)
        }

        return results
    }

    /// Reset the draft model's KV cache and position
    public func reset() {
        runner.reset()
    }

    /// Current position in the draft model's sequence
    public var currentPosition: Int {
        runner.currentPosition
    }

    /// Computes the final draft sampling distribution without consuming RNG.
    ///
    /// This mirrors `Sampler.sampleDetailed` through repetition penalty,
    /// top-k/top-p filtering, and temperature scaling so verification receives
    /// the same distribution the draft token was sampled from.
    private static func samplingDistribution(
        logits: Tensor<Float>,
        config: SamplerConfig,
        history: [Int]
    ) -> [Float] {
        var adjustedData = logits.data
        if config.repetitionPenalty != 1.0 && !history.isEmpty {
            let penalty = config.repetitionPenalty
            for tokenId in history where tokenId >= 0 && tokenId < adjustedData.count {
                if adjustedData[tokenId] > 0 {
                    adjustedData[tokenId] /= penalty
                } else {
                    adjustedData[tokenId] *= penalty
                }
            }
        }

        var workingLogits = Tensor<Float>(shape: logits.shape, data: adjustedData)

        if let k = config.topK {
            let sorted = workingLogits.data.enumerated().sorted { $0.element > $1.element }
            let keep = Set(sorted.prefix(max(0, k)).map { $0.offset })
            var filtered = workingLogits.data
            for i in 0..<filtered.count where !keep.contains(i) {
                filtered[i] = -Float.infinity
            }
            workingLogits = Tensor<Float>(shape: workingLogits.shape, data: filtered)
        } else if let p = config.topP {
            let probs = workingLogits.softmax().data
            let sorted = probs.enumerated().sorted { $0.element > $1.element }
            var cumulative: Float = 0
            var cutoff = sorted.count
            for (i, (_, probability)) in sorted.enumerated() {
                cumulative += probability
                if cumulative >= p {
                    cutoff = i + 1
                    break
                }
            }

            let keep = Set(sorted.prefix(cutoff).map { $0.offset })
            var filtered = workingLogits.data
            for i in 0..<filtered.count where !keep.contains(i) {
                filtered[i] = -Float.infinity
            }
            workingLogits = Tensor<Float>(shape: workingLogits.shape, data: filtered)
        }

        let temperature = max(0, config.temperature)
        let scaledData: [Float]
        if temperature < 0.01 {
            scaledData = workingLogits.data
        } else {
            scaledData = workingLogits.data.map { $0 / temperature }
        }

        return Tensor<Float>(shape: workingLogits.shape, data: scaledData).softmax().data
    }
}
