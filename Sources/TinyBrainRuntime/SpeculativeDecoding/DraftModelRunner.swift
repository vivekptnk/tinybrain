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
            let probabilityDistribution = Sampler.samplingDistribution(
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
}
