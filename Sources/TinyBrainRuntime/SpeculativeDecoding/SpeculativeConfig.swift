/// Configuration for speculative decoding
///
/// **TB-Spec-001:** Speculative decoding settings for draft-then-verify generation
///
/// ## Educational Overview
///
/// Speculative decoding (Leviathan et al. 2023) uses a smaller "draft" model
/// to propose K candidate tokens, then a larger "target" model verifies them
/// in a single batched forward pass. Accepted tokens skip full inference.
///
/// **Key insight:** Verifying K tokens costs roughly the same as generating 1,
/// so accepted tokens are essentially free — yielding 1.5-2x throughput.
///
/// **Example:**
/// ```swift
/// let config = SpeculativeConfig(
///     speculationDepth: 4,
///     draftModelPath: "Models/draft-60M.tbf",
///     draftModelConfig: smallConfig
/// )
/// let decoder = SpeculativeDecoder(
///     targetRunner: mainRunner,
///     config: config
/// )
/// for try await token in decoder.generateStream(prompt: tokens, config: genConfig) {
///     print(token.tokenId)
/// }
/// ```

import Foundation

// MARK: - Speculative Config

/// Configuration for speculative decoding
///
/// Controls the draft-then-verify generation strategy.
public struct SpeculativeConfig: Equatable {
    /// Number of tokens the draft model proposes per round (K)
    ///
    /// Higher K = more speculative tokens per round, but higher
    /// rejection risk if the draft model diverges from the target.
    /// Typical values: 3-6. Default: 4.
    public let speculationDepth: Int

    /// Path to the draft model weights file
    ///
    /// The draft model should be a smaller, faster version of the
    /// target model (e.g., fewer layers, smaller hidden dim).
    public let draftModelPath: String

    /// Configuration for the draft model
    ///
    /// Must match the architecture of the draft model weights.
    /// Uses the same `ModelConfig` as the target — just with
    /// smaller dimensions.
    public let draftModelConfig: ModelConfig

    /// Minimum acceptance threshold for p_target / p_draft ratio
    ///
    /// Tokens with ratio below this threshold are always rejected,
    /// even if the random draw would have accepted them.
    /// 0.0 = standard algorithm (accept based purely on ratio).
    /// Higher values = more conservative, reject more aggressively.
    public let acceptanceThreshold: Float

    public init(
        speculationDepth: Int = 4,
        draftModelPath: String,
        draftModelConfig: ModelConfig,
        acceptanceThreshold: Float = 0.0
    ) {
        precondition(speculationDepth >= 1, "Speculation depth must be >= 1")
        precondition(acceptanceThreshold >= 0.0 && acceptanceThreshold <= 1.0,
                     "Acceptance threshold must be in [0, 1]")
        self.speculationDepth = speculationDepth
        self.draftModelPath = draftModelPath
        self.draftModelConfig = draftModelConfig
        self.acceptanceThreshold = acceptanceThreshold
    }
}

// MARK: - ModelConfig Equatable

extension ModelConfig: Equatable {
    public static func == (lhs: ModelConfig, rhs: ModelConfig) -> Bool {
        lhs.numLayers == rhs.numLayers &&
        lhs.hiddenDim == rhs.hiddenDim &&
        lhs.numHeads == rhs.numHeads &&
        lhs.numKVHeads == rhs.numKVHeads &&
        lhs.vocabSize == rhs.vocabSize &&
        lhs.maxSeqLen == rhs.maxSeqLen &&
        lhs.intermediateDim == rhs.intermediateDim
    }
}
