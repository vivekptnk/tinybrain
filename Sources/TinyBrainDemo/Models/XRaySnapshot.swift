/// X-Ray Snapshot — Single-token observation from the transformer
///
/// **TB-010: X-Ray Mode**
///
/// Captures a snapshot of transformer internals at one inference step.
/// Each snapshot contains attention patterns, layer activation magnitudes,
/// and the top token candidates from the output distribution.

import Foundation

/// A single observation from one inference step
public struct XRaySnapshot: Identifiable {
    public let id = UUID()

    /// Token position in the sequence (0-based)
    public let position: Int

    /// Timestamp when this snapshot was captured
    public let timestamp: Date

    /// Attention weights per layer: `attentionWeights[layer][seqPosition]`
    ///
    /// Each inner array sums to ~1.0 (softmax output).
    /// Shows which past tokens the model attends to at each layer.
    public let attentionWeights: [[Float]]

    /// L2 norm of hidden state entering each layer: `layerNorms[layer]`
    ///
    /// Shows how the signal magnitude evolves through the transformer.
    public let layerNorms: [Float]

    /// Top token candidates sorted by probability (descending)
    public let topCandidates: [TokenCandidate]

    /// Shannon entropy of the output distribution (higher = more uncertain)
    public let entropy: Float
}

/// A candidate token from the output distribution
public struct TokenCandidate: Identifiable {
    public let id: Int  // tokenId doubles as identity
    public let tokenId: Int
    public let probability: Float

    public init(tokenId: Int, probability: Float) {
        self.id = tokenId
        self.tokenId = tokenId
        self.probability = probability
    }
}
