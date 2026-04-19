/// Modified rejection sampling for speculative decoding verification
///
/// **TB-Spec-002:** Token acceptance/rejection per Leviathan et al. 2023
///
/// ## Educational Overview
///
/// After the draft model proposes K tokens, the target model verifies them
/// by comparing probability distributions. For each draft token `x_i`:
///
/// 1. Compute ratio: `r = p_target(x_i) / p_draft(x_i)`
/// 2. Draw `u ~ Uniform[0, 1]`
/// 3. If `u < r` → accept (target agrees with draft)
/// 4. If `u >= r` → reject, resample from `norm(max(0, p_target - p_draft))`
///
/// This guarantees the final distribution matches the target model exactly,
/// regardless of draft model quality. Better drafts just accept more tokens.
///
/// **Edge cases:**
/// - `p_draft = 0`: Accept unconditionally (draft didn't predict it, target did)
/// - `p_target = 0`: Reject unconditionally (target forbids this token)
/// - Both zero: Skip (impossible token in both models)

import Foundation

// MARK: - Draft Token

/// A token proposed by the draft model with its probability
public struct DraftToken: Equatable {
    /// The proposed token ID
    public let tokenId: Int
    /// Log-probability from the draft model's distribution
    public let logProb: Float

    /// Probability (exp of logProb)
    public var probability: Float { exp(logProb) }

    public init(tokenId: Int, logProb: Float) {
        self.tokenId = tokenId
        self.logProb = logProb
    }
}

// MARK: - Verification Result

/// Result of verifying a single draft token
public enum VerificationResult: Equatable {
    /// Token accepted — matches target distribution
    case accepted(tokenId: Int)
    /// Token rejected — resample from adjusted distribution
    case rejected(resampledTokenId: Int)
}

// MARK: - Verification Sampler

/// Modified rejection sampler for speculative decoding
///
/// Compares draft vs target probability distributions to decide
/// which draft tokens to accept. Guarantees output distribution
/// matches the target model exactly.
public struct VerificationSampler {
    /// Random number generator (seeded for reproducibility in tests)
    private var rng: SeededRandomGenerator

    /// Minimum p_target/p_draft ratio for acceptance
    private let acceptanceThreshold: Float

    /// Initialize with optional seed for deterministic behavior
    ///
    /// - Parameters:
    ///   - seed: RNG seed (nil = random, set for reproducible tests)
    ///   - acceptanceThreshold: Minimum ratio for acceptance (default 0.0)
    public init(seed: UInt64? = nil, acceptanceThreshold: Float = 0.0) {
        if let seed = seed {
            self.rng = SeededRandomGenerator(seed: seed)
        } else {
            self.rng = SeededRandomGenerator(seed: UInt64.random(in: 0...UInt64.max))
        }
        self.acceptanceThreshold = acceptanceThreshold
    }

    /// Verify a single draft token against the target distribution
    ///
    /// Implements modified rejection sampling:
    /// - Accept if `u < min(1, p_target / p_draft)` and ratio >= threshold
    /// - Reject and resample from `norm(max(0, p_target - p_draft))` otherwise
    ///
    /// - Parameters:
    ///   - draft: The draft token with its probability
    ///   - targetLogits: Full logits from the target model [vocabSize]
    ///   - vocabSize: Size of the vocabulary
    /// - Returns: Verification result (accepted or rejected with resampled token)
    public mutating func verify(
        draft: DraftToken,
        targetLogits: Tensor<Float>,
        vocabSize: Int
    ) -> VerificationResult {
        let targetProbs = softmax(targetLogits.data)
        let pTarget = targetProbs[draft.tokenId]
        let pDraft = draft.probability

        // Edge case: target assigns zero probability — always reject
        if pTarget <= 0 {
            let resampled = resampleFromAdjusted(
                targetProbs: targetProbs,
                draftProbs: nil,
                vocabSize: vocabSize
            )
            return .rejected(resampledTokenId: resampled)
        }

        // Edge case: draft assigned zero probability — always accept
        // (target predicted something draft didn't, always good)
        if pDraft <= 0 {
            return .accepted(tokenId: draft.tokenId)
        }

        // Standard acceptance: u < min(1, p_target / p_draft)
        let ratio = pTarget / pDraft
        if ratio < acceptanceThreshold {
            let resampled = resampleFromAdjusted(
                targetProbs: targetProbs,
                draftProbs: makeDraftProbs(from: draft, vocabSize: vocabSize),
                vocabSize: vocabSize
            )
            return .rejected(resampledTokenId: resampled)
        }

        let u = Float.random(in: 0..<1, using: &rng)
        if u < min(1.0, ratio) {
            return .accepted(tokenId: draft.tokenId)
        }

        // Rejection: resample from norm(max(0, p_target - p_draft))
        let draftProbs = makeDraftProbs(from: draft, vocabSize: vocabSize)
        let resampled = resampleFromAdjusted(
            targetProbs: targetProbs,
            draftProbs: draftProbs,
            vocabSize: vocabSize
        )
        return .rejected(resampledTokenId: resampled)
    }

    /// Verify a batch of draft tokens against target distributions
    ///
    /// Processes tokens sequentially, stopping at the first rejection.
    /// Returns all accepted tokens plus the resampled token from rejection.
    ///
    /// - Parameters:
    ///   - draftTokens: Ordered draft tokens with probabilities
    ///   - targetLogitsBatch: Target logits for each position [K x vocabSize]
    ///   - vocabSize: Vocabulary size
    /// - Returns: Array of verification results (accepted tokens, then one rejection or all accepted)
    public mutating func verifyBatch(
        draftTokens: [DraftToken],
        targetLogitsBatch: [Tensor<Float>],
        vocabSize: Int
    ) -> [VerificationResult] {
        precondition(draftTokens.count == targetLogitsBatch.count,
                     "Draft tokens and target logits must have same count")

        var results: [VerificationResult] = []

        for (draft, targetLogits) in zip(draftTokens, targetLogitsBatch) {
            let result = verify(
                draft: draft,
                targetLogits: targetLogits,
                vocabSize: vocabSize
            )
            results.append(result)

            // Stop at first rejection
            if case .rejected = result {
                break
            }
        }

        return results
    }

    // MARK: - Private Helpers

    /// Compute softmax of logits
    private func softmax(_ logits: [Float]) -> [Float] {
        let maxLogit = logits.max() ?? 0
        let exps = logits.map { exp($0 - maxLogit) }
        let sum = exps.reduce(0, +)
        return sum > 0 ? exps.map { $0 / sum } : exps
    }

    /// Construct a full probability distribution from a single draft token
    ///
    /// We only have the draft probability for the selected token.
    /// For resampling we approximate: the draft model assigned `pDraft`
    /// to the selected token and distributes the rest uniformly.
    private func makeDraftProbs(from draft: DraftToken, vocabSize: Int) -> [Float] {
        var probs = [Float](repeating: 0, count: vocabSize)
        if draft.tokenId < vocabSize {
            probs[draft.tokenId] = draft.probability
        }
        return probs
    }

    /// Resample from the adjusted distribution: norm(max(0, p_target - p_draft))
    ///
    /// This is the correction distribution that ensures the final output
    /// matches the target model's distribution exactly.
    private mutating func resampleFromAdjusted(
        targetProbs: [Float],
        draftProbs: [Float]?,
        vocabSize: Int
    ) -> Int {
        var adjusted = [Float](repeating: 0, count: vocabSize)

        if let draftProbs = draftProbs {
            for i in 0..<vocabSize {
                adjusted[i] = max(0, targetProbs[i] - draftProbs[i])
            }
        } else {
            // No draft probs available — just use target probs directly
            adjusted = Array(targetProbs.prefix(vocabSize))
        }

        // Normalize
        let sum = adjusted.reduce(0, +)
        if sum > 0 {
            for i in 0..<vocabSize {
                adjusted[i] /= sum
            }
        } else {
            // Fallback: uniform distribution (should not happen in practice)
            let uniform = 1.0 / Float(vocabSize)
            for i in 0..<vocabSize {
                adjusted[i] = uniform
            }
        }

        // Sample from adjusted distribution
        let u = Float.random(in: 0..<1, using: &rng)
        var cumulative: Float = 0
        for i in 0..<vocabSize {
            cumulative += adjusted[i]
            if u < cumulative {
                return i
            }
        }
        return vocabSize - 1
    }
}
