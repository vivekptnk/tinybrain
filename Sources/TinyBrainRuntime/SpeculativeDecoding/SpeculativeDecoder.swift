/// Speculative decoder — draft-then-verify generation orchestrator
///
/// **TB-Spec-004:** Main orchestrator for speculative decoding
///
/// ## Educational Overview
///
/// Speculative decoding works in rounds:
///
/// ```
/// Round 1:
///   Draft model:  [prompt] → draft t1, t2, t3, t4  (fast, small model)
///   Target model: verify t1, t2, t3, t4 in one batch (expensive, but batched)
///   Accept/reject: t1 ✓, t2 ✓, t3 ✗ → yield t1, t2, resample t3'
///
/// Round 2:
///   Draft model:  [prompt + t1, t2, t3'] → draft t4, t5, t6, t7
///   Target model: verify t4, t5, t6, t7
///   ...
/// ```
///
/// **Why it's fast:** Verifying K tokens in one batched forward pass costs
/// roughly the same as generating 1 token. So accepted tokens are free.
///
/// **Correctness guarantee:** The rejection sampling math ensures the output
/// distribution matches the target model exactly, regardless of draft quality.
///
/// **Example:**
/// ```swift
/// let decoder = SpeculativeDecoder(
///     targetRunner: targetRunner,
///     specConfig: specConfig
/// )
/// for try await token in decoder.generateStream(prompt: tokenIds, config: genConfig) {
///     print(token.tokenId)
/// }
/// ```

import Foundation

// MARK: - Speculative Decoder

/// Orchestrates draft-then-verify speculative generation
///
/// Wraps a target ModelRunner with a draft model to accelerate
/// autoregressive generation through speculative execution.
///
/// **Design:** Composition over inheritance. SpeculativeDecoder owns
/// the target and draft runners, coordinating their KV caches.
public final class SpeculativeDecoder {
    /// The main (target) model runner
    public let targetRunner: ModelRunner

    /// Draft model runner (smaller, faster)
    public let draftRunner: DraftModelRunner?

    /// Speculative decoding configuration
    public let specConfig: SpeculativeConfig?

    /// Verification sampler for acceptance/rejection
    private var verificationSampler: VerificationSampler

    /// Statistics for monitoring acceptance rates
    public private(set) var stats: SpeculativeStats = SpeculativeStats()

    /// Initialize with a target runner and optional speculative config
    ///
    /// If `specConfig` is nil, falls back to standard generation
    /// (equivalent to calling `targetRunner.generateStream()` directly).
    ///
    /// - Parameters:
    ///   - targetRunner: The main model runner (large model)
    ///   - specConfig: Speculative decoding config (nil = no speculation)
    ///   - seed: RNG seed for reproducible verification (nil = random)
    public init(
        targetRunner: ModelRunner,
        specConfig: SpeculativeConfig? = nil,
        seed: UInt64? = nil
    ) {
        self.targetRunner = targetRunner
        self.specConfig = specConfig

        if let config = specConfig {
            self.draftRunner = DraftModelRunner(config: config.draftModelConfig)
            self.verificationSampler = VerificationSampler(
                seed: seed,
                acceptanceThreshold: config.acceptanceThreshold
            )
        } else {
            self.draftRunner = nil
            self.verificationSampler = VerificationSampler(seed: seed)
        }
    }

    /// Initialize with explicit draft model weights
    ///
    /// - Parameters:
    ///   - targetRunner: The main model runner
    ///   - draftWeights: Loaded weights for the draft model
    ///   - specConfig: Speculative config
    ///   - seed: RNG seed for verification
    public init(
        targetRunner: ModelRunner,
        draftWeights: ModelWeights,
        specConfig: SpeculativeConfig,
        seed: UInt64? = nil
    ) {
        self.targetRunner = targetRunner
        self.specConfig = specConfig
        self.draftRunner = DraftModelRunner(weights: draftWeights)
        self.verificationSampler = VerificationSampler(
            seed: seed,
            acceptanceThreshold: specConfig.acceptanceThreshold
        )
    }

    /// Generate a stream of tokens using speculative decoding
    ///
    /// If no draft model is configured, delegates directly to the
    /// target model's `generateStream()` — zero overhead fallback.
    ///
    /// - Parameters:
    ///   - prompt: Input token IDs
    ///   - config: Generation configuration
    /// - Returns: Async stream of token outputs
    public func generateStream(
        prompt: [Int],
        config: GenerationConfig = GenerationConfig()
    ) -> AsyncThrowingStream<TokenOutput, Error> {
        // Fallback: no draft model → standard generation
        guard let draftRunner = draftRunner, specConfig != nil else {
            return targetRunner.generateStream(prompt: prompt, config: config)
        }

        let specDepth = specConfig!.speculationDepth
        let vocabSize = targetRunner.config.vocabSize

        return AsyncThrowingStream { continuation in
            Task {
                var mutableConfig = config
                let sanitizedPrompt = prompt.map { max(0, min($0, vocabSize - 1)) }

                // Reset both models
                self.targetRunner.reset()
                draftRunner.reset()
                self.stats = SpeculativeStats()

                // Process prompt through target model
                for token in sanitizedPrompt.dropLast() {
                    _ = self.targetRunner.step(tokenId: token)
                }

                var currentToken = sanitizedPrompt.last ?? 0
                var history: [Int] = Array(sanitizedPrompt)
                var generated = 0

                while generated < mutableConfig.maxTokens {
                    // === DRAFT PHASE ===
                    // Run draft model for K tokens from current position
                    draftRunner.reset()
                    // Feed prompt + accepted history to draft model
                    let draftPrompt = history
                    let draftTokens = draftRunner.draftTokens(
                        prompt: draftPrompt,
                        count: min(specDepth, mutableConfig.maxTokens - generated),
                        samplerConfig: mutableConfig.sampler
                    )

                    if draftTokens.isEmpty {
                        // No draft tokens — fall back to single-token generation
                        let logits = self.targetRunner.step(tokenId: currentToken)
                        let detailed = Sampler.sampleDetailed(
                            logits: logits,
                            config: &mutableConfig.sampler,
                            history: history
                        )

                        let output = TokenOutput(
                            tokenId: detailed.tokenId,
                            probability: detailed.probability,
                            entropy: detailed.entropy,
                            timestamp: Date()
                        )
                        continuation.yield(output)

                        currentToken = detailed.tokenId
                        history.append(detailed.tokenId)
                        generated += 1

                        if mutableConfig.stopTokens.contains(detailed.tokenId) {
                            break
                        }
                        continue
                    }

                    // === VERIFY PHASE ===
                    // Run target model on all draft tokens, collecting logits
                    var targetLogitsBatch: [Tensor<Float>] = []

                    // First: get logits for current token position
                    let firstLogits = self.targetRunner.step(tokenId: currentToken)
                    targetLogitsBatch.append(firstLogits)

                    // Then: step through remaining draft tokens
                    for i in 1..<draftTokens.count {
                        let logits = self.targetRunner.step(tokenId: draftTokens[i - 1].tokenId)
                        targetLogitsBatch.append(logits)
                    }

                    // === ACCEPT/REJECT ===
                    let results = self.verificationSampler.verifyBatch(
                        draftTokens: draftTokens,
                        targetLogitsBatch: targetLogitsBatch,
                        vocabSize: vocabSize
                    )

                    self.stats.totalDraftTokens += draftTokens.count
                    self.stats.totalRounds += 1

                    // Yield accepted tokens
                    var shouldStop = false
                    for result in results {
                        let tokenId: Int
                        switch result {
                        case .accepted(let id):
                            tokenId = id
                            self.stats.acceptedTokens += 1
                        case .rejected(let resampledId):
                            tokenId = resampledId
                        }

                        let output = TokenOutput(
                            tokenId: tokenId,
                            probability: 0.0, // Probability not tracked in speculative path
                            entropy: 0.0,
                            timestamp: Date()
                        )
                        continuation.yield(output)

                        currentToken = tokenId
                        history.append(tokenId)
                        generated += 1

                        if mutableConfig.stopTokens.contains(tokenId) {
                            shouldStop = true
                            break
                        }

                        if generated >= mutableConfig.maxTokens {
                            shouldStop = true
                            break
                        }
                    }

                    if shouldStop { break }

                    // After a rejection, we need to re-sync: the target model's
                    // KV cache is ahead (it processed all K draft tokens).
                    // Reset and re-feed the accepted history.
                    // This is the simple approach — a production version would
                    // truncate the KV cache instead of resetting.
                    if results.count < draftTokens.count || results.last.map({ if case .rejected = $0 { return true } else { return false } }) ?? false {
                        self.targetRunner.reset()
                        for token in history.dropLast() {
                            _ = self.targetRunner.step(tokenId: token)
                        }
                    }
                }

                continuation.finish()
            }
        }
    }

    /// Reset both target and draft models
    public func reset() {
        targetRunner.reset()
        draftRunner?.reset()
        stats = SpeculativeStats()
    }
}

// MARK: - Speculative Stats

/// Statistics for monitoring speculative decoding performance
public struct SpeculativeStats: Equatable {
    /// Total draft tokens proposed across all rounds
    public var totalDraftTokens: Int = 0

    /// Total draft tokens accepted
    public var acceptedTokens: Int = 0

    /// Total verification rounds
    public var totalRounds: Int = 0

    /// Acceptance rate (0.0 - 1.0)
    public var acceptanceRate: Float {
        totalDraftTokens > 0 ? Float(acceptedTokens) / Float(totalDraftTokens) : 0
    }

    /// Average accepted tokens per round
    public var avgAcceptedPerRound: Float {
        totalRounds > 0 ? Float(acceptedTokens) / Float(totalRounds) : 0
    }
}
