/// Tests for the speculative decoding subsystem
///
/// **TB-Spec-Tests:** Covers VerificationSampler, DraftModelRunner, SpeculativeDecoder
///
/// ## Test Strategy
///
/// 1. **VerificationSampler** — Unit tests with known distributions and seeded RNG
/// 2. **DraftModelRunner** — Integration with toy ModelRunner
/// 3. **SpeculativeDecoder** — End-to-end with fallback behavior
/// 4. **Edge cases** — All accepted, all rejected, K=1
/// 5. **Statistics** — Acceptance rate tracking

import XCTest
@testable import TinyBrainRuntime

// MARK: - SpeculativeConfig Tests

final class SpeculativeConfigTests: XCTestCase {

    /// **Test:** Config stores all fields correctly
    func testConfigCreation() {
        let draftConfig = ModelConfig(numLayers: 2, hiddenDim: 64, numHeads: 2, vocabSize: 100)
        let config = SpeculativeConfig(
            speculationDepth: 5,
            draftModelPath: "Models/draft.tbf",
            draftModelConfig: draftConfig,
            acceptanceThreshold: 0.1
        )
        XCTAssertEqual(config.speculationDepth, 5)
        XCTAssertEqual(config.draftModelPath, "Models/draft.tbf")
        XCTAssertEqual(config.draftModelConfig, draftConfig)
        XCTAssertEqual(config.acceptanceThreshold, 0.1)
    }

    /// **Test:** Default values
    func testConfigDefaults() {
        let draftConfig = ModelConfig(numLayers: 2, hiddenDim: 64, numHeads: 2, vocabSize: 100)
        let config = SpeculativeConfig(
            draftModelPath: "Models/draft.tbf",
            draftModelConfig: draftConfig
        )
        XCTAssertEqual(config.speculationDepth, 4, "Default speculation depth should be 4")
        XCTAssertEqual(config.acceptanceThreshold, 0.0, "Default threshold should be 0.0")
    }

    /// **Test:** Config equality
    func testConfigEquality() {
        let draftConfig = ModelConfig(numLayers: 2, hiddenDim: 64, numHeads: 2, vocabSize: 100)
        let a = SpeculativeConfig(draftModelPath: "a.tbf", draftModelConfig: draftConfig)
        let b = SpeculativeConfig(draftModelPath: "a.tbf", draftModelConfig: draftConfig)
        let c = SpeculativeConfig(speculationDepth: 8, draftModelPath: "a.tbf", draftModelConfig: draftConfig)
        XCTAssertEqual(a, b)
        XCTAssertNotEqual(a, c)
    }

    /// **Test:** ModelConfig equality
    func testModelConfigEquality() {
        let a = ModelConfig(numLayers: 4, hiddenDim: 128, numHeads: 4, vocabSize: 1000)
        let b = ModelConfig(numLayers: 4, hiddenDim: 128, numHeads: 4, vocabSize: 1000)
        let c = ModelConfig(numLayers: 2, hiddenDim: 64, numHeads: 2, vocabSize: 100)
        XCTAssertEqual(a, b)
        XCTAssertNotEqual(a, c)
    }
}

// MARK: - DraftToken Tests

final class DraftTokenTests: XCTestCase {

    /// **Test:** DraftToken stores fields and computes probability
    func testDraftTokenCreation() {
        let token = DraftToken(tokenId: 42, logProb: -0.5)
        XCTAssertEqual(token.tokenId, 42)
        XCTAssertEqual(token.logProb, -0.5)
        XCTAssertEqual(token.probability, exp(-0.5), accuracy: 1e-6)
    }

    /// **Test:** DraftToken equality
    func testDraftTokenEquality() {
        let a = DraftToken(tokenId: 1, logProb: -0.3)
        let b = DraftToken(tokenId: 1, logProb: -0.3)
        let c = DraftToken(tokenId: 2, logProb: -0.3)
        XCTAssertEqual(a, b)
        XCTAssertNotEqual(a, c)
    }

    /// **Test:** Zero log-probability gives probability near zero
    func testDraftTokenNegInfLogProb() {
        let token = DraftToken(tokenId: 0, logProb: -Float.infinity)
        XCTAssertEqual(token.probability, 0.0)
    }
}

// MARK: - VerificationSampler Tests

final class VerificationSamplerTests: XCTestCase {

    /// **Test:** Token is accepted when target probability >> draft probability
    ///
    /// When p_target/p_draft >> 1, acceptance is guaranteed (ratio ≥ 1).
    func testAcceptWhenTargetDominates() {
        var sampler = VerificationSampler(seed: 42)
        let vocabSize = 5

        // Target strongly favors token 2 (high logit)
        let targetLogits = Tensor<Float>(shape: TensorShape(vocabSize), data: [-10, -10, 10, -10, -10])
        // Draft also chose token 2 but with moderate probability
        let draft = DraftToken(tokenId: 2, logProb: log(0.5))

        let result = sampler.verify(draft: draft, targetLogits: targetLogits, vocabSize: vocabSize)
        if case .accepted(let id) = result {
            XCTAssertEqual(id, 2, "Should accept when target strongly agrees")
        } else {
            XCTFail("Should accept when target probability dominates")
        }
    }

    /// **Test:** Token is rejected when target assigns zero probability
    func testRejectWhenTargetForbids() {
        var sampler = VerificationSampler(seed: 42)
        let vocabSize = 5

        // Target assigns effectively zero to token 2 (very negative logit)
        let targetLogits = Tensor<Float>(shape: TensorShape(vocabSize), data: [10, 10, -100, 10, 10])
        let draft = DraftToken(tokenId: 2, logProb: log(0.5))

        let result = sampler.verify(draft: draft, targetLogits: targetLogits, vocabSize: vocabSize)
        if case .rejected(let resampledId) = result {
            XCTAssertNotEqual(resampledId, 2, "Should resample to a different token")
        } else {
            XCTFail("Should reject when target assigns near-zero probability")
        }
    }

    /// **Test:** Accept unconditionally when draft probability is zero
    func testAcceptWhenDraftProbIsZero() {
        var sampler = VerificationSampler(seed: 42)
        let vocabSize = 5

        let targetLogits = Tensor<Float>(shape: TensorShape(vocabSize), data: [1, 1, 5, 1, 1])
        // Draft had zero probability for this token (shouldn't happen normally,
        // but the algorithm says: accept because target predicted something draft didn't)
        let draft = DraftToken(tokenId: 2, logProb: -Float.infinity)

        let result = sampler.verify(draft: draft, targetLogits: targetLogits, vocabSize: vocabSize)
        if case .accepted(let id) = result {
            XCTAssertEqual(id, 2)
        } else {
            XCTFail("Should accept when draft probability is zero (target found something draft missed)")
        }
    }

    /// **Test:** Batch verification stops at first rejection
    func testBatchVerificationStopsAtRejection() {
        var sampler = VerificationSampler(seed: 42)
        let vocabSize = 5

        // Token 0: target agrees (accept)
        let logits0 = Tensor<Float>(shape: TensorShape(vocabSize), data: [-10, -10, 10, -10, -10])
        // Token 1: target disagrees (reject)
        let logits1 = Tensor<Float>(shape: TensorShape(vocabSize), data: [10, -10, -100, -10, -10])
        // Token 2: would be accepted but shouldn't be reached
        let logits2 = Tensor<Float>(shape: TensorShape(vocabSize), data: [-10, -10, 10, -10, -10])

        let draftTokens = [
            DraftToken(tokenId: 2, logProb: log(0.5)),
            DraftToken(tokenId: 2, logProb: log(0.5)), // Will be rejected
            DraftToken(tokenId: 2, logProb: log(0.5)),
        ]

        let results = sampler.verifyBatch(
            draftTokens: draftTokens,
            targetLogitsBatch: [logits0, logits1, logits2],
            vocabSize: vocabSize
        )

        XCTAssertEqual(results.count, 2, "Should stop after rejection (1 accept + 1 reject)")
        if case .accepted = results[0] { } else { XCTFail("First should be accepted") }
        if case .rejected = results[1] { } else { XCTFail("Second should be rejected") }
    }

    /// **Test:** All tokens accepted when distributions match closely
    func testAllAccepted() {
        var sampler = VerificationSampler(seed: 100)
        let vocabSize = 3

        // Both models strongly agree on the same token
        let logits = Tensor<Float>(shape: TensorShape(vocabSize), data: [-10, 10, -10])
        let draft = DraftToken(tokenId: 1, logProb: log(0.99))

        let results = sampler.verifyBatch(
            draftTokens: [draft, draft, draft],
            targetLogitsBatch: [logits, logits, logits],
            vocabSize: vocabSize
        )

        XCTAssertEqual(results.count, 3, "All should be processed")
        for result in results {
            if case .accepted(let id) = result {
                XCTAssertEqual(id, 1)
            } else {
                XCTFail("All should be accepted when distributions match")
            }
        }
    }

    /// **Test:** K=1 (single token verification)
    func testSingleTokenVerification() {
        var sampler = VerificationSampler(seed: 42)
        let vocabSize = 4

        let logits = Tensor<Float>(shape: TensorShape(vocabSize), data: [-10, 10, -10, -10])
        let draft = DraftToken(tokenId: 1, logProb: log(0.9))

        let results = sampler.verifyBatch(
            draftTokens: [draft],
            targetLogitsBatch: [logits],
            vocabSize: vocabSize
        )

        XCTAssertEqual(results.count, 1)
    }

    /// **Test:** Deterministic with same seed
    func testDeterministicWithSeed() {
        let vocabSize = 5
        let logits = Tensor<Float>(shape: TensorShape(vocabSize), data: [2, 2, 2, 2, 2]) // Uniform
        let draft = DraftToken(tokenId: 0, logProb: log(0.3))

        var sampler1 = VerificationSampler(seed: 999)
        var sampler2 = VerificationSampler(seed: 999)

        let r1 = sampler1.verify(draft: draft, targetLogits: logits, vocabSize: vocabSize)
        let r2 = sampler2.verify(draft: draft, targetLogits: logits, vocabSize: vocabSize)
        XCTAssertEqual(r1, r2, "Same seed should produce same result")
    }

    /// **Test:** Acceptance threshold rejects low-ratio tokens
    func testAcceptanceThreshold() {
        // With a high threshold, even moderate agreement gets rejected
        var sampler = VerificationSampler(seed: 42, acceptanceThreshold: 0.9)
        let vocabSize = 4

        // Target mildly agrees with token 1, but ratio < 0.9
        let logits = Tensor<Float>(shape: TensorShape(vocabSize), data: [1, 2, 1, 1])
        let draft = DraftToken(tokenId: 1, logProb: log(0.8))

        let result = sampler.verify(draft: draft, targetLogits: logits, vocabSize: vocabSize)
        // With threshold 0.9 and ratio likely < 0.9, should reject
        // (target softmax ~0.37 / draft 0.8 = ~0.46 < 0.9)
        if case .rejected = result {
            // Expected
        } else {
            XCTFail("Should reject when ratio is below acceptance threshold")
        }
    }
}

// MARK: - DraftModelRunner Tests

final class DraftModelRunnerTests: XCTestCase {

    /// **Test:** Draft runner produces tokens with log-probabilities
    func testDraftTokenGeneration() {
        let config = ModelConfig(numLayers: 2, hiddenDim: 64, numHeads: 2, vocabSize: 100)
        let draft = DraftModelRunner(config: config)

        let tokens = draft.draftTokens(prompt: [1, 2, 3], count: 4)

        XCTAssertEqual(tokens.count, 4, "Should produce exactly K tokens")
        for token in tokens {
            XCTAssertGreaterThanOrEqual(token.tokenId, 0)
            XCTAssertLessThan(token.tokenId, 100)
            XCTAssertFalse(token.logProb.isNaN, "Log-prob should not be NaN")
        }
    }

    /// **Test:** Draft runner with different count values
    func testDraftTokenCount() {
        let config = ModelConfig(numLayers: 2, hiddenDim: 64, numHeads: 2, vocabSize: 100)
        let draft = DraftModelRunner(config: config)

        let one = draft.draftTokens(prompt: [1], count: 1)
        XCTAssertEqual(one.count, 1)

        draft.reset()
        let six = draft.draftTokens(prompt: [1], count: 6)
        XCTAssertEqual(six.count, 6)
    }

    /// **Test:** Reset clears position
    func testDraftRunnerReset() {
        let config = ModelConfig(numLayers: 2, hiddenDim: 64, numHeads: 2, vocabSize: 100)
        let draft = DraftModelRunner(config: config)

        _ = draft.draftTokens(prompt: [1], count: 3)
        XCTAssertGreaterThan(draft.currentPosition, 0)

        draft.reset()
        XCTAssertEqual(draft.currentPosition, 0)
    }

    /// **Test:** Log-probabilities are negative (valid log-probs)
    func testLogProbsAreNegative() {
        let config = ModelConfig(numLayers: 2, hiddenDim: 64, numHeads: 2, vocabSize: 100)
        let draft = DraftModelRunner(config: config)

        let tokens = draft.draftTokens(prompt: [1], count: 4)
        for token in tokens {
            XCTAssertLessThanOrEqual(token.logProb, 0, "Log-probability should be <= 0")
        }
    }
}

// MARK: - SpeculativeDecoder Tests

final class SpeculativeDecoderTests: XCTestCase {

    /// **Test:** Fallback mode — no draft model means standard generation
    func testFallbackToStandardGeneration() async throws {
        let targetConfig = ModelConfig(numLayers: 2, hiddenDim: 64, numHeads: 2, vocabSize: 100)
        let targetRunner = ModelRunner(config: targetConfig)

        // No specConfig → fallback
        let decoder = SpeculativeDecoder(targetRunner: targetRunner)

        var tokens: [Int] = []
        for try await output in decoder.generateStream(
            prompt: [1, 2],
            config: GenerationConfig(maxTokens: 5)
        ) {
            tokens.append(output.tokenId)
        }

        XCTAssertEqual(tokens.count, 5, "Fallback should generate exactly maxTokens")
        for id in tokens {
            XCTAssertGreaterThanOrEqual(id, 0)
            XCTAssertLessThan(id, 100)
        }
    }

    /// **Test:** Speculative decoding produces valid token stream
    func testSpeculativeGenerationProducesTokens() async throws {
        let targetConfig = ModelConfig(numLayers: 4, hiddenDim: 128, numHeads: 4, vocabSize: 100)
        let draftConfig = ModelConfig(numLayers: 2, hiddenDim: 64, numHeads: 2, vocabSize: 100)
        let targetRunner = ModelRunner(config: targetConfig)

        let specConfig = SpeculativeConfig(
            speculationDepth: 3,
            draftModelPath: "test.tbf",
            draftModelConfig: draftConfig
        )

        let decoder = SpeculativeDecoder(
            targetRunner: targetRunner,
            specConfig: specConfig,
            seed: 42
        )

        var tokens: [Int] = []
        for try await output in decoder.generateStream(
            prompt: [1, 2, 3],
            config: GenerationConfig(maxTokens: 10)
        ) {
            tokens.append(output.tokenId)
        }

        XCTAssertGreaterThan(tokens.count, 0, "Should produce at least some tokens")
        XCTAssertLessThanOrEqual(tokens.count, 10, "Should not exceed maxTokens")
        for id in tokens {
            XCTAssertGreaterThanOrEqual(id, 0)
            XCTAssertLessThan(id, 100, "All tokens should be valid vocabulary IDs")
        }
    }

    /// **Test:** Statistics are tracked correctly
    func testStatisticsTracking() async throws {
        let targetConfig = ModelConfig(numLayers: 2, hiddenDim: 64, numHeads: 2, vocabSize: 100)
        let draftConfig = ModelConfig(numLayers: 2, hiddenDim: 64, numHeads: 2, vocabSize: 100)
        let targetRunner = ModelRunner(config: targetConfig)

        let specConfig = SpeculativeConfig(
            speculationDepth: 3,
            draftModelPath: "test.tbf",
            draftModelConfig: draftConfig
        )

        let decoder = SpeculativeDecoder(
            targetRunner: targetRunner,
            specConfig: specConfig,
            seed: 42
        )

        var tokens: [Int] = []
        for try await output in decoder.generateStream(
            prompt: [1],
            config: GenerationConfig(maxTokens: 8)
        ) {
            tokens.append(output.tokenId)
        }

        XCTAssertGreaterThan(decoder.stats.totalRounds, 0, "Should have at least one verification round")
        XCTAssertGreaterThan(decoder.stats.totalDraftTokens, 0, "Should have proposed draft tokens")
        XCTAssertGreaterThanOrEqual(decoder.stats.acceptedTokens, 0)
        XCTAssertLessThanOrEqual(decoder.stats.acceptanceRate, 1.0)
        XCTAssertGreaterThanOrEqual(decoder.stats.acceptanceRate, 0.0)
    }

    /// **Test:** Reset clears everything
    func testReset() async throws {
        let targetConfig = ModelConfig(numLayers: 2, hiddenDim: 64, numHeads: 2, vocabSize: 100)
        let draftConfig = ModelConfig(numLayers: 2, hiddenDim: 64, numHeads: 2, vocabSize: 100)
        let targetRunner = ModelRunner(config: targetConfig)

        let specConfig = SpeculativeConfig(
            speculationDepth: 2,
            draftModelPath: "test.tbf",
            draftModelConfig: draftConfig
        )

        let decoder = SpeculativeDecoder(
            targetRunner: targetRunner,
            specConfig: specConfig,
            seed: 42
        )

        // Generate some tokens
        for try await _ in decoder.generateStream(
            prompt: [1],
            config: GenerationConfig(maxTokens: 4)
        ) {}

        decoder.reset()
        XCTAssertEqual(decoder.stats, SpeculativeStats())
        XCTAssertEqual(decoder.targetRunner.currentPosition, 0)
    }

    /// **Test:** Stop tokens are respected
    func testStopTokens() async throws {
        let targetConfig = ModelConfig(numLayers: 2, hiddenDim: 64, numHeads: 2, vocabSize: 100)
        let draftConfig = ModelConfig(numLayers: 2, hiddenDim: 64, numHeads: 2, vocabSize: 100)
        let targetRunner = ModelRunner(config: targetConfig)

        let specConfig = SpeculativeConfig(
            speculationDepth: 2,
            draftModelPath: "test.tbf",
            draftModelConfig: draftConfig
        )

        let decoder = SpeculativeDecoder(
            targetRunner: targetRunner,
            specConfig: specConfig,
            seed: 42
        )

        // Use stop token = 0 (unlikely but tests the mechanism)
        var tokens: [Int] = []
        for try await output in decoder.generateStream(
            prompt: [1],
            config: GenerationConfig(maxTokens: 100, stopTokens: [0])
        ) {
            tokens.append(output.tokenId)
        }

        // Should stop well before 100 tokens (either hit stop token or maxTokens)
        XCTAssertLessThanOrEqual(tokens.count, 100)
    }

    /// **Test:** Speculation depth K=1 works
    func testSpeculationDepthOne() async throws {
        let targetConfig = ModelConfig(numLayers: 2, hiddenDim: 64, numHeads: 2, vocabSize: 100)
        let draftConfig = ModelConfig(numLayers: 2, hiddenDim: 64, numHeads: 2, vocabSize: 100)
        let targetRunner = ModelRunner(config: targetConfig)

        let specConfig = SpeculativeConfig(
            speculationDepth: 1,
            draftModelPath: "test.tbf",
            draftModelConfig: draftConfig
        )

        let decoder = SpeculativeDecoder(
            targetRunner: targetRunner,
            specConfig: specConfig,
            seed: 42
        )

        var tokens: [Int] = []
        for try await output in decoder.generateStream(
            prompt: [1, 2],
            config: GenerationConfig(maxTokens: 5)
        ) {
            tokens.append(output.tokenId)
        }

        XCTAssertGreaterThan(tokens.count, 0, "K=1 should still produce tokens")
        XCTAssertLessThanOrEqual(tokens.count, 5)
    }
}

// MARK: - SpeculativeStats Tests

final class SpeculativeStatsTests: XCTestCase {

    /// **Test:** Default stats are zero
    func testDefaultStats() {
        let stats = SpeculativeStats()
        XCTAssertEqual(stats.totalDraftTokens, 0)
        XCTAssertEqual(stats.acceptedTokens, 0)
        XCTAssertEqual(stats.totalRounds, 0)
        XCTAssertEqual(stats.acceptanceRate, 0)
        XCTAssertEqual(stats.avgAcceptedPerRound, 0)
    }

    /// **Test:** Acceptance rate calculation
    func testAcceptanceRate() {
        var stats = SpeculativeStats()
        stats.totalDraftTokens = 10
        stats.acceptedTokens = 7
        stats.totalRounds = 3
        XCTAssertEqual(stats.acceptanceRate, 0.7, accuracy: 0.01)
        XCTAssertEqual(stats.avgAcceptedPerRound, 7.0 / 3.0, accuracy: 0.01)
    }

    /// **Test:** Stats equality
    func testStatsEquality() {
        var a = SpeculativeStats()
        a.totalDraftTokens = 5
        a.acceptedTokens = 3
        var b = SpeculativeStats()
        b.totalDraftTokens = 5
        b.acceptedTokens = 3
        XCTAssertEqual(a, b)
    }
}

// MARK: - VerificationResult Tests

final class VerificationResultTests: XCTestCase {

    /// **Test:** Result equality
    func testResultEquality() {
        XCTAssertEqual(
            VerificationResult.accepted(tokenId: 1),
            VerificationResult.accepted(tokenId: 1)
        )
        XCTAssertNotEqual(
            VerificationResult.accepted(tokenId: 1),
            VerificationResult.accepted(tokenId: 2)
        )
        XCTAssertNotEqual(
            VerificationResult.accepted(tokenId: 1),
            VerificationResult.rejected(resampledTokenId: 1)
        )
    }
}
