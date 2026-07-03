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

    /// **Test:** Speculative sampling preserves the target distribution when it subtracts the full draft distribution.
    ///
    /// Math under test for one speculative position:
    /// - Target distribution p = [0.02, 0.08, 0.10, 0.12, 0.18, 0.20, 0.14, 0.16]
    /// - Draft distribution q = [0.42, 0.18, 0.14, 0.10, 0.06, 0.04, 0.03, 0.03]
    /// - Acceptance uses a(x) = min(1, p(x) / q(x)); expected acceptance mass is
    ///   sum_x min(p(x), q(x)) = 0.46, so expected rejection rate is 0.54.
    /// - On rejection, Leviathan et al. require sampling from normalize(max(0, p - q)).
    ///   That correction mass is concentrated on tokens 3...7.
    /// - The old point-mass approximation subtracted only q(x) at the sampled token x.
    ///   Its analytical output distribution is approximately
    ///   [0.0231, 0.1162, 0.1517, 0.1674, 0.1610, 0.1523, 0.1086, 0.1198],
    ///   with max absolute deviation about 0.0517 from p, so N=50k should fail
    ///   a 0.01 max-deviation bound by a wide margin.
    func testSpeculativeSamplingFullDraftDistributionMatchesTarget() {
        let targetProbs: [Float] = [0.02, 0.08, 0.10, 0.12, 0.18, 0.20, 0.14, 0.16]
        let draftProbs: [Float] = [0.42, 0.18, 0.14, 0.10, 0.06, 0.04, 0.03, 0.03]
        let vocabSize = targetProbs.count
        let targetLogits = Tensor<Float>(
            shape: TensorShape(vocabSize),
            data: targetProbs.map { log($0) }
        )

        var draftRNG = SeededRandomGenerator(seed: 0xC1C)
        var sampler = VerificationSampler(seed: 0xC1C_FEED)
        var histogram = [Int](repeating: 0, count: vocabSize)
        let rounds = 50_000

        for _ in 0..<rounds {
            let tokenId = sampleIndex(from: draftProbs, rng: &draftRNG)
            let draft = DraftToken(
                tokenId: tokenId,
                logProb: log(draftProbs[tokenId]),
                probabilityDistribution: draftProbs
            )

            let result = sampler.verify(
                draft: draft,
                targetLogits: targetLogits,
                vocabSize: vocabSize
            )

            switch result {
            case .accepted(let acceptedId):
                histogram[acceptedId] += 1
            case .rejected(let resampledId):
                histogram[resampledId] += 1
            }
        }

        let observed = histogram.map { Float($0) / Float(rounds) }
        let deviations = zip(observed, targetProbs).map { abs($0 - $1) }
        let maxDeviation = deviations.max() ?? 0

        XCTAssertLessThan(
            maxDeviation,
            0.01,
            "Observed \(observed) should stay within 0.01 of target \(targetProbs); deviations \(deviations)"
        )
    }

    /// **Test:** When target and draft distributions match, rejection correction has zero mass.
    ///
    /// The mathematically safe fallback is to sample from the target distribution, not
    /// from a uniform distribution that can introduce tokens the target assigns zero
    /// probability.
    func testAdjustedResampleFallsBackToTargetDistributionWhenCorrectionMassIsZero() {
        let vocabSize = 4
        let targetLogits = Tensor<Float>(
            shape: TensorShape(vocabSize),
            data: [-Float.infinity, -Float.infinity, 0.0, -Float.infinity]
        )
        let draft = DraftToken(
            tokenId: 2,
            logProb: 0.0,
            probabilityDistribution: [0.0, 0.0, 1.0, 0.0]
        )

        for seed in UInt64(0)..<UInt64(10) {
            var sampler = VerificationSampler(seed: seed, acceptanceThreshold: 1.1)
            let result = sampler.verify(draft: draft, targetLogits: targetLogits, vocabSize: vocabSize)

            if case .rejected(let resampledId) = result {
                XCTAssertEqual(resampledId, 2, "Fallback should resample from target distribution")
            } else {
                XCTFail("Acceptance threshold should force rejection before fallback sampling")
            }
        }
    }

    private func sampleIndex(from probabilities: [Float], rng: inout SeededRandomGenerator) -> Int {
        let threshold = Float(rng.next()) / Float(UInt64.max)
        var cumulative: Float = 0

        for (index, probability) in probabilities.enumerated() {
            cumulative += probability
            if threshold <= cumulative {
                return index
            }
        }

        return probabilities.count - 1
    }
}

// MARK: - DraftModelRunner Tests

final class DraftModelRunnerTests: XCTestCase {

    /// **Test:** The detailed sampler and reusable distribution helper stay exactly aligned.
    ///
    /// Speculative decoding stores the draft distribution separately from the sampled
    /// token. This regression proves both values come from the same post-processing
    /// pipeline and the same seeded RNG draw.
    func testSamplerSamplingDistributionMatchesDetailedSeededDraw() {
        let logits = Tensor<Float>(
            shape: TensorShape(7),
            data: [0.25, -0.5, 1.4, 0.8, -1.2, 0.05, 1.1]
        )
        let history = [2, 2, 4, -1, 99]

        let cases: [(name: String, config: SamplerConfig, seed: UInt64)] = [
            (
                name: "top-k",
                config: SamplerConfig(temperature: 0.7, topK: 4, repetitionPenalty: 1.25, seed: 0xC1C),
                seed: 0xC1C
            ),
            (
                name: "top-p",
                config: SamplerConfig(temperature: 0.9, topP: 0.72, repetitionPenalty: 1.25, seed: 0xC1D),
                seed: 0xC1D
            )
        ]

        for testCase in cases {
            let distribution = Sampler.samplingDistribution(
                logits: logits,
                config: testCase.config,
                history: history
            )

            var detailedConfig = testCase.config
            let detailed = Sampler.sampleDetailed(
                logits: logits,
                config: &detailedConfig,
                history: history
            )

            var rng = SeededRandomGenerator(seed: testCase.seed)
            let threshold = Float(rng.next()) / Float(UInt64.max)
            var cumulative: Float = 0
            var expectedToken = distribution.count - 1
            for (index, probability) in distribution.enumerated() {
                cumulative += probability
                if threshold <= cumulative {
                    expectedToken = index
                    break
                }
            }

            let expectedEntropy = distribution.reduce(Float(0)) { entropy, probability in
                probability > 0 ? entropy - probability * log(probability) : entropy
            }

            XCTAssertEqual(detailed.tokenId, expectedToken, "\(testCase.name) should use the helper distribution")
            XCTAssertEqual(detailed.probability.bitPattern, distribution[detailed.tokenId].bitPattern)
            XCTAssertEqual(detailed.entropy.bitPattern, expectedEntropy.bitPattern)
        }
    }

    /// **Test:** Draft runner produces tokens with log-probabilities
    func testDraftTokenGeneration() throws {
        let config = ModelConfig(numLayers: 2, hiddenDim: 64, numHeads: 2, vocabSize: 100)
        let draft = DraftModelRunner(config: config)

        let tokens = draft.draftTokens(prompt: [1, 2, 3], count: 4)

        XCTAssertEqual(tokens.count, 4, "Should produce exactly K tokens")
        for token in tokens {
            XCTAssertGreaterThanOrEqual(token.tokenId, 0)
            XCTAssertLessThan(token.tokenId, 100)
            XCTAssertFalse(token.logProb.isNaN, "Log-prob should not be NaN")
            let distribution = try XCTUnwrap(token.probabilityDistribution)
            XCTAssertEqual(distribution.count, 100)
            XCTAssertEqual(distribution.reduce(0, +), 1.0, accuracy: 1e-4)
            XCTAssertEqual(distribution[token.tokenId], token.probability, accuracy: 1e-5)
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

    /// **Test:** Cancelling the consumer terminates the speculative producer task.
    func testGenerationStreamCancellationStopsProducer() async throws {
        let targetConfig = ModelConfig(numLayers: 2, hiddenDim: 64, numHeads: 2, vocabSize: 64)
        let draftConfig = ModelConfig(numLayers: 1, hiddenDim: 32, numHeads: 2, vocabSize: 64)
        let targetRunner = ModelRunner(config: targetConfig)

        let specConfig = SpeculativeConfig(
            speculationDepth: 4,
            draftModelPath: "test.tbf",
            draftModelConfig: draftConfig
        )

        let decoder = SpeculativeDecoder(
            targetRunner: targetRunner,
            specConfig: specConfig,
            seed: 42
        )

        let firstToken = expectation(description: "first speculative token")
        let consumer = Task {
            var emitted = 0
            for try await _ in decoder.generateStream(
                prompt: [1, 2, 3],
                config: GenerationConfig(maxTokens: 5_000)
            ) {
                emitted += 1
                firstToken.fulfill()
                try await Task.sleep(nanoseconds: 5_000_000_000)
            }
            return emitted
        }

        await fulfillment(of: [firstToken], timeout: 2.0)
        consumer.cancel()

        let emitted: Int
        do {
            emitted = try await consumer.value
        } catch is CancellationError {
            emitted = 1
        }
        XCTAssertEqual(emitted, 1, "Consumer should stop after the first token")

        try await Task.sleep(nanoseconds: 250_000_000)
        let roundsAfterCancellation = decoder.stats.totalRounds

        try await Task.sleep(nanoseconds: 250_000_000)
        XCTAssertEqual(
            decoder.stats.totalRounds,
            roundsAfterCancellation,
            "Speculative loop should stop advancing after stream cancellation"
        )
        XCTAssertLessThan(
            roundsAfterCancellation,
            100,
            "Cancelled generation should not run most of the requested 5,000-token workload"
        )
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
