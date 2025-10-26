import XCTest
@testable import TinyBrainRuntime

/// Tests for sampling strategies used in LLM text generation
///
/// **TDD Approach:**
/// These tests are written FIRST to define expected behavior.
///
/// **Educational Note:**
/// Sampling is how we convert logits (raw model outputs) into actual tokens:
/// - **Greedy**: Always pick highest logit (deterministic, boring)
/// - **Temperature**: Scale logits to control randomness
/// - **Top-K**: Only sample from K highest logits
/// - **Top-P (Nucleus)**: Sample from smallest set with cumulative prob > P
/// - **Repetition Penalty**: Reduce probability of recently used tokens
final class SamplerTests: XCTestCase {
    
    // MARK: - Greedy Sampling Tests
    
    /// **Test:** Greedy sampling always selects argmax
    ///
    /// **Educational:**
    /// Greedy is deterministic - always picks the highest logit.
    /// Good for reproducibility, bad for creativity.
    func testGreedySelectsArgmax() {
        let logits = Tensor<Float>(shape: TensorShape(5), data: [0.1, 0.5, 0.2, 0.8, 0.3])
        let token = Sampler.greedy(logits: logits)
        
        XCTAssertEqual(token, 3, "Greedy should select index 3 (highest logit 0.8)")
    }
    
    /// **Test:** Greedy handles negative logits
    func testGreedyWithNegativeLogits() {
        let logits = Tensor<Float>(shape: TensorShape(4), data: [-2.0, -1.0, -3.0, -0.5])
        let token = Sampler.greedy(logits: logits)
        
        XCTAssertEqual(token, 3, "Should select index 3 (highest value -0.5)")
    }
    
    /// **Test:** Greedy handles tied values (picks first)
    func testGreedyWithTiedValues() {
        let logits = Tensor<Float>(shape: TensorShape(4), data: [0.5, 0.8, 0.8, 0.3])
        let token = Sampler.greedy(logits: logits)
        
        XCTAssertTrue([1, 2].contains(token), "Should pick one of the tied maximum values")
    }
    
    // MARK: - Temperature Sampling Tests
    
    /// **Test:** Temperature = 0 behaves like greedy
    ///
    /// **Educational:**
    /// As temperature → 0, softmax becomes more peaked, approaching greedy.
    func testTemperatureZeroIsGreedy() {
        let logits = Tensor<Float>(shape: TensorShape(5), data: [0.1, 0.5, 0.2, 0.8, 0.3])
        
        // With very low temperature, should consistently pick argmax
        var selectedTokens: Set<Int> = []
        for _ in 0..<10 {
            let token = Sampler.temperature(logits: logits, temp: 0.01)
            selectedTokens.insert(token)
        }
        
        XCTAssertEqual(selectedTokens.count, 1, "Low temperature should be deterministic")
        XCTAssertEqual(selectedTokens.first, 3, "Should pick argmax")
    }
    
    /// **Test:** Temperature = 1.0 is standard softmax
    func testTemperatureOne() {
        let logits = Tensor<Float>(shape: TensorShape(5), data: [0.1, 0.5, 0.2, 0.8, 0.3])
        
        var selectedTokens: Set<Int> = []
        for _ in 0..<100 {
            let token = Sampler.temperature(logits: logits, temp: 1.0)
            selectedTokens.insert(token)
        }
        
        // With temp=1, should sample from distribution (not always argmax)
        XCTAssertGreaterThan(selectedTokens.count, 1, "Should sample from distribution")
    }
    
    /// **Test:** High temperature increases randomness
    ///
    /// **Educational:**
    /// High temperature flattens the distribution → more random/creative.
    func testHighTemperatureIncreasesRandomness() {
        let logits = Tensor<Float>(shape: TensorShape(5), data: [0.1, 0.5, 0.2, 0.8, 0.3])
        
        var selectedTokens: Set<Int> = []
        for _ in 0..<200 {
            let token = Sampler.temperature(logits: logits, temp: 2.0)
            selectedTokens.insert(token)
        }
        
        // High temperature should sample more uniformly
        XCTAssertGreaterThanOrEqual(selectedTokens.count, 3, 
                                    "High temp should explore more tokens")
    }
    
    // MARK: - Top-K Sampling Tests
    
    /// **Test:** Top-K limits sampling to K highest logits
    ///
    /// **Educational:**
    /// Top-K zeroes out all but the K highest logits before sampling.
    /// Prevents sampling very unlikely tokens.
    func testTopKLimitsOptions() {
        let logits = Tensor<Float>(shape: TensorShape(5), data: [0.1, 0.5, 0.2, 0.8, 0.3])
        
        var seenTokens: Set<Int> = []
        for _ in 0..<100 {
            let token = Sampler.topK(logits: logits, k: 2, temp: 1.0)
            seenTokens.insert(token)
        }
        
        // Should only sample from top-2 (indices 1 and 3)
        XCTAssertLessThanOrEqual(seenTokens.count, 2, "Should only sample from top-K tokens")
        XCTAssertTrue(seenTokens.isSubset(of: [1, 3]), "Should be top-2 logits")
    }
    
    /// **Test:** Top-K with K=1 is greedy
    func testTopKOneIsGreedy() {
        let logits = Tensor<Float>(shape: TensorShape(5), data: [0.1, 0.5, 0.2, 0.8, 0.3])
        
        let token = Sampler.topK(logits: logits, k: 1, temp: 1.0)
        
        XCTAssertEqual(token, 3, "Top-1 should be greedy")
    }
    
    /// **Test:** Top-K with K >= vocab_size samples from all
    func testTopKLargeK() {
        let logits = Tensor<Float>(shape: TensorShape(5), data: [0.1, 0.5, 0.2, 0.8, 0.3])
        
        var seenTokens: Set<Int> = []
        for _ in 0..<200 {
            let token = Sampler.topK(logits: logits, k: 10, temp: 1.5)
            seenTokens.insert(token)
        }
        
        // With K > vocab_size, should behave like standard temperature sampling
        XCTAssertGreaterThan(seenTokens.count, 2, "Should sample broadly")
    }
    
    // MARK: - Top-P (Nucleus) Sampling Tests
    
    /// **Test:** Top-P samples from smallest set with cumulative prob > P
    ///
    /// **Educational:**
    /// Nucleus sampling is adaptive - selects fewest tokens whose
    /// cumulative probability exceeds threshold P.
    func testTopPNucleusSampling() {
        // Create logits where we know the softmax distribution
        let logits = Tensor<Float>(shape: TensorShape(5), data: [0.0, 1.0, 0.0, 2.0, 0.0])
        
        var seenTokens: Set<Int> = []
        for _ in 0..<100 {
            // With P=0.9, should sample from high-probability tokens
            let token = Sampler.topP(logits: logits, p: 0.9, temp: 1.0)
            seenTokens.insert(token)
        }
        
        // Should focus on high-probability tokens (allowing some flexibility)
        XCTAssertLessThanOrEqual(seenTokens.count, 4, 
                                "Top-P should limit to nucleus (allowing ~90% mass)")
        // Should definitely sample the highest prob token
        XCTAssertTrue(seenTokens.contains(3), "Should sample highest probability token")
    }
    
    /// **Test:** Top-P with P=1.0 samples from all tokens
    func testTopPOne() {
        let logits = Tensor<Float>(shape: TensorShape(5), data: [0.1, 0.5, 0.2, 0.8, 0.3])
        
        var seenTokens: Set<Int> = []
        for _ in 0..<200 {
            let token = Sampler.topP(logits: logits, p: 1.0, temp: 1.5)
            seenTokens.insert(token)
        }
        
        // P=1.0 includes all tokens
        XCTAssertGreaterThan(seenTokens.count, 3, "Should sample from all tokens")
    }
    
    /// **Test:** Top-P with very low P approaches greedy
    func testTopPLow() {
        let logits = Tensor<Float>(shape: TensorShape(5), data: [0.1, 0.5, 0.2, 0.8, 0.3])
        
        var seenTokens: Set<Int> = []
        for _ in 0..<50 {
            let token = Sampler.topP(logits: logits, p: 0.1, temp: 1.0)
            seenTokens.insert(token)
        }
        
        // Very low P should be very focused
        XCTAssertLessThanOrEqual(seenTokens.count, 2, "Low P should be focused")
    }
    
    // MARK: - Repetition Penalty Tests
    
    /// **Test:** Repetition penalty reduces probability of repeated tokens
    ///
    /// **Educational:**
    /// Repetition penalty divides logits of previously seen tokens,
    /// discouraging the model from repeating itself.
    func testRepetitionPenaltyReducesRepeats() {
        let logits = Tensor<Float>(shape: TensorShape(3), data: [0.8, 0.1, 0.1])
        var config = SamplerConfig(repetitionPenalty: 2.0)
        let history = [0, 0, 0]  // Token 0 repeated many times
        
        var seenTokens: Set<Int> = []
        for _ in 0..<50 {
            let token = Sampler.sample(logits: logits, config: &config, history: history)
            seenTokens.insert(token)
        }
        
        // Should avoid token 0 due to penalty
        XCTAssertTrue(seenTokens.contains(1) || seenTokens.contains(2),
                     "Should sample non-repeated tokens")
    }
    
    /// **Test:** Repetition penalty = 1.0 has no effect
    func testRepetitionPenaltyOne() {
        let logits = Tensor<Float>(shape: TensorShape(3), data: [0.8, 0.1, 0.1])
        var config = SamplerConfig(repetitionPenalty: 1.0)
        let history = [0, 0, 0]
        
        // With penalty=1.0, should still prefer token 0
        var counts = [0, 0, 0]
        for _ in 0..<100 {
            let token = Sampler.sample(logits: logits, config: &config, history: history)
            counts[token] += 1
        }
        
        // Token 0 should be most frequent (but allow statistical variance)
        XCTAssertGreaterThan(counts[0], counts[1], "Token 0 should be more frequent than others")
        XCTAssertGreaterThan(counts[0], counts[2], "Token 0 should be more frequent than others")
    }
    
    /// **Test:** Empty history means no penalty applied
    func testRepetitionPenaltyEmptyHistory() {
        let logits = Tensor<Float>(shape: TensorShape(3), data: [0.8, 0.1, 0.1])
        var config = SamplerConfig(repetitionPenalty: 2.0)
        let history: [Int] = []
        
        var counts = [0, 0, 0]
        for _ in 0..<100 {
            let token = Sampler.sample(logits: logits, config: &config, history: history)
            counts[token] += 1
        }
        
        // Empty history → no penalty → prefer token 0 (but allow some randomness)
        XCTAssertGreaterThan(counts[0], counts[1], "Token 0 should be most frequent")
        XCTAssertGreaterThan(counts[0], counts[2], "Token 0 should be most frequent")
    }
    
    // MARK: - Combined Configuration Tests
    
    /// **Test:** Full sampler with all strategies combined
    func testCombinedSamplerConfig() {
        let logits = Tensor<Float>(shape: TensorShape(5), data: [0.5, 0.6, 0.7, 0.8, 0.9])
        var config = SamplerConfig(
            temperature: 0.8,
            topK: 3,
            topP: 0.95,
            repetitionPenalty: 1.2
        )
        let history = [4]  // Penalize token 4
        
        var seenTokens: Set<Int> = []
        for _ in 0..<100 {
            let token = Sampler.sample(logits: logits, config: &config, history: history)
            seenTokens.insert(token)
        }
        
        // Should work without crashing and produce valid tokens
        XCTAssertFalse(seenTokens.isEmpty, "Should produce tokens")
        XCTAssertTrue(seenTokens.allSatisfy { $0 >= 0 && $0 < 5 }, 
                     "Tokens should be in valid range")
    }
    
    /// **Test:** Deterministic sampling with seed
    ///
    /// **REVIEW HITLER FIX:** RNG is now stateful, so sequence is deterministic but advances
    func testDeterministicWithSeed() {
        let logits = Tensor<Float>(shape: TensorShape(5), data: [0.1, 0.5, 0.2, 0.8, 0.3])
        
        // First sequence with seed=42
        var config1 = SamplerConfig(temperature: 1.0, seed: 42)
        var sequence1: [Int] = []
        for _ in 0..<5 {
            let token = Sampler.sample(logits: logits, config: &config1, history: [])
            sequence1.append(token)
        }
        
        // Second sequence with same seed - should match
        var config2 = SamplerConfig(temperature: 1.0, seed: 42)
        var sequence2: [Int] = []
        for _ in 0..<5 {
            let token = Sampler.sample(logits: logits, config: &config2, history: [])
            sequence2.append(token)
        }
        
        XCTAssertEqual(sequence1, sequence2, "Same seed should produce same deterministic sequence")
        
        // Sequence should have diversity (not all same token)
        let uniqueTokens = Set(sequence1)
        XCTAssertGreaterThan(uniqueTokens.count, 1, "Should produce diverse sequence")
    }
    
    // MARK: - Edge Cases
    
    /// **Test:** Sampling with uniform logits
    func testUniformLogits() {
        let logits = Tensor<Float>(shape: TensorShape(4), data: [0.5, 0.5, 0.5, 0.5])
        
        var seenTokens: Set<Int> = []
        for _ in 0..<100 {
            let token = Sampler.temperature(logits: logits, temp: 1.0)
            seenTokens.insert(token)
        }
        
        // Uniform distribution should sample all tokens roughly equally
        XCTAssertGreaterThanOrEqual(seenTokens.count, 3, 
                                    "Should sample from uniform distribution")
    }
    
    /// **Test:** Sampling with single token (vocab size = 1)
    func testSingleToken() {
        let logits = Tensor<Float>(shape: TensorShape(1), data: [1.0])
        
        let token = Sampler.greedy(logits: logits)
        XCTAssertEqual(token, 0, "Single token should always be selected")
    }
}

