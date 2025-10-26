import XCTest
@testable import TinyBrainRuntime

/// Tests for sampler edge cases and bug fixes
///
/// **REVIEW HITLER:** These tests expose critical bugs in the initial implementation
final class SamplerBugTests: XCTestCase {
    
    // MARK: - Bug #2: Top-K Doesn't Actually Limit to K
    
    /// **Bug:** Top-K uses threshold comparison, so tied values
    /// can result in > K tokens being kept
    ///
    /// **Expected:** Exactly K tokens (highest by index if tied)
    func testTopKExactlyKTokens() {
        // Create logits with many tied values
        let logits = Tensor<Float>(shape: TensorShape(10), data: [
            0.5, 0.5, 0.5, 0.5, 0.5,  // 5 tied at 0.5
            0.3, 0.3, 0.3,              // 3 tied at 0.3
            0.1, 0.1                    // 2 tied at 0.1
        ])
        
        // With K=3 and threshold-based filtering, might keep all 5 tokens at 0.5
        // Should keep EXACTLY 3 tokens (first 3 indices: 0, 1, 2)
        var seenTokens: Set<Int> = []
        for _ in 0..<200 {
            let token = Sampler.topK(logits: logits, k: 3, temp: 1.0)
            seenTokens.insert(token)
        }
        
        // Should see EXACTLY 3 unique tokens
        XCTAssertEqual(seenTokens.count, 3,
                      "Top-K should limit to exactly K tokens, not K+ties")
        
        // Should be the first K indices (0, 1, 2)
        XCTAssertTrue(seenTokens.isSubset(of: Set(0...2)),
                     "Should keep first K indices when tied")
    }
    
    /// **Bug:** Top-K with all equal logits keeps all tokens
    func testTopKWithAllEqualLogits() {
        let logits = Tensor<Float>(shape: TensorShape(50), data: Array(repeating: 1.0, count: 50))
        
        var seenTokens: Set<Int> = []
        for _ in 0..<200 {
            let token = Sampler.topK(logits: logits, k: 5, temp: 1.0)
            seenTokens.insert(token)
        }
        
        // Should see EXACTLY 5 tokens (first 5 indices)
        XCTAssertEqual(seenTokens.count, 5,
                      "Top-K=5 with uniform logits should keep exactly 5 tokens")
        XCTAssertTrue(seenTokens.isSubset(of: Set(0..<5)),
                     "Should keep first K indices")
    }
    
    // MARK: - Bug #3: Seeded Sampling Degenerates
    
    /// **Bug:** Each sample call creates new RNG with same seed,
    /// so all samples produce identical output
    ///
    /// **Expected:** Seed should produce deterministic SEQUENCE,
    /// not same value every time
    func testSeededSamplingProducesSequence() {
        let logits = Tensor<Float>(shape: TensorShape(5), data: [0.2, 0.2, 0.2, 0.2, 0.2])
        var config = SamplerConfig(temperature: 1.0, seed: 42)  // **FIX:** var for inout
        
        // Sample 10 times with same seed
        var tokens: [Int] = []
        for _ in 0..<10 {
            let token = Sampler.sample(logits: logits, config: &config, history: [])
            tokens.append(token)
        }
        
        // Should get DIFFERENT tokens (deterministic sequence)
        // NOT the same token 10 times!
        let uniqueTokens = Set(tokens)
        XCTAssertGreaterThan(uniqueTokens.count, 1,
                            "Seeded sampling should produce a sequence, not same value")
        
        // But repeating with same seed should give same sequence
        var config2 = SamplerConfig(temperature: 1.0, seed: 42)  // Fresh config
        var tokens2: [Int] = []
        for _ in 0..<10 {
            let token = Sampler.sample(logits: logits, config: &config2, history: [])
            tokens2.append(token)
        }
        
        XCTAssertEqual(tokens, tokens2,
                      "Same seed should produce same sequence")
    }
    
    /// **Bug:** ModelRunner streaming with seed produces identical tokens
    func testStreamingWithSeedProducesDiverseSequence() async throws {
        let config = ModelConfig(
            numLayers: 2,
            hiddenDim: 64,
            numHeads: 4,
            vocabSize: 100,
            maxSeqLen: 128
        )
        let runner = ModelRunner(config: config)
        
        let genConfig = GenerationConfig(
            maxTokens: 20,
            sampler: SamplerConfig(temperature: 1.0, seed: 42)
        )
        
        var tokens: [Int] = []
        for try await output in runner.generateStream(prompt: [1], config: genConfig) {
            tokens.append(output.tokenId)
        }
        
        // Should generate a diverse sequence, not all the same token
        let uniqueTokens = Set(tokens)
        XCTAssertGreaterThan(uniqueTokens.count, 3,
                            "Seeded streaming should produce diverse sequence (found \(uniqueTokens.count) unique tokens)")
    }
}

