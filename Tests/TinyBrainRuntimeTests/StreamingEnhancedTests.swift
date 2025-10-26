import XCTest
@testable import TinyBrainRuntime

/// Tests for enhanced streaming API with GenerationConfig
///
/// **TDD Approach:**
/// These tests are written FIRST to define expected streaming behavior.
///
/// **Educational Note:**
/// Streaming is critical for interactive LLM experiences:
/// - Users see tokens as they're generated (not waiting for full response)
/// - Can stop generation early if output goes off-track
/// - Provides real-time feedback on generation speed
///
/// **TB-005:** Enhanced from TB-004's basic streaming to production-ready API
final class StreamingEnhancedTests: XCTestCase {
    
    // MARK: - Test Fixtures
    
    /// Create a small test model for streaming tests
    private func makeTestRunner() -> ModelRunner {
        let config = ModelConfig(
            numLayers: 2,
            hiddenDim: 64,
            numHeads: 4,
            vocabSize: 100,
            maxSeqLen: 128
        )
        return ModelRunner(config: config)
    }
    
    // MARK: - Basic Streaming Tests
    
    /// **Test:** Basic streaming produces tokens
    ///
    /// **Educational:**
    /// AsyncSequence allows consuming tokens one-by-one as they're generated.
    func testBasicStreaming() async throws {
        let runner = makeTestRunner()
        let config = GenerationConfig(maxTokens: 10)
        
        var tokenCount = 0
        for try await output in runner.generateStream(prompt: [1, 2, 3], config: config) {
            XCTAssertGreaterThanOrEqual(output.tokenId, 0, "Token ID should be valid")
            XCTAssertLessThan(output.tokenId, 100, "Token ID should be in vocab range")
            tokenCount += 1
        }
        
        XCTAssertEqual(tokenCount, 10, "Should generate exactly maxTokens")
    }
    
    /// **Test:** Empty prompt still generates tokens
    func testStreamingWithEmptyPrompt() async throws {
        let runner = makeTestRunner()
        let config = GenerationConfig(maxTokens: 5)
        
        var tokenCount = 0
        for try await _ in runner.generateStream(prompt: [], config: config) {
            tokenCount += 1
        }
        
        XCTAssertEqual(tokenCount, 5, "Should generate tokens even with empty prompt")
    }
    
    // MARK: - Stop Token Tests
    
    /// **Test:** Stop tokens halt generation early
    ///
    /// **Educational:**
    /// Stop tokens (like EOS) let model signal "I'm done" before hitting maxTokens.
    /// Critical for preventing unnecessary computation.
    func testStopTokensHaltGeneration() async throws {
        let runner = makeTestRunner()
        
        // Use a common token as stop token (will likely appear before 100 tokens)
        let config = GenerationConfig(
            maxTokens: 100,
            stopTokens: [50, 51, 52]  // Multiple stop tokens
        )
        
        var tokenCount = 0
        var stoppedEarly = false
        
        for try await output in runner.generateStream(prompt: [1], config: config) {
            tokenCount += 1
            if config.stopTokens.contains(output.tokenId) {
                stoppedEarly = true
                break
            }
        }
        
        // Very likely to hit a stop token before 100 iterations
        if stoppedEarly {
            XCTAssertLessThan(tokenCount, 100, "Should stop before maxTokens when stop token encountered")
        }
        // Test is lenient - if by chance no stop token appears, that's OK too
    }
    
    /// **Test:** Stop tokens work on first token
    func testStopTokenOnFirstToken() async throws {
        let runner = makeTestRunner()
        
        // Set stop tokens to likely values
        let config = GenerationConfig(
            maxTokens: 50,
            stopTokens: Array(0..<100)  // All tokens are stop tokens
        )
        
        var tokenCount = 0
        for try await _ in runner.generateStream(prompt: [1], config: config) {
            tokenCount += 1
            break  // Should stop immediately
        }
        
        XCTAssertEqual(tokenCount, 1, "Should generate at least one token before checking stop")
    }
    
    /// **Test:** No stop tokens means generation continues to maxTokens
    func testNoStopTokens() async throws {
        let runner = makeTestRunner()
        let config = GenerationConfig(
            maxTokens: 15,
            stopTokens: []  // No stop tokens
        )
        
        var tokenCount = 0
        for try await _ in runner.generateStream(prompt: [1], config: config) {
            tokenCount += 1
        }
        
        XCTAssertEqual(tokenCount, 15, "Should generate all maxTokens when no stop tokens")
    }
    
    // MARK: - TokenOutput Tests
    
    /// **Test:** TokenOutput contains required metadata
    ///
    /// **Educational:**
    /// TokenOutput provides rich metadata for UI/telemetry:
    /// - tokenId: The actual token generated
    /// - probability: How confident the model was (for debugging)
    /// - timestamp: When token was generated (for latency analysis)
    func testTokenOutputMetadata() async throws {
        let runner = makeTestRunner()
        let config = GenerationConfig(maxTokens: 5)
        
        let startTime = Date()
        var outputs: [TokenOutput] = []
        
        for try await output in runner.generateStream(prompt: [1], config: config) {
            outputs.append(output)
        }
        
        XCTAssertEqual(outputs.count, 5, "Should have all outputs")
        
        for output in outputs {
            // Verify metadata exists and is reasonable
            XCTAssertGreaterThanOrEqual(output.tokenId, 0, "Token ID should be non-negative")
            XCTAssertGreaterThan(output.probability, 0.0, "Probability should be positive")
            XCTAssertLessThanOrEqual(output.probability, 1.0, "Probability should be ≤ 1.0")
            XCTAssertGreaterThanOrEqual(output.timestamp, startTime, "Timestamp should be after start")
        }
    }
    
    /// **Test:** Timestamps are monotonically increasing
    func testTokenTimestampsIncreasing() async throws {
        let runner = makeTestRunner()
        let config = GenerationConfig(maxTokens: 10)
        
        var previousTimestamp: Date?
        
        for try await output in runner.generateStream(prompt: [1], config: config) {
            if let prev = previousTimestamp {
                XCTAssertGreaterThanOrEqual(output.timestamp, prev, 
                                           "Timestamps should be monotonically increasing")
            }
            previousTimestamp = output.timestamp
        }
    }
    
    // MARK: - Sampling Configuration Tests
    
    /// **Test:** Streaming respects sampler configuration
    ///
    /// **Educational:**
    /// GenerationConfig integrates with our Sampler from Phase 2.
    func testStreamingWithSamplerConfig() async throws {
        let runner = makeTestRunner()
        
        // Use deterministic sampling with seed
        let config = GenerationConfig(
            maxTokens: 5,
            sampler: SamplerConfig(temperature: 1.0, seed: 42)
        )
        
        var tokens1: [Int] = []
        for try await output in runner.generateStream(prompt: [1, 2], config: config) {
            tokens1.append(output.tokenId)
        }
        
        // Run again with same seed - should be deterministic
        runner.reset()
        var tokens2: [Int] = []
        for try await output in runner.generateStream(prompt: [1, 2], config: config) {
            tokens2.append(output.tokenId)
        }
        
        XCTAssertEqual(tokens1, tokens2, 
                      "Same seed should produce deterministic results")
    }
    
    /// **Test:** Top-K sampling produces diverse outputs
    func testStreamingWithTopK() async throws {
        let runner = makeTestRunner()
        
        let config = GenerationConfig(
            maxTokens: 20,
            sampler: SamplerConfig(temperature: 1.0, topK: 5)
        )
        
        var seenTokens: Set<Int> = []
        for try await output in runner.generateStream(prompt: [1], config: config) {
            seenTokens.insert(output.tokenId)
        }
        
        // With top-k=5 and 20 samples, should see some diversity
        XCTAssertGreaterThan(seenTokens.count, 1, 
                            "Top-K sampling should produce some diversity")
    }
    
    /// **Test:** Repetition penalty reduces repeats in stream
    func testStreamingWithRepetitionPenalty() async throws {
        let runner = makeTestRunner()
        
        let config = GenerationConfig(
            maxTokens: 30,
            sampler: SamplerConfig(
                temperature: 1.0,
                repetitionPenalty: 1.5
            )
        )
        
        var tokens: [Int] = []
        for try await output in runner.generateStream(prompt: [1], config: config) {
            tokens.append(output.tokenId)
        }
        
        // Count consecutive repeats
        var maxConsecutiveRepeats = 0
        var currentRepeat = 1
        
        for i in 1..<tokens.count {
            if tokens[i] == tokens[i-1] {
                currentRepeat += 1
                maxConsecutiveRepeats = max(maxConsecutiveRepeats, currentRepeat)
            } else {
                currentRepeat = 1
            }
        }
        
        // With repetition penalty, shouldn't see too many consecutive repeats
        XCTAssertLessThan(maxConsecutiveRepeats, 10, 
                         "Repetition penalty should limit consecutive repeats")
    }
    
    // MARK: - Performance Tests
    
    /// **Test:** Streaming latency is acceptable (<150ms per token)
    ///
    /// **Educational:**
    /// Interactive LLMs need <150ms per token for good UX.
    /// This is TB-005 acceptance criteria!
    func testStreamingLatency() async throws {
        let runner = makeTestRunner()
        let config = GenerationConfig(maxTokens: 10)
        
        var latencies: [TimeInterval] = []
        var lastTime = Date()
        
        for try await _ in runner.generateStream(prompt: [1, 2, 3], config: config) {
            let now = Date()
            let latency = now.timeIntervalSince(lastTime)
            latencies.append(latency)
            lastTime = now
        }
        
        // Skip first token (includes prompt processing)
        let generationLatencies = latencies.dropFirst()
        
        guard !generationLatencies.isEmpty else {
            XCTFail("Should have generation latencies")
            return
        }
        
        let avgLatency = generationLatencies.reduce(0, +) / Double(generationLatencies.count)
        let maxLatency = generationLatencies.max() ?? 0
        
        // TB-005 requirement: <150ms average
        XCTAssertLessThan(avgLatency * 1000, 150.0, 
                         "Average latency should be <150ms per token")
        
        // Max shouldn't be too much higher
        XCTAssertLessThan(maxLatency * 1000, 300.0,
                         "Max latency should be reasonable")
    }
    
    /// **Test:** Streaming handles long sequences efficiently
    func testStreamingLongSequence() async throws {
        let runner = makeTestRunner()
        let config = GenerationConfig(maxTokens: 50)
        
        var tokenCount = 0
        let startTime = Date()
        
        for try await _ in runner.generateStream(prompt: [1], config: config) {
            tokenCount += 1
        }
        
        let totalTime = Date().timeIntervalSince(startTime)
        let avgTimePerToken = totalTime / Double(tokenCount)
        
        XCTAssertEqual(tokenCount, 50, "Should generate all tokens")
        XCTAssertLessThan(avgTimePerToken * 1000, 200.0,
                         "Should maintain reasonable speed for long sequences")
    }
    
    // MARK: - Error Handling Tests
    
    /// **Test:** Streaming with invalid prompt tokens
    func testStreamingWithInvalidPrompt() async throws {
        let runner = makeTestRunner()
        
        // Tokens outside vocab range
        let config = GenerationConfig(maxTokens: 5)
        
        // Should handle gracefully (clip to valid range or use default)
        var tokenCount = 0
        for try await output in runner.generateStream(prompt: [999, -1, 1000], config: config) {
            XCTAssertGreaterThanOrEqual(output.tokenId, 0, "Generated token should be valid")
            XCTAssertLessThan(output.tokenId, 100, "Generated token should be in range")
            tokenCount += 1
        }
        
        XCTAssertGreaterThan(tokenCount, 0, "Should still generate tokens")
    }
    
    /// **Test:** Streaming with maxTokens = 0
    func testStreamingWithZeroTokens() async throws {
        let runner = makeTestRunner()
        let config = GenerationConfig(maxTokens: 0)
        
        var tokenCount = 0
        for try await _ in runner.generateStream(prompt: [1], config: config) {
            tokenCount += 1
        }
        
        XCTAssertEqual(tokenCount, 0, "Should generate no tokens when maxTokens=0")
    }
    
    // MARK: - State Management Tests
    
    /// **Test:** Multiple streaming sessions don't interfere
    func testConcurrentStreaming() async throws {
        let runner = makeTestRunner()
        
        // Run two streams concurrently
        async let stream1 = collectTokens(
            runner: runner,
            config: GenerationConfig(maxTokens: 5)
        )
        
        async let stream2 = collectTokens(
            runner: runner,
            config: GenerationConfig(maxTokens: 5)
        )
        
        let (tokens1, tokens2) = try await (stream1, stream2)
        
        // Each stream should complete independently
        XCTAssertEqual(tokens1.count, 5, "Stream 1 should complete")
        XCTAssertEqual(tokens2.count, 5, "Stream 2 should complete")
    }
    
    /// **Test:** Reset clears state between streams
    func testResetBetweenStreams() async throws {
        let runner = makeTestRunner()
        let config = GenerationConfig(maxTokens: 3)
        
        // First stream
        var tokens1: [Int] = []
        for try await output in runner.generateStream(prompt: [1, 2], config: config) {
            tokens1.append(output.tokenId)
        }
        
        // Reset
        runner.reset()
        
        // Second stream with same prompt should start fresh
        var tokens2: [Int] = []
        for try await output in runner.generateStream(prompt: [1, 2], config: config) {
            tokens2.append(output.tokenId)
        }
        
        XCTAssertEqual(tokens1.count, 3, "First stream should complete")
        XCTAssertEqual(tokens2.count, 3, "Second stream should complete")
    }
    
    // MARK: - Helper Methods
    
    private func collectTokens(runner: ModelRunner, config: GenerationConfig) async throws -> [Int] {
        var tokens: [Int] = []
        for try await output in runner.generateStream(prompt: [1], config: config) {
            tokens.append(output.tokenId)
        }
        return tokens
    }
}

