/// Tests for streaming inference API
///
/// **TB-004 Phase 5:** Validates ModelRunner with KV cache reuse for efficient token generation
///
/// ## What is Streaming Inference?
///
/// Instead of waiting for the entire response:
/// ```
/// Non-streaming: "....................." [3 seconds] "Hello world!"
/// Streaming:     "H" "e" "l" "l" "o" " " "w" "o" "r" "l" "d" "!"
///                ↑ Each token appears immediately!
/// ```
///
/// **The Key:** Reuse KV cache so we don't recompute past tokens!
///
/// ```
/// Token 0: Compute from scratch
/// Token 1: Reuse Token 0's K/V, only compute Token 1
/// Token 2: Reuse Tokens 0-1's K/V, only compute Token 2
/// ...
/// ```

import XCTest
@testable import TinyBrainRuntime

final class StreamingTests: XCTestCase {
    
    // MARK: - ModelRunner.step() Tests
    
    func testStepReusesKVCache() {
        // WHAT: step() doesn't recompute past tokens
        // WHY: KV cache performance benefit - O(n) not O(n²)
        // HOW: Track cache size, verify it grows by 1 per step
        
        let config = ModelConfig(
            numLayers: 2,
            hiddenDim: 64,
            numHeads: 2,
            vocabSize: 1000
        )
        
        let runner = ModelRunner(config: config)
        
        // First step
        let position0 = runner.currentPosition
        runner.step(tokenId: 100)
        
        XCTAssertEqual(runner.currentPosition, position0 + 1, "Position should increment")
        XCTAssertEqual(runner.kvCache.length, 1, "Cache should have 1 token")
        
        // Second step - should reuse first token's K/V
        runner.step(tokenId: 200)
        
        XCTAssertEqual(runner.currentPosition, position0 + 2, "Position should increment again")
        XCTAssertEqual(runner.kvCache.length, 2, "Cache should have 2 tokens")
    }
    
    func testStepGeneratesLogits() {
        // WHAT: step() returns logits for next token prediction
        // WHY: Need logits to sample next token
        // HOW: Verify shape is [vocabSize]
        
        let config = ModelConfig(
            numLayers: 2,
            hiddenDim: 64,
            numHeads: 2,
            vocabSize: 1000
        )
        
        let runner = ModelRunner(config: config)
        let logits = runner.step(tokenId: 42)
        
        XCTAssertEqual(logits.shape, TensorShape(1000), "Logits should be [vocabSize]")
    }
    
    func testMultipleSteps() {
        // WHAT: Multiple step() calls build up context
        // WHY: Token-by-token generation
        // HOW: 10 steps, verify cache grows to 10
        
        let config = ModelConfig(
            numLayers: 2,
            hiddenDim: 64,
            numHeads: 2,
            vocabSize: 1000
        )
        
        let runner = ModelRunner(config: config)
        
        for i in 0..<10 {
            let _ = runner.step(tokenId: i)
        }
        
        XCTAssertEqual(runner.currentPosition, 10, "Should have processed 10 tokens")
        XCTAssertEqual(runner.kvCache.length, 10, "Cache should have 10 tokens")
    }
    
    func testResetClearsState() {
        // WHAT: reset() clears KV cache and position
        // WHY: Start new sequence/conversation
        // HOW: Generate tokens, reset, verify empty
        
        let config = ModelConfig(
            numLayers: 2,
            hiddenDim: 64,
            numHeads: 2,
            vocabSize: 1000
        )
        
        let runner = ModelRunner(config: config)
        
        // Generate 20 tokens
        for i in 0..<20 {
            let _ = runner.step(tokenId: i)
        }
        
        XCTAssertEqual(runner.currentPosition, 20)
        XCTAssertEqual(runner.kvCache.length, 20)
        
        // Reset
        runner.reset()
        
        XCTAssertEqual(runner.currentPosition, 0, "Position should reset to 0")
        XCTAssertEqual(runner.kvCache.length, 0, "Cache should be empty")
    }
    
    // MARK: - AsyncSequence Streaming Tests
    
    func testStreamGeneration() async throws {
        // WHAT: Generate stream of tokens using AsyncSequence
        // WHY: Progressive output for UI (SwiftUI updates)
        // HOW: Collect tokens from stream, verify count
        
        let config = ModelConfig(
            numLayers: 2,
            hiddenDim: 64,
            numHeads: 2,
            vocabSize: 1000
        )
        
        let runner = ModelRunner(config: config)
        var tokens: [Int] = []
        
        for try await tokenId in runner.generateStream(prompt: [1, 2, 3], maxTokens: 10) {
            tokens.append(tokenId)
            if tokens.count >= 10 { break }
        }
        
        XCTAssertEqual(tokens.count, 10, "Should generate 10 tokens")
    }
    
    func testStreamCancellation() async throws {
        // WHAT: Stream can be cancelled mid-generation
        // WHY: User stops generation early
        // HOW: Break from loop, verify partial results
        
        let config = ModelConfig(
            numLayers: 2,
            hiddenDim: 64,
            numHeads: 2,
            vocabSize: 1000
        )
        
        let runner = ModelRunner(config: config)
        var tokens: [Int] = []
        
        for try await tokenId in runner.generateStream(prompt: [1, 2, 3], maxTokens: 100) {
            tokens.append(tokenId)
            if tokens.count >= 5 {
                break  // Early stop
            }
        }
        
        XCTAssertEqual(tokens.count, 5, "Should stop at 5 tokens")
        XCTAssertLessThan(runner.currentPosition, 100, "Should not generate all 100")
    }
    
    func testMultipleStreams() async throws {
        // WHAT: Reset between streams
        // WHY: Multiple conversations/prompts
        // HOW: Generate, reset, generate again
        
        let config = ModelConfig(
            numLayers: 2,
            hiddenDim: 64,
            numHeads: 2,
            vocabSize: 1000
        )
        
        let runner = ModelRunner(config: config)
        
        // First stream
        var stream1: [Int] = []
        for try await tokenId in runner.generateStream(prompt: [1], maxTokens: 5) {
            stream1.append(tokenId)
        }
        
        runner.reset()
        
        // Second stream
        var stream2: [Int] = []
        for try await tokenId in runner.generateStream(prompt: [2], maxTokens: 5) {
            stream2.append(tokenId)
        }
        
        XCTAssertEqual(stream1.count, 5)
        XCTAssertEqual(stream2.count, 5)
    }
}

// ModelConfig is now in ModelRunner.swift (no longer mocked)

