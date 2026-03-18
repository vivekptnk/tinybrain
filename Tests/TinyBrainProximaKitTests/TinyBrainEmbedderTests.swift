// TinyBrainEmbedderTests.swift
// TinyBrainProximaKitTests

import XCTest
import ProximaKit
import TinyBrainRuntime
import TinyBrainTokenizer
@testable import TinyBrainProximaKit

// MARK: - Test Tokenizer

/// Minimal tokenizer for testing: maps each character's Unicode scalar value to a token ID.
private struct CharTokenizer: Tokenizer {
    let vocabularySize: Int = 256

    func encode(_ text: String) -> [Int] {
        text.unicodeScalars.map { min(Int($0.value), vocabularySize - 1) }
    }

    func decode(_ tokens: [Int]) -> String {
        String(tokens.compactMap { UnicodeScalar($0) }.map { Character($0) })
    }
}

/// Tokenizer that always returns an empty array.
private struct EmptyTokenizer: Tokenizer {
    let vocabularySize: Int = 256
    func encode(_ text: String) -> [Int] { [] }
    func decode(_ tokens: [Int]) -> String { "" }
}

// MARK: - Tests

final class TinyBrainEmbedderTests: XCTestCase {

    private let config = ModelConfig(numLayers: 2, hiddenDim: 64, numHeads: 2, vocabSize: 256)

    private func makeEmbedder(tokenizer: (any Tokenizer)? = nil) -> TinyBrainEmbedder {
        let weights = ModelWeights.makeToyModel(config: config, seed: 42)
        let runner = ModelRunner(weights: weights)
        return TinyBrainEmbedder(runner: runner, tokenizer: tokenizer ?? CharTokenizer())
    }

    // MARK: - Dimension

    func testDimensionMatchesModelConfig() {
        let embedder = makeEmbedder()
        XCTAssertEqual(embedder.dimension, 64)
    }

    // MARK: - Single Embedding

    func testEmbedReturnsCorrectDimension() async throws {
        let embedder = makeEmbedder()
        let vector = try await embedder.embed("hello")
        XCTAssertEqual(vector.dimension, 64)
    }

    func testEmbedIsDeterministic() async throws {
        let embedder = makeEmbedder()
        let v1 = try await embedder.embed("test")
        let v2 = try await embedder.embed("test")
        XCTAssertEqual(v1, v2, "Same input should produce identical vectors")
    }

    func testEmbedDiffersForDifferentInputs() async throws {
        let embedder = makeEmbedder()
        let v1 = try await embedder.embed("hello")
        let v2 = try await embedder.embed("world")
        XCTAssertNotEqual(v1, v2, "Different inputs should produce different vectors")
    }

    func testEmbedProducesNonZeroVector() async throws {
        let embedder = makeEmbedder()
        let vector = try await embedder.embed("abc")
        let hasNonZero = vector.components.contains { $0 != 0.0 }
        XCTAssertTrue(hasNonZero, "Embedding should not be all zeros")
    }

    // MARK: - Batch Embedding

    func testEmbedBatchCountMatchesInput() async throws {
        let embedder = makeEmbedder()
        let vectors = try await embedder.embedBatch(["a", "b", "c"])
        XCTAssertEqual(vectors.count, 3)
    }

    func testEmbedBatchPreservesOrder() async throws {
        let embedder = makeEmbedder()
        let texts = ["alpha", "beta", "gamma"]
        let batch = try await embedder.embedBatch(texts)

        // Each batch result should match individual embed
        for (i, text) in texts.enumerated() {
            let single = try await embedder.embed(text)
            XCTAssertEqual(batch[i], single,
                           "Batch[\(i)] should match individual embed for \"\(text)\"")
        }
    }

    func testEmbedBatchEmpty() async throws {
        let embedder = makeEmbedder()
        let vectors = try await embedder.embedBatch([])
        XCTAssertEqual(vectors.count, 0)
    }

    // MARK: - Error Cases

    func testEmptyTokenizationThrows() async {
        let embedder = makeEmbedder(tokenizer: EmptyTokenizer())
        do {
            _ = try await embedder.embed("anything")
            XCTFail("Expected emptyTokenization error")
        } catch let error as TinyBrainEmbedderError {
            if case .emptyTokenization = error {
                // Expected
            } else {
                XCTFail("Wrong error case: \(error)")
            }
        } catch {
            XCTFail("Unexpected error type: \(error)")
        }
    }

    // MARK: - ProximaKit Integration

    func testVectorIsUsableWithCosineSimilarity() async throws {
        let embedder = makeEmbedder()
        let v1 = try await embedder.embed("hello")
        let v2 = try await embedder.embed("hello")
        let similarity = v1.cosineSimilarity(v2)
        XCTAssertEqual(similarity, 1.0, accuracy: 1e-5,
                       "Identical inputs should have cosine similarity of 1.0")
    }
}
