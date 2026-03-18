// TinyBrainEmbedder.swift
// TinyBrainProximaKit
//
// Bridges TinyBrain inference to ProximaKit's TextEmbedder protocol,
// enabling on-device embedding generation for vector similarity search.

import Foundation
import ProximaKit
import TinyBrainRuntime
import TinyBrainTokenizer

/// Generates text embeddings using a TinyBrain `ModelRunner`.
///
/// Conforms to ProximaKit's ``TextEmbedder`` protocol, allowing any
/// TinyBrain-compatible model to power vector similarity search through
/// ProximaKit indices and stores.
///
/// ```swift
/// let runner = ModelRunner(weights: myWeights)
/// let tokenizer = BPETokenizer(vocabulary: vocab)
/// let embedder = TinyBrainEmbedder(runner: runner, tokenizer: tokenizer)
///
/// let vector = try await embedder.embed("hello world")
/// // vector.dimension == runner.config.hiddenDim
/// ```
///
/// - Important: `ModelRunner` is **not** thread-safe. The embedder serializes
///   all calls to `extractEmbedding` through an internal actor. This is safe
///   when used with ProximaKit's actor-isolated indices.
public final class TinyBrainEmbedder: TextEmbedder, @unchecked Sendable {

    // MARK: - Private State

    private let runner: ModelRunner
    private let tokenizer: any Tokenizer
    private let lock = NSLock()

    // MARK: - TextEmbedder

    /// The dimension of the embedding vectors produced by this embedder.
    ///
    /// Matches the model's hidden dimension (`ModelConfig.hiddenDim`).
    public var dimension: Int { runner.config.hiddenDim }

    // MARK: - Initialization

    /// Creates an embedder backed by the given model runner and tokenizer.
    ///
    /// - Parameters:
    ///   - runner: A configured `ModelRunner` with loaded weights.
    ///   - tokenizer: A tokenizer compatible with the model's vocabulary.
    public init(runner: ModelRunner, tokenizer: any Tokenizer) {
        self.runner = runner
        self.tokenizer = tokenizer
    }

    // MARK: - Embedding

    /// Embeds a single text string into a vector.
    ///
    /// Tokenizes the input, runs a forward pass through all transformer layers,
    /// extracts the final hidden state (post-RMSNorm, pre-projection), and
    /// returns it as a ProximaKit `Vector`.
    ///
    /// - Parameter text: The text to embed.
    /// - Returns: A `Vector` of dimension `hiddenDim`.
    public func embed(_ text: String) async throws -> Vector {
        let tokenIds = tokenizer.encode(text)
        guard !tokenIds.isEmpty else {
            throw TinyBrainEmbedderError.emptyTokenization(text: text)
        }

        let floats: [Float] = lock.withLock {
            let tensor = runner.extractEmbedding(for: tokenIds)
            return tensor.scalars
        }

        return Vector(floats)
    }

    /// Embeds multiple texts sequentially.
    ///
    /// Overrides the default concurrent `embedBatch` from `TextEmbedder`
    /// because `ModelRunner` is not thread-safe — concurrent extraction
    /// would corrupt the KV cache.
    ///
    /// - Parameter texts: The texts to embed.
    /// - Returns: An array of vectors, one per input text, in order.
    public func embedBatch(_ texts: [String]) async throws -> [Vector] {
        var vectors: [Vector] = []
        vectors.reserveCapacity(texts.count)
        for text in texts {
            try await vectors.append(embed(text))
        }
        return vectors
    }
}

// MARK: - Errors

/// Errors specific to TinyBrain embedding generation.
public enum TinyBrainEmbedderError: Error, LocalizedError {
    /// The tokenizer produced zero tokens for the given text.
    case emptyTokenization(text: String)

    public var errorDescription: String? {
        switch self {
        case .emptyTokenization(let text):
            return "Tokenizer produced no tokens for input: \"\(text)\""
        }
    }
}
