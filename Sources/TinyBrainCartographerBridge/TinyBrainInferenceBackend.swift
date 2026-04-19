// TinyBrainInferenceBackend — production `InferenceBackend` backed by
// `TinyBrainRuntime` + `TinyBrainTokenizer`. Lazy-loads weights on first
// `prewarm`, then keeps them resident for the process lifetime unless the
// adapter actor evicts.
//
// The backend is `@unchecked Sendable` because the underlying ModelRunner
// is not thread-safe but the owning `TinyBrainSmartService` actor already
// serializes every call. An internal NSLock guards state transitions that
// the actor contract alone does not cover (e.g. concurrent
// `estimatedTokenCount` calls from non-isolated contexts).

import Foundation
import TinyBrainRuntime
import TinyBrainTokenizer

/// Production backend that loads a TBF model + tokenizer and streams
/// tokens through `ModelRunner`.
public final class TinyBrainInferenceBackend: InferenceBackend, @unchecked Sendable {

    // MARK: - Config

    private let modelURL: URL
    private let tokenizerURL: URL?

    // MARK: - State (guarded by `lock`)

    private let lock = NSLock()
    private var runner: ModelRunner?
    private var tokenizer: (any Tokenizer)?

    // MARK: - Init

    public init(modelURL: URL, tokenizerURL: URL? = nil) {
        self.modelURL = modelURL
        self.tokenizerURL = tokenizerURL
    }

    // MARK: - InferenceBackend

    public func estimatedTokenCount(_ text: String) -> Int {
        let encoder: (any Tokenizer)? = lock.withLock { tokenizer }
        if let encoder {
            return encoder.encode(text).count
        }
        // §4.1 pessimistic heuristic until the tokenizer is loaded.
        return max(1, (text.count + 3) / 4)
    }

    public func prewarm() async throws {
        let alreadyLoaded: Bool = lock.withLock { runner != nil && tokenizer != nil }
        if alreadyLoaded { return }

        try Task.checkCancellation()

        let weights = try ModelWeights.load(from: modelURL.path)
        let loadedRunner = ModelRunner(weights: weights)

        let loadedTokenizer = try await loadTokenizer()

        lock.withLock {
            self.runner = loadedRunner
            self.tokenizer = loadedTokenizer
        }
    }

    public func generate(
        prompt: String,
        maxOutputTokens: Int
    ) async throws -> String {
        try await prewarm()
        try Task.checkCancellation()

        let loaded: (ModelRunner, any Tokenizer)? = lock.withLock {
            guard let runner = self.runner, let tokenizer = self.tokenizer else {
                return nil
            }
            return (runner, tokenizer)
        }
        guard let (runner, tokenizer) = loaded else {
            throw BackendError.notLoaded
        }

        let promptTokens = tokenizer.encode(prompt)
        var collected: [Int] = []
        collected.reserveCapacity(maxOutputTokens)

        let config = GenerationConfig(maxTokens: max(1, maxOutputTokens))
        let stream = runner.generateStream(prompt: promptTokens, config: config)

        for try await output in stream {
            try Task.checkCancellation()
            collected.append(output.tokenId)
            if collected.count >= maxOutputTokens { break }
        }

        return tokenizer.decode(collected)
    }

    // MARK: - Tokenizer resolution

    private func loadTokenizer() async throws -> any Tokenizer {
        if let tokenizerURL {
            return try TokenizerLoader.load(from: tokenizerURL.path)
        }
        // Try sibling file conventions: `<model-stem>.tokenizer.json`,
        // `tokenizer.json`, `tokenizer.model` next to the .tbf.
        let directory = modelURL.deletingLastPathComponent()
        let candidates: [String] = [
            modelURL.deletingPathExtension().appendingPathExtension("tokenizer.json").path,
            directory.appendingPathComponent("tokenizer.json").path,
            directory.appendingPathComponent("tokenizer.model").path
        ]
        for candidate in candidates {
            if FileManager.default.fileExists(atPath: candidate) {
                return try TokenizerLoader.load(from: candidate)
            }
        }
        throw BackendError.tokenizerNotFound(searched: candidates)
    }

    // MARK: - Errors

    public enum BackendError: Error, CustomStringConvertible {
        case notLoaded
        case tokenizerNotFound(searched: [String])

        public var description: String {
            switch self {
            case .notLoaded:
                return "TinyBrainInferenceBackend not loaded"
            case .tokenizerNotFound(let searched):
                return "tokenizer not found (tried: \(searched.joined(separator: ", ")))"
            }
        }
    }
}
