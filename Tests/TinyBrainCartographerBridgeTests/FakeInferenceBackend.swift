// FakeInferenceBackend — deterministic `InferenceBackend` for the bridge
// contract suite. Lets us assert prompt content, output parsing, budget
// enforcement, eviction, and cancellation behavior without a real model.

import Foundation
@testable import TinyBrainCartographerBridge

final class FakeInferenceBackend: InferenceBackend, @unchecked Sendable {

    // MARK: - Hooks

    /// Produces the raw model output for a prompt. Called serially from
    /// the owning actor.
    var handler: @Sendable (String, Int) async throws -> String

    /// Captures every prompt/maxTokens pair the adapter sent.
    private(set) var calls: [(prompt: String, maxOutputTokens: Int)] = []

    /// Number of times `prewarm` was called.
    private(set) var prewarmCount: Int = 0

    /// Override estimated token count; defaults to whitespace-split word count.
    var tokenCounter: @Sendable (String) -> Int = { text in
        text.split(whereSeparator: { $0.isWhitespace }).count
    }

    // MARK: - Init

    init(handler: @Sendable @escaping (String, Int) async throws -> String = { _, _ in "[]" }) {
        self.handler = handler
    }

    // MARK: - InferenceBackend

    func estimatedTokenCount(_ text: String) -> Int {
        tokenCounter(text)
    }

    func prewarm() async throws {
        prewarmCount += 1
    }

    func generate(prompt: String, maxOutputTokens: Int) async throws -> String {
        calls.append((prompt, maxOutputTokens))
        return try await handler(prompt, maxOutputTokens)
    }
}
