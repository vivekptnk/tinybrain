// InferenceBackend — the seam between TinyBrainSmartService and the
// underlying language-model runtime.
//
// The adapter actor is deliberately runtime-agnostic: it shapes prompts,
// enforces token budgets, and parses outputs. Everything model-specific —
// loading weights, running the forward pass, decoding tokens — lives behind
// this protocol. Tests inject a deterministic backend; production uses
// `TinyBrainInferenceBackend`.

import Foundation

/// A pluggable inference surface.
///
/// Implementations must be safe to call from an actor context (the adapter
/// awaits its methods on its own isolation) and must honor `Task` cancellation
/// within one token-generation iteration — per INTEGRATION-TINYBRAIN.md §3.
public protocol InferenceBackend: Sendable {

    /// Approximate number of tokens in `text`. Called frequently during
    /// budget checks; implementations should be cheap.
    func estimatedTokenCount(_ text: String) -> Int

    /// Load heavy resources (weights, tokenizer tables). Called at most once
    /// before the first `generate` call. Subsequent calls are no-ops.
    func prewarm() async throws

    /// Run a completion. Must stop when `maxOutputTokens` is reached or
    /// `Task.isCancelled` becomes true, whichever comes first. On
    /// cancellation, throw `CancellationError`.
    ///
    /// - Parameters:
    ///   - prompt: The full prompt (system + user concatenation is the
    ///     caller's responsibility).
    ///   - maxOutputTokens: Upper bound on generated tokens.
    /// - Returns: The decoded string (excluding the prompt).
    func generate(prompt: String, maxOutputTokens: Int) async throws -> String
}
