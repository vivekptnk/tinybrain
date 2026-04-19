// TinyBrainSmartService — actor conforming to Cartographer's
// SmartAnnotationService protocol, backed by a TinyBrain language model.
//
// Contract: docs/INTEGRATION-TINYBRAIN.md in the Cartographer repo.
// Summary:
//   §2 lifecycle   — cheap init, lazy weight load, resident for process life,
//                    evict on memory pressure.
//   §3 threading   — actor-isolated; inference off the main actor; honor Task
//                    cancellation within one token iteration.
//   §4 budgets     — ≤ 1,536 input / ≤ 256 search-output / ≤ 384 summarize-output
//                    tokens; overflow throws `.invalidQuery`.
//   §5 fallback    — evicted calls throw `.modelUnavailable`; Cartographer's
//                    wrapper catches and falls through to the mock.

import Cartographer
import Foundation

/// On-device, TinyBrain-backed implementation of
/// `Cartographer.SmartAnnotationService`.
///
/// ```swift
/// let url = Bundle.main.url(forResource: "tinyllama-int4", withExtension: "tbf")!
/// let smart = try TinyBrainSmartService(modelURL: url)
///
/// let ids = try await smart.search(query: "coffee", in: corpus)
/// let summary = try await smart.summarize(annotations: corpus)
/// ```
///
/// Construct once per process. Holding a reference keeps the model resident.
/// Memory-pressure eviction is sticky — drop this instance and build a fresh
/// one once the host app is ready to reload.
public actor TinyBrainSmartService: SmartAnnotationService {

    // MARK: - Private state

    private let backend: any InferenceBackend
    private let configuration: Configuration
    private var evicted: Bool = false
    private var pressureMonitor: MemoryPressureMonitor?

    // MARK: - Init (production)

    /// Construct from a `.tbf` model URL. The file is opened to verify the
    /// magic header but **weights are not loaded**; the first `search` or
    /// `summarize` call triggers loading.
    ///
    /// - Throws: `SmartAnnotationServiceError.modelUnavailable` if the URL
    ///   is missing, unreadable, or not a valid TBF file.
    public init(
        modelURL: URL,
        configuration: Configuration = .default
    ) throws {
        try TinyBrainSmartService.validateTBFHeader(at: modelURL)
        self.backend = TinyBrainInferenceBackend(modelURL: modelURL)
        self.configuration = configuration
        if configuration.enableMemoryPressureEviction {
            self.pressureMonitor = MemoryPressureMonitor()
        }
    }

    // MARK: - Init (custom backend)

    /// Construct with a caller-supplied backend. Used by tests and by hosts
    /// that want to wire a non-TBF inference provider through the same
    /// prompt-shaping / budget-enforcement plumbing.
    public init(
        backend: some InferenceBackend,
        configuration: Configuration = .default
    ) {
        self.backend = backend
        self.configuration = configuration
        if configuration.enableMemoryPressureEviction {
            self.pressureMonitor = MemoryPressureMonitor()
        }
    }

    // MARK: - Public API

    /// Approximate token count for `text`, delegating to the backing
    /// tokenizer. Cartographer uses this while chunking a corpus into
    /// prompt-sized pieces; see INTEGRATION-TINYBRAIN.md §4.2.
    public nonisolated func estimatedTokenCount(_ text: String) -> Int {
        backend.estimatedTokenCount(text)
    }

    // MARK: - SmartAnnotationService conformance

    public func search(
        query: String,
        in corpus: [Annotation]
    ) async throws -> [EntityID] {
        try checkLive()

        let trimmed = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            throw SmartAnnotationServiceError.invalidQuery("empty query")
        }
        guard !corpus.isEmpty else { return [] }

        let prompt = PromptBuilder.search(query: trimmed, corpus: corpus)
        try enforceInputBudget(prompt, budget: configuration.searchInputBudget)

        try Task.checkCancellation()
        try await prewarmOrEvict()

        let raw: String
        do {
            raw = try await backend.generate(
                prompt: prompt,
                maxOutputTokens: configuration.searchOutputBudget
            )
        } catch is CancellationError {
            throw CancellationError()
        } catch {
            throw SmartAnnotationServiceError.inferenceFailed("\(error)")
        }

        return ResponseParser.parseSearchIndices(raw, corpus: corpus)
    }

    public func summarize(
        annotations: [Annotation]
    ) async throws -> String {
        try checkLive()

        guard !annotations.isEmpty else { return "" }

        let prompt = PromptBuilder.summarize(annotations: annotations)
        try enforceInputBudget(prompt, budget: configuration.summarizeInputBudget)

        try Task.checkCancellation()
        try await prewarmOrEvict()

        let raw: String
        do {
            raw = try await backend.generate(
                prompt: prompt,
                maxOutputTokens: configuration.summarizeOutputBudget
            )
        } catch is CancellationError {
            throw CancellationError()
        } catch {
            throw SmartAnnotationServiceError.inferenceFailed("\(error)")
        }

        return raw.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    // MARK: - Eviction (test hook + pressure drain)

    /// Force the service into the evicted state. Subsequent calls throw
    /// `SmartAnnotationServiceError.modelUnavailable`. Mirrors what the
    /// memory-pressure handler does on critical pressure; exposed so
    /// hosts (and tests) can simulate the condition deterministically.
    public func evictForMemoryPressure() {
        evicted = true
        pressureMonitor = nil
    }

    /// Drain the pressure monitor queue during this heartbeat. Tests call
    /// this to process any queued critical events before asserting state.
    internal func drainPressureEvents() async {
        guard let monitor = pressureMonitor, monitor.didFire() else { return }
        evictForMemoryPressure()
    }

    // MARK: - Helpers

    private func checkLive() throws {
        if evicted {
            throw SmartAnnotationServiceError.modelUnavailable("memory pressure eviction")
        }
    }

    private func enforceInputBudget(
        _ prompt: String,
        budget: Int
    ) throws {
        let count = backend.estimatedTokenCount(prompt)
        if count > budget {
            throw SmartAnnotationServiceError.invalidQuery(
                "prompt exceeds input budget: \(count) > \(budget) tokens"
            )
        }
    }

    private func prewarmOrEvict() async throws {
        do {
            try await backend.prewarm()
        } catch is CancellationError {
            throw CancellationError()
        } catch {
            evicted = true
            throw SmartAnnotationServiceError.modelUnavailable("\(error)")
        }
    }

    // MARK: - TBF header validation

    private static func validateTBFHeader(at url: URL) throws {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw SmartAnnotationServiceError.modelUnavailable(
                "model file missing at \(url.path)"
            )
        }
        let handle: FileHandle
        do {
            handle = try FileHandle(forReadingFrom: url)
        } catch {
            throw SmartAnnotationServiceError.modelUnavailable(
                "could not open model file: \(error)"
            )
        }
        defer { try? handle.close() }

        let magic: Data
        do {
            magic = try handle.read(upToCount: 4) ?? Data()
        } catch {
            throw SmartAnnotationServiceError.modelUnavailable(
                "could not read model header: \(error)"
            )
        }
        guard magic.count == 4, String(data: magic, encoding: .utf8) == "TBFM" else {
            throw SmartAnnotationServiceError.modelUnavailable(
                "not a TBF file (magic mismatch)"
            )
        }
    }
}
