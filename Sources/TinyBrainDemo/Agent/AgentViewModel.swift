/// View model for the TinyBrain Agent workbench.
///
/// Owns the agent runtime, the bundled corpus lifecycle, and the live trace
/// observer. The agent uses a separate `ModelRunner` from chat generation.

import Combine
import Foundation
import TinyBrainRuntime
import TinyBrainTokenizer

/// Orchestrates the in-app Agent demo over the bundled corpus.
@MainActor
public final class AgentViewModel: ObservableObject {
    /// Prompt currently in the agent composer.
    @Published public var promptText: String = ""

    /// Whether the corpus/runtime is being prepared.
    @Published public private(set) var isPreparing: Bool = false

    /// Whether an agent run is active.
    @Published public private(set) var isRunning: Bool = false

    /// Truthful reason the agent is unavailable.
    @Published public private(set) var disabledReason: String?

    /// Last runtime error, if any.
    @Published public private(set) var errorMessage: String?

    /// Live trace view model, strongly retained because `AgentLoop` stores the
    /// observer weakly.
    public let trace: AgentTraceViewModel

    private var activeWeights: ModelWeights?
    private var tokenizer: (any Tokenizer)?
    private var promptStyle: ModelPromptStyle
    private var sampler: SamplerConfig
    private var activeModelName: String
    private var runtime: AgentRuntimeContext?
    private var runTask: Task<Void, Never>?

    /// Creates an agent workbench view model for the active chat model.
    public init(
        activeWeights: ModelWeights?,
        tokenizer: (any Tokenizer)?,
        promptStyle: ModelPromptStyle,
        sampler: SamplerConfig,
        activeModelName: String,
        isToyModel: Bool,
        trace: AgentTraceViewModel? = nil
    ) {
        self.activeWeights = activeWeights
        self.tokenizer = tokenizer
        self.promptStyle = promptStyle
        self.sampler = sampler
        self.activeModelName = activeModelName
        self.trace = trace ?? AgentTraceViewModel()
        applyAvailability(isToyModel: isToyModel)
    }

    /// Rebuilds model-dependent agent state after a chat model switch.
    public func reconfigure(
        weights: ModelWeights?,
        tokenizer: (any Tokenizer)?,
        promptStyle: ModelPromptStyle,
        sampler: SamplerConfig,
        activeModelName: String,
        isToyModel: Bool
    ) {
        cancel()
        self.activeWeights = weights
        self.tokenizer = tokenizer
        self.promptStyle = promptStyle
        self.sampler = sampler
        self.activeModelName = activeModelName
        self.runtime = nil
        self.errorMessage = nil
        trace.reset()
        trace.updateCorpusStatus(.idle(noteCount: AgentDemoCorpus.notes.count))
        applyAvailability(isToyModel: isToyModel)
    }

    /// Updates the sampler used for the next runtime build.
    public func updateSampler(_ sampler: SamplerConfig) {
        self.sampler = sampler
        runtime = nil
    }

    /// Lazily indexes the bundled corpus and constructs the agent loop.
    public func prepareIfNeeded() async {
        guard disabledReason == nil else { return }
        guard runtime == nil, !isPreparing else { return }
        guard let activeWeights, let tokenizer else {
            disabledReason = "Agent runs require loaded model weights and a tokenizer."
            return
        }

        isPreparing = true
        errorMessage = nil
        trace.updateCorpusStatus(.indexing(noteCount: AgentDemoCorpus.notes.count))

        do {
            let built = try await AgentRuntimeFactory.makeRuntime(
                weights: activeWeights,
                tokenizer: tokenizer,
                promptStyle: promptStyle,
                sampler: sampler,
                observer: trace
            )
            runtime = built
            trace.updateCorpusStatus(.ready(
                noteCount: built.corpus.noteCount,
                chunkCount: built.corpus.chunkCount,
                embedder: built.corpus.embedder,
                indexPreparation: built.corpus.indexPreparation
            ))
        } catch {
            let message = "Agent index failed: \(error.localizedDescription)"
            errorMessage = message
            trace.updateCorpusStatus(.failed(message))
        }

        isPreparing = false
    }

    /// Starts an agent run over the bundled corpus.
    public func startRun() {
        let prompt = promptText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !prompt.isEmpty, !isRunning else { return }
        guard disabledReason == nil else { return }

        runTask = Task { [weak self] in
            await self?.run(prompt)
        }
    }

    /// Cancels the active agent run.
    public func cancel() {
        guard isRunning || runTask != nil else { return }
        runTask?.cancel()
        runTask = nil
        isRunning = false
        trace.cancelLocally(reason: "Cancelled by user")
    }

    /// Inserts a suggested prompt.
    public func usePrompt(_ prompt: String) {
        promptText = prompt
    }

    private func run(_ prompt: String) async {
        await prepareIfNeeded()
        guard let runtime, disabledReason == nil else { return }

        isRunning = true
        errorMessage = nil
        trace.beginRun(maxSteps: AgentRuntimeFactory.maxSteps)

        do {
            let stream = runtime.loop.run(prompt)
            for try await _ in stream {
                try Task.checkCancellation()
            }
        } catch is CancellationError {
            trace.cancelLocally(reason: "Cancelled by user")
        } catch {
            let message = "Agent run failed: \(error.localizedDescription)"
            errorMessage = message
            trace.failRun(message: message)
        }

        isRunning = false
        runTask = nil
    }

    private func applyAvailability(isToyModel: Bool) {
        if isToyModel {
            disabledReason = "Agent runs require a chat/instruct model that can emit tool calls."
        } else {
            disabledReason = nil
        }
    }
}
