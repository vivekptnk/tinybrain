/// SwiftUI bridge for TinyBrain agent trace observer callbacks.
///
/// The observer methods are intentionally nonisolated because `AgentLoop` calls
/// them from its actor executor. Events are reduced off-main, then published by
/// hopping back to `MainActor`, matching the X-Ray view model pattern.

import Combine
import Foundation
import TinyBrainAgent
import TinyBrainRuntime

/// Current status of the bundled demo corpus.
public enum AgentCorpusStatus: Equatable, Sendable {
    case idle(noteCount: Int)
    case indexing(noteCount: Int)
    case ready(noteCount: Int, chunkCount: Int, embedder: String)
    case failed(String)

    /// Whether retrieval can run.
    public var isReady: Bool {
        if case .ready = self { return true }
        return false
    }
}

/// Observable trace state for the Agent Trace panel.
@MainActor
public final class AgentTraceViewModel: ObservableObject {
    /// Whether an agent run is active.
    @Published public private(set) var isRunning: Bool = false

    /// Visible timeline steps.
    @Published public private(set) var steps: [AgentVisibleStep] = []

    /// Active step index.
    @Published public private(set) var activeStepIndex: Int?

    /// Final answer text.
    @Published public private(set) var finalAnswer: String?

    /// Human-readable termination reason.
    @Published public private(set) var terminationReason: String?

    /// User-facing error or cancellation message.
    @Published public private(set) var errorMessage: String?

    /// Aggregate run metrics.
    @Published public private(set) var runMetrics: AgentRunMetrics = AgentRunMetrics()

    /// Corpus status shown by the workbench.
    @Published public private(set) var corpusStatus: AgentCorpusStatus = .idle(noteCount: AgentDemoCorpus.notes.count)

    /// Whether the step budget warning band is visible.
    @Published public private(set) var isBudgetExhausted: Bool = false

    /// Configured maximum step count.
    @Published public private(set) var maxSteps: Int = 3

    /// Run start time.
    @Published public private(set) var startedAt: Date?

    /// Run end time.
    @Published public private(set) var endedAt: Date?

    private let accumulator = TraceAccumulator()

    /// Creates an empty trace view model.
    public init() {}

    /// Prepares the trace for a new run.
    public func beginRun(maxSteps: Int = 3) {
        publish(accumulator.beginRun(maxSteps: maxSteps))
    }

    /// Clears all trace state.
    public func reset() {
        publish(accumulator.clear())
    }

    /// Updates corpus status without touching timeline state.
    public func updateCorpusStatus(_ status: AgentCorpusStatus) {
        corpusStatus = status
    }

    /// Applies a local cancellation when the async stream is terminated before
    /// the agent observer can publish its cancellation event.
    public func cancelLocally(reason: String) {
        publish(accumulator.apply(.cancelled(reason: reason)))
    }

    /// Applies a local stream/runtime failure.
    public func failRun(message: String) {
        publish(accumulator.apply(.failed(message: message)))
    }

    private func publish(_ snapshot: AgentTraceSnapshot) {
        isRunning = snapshot.isRunning
        steps = snapshot.steps
        activeStepIndex = snapshot.activeStepIndex
        finalAnswer = snapshot.finalAnswer
        terminationReason = snapshot.terminationReason
        errorMessage = snapshot.errorMessage
        runMetrics = snapshot.runMetrics
        isBudgetExhausted = snapshot.isBudgetExhausted
        maxSteps = snapshot.maxSteps
        startedAt = snapshot.startedAt
        endedAt = snapshot.endedAt
    }

    nonisolated private func publishFromObserver(_ snapshot: AgentTraceSnapshot) {
        Task { @MainActor [weak self] in
            self?.publish(snapshot)
        }
    }
}

// MARK: - AgentTraceObserver

extension AgentTraceViewModel: AgentTraceObserver {
    nonisolated public func stepStarted(_ event: AgentStepStarted) {
        publishFromObserver(accumulator.apply(.stepStarted(
            index: event.index,
            promptTokens: event.promptTokens
        )))
    }

    nonisolated public func toolCallProposed(_ event: AgentToolCallProposed) {
        let argumentsJSON = Self.prettyJSONString(event.call.arguments)
        publishFromObserver(accumulator.apply(.toolCallProposed(
            index: event.index,
            toolName: event.call.name,
            argumentsJSON: argumentsJSON,
            query: event.call.arguments["query"] as? String,
            k: Self.intArgument(event.call.arguments["k"]),
            rawOutput: event.rawOutput
        )))
    }

    nonisolated public func toolExecuted(_ event: AgentToolExecuted) {
        publishFromObserver(accumulator.apply(.toolExecuted(
            index: event.index,
            toolName: event.call?.name,
            resultContent: event.result.content,
            isError: event.result.isError,
            elapsedMs: event.elapsedMs,
            resultTokens: event.resultTokens
        )))
    }

    nonisolated public func finalAnswer(_ event: AgentFinalAnswer) {
        publishFromObserver(accumulator.apply(.finalAnswer(
            answer: event.answer,
            terminationReason: Self.readableTermination(event.transcript.terminationReason),
            completedStepCount: event.transcript.steps.count
        )))
    }

    nonisolated public func budgetExhausted(_ event: AgentBudgetExhausted) {
        publishFromObserver(accumulator.apply(.budgetExhausted(maxSteps: event.maxSteps)))
    }

    nonisolated public func cancelled(_ event: AgentCancelled) {
        publishFromObserver(accumulator.apply(.cancelled(reason: event.reason)))
    }

    private nonisolated static func prettyJSONString(_ object: [String: Any]) -> String {
        guard JSONSerialization.isValidJSONObject(object),
              let data = try? JSONSerialization.data(
                withJSONObject: object,
                options: [.prettyPrinted, .sortedKeys]
              ),
              let string = String(data: data, encoding: .utf8) else {
            return "{}"
        }
        return string
    }

    private nonisolated static func intArgument(_ value: Any?) -> Int? {
        switch value {
        case let int as Int:
            return int
        case let number as NSNumber:
            return number.intValue
        case let string as String:
            return Int(string)
        default:
            return nil
        }
    }

    private nonisolated static func readableTermination(_ reason: AgentTerminationReason?) -> String {
        switch reason {
        case .finalAnswer:
            return "final answer"
        case .budgetExhausted:
            return "budget exhausted"
        case .cancelled:
            return "cancelled"
        case nil:
            return "complete"
        }
    }
}

// MARK: - Accumulator

private final class TraceAccumulator: @unchecked Sendable {
    private let lock = NSLock()
    private var reducer = AgentTraceReducer()

    func beginRun(maxSteps: Int) -> AgentTraceSnapshot {
        lock.lock()
        defer { lock.unlock() }
        return reducer.reset(maxSteps: maxSteps)
    }

    func clear() -> AgentTraceSnapshot {
        lock.lock()
        defer { lock.unlock() }
        return reducer.clear()
    }

    func apply(_ event: AgentTraceReducerEvent) -> AgentTraceSnapshot {
        lock.lock()
        defer { lock.unlock() }
        return reducer.reduce(event)
    }
}
