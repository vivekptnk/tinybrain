import Foundation
import TinyBrainRuntime

/// Event emitted when an agent step begins.
public struct AgentStepStarted: Equatable {
    /// Zero-based step index.
    public let index: Int

    /// Approximate prompt-token count.
    public let promptTokens: Int

    /// Creates a step-start event.
    public init(index: Int, promptTokens: Int) {
        self.index = index
        self.promptTokens = promptTokens
    }
}

/// Event emitted after the model proposes a tool call.
public struct AgentToolCallProposed: Equatable {
    /// Zero-based step index.
    public let index: Int

    /// Parsed tool call.
    public let call: ToolCall

    /// Raw model output that contained the call.
    public let rawOutput: String

    /// Creates a proposed-tool-call event.
    public init(index: Int, call: ToolCall, rawOutput: String) {
        self.index = index
        self.call = call
        self.rawOutput = rawOutput
    }
}

/// Event emitted after a tool finishes.
public struct AgentToolExecuted: Equatable {
    /// Zero-based step index.
    public let index: Int

    /// Tool call that was dispatched, when parsing succeeded.
    public let call: ToolCall?

    /// Tool result fed back into the transcript.
    public let result: ToolResult

    /// Tool execution latency in milliseconds.
    public let elapsedMs: Double

    /// Approximate result-token count.
    public let resultTokens: Int

    /// Creates a tool-executed event.
    public init(
        index: Int,
        call: ToolCall?,
        result: ToolResult,
        elapsedMs: Double,
        resultTokens: Int
    ) {
        self.index = index
        self.call = call
        self.result = result
        self.elapsedMs = elapsedMs
        self.resultTokens = resultTokens
    }
}

/// Event emitted when the model produces a final answer.
public struct AgentFinalAnswer: Equatable {
    /// Final answer text.
    public let answer: String

    /// Completed transcript at termination.
    public let transcript: AgentTranscript

    /// Creates a final-answer event.
    public init(answer: String, transcript: AgentTranscript) {
        self.answer = answer
        self.transcript = transcript
    }
}

/// Event emitted when the step budget is reached.
public struct AgentBudgetExhausted: Equatable {
    /// Configured step budget.
    public let maxSteps: Int

    /// Transcript at the point the budget was exhausted.
    public let transcript: AgentTranscript

    /// Creates a budget-exhausted event.
    public init(maxSteps: Int, transcript: AgentTranscript) {
        self.maxSteps = maxSteps
        self.transcript = transcript
    }
}

/// Event emitted when a run is cancelled.
public struct AgentCancelled: Equatable {
    /// Human-readable cancellation reason.
    public let reason: String

    /// Transcript at cancellation.
    public let transcript: AgentTranscript

    /// Creates a cancellation event.
    public init(reason: String, transcript: AgentTranscript) {
        self.reason = reason
        self.transcript = transcript
    }
}

/// Streamed event surface for agent runs.
public enum AgentEvent: Equatable {
    /// A plan-act-observe step started.
    case stepStarted(AgentStepStarted)
    /// The model proposed a tool call.
    case toolCallProposed(AgentToolCallProposed)
    /// A tool returned a success or error result.
    case toolExecuted(AgentToolExecuted)
    /// The model produced the final answer.
    case finalAnswer(AgentFinalAnswer)
    /// The configured tool-step budget was exhausted.
    case budgetExhausted(AgentBudgetExhausted)
    /// The run was cancelled.
    case cancelled(AgentCancelled)
}

/// X-Ray observer for the agent-level plan-act-observe timeline.
///
/// Implementations should keep these callbacks cheap; ``AgentLoop`` checks for
/// a nil observer before building UI-only state.
public protocol AgentTraceObserver: AnyObject {
    /// Called when a planning step begins.
    func stepStarted(_ event: AgentStepStarted)

    /// Called after the model proposes a tool call.
    func toolCallProposed(_ event: AgentToolCallProposed)

    /// Called after a tool returns a result.
    func toolExecuted(_ event: AgentToolExecuted)

    /// Called when the loop terminates with a final answer.
    func finalAnswer(_ event: AgentFinalAnswer)

    /// Called when the loop reaches its hard step budget.
    func budgetExhausted(_ event: AgentBudgetExhausted)

    /// Called when the loop is cancelled.
    func cancelled(_ event: AgentCancelled)
}
