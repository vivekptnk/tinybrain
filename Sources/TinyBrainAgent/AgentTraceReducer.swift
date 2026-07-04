/// Pure reducer for the visible TinyBrain Agent trace.
///
/// This file has no SwiftUI dependencies so the plan-act-observe state mapping
/// can be tested without rendering the demo app.

import Foundation

/// High-level state shown on an agent step badge.
public enum AgentStepVisualState: String, Equatable, Sendable {
    case planning
    case calling
    case observed
    case error
    case done
    case cancelled

    /// Uppercase badge text.
    public var badge: String { rawValue.uppercased() }
}

/// A parsed passage row extracted from the P1 retrieve tool output.
public struct AgentRetrievedPassage: Identifiable, Equatable, Sendable {
    /// Zero-based retrieval rank.
    public let rank: Int

    /// Source path from the retrieve output.
    public let source: String

    /// Lower-is-better vector distance.
    public let distance: Double

    /// Passage excerpt.
    public let excerpt: String

    /// Stable row identifier.
    public var id: String { "\(rank)-\(source)-\(String(format: "%.3f", distance))" }

    /// Creates a visible passage row.
    public init(rank: Int, source: String, distance: Double, excerpt: String) {
        self.rank = rank
        self.source = source
        self.distance = distance
        self.excerpt = excerpt
    }
}

/// One visible tile in the agent timeline.
public struct AgentVisibleStep: Identifiable, Equatable, Sendable {
    /// Zero-based step index.
    public let index: Int

    /// Stable tile identifier.
    public var id: Int { index }

    /// Current visual state.
    public var state: AgentStepVisualState

    /// Approximate prompt-token count for this step.
    public var promptTokens: Int

    /// Generated-token count when known from a completed transcript.
    public var generatedTokens: Int?

    /// Result-token count returned by a tool.
    public var resultTokens: Int?

    /// Tool execution latency.
    public var elapsedMs: Double?

    /// Time the step tile was created.
    public var startedAt: Date

    /// Tool name proposed by the model.
    public var toolName: String?

    /// Pretty JSON arguments for the proposed tool call.
    public var argumentsJSON: String?

    /// First-class retrieve query label.
    public var liftedQuery: String?

    /// First-class retrieve k label.
    public var liftedK: Int?

    /// Raw model output that produced the tool call.
    public var rawOutput: String?

    /// Tool result content.
    public var resultContent: String?

    /// Whether the tool result was an error.
    public var isError: Bool

    /// Parsed retrieve passages.
    public var passages: [AgentRetrievedPassage]

    /// Creates a planning step.
    public init(index: Int, promptTokens: Int, startedAt: Date) {
        self.index = index
        self.state = .planning
        self.promptTokens = promptTokens
        self.startedAt = startedAt
        self.isError = false
        self.passages = []
    }
}

/// Compact aggregate metrics for the trace header.
public struct AgentRunMetrics: Equatable, Sendable {
    /// Number of visible steps.
    public var steps: Int = 0

    /// Total prompt tokens across visible steps.
    public var promptTokens: Int = 0

    /// Total tool latency.
    public var toolElapsedMs: Double = 0

    /// Total tool-result tokens.
    public var resultTokens: Int = 0

    /// Creates aggregate run metrics.
    public init(
        steps: Int = 0,
        promptTokens: Int = 0,
        toolElapsedMs: Double = 0,
        resultTokens: Int = 0
    ) {
        self.steps = steps
        self.promptTokens = promptTokens
        self.toolElapsedMs = toolElapsedMs
        self.resultTokens = resultTokens
    }
}

/// Full reducer output published to SwiftUI.
public struct AgentTraceSnapshot: Equatable, Sendable {
    /// Whether a run is active.
    public var isRunning: Bool = false

    /// Visible timeline steps.
    public var steps: [AgentVisibleStep] = []

    /// Active step index, when any.
    public var activeStepIndex: Int?

    /// Final answer text, if produced.
    public var finalAnswer: String?

    /// Human-readable termination reason.
    public var terminationReason: String?

    /// User-facing error or cancellation message.
    public var errorMessage: String?

    /// Aggregate trace metrics.
    public var runMetrics: AgentRunMetrics = AgentRunMetrics()

    /// Configured maximum plan-act-observe steps.
    public var maxSteps: Int = 3

    /// Whether the budget-exhausted warning band should be shown.
    public var isBudgetExhausted: Bool = false

    /// Run start time.
    public var startedAt: Date?

    /// Run end time.
    public var endedAt: Date?

    /// Creates a trace snapshot.
    public init(
        isRunning: Bool = false,
        steps: [AgentVisibleStep] = [],
        activeStepIndex: Int? = nil,
        finalAnswer: String? = nil,
        terminationReason: String? = nil,
        errorMessage: String? = nil,
        runMetrics: AgentRunMetrics = AgentRunMetrics(),
        maxSteps: Int = 3,
        isBudgetExhausted: Bool = false,
        startedAt: Date? = nil,
        endedAt: Date? = nil
    ) {
        self.isRunning = isRunning
        self.steps = steps
        self.activeStepIndex = activeStepIndex
        self.finalAnswer = finalAnswer
        self.terminationReason = terminationReason
        self.errorMessage = errorMessage
        self.runMetrics = runMetrics
        self.maxSteps = maxSteps
        self.isBudgetExhausted = isBudgetExhausted
        self.startedAt = startedAt
        self.endedAt = endedAt
    }
}

/// Minimal reducer input derived from agent observer callbacks.
public enum AgentTraceReducerEvent: Equatable, Sendable {
    case stepStarted(index: Int, promptTokens: Int)
    case toolCallProposed(
        index: Int,
        toolName: String,
        argumentsJSON: String,
        query: String?,
        k: Int?,
        rawOutput: String
    )
    case toolExecuted(
        index: Int,
        toolName: String?,
        resultContent: String,
        isError: Bool,
        elapsedMs: Double,
        resultTokens: Int
    )
    case budgetExhausted(maxSteps: Int)
    case finalAnswer(answer: String, terminationReason: String, completedStepCount: Int)
    case cancelled(reason: String)
    case failed(message: String)
}

/// Deterministic reducer that maps agent events to visible trace state.
public struct AgentTraceReducer: Equatable, Sendable {
    /// Current reducer snapshot.
    public private(set) var snapshot: AgentTraceSnapshot

    /// Creates a reducer with an optional initial snapshot.
    public init(snapshot: AgentTraceSnapshot = AgentTraceSnapshot()) {
        self.snapshot = snapshot
    }

    /// Resets state for a new run.
    public mutating func reset(maxSteps: Int = 3, now: Date = Date()) -> AgentTraceSnapshot {
        snapshot = AgentTraceSnapshot(
            isRunning: true,
            maxSteps: maxSteps,
            startedAt: now
        )
        return snapshot
    }

    /// Clears all visible state.
    public mutating func clear() -> AgentTraceSnapshot {
        snapshot = AgentTraceSnapshot(maxSteps: snapshot.maxSteps)
        return snapshot
    }

    /// Applies one observer-derived event.
    @discardableResult
    public mutating func reduce(
        _ event: AgentTraceReducerEvent,
        now: Date = Date()
    ) -> AgentTraceSnapshot {
        switch event {
        case .stepStarted(let index, let promptTokens):
            upsertStep(index: index, now: now) { step in
                step.state = .planning
                step.promptTokens = promptTokens
                step.startedAt = now
            } create: {
                AgentVisibleStep(index: index, promptTokens: promptTokens, startedAt: now)
            }
            snapshot.isRunning = true
            snapshot.activeStepIndex = index
            if snapshot.startedAt == nil {
                snapshot.startedAt = now
            }

        case .toolCallProposed(let index, let toolName, let argumentsJSON, let query, let k, let rawOutput):
            upsertStep(index: index, now: now) { step in
                step.state = .calling
                step.toolName = toolName
                step.argumentsJSON = argumentsJSON
                step.liftedQuery = query
                step.liftedK = k
                step.rawOutput = rawOutput
            } create: {
                var step = AgentVisibleStep(index: index, promptTokens: 0, startedAt: now)
                step.state = .calling
                step.toolName = toolName
                step.argumentsJSON = argumentsJSON
                step.liftedQuery = query
                step.liftedK = k
                step.rawOutput = rawOutput
                return step
            }
            snapshot.activeStepIndex = index

        case .toolExecuted(let index, let toolName, let resultContent, let isError, let elapsedMs, let resultTokens):
            upsertStep(index: index, now: now) { step in
                step.state = isError ? .error : .observed
                step.toolName = step.toolName ?? toolName
                step.resultContent = resultContent
                step.isError = isError
                step.elapsedMs = elapsedMs
                step.resultTokens = resultTokens
                step.passages = Self.parseRetrievedPassages(from: resultContent)
            } create: {
                var step = AgentVisibleStep(index: index, promptTokens: 0, startedAt: now)
                step.state = isError ? .error : .observed
                step.toolName = toolName
                step.resultContent = resultContent
                step.isError = isError
                step.elapsedMs = elapsedMs
                step.resultTokens = resultTokens
                step.passages = Self.parseRetrievedPassages(from: resultContent)
                return step
            }
            snapshot.activeStepIndex = index

        case .budgetExhausted(let maxSteps):
            snapshot.isBudgetExhausted = true
            snapshot.maxSteps = maxSteps
            snapshot.terminationReason = "budget exhausted"
            snapshot.isRunning = true

        case .finalAnswer(let answer, let terminationReason, let completedStepCount):
            snapshot.finalAnswer = answer
            snapshot.terminationReason = terminationReason
            snapshot.isRunning = false
            snapshot.endedAt = now
            markOpenStepsDone(completedStepCount: completedStepCount)

        case .cancelled(let reason):
            snapshot.errorMessage = reason
            snapshot.terminationReason = "cancelled"
            snapshot.isRunning = false
            snapshot.endedAt = now
            if let active = snapshot.activeStepIndex {
                updateStep(at: active) { step in
                    step.state = .cancelled
                }
            }

        case .failed(let message):
            snapshot.errorMessage = message
            snapshot.terminationReason = "error"
            snapshot.isRunning = false
            snapshot.endedAt = now
            if let active = snapshot.activeStepIndex {
                updateStep(at: active) { step in
                    step.state = .error
                    step.isError = true
                    step.resultContent = message
                }
            }
        }

        snapshot.runMetrics = Self.metrics(for: snapshot.steps)
        return snapshot
    }

    /// Parses P1 retrieve output lines:
    /// `[1] excerpt (source: path, distance: 0.123)`.
    public static func parseRetrievedPassages(from content: String) -> [AgentRetrievedPassage] {
        let pattern = #"^\[(\d+)\]\s+(.+)\s+\(source:\s*(.+),\s*distance:\s*([0-9.+\-eE]+)\)$"#
        guard let regex = try? NSRegularExpression(pattern: pattern) else {
            return []
        }

        return content.split(whereSeparator: \.isNewline).compactMap { line in
            let string = String(line)
            let range = NSRange(string.startIndex..<string.endIndex, in: string)
            guard let match = regex.firstMatch(in: string, range: range),
                  match.numberOfRanges == 5,
                  let rankRange = Range(match.range(at: 1), in: string),
                  let excerptRange = Range(match.range(at: 2), in: string),
                  let sourceRange = Range(match.range(at: 3), in: string),
                  let distanceRange = Range(match.range(at: 4), in: string),
                  let printedRank = Int(string[rankRange]),
                  let distance = Double(string[distanceRange]) else {
                return nil
            }

            return AgentRetrievedPassage(
                rank: max(0, printedRank - 1),
                source: String(string[sourceRange]),
                distance: distance,
                excerpt: String(string[excerptRange])
            )
        }
    }

    private mutating func upsertStep(
        index: Int,
        now: Date,
        update: (inout AgentVisibleStep) -> Void,
        create: () -> AgentVisibleStep
    ) {
        if let existing = snapshot.steps.firstIndex(where: { $0.index == index }) {
            update(&snapshot.steps[existing])
        } else {
            var step = create()
            update(&step)
            snapshot.steps.append(step)
            snapshot.steps.sort { $0.index < $1.index }
        }
    }

    private mutating func updateStep(at index: Int, update: (inout AgentVisibleStep) -> Void) {
        guard let existing = snapshot.steps.firstIndex(where: { $0.index == index }) else {
            return
        }
        update(&snapshot.steps[existing])
    }

    private mutating func markOpenStepsDone(completedStepCount: Int) {
        for offset in snapshot.steps.indices {
            switch snapshot.steps[offset].state {
            case .planning, .calling, .observed:
                snapshot.steps[offset].state = .done
            case .error, .done, .cancelled:
                break
            }
        }

        for transcriptStep in snapshot.steps.indices where snapshot.steps[transcriptStep].index < completedStepCount {
            if snapshot.steps[transcriptStep].state == .observed {
                snapshot.steps[transcriptStep].state = .done
            }
        }
    }

    private static func metrics(for steps: [AgentVisibleStep]) -> AgentRunMetrics {
        AgentRunMetrics(
            steps: steps.count,
            promptTokens: steps.reduce(0) { $0 + $1.promptTokens },
            toolElapsedMs: steps.reduce(0) { $0 + ($1.elapsedMs ?? 0) },
            resultTokens: steps.reduce(0) { $0 + ($1.resultTokens ?? 0) }
        )
    }
}
