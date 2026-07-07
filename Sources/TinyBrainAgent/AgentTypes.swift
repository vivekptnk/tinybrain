import Foundation
import TinyBrainRAG
import TinyBrainRuntime

/// Prompt envelope used by ``AgentLoop`` when rendering the agent transcript.
public enum AgentPromptStyle: Equatable {
    /// Qwen ChatML format: `<|im_start|>role ... <|im_end|>`.
    case qwenChatML
    /// TinyLlama/Zephyr chat format used by TinyBrain Chat.
    case zephyrChat
    /// Plain-text transcript rendering for base completion models.
    case rawCompletion
}

/// Runtime controls for ``AgentLoop``.
public struct AgentConfig {
    /// Maximum number of tool-planning steps before the loop forces a final answer.
    public var maxSteps: Int

    /// Initial tool policy supplied to the model.
    ///
    /// Strict or guided constrained decoding only applies when this choice names
    /// a single unambiguous tool: ``ToolChoice/specific(name:)``, or
    /// ``ToolChoice/required`` with exactly one registered tool. ``ToolChoice/auto``
    /// and multi-tool required turns generate unconstrained text that the agent
    /// validates by parsing.
    public var toolChoice: ToolChoice

    /// Constraint strictness for tool-call generations.
    ///
    /// This is a best-effort decoding constraint, not a universal guarantee.
    /// Strict and guided modes are active only for single unambiguous tool calls:
    /// ``ToolChoice/specific(name:)``, or ``ToolChoice/required`` when exactly one
    /// tool is registered. Auto tool choice and required turns with multiple
    /// tools are generated unconstrained and then accepted or rejected by the
    /// tool-call parser.
    public var constraintMode: ConstraintMode

    /// Maximum tokens generated during each planning step.
    public var perStepTokenBudget: Int

    /// Approximate prompt context budget used for transcript compaction.
    public var contextBudget: Int

    /// Sampling policy for model-backed generation.
    public var sampler: SamplerConfig

    /// Stop tokens passed through to model-backed generation.
    public var stopTokens: [Int]

    /// Chat-template format used to render the transcript.
    public var promptStyle: AgentPromptStyle

    /// Creates an agent-loop configuration.
    ///
    /// - Precondition: `maxSteps`, `perStepTokenBudget`, and `contextBudget` must be positive.
    public init(
        maxSteps: Int = 6,
        toolChoice: ToolChoice = .auto,
        constraintMode: ConstraintMode = .strict,
        perStepTokenBudget: Int = 256,
        contextBudget: Int = 2_048,
        sampler: SamplerConfig = SamplerConfig(temperature: 0.7, topK: 20, topP: 0.8),
        stopTokens: [Int] = [],
        promptStyle: AgentPromptStyle = .qwenChatML
    ) {
        precondition(maxSteps > 0, "maxSteps must be positive")
        precondition(perStepTokenBudget > 0, "perStepTokenBudget must be positive")
        precondition(contextBudget > 0, "contextBudget must be positive")
        self.maxSteps = maxSteps
        self.toolChoice = toolChoice
        self.constraintMode = constraintMode
        self.perStepTokenBudget = perStepTokenBudget
        self.contextBudget = contextBudget
        self.sampler = sampler
        self.stopTokens = stopTokens
        self.promptStyle = promptStyle
    }
}

/// Whether a generation turn may propose a tool call or must produce text.
public enum AgentGenerationMode: Equatable {
    /// The model may return a JSON tool call or natural-language final answer.
    case toolOrFinalAnswer
    /// The model must produce natural-language text from the current transcript.
    case finalAnswer
}

/// Input supplied by ``AgentLoop`` to an ``AgentGenerating`` implementation.
public struct AgentGenerationRequest {
    /// Zero-based planning-step index.
    public let stepIndex: Int

    /// Rendered prompt sent to the model.
    public let prompt: String

    /// Generation mode for this turn.
    public let mode: AgentGenerationMode

    /// Maximum tokens to produce.
    public let maxTokens: Int

    /// Sampling policy for this generation.
    public let sampler: SamplerConfig

    /// Stop tokens for this generation.
    public let stopTokens: [Int]

    /// Tool definitions visible to the model.
    public let toolDefinitions: [ToolDefinition]

    /// Tool-choice policy visible to the model.
    public let toolChoice: ToolChoice

    /// Constraint strictness for this generation.
    public let constraintMode: ConstraintMode

    /// Creates a model generation request.
    public init(
        stepIndex: Int,
        prompt: String,
        mode: AgentGenerationMode,
        maxTokens: Int,
        sampler: SamplerConfig,
        stopTokens: [Int],
        toolDefinitions: [ToolDefinition],
        toolChoice: ToolChoice,
        constraintMode: ConstraintMode
    ) {
        self.stepIndex = stepIndex
        self.prompt = prompt
        self.mode = mode
        self.maxTokens = maxTokens
        self.sampler = sampler
        self.stopTokens = stopTokens
        self.toolDefinitions = toolDefinitions
        self.toolChoice = toolChoice
        self.constraintMode = constraintMode
    }
}

/// Text generated for one agent turn.
public struct AgentGenerationResult: Equatable {
    /// Complete decoded model text for the turn.
    public let text: String

    /// Number of generated tokens, when known.
    public let tokenCount: Int

    /// Creates a generated-turn result.
    public init(text: String, tokenCount: Int) {
        self.text = text
        self.tokenCount = tokenCount
    }
}

/// Protocol used by ``AgentLoop`` to generate one model turn.
public protocol AgentGenerating {
    /// Generates one complete turn for a rendered prompt.
    func generate(_ request: AgentGenerationRequest) async throws -> AgentGenerationResult
}

/// Role for a message stored in ``AgentTranscript``.
public enum AgentTranscriptRole: String, Codable, Equatable {
    /// User task or follow-up instruction.
    case user
    /// Assistant model output.
    case assistant
    /// Tool observation fed back to the model.
    case tool
}

/// One message in the agent transcript.
public struct AgentTranscriptMessage: Codable, Equatable {
    /// Message role.
    public let role: AgentTranscriptRole

    /// Message content.
    public let content: String

    /// Creates a transcript message.
    public init(role: AgentTranscriptRole, content: String) {
        self.role = role
        self.content = content
    }
}

/// Serializable snapshot of a runtime ``ToolCall``.
public struct AgentToolCallSnapshot: Codable, Equatable {
    /// Tool-call ID.
    public let id: String

    /// Tool name.
    public let name: String

    /// Canonical JSON object containing the call arguments.
    public let argumentsJSON: String

    /// Creates a snapshot from a runtime tool call.
    public init(call: ToolCall) {
        self.id = call.id
        self.name = call.name
        self.argumentsJSON = AgentJSON.canonicalObjectString(call.arguments)
    }
}

/// Serializable snapshot of a runtime ``ToolResult``.
public struct AgentToolResultSnapshot: Codable, Equatable {
    /// Correlating tool-call ID.
    public let callId: String

    /// Result content returned to the model.
    public let content: String

    /// Whether the result is an error observation.
    public let isError: Bool

    /// Structured retrieved passages carried beside ``content``, when available.
    public let passages: [RetrievedPassageRecord]?

    /// Creates a snapshot from a runtime tool result.
    public init(result: ToolResult, passages: [RetrievedPassageRecord]? = nil) {
        self.callId = result.callId
        self.content = result.content
        self.isError = result.isError
        self.passages = passages
    }
}

/// One completed plan-act-observe step.
public struct AgentTranscriptStep: Codable, Equatable {
    /// Zero-based step index.
    public let index: Int

    /// Raw model output for the planning turn.
    public let modelOutput: String

    /// Tool call proposed by the model, if parsing succeeded.
    public let toolCall: AgentToolCallSnapshot?

    /// Tool result observed by the model, including error results.
    public let toolResult: AgentToolResultSnapshot?

    /// Approximate prompt-token count for this step.
    public let promptTokens: Int

    /// Generated-token count reported by the generator.
    public let generatedTokens: Int

    /// Approximate result-token count fed back to the model.
    public let resultTokens: Int

    /// Tool execution latency in milliseconds.
    public let elapsedMs: Double

    /// Creates a completed step record.
    public init(
        index: Int,
        modelOutput: String,
        toolCall: AgentToolCallSnapshot?,
        toolResult: AgentToolResultSnapshot?,
        promptTokens: Int,
        generatedTokens: Int,
        resultTokens: Int,
        elapsedMs: Double
    ) {
        self.index = index
        self.modelOutput = modelOutput
        self.toolCall = toolCall
        self.toolResult = toolResult
        self.promptTokens = promptTokens
        self.generatedTokens = generatedTokens
        self.resultTokens = resultTokens
        self.elapsedMs = elapsedMs
    }
}

/// Reason the agent loop stopped.
public enum AgentTerminationReason: String, Codable, Equatable {
    /// The model produced a final answer before exhausting the step budget.
    case finalAnswer
    /// The step budget was exhausted and the loop forced a final answer turn.
    case budgetExhausted
    /// The task was cancelled.
    case cancelled
}

/// Ordered, serializable record of an agent run.
public struct AgentTranscript: Codable, Equatable {
    /// Original user task.
    public let task: String

    /// Messages rendered into prompts for future turns.
    public var messages: [AgentTranscriptMessage]

    /// Completed tool steps.
    public var steps: [AgentTranscriptStep]

    /// Final answer text, when one was produced.
    public var finalAnswer: String?

    /// Reason the loop stopped.
    public var terminationReason: AgentTerminationReason?

    /// Creates a transcript for a new task.
    public init(
        task: String,
        messages: [AgentTranscriptMessage] = [],
        steps: [AgentTranscriptStep] = [],
        finalAnswer: String? = nil,
        terminationReason: AgentTerminationReason? = nil
    ) {
        self.task = task
        self.messages = messages
        self.steps = steps
        self.finalAnswer = finalAnswer
        self.terminationReason = terminationReason
    }
}

enum AgentJSON {
    static func canonicalObjectString(_ object: [String: Any]) -> String {
        guard JSONSerialization.isValidJSONObject(object),
              let data = try? JSONSerialization.data(
                withJSONObject: object,
                options: [.sortedKeys]
              ),
              let string = String(data: data, encoding: .utf8) else {
            return "{}"
        }
        return string
    }

    static func objectString(_ object: [String: Any]) -> String {
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
}
