import Foundation
import TinyBrainRuntime
import TinyBrainTokenizer

/// Actor that runs TinyBrain's plan-act-observe agent loop.
public actor AgentLoop {
    private let generator: any AgentGenerating
    private let tokenizer: any Tokenizer
    private let registry: ToolRegistry
    private let config: AgentConfig
    private weak var observer: AgentTraceObserver?
    private var activeTranscript: AgentTranscript

    /// Creates an agent loop around an injected generator.
    public init(
        generator: any AgentGenerating,
        tokenizer: any Tokenizer,
        registry: ToolRegistry,
        config: AgentConfig = AgentConfig(),
        observer: AgentTraceObserver? = nil
    ) {
        self.generator = generator
        self.tokenizer = tokenizer
        self.registry = registry
        self.config = config
        self.observer = observer
        self.activeTranscript = AgentTranscript(task: "")
    }

    /// Creates an agent loop around a concrete TinyBrain model runner.
    public init(
        runner: ModelRunner,
        tokenizer: any Tokenizer,
        registry: ToolRegistry,
        config: AgentConfig = AgentConfig(),
        observer: AgentTraceObserver? = nil
    ) {
        self.generator = ModelRunnerAgentGenerator(runner: runner, tokenizer: tokenizer)
        self.tokenizer = tokenizer
        self.registry = registry
        self.config = config
        self.observer = observer
        self.activeTranscript = AgentTranscript(task: "")
    }

    /// Current transcript for the active or most recent run.
    public var transcript: AgentTranscript {
        activeTranscript
    }

    /// Runs until the model emits a final answer, the budget is exhausted, or the task is cancelled.
    public nonisolated func run(_ task: String) -> AsyncThrowingStream<AgentEvent, Error> {
        AsyncThrowingStream { continuation in
            let runTask = Task {
                await self.execute(task, continuation: continuation)
            }
            continuation.onTermination = { @Sendable _ in
                runTask.cancel()
            }
        }
    }

    private func execute(
        _ task: String,
        continuation: AsyncThrowingStream<AgentEvent, Error>.Continuation
    ) async {
        activeTranscript = AgentTranscript(
            task: task,
            messages: [AgentTranscriptMessage(role: .user, content: task)]
        )

        do {
            try await runPlanActObserveLoop(continuation: continuation)
            continuation.finish()
        } catch is CancellationError {
            activeTranscript.terminationReason = .cancelled
            let event = AgentCancelled(reason: "Task cancelled", transcript: activeTranscript)
            emit(.cancelled(event), continuation: continuation)
            continuation.finish()
        } catch {
            continuation.finish(throwing: error)
        }
    }

    private func runPlanActObserveLoop(
        continuation: AsyncThrowingStream<AgentEvent, Error>.Continuation
    ) async throws {
        var stepIndex = 0

        while stepIndex < config.maxSteps {
            try Task.checkCancellation()

            let definitions = await registry.definitions
            let dispatcher = await registry.dispatcher()
            let stepToolChoice = effectiveToolChoice()
            let prompt = await renderPrompt(toolChoice: stepToolChoice, mode: .toolOrFinalAnswer)
            let promptTokens = tokenCount(prompt)
            emit(
                .stepStarted(AgentStepStarted(index: stepIndex, promptTokens: promptTokens)),
                continuation: continuation
            )

            let generation = try await generator.generate(
                AgentGenerationRequest(
                    stepIndex: stepIndex,
                    prompt: prompt,
                    mode: .toolOrFinalAnswer,
                    maxTokens: config.perStepTokenBudget,
                    sampler: config.sampler,
                    stopTokens: config.stopTokens,
                    toolDefinitions: definitions,
                    toolChoice: stepToolChoice,
                    constraintMode: config.constraintMode
                )
            )
            try Task.checkCancellation()

            switch modelOutcome(from: generation.text, definitions: definitions, stepIndex: stepIndex) {
            case .finalAnswer(let answer):
                finishWithFinalAnswer(
                    answer,
                    reason: .finalAnswer,
                    continuation: continuation
                )
                return

            case .toolCall(let call):
                emit(
                    .toolCallProposed(
                        AgentToolCallProposed(index: stepIndex, call: call, rawOutput: generation.text)
                    ),
                    continuation: continuation
                )
                let started = Date()
                let result = await dispatchSafely(call, dispatcher: dispatcher)
                let elapsedMs = Date().timeIntervalSince(started) * 1_000
                appendStep(
                    index: stepIndex,
                    modelOutput: generation.text,
                    call: call,
                    result: result,
                    promptTokens: promptTokens,
                    generatedTokens: generation.tokenCount,
                    elapsedMs: elapsedMs
                )
                emit(
                    .toolExecuted(
                        AgentToolExecuted(
                            index: stepIndex,
                            call: call,
                            result: result,
                            elapsedMs: elapsedMs,
                            resultTokens: tokenCount(result.content)
                        )
                    ),
                    continuation: continuation
                )
                stepIndex += 1

            case .toolError(let result):
                appendStep(
                    index: stepIndex,
                    modelOutput: generation.text,
                    call: nil,
                    result: result,
                    promptTokens: promptTokens,
                    generatedTokens: generation.tokenCount,
                    elapsedMs: 0
                )
                emit(
                    .toolExecuted(
                        AgentToolExecuted(
                            index: stepIndex,
                            call: nil,
                            result: result,
                            elapsedMs: 0,
                            resultTokens: tokenCount(result.content)
                        )
                    ),
                    continuation: continuation
                )
                stepIndex += 1
            }
        }

        activeTranscript.terminationReason = .budgetExhausted
        emit(
            .budgetExhausted(
                AgentBudgetExhausted(maxSteps: config.maxSteps, transcript: activeTranscript)
            ),
            continuation: continuation
        )
        try await forceFinalAnswer(continuation: continuation)
    }

    private func forceFinalAnswer(
        continuation: AsyncThrowingStream<AgentEvent, Error>.Continuation
    ) async throws {
        try Task.checkCancellation()
        let prompt = await renderPrompt(toolChoice: .none, mode: .finalAnswer)
        let promptTokens = tokenCount(prompt)
        emit(
            .stepStarted(AgentStepStarted(index: activeTranscript.steps.count, promptTokens: promptTokens)),
            continuation: continuation
        )
        let generation = try await generator.generate(
            AgentGenerationRequest(
                stepIndex: activeTranscript.steps.count,
                prompt: prompt,
                mode: .finalAnswer,
                maxTokens: config.perStepTokenBudget,
                sampler: config.sampler,
                stopTokens: config.stopTokens,
                toolDefinitions: [],
                toolChoice: .none,
                constraintMode: .none
            )
        )
        finishWithFinalAnswer(
            generation.text,
            reason: .budgetExhausted,
            continuation: continuation
        )
    }

    private func effectiveToolChoice() -> ToolChoice {
        if config.toolChoice == .required, !activeTranscript.steps.isEmpty {
            return .auto
        }
        return config.toolChoice
    }

    private func dispatchSafely(_ call: ToolCall, dispatcher: ToolDispatcher) async -> ToolResult {
        do {
            return try await dispatcher.dispatch(call)
        } catch {
            return ToolResult(
                callId: call.id,
                content: "Error: \(error.localizedDescription)",
                isError: true
            )
        }
    }

    private func appendStep(
        index: Int,
        modelOutput: String,
        call: ToolCall?,
        result: ToolResult,
        promptTokens: Int,
        generatedTokens: Int,
        elapsedMs: Double
    ) {
        activeTranscript.messages.append(
            AgentTranscriptMessage(role: .assistant, content: modelOutput)
        )
        activeTranscript.messages.append(
            AgentTranscriptMessage(
                role: .tool,
                content: formatObservation(call: call, result: result)
            )
        )
        activeTranscript.steps.append(
            AgentTranscriptStep(
                index: index,
                modelOutput: modelOutput,
                toolCall: call.map(AgentToolCallSnapshot.init(call:)),
                toolResult: AgentToolResultSnapshot(result: result),
                promptTokens: promptTokens,
                generatedTokens: generatedTokens,
                resultTokens: tokenCount(result.content),
                elapsedMs: elapsedMs
            )
        )
    }

    private func finishWithFinalAnswer(
        _ rawAnswer: String,
        reason: AgentTerminationReason,
        continuation: AsyncThrowingStream<AgentEvent, Error>.Continuation
    ) {
        let answer = sanitizeFinalAnswer(rawAnswer)
        activeTranscript.messages.append(
            AgentTranscriptMessage(role: .assistant, content: answer)
        )
        activeTranscript.finalAnswer = answer
        activeTranscript.terminationReason = reason
        emit(
            .finalAnswer(AgentFinalAnswer(answer: answer, transcript: activeTranscript)),
            continuation: continuation
        )
    }

    private func modelOutcome(
        from text: String,
        definitions: [ToolDefinition],
        stepIndex: Int
    ) -> ModelOutcome {
        var parser = ToolCallParser(tools: definitions)
        parser.feed(text)

        guard parser.hasCompleteJSON else {
            return .finalAnswer(text)
        }

        switch parser.extractToolCall() {
        case .success(let call):
            if let final = finalAnswer(from: call) {
                return .finalAnswer(final)
            }
            return .toolCall(call)
        case .failure(let error):
            return .toolError(
                ToolResult(
                    callId: "parse_error_\(stepIndex)",
                    content: "Error: \(error.description)",
                    isError: true
                )
            )
        }
    }

    private func finalAnswer(from call: ToolCall) -> String? {
        guard call.name == "final_answer" || call.name == "final" else {
            return nil
        }
        return call.arguments["answer"] as? String ?? call.arguments["content"] as? String
    }

    private func renderPrompt(
        toolChoice: ToolChoice,
        mode: AgentGenerationMode
    ) async -> String {
        let toolPrompt = await registry.systemPrompt(choice: toolChoice)
        let system = systemPrompt(toolPrompt: toolPrompt, mode: mode)
        let messages = compactedMessages(systemPrompt: system, mode: mode)

        switch config.promptStyle {
        case .qwenChatML:
            return renderQwen(systemPrompt: system, messages: messages)
        case .zephyrChat:
            return renderZephyr(systemPrompt: system, messages: messages)
        case .rawCompletion:
            return renderRaw(systemPrompt: system, messages: messages)
        }
    }

    private func systemPrompt(toolPrompt: String, mode: AgentGenerationMode) -> String {
        var prompt = """
        You are TinyBrain Agent, a private on-device assistant. Work one step at a time.
        Use exactly one tool call when a tool is needed, then wait for the observation.
        When you have enough evidence, answer the user directly in natural language.
        Tool errors are observations; explain what you can from the transcript.
        """

        if !toolPrompt.isEmpty {
            prompt += "\n\n\(toolPrompt)"
        }

        switch mode {
        case .toolOrFinalAnswer:
            prompt += "\n\nReturn either one JSON tool call or a final natural-language answer. Do not include more than one tool call."
        case .finalAnswer:
            prompt += "\n\nThe step budget is exhausted. Do not call tools. Answer with the evidence already in the transcript."
        }
        return prompt
    }

    private func compactedMessages(
        systemPrompt: String,
        mode: AgentGenerationMode
    ) -> [AgentTranscriptMessage] {
        var messages = activeTranscript.messages
        while tokenCount(renderRaw(systemPrompt: systemPrompt, messages: messages)) > config.contextBudget,
              messages.count > 1 {
            if let removable = messages.firstIndex(where: { $0.role != .user }) {
                messages.remove(at: removable)
            } else {
                break
            }
        }
        return messages
    }

    private func renderQwen(
        systemPrompt: String,
        messages: [AgentTranscriptMessage]
    ) -> String {
        var prompt = "<|im_start|>system\n\(systemPrompt)<|im_end|>\n"
        var index = 0
        while index < messages.count {
            let message = messages[index]
            switch message.role {
            case .user, .assistant:
                prompt += "<|im_start|>\(message.role.rawValue)\n\(message.content)<|im_end|>\n"
                index += 1
            case .tool:
                prompt += "<|im_start|>user\n"
                var isFirstToolResponse = true
                while index < messages.count, messages[index].role == .tool {
                    if !isFirstToolResponse {
                        prompt += "\n"
                    }
                    prompt += "<tool_response>\n\(messages[index].content)\n</tool_response>"
                    isFirstToolResponse = false
                    index += 1
                }
                prompt += "<|im_end|>\n"
            }
        }
        prompt += "<|im_start|>assistant\n"
        return prompt
    }

    private func renderZephyr(
        systemPrompt: String,
        messages: [AgentTranscriptMessage]
    ) -> String {
        var prompt = "<|system|>\n\(systemPrompt)</s>\n"
        for message in messages {
            switch message.role {
            case .user:
                prompt += "<|user|>\n\(message.content)</s>\n"
            case .assistant:
                prompt += "<|assistant|>\n\(message.content)</s>\n"
            case .tool:
                prompt += "<|user|>\nTool observation:\n\(message.content)</s>\n"
            }
        }
        prompt += "<|assistant|>\n"
        return prompt
    }

    private func renderRaw(
        systemPrompt: String,
        messages: [AgentTranscriptMessage]
    ) -> String {
        var prompt = "System:\n\(systemPrompt)\n\n"
        for message in messages {
            prompt += "\(message.role.rawValue.capitalized):\n\(message.content)\n\n"
        }
        prompt += "Assistant:\n"
        return prompt
    }

    private func formatObservation(call: ToolCall?, result: ToolResult) -> String {
        result.content
    }

    private func sanitizeFinalAnswer(_ answer: String) -> String {
        var text = answer
        for marker in ["<|im_end|>", "<|endoftext|>", "</s>"] {
            if let range = text.range(of: marker) {
                text = String(text[..<range.lowerBound])
            }
        }
        return text.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func tokenCount(_ text: String) -> Int {
        tokenizer.encode(text).count
    }

    private func emit(
        _ event: AgentEvent,
        continuation: AsyncThrowingStream<AgentEvent, Error>.Continuation
    ) {
        switch continuation.yield(event) {
        case .terminated:
            return
        case .enqueued, .dropped:
            break
        @unknown default:
            break
        }

        guard let observer else {
            return
        }
        switch event {
        case .stepStarted(let payload):
            observer.stepStarted(payload)
        case .toolCallProposed(let payload):
            observer.toolCallProposed(payload)
        case .toolExecuted(let payload):
            observer.toolExecuted(payload)
        case .finalAnswer(let payload):
            observer.finalAnswer(payload)
        case .budgetExhausted(let payload):
            observer.budgetExhausted(payload)
        case .cancelled(let payload):
            observer.cancelled(payload)
        }
    }
}

private enum ModelOutcome {
    case finalAnswer(String)
    case toolCall(ToolCall)
    case toolError(ToolResult)
}
