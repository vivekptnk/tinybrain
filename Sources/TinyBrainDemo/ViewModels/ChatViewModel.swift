/// Enhanced Chat View Model
///
/// **TDD Phase: GREEN**
/// Orchestrates chat UI, streaming generation, and telemetry.
///
/// Integrates:
/// - Message history management
/// - Real-time streaming generation
/// - Telemetry tracking
/// - Sampler configuration
/// - Error handling
///
/// **TB-006:** Complete rebuild from TB-005 demo version

import Foundation
import SwiftUI
import Combine
import TinyBrainRuntime
import TinyBrainTokenizer

/// Main view model for the TinyBrain Chat interface
@MainActor
public final class ChatViewModel: ObservableObject {

    // MARK: - Published Properties

    /// All messages in the conversation
    @Published public var messages: [Message] = []

    /// Current user input
    @Published public var promptText: String = ""

    /// Whether AI is currently generating
    @Published public var isGenerating: Bool = false

    /// Error state
    @Published public var hasError: Bool = false
    @Published public var errorMessage: String = ""

    /// Sampler configuration
    @Published public var temperature: Float = ModelPromptStyle.rawCompletion.samplingDefaults.temperature
    @Published public var topK: Int = 40
    @Published public var topP: Float = 0.9
    @Published public var useTopK: Bool = true

    /// Display name for the model that is actually backing the runner.
    @Published public private(set) var activeModelName: String

    /// Honest precision badge for the active model.
    @Published public private(set) var activeQuant: QuantBadge

    /// Active file-backed model path, or nil when the built-in toy model runs.
    @Published public private(set) var activeModelPath: String?

    /// Whether a model switch is in progress
    @Published public private(set) var isSwitchingModel: Bool = false

    /// Path currently being loaded, used by the picker for inline progress.
    @Published public private(set) var pendingModelPath: String?

    /// Model target from the last failed switch, used by the banner Retry action.
    @Published public private(set) var failedModelSwitchTarget: ModelInfo?

    /// Assistant messages whose generation failed mid-stream.
    @Published public private(set) var failedMessageIDs: Set<UUID> = []

    /// Telemetry view model
    @Published public private(set) var telemetry: TelemetryViewModel

    /// X-Ray visualization view model (TB-010)
    @Published public private(set) var xRay: XRayViewModel

    /// Agent workbench view model.
    @Published public private(set) var agent: AgentViewModel

    /// Model picker view model
    public let modelPicker: ModelPickerViewModel

    // MARK: - Private State

    /// Model runner (mutable to allow hot-swapping models)
    private var runner: ModelRunner

    /// Active weights used to build a separate agent runner.
    private var activeWeights: ModelWeights?

    /// Optional tokenizer (real or mock, mutable for hot-swapping)
    private var tokenizer: (any Tokenizer)?

    /// Prompt formatting mode for the active model family.
    private var activePromptStyle: ModelPromptStyle

    /// Current generation task
    private var generationTask: Task<Void, Never>?

    /// Assistant message currently receiving streamed tokens.
    private var currentStreamingMessageID: UUID?

    /// Decode a single token ID to text (for X-Ray display)
    public func decodeToken(_ tokenId: Int) -> String {
        tokenizer?.decode([tokenId]) ?? "[\(tokenId)]"
    }

    // MARK: - Initialization

    /// Initialize with pre-configured model runner and optional tokenizer.
    ///
    /// - Parameters:
    ///   - runner: Model runner already configured with loaded weights.
    ///   - tokenizer: Optional tokenizer for prompt encoding/decoding.
    ///   - activeModelName: Display name for the weights backing `runner`.
    ///   - activeQuant: Badge derived by the caller from loaded weights, or `.toy` for fallback.
    ///   - activeModelPath: Absolute path for a real `.tbf` model, or nil for toy.
    public init(
        runner: ModelRunner,
        activeWeights: ModelWeights? = nil,
        tokenizer: (any Tokenizer)? = nil,
        activeModelName: String = "Toy Model",
        activeQuant: QuantBadge = .toy,
        activeModelPath: String? = nil
    ) {
        self.runner = runner
        self.activeWeights = activeWeights
        self.tokenizer = tokenizer
        self.activeModelName = activeModelName
        self.activeQuant = activeQuant
        self.activeModelPath = activeModelPath
        self.activePromptStyle = activeModelPath.map { ModelInfo(path: $0).promptStyle } ?? .rawCompletion
        self.telemetry = TelemetryViewModel()
        self.xRay = XRayViewModel(numLayers: runner.config.numLayers)
        self.modelPicker = ModelPickerViewModel()
        let defaults = activePromptStyle.samplingDefaults
        self.temperature = defaults.temperature
        self.topK = defaults.topK ?? 0
        self.topP = defaults.topP ?? 1.0
        self.useTopK = defaults.topK != nil
        self.agent = AgentViewModel(
            activeWeights: activeWeights,
            tokenizer: tokenizer,
            promptStyle: activePromptStyle,
            sampler: Self.samplerConfig(
                temperature: defaults.temperature,
                topK: defaults.topK,
                topP: defaults.topP,
                useTopK: defaults.topK != nil,
                includesTopPWithTopK: defaults.includesTopPWithTopK,
                repetitionPenalty: defaults.repetitionPenalty
            ),
            activeModelName: activeModelName,
            isToyModel: activeModelPath == nil
        )
        self.modelPicker.refresh()
        self.modelPicker.select(path: activeModelPath)
    }

    // MARK: - Model Switching

    /// Switch to a different model file at runtime.
    ///
    /// This resets the conversation, loads the new weights + tokenizer,
    /// and rebuilds the runner. If `model` is nil, reverts to the toy model.
    ///
    /// - Parameter model: The ModelInfo to load, or nil for the toy model.
    public func switchModel(_ model: ModelInfo?) async {
        guard !isGenerating, !agent.isRunning else { return }

        isSwitchingModel = true
        pendingModelPath = model?.path
        failedModelSwitchTarget = nil
        let previousModelPath = activeModelPath
        modelPicker.select(path: model?.path)

        do {
            let (weights, newTokenizer) = try await modelPicker.loadSelected()

            // Rebuild runner with new weights and matching tokenizer atomically.
            runner = ModelRunner(weights: weights)
            activeWeights = model == nil ? nil : weights
            tokenizer = newTokenizer
            activePromptStyle = model?.promptStyle ?? .rawCompletion
            applySamplerDefaults(for: activePromptStyle)

            // Rebuild X-Ray for new layer count.
            xRay = XRayViewModel(numLayers: runner.config.numLayers)
            clearConversation()

            if let model {
                activeModelName = model.displayName
                activeQuant = QuantBadge.derived(from: weights, fallback: QuantBadge(hint: model.quantization))
                activeModelPath = model.path
            } else {
                activeModelName = "Toy Model"
                activeQuant = .toy
                activeModelPath = nil
            }

            agent.reconfigure(
                weights: activeWeights,
                tokenizer: tokenizer,
                promptStyle: activePromptStyle,
                sampler: currentSamplerConfig,
                activeModelName: activeModelName,
                isToyModel: activeModelPath == nil
            )
        } catch {
            modelPicker.select(path: previousModelPath)
            failedModelSwitchTarget = model
            handleError(message: modelPicker.switchError ?? error.localizedDescription)
        }

        isSwitchingModel = false
        pendingModelPath = nil
    }

    /// Retry the last failed model switch, if one is available.
    public func retryLastModelSwitch() async {
        guard let failedModelSwitchTarget else { return }
        await switchModel(failedModelSwitchTarget)
    }
    
    // MARK: - Message Management
    
    /// Add user message from current prompt
    public func addUserMessage() {
        let trimmed = promptText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        
        let message = Message(role: .user, content: trimmed)
        messages.append(message)
        promptText = ""
    }
    
    /// Add assistant message
    public func addAssistantMessage(content: String) {
        let message = Message(role: .assistant, content: content)
        messages.append(message)
    }
    
    /// Clear all messages and reset state
    public func clearConversation() {
        messages.removeAll()
        isGenerating = false
        generationTask?.cancel()
        generationTask = nil
        currentStreamingMessageID = nil
        failedMessageIDs.removeAll()
        telemetry.reset()
        xRay.reset()
        runner.reset()
        clearError()
    }

    /// Toggle X-Ray observation on/off
    public func setXRayEnabled(_ enabled: Bool) {
        xRay.isEnabled = enabled
        runner.observer = enabled ? xRay : nil
    }
    
    // MARK: - Generation
    
    /// Generate response for current prompt
    public func generate() async {
        guard !promptText.isEmpty else { return }
        guard !isGenerating else { return }
        guard !agent.isRunning else { return }
        
        // Add user message
        addUserMessage()
        
        // Start generation
        setGenerating(true)
        telemetry.reset()
        clearError()
        
        let task = Task {
            do {
                try await performGeneration()
            } catch is CancellationError {
                // Stop is a cancellation path, not an error state.
            } catch {
                if let currentStreamingMessageID {
                    failedMessageIDs.insert(currentStreamingMessageID)
                }
                handleError(message: "Generation failed: \(error.localizedDescription)")
            }
            setGenerating(false)
            currentStreamingMessageID = nil
        }

        generationTask = task
        await task.value
    }

    /// Cancel the active generation without clearing the conversation.
    public func stopGeneration() {
        guard isGenerating else { return }
        generationTask?.cancel()
        generationTask = nil
        setGenerating(false)
    }
    
    /// Format conversation history using TinyLlama/Zephyr chat template.
    private func formatZephyrChatPrompt() -> String {
        TinyBrainChatTemplate.format(messages: messages)
    }

    /// Format conversation history using Qwen ChatML.
    private func formatQwenChatPrompt() -> String {
        TinyBrainQwenChatTemplate.format(messages: messages)
    }

    /// Format a raw completion prompt for base models.
    private func formatRawPrompt() throws -> String {
        guard let lastUserMessage = messages.last(where: { $0.isUser }) else {
            throw ChatError.noUserMessage
        }
        return lastUserMessage.content
    }

    private func formattedPromptForActiveModel() throws -> String {
        switch activePromptStyle {
        case .zephyrChat:
            return formatZephyrChatPrompt()
        case .qwenChatML:
            return formatQwenChatPrompt()
        case .rawCompletion:
            return try formatRawPrompt()
        }
    }

    private var activeGenerationMaxTokens: Int {
        switch activePromptStyle {
        case .zephyrChat:
            return TinyBrainChatDefaults.chatMaxTokens
        case .qwenChatML:
            return TinyBrainChatDefaults.qwenChatMaxTokens
        case .rawCompletion:
            return 64
        }
    }

    private func performGeneration() async throws {
        // Get last user message
        guard messages.last(where: { $0.isUser }) != nil else {
            throw ChatError.noUserMessage
        }

        // Build prompt with the active model family's expected formatting.
        let prompt = try formattedPromptForActiveModel()

        // Tokenize with the active tokenizer's source-configured BOS behavior.
        let promptTokens = TinyBrainPromptTokenizer.encode(
            prompt: prompt,
            tokenizer: tokenizer,
            fallbackVocabSize: runner.config.vocabSize
        )

        // Reset runner for fresh generation (clear KV cache from previous turns)
        runner.reset()

        let stopTokens = TinyBrainChatStops.stopTokenIDs(
            for: tokenizer,
            promptStyle: activePromptStyle
        )

        let stopSequences = TinyBrainChatStops.stopSequences(
            for: tokenizer,
            promptStyle: activePromptStyle,
            eosTokens: stopTokens
        )

        // Configure generation
        let generationConfig = GenerationConfig(
            maxTokens: activeGenerationMaxTokens,
            sampler: currentSamplerConfig,
            stopTokens: stopTokens
        )
        
        // Create assistant message to accumulate response
        var responseContent = ""
        addAssistantMessage(content: responseContent)
        let assistantIndex = messages.count - 1
        let assistantID = messages[assistantIndex].id
        currentStreamingMessageID = assistantID
        
        var detokenizer: IncrementalDetokenizer?
        if let tokenizer {
            detokenizer = IncrementalDetokenizer(tokenizer: tokenizer)
        }

        var stopMatcher = StopSequenceMatcher<TokenOutput>(
            stopSequences: stopSequences,
            tokenID: { $0.tokenId }
        )
        var stoppedBySequence = false

        // Stream generation
        for try await output in runner.generateStream(prompt: promptTokens, config: generationConfig) {
            // Check cancellation
            if Task.isCancelled { break }

            switch stopMatcher.append(output) {
            case .emit(let outputs):
                for safeOutput in outputs {
                    appendGeneratedToken(
                        safeOutput,
                        detokenizer: &detokenizer,
                        responseContent: &responseContent,
                        assistantIndex: assistantIndex,
                        assistantID: assistantID
                    )
                }
            case .stop(let outputsBeforeStop):
                for safeOutput in outputsBeforeStop {
                    appendGeneratedToken(
                        safeOutput,
                        detokenizer: &detokenizer,
                        responseContent: &responseContent,
                        assistantIndex: assistantIndex,
                        assistantID: assistantID
                    )
                }
                stoppedBySequence = true
            }
            
            if stoppedBySequence {
                break
            }

            // Small delay for animation smoothness
            try? await Task.sleep(nanoseconds: 50_000_000) // 50ms
        }

        if !stoppedBySequence {
            for safeOutput in stopMatcher.flush() {
                appendGeneratedToken(
                    safeOutput,
                    detokenizer: &detokenizer,
                    responseContent: &responseContent,
                    assistantIndex: assistantIndex,
                    assistantID: assistantID
                )
            }
        }
    }

    private func appendGeneratedToken(
        _ output: TokenOutput,
        detokenizer: inout IncrementalDetokenizer?,
        responseContent: inout String,
        assistantIndex: Int,
        assistantID: UUID
    ) {
        // Detokenize
        let text: String
        if tokenizer != nil {
            guard let delta = detokenizer?.append(output.tokenId) else {
                return
            }
            text = delta
        } else {
            // Fallback: character-based
            let char = Character(UnicodeScalar(UInt8(output.tokenId % 94 + 33)))
            text = String(char)
        }

        responseContent += text

        // Update message
        if assistantIndex < messages.count {
            messages[assistantIndex] = Message(
                id: assistantID,
                role: .assistant,
                content: responseContent,
                timestamp: messages[assistantIndex].timestamp
            )
        }

        // Update telemetry
        telemetry.recordTokenWithProbability(
            tokenId: output.tokenId,
            probability: output.probability,
            at: Date()
        )
        telemetry.calculateMetrics()

        // Update X-Ray KV cache visualization
        if xRay.isEnabled {
            xRay.kvCachePages = runner.kvCache.pageAllocationStatus()
        }
    }
    
    // MARK: - Sampler Configuration
    
    /// Current sampler configuration based on UI settings
    public var currentSamplerConfig: SamplerConfig {
        let defaults = activePromptStyle.samplingDefaults
        let topPValue = (!useTopK || defaults.includesTopPWithTopK) ? topP : nil
        return SamplerConfig(
            temperature: temperature,
            topK: useTopK ? topK : nil,
            topP: topPValue,
            repetitionPenalty: defaults.repetitionPenalty
        )
    }

    private func applySamplerDefaults(for promptStyle: ModelPromptStyle) {
        let defaults = promptStyle.samplingDefaults
        temperature = defaults.temperature
        topK = defaults.topK ?? 0
        topP = defaults.topP ?? 1.0
        useTopK = defaults.topK != nil
        agent.updateSampler(currentSamplerConfig)
    }

    private static func samplerConfig(
        temperature: Float,
        topK: Int?,
        topP: Float?,
        useTopK: Bool,
        includesTopPWithTopK: Bool,
        repetitionPenalty: Float
    ) -> SamplerConfig {
        let topPValue = (!useTopK || includesTopPWithTopK) ? topP : nil
        return SamplerConfig(
            temperature: temperature,
            topK: useTopK ? topK : nil,
            topP: topPValue,
            repetitionPenalty: repetitionPenalty
        )
    }
    
    /// Apply a preset sampler configuration
    public func applySamplerPreset(_ preset: SamplerPreset) {
        switch preset {
        case .balanced:
            temperature = 0.7
            topK = 40
            topP = 0.9
            useTopK = true
        case .creative:
            temperature = 1.2
            topK = 100
            topP = 0.95
            useTopK = false
        case .precise:
            temperature = 0.3
            topK = 10
            topP = 0.8
            useTopK = true
        }
    }
    
    // MARK: - State Management
    
    /// Set generation state
    public func setGenerating(_ generating: Bool) {
        isGenerating = generating
    }
    
    /// Handle error
    public func handleError(message: String) {
        hasError = true
        errorMessage = message
    }
    
    /// Clear error state
    public func clearError() {
        hasError = false
        errorMessage = ""
        failedModelSwitchTarget = nil
    }

    // MARK: - Screenshot Automation

    /// Seed a deterministic transcript and live telemetry for screenshot verification.
    public func seedDemoTranscriptForScreenshots() {
        let now = Date()
        messages = [
            Message(role: .user, content: "Explain what TinyBrain is doing on-device.", timestamp: now.addingTimeInterval(-150)),
            Message(role: .assistant, content: "TinyBrain is running a compact transformer locally, keeping prompts and weights on this Mac while streaming each decoded token.", timestamp: now.addingTimeInterval(-132)),
            Message(role: .user, content: "Show me the live signals.", timestamp: now.addingTimeInterval(-34)),
            Message(role: .assistant, content: "The X-Ray panel is tracking attention, layer norms, candidate tokens, and KV cache pages as the next token forms", timestamp: now.addingTimeInterval(-8))
        ]
        failedMessageIDs.removeAll()
        telemetry.seedDemoValues(
            tokensPerSecond: 31.8,
            millisecondsPerToken: 42,
            energyEstimate: 1.72,
            kvCacheUsagePercent: 38
        )
        isGenerating = true
        seedDemoXRaySnapshot()
    }

    /// Surface a deterministic sample error for screenshot verification.
    public func seedDemoErrorForScreenshots() {
        handleError(message: "Sample model switch failed: incompatible tensor metadata.")
    }

    private func seedDemoXRaySnapshot() {
        let layerCount = max(runner.config.numLayers, 2)
        let snapshots = (0..<8).map { position in
            XRaySnapshot(
                position: position,
                timestamp: Date().addingTimeInterval(Double(position - 8)),
                attentionWeights: (0..<layerCount).map { layer in
                    (0...position).map { column in
                        let distance = abs(position - column)
                        let base = max(0.05, 1.0 - Float(distance) * 0.14)
                        return min(1.0, base * (0.75 + Float(layer % 3) * 0.08))
                    }
                },
                layerNorms: (0..<layerCount).map { Float($0 + 1) / Float(layerCount) * 2.2 },
                topCandidates: [
                    TokenCandidate(tokenId: 318, probability: 0.42),
                    TokenCandidate(tokenId: 262, probability: 0.21),
                    TokenCandidate(tokenId: 1402, probability: 0.12),
                    TokenCandidate(tokenId: 29889, probability: 0.08)
                ],
                entropy: 2.4
            )
        }
        xRay.snapshotHistory = snapshots
        xRay.latestSnapshot = snapshots.last
        xRay.kvCachePages = (0..<64).map { $0 < 24 }
    }
}

// MARK: - Supporting Types

enum TinyBrainChatDefaults {
    // TinyLlama-1.1B can still wander; these defaults aim for short, bounded
    // replies rather than promising high factual quality.
    static let systemPrompt = "You are TinyBrain, a concise on-device assistant. Answer the user directly in a sentence or two. If you don't know, say so briefly."
    static let temperature: Float = 0.4
    static let chatMaxTokens = 128
    static let qwenChatMaxTokens = 200
}

struct TinyBrainSamplingDefaults {
    let temperature: Float
    let topK: Int?
    let topP: Float?
    let includesTopPWithTopK: Bool
    let repetitionPenalty: Float
}

extension ModelPromptStyle {
    var samplingDefaults: TinyBrainSamplingDefaults {
        switch self {
        case .zephyrChat, .rawCompletion:
            return TinyBrainSamplingDefaults(
                temperature: TinyBrainChatDefaults.temperature,
                topK: 40,
                topP: 0.9,
                includesTopPWithTopK: false,
                repetitionPenalty: 1.2
            )
        case .qwenChatML:
            return TinyBrainSamplingDefaults(
                temperature: 0.7,
                topK: 20,
                topP: 0.8,
                includesTopPWithTopK: true,
                repetitionPenalty: 1.05
            )
        }
    }
}

enum TinyBrainChatTemplate {
    static func format(
        messages: [Message],
        systemPrompt: String = TinyBrainChatDefaults.systemPrompt
    ) -> String {
        var prompt = ""
        prompt += "<|system|>\n\(systemPrompt)</s>\n"
        for message in messages {
            if message.isUser {
                prompt += "<|user|>\n\(message.content)</s>\n"
            } else if !message.content.isEmpty {
                prompt += "<|assistant|>\n\(message.content)</s>\n"
            }
        }
        prompt += "<|assistant|>\n"
        return prompt
    }
}

enum TinyBrainQwenChatTemplate {
    static func format(
        messages: [Message],
        systemPrompt: String = TinyBrainChatDefaults.systemPrompt
    ) -> String {
        var prompt = ""
        prompt += "<|im_start|>system\n\(systemPrompt)<|im_end|>\n"
        for message in messages {
            if message.isUser {
                prompt += "<|im_start|>user\n\(message.content)<|im_end|>\n"
            } else if !message.content.isEmpty {
                prompt += "<|im_start|>assistant\n\(message.content)<|im_end|>\n"
            }
        }
        prompt += "<|im_start|>assistant\n"
        return prompt
    }
}

enum TinyBrainPromptTokenizer {
    static func encode(
        prompt: String,
        tokenizer: (any Tokenizer)?,
        fallbackVocabSize: Int
    ) -> [Int] {
        if let tokenizer {
            var encoded = tokenizer.encode(prompt)
            if let bpeTokenizer = tokenizer as? BPETokenizer, bpeTokenizer.addsBosToken {
                encoded.insert(bpeTokenizer.bosToken, at: 0)
            }
            return encoded
        }

        return Array(prompt.prefix(50)).map { char in
            Int(char.asciiValue ?? 0) % fallbackVocabSize
        }
    }
}

enum TinyBrainChatStops {
    static let endOfSequenceMarker = "</s>"
    static let turnBoundaryMarkers = ["<|user|>", "<|system|>"]
    static let qwenEndOfTextToken = 151_643
    static let qwenTurnBoundaryMarker = "<|im_end|>"

    static func stopTokenIDs(
        for tokenizer: (any Tokenizer)?,
        promptStyle: ModelPromptStyle
    ) -> [Int] {
        guard let bpeTokenizer = tokenizer as? BPETokenizer else {
            return []
        }

        var tokens = [bpeTokenizer.eosToken]
        if promptStyle == .qwenChatML {
            tokens.append(qwenEndOfTextToken)
        }
        return uniqueTokenIDs(tokens)
    }

    static func stopSequences(
        for tokenizer: (any Tokenizer)?,
        promptStyle: ModelPromptStyle,
        eosTokens: [Int]
    ) -> [[Int]] {
        var sequences = eosTokens.map { [$0] }

        guard let tokenizer else {
            return uniqueSequences(sequences)
        }

        switch promptStyle {
        case .zephyrChat:
            sequences.append(tokenizer.encode(endOfSequenceMarker))
            sequences.append(contentsOf: turnBoundaryMarkers.map { tokenizer.encode($0) })
        case .qwenChatML:
            sequences.append(tokenizer.encode(qwenTurnBoundaryMarker))
        case .rawCompletion:
            sequences.append(tokenizer.encode(endOfSequenceMarker))
        }
        return uniqueSequences(sequences.filter { !$0.isEmpty })
    }

    private static func uniqueTokenIDs(_ tokens: [Int]) -> [Int] {
        var seen: Set<Int> = []
        var result: [Int] = []
        for token in tokens where seen.insert(token).inserted {
            result.append(token)
        }
        return result
    }

    private static func uniqueSequences(_ sequences: [[Int]]) -> [[Int]] {
        var seen: Set<String> = []
        var result: [[Int]] = []
        for sequence in sequences {
            let key = sequence.map(String.init).joined(separator: ",")
            if seen.insert(key).inserted {
                result.append(sequence)
            }
        }
        return result
    }
}

enum StopSequenceDecision<Element> {
    case emit([Element])
    case stop([Element])
}

struct StopSequenceMatcher<Element> {
    private let stopSequences: [[Int]]
    private let tokenID: (Element) -> Int
    private var pending: [Element] = []

    init(stopSequences: [[Int]], tokenID: @escaping (Element) -> Int) {
        self.stopSequences = stopSequences
            .filter { !$0.isEmpty }
            .sorted { $0.count > $1.count }
        self.tokenID = tokenID
    }

    mutating func append(_ element: Element) -> StopSequenceDecision<Element> {
        guard !stopSequences.isEmpty else {
            return .emit([element])
        }

        pending.append(element)
        let ids = pending.map(tokenID)

        if let stopLength = matchingStopSuffixLength(in: ids) {
            let emitCount = pending.count - stopLength
            let outputsBeforeStop = Array(pending.prefix(emitCount))
            pending.removeAll()
            return .stop(outputsBeforeStop)
        }

        let keepCount = longestStopPrefixSuffixLength(in: ids)
        let emitCount = pending.count - keepCount
        guard emitCount > 0 else {
            return .emit([])
        }

        let safeOutputs = Array(pending.prefix(emitCount))
        pending = Array(pending.suffix(keepCount))
        return .emit(safeOutputs)
    }

    mutating func flush() -> [Element] {
        defer { pending.removeAll() }
        return pending
    }

    private func matchingStopSuffixLength(in ids: [Int]) -> Int? {
        for sequence in stopSequences where ids.count >= sequence.count {
            if Array(ids.suffix(sequence.count)) == sequence {
                return sequence.count
            }
        }
        return nil
    }

    private func longestStopPrefixSuffixLength(in ids: [Int]) -> Int {
        let maxLength = min(ids.count, stopSequences.map(\.count).max() ?? 0)
        guard maxLength > 0 else { return 0 }

        for length in stride(from: maxLength, through: 1, by: -1) {
            let suffix = Array(ids.suffix(length))
            if stopSequences.contains(where: { sequence in
                sequence.count >= length && Array(sequence.prefix(length)) == suffix
            }) {
                return length
            }
        }
        return 0
    }
}

/// Sampler presets for quick configuration
public enum SamplerPreset {
    case balanced
    case creative
    case precise
}

/// Chat-specific errors
enum ChatError: Error {
    case noUserMessage
    case generationCancelled
}
