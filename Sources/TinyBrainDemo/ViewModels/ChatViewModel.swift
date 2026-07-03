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
    @Published public var temperature: Float = 0.7
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

    /// Model picker view model
    public let modelPicker: ModelPickerViewModel

    // MARK: - Private State

    /// Model runner (mutable to allow hot-swapping models)
    private var runner: ModelRunner

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
        tokenizer: (any Tokenizer)? = nil,
        activeModelName: String = "Toy Model",
        activeQuant: QuantBadge = .toy,
        activeModelPath: String? = nil
    ) {
        self.runner = runner
        self.tokenizer = tokenizer
        self.activeModelName = activeModelName
        self.activeQuant = activeQuant
        self.activeModelPath = activeModelPath
        self.activePromptStyle = activeModelPath.map { ModelInfo(path: $0).promptStyle } ?? .rawCompletion
        self.telemetry = TelemetryViewModel()
        self.xRay = XRayViewModel(numLayers: runner.config.numLayers)
        self.modelPicker = ModelPickerViewModel()
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
        guard !isGenerating else { return }

        isSwitchingModel = true
        pendingModelPath = model?.path
        failedModelSwitchTarget = nil
        let previousModelPath = activeModelPath
        modelPicker.select(path: model?.path)

        do {
            let (weights, newTokenizer) = try await modelPicker.loadSelected()

            // Rebuild runner with new weights and matching tokenizer atomically.
            runner = ModelRunner(weights: weights)
            tokenizer = newTokenizer
            activePromptStyle = model?.promptStyle ?? .rawCompletion

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
    private func formatChatPrompt() -> String {
        var prompt = ""
        // System message
        prompt += "<|system|>\nYou are a friendly, helpful assistant.</s>\n"
        // Conversation turns
        for message in messages {
            if message.isUser {
                prompt += "<|user|>\n\(message.content)</s>\n"
            } else if !message.content.isEmpty {
                prompt += "<|assistant|>\n\(message.content)</s>\n"
            }
        }
        // Generation prompt
        prompt += "<|assistant|>\n"
        return prompt
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
            return formatChatPrompt()
        case .rawCompletion:
            return try formatRawPrompt()
        }
    }

    private var activeGenerationMaxTokens: Int {
        switch activePromptStyle {
        case .zephyrChat:
            return 200
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

        // Tokenize with the active tokenizer's BOS token when available.
        let promptTokens: [Int]
        if let tokenizer = tokenizer {
            var encoded = tokenizer.encode(prompt)
            if let bpeTokenizer = tokenizer as? BPETokenizer {
                encoded.insert(bpeTokenizer.bosToken, at: 0)
            }
            promptTokens = encoded
        } else {
            // Fallback: character-based
            promptTokens = Array(prompt.prefix(50)).map { char in
                Int(char.asciiValue ?? 0) % runner.config.vocabSize
            }
        }

        // Reset runner for fresh generation (clear KV cache from previous turns)
        runner.reset()

        let stopTokens: [Int]
        if let bpeTokenizer = tokenizer as? BPETokenizer {
            stopTokens = [bpeTokenizer.eosToken]
        } else {
            stopTokens = []
        }

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

        // Stream generation
        for try await output in runner.generateStream(prompt: promptTokens, config: generationConfig) {
            // Check cancellation
            if Task.isCancelled { break }

            if stopTokens.contains(output.tokenId) {
                break
            }
            
            // Detokenize
            let text: String
            if tokenizer != nil {
                guard let delta = detokenizer?.append(output.tokenId) else {
                    continue
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

            // Small delay for animation smoothness
            try? await Task.sleep(nanoseconds: 50_000_000) // 50ms
        }
    }
    
    // MARK: - Sampler Configuration
    
    /// Current sampler configuration based on UI settings
    public var currentSamplerConfig: SamplerConfig {
        SamplerConfig(
            temperature: temperature,
            topK: useTopK ? topK : nil,
            topP: useTopK ? nil : topP,
            repetitionPenalty: 1.2
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
