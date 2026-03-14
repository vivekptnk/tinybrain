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
    
    /// Quantization mode (for UI display)
    @Published public var quantizationMode: QuantizationMode = .int8
    
    /// Telemetry view model
    @Published public private(set) var telemetry: TelemetryViewModel
    
    // MARK: - Private State
    
    /// Model runner
    private let runner: ModelRunner
    
    /// Optional tokenizer (real or mock)
    private let tokenizer: (any Tokenizer)?
    
    /// Current generation task
    private var generationTask: Task<Void, Never>?
    
    // MARK: - Initialization
    
    /// Initialize with pre-configured model runner and optional tokenizer
    public init(runner: ModelRunner, tokenizer: (any Tokenizer)? = nil) {
        self.runner = runner
        self.tokenizer = tokenizer
        self.telemetry = TelemetryViewModel()
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
        telemetry.reset()
        runner.reset()
        clearError()
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
        
        generationTask = Task {
            do {
                try await performGeneration()
            } catch {
                handleError(message: "Generation failed: \(error.localizedDescription)")
            }
            setGenerating(false)
        }
        
        await generationTask?.value
    }
    
    private func performGeneration() async throws {
        // Get last user message
        guard let lastUserMessage = messages.last(where: { $0.isUser }) else {
            throw ChatError.noUserMessage
        }
        
        // Tokenize
        let promptTokens: [Int]
        if let tokenizer = tokenizer {
            promptTokens = tokenizer.encode(lastUserMessage.content)
        } else {
            // Fallback: character-based
            promptTokens = Array(lastUserMessage.content.prefix(10)).map { char in
                Int(char.asciiValue ?? 0) % runner.config.vocabSize
            }
        }
        
        // Configure generation
        let generationConfig = GenerationConfig(
            maxTokens: 50,
            sampler: currentSamplerConfig,
            stopTokens: []
        )
        
        // Create assistant message to accumulate response
        var responseContent = ""
        addAssistantMessage(content: responseContent)
        let assistantIndex = messages.count - 1
        
        // Stream generation
        for try await output in runner.generateStream(prompt: promptTokens, config: generationConfig) {
            // Check cancellation
            if Task.isCancelled { break }
            
            // Detokenize
            let text: String
            if let tokenizer = tokenizer {
                text = tokenizer.decode([output.tokenId])
            } else {
                // Fallback: character-based
                let char = Character(UnicodeScalar(UInt8(output.tokenId % 94 + 33)))
                text = String(char)
            }
            
            responseContent += text
            
            // Update message
            if assistantIndex < messages.count {
                messages[assistantIndex] = Message(
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
    }
}

// MARK: - Supporting Types

/// Sampler presets for quick configuration
public enum SamplerPreset {
    case balanced
    case creative
    case precise
}

/// Quantization modes (for UI display)
public enum QuantizationMode: String, CaseIterable {
    case fp16 = "FP16"
    case int8 = "INT8"
    case int4 = "INT4"
    
    public var memoryMultiplier: Double {
        switch self {
        case .fp16: return 1.0
        case .int8: return 0.5
        case .int4: return 0.25
        }
    }
}

/// Chat-specific errors
enum ChatError: Error {
    case noUserMessage
    case generationCancelled
}

