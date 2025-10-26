/// SwiftUI view model for TinyBrain Chat demo
///
/// **TB-005:** Now integrated with real tokenizer and sampler!
///
/// Manages the chat interface state and coordinates with the inference runtime.

import Foundation
import SwiftUI
import Combine
import TinyBrainRuntime
import TinyBrainTokenizer

/// View model for the TinyBrain Chat demo app
@MainActor
public class ChatViewModel: ObservableObject {
    /// Current prompt input
    @Published public var promptText: String = ""
    
    /// Generated response text
    @Published public var responseText: String = ""
    
    /// Whether inference is currently running
    @Published public var isGenerating: Bool = false
    
    /// Tokens per second metric
    @Published public var tokensPerSecond: Double = 0.0
    
    /// Average token probability (for confidence indication)
    @Published public var averageProbability: Double = 0.0
    
    /// **TB-005:** Sampling configuration (exposed for UI controls)
    @Published public var temperature: Float = 0.7
    @Published public var topK: Int = 40
    @Published public var useTopK: Bool = true
    
    /// Model runner with quantized weights
    private let runner: ModelRunner
    
    /// **TB-005:** Optional tokenizer (would load from file in production)
    private let tokenizer: (any Tokenizer)?
    
    /// Initialize the view model
    ///
    /// **TB-005 Integration:**
    /// - Real BPETokenizer (if vocabulary file available)
    /// - Configurable sampling (temperature, top-k)
    /// - Rich token metadata (probabilities)
    ///
    /// **Production Usage:**
    /// ```swift
    /// let tokenizer = try? BPETokenizer(vocabularyPath: "tinyllama-vocab.json")
    /// let viewModel = ChatViewModel(tokenizer: tokenizer)
    /// ```
    public init(tokenizer: (any Tokenizer)? = nil) {
        // Create real toy model with quantized weights
        let config = ModelConfig(
            numLayers: 2,      // Small for demo
            hiddenDim: 128,
            numHeads: 4,
            vocabSize: 100,
            maxSeqLen: 256
        )
        
        let weights = ModelWeights.makeToyModel(config: config, seed: 42)
        self.runner = ModelRunner(weights: weights)
        self.tokenizer = tokenizer
    }
    
    /// Generate a response for the current prompt
    ///
    /// **TB-005 Enhanced:**
    /// - Uses real tokenizer if available
    /// - Configurable sampling (temperature, top-k)
    /// - Rich metadata (probabilities, timing)
    /// - Stop tokens for graceful termination
    public func generate() async {
        isGenerating = true
        responseText = ""
        averageProbability = 0.0
        
        defer { isGenerating = false }
        
        // **TB-005:** Tokenization (real or fallback)
        let promptTokens: [Int]
        if let tokenizer = tokenizer {
            // Use real BPE tokenizer
            promptTokens = tokenizer.encode(promptText)
        } else {
            // Fallback: character-based tokenization (demo mode)
            promptTokens = Array(promptText.prefix(10)).map { char in
                Int(char.asciiValue ?? 0) % runner.config.vocabSize
            }
        }
        
        // **TB-005:** Configure sampling
        let samplerConfig = SamplerConfig(
            temperature: temperature,
            topK: useTopK ? topK : nil,
            topP: useTopK ? nil : 0.9,  // Use top-p if not using top-k
            repetitionPenalty: 1.2  // Discourage loops
        )
        
        let generationConfig = GenerationConfig(
            maxTokens: 50,
            sampler: samplerConfig,
            stopTokens: []  // Would include EOS token in production
        )
        
        // Track timing and probabilities
        let startTime = Date()
        var tokenCount = 0
        var totalProbability: Float = 0.0
        
        // **TB-005:** Enhanced streaming with metadata
        do {
            for try await output in runner.generateStream(prompt: promptTokens, config: generationConfig) {
                // Detokenization
                if let tokenizer = tokenizer {
                    // Use real tokenizer
                    let text = tokenizer.decode([output.tokenId])
                    responseText += text
                } else {
                    // Fallback: character-based (demo mode)
                    let char = Character(UnicodeScalar(UInt8(output.tokenId % 94 + 33)))
                    responseText.append(char)
                }
                
                tokenCount += 1
                totalProbability += output.probability
                
                // Update metrics
                let elapsed = Date().timeIntervalSince(startTime)
                tokensPerSecond = Double(tokenCount) / elapsed
                averageProbability = Double(totalProbability) / Double(tokenCount)
            }
        } catch {
            responseText = "Error: \(error.localizedDescription)"
        }
        
        // Reset for next generation
        runner.reset()
    }
}

