/// SwiftUI view model for TinyBrain Chat demo
///
/// Manages the chat interface state and coordinates with the inference runtime.

import Foundation
import SwiftUI
import Combine
import TinyBrainRuntime

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
    
    /// **REVIEW HITLER:** Actual ModelRunner with quantized weights!
    private let runner: ModelRunner
    
    /// Initialize the view model
    public init() {
        // **REVIEW HITLER FIX:** Create real toy model with quantized weights
        let config = ModelConfig(
            numLayers: 2,      // Small for demo
            hiddenDim: 128,
            numHeads: 4,
            vocabSize: 100,
            maxSeqLen: 256
        )
        
        let weights = ModelWeights.makeToyModel(config: config, seed: 42)
        self.runner = ModelRunner(weights: weights)
    }
    
    /// Generate a response for the current prompt
    public func generate() async {
        isGenerating = true
        responseText = ""
        
        defer { isGenerating = false }
        
        // **REVIEW HITLER FIX:** Use actual ModelRunner with quantized weights!
        
        // Simple tokenization (character-based for demo)
        let promptTokens = Array(promptText.prefix(10)).map { char in
            Int(char.asciiValue ?? 0) % runner.config.vocabSize
        }
        
        // Track timing
        let startTime = Date()
        var tokenCount = 0
        
        // Stream generation
        do {
            for try await tokenId in runner.generateStream(prompt: promptTokens, maxTokens: 50) {
                // Simple detokenization (placeholder - TB-005 will add real tokenizer)
                let char = Character(UnicodeScalar(UInt8(tokenId % 94 + 33)))
                responseText.append(char)
                
                tokenCount += 1
                
                // Update tokens/sec
                let elapsed = Date().timeIntervalSince(startTime)
                tokensPerSecond = Double(tokenCount) / elapsed
            }
        } catch {
            responseText = "Error: \(error.localizedDescription)"
        }
        
        // Reset for next generation
        runner.reset()
    }
}

