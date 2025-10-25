/// SwiftUI view model for TinyBrain Chat demo
///
/// Manages the chat interface state and coordinates with the inference runtime.

import Foundation
import SwiftUI
import Combine

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
    
    /// Initialize the view model
    public init() {}
    
    /// Generate a response for the current prompt
    public func generate() async {
        isGenerating = true
        responseText = ""
        
        // Placeholder - actual implementation tracked in TB-007
        defer { isGenerating = false }
        
        // Simulate streaming response
        let placeholderResponse = "This is a placeholder response. Actual inference will be implemented in TB-007."
        for char in placeholderResponse {
            responseText.append(char)
            try? await Task.sleep(nanoseconds: 50_000_000) // 50ms delay
        }
    }
}

