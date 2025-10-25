/// TinyBrain – Swift-Native On-Device LLM Inference Kit
///
/// TinyBrain provides a transparent, educational runtime for running large language models
/// entirely on-device on iOS and macOS, optimized for Apple Silicon.
///
/// ## Topics
/// ### Getting Started
/// - ``TinyBrain``
/// - ``ModelRunner``
/// 
/// ### Tensor Operations
/// - ``Tensor``
/// - ``TensorShape``
///
/// ### Model Loading
/// - ``ModelLoader``
/// - ``ModelConfig``

import Foundation

/// Main entry point for TinyBrain inference.
///
/// Use this class to load models and generate text on-device.
///
/// Example usage:
/// ```swift
/// let model = try TinyBrain.load("tinyllama-int8.tbf")
/// let stream = try await model.generateStream(prompt: "Explain gravity.")
///
/// for try await token in stream {
///     print(token, terminator: "")
/// }
/// ```
public struct TinyBrain {
    /// TinyBrain version string
    public static let version = "0.1.0"
    
    /// Load a quantized model from the specified path
    /// - Parameter path: Path to the `.tbf` model file
    /// - Returns: A configured model runner ready for inference
    /// - Throws: `ModelError` if the model cannot be loaded
    public static func load(_ path: String) async throws -> ModelRunner {
        // Placeholder implementation
        fatalError("Not yet implemented – tracked in TB-002")
    }
}

/// Errors that can occur during model operations
public enum ModelError: Error {
    case fileNotFound(String)
    case invalidFormat(String)
    case unsupportedVersion(String)
    case outOfMemory
}

/// Placeholder for model runner
public struct ModelRunner {
    /// Generate text from a prompt with streaming output
    /// - Parameter prompt: Input text prompt
    /// - Returns: AsyncSequence of generated tokens
    public func generateStream(prompt: String) async throws -> AsyncThrowingStream<String, Error> {
        // Placeholder implementation
        fatalError("Not yet implemented – tracked in TB-002")
    }
}

