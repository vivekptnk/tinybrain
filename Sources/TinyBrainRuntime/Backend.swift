/// Backend abstraction for TinyBrain operations
///
/// Allows transparent selection between CPU (Accelerate) and GPU (Metal) implementations.

import Foundation

/// Compute backend for tensor operations
public enum ComputeBackend {
    /// CPU backend using Accelerate framework
    case cpu
    
    /// GPU backend using Metal (if available)
    case metal
    
    /// Automatically select best backend
    case auto
}

/// Global backend configuration
public final class TinyBrainBackend {
    /// Preferred backend (can be changed at runtime)
    public static var preferred: ComputeBackend = .auto
    
    /// Enable debug logging to see which backend is used
    public static var debugLogging: Bool = false
    
    /// Shared Metal backend instance (lazy-initialized)
    /// Type-erased to avoid circular dependency (TinyBrainRuntime → TinyBrainMetal)
    public static var metalBackend: Any?
    
    /// Check if Metal backend is available and configured
    public static var isMetalConfigured: Bool {
        metalBackend != nil
    }
    
    /// Log backend selection (if debugging enabled)
    static func log(_ message: String) {
        if debugLogging {
            print("[TinyBrain Backend] \(message)")
        }
    }
}

/// Protocol for backend implementations to conform to
///
/// Allows runtime polymorphism without compile-time dependency
public protocol MatMulBackend {
    func matmul(_ a: Tensor, _ b: Tensor) throws -> Tensor
}

