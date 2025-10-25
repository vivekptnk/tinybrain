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
    
    /// Log backend selection (if debugging enabled)
    static func log(_ message: String) {
        if debugLogging {
            print("[TinyBrain Backend] \(message)")
        }
    }
}

