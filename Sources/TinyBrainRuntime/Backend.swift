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
    
    /// Size threshold for GPU (matrices smaller than this use CPU)
    /// Default: 512×512 (GPU overhead dominates below this)
    public static var metalSizeThreshold: Int = 512
    
    /// Check if Metal backend is available and configured
    public static var isMetalConfigured: Bool {
        metalBackend != nil
    }
    
    /// Check if matrix is large enough to benefit from GPU
    ///
    /// Small matrices (< 512×512) are faster on CPU due to overhead.
    /// - Parameters:
    ///   - m: Rows
    ///   - n: Columns
    /// - Returns: True if matrix should use Metal
    public static func shouldUseMetal(m: Int, n: Int) -> Bool {
        let size = max(m, n)
        return size >= metalSizeThreshold
    }
    
    /// Enable Metal GPU acceleration (opt-in)
    ///
    /// Call this once at app startup to enable GPU acceleration.
    /// If Metal is unavailable, falls back to CPU silently.
    ///
    /// Example:
    /// ```swift
    /// // In your app's init:
    /// TinyBrainBackend.enableMetal()
    ///
    /// // Now matmul automatically uses GPU for large matrices!
    /// let c = a.matmul(b)  // GPU if ≥1024×1024
    /// ```
    ///
    /// - Returns: True if Metal was successfully enabled
    @discardableResult
    public static func enableMetal() -> Bool {
        guard metalBackend == nil else {
            return true  // Already enabled
        }
        
        // Try to create Metal backend
        // Note: Actual creation happens in TinyBrainMetal module
        // This is a placeholder - umbrella module will override
        return false
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

