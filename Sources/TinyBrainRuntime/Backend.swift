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
    /// **TB-004:** This is implemented by the TinyBrain umbrella module.
    ///
    /// Call this once at app startup to enable GPU acceleration.
    /// If Metal is unavailable, falls back to CPU silently.
    ///
    /// Example:
    /// ```swift
    /// import TinyBrain
    ///
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
        // Stub implementation - overridden by TinyBrain umbrella module
        log("enableMetal() called but not overridden by umbrella module")
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
///
/// **TB-004:** Currently supports Float32 tensors (Float16/Int8 coming later)
public protocol MatMulBackend {
    func matmul(_ a: Tensor<Float>, _ b: Tensor<Float>) throws -> Tensor<Float>
}

/// **TB-004:** Protocol for uploading tensors to GPU
public protocol TensorUploader {
    func uploadTensor(_ tensor: Tensor<Float>) throws -> Tensor<Float>
}

/// **TB-004:** Protocol for downloading tensors from GPU
public protocol TensorDownloader {
    func downloadTensor(_ tensor: Tensor<Float>) -> Tensor<Float>
}

