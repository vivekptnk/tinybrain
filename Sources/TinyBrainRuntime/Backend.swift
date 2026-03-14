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

    // MARK: - New-op dispatch helpers (Phase 3)

    /// Dispatch softmax to GPU if the backend supports it
    ///
    /// Falls back to the closure `cpuFallback` when Metal is unavailable.
    public static func dispatchSoftmax(
        _ input: Tensor<Float>,
        cpuFallback: () -> Tensor<Float>
    ) -> Tensor<Float> {
        guard let backend = metalBackend as? SoftmaxBackend else {
            return cpuFallback()
        }
        return (try? backend.softmax(input)) ?? cpuFallback()
    }

    /// Dispatch RMSNorm to GPU if the backend supports it
    public static func dispatchRMSNorm(
        _ input: Tensor<Float>,
        weight: Tensor<Float>,
        eps: Float,
        cpuFallback: () -> Tensor<Float>
    ) -> Tensor<Float> {
        guard let backend = metalBackend as? RMSNormBackend else {
            return cpuFallback()
        }
        return (try? backend.rmsnorm(input, weight: weight, eps: eps)) ?? cpuFallback()
    }

    /// Dispatch SiLU activation to GPU if the backend supports it
    public static func dispatchSiLU(
        _ input: Tensor<Float>,
        cpuFallback: () -> Tensor<Float>
    ) -> Tensor<Float> {
        guard let backend = metalBackend as? ActivationBackend else {
            return cpuFallback()
        }
        return (try? backend.silu(input)) ?? cpuFallback()
    }

    /// Dispatch GELU activation to GPU if the backend supports it
    public static func dispatchGELU(
        _ input: Tensor<Float>,
        cpuFallback: () -> Tensor<Float>
    ) -> Tensor<Float> {
        guard let backend = metalBackend as? ActivationBackend else {
            return cpuFallback()
        }
        return (try? backend.gelu(input)) ?? cpuFallback()
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

/// **REVIEW HITLER FIX:** Protocol for INT8 quantized operations
public protocol QuantizedMatMulBackend {
    func matmulQuantized(_ input: Tensor<Float>, _ quantized: QuantizedTensor) throws -> Tensor<Float>
}

/// Protocol for GPU-accelerated softmax
///
/// Implementations should apply numerically stable softmax along the last dimension.
public protocol SoftmaxBackend {
    /// Apply softmax along last dimension
    /// - Parameter input: Tensor of any shape; softmax is applied per-row (last dim)
    /// - Returns: Tensor of same shape with rows summing to 1
    func softmax(_ input: Tensor<Float>) throws -> Tensor<Float>
}

/// Protocol for GPU-accelerated RMSNorm
///
/// Used by Llama-family models for pre-normalization.
public protocol RMSNormBackend {
    /// Apply RMSNorm: out = (x / rms(x)) * weight
    /// - Parameters:
    ///   - input: Input tensor [numTokens, hiddenDim]
    ///   - weight: Per-feature scale weights [hiddenDim]
    ///   - eps: Epsilon for numerical stability (default 1e-5)
    /// - Returns: Normalized tensor of same shape
    func rmsnorm(_ input: Tensor<Float>, weight: Tensor<Float>, eps: Float) throws -> Tensor<Float>
}

/// Protocol for GPU-accelerated element-wise activation functions
public protocol ActivationBackend {
    /// SiLU activation: out[i] = x[i] * sigmoid(x[i])
    func silu(_ input: Tensor<Float>) throws -> Tensor<Float>

    /// GELU activation (tanh approximation)
    func gelu(_ input: Tensor<Float>) throws -> Tensor<Float>
}

