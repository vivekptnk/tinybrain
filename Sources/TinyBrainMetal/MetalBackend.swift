/// Metal acceleration backend for TinyBrain
///
/// Provides GPU-accelerated operations for inference using Metal Performance Shaders
/// and custom kernels.

import Metal
import Foundation

/// Metal backend for accelerated tensor operations
public final class MetalBackend {
    /// Shared Metal device
    private let device: MTLDevice
    
    /// Command queue for GPU operations
    private let commandQueue: MTLCommandQueue
    
    /// Initialize the Metal backend
    /// - Throws: `MetalError.deviceNotFound` if Metal is unavailable
    public init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MetalError.deviceNotFound
        }
        
        guard let commandQueue = device.makeCommandQueue() else {
            throw MetalError.commandQueueCreationFailed
        }
        
        self.device = device
        self.commandQueue = commandQueue
    }
    
    /// Check if Metal is available on this device
    public static var isAvailable: Bool {
        MTLCreateSystemDefaultDevice() != nil
    }
    
    /// Get device information
    public var deviceInfo: String {
        device.name
    }
}

/// Errors specific to Metal operations
public enum MetalError: Error {
    case deviceNotFound
    case commandQueueCreationFailed
    case shaderCompilationFailed(String)
    case bufferCreationFailed
}

