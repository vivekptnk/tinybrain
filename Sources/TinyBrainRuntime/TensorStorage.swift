/// Storage abstraction for Tensor data
///
/// Allows tensors to live on CPU or GPU with lazy synchronization.
/// This is the key to eliminating the 0.45ms transfer overhead that made
/// Metal slower than CPU in TB-003.

import Foundation
#if canImport(Metal)
import Metal
#endif

/// Defines where tensor data resides
public enum TensorLocation {
    case cpu       // Data on CPU memory
    case gpu       // Data on GPU memory (Metal buffer)
    case both      // Data synchronized on both (rare)
}

/// **TB-004 Phase 2:** Generic storage backend for tensor data
///
/// This class uses reference semantics (class, not struct) because:
/// - Actual storage buffer should be shared until mutation (CoW)
/// - Copy-on-write optimization needs reference counting
/// - GPU buffers are inherently reference types
public final class TensorStorage<Element: TensorElement> {
    /// CPU data (may be nil if data only on GPU)
    public var cpuData: [Element]?
    
    /// GPU buffer ID (stored as Any to avoid circular dependency)
    /// In practice, this is an MTLBuffer when data is on GPU
    public var gpuBuffer: Any?
    
    /// Track which location has the "source of truth"
    public var location: TensorLocation
    
    /// Number of elements
    public let count: Int
    
    /// **REVIEW HITLER FIX:** Callback to release GPU buffer
    /// Called when storage is destroyed to return buffer to pool
    public var releaseCallback: (() -> Void)?
    
    /// Initialize with CPU data
    public init(cpuData: [Element]) {
        self.cpuData = cpuData
        self.gpuBuffer = nil
        self.location = .cpu
        self.count = cpuData.count
    }
    
    /// Initialize with GPU buffer
    public init(gpuBuffer: Any, count: Int) {
        self.cpuData = nil
        self.gpuBuffer = gpuBuffer
        self.location = .gpu
        self.count = count
    }
    
    /// Check if data is on CPU
    public var isOnCPU: Bool {
        location == .cpu || location == .both
    }
    
    /// Check if data is on GPU
    public var isOnGPU: Bool {
        location == .gpu || location == .both
    }
    
    /// Get CPU data, syncing from GPU if needed
    ///
    /// This is the "lazy synchronization" that eliminates unnecessary transfers.
    public func getCPUData() -> [Element] {
        if let data = cpuData {
            return data
        }
        
        // Need to sync from GPU
        guard let buffer = gpuBuffer else {
            fatalError("Storage has no data on CPU or GPU")
        }
        
        // Sync from GPU using Metal if available
        #if canImport(Metal)
        if let mtlBuffer = buffer as? MTLBuffer {
            let pointer = mtlBuffer.contents().bindMemory(to: Element.self, capacity: count)
            let data = Array(UnsafeBufferPointer(start: pointer, count: count))
            
            // Cache CPU data for next access
            self.cpuData = data
            self.location = .both
            
            return data
        }
        #endif
        
        // Fallback: no GPU data available
        fatalError("Cannot sync from GPU - Metal not available or buffer invalid")
    }
    
    /// Create a copy of this storage
    public func copy() -> TensorStorage<Element> {
        let storage = TensorStorage(cpuData: getCPUData())
        storage.location = .cpu  // Copy always starts on CPU
        return storage
    }
    
    /// **REVIEW HITLER FIX:** Release GPU buffer when storage is destroyed
    deinit {
        // Call release callback if set (releases buffer back to pool)
        releaseCallback?()
    }
}

// MARK: - Type Aliases for Backward Compatibility

/// Storage for Float32 tensors (backward compatibility)
public typealias FloatTensorStorage = TensorStorage<Float>

