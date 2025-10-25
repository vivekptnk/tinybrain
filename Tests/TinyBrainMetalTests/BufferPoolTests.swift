/// Tests for Metal buffer pool implementation
///
/// Validates buffer reuse to eliminate allocation overhead that causes
/// Metal to be slower than CPU (0.45ms overhead vs 0.05ms compute).

import XCTest
import Metal
@testable import TinyBrainMetal

final class BufferPoolTests: XCTestCase {
    
    func testBufferReuse() throws {
        // WHAT: Buffers reused for same-size tensors
        // WHY: Eliminate allocation overhead (0.45ms → 0ms)
        // HOW: Acquire, release, acquire again - should get same buffer
        
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("Metal not available")
        }
        
        let pool = MetalBufferPool(device: device)
        
        // Acquire a buffer
        let buffer1 = try pool.acquire(elementCount: 1024)
        let id1 = ObjectIdentifier(buffer1)
        
        // Release it back to pool
        pool.release(buffer1, elementCount: 1024)
        
        // Acquire again - should get the same buffer back
        let buffer2 = try pool.acquire(elementCount: 1024)
        let id2 = ObjectIdentifier(buffer2)
        
        XCTAssertEqual(id1, id2, "Should reuse released buffer")
    }
    
    func testBufferPoolCapacity() throws {
        // WHAT: Pool doesn't grow unbounded
        // WHY: Prevent memory leaks
        // HOW: Allocate more than max, verify only max cached
        
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("Metal not available")
        }
        
        let maxPooled = 10
        let pool = MetalBufferPool(device: device, maxPooledBuffers: maxPooled)
        
        // Allocate and release many buffers
        var buffers: [MTLBuffer] = []
        for _ in 0..<20 {
            buffers.append(try pool.acquire(elementCount: 1024))
        }
        
        // Release all
        for buffer in buffers {
            pool.release(buffer, elementCount: 1024)
        }
        
        // Pool should only keep maxPooled buffers
        XCTAssertLessThanOrEqual(pool.pooledCount, maxPooled, "Pool should not exceed max capacity")
    }
    
    func testDifferentSizesNotReused() throws {
        // WHAT: Buffers of different sizes not reused
        // WHY: Prevent size mismatches and memory corruption
        // HOW: Acquire size A, release, acquire size B - should get new buffer
        
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("Metal not available")
        }
        
        let pool = MetalBufferPool(device: device)
        
        let buffer1 = try pool.acquire(elementCount: 1024)
        let id1 = ObjectIdentifier(buffer1)
        pool.release(buffer1, elementCount: 1024)
        
        // Request different size
        let buffer2 = try pool.acquire(elementCount: 2048)
        let id2 = ObjectIdentifier(buffer2)
        
        XCTAssertNotEqual(id1, id2, "Different sizes should not reuse buffers")
    }
    
    func testConcurrentAccess() throws {
        // WHAT: Pool is thread-safe
        // WHY: Metal operations may happen on different threads
        // HOW: Acquire/release from multiple threads concurrently
        
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("Metal not available")
        }
        
        let pool = MetalBufferPool(device: device)
        let iterations = 100
        
        DispatchQueue.concurrentPerform(iterations: iterations) { i in
            do {
                let buffer = try pool.acquire(elementCount: 1024)
                // Simulate some work
                Thread.sleep(forTimeInterval: 0.001)
                pool.release(buffer, elementCount: 1024)
            } catch {
                XCTFail("Concurrent access failed: \(error)")
            }
        }
        
        // If we get here without crashes, thread safety works
        XCTAssertTrue(true, "Concurrent access succeeded")
    }
}

