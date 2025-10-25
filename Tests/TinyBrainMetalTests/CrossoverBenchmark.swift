/// Find the crossover point where GPU becomes faster than CPU
///
/// Tests multiple matrix sizes to determine when GPU parallelism
/// overcomes the AMX advantage on M4.

import XCTest
import TinyBrainMetal
@testable import TinyBrainRuntime

final class CrossoverBenchmark: XCTestCase {
    
    func testGPUCPUCrossover() throws {
        guard MetalBackend.isAvailable else {
            throw XCTSkip("Metal not available")
        }
        
        let backend = try MetalBackend()
        TinyBrainBackend.metalBackend = backend
        TinyBrainBackend.preferred = .metal
        
        print("")
        print("🧪 Testing GPU vs CPU Crossover on M4")
        print("")
        print("Size\t\tCPU (ms)\tGPU (ms)\tSpeedup\tWinner")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        
        let sizes = [256, 512, 768, 1024, 1536, 2048, 3072, 4096]
        
        for size in sizes {
            let a = Tensor.random(shape: TensorShape(size, size))
            let b = Tensor.random(shape: TensorShape(size, size))
            
            // CPU benchmark
            _ = a.matmulCPU(b)  // Warmup
            let cpuStart = Date()
            _ = a.matmulCPU(b)
            let cpuTime = Date().timeIntervalSince(cpuStart) * 1000
            
            // GPU benchmark with persistent buffers
            let gpuA = a.toGPU()
            let gpuB = b.toGPU()
            _ = gpuA.matmul(gpuB)  // Warmup
            
            let gpuStart = Date()
            _ = gpuA.matmul(gpuB)
            let gpuTime = Date().timeIntervalSince(gpuStart) * 1000
            
            let speedup = cpuTime / gpuTime
            let winner = speedup >= 1.0 ? "🎯 GPU" : "🏃 CPU"
            
            print("\(size)×\(size)\t\t\(String(format: "%.2f", cpuTime))\t\t\(String(format: "%.2f", gpuTime))\t\t\(String(format: "%.2f×", speedup))\t\(winner)")
        }
        
        print("")
        print("💡 M4 has AMX (Apple Matrix Extension) - dedicated matrix hardware")
        print("   AMX is optimized for certain sizes and often beats GPU!")
        print("")
        
        // Test passes if we successfully measured all sizes
        XCTAssertTrue(true, "Crossover benchmark completed")
    }
}

