#!/usr/bin/env swift

import TinyBrain

// Find the crossover point where GPU becomes faster than CPU

TinyBrainBackend.enableMetal()
TinyBrainBackend.debugLogging = false

print("Testing GPU vs CPU crossover point on M4...")
print("")
print("Size\t\tCPU (ms)\tGPU (ms)\tSpeedup\tWinner")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

let sizes = [256, 512, 768, 1024, 1536, 2048, 3072, 4096]

for size in sizes {
    let a = Tensor.random(shape: TensorShape(size, size))
    let b = Tensor.random(shape: TensorShape(size, size))
    
    // CPU benchmark
    _ = a.matmulCPU(b)  // Warmup
    let cpuStart = Date()
    let cpuResult = a.matmulCPU(b)
    let cpuTime = Date().timeIntervalSince(cpuStart) * 1000
    
    // GPU benchmark (persistent buffers)
    let gpuA = a.toGPU()
    let gpuB = b.toGPU()
    _ = gpuA.matmul(gpuB)  // Warmup
    
    let gpuStart = Date()
    let gpuResult = gpuA.matmul(gpuB).toCPU()
    let gpuTime = Date().timeIntervalSince(gpuStart) * 1000
    
    let speedup = cpuTime / gpuTime
    let winner = speedup >= 1.0 ? "🎯 GPU" : "🏃 CPU"
    
    print("\(size)×\(size)\t\(String(format: "%.2f", cpuTime))\t\t\(String(format: "%.2f", gpuTime))\t\t\(String(format: "%.2f×", speedup))\t\(winner)")
}

print("")
print("M4 has AMX (Apple Matrix Extension) - dedicated matrix hardware")
print("AMX is faster than GPU for many workloads!")

