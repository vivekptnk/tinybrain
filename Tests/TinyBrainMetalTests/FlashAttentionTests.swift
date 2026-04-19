/// FlashAttention Metal Kernel Tests
///
/// Verifies numerical accuracy of GPU FlashAttention against a CPU reference
/// implementation, including GQA support and edge cases.
///
/// **Tolerance:** relative error < 5e-4 (matching Phase 3 kernel standards).
///
/// **Design pattern:**
/// - Skipped automatically when Metal is unavailable (CI without GPU).
/// - Each test runs both CPU and GPU implementations on the same input
///   and checks element-wise agreement within tolerance.

import XCTest
@testable import TinyBrainMetal
@testable import TinyBrainRuntime

final class FlashAttentionTests: XCTestCase {

    // ── Helpers ────────────────────────────────────────────────────────────

    private let tolerance: Float = 5e-4

    /// Assert that two Float tensors agree element-wise within `tolerance`.
    private func assertClose(
        _ gpu: Tensor<Float>,
        _ cpu: Tensor<Float>,
        tolerance: Float,
        context: String,
        file: StaticString = #file,
        line: UInt = #line
    ) {
        XCTAssertEqual(gpu.shape, cpu.shape,
                       "Shape mismatch in \(context)", file: file, line: line)
        let gpuData = gpu.rawData
        let cpuData = cpu.rawData
        // Combined absolute + relative tolerance:
        // Pass if |gpu - cpu| < atol OR |gpu - cpu| / max(|cpu|, atol) < tolerance
        // This avoids false positives at near-zero values where relative error is magnified.
        let atol: Float = 1e-5
        for i in 0..<cpuData.count {
            let g = gpuData[i]
            let c = cpuData[i]
            let absDiff = abs(g - c)
            if absDiff < atol { continue }
            let relErr = absDiff / max(abs(c), atol)
            XCTAssertLessThan(relErr, tolerance,
                              "\(context): index \(i) GPU=\(g) CPU=\(c) relErr=\(relErr)",
                              file: file, line: line)
        }
    }

    /// CPU reference implementation of multi-head attention
    ///
    /// Computes: softmax(Q @ K^T / sqrt(headDim)) @ V per head, with GQA support.
    private func cpuAttention(
        query: Tensor<Float>,
        keys: Tensor<Float>,
        values: Tensor<Float>,
        mask: Tensor<Float>?,
        headDim: Int,
        numHeads: Int,
        numKVHeads: Int
    ) -> Tensor<Float> {
        let qDims = query.shape.dimensions
        let batch: Int
        let seqLen: Int
        let kvSeqLen: Int

        if qDims.count == 3 {
            batch = qDims[0]
            seqLen = qDims[1]
            kvSeqLen = keys.shape.dimensions[1]
        } else {
            batch = 1
            seqLen = qDims[0]
            kvSeqLen = keys.shape.dimensions[0]
        }

        let qStride = numHeads * headDim
        let kvStride = numKVHeads * headDim
        let repeats = numHeads / numKVHeads
        let scale = 1.0 / sqrt(Float(headDim))

        let qData = query.rawData
        let kData = keys.rawData
        let vData = values.rawData
        let maskData = mask?.rawData

        var outputData = [Float](repeating: 0, count: batch * seqLen * qStride)

        for b in 0..<batch {
            for q in 0..<seqLen {
                for head in 0..<numHeads {
                    let kvHead = head / repeats
                    let qOffset = b * seqLen * qStride + q * qStride + head * headDim

                    // Compute attention scores
                    var scores = [Float](repeating: 0, count: kvSeqLen)
                    for kv in 0..<kvSeqLen {
                        var dot: Float = 0
                        let kOffset = b * kvSeqLen * kvStride + kv * kvStride + kvHead * headDim
                        for d in 0..<headDim {
                            dot += qData[qOffset + d] * kData[kOffset + d]
                        }
                        scores[kv] = dot * scale
                        if let m = maskData {
                            scores[kv] += m[kv]
                        }
                    }

                    // Softmax
                    let maxScore = scores.max() ?? 0
                    var expScores = scores.map { exp($0 - maxScore) }
                    let sumExp = expScores.reduce(0, +)
                    expScores = expScores.map { $0 / sumExp }

                    // Weighted sum of values
                    let oOffset = b * seqLen * qStride + q * qStride + head * headDim
                    for kv in 0..<kvSeqLen {
                        let w = expScores[kv]
                        let vOffset = b * kvSeqLen * kvStride + kv * kvStride + kvHead * headDim
                        for d in 0..<headDim {
                            outputData[oOffset + d] += w * vData[vOffset + d]
                        }
                    }
                }
            }
        }

        return Tensor<Float>(shape: query.shape, data: outputData)
    }

    // ── Basic Correctness ─────────────────────────────────────────────────

    /// Basic MHA: 4 heads, seqLen=8, headDim=16 — standard multi-head attention
    func testBasicMultiHeadAttention() throws {
        guard MetalBackend.isAvailable else { throw XCTSkip("Metal not available") }
        let backend = try MetalBackend()

        let seqLen = 8
        let numHeads = 4
        let numKVHeads = 4
        let headDim = 16
        let dim = numHeads * headDim

        let query = Tensor<Float>.random(shape: TensorShape(seqLen, dim))
        let keys = Tensor<Float>.random(shape: TensorShape(seqLen, dim))
        let values = Tensor<Float>.random(shape: TensorShape(seqLen, dim))

        let gpuOut = try backend.attention(
            query: query, keys: keys, values: values,
            mask: nil, headDim: headDim, numHeads: numHeads, numKVHeads: numKVHeads)

        let cpuOut = cpuAttention(
            query: query, keys: keys, values: values,
            mask: nil, headDim: headDim, numHeads: numHeads, numKVHeads: numKVHeads)

        assertClose(gpuOut, cpuOut, tolerance: tolerance, context: "basic MHA (4 heads, seq=8)")
        print("✓ Basic MHA test passed")
    }

    // ── GQA Tests ─────────────────────────────────────────────────────────

    /// GQA: 32 query heads, 8 KV heads (4x repeat) — typical Llama configuration
    func testGroupedQueryAttention() throws {
        guard MetalBackend.isAvailable else { throw XCTSkip("Metal not available") }
        let backend = try MetalBackend()

        let seqLen = 16
        let numHeads = 32
        let numKVHeads = 8
        let headDim = 16
        let qDim = numHeads * headDim
        let kvDim = numKVHeads * headDim

        let query = Tensor<Float>.random(shape: TensorShape(seqLen, qDim))
        let keys = Tensor<Float>.random(shape: TensorShape(seqLen, kvDim))
        let values = Tensor<Float>.random(shape: TensorShape(seqLen, kvDim))

        let gpuOut = try backend.attention(
            query: query, keys: keys, values: values,
            mask: nil, headDim: headDim, numHeads: numHeads, numKVHeads: numKVHeads)

        let cpuOut = cpuAttention(
            query: query, keys: keys, values: values,
            mask: nil, headDim: headDim, numHeads: numHeads, numKVHeads: numKVHeads)

        assertClose(gpuOut, cpuOut, tolerance: tolerance,
                    context: "GQA (32 heads, 8 KV heads, 4x repeat)")
        print("✓ GQA test passed (numHeads=32, numKVHeads=8)")
    }

    /// GQA: 8 query heads, 2 KV heads (4x repeat)
    func testGQASmallConfig() throws {
        guard MetalBackend.isAvailable else { throw XCTSkip("Metal not available") }
        let backend = try MetalBackend()

        let seqLen = 12
        let numHeads = 8
        let numKVHeads = 2
        let headDim = 32

        let query = Tensor<Float>.random(shape: TensorShape(seqLen, numHeads * headDim))
        let keys = Tensor<Float>.random(shape: TensorShape(seqLen, numKVHeads * headDim))
        let values = Tensor<Float>.random(shape: TensorShape(seqLen, numKVHeads * headDim))

        let gpuOut = try backend.attention(
            query: query, keys: keys, values: values,
            mask: nil, headDim: headDim, numHeads: numHeads, numKVHeads: numKVHeads)

        let cpuOut = cpuAttention(
            query: query, keys: keys, values: values,
            mask: nil, headDim: headDim, numHeads: numHeads, numKVHeads: numKVHeads)

        assertClose(gpuOut, cpuOut, tolerance: tolerance,
                    context: "GQA (8 heads, 2 KV heads)")
        print("✓ GQA small config test passed")
    }

    // ── Edge Cases ────────────────────────────────────────────────────────

    /// seqLen=1 — single token (decoding step)
    func testSingleTokenDecoding() throws {
        guard MetalBackend.isAvailable else { throw XCTSkip("Metal not available") }
        let backend = try MetalBackend()

        let numHeads = 4
        let numKVHeads = 4
        let headDim = 16
        let dim = numHeads * headDim
        let kvSeqLen = 32  // 32 cached positions

        let query = Tensor<Float>.random(shape: TensorShape(1, dim))
        let keys = Tensor<Float>.random(shape: TensorShape(kvSeqLen, dim))
        let values = Tensor<Float>.random(shape: TensorShape(kvSeqLen, dim))

        let gpuOut = try backend.attention(
            query: query, keys: keys, values: values,
            mask: nil, headDim: headDim, numHeads: numHeads, numKVHeads: numKVHeads)

        let cpuOut = cpuAttention(
            query: query, keys: keys, values: values,
            mask: nil, headDim: headDim, numHeads: numHeads, numKVHeads: numKVHeads)

        assertClose(gpuOut, cpuOut, tolerance: tolerance,
                    context: "single token decoding (seqLen=1, kvSeqLen=32)")
        print("✓ Single token decoding test passed")
    }

    /// seqLen not divisible by tile size (Br=32)
    func testNonDivisibleSeqLen() throws {
        guard MetalBackend.isAvailable else { throw XCTSkip("Metal not available") }
        let backend = try MetalBackend()

        let seqLen = 37  // Not divisible by 32
        let numHeads = 4
        let numKVHeads = 4
        let headDim = 16
        let dim = numHeads * headDim

        let query = Tensor<Float>.random(shape: TensorShape(seqLen, dim))
        let keys = Tensor<Float>.random(shape: TensorShape(seqLen, dim))
        let values = Tensor<Float>.random(shape: TensorShape(seqLen, dim))

        let gpuOut = try backend.attention(
            query: query, keys: keys, values: values,
            mask: nil, headDim: headDim, numHeads: numHeads, numKVHeads: numKVHeads)

        let cpuOut = cpuAttention(
            query: query, keys: keys, values: values,
            mask: nil, headDim: headDim, numHeads: numHeads, numKVHeads: numKVHeads)

        assertClose(gpuOut, cpuOut, tolerance: tolerance,
                    context: "non-divisible seqLen=37")
        print("✓ Non-divisible sequence length test passed")
    }

    /// Single head attention
    func testSingleHead() throws {
        guard MetalBackend.isAvailable else { throw XCTSkip("Metal not available") }
        let backend = try MetalBackend()

        let seqLen = 16
        let numHeads = 1
        let numKVHeads = 1
        let headDim = 32

        let query = Tensor<Float>.random(shape: TensorShape(seqLen, headDim))
        let keys = Tensor<Float>.random(shape: TensorShape(seqLen, headDim))
        let values = Tensor<Float>.random(shape: TensorShape(seqLen, headDim))

        let gpuOut = try backend.attention(
            query: query, keys: keys, values: values,
            mask: nil, headDim: headDim, numHeads: numHeads, numKVHeads: numKVHeads)

        let cpuOut = cpuAttention(
            query: query, keys: keys, values: values,
            mask: nil, headDim: headDim, numHeads: numHeads, numKVHeads: numKVHeads)

        assertClose(gpuOut, cpuOut, tolerance: tolerance,
                    context: "single head attention")
        print("✓ Single head attention test passed")
    }

    /// Batch dimension support
    func testBatchedAttention() throws {
        guard MetalBackend.isAvailable else { throw XCTSkip("Metal not available") }
        let backend = try MetalBackend()

        let batch = 2
        let seqLen = 8
        let numHeads = 4
        let numKVHeads = 4
        let headDim = 16
        let dim = numHeads * headDim

        let query = Tensor<Float>.random(shape: TensorShape(batch, seqLen, dim))
        let keys = Tensor<Float>.random(shape: TensorShape(batch, seqLen, dim))
        let values = Tensor<Float>.random(shape: TensorShape(batch, seqLen, dim))

        let gpuOut = try backend.attention(
            query: query, keys: keys, values: values,
            mask: nil, headDim: headDim, numHeads: numHeads, numKVHeads: numKVHeads)

        let cpuOut = cpuAttention(
            query: query, keys: keys, values: values,
            mask: nil, headDim: headDim, numHeads: numHeads, numKVHeads: numKVHeads)

        assertClose(gpuOut, cpuOut, tolerance: tolerance,
                    context: "batched attention (batch=2)")
        print("✓ Batched attention test passed")
    }

    /// With mask (causal-style)
    func testMaskedAttention() throws {
        guard MetalBackend.isAvailable else { throw XCTSkip("Metal not available") }
        let backend = try MetalBackend()

        let seqLen = 16
        let numHeads = 4
        let numKVHeads = 4
        let headDim = 16
        let dim = numHeads * headDim

        // Create a mask that blocks the last 8 positions
        var maskData = [Float](repeating: 0.0, count: seqLen)
        for i in 8..<seqLen {
            maskData[i] = -Float.infinity
        }
        let mask = Tensor<Float>(shape: TensorShape(seqLen), data: maskData)

        let query = Tensor<Float>.random(shape: TensorShape(seqLen, dim))
        let keys = Tensor<Float>.random(shape: TensorShape(seqLen, dim))
        let values = Tensor<Float>.random(shape: TensorShape(seqLen, dim))

        let gpuOut = try backend.attention(
            query: query, keys: keys, values: values,
            mask: mask, headDim: headDim, numHeads: numHeads, numKVHeads: numKVHeads)

        let cpuOut = cpuAttention(
            query: query, keys: keys, values: values,
            mask: mask, headDim: headDim, numHeads: numHeads, numKVHeads: numKVHeads)

        assertClose(gpuOut, cpuOut, tolerance: tolerance,
                    context: "masked attention")
        print("✓ Masked attention test passed")
    }

    // ── Performance Benchmark ─────────────────────────────────────────────

    /// Benchmark: GPU vs CPU for various sequence lengths
    func testFlashAttentionPerformance() throws {
        guard MetalBackend.isAvailable else { throw XCTSkip("Metal not available") }
        let backend = try MetalBackend()

        let numHeads = 8
        let numKVHeads = 8
        let headDim = 64
        let dim = numHeads * headDim

        print("\n📊 FlashAttention Performance Benchmark")
        print(String(repeating: "=", count: 60))
        print("SeqLen      CPU (ms)     GPU (ms)    Speedup")
        print(String(repeating: "-", count: 60))

        for seqLen in [64, 128, 256, 512] {
            let query = Tensor<Float>.random(shape: TensorShape(1, dim))
            let keys = Tensor<Float>.random(shape: TensorShape(seqLen, dim))
            let values = Tensor<Float>.random(shape: TensorShape(seqLen, dim))

            // Warmup GPU
            _ = try backend.attention(
                query: query, keys: keys, values: values,
                mask: nil, headDim: headDim, numHeads: numHeads, numKVHeads: numKVHeads)

            // Benchmark GPU
            let gpuStart = CFAbsoluteTimeGetCurrent()
            let gpuIters = 10
            for _ in 0..<gpuIters {
                _ = try backend.attention(
                    query: query, keys: keys, values: values,
                    mask: nil, headDim: headDim, numHeads: numHeads, numKVHeads: numKVHeads)
            }
            let gpuMs = (CFAbsoluteTimeGetCurrent() - gpuStart) * 1000.0 / Double(gpuIters)

            // Benchmark CPU
            let cpuStart = CFAbsoluteTimeGetCurrent()
            let cpuIters = 10
            for _ in 0..<cpuIters {
                _ = cpuAttention(
                    query: query, keys: keys, values: values,
                    mask: nil, headDim: headDim, numHeads: numHeads, numKVHeads: numKVHeads)
            }
            let cpuMs = (CFAbsoluteTimeGetCurrent() - cpuStart) * 1000.0 / Double(cpuIters)

            let speedup = cpuMs / gpuMs
            print(String(format: "%-10d %12.3f %12.3f %9.2fx", seqLen, cpuMs, gpuMs, speedup))
        }
        print(String(repeating: "=", count: 60))
    }

    /// Correctness at larger sequence length (256)
    func testLargerSequenceCorrectness() throws {
        guard MetalBackend.isAvailable else { throw XCTSkip("Metal not available") }
        let backend = try MetalBackend()

        let seqLen = 256
        let numHeads = 8
        let numKVHeads = 4
        let headDim = 32
        let qDim = numHeads * headDim
        let kvDim = numKVHeads * headDim

        let query = Tensor<Float>.random(shape: TensorShape(seqLen, qDim))
        let keys = Tensor<Float>.random(shape: TensorShape(seqLen, kvDim))
        let values = Tensor<Float>.random(shape: TensorShape(seqLen, kvDim))

        let gpuOut = try backend.attention(
            query: query, keys: keys, values: values,
            mask: nil, headDim: headDim, numHeads: numHeads, numKVHeads: numKVHeads)

        let cpuOut = cpuAttention(
            query: query, keys: keys, values: values,
            mask: nil, headDim: headDim, numHeads: numHeads, numKVHeads: numKVHeads)

        // Use slightly relaxed tolerance for larger sequences (more accumulation error)
        assertClose(gpuOut, cpuOut, tolerance: 1e-3,
                    context: "larger sequence (seqLen=256, GQA 8:4)")
        print("✓ Larger sequence correctness test passed (seqLen=256)")
    }

    /// Validate headDim=64 (common in real models like TinyLlama)
    func testHeadDim64() throws {
        guard MetalBackend.isAvailable else { throw XCTSkip("Metal not available") }
        let backend = try MetalBackend()

        let seqLen = 16
        let numHeads = 4
        let numKVHeads = 4
        let headDim = 64
        let dim = numHeads * headDim

        let query = Tensor<Float>.random(shape: TensorShape(seqLen, dim))
        let keys = Tensor<Float>.random(shape: TensorShape(seqLen, dim))
        let values = Tensor<Float>.random(shape: TensorShape(seqLen, dim))

        let gpuOut = try backend.attention(
            query: query, keys: keys, values: values,
            mask: nil, headDim: headDim, numHeads: numHeads, numKVHeads: numKVHeads)

        let cpuOut = cpuAttention(
            query: query, keys: keys, values: values,
            mask: nil, headDim: headDim, numHeads: numHeads, numKVHeads: numKVHeads)

        assertClose(gpuOut, cpuOut, tolerance: tolerance,
                    context: "headDim=64")
        print("✓ headDim=64 test passed")
    }

    /// Validate dispatch of attention through the Backend protocol
    func testAttentionBackendDispatch() throws {
        guard MetalBackend.isAvailable else { throw XCTSkip("Metal not available") }
        let backend = try MetalBackend()

        // Register the metal backend
        let previousBackend = TinyBrainBackend.metalBackend
        TinyBrainBackend.metalBackend = backend
        defer { TinyBrainBackend.metalBackend = previousBackend }

        let seqLen = 8
        let numHeads = 4
        let numKVHeads = 4
        let headDim = 16
        let dim = numHeads * headDim

        let query = Tensor<Float>.random(shape: TensorShape(seqLen, dim))
        let keys = Tensor<Float>.random(shape: TensorShape(seqLen, dim))
        let values = Tensor<Float>.random(shape: TensorShape(seqLen, dim))

        var usedFallback = false
        let result = TinyBrainBackend.dispatchAttention(
            query: query, keys: keys, values: values,
            mask: nil, headDim: headDim, numHeads: numHeads, numKVHeads: numKVHeads,
            cpuFallback: {
                usedFallback = true
                return Tensor<Float>.random(shape: TensorShape(seqLen, dim))
            })

        // Should have used GPU, not fallback
        XCTAssertFalse(usedFallback, "Should use GPU backend, not CPU fallback")
        XCTAssertEqual(result.shape, query.shape, "Output shape should match query shape")

        // Verify correctness
        let cpuOut = cpuAttention(
            query: query, keys: keys, values: values,
            mask: nil, headDim: headDim, numHeads: numHeads, numKVHeads: numKVHeads)
        assertClose(result, cpuOut, tolerance: tolerance,
                    context: "dispatch through Backend protocol")
        print("✓ Backend dispatch test passed")
    }
}
