/// Phase 3 Metal Kernel Tests
///
/// Verifies numerical accuracy of GPU softmax, RMSNorm, SiLU, and GELU
/// against CPU reference implementations.
///
/// **Tolerance:** `|gpu - cpu| <= atol + rtol * |cpu|` (numpy-style `allclose`),
/// with `rtol = 5e-4` and `atol = 1e-6`. The absolute floor prevents relative
/// error from inflating to meaningless values when `|cpu|` is near zero — a
/// regime that the GELU tanh approximation hits for Gaussian tail inputs.
///
/// **Design pattern:**
/// - Skipped automatically when Metal is unavailable (CI without GPU).
/// - Each test runs both CPU and GPU implementations on the same input
///   and checks that every element agrees within the tolerance.
/// - Randomized inputs use `SeededGenerator` so CI runs are deterministic.

import XCTest
@testable import TinyBrainMetal
@testable import TinyBrainRuntime

final class Phase3KernelTests: XCTestCase {

    // ── Helpers ────────────────────────────────────────────────────────────

    /// Default relative tolerance between GPU and CPU results.
    private let tolerance: Float = 5e-4

    /// Fixed seed for deterministic random inputs across CI runs (CHA-180).
    private let rngSeed: UInt64 = 2025

    /// Build a seeded RNG for deterministic `Tensor.random` calls in tests.
    private func seededRNG() -> any RandomNumberGenerator {
        SeededGenerator(seed: rngSeed)
    }

    /// Deterministic `Tensor<Float>.random` using the fixed test seed.
    private func randomTensor(shape: TensorShape) -> Tensor<Float> {
        var rng: any RandomNumberGenerator = seededRNG()
        return Tensor<Float>.random(shape: shape, using: &rng)
    }

    /// Assert that two Float tensors agree element-wise.
    ///
    /// Uses combined absolute + relative tolerance (numpy `allclose` semantics):
    /// `|gpu - cpu| <= atol + rtol * |cpu|`. The absolute floor is essential
    /// for activations like GELU whose CPU reference produces values near zero
    /// for Gaussian-tail inputs, where ULP-scale GPU/CPU differences would
    /// otherwise inflate to arbitrary relative error.
    private func assertClose(
        _ gpu: Tensor<Float>,
        _ cpu: Tensor<Float>,
        tolerance: Float,
        atol: Float = 1e-6,
        context: String,
        file: StaticString = #file,
        line: UInt = #line
    ) {
        XCTAssertEqual(gpu.shape, cpu.shape,
                       "Shape mismatch in \(context)", file: file, line: line)
        for i in 0..<cpu.shape.count {
            let g = gpu.rawData[i]
            let c = cpu.rawData[i]
            let diff = abs(g - c)
            let allowed = atol + tolerance * abs(c)
            XCTAssertLessThanOrEqual(
                diff, allowed,
                "\(context): index \(i) GPU=\(g) CPU=\(c) diff=\(diff) allowed=\(allowed)",
                file: file, line: line
            )
        }
    }

    // ── Softmax ────────────────────────────────────────────────────────────

    /// Basic 1-D softmax: small known vector
    func testSoftmax1D() throws {
        guard MetalBackend.isAvailable else { throw XCTSkip("Metal not available") }
        let backend = try MetalBackend()

        let input = Tensor<Float>(shape: TensorShape(4), data: [1.0, 2.0, 3.0, 4.0])
        let gpuOut = try backend.softmax(input)
        let cpuOut = input.softmaxCPU()

        assertClose(gpuOut, cpuOut, tolerance: tolerance, context: "softmax 1-D")

        // Rows must sum to 1
        let sum = gpuOut.rawData.reduce(0, +)
        XCTAssertEqual(sum, 1.0, accuracy: 1e-5, "Softmax 1-D row should sum to 1")
        print("softmax 1-D: GPU=\(gpuOut.rawData)")
    }

    /// 2-D softmax over 8 rows of 16 elements each
    func testSoftmax2D() throws {
        guard MetalBackend.isAvailable else { throw XCTSkip("Metal not available") }
        let backend = try MetalBackend()

        let rows = 8
        let cols = 16
        let input = randomTensor(shape: TensorShape(rows, cols))

        let gpuOut = try backend.softmax(input)
        let cpuOut = input.softmaxCPU()

        assertClose(gpuOut, cpuOut, tolerance: tolerance, context: "softmax 2-D")

        // Each row should sum to 1
        for r in 0..<rows {
            var rowSum: Float = 0.0
            for c in 0..<cols {
                rowSum += gpuOut[r, c]
            }
            XCTAssertEqual(rowSum, 1.0, accuracy: 1e-4, "Row \(r) should sum to 1")
        }
        print("softmax 2-D [\(rows)×\(cols)]: passed")
    }

    /// Numerically challenging: large input values that would overflow naive softmax
    func testSoftmaxNumericalStability() throws {
        guard MetalBackend.isAvailable else { throw XCTSkip("Metal not available") }
        let backend = try MetalBackend()

        // Inputs spanning a wide range — naive exp() would produce NaN/Inf
        let data: [Float] = [1000.0, 999.0, 998.0, 0.0, -100.0, -200.0, 500.0, 501.0]
        let input = Tensor<Float>(shape: TensorShape(data.count), data: data)

        let gpuOut = try backend.softmax(input)

        // No NaN or Inf values
        for (i, v) in gpuOut.rawData.enumerated() {
            XCTAssertFalse(v.isNaN,  "softmax output NaN at index \(i)")
            XCTAssertFalse(v.isInfinite, "softmax output Inf at index \(i)")
        }

        let sum = gpuOut.rawData.reduce(0, +)
        XCTAssertEqual(sum, 1.0, accuracy: 1e-4, "Softmax of large values should still sum to 1")
        print("softmax numerical stability: passed (sum=\(sum))")
    }

    /// Large matrix (256 rows × 512 cols) — validates threadgroup tiling
    func testSoftmaxLarge() throws {
        guard MetalBackend.isAvailable else { throw XCTSkip("Metal not available") }
        let backend = try MetalBackend()

        let rows = 256
        let cols = 512
        let input = randomTensor(shape: TensorShape(rows, cols))

        let gpuOut = try backend.softmax(input)
        let cpuOut = input.softmaxCPU()

        assertClose(gpuOut, cpuOut, tolerance: tolerance, context: "softmax large [\(rows)×\(cols)]")
        print("softmax large [\(rows)×\(cols)]: passed")
    }

    // ── RMSNorm ────────────────────────────────────────────────────────────

    /// Basic RMSNorm: 1 token × 8 features, identity weight
    func testRMSNormIdentityWeight() throws {
        guard MetalBackend.isAvailable else { throw XCTSkip("Metal not available") }
        let backend = try MetalBackend()

        let hidden = 8
        let data: [Float] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        let input  = Tensor<Float>(shape: TensorShape(1, hidden), data: data)
        let weight = Tensor<Float>(shape: TensorShape(hidden), data: [Float](repeating: 1.0, count: hidden))

        let gpuOut = try backend.rmsnorm(input, weight: weight, eps: 1e-5)
        let cpuOut = input.rmsNormCPU(weight: weight, epsilon: 1e-5)

        assertClose(gpuOut, cpuOut, tolerance: tolerance, context: "rmsnorm identity")
        print("RMSNorm identity weight: GPU=\(gpuOut.rawData)")
    }

    /// RMSNorm with learnable scale weights != 1
    func testRMSNormScaledWeights() throws {
        guard MetalBackend.isAvailable else { throw XCTSkip("Metal not available") }
        let backend = try MetalBackend()

        let tokens = 4
        let hidden = 32
        var rng: any RandomNumberGenerator = seededRNG()
        let input  = Tensor<Float>.random(shape: TensorShape(tokens, hidden), using: &rng)
        // Random weights in [0.5, 1.5] — also seeded for determinism.
        var wData = [Float](repeating: 0.0, count: hidden)
        for i in 0..<hidden { wData[i] = Float.random(in: 0.5...1.5, using: &rng) }
        let weight = Tensor<Float>(shape: TensorShape(hidden), data: wData)

        let gpuOut = try backend.rmsnorm(input, weight: weight, eps: 1e-5)
        let cpuOut = input.rmsNormCPU(weight: weight, epsilon: 1e-5)

        assertClose(gpuOut, cpuOut, tolerance: tolerance, context: "rmsnorm scaled [\(tokens)×\(hidden)]")
        print("RMSNorm scaled weights [\(tokens)×\(hidden)]: passed")
    }

    /// Large RMSNorm (typical Llama hidden_dim = 4096)
    func testRMSNormLlama() throws {
        guard MetalBackend.isAvailable else { throw XCTSkip("Metal not available") }
        let backend = try MetalBackend()

        let tokens = 32
        let hidden = 256   // Smaller than real Llama but exercises the reduction path
        let input  = randomTensor(shape: TensorShape(tokens, hidden))
        let weight = Tensor<Float>(shape: TensorShape(hidden), data: [Float](repeating: 1.0, count: hidden))

        let gpuOut = try backend.rmsnorm(input, weight: weight, eps: 1e-6)
        let cpuOut = input.rmsNormCPU(weight: weight, epsilon: 1e-6)

        assertClose(gpuOut, cpuOut, tolerance: tolerance, context: "rmsnorm large [\(tokens)×\(hidden)]")
        print("RMSNorm large [\(tokens)×\(hidden)]: passed")
    }

    // ── SiLU ───────────────────────────────────────────────────────────────

    /// SiLU on a small known vector
    func testSiLUBasic() throws {
        guard MetalBackend.isAvailable else { throw XCTSkip("Metal not available") }
        let backend = try MetalBackend()

        let data: [Float] = [-3.0, -1.0, 0.0, 1.0, 3.0]
        let input = Tensor<Float>(shape: TensorShape(data.count), data: data)

        let gpuOut = try backend.silu(input)
        let cpuOut = input.siluCPU()

        assertClose(gpuOut, cpuOut, tolerance: tolerance, context: "silu basic")

        // Verify SiLU(0) == 0 exactly
        XCTAssertEqual(gpuOut.rawData[2], 0.0, accuracy: 1e-6, "SiLU(0) should be 0")
        print("SiLU basic: GPU=\(gpuOut.rawData)")
    }

    /// SiLU on a large flat tensor (FFN gate projection size)
    func testSiLULarge() throws {
        guard MetalBackend.isAvailable else { throw XCTSkip("Metal not available") }
        let backend = try MetalBackend()

        let n = 4096
        let input = randomTensor(shape: TensorShape(n))

        let gpuOut = try backend.silu(input)
        let cpuOut = input.siluCPU()

        assertClose(gpuOut, cpuOut, tolerance: tolerance, context: "silu large n=\(n)")
        print("SiLU large (n=\(n)): passed")
    }

    /// SiLU output is always x * σ(x); for positive x it should be in (0, x]
    func testSiLUOutputRange() throws {
        guard MetalBackend.isAvailable else { throw XCTSkip("Metal not available") }
        let backend = try MetalBackend()

        var posData = [Float](repeating: 0.0, count: 64)
        for i in 0..<64 { posData[i] = Float(i) * 0.1 }
        let input = Tensor<Float>(shape: TensorShape(64), data: posData)

        let gpuOut = try backend.silu(input)

        for (i, v) in gpuOut.rawData.enumerated() {
            let x = posData[i]
            // For x >= 0: SiLU(x) is in [0, x]
            XCTAssertGreaterThanOrEqual(v, 0.0 - 1e-5, "SiLU should be non-negative for positive x at \(i)")
            XCTAssertLessThanOrEqual(v, x + 1e-5, "SiLU(x) should be <= x for positive x at \(i)")
        }
        print("SiLU output range check: passed")
    }

    // ── GELU ───────────────────────────────────────────────────────────────

    /// GELU on a small known vector
    func testGELUBasic() throws {
        guard MetalBackend.isAvailable else { throw XCTSkip("Metal not available") }
        let backend = try MetalBackend()

        let data: [Float] = [-3.0, -1.0, 0.0, 1.0, 3.0]
        let input = Tensor<Float>(shape: TensorShape(data.count), data: data)

        let gpuOut = try backend.gelu(input)
        let cpuOut = input.geluCPU()

        assertClose(gpuOut, cpuOut, tolerance: tolerance, context: "gelu basic")

        // GELU(0) == 0 exactly
        XCTAssertEqual(gpuOut.rawData[2], 0.0, accuracy: 1e-6, "GELU(0) should be 0")
        print("GELU basic: GPU=\(gpuOut.rawData)")
    }

    /// GELU on a large flat tensor
    ///
    /// **CHA-180:** Previously used an unseeded `Tensor.random` and a
    /// pure-relative tolerance, so Gaussian-tail samples (|x| ≳ 4) — where
    /// the tanh approximation yields CPU outputs on the order of 1e-4 to 1e-5
    /// — occasionally produced CI flakes as ULP-scale GPU/CPU differences
    /// got amplified by the `1/|c|` denominator. The test now uses a seeded
    /// RNG and a combined atol+rtol `assertClose`.
    func testGELULarge() throws {
        guard MetalBackend.isAvailable else { throw XCTSkip("Metal not available") }
        let backend = try MetalBackend()

        let n = 4096
        let input = randomTensor(shape: TensorShape(n))

        let gpuOut = try backend.gelu(input)
        let cpuOut = input.geluCPU()

        assertClose(gpuOut, cpuOut, tolerance: tolerance, context: "gelu large n=\(n)")
        print("GELU large (n=\(n)): passed")
    }

    /// GELU(x) ~ x for moderately large positive x (asymptotic behaviour)
    ///
    /// Note: the tanh approximation overflows Float32 for |x| > ~8 because
    /// tanh(√(2/π) * (x + 0.044715·x³)) becomes exp(very_large)/exp(very_large).
    /// We therefore test with x in [3, 6] where the approximation is valid and
    /// GELU(x) ≈ x (relative error < 1%).
    func testGELUAsymptotic() throws {
        guard MetalBackend.isAvailable else { throw XCTSkip("Metal not available") }
        let backend = try MetalBackend()

        // For moderate x > 0, GELU(x) ≈ x (tanh approximation is stable here)
        let data: [Float] = [3.0, 4.0, 5.0, 6.0]
        let input = Tensor<Float>(shape: TensorShape(data.count), data: data)
        let gpuOut = try backend.gelu(input)

        for (i, v) in gpuOut.rawData.enumerated() {
            let x = data[i]
            let relErr = abs(v - x) / x
            XCTAssertLessThan(relErr, 0.01, "GELU(\(x)) should be ~\(x) but got \(v)")
        }
        print("GELU asymptotic: passed")
    }

    // ── Tensor-level dispatch tests (via Tensor public API) ────────────────

    /// Verify that Tensor.softmax() dispatches correctly (GPU when available)
    func testTensorSoftmaxDispatch() throws {
        guard MetalBackend.isAvailable else { throw XCTSkip("Metal not available") }
        let backend = try MetalBackend()

        // Register the backend so TinyBrainBackend.metalBackend is set.
        // Use defer so the cleanup runs even if the test throws.
        let previous = TinyBrainBackend.metalBackend
        TinyBrainBackend.metalBackend = backend
        defer { TinyBrainBackend.metalBackend = previous }

        let input = randomTensor(shape: TensorShape(8, 32))
        let result = input.softmax()  // Should use GPU internally

        // Cross-check against direct CPU call
        let cpuRef = input.softmaxCPU()
        assertClose(result, cpuRef, tolerance: tolerance, context: "Tensor.softmax() dispatch")
        print("Tensor.softmax() dispatch: passed")
    }

    /// Verify that Tensor.gelu() dispatches correctly
    func testTensorGELUDispatch() throws {
        guard MetalBackend.isAvailable else { throw XCTSkip("Metal not available") }
        let backend = try MetalBackend()
        let previous = TinyBrainBackend.metalBackend
        TinyBrainBackend.metalBackend = backend
        defer { TinyBrainBackend.metalBackend = previous }

        let input = randomTensor(shape: TensorShape(512))
        let result = input.gelu()
        let cpuRef = input.geluCPU()
        assertClose(result, cpuRef, tolerance: tolerance, context: "Tensor.gelu() dispatch")
        print("Tensor.gelu() dispatch: passed")
    }

    /// Verify that Tensor.silu() dispatches correctly
    func testTensorSiLUDispatch() throws {
        guard MetalBackend.isAvailable else { throw XCTSkip("Metal not available") }
        let backend = try MetalBackend()
        let previous = TinyBrainBackend.metalBackend
        TinyBrainBackend.metalBackend = backend
        defer { TinyBrainBackend.metalBackend = previous }

        let input = randomTensor(shape: TensorShape(512))
        let result = input.silu()
        let cpuRef = input.siluCPU()
        assertClose(result, cpuRef, tolerance: tolerance, context: "Tensor.silu() dispatch")
        print("Tensor.silu() dispatch: passed")
    }

    /// Verify that Tensor.rmsNorm(weight:) dispatches correctly
    func testTensorRMSNormDispatch() throws {
        guard MetalBackend.isAvailable else { throw XCTSkip("Metal not available") }
        let backend = try MetalBackend()
        let previous = TinyBrainBackend.metalBackend
        TinyBrainBackend.metalBackend = backend
        defer { TinyBrainBackend.metalBackend = previous }

        let tokens = 4
        let hidden = 64
        let input  = randomTensor(shape: TensorShape(tokens, hidden))
        let weight = Tensor<Float>(shape: TensorShape(hidden), data: [Float](repeating: 1.0, count: hidden))

        let result = input.rmsNorm(weight: weight)
        let cpuRef = input.rmsNormCPU(weight: weight, epsilon: 1e-5)
        assertClose(result, cpuRef, tolerance: tolerance, context: "Tensor.rmsNorm() dispatch")
        print("Tensor.rmsNorm() dispatch: passed")
    }

    // ── CPU fallback (no Metal) tests ──────────────────────────────────────

    /// CPU-only softmax (always runs, validates the fallback path)
    func testSoftmaxCPUFallback() {
        let input = Tensor<Float>(shape: TensorShape(5), data: [2.0, 1.0, 0.1, -1.0, 3.0])
        let result = input.softmaxCPU()

        let sum = result.rawData.reduce(0, +)
        XCTAssertEqual(sum, 1.0, accuracy: 1e-5, "CPU softmax should sum to 1")
        for v in result.rawData {
            XCTAssertGreaterThan(v, 0.0, "CPU softmax values should be positive")
        }
        print("CPU softmax fallback: passed")
    }

    /// CPU-only RMSNorm (always runs)
    func testRMSNormCPUFallback() {
        let input  = Tensor<Float>(shape: TensorShape(1, 4), data: [1.0, 2.0, 3.0, 4.0])
        let weight = Tensor<Float>(shape: TensorShape(4), data: [1.0, 1.0, 1.0, 1.0])
        let result = input.rmsNormCPU(weight: weight, epsilon: 1e-5)

        // After RMSNorm the RMS of the output (before weight scaling) is ~1
        var ss: Float = 0.0
        for v in result.rawData { ss += v * v }
        let rms = sqrt(ss / 4.0)
        XCTAssertEqual(rms, 1.0, accuracy: 0.01, "CPU RMSNorm output should have RMS ≈ 1")
        print("CPU RMSNorm fallback: passed (output RMS=\(rms))")
    }

    /// CPU-only SiLU
    func testSiLUCPUFallback() {
        let input  = Tensor<Float>(shape: TensorShape(3), data: [-1.0, 0.0, 1.0])
        let result = input.siluCPU()

        // SiLU(-1) ≈ -0.269
        XCTAssertEqual(result.rawData[0], -1.0 / (1.0 + exp(1.0)), accuracy: 1e-5)
        XCTAssertEqual(result.rawData[1], 0.0, accuracy: 1e-6)
        // SiLU(1) ≈ 0.731
        XCTAssertEqual(result.rawData[2], 1.0 / (1.0 + exp(-1.0)), accuracy: 1e-5)
        print("CPU SiLU fallback: passed")
    }

    /// CPU-only GELU
    func testGELUCPUFallback() {
        let input  = Tensor<Float>(shape: TensorShape(3), data: [-1.0, 0.0, 1.0])
        let result = input.geluCPU()

        XCTAssertEqual(result.rawData[1], 0.0, accuracy: 1e-6, "GELU(0) == 0")
        XCTAssertLessThan(result.rawData[0], 0.0, "GELU(-1) should be negative")
        XCTAssertGreaterThan(result.rawData[2], 0.0, "GELU(1) should be positive")
        print("CPU GELU fallback: passed")
    }
}
