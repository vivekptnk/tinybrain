import XCTest
@testable import TinyBrainRuntime

/// Hermetic perf-regression tests for the quantized linear path.
///
/// These do NOT assert wall-clock time (brittle in CI). Instead they assert the
/// *structural* property that produced the >11-minute RAG demo: the generation
/// linear path must never re-materialize an entire quantized weight matrix to
/// Float32. We verify this with debug counters (`QuantizedMatmulStats`) and a
/// numerical-equivalence check against the reference dequantize path.
///
/// All tests force the CPU path (Metal backend nil) so they exercise the
/// streaming quantized kernels deterministically regardless of host GPU state.
final class QuantizedMatmulPerfTests: XCTestCase {

    private var savedBackend: Any?

    override func setUp() {
        super.setUp()
        // Force the CPU quantized path: the streaming kernel only runs when the
        // Metal backend is unavailable (exactly the tinybrain-rag configuration).
        savedBackend = TinyBrainBackend.metalBackend
        TinyBrainBackend.metalBackend = nil
        QuantizedMatmulStats.reset()
    }

    override func tearDown() {
        TinyBrainBackend.metalBackend = savedBackend
        QuantizedMatmulStats.reset()
        super.tearDown()
    }

    // MARK: - Streaming path correctness

    /// The streaming INT8 kernel must be numerically equivalent to the reference
    /// `dequantize()` + CPU matmul, and must NOT take the full-dequant fallback.
    func testStreamingINT8MatchesDequantReference() {
        var rng: any RandomNumberGenerator = SeededGenerator(seed: 7)
        let input = Tensor<Float>.random(shape: TensorShape(1, 256),
                                         mean: 0, std: 1.0, using: &rng)
        let weightsF = Tensor<Float>.random(shape: TensorShape(256, 512),
                                            mean: 0, std: 0.05, using: &rng)
        let q = weightsF.quantize(mode: .perChannel)   // per-channel INT8, zero-point 0

        QuantizedMatmulStats.reset()
        let streamed = input.matmul(q)                 // fast streaming path
        let reference = input.matmulCPU(q.dequantize()) // reference path

        // Fast path handled it: zero fallbacks, one streaming matmul.
        XCTAssertEqual(QuantizedMatmulStats.fullDequantFallbackCount, 0,
                       "Streaming INT8 path must not fall back to full FP32 materialization")
        XCTAssertEqual(QuantizedMatmulStats.streamingINT8Count, 1,
                       "Streaming INT8 path should have handled the matmul")

        // Numerical equivalence (float reassociation only — well inside quant error).
        XCTAssertEqual(streamed.shape, reference.shape)
        let err = relativeError(reference, streamed)
        XCTAssertLessThan(err, 1e-4,
                          "Streaming INT8 must match dequantize reference (rel err \(err))")
    }

    /// Multi-row (M > 1) inputs must also route through the streaming path.
    func testStreamingINT8HandlesMultiRow() {
        var rng: any RandomNumberGenerator = SeededGenerator(seed: 11)
        let input = Tensor<Float>.random(shape: TensorShape(4, 128),
                                         mean: 0, std: 1.0, using: &rng)
        let weightsF = Tensor<Float>.random(shape: TensorShape(128, 96),
                                            mean: 0, std: 0.05, using: &rng)
        let q = weightsF.quantize(mode: .perChannel)

        QuantizedMatmulStats.reset()
        let streamed = input.matmul(q)
        let reference = input.matmulCPU(q.dequantize())

        XCTAssertEqual(QuantizedMatmulStats.fullDequantFallbackCount, 0)
        XCTAssertEqual(QuantizedMatmulStats.streamingINT8Count, 1)
        XCTAssertLessThan(relativeError(reference, streamed), 1e-4)
    }

    // MARK: - Generation path never re-materializes full weights

    /// The whole generation loop over a toy INT8 model must never materialize
    /// full FP32 weight matrices — the property whose absence made real-model
    /// generation take minutes per prefill.
    func testGenerationLoopNeverFullyDequantizes() {
        let config = ModelConfig(
            numLayers: 3,
            hiddenDim: 64,
            numHeads: 8,
            vocabSize: 128,
            maxSeqLen: 64
        )
        // makeToyModel quantizes every projection to per-channel INT8.
        let weights = ModelWeights.makeToyModel(config: config, seed: 99)
        let runner = ModelRunner(weights: weights)

        QuantizedMatmulStats.reset()

        // Simulate a small "prefill + decode": feed several tokens through step().
        let tokens = [1, 5, 9, 13, 17, 21, 25, 29]
        for t in tokens {
            _ = runner.step(tokenId: t)
        }

        // Every linear (attention q/k/v/o, ffn up/down, output projection) is
        // per-channel INT8 → must be handled by the streaming path.
        XCTAssertEqual(QuantizedMatmulStats.fullMatrixDequantizeCount, 0,
                       "Generation linear path must never re-materialize full INT8 weights to FP32")
        XCTAssertGreaterThan(QuantizedMatmulStats.streamingINT8Count, 0,
                             "Streaming INT8 path should have been exercised during generation")
    }

    // MARK: - INT4 streaming

    /// INT4 per-group weights must also avoid full-matrix dequantization.
    func testStreamingINT4MatchesDequantReference() {
        var rng: any RandomNumberGenerator = SeededGenerator(seed: 3)
        let input = Tensor<Float>.random(shape: TensorShape(1, 128),
                                         mean: 0, std: 1.0, using: &rng)
        let weightsF = Tensor<Float>.random(shape: TensorShape(128, 128),
                                            mean: 0, std: 0.05, using: &rng)
        let q = weightsF.quantize(mode: .int4, groupSize: 32)

        QuantizedMatmulStats.reset()
        let out = input.matmul(q)
        let reference = input.matmulCPU(q.dequantize())

        XCTAssertEqual(QuantizedMatmulStats.fullMatrixDequantizeCount, 0,
                       "INT4 streaming path must not materialize the full FP32 matrix")
        XCTAssertEqual(QuantizedMatmulStats.streamingINT8Count, 0)
        XCTAssertEqual(QuantizedMatmulStats.streamingINT4Count, 1)
        XCTAssertLessThan(relativeError(reference, out), 1e-4,
                          "INT4 streaming path must match the reference dequant path")
    }
}
