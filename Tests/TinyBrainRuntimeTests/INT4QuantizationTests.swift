/// Tests for INT4 (4-bit) quantization functionality
///
/// **Phase 1:** Validates per-group INT4 quantization with packed byte storage.
///
/// ## INT4 vs INT8 Comparison
///
/// | Precision | Bits | Bytes/element | Savings vs FP32 | Max value |
/// |-----------|------|---------------|-----------------|-----------|
/// | FP32      |  32  | 4.0           | 0%              | ~3.4e38   |
/// | INT8      |   8  | 1.0           | 75%             | 127       |
/// | INT4      |   4  | 0.5           | 87.5%           | 7         |
///
/// INT4 packs TWO values into each byte, halving storage vs INT8.

import XCTest
@testable import TinyBrainRuntime
import TinyBrainMetal

final class INT4QuantizationTests: XCTestCase {

    override func setUp() {
        super.setUp()
        // Initialize Metal backend for GPU kernel tests
        if TinyBrainBackend.metalBackend == nil {
            if MetalBackend.isAvailable {
                do {
                    TinyBrainBackend.metalBackend = try MetalBackend()
                } catch {
                    print("Metal init failed: \(error)")
                }
            }
        }
    }

    // MARK: - INT4 Pack/Unpack Tests

    func testPackUnpackRoundTrip() {
        // Verify that packing two INT4 values into a byte and unpacking
        // recovers the original values exactly
        let testPairs: [(Int8, Int8)] = [
            (0, 0), (7, -7), (-7, 7), (3, -3), (1, 2), (-1, -2), (5, -3)
        ]

        for (v1, v2) in testPairs {
            let packed = QuantizedTensor.packInt4(v1, v2)
            let (u1, u2) = QuantizedTensor.unpackInt4(packed)
            XCTAssertEqual(u1, v1, "Unpack high nibble: expected \(v1), got \(u1)")
            XCTAssertEqual(u2, v2, "Unpack low nibble: expected \(v2), got \(u2)")
        }
    }

    func testPackClampsBeyondRange() {
        // Values outside [-7, 7] should be clamped
        let packed = QuantizedTensor.packInt4(10, -10)
        let (u1, u2) = QuantizedTensor.unpackInt4(packed)
        XCTAssertEqual(u1, 7, "Should clamp 10 to 7")
        XCTAssertEqual(u2, -7, "Should clamp -10 to -7")
    }

    // MARK: - INT4 Quantize/Dequantize Round-Trip

    func testINT4QuantizeDequantizeRoundTrip() {
        // WHAT: Float32 -> INT4 -> Float32 round-trip
        // WHY: Verify quantization preserves values within acceptable error
        //
        // NOTE: INT4 has only 15 quantization levels (-7 to 7), so the element-wise
        // error for random normal data is naturally ~10-12%. This is expected!
        // In practice, perplexity degradation on real models is < 5% because:
        //   1. Model weights tend to cluster around zero
        //   2. Per-group quantization adapts to local distributions
        //   3. Errors tend to cancel during matmul accumulation

        let original = Tensor<Float>.random(shape: TensorShape(128, 256))
        let quantized = original.quantize(mode: .int4, groupSize: 128)
        let dequantized = quantized.dequantize()

        // Verify shapes match
        XCTAssertEqual(original.shape, dequantized.shape, "Shape must be preserved")

        // Verify precision metadata
        XCTAssertEqual(quantized.precision, .int4, "Precision should be INT4")
        XCTAssertEqual(quantized.mode, .int4, "Mode should be .int4")
        XCTAssertEqual(quantized.groupSize, 128, "Group size should be 128")

        // Verify packed data size (2 values per byte)
        let expectedPackedCount = (128 * 256 + 1) / 2
        XCTAssertEqual(quantized.data.count, expectedPackedCount,
                       "Packed data should have ceil(N/2) bytes")

        // INT4 round-trip error: ~12% for random normal data is expected
        // (15 levels vs 255 for INT8). Real model weights perform much better.
        let error = relativeError(original, dequantized)
        XCTAssertLessThan(error, 0.15,
                          "INT4 round-trip error should be < 15%, got \(error * 100)%")
    }

    func testINT4SmallGroupSize() {
        // Smaller group sizes should yield better accuracy
        // (more scale factors = finer granularity)
        let original = Tensor<Float>.random(shape: TensorShape(64, 128))

        let q32 = original.quantize(mode: .int4, groupSize: 32)
        let q128 = original.quantize(mode: .int4, groupSize: 128)

        let error32 = relativeError(original, q32.dequantize())
        let error128 = relativeError(original, q128.dequantize())

        // Both should be within INT4 expected range
        XCTAssertLessThan(error32, 0.15, "Group-32 error should be < 15%")
        XCTAssertLessThan(error128, 0.15, "Group-128 error should be < 15%")

        // Verify scale counts match expected groups
        let expectedGroups32 = (64 * 128 + 31) / 32
        let expectedGroups128 = (64 * 128 + 127) / 128
        XCTAssertEqual(q32.scales.count, expectedGroups32)
        XCTAssertEqual(q128.scales.count, expectedGroups128)
    }

    func testINT4PerChannelQuantization() {
        // Per-channel INT4: one scale per row
        let original = Tensor<Float>(shape: TensorShape(3, 4), data: [
            1.0, 2.0, 3.0, 4.0,           // Row 0: small values
            100.0, 200.0, 300.0, 400.0,    // Row 1: large values
            0.01, 0.02, 0.03, 0.04         // Row 2: tiny values
        ])

        let quantized = original.quantize(mode: .int4PerChannel)

        XCTAssertEqual(quantized.precision, .int4, "Precision should be INT4")
        XCTAssertEqual(quantized.scales.count, 3, "One scale per channel (row)")

        // Scales should differ for different magnitude rows
        XCTAssertNotEqual(quantized.scales[0], quantized.scales[1])
        XCTAssertNotEqual(quantized.scales[1], quantized.scales[2])

        // Round-trip check
        let dequantized = quantized.dequantize()
        XCTAssertEqual(dequantized.shape, original.shape)
    }

    // MARK: - Memory Savings Tests

    func testINT4MemorySavings() {
        // WHAT: INT4 should save ~87.5% vs Float32
        // WHY: Primary motivation for INT4 quantization
        //
        // Math: FP32 = 4 bytes/value, INT4 = 0.5 bytes/value
        //   Savings = (4 - 0.5) / 4 = 87.5%
        //   Plus small overhead for scales (1 float per 128 elements = ~3%)
        //   Net savings: ~85%+

        let tensor = Tensor<Float>.random(shape: TensorShape(1000, 1000))
        let quantized = tensor.quantize(mode: .int4, groupSize: 128)

        let float32Bytes = 1000 * 1000 * MemoryLayout<Float>.size  // 4,000,000 bytes
        let int4Bytes = quantized.byteSize

        let savings = Double(float32Bytes - int4Bytes) / Double(float32Bytes)

        // Should save at least 85% (87.5% minus scale overhead)
        XCTAssertGreaterThanOrEqual(savings, 0.85,
            "INT4 should save >= 85% vs FP32 (got \(savings * 100)%)")

        // INT4 should save significantly more than INT8
        let int8Quantized = tensor.quantize(mode: .perChannel)
        XCTAssertLessThan(quantized.byteSize, int8Quantized.byteSize,
            "INT4 should use less memory than INT8")
    }

    func testINT4VsINT8Savings() {
        // INT4 should be roughly half the size of INT8
        let tensor = Tensor<Float>.random(shape: TensorShape(512, 512))

        let int8 = tensor.quantize(mode: .perChannel)
        let int4 = tensor.quantize(mode: .int4, groupSize: 128)

        // INT4 data is half the size of INT8 data (2 values per byte vs 1)
        let int8DataBytes = int8.data.count
        let int4DataBytes = int4.data.count

        // INT4 packed data should be approximately half of INT8
        let ratio = Double(int4DataBytes) / Double(int8DataBytes)
        XCTAssertLessThan(ratio, 0.6,
            "INT4 data should be ~50% of INT8 data (got \(ratio * 100)%)")
    }

    // MARK: - INT4 MatMul Tests

    func testINT4MatMulCPUFallback() {
        // Test INT4 matmul via CPU dequantize path
        let input = Tensor<Float>.random(shape: TensorShape(4, 128))
        let weights = Tensor<Float>.random(shape: TensorShape(128, 64))

        // FP32 reference
        let fp32Result = input.matmul(weights)

        // INT4 quantized matmul
        let weightsINT4 = weights.quantize(mode: .int4, groupSize: 128)
        let int4Result = input.matmul(weightsINT4)

        // INT4 has only 15 quantization levels, so matmul error is higher than INT8.
        // In real models, errors cancel during accumulation. For random data, ~12% is expected.
        let error = relativeError(fp32Result, int4Result)
        XCTAssertLessThan(error, 0.15,
            "INT4 matmul error should be < 15% for random data, got \(error * 100)%")
    }

    func testINT4MatMulMetalKernel() {
        // Test INT4 matmul via Metal GPU kernel
        guard TinyBrainBackend.metalBackend != nil else {
            print("Skipping Metal INT4 test - no GPU available")
            return
        }

        let input = Tensor<Float>.random(shape: TensorShape(16, 256))
        let weights = Tensor<Float>.random(shape: TensorShape(256, 128))

        // FP32 reference (CPU)
        let savedBackend = TinyBrainBackend.metalBackend
        TinyBrainBackend.metalBackend = nil
        let fp32Result = input.matmul(weights)
        TinyBrainBackend.metalBackend = savedBackend

        // INT4 Metal matmul
        let weightsINT4 = weights.quantize(mode: .int4, groupSize: 128)
        let int4Result = input.matmul(weightsINT4)

        let error = relativeError(fp32Result, int4Result)
        XCTAssertLessThan(error, 0.15,
            "INT4 Metal matmul error should be < 15% for random data, got \(error * 100)%")
    }

    func testINT4MetalVsCPUConsistency() {
        // Verify Metal INT4 kernel produces same results as CPU dequantize path
        guard TinyBrainBackend.metalBackend != nil else {
            print("Skipping Metal consistency test - no GPU available")
            return
        }

        let input = Tensor<Float>.random(shape: TensorShape(8, 128))
        let weights = Tensor<Float>.random(shape: TensorShape(128, 64))
        let weightsINT4 = weights.quantize(mode: .int4, groupSize: 128)

        // CPU path: dequantize then matmul
        let dequantized = weightsINT4.dequantize()
        let cpuResult = input.matmulCPU(dequantized)

        // Metal path: fused INT4 dequant+matmul kernel
        let metalResult = input.matmul(weightsINT4)

        let error = relativeError(cpuResult, metalResult)
        XCTAssertLessThan(error, 0.01,
            "Metal and CPU INT4 results should match within 1%, got \(error * 100)%")
    }

    // MARK: - Edge Cases

    func testINT4AllZeros() {
        let zeros = Tensor<Float>.zeros(shape: TensorShape(64, 64))
        let quantized = zeros.quantize(mode: .int4, groupSize: 128)
        let dequantized = quantized.dequantize()

        for i in 0..<dequantized.rawData.count {
            XCTAssertEqual(dequantized.rawData[i], 0.0, accuracy: 1e-6,
                           "Zeros should stay zeros after INT4 round-trip")
        }
    }

    func testINT4NegativeValues() {
        let negative = Tensor<Float>.filled(shape: TensorShape(32, 32), value: -3.5)
        let quantized = negative.quantize(mode: .int4, groupSize: 128)
        let dequantized = quantized.dequantize()

        // INT4 has coarser granularity, allow more tolerance
        XCTAssertEqual(dequantized[0, 0], -3.5, accuracy: 1.0,
                       "Negative values should be approximately preserved")
        XCTAssertLessThan(dequantized[0, 0], 0,
                          "Negative values should remain negative")
    }

    func testINT4OddElementCount() {
        // Odd number of elements: last byte has only one value in high nibble
        let tensor = Tensor<Float>(shape: TensorShape(3, 3), data: [
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0
        ])
        let quantized = tensor.quantize(mode: .int4, groupSize: 128)

        // 9 elements -> ceil(9/2) = 5 packed bytes
        XCTAssertEqual(quantized.data.count, 5, "Odd element count rounds up")

        let dequantized = quantized.dequantize()
        XCTAssertEqual(dequantized.shape, tensor.shape)

        // Values should be approximately preserved
        let error = relativeError(tensor, dequantized)
        XCTAssertLessThan(error, 0.15,
            "Odd-count INT4 error should be reasonable, got \(error * 100)%")
    }

    func testINT4LargeMatrix() {
        // Real-world sized matrix (like TinyLlama weights)
        let weights = Tensor<Float>.random(shape: TensorShape(768, 3072))
        let quantized = weights.quantize(mode: .int4, groupSize: 128)

        // Verify packed size
        let expectedPacked = (768 * 3072 + 1) / 2
        XCTAssertEqual(quantized.data.count, expectedPacked)

        // Round-trip accuracy (INT4 has ~12% error for random normal data)
        let dequantized = quantized.dequantize()
        let error = relativeError(weights, dequantized)
        XCTAssertLessThan(error, 0.15,
            "Large matrix INT4 error should be < 15%, got \(error * 100)%")
    }

    // MARK: - Controlled Accuracy Test

    func testINT4AccuracyWithUniformData() {
        // Test with uniform-range data where INT4 quantization is well-behaved.
        // Values in a narrow range map cleanly to 15 levels.
        let data = (0..<256).map { Float($0) / 255.0 }  // [0, 1] range
        let tensor = Tensor<Float>(shape: TensorShape(16, 16), data: data)

        let quantized = tensor.quantize(mode: .int4, groupSize: 128)
        let dequantized = quantized.dequantize()

        // With uniform [0,1] data and 7 positive levels, step size is ~0.143
        // Max element-wise error is ~0.071 out of range 1.0 = ~7%
        let error = relativeError(tensor, dequantized)
        XCTAssertLessThan(error, 0.10,
            "INT4 with uniform data should have < 10% error, got \(error * 100)%")
    }

    func testINT4AccuracyWithConstantData() {
        // Constant data should quantize perfectly (one quantization level used)
        let constant = Tensor<Float>.filled(shape: TensorShape(64, 64), value: 5.0)
        let quantized = constant.quantize(mode: .int4, groupSize: 128)
        let dequantized = quantized.dequantize()

        let error = relativeError(constant, dequantized)
        XCTAssertLessThan(error, 0.001,
            "Constant data should round-trip nearly perfectly, got \(error * 100)%")
    }

    // MARK: - Precision Enum Tests

    func testPrecisionEnum() {
        XCTAssertEqual(QuantizationPrecision.int4.bits, 4)
        XCTAssertEqual(QuantizationPrecision.int4.maxValue, 7)
        XCTAssertEqual(QuantizationPrecision.int8.bits, 8)
        XCTAssertEqual(QuantizationPrecision.int8.maxValue, 127)
    }
}
