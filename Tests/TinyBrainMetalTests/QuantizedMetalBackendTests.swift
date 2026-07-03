import XCTest
@testable import TinyBrainMetal
@testable import TinyBrainRuntime

final class QuantizedMetalBackendTests: XCTestCase {
    private func assertClose(
        _ actual: Tensor<Float>,
        _ expected: Tensor<Float>,
        tolerance: Float,
        context: String,
        file: StaticString = #file,
        line: UInt = #line
    ) {
        XCTAssertEqual(actual.shape, expected.shape, "Shape mismatch in \(context)", file: file, line: line)
        let actualData = actual.rawData
        let expectedData = expected.rawData
        for i in 0..<expectedData.count {
            XCTAssertEqual(actualData[i], expectedData[i], accuracy: tolerance,
                           "\(context): index \(i), actual \(actualData[i]), expected \(expectedData[i])",
                           file: file, line: line)
        }
    }

    private func dequantizeINT4PerColumnReference(_ quantized: QuantizedTensor) -> Tensor<Float> {
        let rows = quantized.shape.dimensions[0]
        let cols = quantized.shape.dimensions[1]
        var values = [Float](repeating: 0, count: rows * cols)

        for row in 0..<rows {
            for col in 0..<cols {
                let linearIndex = row * cols + col
                let packed = UInt8(bitPattern: quantized.data[linearIndex / 2])
                let rawNibble = linearIndex.isMultiple(of: 2)
                    ? Int((packed >> 4) & 0x0F)
                    : Int(packed & 0x0F)
                let signed = rawNibble > 7 ? rawNibble - 16 : rawNibble
                let zeroPoint = Int(quantized.zeroPoints?[col] ?? 0)
                values[linearIndex] = Float(signed - zeroPoint) * quantized.scales[col]
            }
        }

        return Tensor<Float>(shape: quantized.shape, data: values)
    }

    func testINT4PerChannelMatmulMatchesPerColumnCPUReference() throws {
        guard MetalBackend.isAvailable else { throw XCTSkip("Metal not available") }

        let backend = try MetalBackend()
        let input = Tensor<Float>(shape: TensorShape(2, 3), data: [
            1.0, 1.0, 1.0,
            -2.0, 0.5, 3.0
        ])
        let weights = Tensor<Float>(shape: TensorShape(3, 4), data: [
            1.0, 100.0, 0.01, -1_000.0,
            2.0, 200.0, 0.02, -2_000.0,
            3.0, 300.0, 0.03, -3_000.0
        ])
        let quantized = weights.quantize(mode: .int4PerChannel)
        let scaleRatio = (quantized.scales.max() ?? 0) / max(quantized.scales.min() ?? 1, Float.leastNonzeroMagnitude)
        XCTAssertGreaterThan(scaleRatio, 10_000, "Test fixture must have per-column scales that differ by orders of magnitude")

        let cpuReference = input.matmulCPU(dequantizeINT4PerColumnReference(quantized))
        let gpuResult = try backend.matmulQuantized(input, quantized)

        assertClose(gpuResult, cpuReference, tolerance: 1e-4, context: "INT4 per-channel matmul")
    }

    func testINT4PerGroupMatmulReusesCachedBuffers() throws {
        guard MetalBackend.isAvailable else { throw XCTSkip("Metal not available") }

        let backend = try MetalBackend()
        let input = Tensor<Float>.random(shape: TensorShape(4, 8))
        let weights = Tensor<Float>.random(shape: TensorShape(8, 6))
        let quantized = weights.quantize(mode: .int4, groupSize: 8)

        XCTAssertEqual(backend.int4QuantizedBufferCacheCount, 0)
        XCTAssertEqual(backend.int4QuantizedBufferUploadCount, 0)

        _ = try backend.matmulQuantized(input, quantized)
        XCTAssertEqual(backend.int4QuantizedBufferCacheCount, 1)
        XCTAssertEqual(backend.int4QuantizedBufferUploadCount, 1)

        _ = try backend.matmulQuantized(input, quantized)
        XCTAssertEqual(backend.int4QuantizedBufferCacheCount, 1,
                       "Second matmul with the same QuantizedTensor should reuse the existing INT4 cache entry")
        XCTAssertEqual(backend.int4QuantizedBufferUploadCount, 1,
                       "Second matmul with the same QuantizedTensor must not upload INT4 buffers again")
    }
}
