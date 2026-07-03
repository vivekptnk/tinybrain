import XCTest
@testable import TinyBrainRuntime

final class AsymmetricQuantizationZeroPointTests: XCTestCase {
    func testAsymmetricQuantizationPositiveRangeSaturatesZeroPoint() {
        // WHAT: All-positive activations quantize without an Int8 zero-point trap.
        // WHY: Post-ReLU ranges do not include zero, so the ideal zero-point is below Int8.min.
        // HOW: The representable asymmetric range expands to include zero and saturates at -128.
        assertAsymmetricRoundTrip(
            values: [0.5, 1.0, 2.0, 4.0],
            expectedZeroPoint: -128
        )
    }

    func testAsymmetricQuantizationNegativeRangeSaturatesZeroPoint() {
        // WHAT: All-negative tensors quantize without an Int8 zero-point trap.
        // WHY: Negative-only ranges put the ideal zero-point above Int8.max.
        // HOW: The representable asymmetric range expands to include zero and saturates at 127.
        assertAsymmetricRoundTrip(
            values: [-4.0, -2.0, -1.0, -0.5],
            expectedZeroPoint: 127
        )
    }

    func testAsymmetricQuantizationDegenerateAndMixedRangesDoNotTrap() {
        // WHAT: Degenerate, zero, and mixed-sign tensors remain stable.
        // WHY: Single-value tensors have min == max, and mixed tensors are the normal regression path.
        // HOW: Each case round-trips within one asymmetric quantization step.
        assertAsymmetricRoundTrip(values: [3.25], expectedZeroPoint: -128)
        assertAsymmetricRoundTrip(values: [-3.25], expectedZeroPoint: 127)
        assertAsymmetricRoundTrip(values: [0.0, 0.0, 0.0], expectedZeroPoint: -128)
        assertAsymmetricRoundTrip(values: [-2.0, -0.25, 0.0, 0.75, 3.0])
    }

    private func assertAsymmetricRoundTrip(
        values: [Float],
        expectedZeroPoint: Int8? = nil,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        let original = Tensor<Float>(shape: TensorShape(values.count), data: values)
        let quantized = original.quantize(mode: .asymmetric)
        let dequantized = quantized.dequantize()

        XCTAssertEqual(quantized.mode, .asymmetric, file: file, line: line)
        XCTAssertEqual(quantized.scales.count, 1, file: file, line: line)
        XCTAssertGreaterThan(quantized.scales[0], 0, file: file, line: line)
        XCTAssertEqual(quantized.zeroPoints?.count, 1, file: file, line: line)

        if let expectedZeroPoint {
            XCTAssertEqual(quantized.zeroPoints?[0], expectedZeroPoint, file: file, line: line)
        }

        let scale = quantized.scales[0]
        for (index, expected) in values.enumerated() {
            let actual = dequantized.rawData[index]
            XCTAssertEqual(
                actual,
                expected,
                accuracy: scale + 1e-6,
                "Expected \(expected) to round-trip within one quantization step (\(scale)); got \(actual)",
                file: file,
                line: line
            )
        }
    }
}
