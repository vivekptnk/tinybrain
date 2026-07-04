import XCTest
@testable import TinyBrainRuntime

final class RoPEParityTests: XCTestCase {

    func testApplyRoPEMatchesHFRotateHalfReference() {
        let headDim = 8
        let rotaryDims = 8
        let position = 3
        let input: [Float] = [0.25, -1.5, 2.0, -3.25, 4.5, -5.75, 6.125, -7.0]

        let runner = ModelRunner(config: ModelConfig(numLayers: 1,
                                                     hiddenDim: headDim,
                                                     numHeads: 1,
                                                     vocabSize: 16))

        let actual = runner.applyRoPE(input,
                                      headDim: headDim,
                                      numHeads: 1,
                                      position: position,
                                      rotaryDims: rotaryDims)
        let hfReference = hfRotateHalfReference(input,
                                                headDim: headDim,
                                                numHeads: 1,
                                                position: position,
                                                rotaryDims: rotaryDims)
        let interleavedReference = interleavedPairReference(input,
                                                           headDim: headDim,
                                                           numHeads: 1,
                                                           position: position,
                                                           rotaryDims: rotaryDims)

        XCTAssertGreaterThan(maxAbsoluteDelta(hfReference, interleavedReference), 1.0,
                             "The tiny fixture must distinguish rotate-half from adjacent-pair RoPE")

        for index in 0..<input.count {
            XCTAssertEqual(actual[index], hfReference[index], accuracy: 1e-5,
                           "applyRoPE must match HF rotate-half at index \(index); " +
                           "actual=\(actual[index]), hf=\(hfReference[index]), " +
                           "interleaved=\(interleavedReference[index])")
        }
    }

    func testApplyRoPEMatchesHFRotateHalfReferenceWithQwenTheta() {
        let headDim = 8
        let rotaryDims = 8
        let position = 5
        let input: [Float] = [0.5, -1.0, 2.0, -4.0, 1.5, -2.5, 3.0, -3.5]

        let runner = ModelRunner(config: ModelConfig(numLayers: 1,
                                                     hiddenDim: headDim,
                                                     numHeads: 1,
                                                     vocabSize: 16,
                                                     ropeTheta: 1_000_000.0))

        let actual = runner.applyRoPE(input,
                                      headDim: headDim,
                                      numHeads: 1,
                                      position: position,
                                      rotaryDims: rotaryDims)

        // Hand-computed HF rotate-half reference for theta=1e6:
        // angles = [5, 0.15811387, 0.0050000004, 0.0001581139].
        let expected: [Float] = [
             1.580217600,
            -0.593886316,
             1.984974980,
            -3.999446630,
            -0.053968847,
            -2.626271009,
             3.009962320,
            -3.500632524,
        ]

        for index in 0..<input.count {
            XCTAssertEqual(actual[index], expected[index], accuracy: 1e-5,
                           "applyRoPE must honor config.ropeTheta=1e6 at index \(index)")
        }

        let defaultThetaReference = hfRotateHalfReference(input,
                                                          headDim: headDim,
                                                          numHeads: 1,
                                                          position: position,
                                                          rotaryDims: rotaryDims,
                                                          ropeTheta: 10000.0)
        XCTAssertGreaterThan(maxAbsoluteDelta(actual, defaultThetaReference), 0.02,
                             "Qwen theta=1e6 fixture must distinguish config-driven RoPE from the default 10000 base")
    }

    private func hfRotateHalfReference(_ input: [Float],
                                       headDim: Int,
                                       numHeads: Int,
                                       position: Int,
                                       rotaryDims: Int,
                                       ropeTheta: Float = 10000.0) -> [Float] {
        var output = input
        let halfRotaryDims = rotaryDims / 2

        for head in 0..<numHeads {
            let offset = head * headDim
            for d in 0..<halfRotaryDims {
                let frequency = pow(ropeTheta, -Float(2 * d) / Float(rotaryDims))
                let angle = Float(position) * frequency
                let cosAngle = cos(angle)
                let sinAngle = sin(angle)

                let first = input[offset + d]
                let second = input[offset + d + halfRotaryDims]

                output[offset + d] = first * cosAngle - second * sinAngle
                output[offset + d + halfRotaryDims] = first * sinAngle + second * cosAngle
            }
        }

        return output
    }

    private func interleavedPairReference(_ input: [Float],
                                          headDim: Int,
                                          numHeads: Int,
                                          position: Int,
                                          rotaryDims: Int,
                                          ropeTheta: Float = 10000.0) -> [Float] {
        var output = input

        for head in 0..<numHeads {
            let offset = head * headDim
            for i in stride(from: 0, to: rotaryDims, by: 2) {
                let frequency = pow(ropeTheta, -Float(i) / Float(rotaryDims))
                let angle = Float(position) * frequency
                let cosAngle = cos(angle)
                let sinAngle = sin(angle)

                let first = input[offset + i]
                let second = input[offset + i + 1]

                output[offset + i] = first * cosAngle - second * sinAngle
                output[offset + i + 1] = first * sinAngle + second * cosAngle
            }
        }

        return output
    }

    private func maxAbsoluteDelta(_ lhs: [Float], _ rhs: [Float]) -> Float {
        zip(lhs, rhs).map { abs($0 - $1) }.max() ?? 0
    }
}
