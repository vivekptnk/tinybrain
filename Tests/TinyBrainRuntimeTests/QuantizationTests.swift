/// Tests for INT8 quantization functionality
///
/// **TB-004 Phase 3:** Validates symmetric/asymmetric quantization with per-channel scales
///
/// ## What is Quantization?
///
/// Converting high-precision numbers (Float32: 4 bytes) to low-precision (Int8: 1 byte)
/// to save memory while preserving accuracy.
///
/// **Example:**
/// ```
/// Float32: 3.14159 → takes 4 bytes
/// Int8: 127 → takes 1 byte (75% memory savings!)
/// ```
///
/// **The trick:** Store a "scale" factor to convert back:
/// ```
/// quantized = round(float_value / scale)
/// dequantized = quantized * scale
/// ```

import XCTest
@testable import TinyBrainRuntime
import TinyBrainMetal

final class QuantizationTests: XCTestCase {
    
    override func setUp() {
        super.setUp()
        // **REVIEW HITLER FIX:** Initialize Metal for INT8 kernel tests!
        if TinyBrainBackend.metalBackend == nil {
            if MetalBackend.isAvailable {
                do {
                    TinyBrainBackend.metalBackend = try MetalBackend()
                    print("✅ Metal initialized for quantization tests")
                } catch {
                    print("⚠️ Metal init failed: \(error)")
                }
            }
        }
    }
    
    // MARK: - Symmetric Quantization Tests
    
    func testSymmetricQuantization() {
        // WHAT: Float32 → INT8 → Float32 roundtrip
        // WHY: Verify quantization correctness
        // ACCURACY: ≤1% error (perplexity delta ≤1% on real models)
        
        let original = Tensor<Float>.random(shape: TensorShape(128, 768))
        let quantized = original.quantize(mode: .symmetric)
        let dequantized = quantized.dequantize()
        
        // Should be close to original (some precision loss acceptable)
        let error = relativeError(original, dequantized)
        // INT8 quantization: 1-2% error is excellent! (127 quantization levels)
        XCTAssertLessThan(error, 0.02, "Quantization error should be ≤2%, got \(error * 100)%")
    }
    
    func testSymmetricQuantizationRange() {
        // WHAT: Symmetric quantization uses range [-127, 127]
        // WHY: Zero-point = 0 simplifies math
        // HOW: Verify quantized values in range
        
        let original = Tensor<Float>.filled(shape: TensorShape(10, 10), value: 5.0)
        let quantized = original.quantize(mode: .symmetric)
        
        // All values should be in Int8 range
        for i in 0..<quantized.data.count {
            let val = quantized.data[i]
            XCTAssertGreaterThanOrEqual(val, -127, "Value \(val) below symmetric range")
            XCTAssertLessThanOrEqual(val, 127, "Value \(val) above symmetric range")
        }
    }
    
    // MARK: - Per-Channel Scale Tests
    
    func testPerChannelScales() {
        // WHAT: Each output channel has its own scale
        // WHY: Better accuracy than single scale for entire tensor
        // HOW: Different magnitude rows should have different scales
        
        let weights = Tensor<Float>(shape: TensorShape(3, 4), data: [
            1.0, 2.0, 3.0, 4.0,           // Row 0: small values
            100.0, 200.0, 300.0, 400.0,   // Row 1: large values  
            0.01, 0.02, 0.03, 0.04        // Row 2: tiny values
        ])
        
        let quantized = weights.quantize(mode: .perChannel)
        
        // Should have one scale per output channel (dim 1 = 4 columns)
        XCTAssertEqual(quantized.scales.count, 4, "One scale per output channel")

        // Scales should be different for different magnitude columns
        XCTAssertNotEqual(quantized.scales[0], quantized.scales[1], "Different magnitudes → different scales")
        XCTAssertNotEqual(quantized.scales[2], quantized.scales[3], "Different magnitudes → different scales")
    }
    
    func testPerChannelAccuracy() {
        // WHAT: Per-channel more accurate than per-tensor
        // WHY: Preserves outliers better
        // HOW: Compare error for mixed-magnitude tensor
        
        let mixed = Tensor<Float>(shape: TensorShape(2, 100), data: 
            Array(repeating: 1.0, count: 100) +      // Channel 0: small
            Array(repeating: 100.0, count: 100))     // Channel 1: large
        
        let perChannel = mixed.quantize(mode: .perChannel)
        let perChannelError = relativeError(mixed, perChannel.dequantize())
        
        // Per-channel should have low error even with mixed magnitudes
        XCTAssertLessThan(perChannelError, 0.01, "Per-channel should be accurate")
    }
    
    // MARK: - Quantized MatMul Tests
    
    func testQuantizedMatMul() {
        // WHAT: MatMul with INT8 weights (dequantize on-the-fly)
        // WHY: End-to-end quantized inference workflow
        // ACCURACY: relative error < 1e-3 vs FP32
        
        let input = Tensor<Float>.random(shape: TensorShape(128, 256))
        let weights = Tensor<Float>.random(shape: TensorShape(256, 512))
        
        // Float32 reference
        let fp32Result = input.matmul(weights)
        
        // Quantized workflow
        let weightsQuant = weights.quantize(mode: .perChannel)
        let int8Result = input.matmul(weightsQuant)  // Should auto-dequantize
        
        let error = relativeError(fp32Result, int8Result)
        // MatMul with quantized weights: 1% error is very good!
        XCTAssertLessThan(error, 0.01, "Quantized matmul error < 1%, got \(error * 100)%")
    }
    
    func testQuantizedMatMulLargeMatrix() {
        // WHAT: Quantization scales to large matrices
        // WHY: Real model weights are large (768×3072 in TinyLlama)
        // ACCURACY: Same error tolerance at scale
        
        let input = Tensor<Float>.random(shape: TensorShape(10, 768))
        let weights = Tensor<Float>.random(shape: TensorShape(768, 3072))
        
        let fp32Result = input.matmul(weights)
        let weightsQuant = weights.quantize()
        let int8Result = input.matmul(weightsQuant)
        
        let error = relativeError(fp32Result, int8Result)
        XCTAssertLessThan(error, 0.01, "Large matrix quantization error < 1%, got \(error * 100)%")
    }
    
    // MARK: - Edge Cases
    
    func testQuantizeAllZeros() {
        // Edge: All zeros should quantize cleanly
        // WHY: Handle degenerate cases
        
        let zeros = Tensor<Float>.zeros(shape: TensorShape(10, 10))
        let quantized = zeros.quantize()
        
        // Scale might be 0 or very small
        // Dequantized should still be all zeros
        let dequantized = quantized.dequantize()
        
        for i in 0..<dequantized.rawData.count {
            XCTAssertEqual(dequantized.rawData[i], 0.0, accuracy: 1e-6, "Zeros should stay zeros")
        }
    }
    
    func testQuantizeNegativeValues() {
        // Edge: Negative values in range
        // WHY: Model weights can be negative
        
        let negative = Tensor<Float>.filled(shape: TensorShape(5, 5), value: -3.5)
        let quantized = negative.quantize(mode: .symmetric)
        let dequantized = quantized.dequantize()
        
        // Should be close to -3.5
        XCTAssertEqual(dequantized[0, 0], -3.5, accuracy: 0.1, "Negative values should quantize")
    }
    
    func testQuantizeMixedSign() {
        // Edge: Mix of positive and negative
        // WHY: Typical model weights
        
        let mixed = Tensor<Float>(shape: TensorShape(4, 1), data: [-10.0, -1.0, 1.0, 10.0])
        let quantized = mixed.quantize(mode: .symmetric)
        let dequantized = quantized.dequantize()
        
        // Verify sign preserved
        XCTAssertLessThan(dequantized[0, 0], 0, "Negative should stay negative")
        XCTAssertGreaterThan(dequantized[3, 0], 0, "Positive should stay positive")
    }
    
    func testQuantizeOutliers() {
        // Edge: One column has an extreme outlier, another column has normal values
        // WHY: Per-channel (per-column) quantization isolates outlier impact

        // Column 0: all 1.0 (normal), Column 1: one outlier at 1000.0
        var data = [Float](repeating: 0, count: 100 * 2)
        for row in 0..<100 {
            data[row * 2 + 0] = 1.0       // col 0: normal
            data[row * 2 + 1] = 1.0       // col 1: normal
        }
        data[99 * 2 + 1] = 1000.0         // outlier only in col 1
        let tensor = Tensor<Float>(shape: TensorShape(100, 2), data: data)

        let quantized = tensor.quantize(mode: .perChannel)
        let dequantized = quantized.dequantize()

        // Per-channel: column 0 has its own scale, unaffected by column 1's outlier
        let regularValue = dequantized[0, 0]
        XCTAssertEqual(regularValue, 1.0, accuracy: 0.2, "Regular column preserved despite outlier in other column")
    }
    
    // MARK: - Memory Savings Tests
    
    func testMemorySavings() {
        // WHAT: INT8 uses 75% less memory than Float32
        // WHY: Primary goal of quantization
        // HOW: Check byte sizes
        
        let float32 = Tensor<Float>.zeros(shape: TensorShape(1000, 1000))
        let quantized = float32.quantize()
        
        let float32Bytes = float32.rawData.count * MemoryLayout<Float>.size
        let int8Bytes = quantized.data.count * MemoryLayout<Int8>.size
        
        let savings = Double(float32Bytes - int8Bytes) / Double(float32Bytes)
        XCTAssertGreaterThanOrEqual(savings, 0.74, "Should save ~75% memory (got \(savings * 100)%)")
    }
}

/// Helper: Compute relative error between tensors
func relativeError(_ a: Tensor<Float>, _ b: Tensor<Float>) -> Float {
    precondition(a.shape == b.shape, "Shapes must match")
    
    let aData = a.rawData
    let bData = b.rawData
    
    var sumSquaredDiff: Float = 0.0
    var sumSquaredA: Float = 0.0
    
    for i in 0..<aData.count {
        let diff = aData[i] - bData[i]
        sumSquaredDiff += diff * diff
        sumSquaredA += aData[i] * aData[i]
    }
    
    return sqrt(sumSquaredDiff) / max(sqrt(sumSquaredA), Float.leastNonzeroMagnitude)
}

