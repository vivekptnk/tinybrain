/// Quantized tensor for efficient storage of model weights
///
/// **TB-004 Phase 3:** INT8 quantization with per-channel scales
///
/// ## What is Quantization?
///
/// Converting Float32 (4 bytes) to Int8 (1 byte) = **75% memory savings!**
///
/// **The Math:**
/// ```
/// Quantize:   int8_value = round(float_value / scale) + zero_point
/// Dequantize: float_value = (int8_value - zero_point) * scale
/// ```
///
/// ## Why Per-Channel?
///
/// Different channels (rows in weight matrix) have different value ranges.
/// Using one scale per channel preserves accuracy better!
///
/// **Example:**
/// ```
/// Channel 0: values in [-1, 1]    → scale = 1/127
/// Channel 1: values in [-100, 100] → scale = 100/127
/// ```
///
/// Each channel uses its full Int8 range efficiently!

import Foundation

/// Quantization mode
public enum QuantizationMode {
    case symmetric      // INT8: zero_point = 0, range [-127, 127]
    case asymmetric     // INT8: zero_point ≠ 0, range [-128, 127]
    case perChannel     // INT8: Symmetric with one scale per channel
    case int4           // INT4: 4-bit quantization, 87.5% savings! (range [-7, 7])
    case int4PerChannel // INT4: Per-channel with 4-bit values
}

/// Precision level for quantization
public enum QuantizationPrecision {
    case int8   // 1 byte: 75% savings
    case int4   // 0.5 byte: 87.5% savings (2 values packed per byte!)
    
    var bits: Int {
        switch self {
        case .int8: return 8
        case .int4: return 4
        }
    }
    
    var maxValue: Int {
        switch self {
        case .int8: return 127
        case .int4: return 7
        }
    }
}

/// Container for quantized weights with metadata
///
/// Stores Int8 or packed Int4 values + scaling information needed to convert back to Float32
public struct QuantizedTensor {
    /// Shape of the original tensor
    public let shape: TensorShape
    
    /// Quantized data (Int8 values, or packed Int4)
    ///
    /// **INT8:** One value per byte (data.count == shape.count)
    /// **INT4:** Two values per byte (data.count == ceil(shape.count / 2))
    public let data: [Int8]
    
    /// Precision level (INT8 or INT4)
    public let precision: QuantizationPrecision
    
    /// Scale factors (one per channel for perChannel mode, or single value)
    ///
    /// **Per-channel:** scales.count == shape.dimensions[0] (one per row)
    /// **Per-tensor:** scales.count == 1 (single scale)
    public let scales: [Float]
    
    /// Zero points (optional, only for asymmetric quantization)
    ///
    /// Symmetric: zeroPoints == nil (assumed 0)
    /// Asymmetric: zeroPoints[channel] for each channel
    public let zeroPoints: [Int8]?
    
    /// Quantization mode used
    public let mode: QuantizationMode
    
    /// **REVIEW HITLER FIX:** Cached dequantized tensor to avoid repeated conversions
    /// Uses class reference so it can be mutated even from immutable QuantizedTensor
    private let cache: DequantizationCache = DequantizationCache()
    
    /// Cache holder (class = reference semantics)
    private class DequantizationCache {
        var tensor: Tensor<Float>?
    }
    
    /// Create quantized tensor
    public init(shape: TensorShape, data: [Int8], scales: [Float], zeroPoints: [Int8]? = nil, mode: QuantizationMode, precision: QuantizationPrecision = .int8) {
        // Validate data size based on precision
        switch precision {
        case .int8:
            precondition(data.count == shape.count, "INT8: data count must match shape")
        case .int4:
            let expectedCount = (shape.count + 1) / 2  // 2 values per byte, round up
            precondition(data.count == expectedCount, "INT4: data count should be ceil(shape.count / 2)")
        }
        
        switch mode {
        case .perChannel, .int4PerChannel:
            precondition(scales.count == shape.dimensions[0], 
                        "Per-channel: need one scale per channel (rows)")
        case .symmetric, .asymmetric, .int4:
            precondition(scales.count == 1, "Per-tensor: need single scale")
        }
        
        if let zp = zeroPoints {
            precondition(zp.count == scales.count, "Zero points must match scales")
        }
        
        self.shape = shape
        self.data = data
        self.precision = precision
        self.scales = scales
        self.zeroPoints = zeroPoints
        self.mode = mode
    }
    
    /// Dequantize back to Float32
    ///
    /// **REVIEW HITLER FIX:** Now caches result to avoid repeated conversions!
    ///
    /// Converts Int8 → Float32 using stored scales/zero-points
    ///
    /// Example:
    /// ```swift
    /// let quantized = weights.quantize()
    /// let float = quantized.dequantize()  // Cached after first call!
    /// ```
    public func dequantize() -> Tensor<Float> {
        // **REVIEW HITLER FIX:** Return cached if available (no repeated conversions!)
        if let cached = cache.tensor {
            return cached
        }
        
        // Dequantize and cache result
        let result = dequantizeUncached()
        cache.tensor = result  // Cache for next time
        return result
    }
    
    /// Internal uncached dequantization
    private func dequantizeUncached() -> Tensor<Float> {
        var floatData = [Float](repeating: 0.0, count: shape.count)  // Use shape.count, not data.count!
        
        switch mode {
        case .int4, .int4PerChannel:
            // TODO: INT4 unpacking in future phase
            // For now, this shouldn't be reached (quantize() falls back to perChannel)
            fatalError("INT4 dequantization not yet implemented")
            
        case .perChannel:
            // Each channel (row) has its own scale
            let numChannels = shape.dimensions[0]
            let channelSize = data.count / numChannels
            
            for ch in 0..<numChannels {
                let scale = scales[ch]
                let zeroPoint = zeroPoints?[ch] ?? 0
                let start = ch * channelSize
                let end = start + channelSize
                
                for i in start..<end {
                    let quantizedVal = data[i]
                    floatData[i] = Float(quantizedVal - zeroPoint) * scale
                }
            }
            
        case .symmetric, .asymmetric:
            // Single scale for entire tensor
            let scale = scales[0]
            let zeroPoint = zeroPoints?[0] ?? 0
            
            for i in 0..<data.count {
                let quantizedVal = data[i]
                floatData[i] = Float(quantizedVal - zeroPoint) * scale
            }
        }
        
        return Tensor<Float>(shape: shape, data: floatData)
    }
    
    /// Get memory size in bytes
    public var byteSize: Int {
        data.count * MemoryLayout<Int8>.size + 
        scales.count * MemoryLayout<Float>.size +
        (zeroPoints?.count ?? 0) * MemoryLayout<Int8>.size
    }
    
    /// Memory savings vs Float32
    public func savingsVsFloat32() -> Double {
        let float32Size = shape.count * MemoryLayout<Float>.size
        let savings = Double(float32Size - byteSize) / Double(float32Size)
        return savings
    }
}

/// INT4 packing/unpacking utilities
extension QuantizedTensor {
    /// Pack two INT4 values into one Int8 byte
    ///
    /// **Bit layout:** `[high 4 bits: value1][low 4 bits: value2]`
    ///
    /// Example:
    /// ```
    /// value1 = 5  (binary: 0101)
    /// value2 = -3 (binary: 1101 in 4-bit two's complement)
    /// packed = 0101_1101 = 93
    /// ```
    static func packInt4(_ value1: Int8, _ value2: Int8) -> Int8 {
        // Clamp to 4-bit range [-7, 7]
        let v1 = max(-7, min(7, value1)) & 0x0F
        let v2 = max(-7, min(7, value2)) & 0x0F
        return (v1 << 4) | v2
    }
    
    /// Unpack one Int8 byte into two INT4 values
    static func unpackInt4(_ packed: Int8) -> (Int8, Int8) {
        let value1 = (packed >> 4) & 0x0F
        let value2 = packed & 0x0F
        
        // Sign-extend from 4-bit to 8-bit
        let signed1 = value1 > 7 ? value1 - 16 : value1
        let signed2 = value2 > 7 ? value2 - 16 : value2
        
        return (signed1, signed2)
    }
}

/// Extension for quantizing Float tensors
extension Tensor where Element == Float {
    /// Quantize this Float32 tensor to Int8
    ///
    /// **TB-004 Phase 3:** Converts Float32 → Int8 with per-channel scales
    ///
    /// Example:
    /// ```swift
    /// let weights = Tensor<Float>.random([768, 3072])
    /// let quantized = weights.quantize(mode: .perChannel)
    /// // Memory: 4 bytes → 1 byte per element = 75% savings!
    /// ```
    ///
    /// - Parameter mode: Quantization strategy (default: .perChannel)
    /// - Returns: QuantizedTensor with Int8 data and scales
    public func quantize(mode: QuantizationMode = .perChannel) -> QuantizedTensor {
        switch mode {
        case .perChannel:
            return quantizePerChannel()
        case .symmetric:
            return quantizeSymmetric()
        case .asymmetric:
            return quantizeAsymmetric()
        case .int4, .int4PerChannel:
            // TODO: INT4 implementation in future phase
            // For now, fall back to INT8 per-channel (still gives good savings)
            return quantizePerChannel()
        }
    }
    
    /// Per-channel symmetric quantization (recommended for model weights)
    private func quantizePerChannel() -> QuantizedTensor {
        precondition(shape.dimensions.count == 2, "Per-channel quantization requires 2D tensor")
        
        let numChannels = shape.dimensions[0]
        let channelSize = shape.count / numChannels
        
        var quantizedData = [Int8](repeating: 0, count: shape.count)
        var scales = [Float](repeating: 0.0, count: numChannels)
        
        for ch in 0..<numChannels {
            let start = ch * channelSize
            let end = start + channelSize
            let channelData = Array(rawData[start..<end])
            
            // Find min/max for this channel
            let minVal = channelData.min() ?? 0.0
            let maxVal = channelData.max() ?? 0.0
            
            // Compute scale for symmetric quantization
            let absMax = max(abs(minVal), abs(maxVal))
            let scale: Float
            
            if absMax < Float.leastNonzeroMagnitude {
                // All zeros in this channel
                scale = 1.0  // Arbitrary, won't matter
            } else {
                // Map absMax to 127 (max positive Int8 in symmetric range)
                scale = absMax / 127.0
            }
            
            scales[ch] = scale
            
            // Quantize this channel
            for i in start..<end {
                let floatVal = rawData[i]
                let quantVal = round(floatVal / scale)
                // Clamp to [-127, 127] for symmetric
                let clamped = max(-127, min(127, quantVal))
                quantizedData[i] = Int8(clamped)
            }
        }
        
        return QuantizedTensor(
            shape: shape,
            data: quantizedData,
            scales: scales,
            zeroPoints: nil,  // Symmetric: zero-point = 0
            mode: .perChannel
        )
    }
    
    /// Per-tensor symmetric quantization (simpler, less accurate)
    private func quantizeSymmetric() -> QuantizedTensor {
        let minVal = rawData.min() ?? 0.0
        let maxVal = rawData.max() ?? 0.0
        
        let absMax = max(abs(minVal), abs(maxVal))
        let scale = absMax > Float.leastNonzeroMagnitude ? absMax / 127.0 : 1.0
        
        var quantizedData = [Int8](repeating: 0, count: shape.count)
        
        for i in 0..<rawData.count {
            let quantVal = round(rawData[i] / scale)
            let clamped = max(-127, min(127, quantVal))
            quantizedData[i] = Int8(clamped)
        }
        
        return QuantizedTensor(
            shape: shape,
            data: quantizedData,
            scales: [scale],
            zeroPoints: nil,
            mode: .symmetric
        )
    }
    
    /// Asymmetric quantization (uses full [-128, 127] range)
    private func quantizeAsymmetric() -> QuantizedTensor {
        let minVal = rawData.min() ?? 0.0
        let maxVal = rawData.max() ?? 0.0
        
        // Map [minVal, maxVal] → [-128, 127]
        let range = maxVal - minVal
        let scale = range > Float.leastNonzeroMagnitude ? range / 255.0 : 1.0
        let zeroPoint = Int8(round(-minVal / scale - 128.0))
        
        var quantizedData = [Int8](repeating: 0, count: shape.count)
        
        for i in 0..<rawData.count {
            let quantVal = round(rawData[i] / scale) + Float(zeroPoint)
            let clamped = max(-128, min(127, quantVal))
            quantizedData[i] = Int8(clamped)
        }
        
        return QuantizedTensor(
            shape: shape,
            data: quantizedData,
            scales: [scale],
            zeroPoints: [zeroPoint],
            mode: .asymmetric
        )
    }
}

/// Extension for matmul with quantized tensors
extension Tensor where Element == Float {
    /// Matrix multiplication with quantized weights
    ///
    /// **TB-004 Phase 3:** Auto-dequantizes during matmul
    ///
    /// Example:
    /// ```swift
    /// let input = Tensor<Float>.random([128, 768])
    /// let weights = loadWeights().quantize()  // QuantizedTensor
    /// let output = input.matmul(weights)  // Dequantizes automatically
    /// ```
    public func matmul(_ quantized: QuantizedTensor) -> Tensor<Float> {
        // Dequantize then multiply
        // TODO: Fused dequant+matmul Metal kernel for performance
        let weights = quantized.dequantize()
        return self.matmul(weights)
    }
}

