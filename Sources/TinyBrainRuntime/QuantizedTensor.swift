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
    /// Stable identifier used for GPU cache lookups
    ///
    /// INT8 buffers are immutable after creation, so we can safely upload them
    /// to the GPU once and reuse the same Metal buffers for the lifetime of
    /// the tensor. Whenever a quantized tensor is (re)created we stamp it with
    /// a new UUID so caches can differentiate between different blobs that
    /// might share the same shape or data length.
    public let identifier: UUID
    
    /// Shape of the original tensor
    public let shape: TensorShape
    
    /// Quantized data (Int8 values, or packed Int4)
    ///
    /// **INT8:** One value per byte (data.count == shape.count)
    /// **INT4:** Two values per byte (data.count == ceil(shape.count / 2))
    public let data: [Int8]
    
    /// Precision level (INT8 or INT4)
    public let precision: QuantizationPrecision
    
    /// Scale factors (one per channel for perChannel mode, per group for INT4, or single value)
    ///
    /// **Per-channel:** scales.count == shape.dimensions[0] (one per row)
    /// **Per-group INT4:** scales.count == totalElements / groupSize (one per group)
    /// **Per-tensor:** scales.count == 1 (single scale)
    public let scales: [Float]

    /// Group size for per-group quantization (INT4 uses 128 by default)
    ///
    /// **Why groups?** INT4 has only 16 levels (-7 to 7), so per-tensor
    /// quantization loses too much precision. Grouping 128 consecutive
    /// elements shares one scale factor, balancing accuracy vs overhead.
    ///
    /// Group size of 0 means not using per-group quantization.
    public let groupSize: Int

    /// Zero points (optional, only for asymmetric or INT4 per-group quantization)
    ///
    /// Symmetric: zeroPoints == nil (assumed 0)
    /// Asymmetric: zeroPoints[channel] for each channel
    /// INT4 per-group: zeroPoints[group] for each group (stored as INT4 in practice)
    public let zeroPoints: [Int8]?

    /// Quantization mode used
    public let mode: QuantizationMode

    /// **REVIEW HITLER FIX COMPLETE:** NO cache needed!
    /// We have INT8/INT4 Metal kernels that compute directly from quantized data.
    /// Memory: Just the quantized data - no Float32 materialization!

    /// Create quantized tensor
    public init(shape: TensorShape,
                data: [Int8],
                scales: [Float],
                zeroPoints: [Int8]? = nil,
                mode: QuantizationMode,
                precision: QuantizationPrecision = .int8,
                groupSize: Int = 0,
                identifier: UUID = UUID()) {
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
        case .symmetric, .asymmetric:
            precondition(scales.count == 1, "Per-tensor: need single scale")
        case .int4:
            // Per-group: scales.count == numGroups = ceil(totalElements / groupSize)
            if groupSize > 0 {
                let numGroups = (shape.count + groupSize - 1) / groupSize
                precondition(scales.count == numGroups,
                            "INT4 per-group: need one scale per group (\(numGroups)), got \(scales.count)")
            } else {
                precondition(scales.count == 1, "INT4 per-tensor: need single scale")
            }
        }

        if let zp = zeroPoints {
            precondition(zp.count == scales.count, "Zero points must match scales")
        }

        self.shape = shape
        self.data = data
        self.precision = precision
        self.scales = scales
        self.groupSize = groupSize
        self.zeroPoints = zeroPoints
        self.mode = mode
        self.identifier = identifier
    }
    
    /// Dequantize back to Float32
    ///
    /// **NOTE:** This is only for debugging/testing. 
    /// Inference uses INT8 Metal kernel via matmul(quantized) - NO dequantization!
    ///
    /// Converts Int8 → Float32 using stored scales/zero-points
    ///
    /// Example:
    /// ```swift
    /// // For inference: Use INT8 kernel (no dequantization!)
    /// let output = input.matmul(quantized)  // INT8 Metal kernel
    ///
    /// // For debugging: Explicit dequantization
    /// let float = quantized.dequantize()  // Converts to Float32
    /// ```
    public func dequantize() -> Tensor<Float> {
        var floatData = [Float](repeating: 0.0, count: shape.count)  // Use shape.count, not data.count!

        switch mode {
        case .int4:
            // INT4 per-group dequantization
            //
            // Each packed byte contains two 4-bit values:
            //   high nibble = element at even index
            //   low nibble  = element at odd index
            //
            // Each group of `groupSize` elements shares one scale and zero point.
            // Formula: float_value = (int4_value - zero_point) * scale
            let gs = groupSize > 0 ? groupSize : shape.count
            for i in 0..<shape.count {
                let groupIdx = i / gs
                let scale = scales[groupIdx]
                let zeroPoint = zeroPoints?[groupIdx] ?? 0

                // Unpack: byte index = i / 2, high nibble for even i, low nibble for odd i
                let byteIdx = i / 2
                let packed = data[byteIdx]
                let int4Val: Int8
                if i % 2 == 0 {
                    // High nibble (first value in the packed pair)
                    let raw = (packed >> 4) & 0x0F
                    int4Val = raw > 7 ? raw - 16 : raw
                } else {
                    // Low nibble (second value in the packed pair)
                    let raw = packed & 0x0F
                    int4Val = raw > 7 ? raw - 16 : raw
                }

                floatData[i] = Float(int4Val - zeroPoint) * scale
            }

        case .int4PerChannel:
            // INT4 per-channel dequantization (one scale per row)
            let numChannels = shape.dimensions[0]
            let channelSize = shape.count / numChannels
            for ch in 0..<numChannels {
                let scale = scales[ch]
                let zeroPoint = zeroPoints?[ch] ?? 0
                for j in 0..<channelSize {
                    let i = ch * channelSize + j
                    let byteIdx = i / 2
                    let packed = data[byteIdx]
                    let int4Val: Int8
                    if i % 2 == 0 {
                        let raw = (packed >> 4) & 0x0F
                        int4Val = raw > 7 ? raw - 16 : raw
                    } else {
                        let raw = packed & 0x0F
                        int4Val = raw > 7 ? raw - 16 : raw
                    }
                    floatData[i] = Float(int4Val - zeroPoint) * scale
                }
            }

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
    /// Quantize this Float32 tensor
    ///
    /// Supports INT8 (per-channel, symmetric, asymmetric) and INT4 (per-group).
    ///
    /// - Parameters:
    ///   - mode: Quantization strategy (default: .perChannel for INT8)
    ///   - groupSize: Group size for INT4 per-group quantization (default: 128)
    /// - Returns: QuantizedTensor with quantized data and scales
    public func quantize(mode: QuantizationMode = .perChannel, groupSize: Int = 128) -> QuantizedTensor {
        switch mode {
        case .perChannel:
            return quantizePerChannel()
        case .symmetric:
            return quantizeSymmetric()
        case .asymmetric:
            return quantizeAsymmetric()
        case .int4:
            return quantizeINT4(groupSize: groupSize)
        case .int4PerChannel:
            return quantizeINT4PerChannel()
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

    // MARK: - INT4 Quantization

    /// Per-group INT4 quantization (recommended for 4-bit models)
    ///
    /// ## How Per-Group INT4 Works
    ///
    /// 1. Flatten the tensor into a 1D array of floats
    /// 2. Divide into groups of `groupSize` (default 128) consecutive elements
    /// 3. For each group, compute a scale factor from the max absolute value
    /// 4. Quantize each value to 4-bit range [-7, 7]
    /// 5. Pack two INT4 values into each byte (high nibble + low nibble)
    ///
    /// **Memory layout:**
    /// ```
    /// Original: [f0, f1, f2, f3, ...]  (4 bytes each = 4N bytes)
    /// Packed:   [p0, p1, ...]           (1 byte per 2 values = N/2 bytes)
    ///   where p0 = (quant(f0) << 4) | quant(f1)
    /// ```
    ///
    /// **Overhead:** One FP32 scale + one INT4 zero point per group
    ///   For group_size=128: 5 bytes / 128 values = ~4% overhead
    ///
    /// - Parameter groupSize: Number of elements per quantization group (default: 128)
    /// - Returns: QuantizedTensor with packed INT4 data
    private func quantizeINT4(groupSize: Int = 128) -> QuantizedTensor {
        let totalElements = shape.count
        let numGroups = (totalElements + groupSize - 1) / groupSize

        var scales = [Float](repeating: 0.0, count: numGroups)
        var zeroPoints = [Int8](repeating: 0, count: numGroups)
        let packedCount = (totalElements + 1) / 2
        var packedData = [Int8](repeating: 0, count: packedCount)

        // Step 1: Compute per-group scales and zero points
        for g in 0..<numGroups {
            let start = g * groupSize
            let end = min(start + groupSize, totalElements)
            let groupData = Array(rawData[start..<end])

            // Find the range of this group
            let minVal = groupData.min() ?? 0.0
            let maxVal = groupData.max() ?? 0.0

            // Symmetric quantization: map absMax to 7 (max positive INT4)
            let absMax = max(abs(minVal), abs(maxVal))
            let scale: Float
            if absMax < Float.leastNonzeroMagnitude {
                scale = 1.0
            } else {
                scale = absMax / 7.0
            }
            scales[g] = scale
            zeroPoints[g] = 0  // Symmetric: zero point is 0

            // Step 2: Quantize and pack pairs of values into bytes
            for i in start..<end {
                let floatVal = rawData[i]
                let quantVal = round(floatVal / scale)
                // Clamp to INT4 range [-7, 7]
                let clamped = Int8(max(-7, min(7, quantVal)))

                // Pack into the appropriate nibble of the output byte
                //   Even index → high nibble (bits 7-4)
                //   Odd index  → low nibble  (bits 3-0)
                let byteIdx = i / 2
                if i % 2 == 0 {
                    packedData[byteIdx] = (clamped & 0x0F) << 4
                } else {
                    packedData[byteIdx] |= (clamped & 0x0F)
                }
            }
        }

        return QuantizedTensor(
            shape: shape,
            data: packedData,
            scales: scales,
            zeroPoints: zeroPoints,
            mode: .int4,
            precision: .int4,
            groupSize: groupSize
        )
    }

    /// Per-channel INT4 quantization (one scale per row, 4-bit values)
    private func quantizeINT4PerChannel() -> QuantizedTensor {
        precondition(shape.dimensions.count == 2, "Per-channel quantization requires 2D tensor")

        let numChannels = shape.dimensions[0]
        let channelSize = shape.count / numChannels
        let packedCount = (shape.count + 1) / 2
        var packedData = [Int8](repeating: 0, count: packedCount)
        var scales = [Float](repeating: 0.0, count: numChannels)

        for ch in 0..<numChannels {
            let start = ch * channelSize
            let end = start + channelSize
            let channelData = Array(rawData[start..<end])

            let absMax = max(abs(channelData.min() ?? 0.0), abs(channelData.max() ?? 0.0))
            let scale: Float = absMax < Float.leastNonzeroMagnitude ? 1.0 : absMax / 7.0
            scales[ch] = scale

            for i in start..<end {
                let quantVal = round(rawData[i] / scale)
                let clamped = Int8(max(-7, min(7, quantVal)))
                let byteIdx = i / 2
                if i % 2 == 0 {
                    packedData[byteIdx] = (clamped & 0x0F) << 4
                } else {
                    packedData[byteIdx] |= (clamped & 0x0F)
                }
            }
        }

        return QuantizedTensor(
            shape: shape,
            data: packedData,
            scales: scales,
            zeroPoints: nil,
            mode: .int4PerChannel,
            precision: .int4
        )
    }
}

/// Extension for matmul with quantized tensors
extension Tensor where Element == Float {
    /// Matrix multiplication with quantized weights
    ///
    /// **REVIEW HITLER FIX:** Now uses INT8 Metal kernel (no Float32 materialization!)
    ///
    /// Example:
    /// ```swift
    /// let input = Tensor<Float>.random([128, 768])
    /// let weights = loadWeights().quantize()  // QuantizedTensor
    /// let output = input.matmul(weights)  // Computes directly from INT8!
    /// ```
    public func matmul(_ quantized: QuantizedTensor) -> Tensor<Float> {
        // Try Metal kernel first (supports both INT8 and INT4)
        if let metalBackend = TinyBrainBackend.metalBackend as? QuantizedMatMulBackend {
            do {
                let label = quantized.precision == .int4 ? "INT4" : "INT8"
                TinyBrainBackend.log("Using \(label) Metal kernel (no Float32 materialization!)")
                return try metalBackend.matmulQuantized(self, quantized)
            } catch {
                let label = quantized.precision == .int4 ? "INT4" : "INT8"
                print("\(label) kernel failed: \(error), falling back to dequant+CPU")
                TinyBrainBackend.log("\(label) kernel failed: \(error), falling back to dequant+CPU")
            }
        } else {
            TinyBrainBackend.log("No QuantizedMatMulBackend available, falling back to CPU")
        }

        // Fallback: Dequantize then CPU matmul
        TinyBrainBackend.log("Falling back to dequant+CPU matmul")
        let weights = quantized.dequantize()
        return self.matmulCPU(weights)
    }
}
