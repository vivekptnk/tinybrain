/// Protocol defining types that can be used as Tensor elements
///
/// **TB-004 Phase 2:** Enables generic Tensor<Element> supporting Float32, Float16, Int8
///
/// ## Why Generic Tensors?
///
/// Different data types serve different purposes:
/// - **Float32 (Float):** Standard precision, 4 bytes - default for most compute
/// - **Float16:** Half precision, 2 bytes - 50% memory savings
/// - **Int8:** Quantized weights, 1 byte - 75% memory savings
///
/// Generic tensors let us use the same tensor API for all types!

import Foundation
#if canImport(Metal)
import Metal
#endif

/// Elements that can be stored in tensors
public protocol TensorElement {
    /// Zero value for this type
    static var zero: Self { get }
    
    /// One value for this type (for identity/scaling)
    static var one: Self { get }
    
    /// Metal data type (for GPU kernels)
    static var metalType: MTLDataType { get }
    
    /// Size in bytes
    static var byteSize: Int { get }
    
    /// Type name for debugging
    static var typeName: String { get }
}

// MARK: - Float32 Conformance

extension Float: TensorElement {
    public static var zero: Float { 0.0 }
    public static var one: Float { 1.0 }
    
    #if canImport(Metal)
    public static var metalType: MTLDataType { .float }
    #else
    public static var metalType: MTLDataType { fatalError("Metal not available") }
    #endif
    
    public static var byteSize: Int { 4 }
    public static var typeName: String { "Float32" }
}

// MARK: - Float16 Conformance

#if canImport(_Float16)
import _Float16

extension Float16: TensorElement {
    public static var zero: Float16 { 0.0 }
    public static var one: Float16 { 1.0 }
    
    #if canImport(Metal)
    public static var metalType: MTLDataType { .half }
    #else
    public static var metalType: MTLDataType { fatalError("Metal not available") }
    #endif
    
    public static var byteSize: Int { 2 }
    public static var typeName: String { "Float16" }
}
#endif

// MARK: - Int8 Conformance

extension Int8: TensorElement {
    public static var zero: Int8 { 0 }
    public static var one: Int8 { 1 }
    
    #if canImport(Metal)
    public static var metalType: MTLDataType { .char }
    #else
    public static var metalType: MTLDataType { fatalError("Metal not available") }
    #endif
    
    public static var byteSize: Int { 1 }
    public static var typeName: String { "Int8" }
}

// MARK: - Type Conversion Utilities

/// Convert between tensor element types
public protocol TensorElementConvertible {
    func toFloat() -> Float
    func toFloat16() -> Float16
    func toInt8() -> Int8
}

extension Float: TensorElementConvertible {
    public func toFloat() -> Float { self }
    
    #if canImport(_Float16)
    public func toFloat16() -> Float16 { Float16(self) }
    #else
    public func toFloat16() -> Float16 { fatalError("Float16 not available") }
    #endif
    
    public func toInt8() -> Int8 {
        // Clamp to Int8 range
        let clamped = max(-128.0, min(127.0, self))
        return Int8(clamped)
    }
}

#if canImport(_Float16)
extension Float16: TensorElementConvertible {
    public func toFloat() -> Float { Float(self) }
    public func toFloat16() -> Float16 { self }
    
    public func toInt8() -> Int8 {
        let clamped = max(-128.0, min(127.0, Float(self)))
        return Int8(clamped)
    }
}
#endif

extension Int8: TensorElementConvertible {
    public func toFloat() -> Float { Float(self) }
    
    #if canImport(_Float16)
    public func toFloat16() -> Float16 { Float16(self) }
    #else
    public func toFloat16() -> Float16 { fatalError("Float16 not available") }
    #endif
    
    public func toInt8() -> Int8 { self }
}

