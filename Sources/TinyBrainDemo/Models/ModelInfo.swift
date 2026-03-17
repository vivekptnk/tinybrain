/// Model Info
///
/// Metadata about a TinyBrain model file discovered on disk.
/// Used by the model picker to display available models.

import Foundation

/// Describes a `.tbf` model file available for loading
public struct ModelInfo: Identifiable, Equatable {

    // MARK: - Properties

    /// Stable unique identifier (based on file URL)
    public let id: String

    /// Display name derived from filename (e.g. "tinyllama-1.1b-int8")
    public let displayName: String

    /// Absolute path to the `.tbf` file
    public let path: String

    /// File size in bytes (nil if unavailable)
    public let fileSizeBytes: Int?

    /// Detected quantization type parsed from the filename
    public let quantization: QuantizationHint

    // MARK: - Init

    public init(path: String) {
        let url = URL(fileURLWithPath: path)
        self.id = path
        self.path = path

        // Build display name from filename, strip extension
        let filename = url.deletingPathExtension().lastPathComponent
        self.displayName = filename

        // File size
        let attrs = try? FileManager.default.attributesOfItem(atPath: path)
        self.fileSizeBytes = attrs?[.size] as? Int

        // Detect quantization from filename
        self.quantization = QuantizationHint.detect(from: filename)
    }

    // MARK: - Formatted Helpers

    /// Human-readable file size (e.g. "1.2 GB", "450 MB")
    public var formattedSize: String {
        guard let bytes = fileSizeBytes else { return "Unknown size" }
        let mb = Double(bytes) / 1_048_576.0
        if mb >= 1024 {
            return String(format: "%.1f GB", mb / 1024.0)
        }
        return String(format: "%.0f MB", mb)
    }
}

// MARK: - QuantizationHint

/// Detected quantization type, parsed heuristically from the model filename.
///
/// Examples:
/// - `tinyllama-1.1b-int8.tbf` → `.int8`
/// - `gemma-2b-int4.tbf`       → `.int4`
/// - `phi-2-fp16.tbf`           → `.fp16`
/// - `my-model.tbf`             → `.unknown`
public enum QuantizationHint: String, Equatable {
    case fp32 = "FP32"
    case fp16 = "FP16"
    case int8 = "INT8"
    case int4 = "INT4"
    case unknown = "?"

    static func detect(from filename: String) -> QuantizationHint {
        let lower = filename.lowercased()
        if lower.contains("int4") || lower.contains("q4") { return .int4 }
        if lower.contains("int8") || lower.contains("q8") { return .int8 }
        if lower.contains("fp16") || lower.contains("f16") { return .fp16 }
        if lower.contains("fp32") || lower.contains("f32") { return .fp32 }
        return .unknown
    }

    /// Badge color hint for the UI
    public var colorDescription: String {
        switch self {
        case .int4:    return "green"
        case .int8:    return "blue"
        case .fp16:    return "orange"
        case .fp32:    return "red"
        case .unknown: return "gray"
        }
    }
}

// MARK: - Model Scanner

/// Scans the `Models/` directory for available `.tbf` files
public enum ModelScanner {

    /// Scan a directory for `.tbf` model files.
    ///
    /// Only files at the top level and one level deep are returned
    /// (we don't recurse into arbitrary depth to stay fast).
    ///
    /// - Parameter directoryPath: Absolute path to directory to scan (defaults to project `Models/`)
    /// - Returns: Sorted list of discovered models (by display name)
    public static func scan(directoryPath: String? = nil) -> [ModelInfo] {
        let root = directoryPath ?? resolveModelsDir()
        guard FileManager.default.fileExists(atPath: root) else { return [] }

        var results: [ModelInfo] = []

        // Scan top-level files
        let topLevel = (try? FileManager.default.contentsOfDirectory(atPath: root)) ?? []
        for name in topLevel {
            let full = (root as NSString).appendingPathComponent(name)
            if name.hasSuffix(".tbf") {
                results.append(ModelInfo(path: full))
            }
        }

        // Scan one level of subdirectories
        for name in topLevel {
            let subdir = (root as NSString).appendingPathComponent(name)
            var isDir: ObjCBool = false
            FileManager.default.fileExists(atPath: subdir, isDirectory: &isDir)
            if isDir.boolValue {
                let subfiles = (try? FileManager.default.contentsOfDirectory(atPath: subdir)) ?? []
                for subname in subfiles where subname.hasSuffix(".tbf") {
                    let full = (subdir as NSString).appendingPathComponent(subname)
                    results.append(ModelInfo(path: full))
                }
            }
        }

        return results.sorted { $0.displayName < $1.displayName }
    }

    // MARK: - Private

    private static func resolveModelsDir() -> String {
        let relative = "Models"

        // Check cwd
        if FileManager.default.fileExists(atPath: relative) {
            return FileManager.default.currentDirectoryPath + "/" + relative
        }

        // Walk up looking for Package.swift
        var current = FileManager.default.currentDirectoryPath
        for _ in 0..<10 {
            let pkgPath = (current as NSString).appendingPathComponent("Package.swift")
            if FileManager.default.fileExists(atPath: pkgPath) {
                return (current as NSString).appendingPathComponent(relative)
            }
            current = (current as NSString).deletingLastPathComponent
            if current == "/" { break }
        }

        // Hardcoded project root
        return "/Users/vivekesque/Desktop/CreativeSpace/CodingProjects/tinybrain/Models"
    }
}
