/// Model Picker Tests
///
/// Tests for ModelInfo, QuantizationHint, and ModelScanner.
///
/// Verifies:
/// - ModelInfo construction from file paths
/// - Quantization detection from filenames
/// - ModelScanner.scan() returns sorted results
/// - Empty/nonexistent directory handling

import XCTest
@testable import TinyBrainDemo

final class ModelPickerTests: XCTestCase {

    // MARK: - ModelInfo Tests

    func testModelInfoDisplayName() throws {
        let tempDir = FileManager.default.temporaryDirectory
        let filePath = tempDir.appendingPathComponent("tinyllama-1.1b-int8.tbf").path

        // Create a minimal placeholder file
        FileManager.default.createFile(atPath: filePath, contents: Data([0]))
        defer { try? FileManager.default.removeItem(atPath: filePath) }

        let info = ModelInfo(path: filePath)
        XCTAssertEqual(info.displayName, "tinyllama-1.1b-int8")
    }

    func testModelInfoFileSizePopulated() throws {
        let tempDir = FileManager.default.temporaryDirectory
        let filePath = tempDir.appendingPathComponent("mymodel.tbf").path
        let content = Data(repeating: 0xAB, count: 1024)
        FileManager.default.createFile(atPath: filePath, contents: content)
        defer { try? FileManager.default.removeItem(atPath: filePath) }

        let info = ModelInfo(path: filePath)
        XCTAssertNotNil(info.fileSizeBytes, "File size should be populated")
        XCTAssertEqual(info.fileSizeBytes, 1024)
    }

    func testModelInfoFormattedSizeMB() throws {
        let tempDir = FileManager.default.temporaryDirectory
        let filePath = tempDir.appendingPathComponent("mb_model.tbf").path
        // Write 2 MB of data (not 500 MB — keep tests fast)
        let content = Data(repeating: 0, count: 2 * 1024 * 1024)
        FileManager.default.createFile(atPath: filePath, contents: content)
        defer { try? FileManager.default.removeItem(atPath: filePath) }

        let info = ModelInfo(path: filePath)
        let size = info.formattedSize
        XCTAssertTrue(size.contains("MB") || size.contains("GB"),
                      "Size should be expressed in MB or GB, got: \(size)")
    }

    func testModelInfoIdEqualsPath() {
        let path = "/some/path/model.tbf"
        let info = ModelInfo(path: path)
        XCTAssertEqual(info.id, path, "id should equal the file path")
    }

    func testModelInfoEquality() {
        let path = "/tmp/model.tbf"
        let a = ModelInfo(path: path)
        let b = ModelInfo(path: path)
        XCTAssertEqual(a, b, "Two ModelInfo objects with the same path should be equal")
    }

    // MARK: - QuantizationHint Tests

    func testDetectInt4FromFilename() {
        XCTAssertEqual(QuantizationHint.detect(from: "gemma-2b-int4"), .int4)
        XCTAssertEqual(QuantizationHint.detect(from: "model-q4"), .int4)
    }

    func testDetectInt8FromFilename() {
        XCTAssertEqual(QuantizationHint.detect(from: "tinyllama-1.1b-int8"), .int8)
        XCTAssertEqual(QuantizationHint.detect(from: "llama-q8"), .int8)
    }

    func testDetectFP16FromFilename() {
        XCTAssertEqual(QuantizationHint.detect(from: "phi-2-fp16"), .fp16)
        XCTAssertEqual(QuantizationHint.detect(from: "model-f16"), .fp16)
    }

    func testDetectFP32FromFilename() {
        XCTAssertEqual(QuantizationHint.detect(from: "model-fp32"), .fp32)
    }

    func testDetectUnknownFromFilename() {
        XCTAssertEqual(QuantizationHint.detect(from: "my-model"), .unknown)
        XCTAssertEqual(QuantizationHint.detect(from: "abc"), .unknown)
    }

    func testCaseInsensitiveDetection() {
        XCTAssertEqual(QuantizationHint.detect(from: "MODEL-INT4"), .int4)
        XCTAssertEqual(QuantizationHint.detect(from: "MODEL-FP16"), .fp16)
    }

    // MARK: - ModelScanner Tests

    func testScanEmptyDirectory() throws {
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let results = ModelScanner.scan(directoryPath: tempDir.path)
        XCTAssertTrue(results.isEmpty, "Empty directory should return no models")
    }

    func testScanNonexistentDirectory() {
        let results = ModelScanner.scan(directoryPath: "/nonexistent/dir/that/does/not/exist")
        XCTAssertTrue(results.isEmpty, "Nonexistent directory should return empty list")
    }

    func testScanFindsTopLevelTBFFiles() throws {
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempDir) }

        // Create two .tbf files and one non-.tbf file
        let files = ["alpha.tbf", "beta.tbf", "readme.txt"]
        for name in files {
            let url = tempDir.appendingPathComponent(name)
            FileManager.default.createFile(atPath: url.path, contents: Data([0]))
        }

        let results = ModelScanner.scan(directoryPath: tempDir.path)
        XCTAssertEqual(results.count, 2, "Should find exactly 2 .tbf files")
    }

    func testScanFindsFilesInSubdirectory() throws {
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
        let subDir = tempDir.appendingPathComponent("tinyllama-raw")
        try FileManager.default.createDirectory(at: subDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let modelURL = subDir.appendingPathComponent("model.tbf")
        FileManager.default.createFile(atPath: modelURL.path, contents: Data([0]))

        let results = ModelScanner.scan(directoryPath: tempDir.path)
        XCTAssertEqual(results.count, 1, "Should find .tbf in subdirectory")
    }

    func testScanResultsSortedByName() throws {
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempDir) }

        for name in ["zeta.tbf", "alpha.tbf", "gamma.tbf"] {
            let url = tempDir.appendingPathComponent(name)
            FileManager.default.createFile(atPath: url.path, contents: Data([0]))
        }

        let results = ModelScanner.scan(directoryPath: tempDir.path)
        XCTAssertEqual(results.map { $0.displayName }, ["alpha", "gamma", "zeta"],
                       "Results should be sorted alphabetically by display name")
    }

    func testScanIgnoresNonTBFFiles() throws {
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempDir) }

        for name in ["model.tbf", "tokenizer.json", "config.json", "notes.txt"] {
            let url = tempDir.appendingPathComponent(name)
            FileManager.default.createFile(atPath: url.path, contents: Data([0]))
        }

        let results = ModelScanner.scan(directoryPath: tempDir.path)
        XCTAssertEqual(results.count, 1)
        XCTAssertEqual(results[0].displayName, "model")
    }

    func testScanPopulatesQuantizationHint() throws {
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let url = tempDir.appendingPathComponent("tinyllama-1.1b-int8.tbf")
        FileManager.default.createFile(atPath: url.path, contents: Data([0]))

        let results = ModelScanner.scan(directoryPath: tempDir.path)
        XCTAssertEqual(results.first?.quantization, .int8,
                       "Should detect INT8 from filename")
    }
}
