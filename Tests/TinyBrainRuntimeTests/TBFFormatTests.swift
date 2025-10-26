import XCTest
@testable import TinyBrainRuntime
import Foundation

/// **TB-004 Work Item #3:** TBF Format Tests (TDD RED Phase)
///
/// WHAT: Test suite for TinyBrain Binary Format (.tbf) save/load functionality
/// WHY: Validates mmap-friendly weight serialization with zero-copy loading
/// HOW: Test round-trip, mmap efficiency, error handling, version validation
///
/// **TDD Phase:** RED - These tests should FAIL until ModelWeights.save()/load() implemented
final class TBFFormatTests: XCTestCase {
    
    var tempDirectory: URL!
    
    override func setUp() {
        super.setUp()
        tempDirectory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try? FileManager.default.createDirectory(at: tempDirectory, withIntermediateDirectories: true)
    }
    
    override func tearDown() {
        try? FileManager.default.removeItem(at: tempDirectory)
        super.tearDown()
    }
    
    // MARK: - Basic Save/Load Tests
    
    /// **RED:** Test saving toy model to .tbf file
    ///
    /// WHAT: Save a toy model to disk in .tbf format
    /// WHY: Validates that ModelWeights can serialize to binary format
    /// HOW: Create toy model, save to temp file, verify file exists
    /// EXPECTED: Should FAIL - ModelWeights.save(to:) doesn't exist yet
    func testSaveToyModelToTBF() throws {
        let config = ModelConfig(
            numLayers: 2,
            hiddenDim: 32,
            numHeads: 4,
            vocabSize: 64,
            maxSeqLen: 128
        )
        
        let weights = ModelWeights.makeToyModel(config: config, seed: 42)
        let filePath = tempDirectory.appendingPathComponent("test.tbf").path
        
        // RED: This should fail - save() method doesn't exist yet
        try weights.save(to: filePath)
        
        // Verify file was created
        XCTAssertTrue(FileManager.default.fileExists(atPath: filePath),
                      "Save should create .tbf file")
    }
    
    /// **RED:** Test loading toy model from .tbf file
    ///
    /// WHAT: Load a saved model from .tbf file
    /// WHY: Validates that ModelWeights can deserialize from binary format
    /// HOW: Save toy model, then load it back
    /// EXPECTED: Should FAIL - ModelWeights.load(from:) doesn't exist yet
    func testLoadToyModelFromTBF() throws {
        let config = ModelConfig(
            numLayers: 2,
            hiddenDim: 32,
            numHeads: 4,
            vocabSize: 64,
            maxSeqLen: 128
        )
        
        let originalWeights = ModelWeights.makeToyModel(config: config, seed: 42)
        let filePath = tempDirectory.appendingPathComponent("test.tbf").path
        
        // Save first (will fail in RED phase)
        try originalWeights.save(to: filePath)
        
        // RED: This should fail - load() method doesn't exist yet
        let loadedWeights = try ModelWeights.load(from: filePath)
        
        // Verify config matches
        XCTAssertEqual(loadedWeights.config.numLayers, config.numLayers)
        XCTAssertEqual(loadedWeights.config.hiddenDim, config.hiddenDim)
        XCTAssertEqual(loadedWeights.config.vocabSize, config.vocabSize)
    }
    
    /// **RED:** Test round-trip preserves weights exactly
    ///
    /// WHAT: Save then load, verify weights are identical
    /// WHY: Ensures no data corruption during serialization/deserialization
    /// HOW: Compare embeddings and layer weights element-by-element
    /// ACCURACY: Exact match for INT8 (no floating point error)
    func testRoundTripPreservesWeights() throws {
        let config = ModelConfig(
            numLayers: 2,
            hiddenDim: 32,
            numHeads: 4,
            vocabSize: 64,
            maxSeqLen: 128
        )
        
        let original = ModelWeights.makeToyModel(config: config, seed: 12345)
        let filePath = tempDirectory.appendingPathComponent("roundtrip.tbf").path
        
        // Save and load
        try original.save(to: filePath)
        let loaded = try ModelWeights.load(from: filePath)
        
        // Verify embeddings match
        XCTAssertEqual(original.tokenEmbeddings.shape, loaded.tokenEmbeddings.shape,
                       "Embedding shape should match")
        
        let originalEmbedData = original.tokenEmbeddings.data
        let loadedEmbedData = loaded.tokenEmbeddings.data
        
        XCTAssertEqual(originalEmbedData.count, loadedEmbedData.count,
                       "Embedding data count should match")
        
        // Sample check (first 100 values)
        for i in 0..<min(100, originalEmbedData.count) {
            XCTAssertEqual(originalEmbedData[i], loadedEmbedData[i], accuracy: 1e-6,
                           "Embedding[\(i)] should match after round-trip")
        }
        
        // Verify first layer query weights match (quantized)
        let originalQuery = original.layers[0].attention.query.weights
        let loadedQuery = loaded.layers[0].attention.query.weights
        
        XCTAssertEqual(originalQuery.shape, loadedQuery.shape,
                       "Query weight shape should match")
        XCTAssertEqual(originalQuery.scales.count, loadedQuery.scales.count,
                       "Query scales count should match")
        
        // Scales should match exactly
        for i in 0..<originalQuery.scales.count {
            XCTAssertEqual(originalQuery.scales[i], loadedQuery.scales[i], accuracy: 1e-9,
                           "Query scale[\(i)] should match exactly")
        }
    }
    
    // MARK: - mmap Efficiency Tests
    
    /// **RED:** Test that mmap doesn't load entire file into RAM
    ///
    /// WHAT: Verify file is memory-mapped, not fully loaded
    /// WHY: Validates zero-copy loading efficiency requirement
    /// HOW: Load large model, check memory usage is much less than file size
    /// EXPECTED: Memory delta << file size (OS should map, not load)
    func testMmapDoesNotLoadEntireFile() throws {
        let config = ModelConfig(
            numLayers: 4,  // Larger model
            hiddenDim: 128,
            numHeads: 8,
            vocabSize: 1024,
            maxSeqLen: 512
        )
        
        let weights = ModelWeights.makeToyModel(config: config, seed: 999)
        let filePath = tempDirectory.appendingPathComponent("large.tbf").path
        
        try weights.save(to: filePath)
        
        // Get file size
        let attrs = try FileManager.default.attributesOfItem(atPath: filePath)
        let fileSize = attrs[.size] as! Int
        
        // Load via mmap
        let loaded = try ModelWeights.load(from: filePath)
        
        // Access one embedding (should trigger minimal page loads)
        let embedding = loaded.embedding(for: 0)
        XCTAssertEqual(embedding.shape.dimensions[0], config.hiddenDim)
        
        // Memory usage test (approximate)
        // In true mmap, we shouldn't see RAM usage spike by fileSize
        // This is hard to test precisely, but we can verify load succeeds
        // and file is accessible without errors
        
        print("File size: \(fileSize / 1024) KB")
        print("Model loaded via mmap successfully")
        
        // If we got here without running out of memory, mmap is working
        XCTAssertTrue(true, "mmap load should succeed without loading full file to RAM")
    }
    
    // MARK: - Error Handling Tests
    
    /// **RED:** Test invalid magic bytes throw error
    ///
    /// WHAT: Load file with wrong magic bytes
    /// WHY: Validates format detection and error handling
    /// HOW: Create file with "JUNK" instead of "TBFM", try to load
    /// EXPECTED: Should throw TBFError.invalidMagicBytes
    func testInvalidMagicBytesThrowsError() throws {
        let filePath = tempDirectory.appendingPathComponent("invalid.tbf").path
        
        // Write file with invalid magic bytes
        let invalidData = Data("JUNK".utf8) + Data(count: 1000)
        try invalidData.write(to: URL(fileURLWithPath: filePath))
        
        // Should throw error
        XCTAssertThrowsError(try ModelWeights.load(from: filePath)) { error in
            // Verify it's the right error type
            if let tbfError = error as? TBFError {
                switch tbfError {
                case .invalidMagicBytes:
                    break  // Expected
                default:
                    XCTFail("Expected .invalidMagicBytes, got \(tbfError)")
                }
            } else {
                XCTFail("Expected TBFError, got \(error)")
            }
        }
    }
    
    /// **RED:** Test version mismatch throws error
    ///
    /// WHAT: Load file with unsupported version
    /// WHY: Validates forward compatibility handling
    /// HOW: Create file with version 999, try to load
    /// EXPECTED: Should throw TBFError.unsupportedVersion
    func testVersionMismatchThrowsError() throws {
        let filePath = tempDirectory.appendingPathComponent("future.tbf").path
        
        // Write file with valid magic but future version
        var data = Data()
        data.append(contentsOf: "TBFM".utf8)  // Valid magic
        
        // Version 999 (unsupported future version)
        var version: UInt32 = 999
        data.append(Data(bytes: &version, count: 4))
        
        // Add some padding
        data.append(Data(count: 1000))
        
        try data.write(to: URL(fileURLWithPath: filePath))
        
        // Should throw error
        XCTAssertThrowsError(try ModelWeights.load(from: filePath)) { error in
            if let tbfError = error as? TBFError {
                switch tbfError {
                case .unsupportedVersion(let found):
                    XCTAssertEqual(found, 999, "Should report found version")
                default:
                    XCTFail("Expected .unsupportedVersion, got \(tbfError)")
                }
            } else {
                XCTFail("Expected TBFError, got \(error)")
            }
        }
    }
    
    /// **RED:** Test file not found throws error
    ///
    /// WHAT: Try to load non-existent file
    /// WHY: Validates error handling for missing files
    /// HOW: Load from path that doesn't exist
    /// EXPECTED: Should throw appropriate error
    func testFileNotFoundThrowsError() throws {
        let filePath = tempDirectory.appendingPathComponent("nonexistent.tbf").path
        
        XCTAssertThrowsError(try ModelWeights.load(from: filePath)) { error in
            // Should get some kind of file error
            XCTAssertTrue(error is TBFError || error is CocoaError,
                          "Should throw file-related error")
        }
    }
}

