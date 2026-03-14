/// Tokenizer Loader Tests
///
/// **TB-009 RED Phase:** Tests for format-agnostic tokenizer loading
///
/// Tests cover:
/// - Format detection (HuggingFace, SentencePiece, TinyBrain)
/// - HuggingFace tokenizer.json parsing
/// - Automatic discovery
/// - Error handling

import XCTest
@testable import TinyBrainTokenizer

final class TokenizerLoaderTests: XCTestCase {
    
    // MARK: - Format Detection
    
    func testDetectHuggingFaceFormat() throws {
        let fixtureURL = Bundle.module.url(forResource: "tinyllama_tokenizer", withExtension: "json")
        XCTAssertNotNil(fixtureURL, "Test fixture should exist")
        
        let format = TokenizerFormat.detect(at: fixtureURL!.path)
        XCTAssertEqual(format, .huggingFace, "Should detect HuggingFace format")
    }
    
    func testDetectTinyBrainFormat() throws {
        let fixtureURL = Bundle.module.url(forResource: "test_vocab", withExtension: "json")!
        
        let format = TokenizerFormat.detect(at: fixtureURL.path)
        XCTAssertEqual(format, .tinyBrain, "Should detect TinyBrain format")
    }
    
    func testDetectInvalidFile() {
        let format = TokenizerFormat.detect(at: "/nonexistent/file.json")
        XCTAssertNil(format, "Should return nil for missing file")
    }
    
    // MARK: - HuggingFace Adapter
    
    func testLoadHuggingFaceTokenizer() throws {
        let fixtureURL = Bundle.module.url(forResource: "tinyllama_tokenizer", withExtension: "json")
        XCTAssertNotNil(fixtureURL, "Test fixture should exist")
        
        let tokenizer = try TokenizerLoader.loadHuggingFace(from: fixtureURL!.path)
        
        // Validate it's a proper tokenizer
        XCTAssertGreaterThan(tokenizer.vocabularySize, 0, "Should have vocabulary")
        
        // Test basic encoding/decoding
        let text = "Hello world"
        let tokens = tokenizer.encode(text)
        XCTAssertGreaterThan(tokens.count, 0, "Should encode to tokens")
        
        let decoded = tokenizer.decode(tokens)
        XCTAssertFalse(decoded.isEmpty, "Should decode back to text")
    }
    
    func testHuggingFaceSpecialTokens() throws {
        let fixtureURL = Bundle.module.url(forResource: "tinyllama_tokenizer", withExtension: "json")!
        let tokenizer = try TokenizerLoader.loadHuggingFace(from: fixtureURL.path)
        
        // HF tokenizers should have BOS, EOS tokens
        XCTAssertNotEqual(tokenizer.bosToken, tokenizer.unkToken, "BOS should be distinct")
        XCTAssertNotEqual(tokenizer.eosToken, tokenizer.unkToken, "EOS should be distinct")
    }
    
    // MARK: - Generic Loader
    
    func testLoadAuto() throws {
        // Should auto-detect and load HuggingFace format
        let fixtureURL = Bundle.module.url(forResource: "tinyllama_tokenizer", withExtension: "json")!
        
        let tokenizer = try TokenizerLoader.load(from: fixtureURL.path)
        XCTAssertGreaterThan(tokenizer.vocabularySize, 0)
    }
    
    func testLoadAutoTinyBrainFormat() throws {
        // Should auto-detect and load TinyBrain format
        let fixtureURL = Bundle.module.url(forResource: "test_vocab", withExtension: "json")!
        
        let tokenizer = try TokenizerLoader.load(from: fixtureURL.path)
        XCTAssertGreaterThan(tokenizer.vocabularySize, 0)
    }
    
    func testLoadBestAvailable() {
        // Should find and load any available tokenizer
        let tokenizer = TokenizerLoader.loadBestAvailable()
        
        // Should at least return something (even if fallback)
        XCTAssertGreaterThan(tokenizer.vocabularySize, 0)
    }
    
    // MARK: - Error Handling
    
    func testLoadInvalidFile() {
        XCTAssertThrowsError(try TokenizerLoader.load(from: "/nonexistent.json")) {
            error in
            XCTAssertTrue(error is CocoaError || error is TokenizerError)
        }
    }
    
    func testLoadInvalidJSON() throws {
        // Create temp file with invalid JSON
        let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent("invalid.json")
        try "{ invalid json }".write(to: tempURL, atomically: true, encoding: .utf8)
        
        XCTAssertThrowsError(try TokenizerLoader.load(from: tempURL.path))
        
        try? FileManager.default.removeItem(at: tempURL)
    }
}

