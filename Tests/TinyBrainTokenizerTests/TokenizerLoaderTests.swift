/// Tokenizer Loader Tests
///
/// **TB-009 / TB-010:** Tests for format-agnostic tokenizer loading
///
/// Tests cover:
/// - Format detection for all four formats (HuggingFace, TinyBrain, SentencePiece, TikToken)
/// - HuggingFace tokenizer.json parsing
/// - SentencePiece .vocab loading
/// - TikToken .tiktoken loading
/// - Automatic discovery (loadBestAvailable)
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
    
    // MARK: - SentencePiece Format Detection

    func testDetectSentencePieceVocabExtension() throws {
        let fixtureURL = Bundle.module.url(forResource: "test_sentencepiece", withExtension: "vocab")
        XCTAssertNotNil(fixtureURL, "test_sentencepiece.vocab fixture should exist")

        let format = TokenizerFormat.detect(at: fixtureURL!.path)
        XCTAssertEqual(format, .sentencePiece, "Should detect .vocab as SentencePiece")
    }

    func testDetectSentencePieceModelFilename() throws {
        // Create a temp file named "tokenizer.model"
        let tempDir = FileManager.default.temporaryDirectory
        let modelURL = tempDir.appendingPathComponent("tokenizer.model")
        // Write minimal valid SP content
        let content = "<unk>\t0\n<s>\t0\n</s>\t0\n▁Hello\t-1.0\n"
        try content.write(to: modelURL, atomically: true, encoding: .utf8)
        defer { try? FileManager.default.removeItem(at: modelURL) }

        let format = TokenizerFormat.detect(at: modelURL.path)
        XCTAssertEqual(format, .sentencePiece, "Should detect 'tokenizer.model' as SentencePiece")
    }

    func testLoadSentencePieceVocab() throws {
        let fixtureURL = Bundle.module.url(forResource: "test_sentencepiece", withExtension: "vocab")!
        let tokenizer = try TokenizerLoader.load(from: fixtureURL.path)
        XCTAssertGreaterThan(tokenizer.vocabularySize, 0, "SentencePiece vocab should load")
    }

    func testLoadSentencePieceRoundTrip() throws {
        let fixtureURL = Bundle.module.url(forResource: "test_sentencepiece", withExtension: "vocab")!
        let tokenizer = try TokenizerLoader.load(from: fixtureURL.path)
        let tokens = tokenizer.encode("Hello")
        XCTAssertFalse(tokens.isEmpty)
        let decoded = tokenizer.decode(tokens)
        XCTAssertFalse(decoded.isEmpty)
    }

    // MARK: - TikToken Format Detection

    func testDetectTikTokenExtension() throws {
        let fixtureURL = Bundle.module.url(forResource: "test_tiktoken", withExtension: "tiktoken")
        XCTAssertNotNil(fixtureURL, "test_tiktoken.tiktoken fixture should exist")

        let format = TokenizerFormat.detect(at: fixtureURL!.path)
        XCTAssertEqual(format, .tiktoken, "Should detect .tiktoken as TikToken format")
    }

    func testLoadTikToken() throws {
        let fixtureURL = Bundle.module.url(forResource: "test_tiktoken", withExtension: "tiktoken")!
        let tokenizer = try TokenizerLoader.load(from: fixtureURL.path)
        XCTAssertGreaterThan(tokenizer.vocabularySize, 0, "TikToken file should load")
    }

    func testLoadTikTokenEncodesDecode() throws {
        let fixtureURL = Bundle.module.url(forResource: "test_tiktoken", withExtension: "tiktoken")!
        let tokenizer = try TokenizerLoader.load(from: fixtureURL.path)
        let tokens = tokenizer.encode("Hello")
        XCTAssertFalse(tokens.isEmpty)
        let decoded = tokenizer.decode(tokens)
        XCTAssertFalse(decoded.isEmpty)
    }

    // MARK: - Priority Ordering

    /// HuggingFace should take priority over TinyBrain when both keys present
    func testHuggingFacePriorityOverTinyBrain() throws {
        // A JSON with both "version"+"model" AND "vocab"+"merges" should be HuggingFace
        let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent("priority_test.json")
        let json = """
        {
          "version": "1.0",
          "model": {"type": "BPE", "vocab": {"a": 0}, "merges": []},
          "vocab": {"a": 0},
          "merges": []
        }
        """
        try json.write(to: tempURL, atomically: true, encoding: .utf8)
        defer { try? FileManager.default.removeItem(at: tempURL) }

        let format = TokenizerFormat.detect(at: tempURL.path)
        XCTAssertEqual(format, .huggingFace, "HuggingFace should take priority")
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

