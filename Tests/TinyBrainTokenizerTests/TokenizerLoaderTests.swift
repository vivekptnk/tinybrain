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
    
    func testLoadBestAvailable() throws {
        let fileManager = FileManager.default
        let tempRoot = fileManager.temporaryDirectory.appendingPathComponent("TinyBrainTokenizerTests-\(UUID().uuidString)")
        let modelsDirectory = tempRoot.appendingPathComponent("Models")
        let tinyLlamaDirectory = modelsDirectory.appendingPathComponent("tinyllama-raw")
        try fileManager.createDirectory(at: tinyLlamaDirectory, withIntermediateDirectories: true)
        defer { try? fileManager.removeItem(at: tempRoot) }

        let tokenizerURL = tinyLlamaDirectory.appendingPathComponent("tokenizer.json")
        let tokenizerJSON = """
        {
          "vocab": {
            "<BOS>": 0,
            "<EOS>": 1,
            "<UNK>": 2,
            "<PAD>": 3,
            "H": 4,
            "i": 5,
            "Hi": 6
          },
          "merges": [["H", "i"]],
          "special_tokens": {
            "bos_token": "<BOS>",
            "eos_token": "<EOS>",
            "unk_token": "<UNK>",
            "pad_token": "<PAD>"
          }
        }
        """
        try tokenizerJSON.write(to: tokenizerURL, atomically: true, encoding: .utf8)

        // Should find and load the best tokenizer from an explicit small Models fixture.
        let tokenizer = TokenizerLoader.loadBestAvailable(in: modelsDirectory.path)
        
        // Should at least return something (even if fallback)
        XCTAssertGreaterThan(tokenizer.vocabularySize, 0)
        XCTAssertEqual(tokenizer.encode("Hi"), [6])
    }

    func testLoadTokenizerForModelPrefersMatchingRawDirectory() throws {
        let fileManager = FileManager.default
        let tempRoot = fileManager.temporaryDirectory.appendingPathComponent("TinyBrainTokenizerPairing-\(UUID().uuidString)")
        let modelsDirectory = tempRoot.appendingPathComponent("Models")
        let tinyLlamaDirectory = modelsDirectory.appendingPathComponent("tinyllama-raw")
        let gemmaDirectory = modelsDirectory.appendingPathComponent("gemma-2b-raw")
        try fileManager.createDirectory(at: tinyLlamaDirectory, withIntermediateDirectories: true)
        try fileManager.createDirectory(at: gemmaDirectory, withIntermediateDirectories: true)
        defer { try? fileManager.removeItem(at: tempRoot) }

        try writeTinyBrainTokenizer(
            to: tinyLlamaDirectory.appendingPathComponent("tokenizer.json"),
            token: "T",
            mergedToken: "Ti"
        )
        try writeTinyBrainTokenizer(
            to: gemmaDirectory.appendingPathComponent("tokenizer.json"),
            token: "G",
            mergedToken: "Go"
        )

        let modelURL = modelsDirectory.appendingPathComponent("gemma-2b-int8.tbf")
        fileManager.createFile(atPath: modelURL.path, contents: Data([0]))

        let tokenizer = try TokenizerLoader.loadTokenizer(forModelAt: modelURL)

        XCTAssertEqual(tokenizer.encode("Go"), [6])
        XCTAssertNotEqual(tokenizer.encode("Ti"), [6], "Gemma model must not pick the TinyLlama tokenizer")
    }

    func testLoadTokenizerForModelFallsBackToKnownTinyLlamaRawName() throws {
        let fileManager = FileManager.default
        let tempRoot = fileManager.temporaryDirectory.appendingPathComponent("TinyBrainTokenizerTinyLlama-\(UUID().uuidString)")
        let modelsDirectory = tempRoot.appendingPathComponent("Models")
        let tinyLlamaDirectory = modelsDirectory.appendingPathComponent("tinyllama-raw")
        try fileManager.createDirectory(at: tinyLlamaDirectory, withIntermediateDirectories: true)
        defer { try? fileManager.removeItem(at: tempRoot) }

        try writeTinyBrainTokenizer(
            to: tinyLlamaDirectory.appendingPathComponent("tokenizer.json"),
            token: "H",
            mergedToken: "Hi"
        )

        let modelURL = modelsDirectory.appendingPathComponent("tinyllama-1.1b-int8.tbf")
        fileManager.createFile(atPath: modelURL.path, contents: Data([0]))

        let tokenizer = try TokenizerLoader.loadTokenizer(forModelAt: modelURL)

        XCTAssertEqual(tokenizer.encode("Hi"), [6])
    }

    func testLoadTokenizerForModelFailsLoudlyWhenMissing() throws {
        let fileManager = FileManager.default
        let tempRoot = fileManager.temporaryDirectory.appendingPathComponent("TinyBrainTokenizerMissing-\(UUID().uuidString)")
        let modelsDirectory = tempRoot.appendingPathComponent("Models")
        try fileManager.createDirectory(at: modelsDirectory, withIntermediateDirectories: true)
        defer { try? fileManager.removeItem(at: tempRoot) }

        let modelURL = modelsDirectory.appendingPathComponent("gemma-2b-int8.tbf")
        fileManager.createFile(atPath: modelURL.path, contents: Data([0]))

        XCTAssertThrowsError(try TokenizerLoader.loadTokenizer(forModelAt: modelURL)) { error in
            let message = String(describing: error)
            XCTAssertTrue(message.contains("No tokenizer found for gemma-2b-int8.tbf"))
            XCTAssertTrue(message.contains("gemma-2b-raw/tokenizer.json"))
            XCTAssertTrue(message.contains("Decoding with a mismatched tokenizer would produce garbage"))
        }
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

    private func writeTinyBrainTokenizer(to url: URL, token: String, mergedToken: String) throws {
        let first = String(mergedToken.prefix(1))
        let rest = String(mergedToken.dropFirst())
        let tokenizerJSON = """
        {
          "vocab": {
            "<BOS>": 0,
            "<EOS>": 1,
            "<UNK>": 2,
            "<PAD>": 3,
            "\(token)": 4,
            "\(rest)": 5,
            "\(mergedToken)": 6
          },
          "merges": [["\(first)", "\(rest)"]],
          "special_tokens": {
            "bos_token": "<BOS>",
            "eos_token": "<EOS>",
            "unk_token": "<UNK>",
            "pad_token": "<PAD>"
          }
        }
        """
        try tokenizerJSON.write(to: url, atomically: true, encoding: .utf8)
    }
}
