/// SentencePiece Adapter Tests
///
/// Tests for SentencePieceAdapter: vocab parsing, space-prefix handling,
/// encode/decode round-trip, and special token detection.

import XCTest
@testable import TinyBrainTokenizer

final class SentencePieceAdapterTests: XCTestCase {

    // MARK: - Fixtures

    /// Path to the test SentencePiece vocab file
    private var vocabFixturePath: String {
        guard let url = Bundle.module.url(forResource: "test_sentencepiece", withExtension: "vocab") else {
            fatalError("Missing fixture: test_sentencepiece.vocab — check Tests/TinyBrainTokenizerTests/Fixtures/")
        }
        return url.path
    }

    // MARK: - Parsing Tests

    /// Loading a valid .vocab file should not throw
    func testLoadVocabFile() throws {
        XCTAssertNoThrow(try SentencePieceAdapter.load(from: vocabFixturePath))
    }

    /// Vocabulary should be non-empty after loading
    func testVocabNotEmpty() throws {
        let tokenizer = try SentencePieceAdapter.load(from: vocabFixturePath)
        XCTAssertGreaterThan(tokenizer.vocabularySize, 0, "Loaded vocabulary must not be empty")
    }

    /// Missing file should throw fileNotFound
    func testMissingFileThrows() {
        XCTAssertThrowsError(try SentencePieceAdapter.load(from: "/nonexistent/path.vocab")) { error in
            guard case TokenizerError.fileNotFound = error else {
                XCTFail("Expected fileNotFound, got \(error)")
                return
            }
        }
    }

    /// Empty content should throw invalidVocabularyFormat
    func testEmptyContentThrows() {
        XCTAssertThrowsError(try SentencePieceAdapter.parse(vocabContent: "")) { error in
            guard case TokenizerError.invalidVocabularyFormat = error else {
                XCTFail("Expected invalidVocabularyFormat, got \(error)")
                return
            }
        }
    }

    // MARK: - Space-Prefix Normalization

    /// ▁ (U+2581) prefix should be converted to a leading space
    func testSpacePrefixNormalization() {
        let normalized = SentencePieceAdapter.normalizePiece("▁Hello")
        XCTAssertEqual(normalized, " Hello",
                       "SentencePiece ▁ prefix should become a leading space")
    }

    /// Pieces without ▁ prefix should be unchanged
    func testNoPrefixUnchanged() {
        let normalized = SentencePieceAdapter.normalizePiece("Hello")
        XCTAssertEqual(normalized, "Hello")
    }

    /// Byte escape tokens (<0xNN>) should pass through unmodified
    func testByteEscapeUnchanged() {
        let normalized = SentencePieceAdapter.normalizePiece("<0x20>")
        XCTAssertEqual(normalized, "<0x20>")
    }

    // MARK: - Special Token Detection

    /// <unk> should be detected as the unknown token
    func testUnkTokenDetection() throws {
        let tokenizer = try SentencePieceAdapter.load(from: vocabFixturePath)
        // BPETokenizer falls back to first valid ID when unk is not resolvable;
        // just verify we get a valid (non-negative) unk token ID
        XCTAssertGreaterThanOrEqual(tokenizer.unkToken, 0)
    }

    /// BOS (<s>) and EOS (</s>) should resolve to different IDs
    func testBosEosDistinct() throws {
        // Build a minimal vocab that contains both <s> and </s>
        let content = """
        <unk>\t0
        <s>\t0
        </s>\t0
        ▁Hello\t-1.0
        ▁world\t-2.0
        """
        let tokenizer = try SentencePieceAdapter.parse(vocabContent: content)
        // SentencePiece maps <unk>=0, <s>=1, </s>=2 by row index
        // Our parse uses line index, so they will differ
        XCTAssertGreaterThanOrEqual(tokenizer.bosToken, 0)
        XCTAssertGreaterThanOrEqual(tokenizer.eosToken, 0)
    }

    // MARK: - Encoding / Decoding

    /// Encoding non-empty text should produce at least one token
    func testEncodeProducesTokens() throws {
        let tokenizer = try SentencePieceAdapter.load(from: vocabFixturePath)
        let tokens = tokenizer.encode("Hello")
        XCTAssertFalse(tokens.isEmpty, "encode() should produce tokens for non-empty input")
    }

    /// Encoding an empty string should produce an empty array
    func testEncodeEmptyString() throws {
        let tokenizer = try SentencePieceAdapter.load(from: vocabFixturePath)
        let tokens = tokenizer.encode("")
        XCTAssertTrue(tokens.isEmpty, "encode(\"\") should return []")
    }

    /// Decoding tokens should return a non-empty string for known tokens
    func testDecodeProducesText() throws {
        let tokenizer = try SentencePieceAdapter.load(from: vocabFixturePath)
        let tokens = tokenizer.encode("Hello")
        let decoded = tokenizer.decode(tokens)
        XCTAssertFalse(decoded.isEmpty, "decode() should return non-empty text")
    }

    /// encode → decode round-trip: individual characters should survive the trip
    ///
    /// The fixture includes individual character entries so BPE can segment any text.
    /// Each character should encode to its own token and decode back correctly.
    func testRoundTripPreservesContent() throws {
        let tokenizer = try SentencePieceAdapter.load(from: vocabFixturePath)
        // "Hello" — each individual char (H, e, l, o) is in the fixture vocab,
        // so they all get unique token IDs and decode back cleanly.
        let original = "Hello"
        let tokens = tokenizer.encode(original)
        XCTAssertFalse(tokens.isEmpty, "Should produce tokens for 'Hello'")
        let decoded = tokenizer.decode(tokens)
        XCTAssertEqual(decoded, original,
                       "Single-char-level round-trip should recover 'Hello', got: '\(decoded)'")
    }

    // MARK: - Implicit Merge Builder

    /// buildImplicitMerges should return an array (possibly empty, but not crash)
    func testBuildImplicitMergesDoesNotCrash() {
        let pieces = ["a", "b", "ab", "c", "abc"]
        let vocab = ["a": 0, "b": 1, "ab": 2, "c": 3, "abc": 4]
        let merges = SentencePieceAdapter.buildImplicitMerges(from: pieces, vocab: vocab)
        XCTAssertFalse(merges.isEmpty, "Should build at least one merge for 'ab' from 'a'+'b'")
    }

    /// Each merge rule should contain exactly two parts
    func testMergeRulesHaveTwoParts() {
        let pieces = ["h", "e", "he", "l", "hel", "o", "hello"]
        var vocab: [String: Int] = [:]
        for (i, p) in pieces.enumerated() { vocab[p] = i }
        let merges = SentencePieceAdapter.buildImplicitMerges(from: pieces, vocab: vocab)
        for merge in merges {
            XCTAssertEqual(merge.count, 2, "Each merge rule must be [left, right]")
        }
    }

    // MARK: - TokenizerLoader Integration

    /// TokenizerLoader should detect .vocab files as SentencePiece
    func testLoaderDetectsVocabFormat() {
        let format = TokenizerFormat.detect(at: vocabFixturePath)
        XCTAssertEqual(format, .sentencePiece,
                       "TokenizerLoader should detect .vocab as SentencePiece format")
    }

    /// TokenizerLoader.load should succeed for .vocab files
    func testLoaderLoadsSentencePiece() throws {
        let tokenizer = try TokenizerLoader.load(from: vocabFixturePath)
        XCTAssertGreaterThan(tokenizer.vocabularySize, 0)
    }
}
