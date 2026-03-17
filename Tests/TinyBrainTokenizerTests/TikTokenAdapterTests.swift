/// TikToken Adapter Tests
///
/// Tests for TikTokenAdapter: base64 parsing, vocab construction,
/// merge reconstruction, encode/decode round-trips, and
/// TokenizerLoader integration.

import XCTest
@testable import TinyBrainTokenizer

final class TikTokenAdapterTests: XCTestCase {

    // MARK: - Fixtures

    /// Raw TikToken content for unit tests (no file I/O needed)
    ///
    /// Each entry is: base64(utf8_bytes) + " " + rank
    /// Tokens:
    ///   SA==  = "H"  (0x48)
    ///   ZQ==  = "e"  (0x65)
    ///   bA==  = "l"  (0x6C)
    ///   bw==  = "o"  (0x6F)
    ///   SGU=  = "He"
    ///   SGVs  = "Hel"
    ///   SGVsbG8= = "Hello"
    ///   IA==  = " "  (0x20)
    ///   dw==  = "w"  (0x77)
    ///   cg==  = "r"  (0x72)
    ///   ZA==  = "d"  (0x64)
    ///   d29y  = "wor"
    ///   d29ybGQ= = "world"
    private let minimalContent = """
    SA== 0
    ZQ== 1
    bA== 2
    bw== 3
    SGU= 4
    SGVs 5
    SGVsbG8= 6
    IA== 7
    dw== 8
    cg== 9
    ZA== 10
    d29y 11
    d29ybGQ= 12
    """

    /// Path to the fixture file
    private var fixturePath: String {
        guard let url = Bundle.module.url(forResource: "test_tiktoken", withExtension: "tiktoken") else {
            fatalError("Missing fixture: test_tiktoken.tiktoken")
        }
        return url.path
    }

    // MARK: - Parsing Tests

    /// Parsing valid TikToken content should succeed
    func testParseValidContent() throws {
        XCTAssertNoThrow(try TikTokenAdapter.parse(tikTokenContent: minimalContent))
    }

    /// Vocabulary should contain the expected number of entries (plus any added specials)
    func testVocabSize() throws {
        let tokenizer = try TikTokenAdapter.parse(tikTokenContent: minimalContent)
        // 13 entries + 2 added special tokens (<|endoftext|>, <|unk|>)
        XCTAssertGreaterThanOrEqual(tokenizer.vocabularySize, 13)
    }

    /// Empty content should throw
    func testEmptyContentThrows() {
        XCTAssertThrowsError(try TikTokenAdapter.parse(tikTokenContent: "")) { error in
            guard case TokenizerError.invalidVocabularyFormat = error else {
                XCTFail("Expected invalidVocabularyFormat, got \(error)")
                return
            }
        }
    }

    /// Content with only blank/comment lines should throw
    func testAllInvalidLinesThrows() {
        let bad = """
        not-base64 abc
        also-not-base64 xyz
        """
        XCTAssertThrowsError(try TikTokenAdapter.parse(tikTokenContent: bad))
    }

    // MARK: - Base64 Token Conversion

    /// Single ASCII byte should decode to its character
    func testSingleByteToken() {
        // "H" = 0x48 → base64 "SA=="
        let data = Data(base64Encoded: "SA==")!
        let str = TikTokenAdapter.tokenStringFromData(data)
        XCTAssertEqual(str, "H")
    }

    /// Multi-byte UTF-8 sequence should decode as the corresponding Unicode string
    func testMultiByteToken() {
        // "Hello" → base64 "SGVsbG8="
        let data = Data(base64Encoded: "SGVsbG8=")!
        let str = TikTokenAdapter.tokenStringFromData(data)
        XCTAssertEqual(str, "Hello")
    }

    /// Invalid UTF-8 bytes should fall back to <0xNN> escape notation
    func testInvalidUTF8FallsBackToEscape() {
        // 0xFF is not valid UTF-8
        let data = Data([0xFF])
        let str = TikTokenAdapter.tokenStringFromData(data)
        XCTAssertEqual(str, "<0xFF>")
    }

    // MARK: - Token Parts Splitting

    /// Single ASCII character should split into one part
    func testSingleCharParts() {
        let parts = TikTokenAdapter.tokenParts("H")
        XCTAssertEqual(parts, ["H"])
    }

    /// Multi-character string should split into one part per character
    func testMultiCharParts() {
        let parts = TikTokenAdapter.tokenParts("Hello")
        XCTAssertEqual(parts, ["H", "e", "l", "l", "o"])
    }

    /// Byte-escape string should be treated as one atomic part
    func testByteEscapePart() {
        let parts = TikTokenAdapter.tokenParts("<0xFF>")
        XCTAssertEqual(parts, ["<0xFF>"])
    }

    /// Mix of regular chars and byte escapes should split correctly
    func testMixedParts() {
        let parts = TikTokenAdapter.tokenParts("a<0x20>b")
        XCTAssertEqual(parts, ["a", "<0x20>", "b"])
    }

    // MARK: - Merge Reconstruction

    /// buildMerges should return rules for tokens with splittable parts
    func testBuildMergesNonEmpty() throws {
        let tokenizer = try TikTokenAdapter.parse(tikTokenContent: minimalContent)
        // Just verify the tokenizer was built without crash and has vocabulary
        XCTAssertGreaterThan(tokenizer.vocabularySize, 5)
    }

    // MARK: - Special Tokens

    /// <|endoftext|> should be added as BOS/EOS token
    func testSpecialTokensAdded() throws {
        let tokenizer = try TikTokenAdapter.parse(tikTokenContent: minimalContent)
        // BOS and EOS should map to the same <|endoftext|> token (GPT style)
        XCTAssertEqual(tokenizer.bosToken, tokenizer.eosToken,
                       "GPT-style tokenizer uses same token for BOS and EOS")
    }

    // MARK: - Encoding / Decoding

    /// encode("Hello") should produce at least one token
    func testEncodeProducesTokens() throws {
        let tokenizer = try TikTokenAdapter.parse(tikTokenContent: minimalContent)
        let tokens = tokenizer.encode("Hello")
        XCTAssertFalse(tokens.isEmpty, "encode() must produce tokens")
    }

    /// encode("") should return empty array
    func testEncodeEmptyString() throws {
        let tokenizer = try TikTokenAdapter.parse(tikTokenContent: minimalContent)
        XCTAssertTrue(tokenizer.encode("").isEmpty)
    }

    /// decode should return non-empty text for a known token ID
    func testDecodeKnownToken() throws {
        let tokenizer = try TikTokenAdapter.parse(tikTokenContent: minimalContent)
        // Token rank 6 is "Hello" (SGVsbG8= with rank 6)
        let decoded = tokenizer.decode([6])
        XCTAssertEqual(decoded, "Hello", "Token 6 should decode to 'Hello'")
    }

    /// Encode → decode should preserve the content
    func testEncodeDecodeRoundTrip() throws {
        let tokenizer = try TikTokenAdapter.parse(tikTokenContent: minimalContent)
        let original = "Hello"
        let tokens = tokenizer.encode(original)
        let decoded = tokenizer.decode(tokens)
        XCTAssertEqual(decoded, original,
                       "Round-trip should recover '\(original)', got '\(decoded)'")
    }

    // MARK: - File Loading

    /// Loading from the fixture file should succeed
    func testLoadFromFile() throws {
        XCTAssertNoThrow(try TikTokenAdapter.load(from: fixturePath))
    }

    /// Missing file should throw fileNotFound
    func testMissingFileThrows() {
        XCTAssertThrowsError(try TikTokenAdapter.load(from: "/no/such/file.tiktoken")) { error in
            guard case TokenizerError.fileNotFound = error else {
                XCTFail("Expected fileNotFound, got \(error)")
                return
            }
        }
    }

    // MARK: - TokenizerLoader Integration

    /// TokenizerLoader should detect .tiktoken extension as tiktoken format
    func testLoaderDetectsTikTokenFormat() {
        let format = TokenizerFormat.detect(at: fixturePath)
        XCTAssertEqual(format, .tiktoken,
                       "TokenizerLoader should detect .tiktoken as tiktoken format")
    }

    /// TokenizerLoader.load should succeed for .tiktoken files
    func testLoaderLoadsTikToken() throws {
        let tokenizer = try TokenizerLoader.load(from: fixturePath)
        XCTAssertGreaterThan(tokenizer.vocabularySize, 0)
    }
}
