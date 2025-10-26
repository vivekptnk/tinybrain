import XCTest
@testable import TinyBrainTokenizer

/// Tests for BPE tokenizer implementation
///
/// **TDD Approach:**
/// These tests are written FIRST to define expected behavior,
/// then the implementation will be written to make them pass.
///
/// **Educational Note:**
/// BPE (Byte Pair Encoding) is a subword tokenization algorithm that:
/// 1. Starts with characters as base tokens
/// 2. Iteratively merges the most frequent adjacent pairs
/// 3. Allows handling unknown words via subword units
final class BPETokenizerTests: XCTestCase {
    
    // MARK: - Test Fixtures
    
    /// Path to test vocabulary file
    private var vocabPath: String {
        // Try without subdirectory first (SPM processes Fixtures differently)
        if let url = Bundle.module.url(forResource: "test_vocab", withExtension: "json") {
            return url.path(percentEncoded: false)
        }
        
        // Try with Fixtures subdirectory
        if let url = Bundle.module.url(forResource: "test_vocab", 
                                       withExtension: "json",
                                       subdirectory: "Fixtures") {
            return url.path(percentEncoded: false)
        }
        
        // Debug: Print all bundle resource URLs
        if let resourceURL = Bundle.module.resourceURL {
            print("Bundle resources at: \(resourceURL)")
            if let files = try? FileManager.default.contentsOfDirectory(at: resourceURL, includingPropertiesForKeys: nil) {
                print("Available files: \(files)")
            }
        }
        
        fatalError("Could not find test_vocab.json in test bundle")
    }
    
    // MARK: - Basic Functionality Tests
    
    /// **Test:** BPE tokenizer can be initialized from vocabulary file
    ///
    /// **Expected behavior:**
    /// - Loads vocabulary successfully
    /// - Parses special tokens
    /// - Vocabulary size matches fixture
    func testBPETokenizerInitialization() throws {
        let tokenizer = try BPETokenizer(vocabularyPath: vocabPath)
        
        XCTAssertGreaterThan(tokenizer.vocabularySize, 0, "Vocabulary should not be empty")
        XCTAssertNotNil(tokenizer.bosToken, "Should have BOS token")
        XCTAssertNotNil(tokenizer.eosToken, "Should have EOS token")
        XCTAssertNotNil(tokenizer.unkToken, "Should have UNK token")
    }
    
    /// **Test:** Encode "Hello" to token IDs
    ///
    /// **Educational:** BPE should merge "H"+"e"→"He", "He"+"l"→"Hel", "Hel"+"l"→"Hell", "l"+"o"→"lo"
    /// Result: "Hello" = "Hell" + "lo" = [101, 103]
    func testBPEEncodeHello() throws {
        let tokenizer = try BPETokenizer(vocabularyPath: vocabPath)
        let tokens = tokenizer.encode("Hello")
        
        // Based on our test vocab merges, "Hello" should become the full token
        XCTAssertFalse(tokens.isEmpty, "Should produce tokens")
        XCTAssertTrue(tokens.contains(where: { $0 == 102 || $0 == 101 || $0 == 103 }), 
                     "Should contain Hello, Hell, or lo tokens")
    }
    
    /// **Test:** Encode "Hello, world!" to token IDs
    ///
    /// **Expected:** Should handle punctuation and spaces
    func testBPEEncodeHelloWorld() throws {
        let tokenizer = try BPETokenizer(vocabularyPath: vocabPath)
        let tokens = tokenizer.encode("Hello, world!")
        
        XCTAssertFalse(tokens.isEmpty, "Should produce tokens")
        XCTAssertGreaterThan(tokens.count, 1, "Multi-word text should produce multiple tokens")
        
        // Should NOT contain UNK token since all characters are in vocab
        XCTAssertFalse(tokens.contains(2), "Known text should not produce UNK tokens")
    }
    
    /// **Test:** Decode token IDs back to text
    ///
    /// **Educational:** Decoding is simpler than encoding - just lookup and concatenate
    func testBPEDecode() throws {
        let tokenizer = try BPETokenizer(vocabularyPath: vocabPath)
        
        // Manual token sequence (based on our vocab)
        let tokens = [102, 8, 9, 105, 13]  // Hello, world!
        let text = tokenizer.decode(tokens)
        
        XCTAssertTrue(text.contains("Hello"), "Should decode to contain 'Hello'")
    }
    
    /// **Test:** Encode → Decode round-trip preserves text
    ///
    /// **Critical property:** encode(decode(x)) should equal x for known text
    func testBPEEncodeDecodeRoundTrip() throws {
        let tokenizer = try BPETokenizer(vocabularyPath: vocabPath)
        let original = "Hello, world!"
        
        let tokens = tokenizer.encode(original)
        let roundTrip = tokenizer.decode(tokens)
        
        XCTAssertEqual(original, roundTrip, "Round-trip should preserve original text")
    }
    
    // MARK: - Special Token Tests
    
    /// **Test:** Special tokens (BOS, EOS, UNK, PAD) are accessible
    func testSpecialTokens() throws {
        let tokenizer = try BPETokenizer(vocabularyPath: vocabPath)
        
        XCTAssertEqual(tokenizer.bosToken, 0, "BOS token should be ID 0")
        XCTAssertEqual(tokenizer.eosToken, 1, "EOS token should be ID 1")
        XCTAssertEqual(tokenizer.unkToken, 2, "UNK token should be ID 2")
        XCTAssertEqual(tokenizer.padToken, 3, "PAD token should be ID 3")
    }
    
    /// **Test:** Unknown characters fallback to UNK token
    ///
    /// **Educational:** When a character isn't in vocab, BPE uses <UNK>
    func testUnknownCharacterHandling() throws {
        let tokenizer = try BPETokenizer(vocabularyPath: vocabPath)
        let tokens = tokenizer.encode("日本語")  // Japanese text not in our vocab
        
        // Should contain UNK tokens for unknown characters
        XCTAssertTrue(tokens.contains(2), "Unknown characters should produce UNK tokens")
    }
    
    // MARK: - Unicode Normalization Tests
    
    /// **Test:** Unicode normalization ensures consistency
    ///
    /// **Educational:** "café" can be represented as:
    /// - NFC (composed): c + a + f + é (single character)
    /// - NFD (decomposed): c + a + f + e + ́ (combining accent)
    /// These should tokenize identically after normalization.
    func testUnicodeNormalization() throws {
        let tokenizer = try BPETokenizer(vocabularyPath: vocabPath)
        
        // NFC form (composed)
        let nfcText = "café"
        
        // NFD form (decomposed) - manually constructed
        let nfdText = "cafe\u{0301}"  // e + combining acute accent
        
        let nfcTokens = tokenizer.encode(nfcText)
        let nfdTokens = tokenizer.encode(nfdText)
        
        XCTAssertEqual(nfcTokens, nfdTokens, 
                      "Unicode normalization should produce identical tokens")
    }
    
    /// **Test:** Emoji and extended Unicode support
    func testEmojiHandling() throws {
        let tokenizer = try BPETokenizer(vocabularyPath: vocabPath)
        let tokens = tokenizer.encode("Hello 🧠 world!")
        
        // Should handle emoji gracefully (likely as UNK)
        XCTAssertFalse(tokens.isEmpty, "Should produce tokens even with emoji")
    }
    
    // MARK: - Edge Case Tests
    
    /// **Test:** Empty string produces empty token array
    func testEmptyString() throws {
        let tokenizer = try BPETokenizer(vocabularyPath: vocabPath)
        let tokens = tokenizer.encode("")
        
        XCTAssertTrue(tokens.isEmpty, "Empty string should produce no tokens")
    }
    
    /// **Test:** Whitespace-only string
    func testWhitespaceOnly() throws {
        let tokenizer = try BPETokenizer(vocabularyPath: vocabPath)
        let tokens = tokenizer.encode("   ")
        
        // Spaces are in our vocab as token 9
        XCTAssertFalse(tokens.isEmpty, "Whitespace should produce tokens")
        XCTAssertTrue(tokens.allSatisfy({ $0 == 9 }), "Should be all space tokens")
    }
    
    /// **Test:** Very long text (stress test)
    func testLongText() throws {
        let tokenizer = try BPETokenizer(vocabularyPath: vocabPath)
        let longText = String(repeating: "Hello world! ", count: 1000)
        
        let tokens = tokenizer.encode(longText)
        
        XCTAssertGreaterThan(tokens.count, 1000, "Long text should produce many tokens")
        
        // Round-trip should work for long text too
        let decoded = tokenizer.decode(tokens)
        XCTAssertEqual(longText, decoded, "Long text round-trip should work")
    }
    
    /// **Test:** Decode handles invalid token IDs gracefully
    func testDecodeInvalidTokens() throws {
        let tokenizer = try BPETokenizer(vocabularyPath: vocabPath)
        
        // Use token ID that doesn't exist in vocab
        let invalidTokens = [9999, 8888]
        
        // Should either use UNK or skip invalid tokens, not crash
        XCTAssertNoThrow(tokenizer.decode(invalidTokens), 
                        "Should handle invalid token IDs gracefully")
    }
    
    // MARK: - Vocabulary Consistency Tests
    
    /// **Test:** Vocabulary size matches fixture
    func testVocabularySize() throws {
        let tokenizer = try BPETokenizer(vocabularyPath: vocabPath)
        
        // Our test vocab has ~40 entries
        XCTAssertGreaterThanOrEqual(tokenizer.vocabularySize, 30, 
                                    "Should have loaded all vocabulary entries")
    }
    
    /// **Test:** All merge rules are loaded
    func testMergeRulesLoaded() throws {
        let tokenizer = try BPETokenizer(vocabularyPath: vocabPath)
        
        // This is more of an integration test - verify merges work
        let text = "Hello"
        let tokens = tokenizer.encode(text)
        
        // With proper merges, "Hello" should be fewer tokens than character count
        XCTAssertLessThan(tokens.count, text.count, 
                         "BPE merges should reduce token count vs character count")
    }
    
    // MARK: - Educational Test Cases
    
    /// **Test:** TinyBrain-specific text
    ///
    /// **Educational:** Test with project-relevant vocabulary
    func testTinyBrainText() throws {
        let tokenizer = try BPETokenizer(vocabularyPath: vocabPath)
        let tokens = tokenizer.encode("TinyBrain")
        
        // Based on our vocab: "Tiny" (307) + "Brain" (310)
        XCTAssertFalse(tokens.isEmpty, "Should tokenize TinyBrain")
        
        // Should use compound tokens, not individual characters
        XCTAssertLessThanOrEqual(tokens.count, 3, 
                                "Should merge into few tokens")
    }
    
    /// **Test:** Multilingual text (within vocab)
    func testMultilingualText() throws {
        let tokenizer = try BPETokenizer(vocabularyPath: vocabPath)
        
        // Test with accented characters in vocab
        let frenchText = "café"
        let tokens = tokenizer.encode(frenchText)
        
        XCTAssertFalse(tokens.isEmpty, "Should handle accented characters")
    }
}

