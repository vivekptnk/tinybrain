import XCTest
@testable import TinyBrainTokenizer

/// Tests for tokenizer edge cases and bug fixes
///
/// **REVIEW HITLER:** These tests expose critical bugs in the initial implementation
final class TokenizerBugTests: XCTestCase {
    
    // MARK: - Bug #1: Hard-coded Special Token IDs
    
    /// **Bug:** Tokenizer assumes IDs 0,1,2,3 for special tokens
    /// even when vocab doesn't have those entries
    ///
    /// **Expected:** Should validate special tokens exist in vocab
    /// or use actual vocab entries
    func testSpecialTokensMustExistInVocab() throws {
        // Create vocab with NO special tokens section
        let vocabJSON = """
        {
          "vocab": {
            "a": 10,
            "b": 20,
            "c": 30
          },
          "merges": []
        }
        """
        
        let tempFile = NSTemporaryDirectory() + "no_special_tokens.json"
        try vocabJSON.write(toFile: tempFile, atomically: true, encoding: .utf8)
        defer { try? FileManager.default.removeItem(atPath: tempFile) }
        
        // This should either:
        // 1. Throw an error (no special tokens defined)
        // 2. Use valid fallback from actual vocab (e.g., first token)
        //
        // It should NOT use IDs 0,1,2,3 which don't exist in vocab!
        do {
            let tokenizer = try BPETokenizer(vocabularyPath: tempFile)
            
            // If it doesn't throw, special tokens must be valid vocab entries
            let allValidIds = Set([10, 20, 30])
            XCTAssertTrue(allValidIds.contains(tokenizer.bosToken), 
                         "BOS token must be a valid vocab entry")
            XCTAssertTrue(allValidIds.contains(tokenizer.eosToken),
                         "EOS token must be a valid vocab entry")
            XCTAssertTrue(allValidIds.contains(tokenizer.unkToken),
                         "UNK token must be a valid vocab entry")
            XCTAssertTrue(allValidIds.contains(tokenizer.padToken),
                         "PAD token must be a valid vocab entry")
        } catch TokenizerError.vocabularyNotFound {
            XCTFail("File should exist")
        } catch {
            // Throwing error for missing special tokens is acceptable
            XCTAssertTrue(true, "Acceptable to require special tokens")
        }
    }
    
    /// **Bug:** Decode with hard-coded special token IDs returns empty string
    /// when those IDs don't exist in vocab
    func testDecodeWithInvalidSpecialTokens() throws {
        let vocabJSON = """
        {
          "vocab": {
            "hello": 100,
            "world": 101
          },
          "merges": []
        }
        """
        
        let tempFile = NSTemporaryDirectory() + "minimal_vocab.json"
        try vocabJSON.write(toFile: tempFile, atomically: true, encoding: .utf8)
        defer { try? FileManager.default.removeItem(atPath: tempFile) }
        
        let tokenizer = try BPETokenizer(vocabularyPath: tempFile)
        
        // Try to decode using the tokenizer's special token IDs
        let decoded = tokenizer.decode([tokenizer.bosToken, 100, tokenizer.eosToken])
        
        // Should either include the special tokens (if they exist in vocab)
        // or gracefully handle missing entries
        XCTAssertTrue(decoded.contains("hello") || decoded == "hello",
                     "Should decode valid token 100")
    }
}

