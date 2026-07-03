import XCTest
@testable import TinyBrainTokenizer

/// HuggingFace tokenizer.json parity tests for SentencePiece-style BPE.
///
/// These cover the TinyLlama tokenizer behavior that differs from a plain BPE:
/// declared added/special tokens are split out before BPE, and unknown pieces use
/// model.byte_fallback UTF-8 byte tokens instead of `<unk>`.
final class HuggingFaceParityTests: XCTestCase {
    private var byteFallbackFixturePath: String {
        guard let url = Bundle.module.url(
            forResource: "hf_byte_fallback_tokenizer",
            withExtension: "json"
        ) else {
            fatalError("Missing fixture: hf_byte_fallback_tokenizer.json")
        }
        return url.path(percentEncoded: false)
    }

    func testHuggingFaceByteFallbackEncodesUnknownScalarsAsUtf8Bytes() throws {
        let tokenizer = try TokenizerLoader.loadHuggingFace(from: byteFallbackFixturePath)

        // "🧠" is not in the vocab, so byte_fallback emits its UTF-8 bytes:
        // F0 9F A7 A0. The normalizer prepends the SentencePiece space marker.
        XCTAssertEqual(tokenizer.encode("🧠"), [100, 243, 162, 170, 163])
    }

    func testHuggingFaceSpecialTokenPreSplittingProtectsAddedTokens() throws {
        let tokenizer = try TokenizerLoader.loadHuggingFace(from: byteFallbackFixturePath)

        // The literal EOS token is declared in added_tokens, so it must map to id
        // 2 exactly and must not be decomposed by BPE or byte fallback.
        XCTAssertEqual(tokenizer.encode("x</s>y"), [120, 2, 121])
    }

    func testHuggingFaceMixedSpecialTokenAndByteFallbackContent() throws {
        let tokenizer = try TokenizerLoader.loadHuggingFace(from: byteFallbackFixturePath)

        // After a special-token split, the following text segment is normalized as
        // its own segment, matching HF tokenizers' leading metaspace behavior.
        XCTAssertEqual(
            tokenizer.encode("x</s>\n🧠"),
            [120, 2, 100, 13, 243, 162, 170, 163]
        )
    }

    func testHuggingFaceByteFallbackDecodeRoundTrip() throws {
        let tokenizer = try TokenizerLoader.loadHuggingFace(from: byteFallbackFixturePath)

        XCTAssertEqual(tokenizer.decode(tokenizer.encode("🧠")), "🧠")
        XCTAssertEqual(tokenizer.decode([243, 162, 170, 163]), "🧠")
    }

    func testUnknownStillFallsBackToUnkWhenByteFallbackDisabled() throws {
        let vocab: [String: Int] = [
            "<BOS>": 0,
            "<EOS>": 1,
            "<UNK>": 2,
            "<PAD>": 3,
            "a": 4
        ]
        let tokenizer = BPETokenizer(
            vocab: vocab,
            merges: [],
            specialTokens: BPEVocabulary.SpecialTokens(
                bos_token: "<BOS>",
                eos_token: "<EOS>",
                unk_token: "<UNK>",
                pad_token: "<PAD>"
            )
        )

        XCTAssertEqual(tokenizer.encode("🧠"), [2])
    }

    func testTinyLlamaChatTemplateMatchesHuggingFaceReference() throws {
        let tokenizer = try loadRealTinyLlamaTokenizerOrSkip()
        let prompt = "<|system|>\nYou are a helpful assistant.</s>\n<|user|>\nWhat is 2+2?</s>\n<|assistant|>\n"

        // Captured with HuggingFace tokenizers 0.22.2 on 2026-07-03:
        // Tokenizer.from_file("Models/tinyllama-raw/tokenizer.json").encode(prompt).ids
        let expected = [
            1, 529, 29989, 5205, 29989, 29958, 13, 3492, 526, 263,
            8444, 20255, 29889, 2, 29871, 13, 29966, 29989, 1792,
            29989, 29958, 13, 5618, 338, 29871, 29906, 29974, 29906,
            29973, 2, 29871, 13, 29966, 29989, 465, 22137, 29989,
            29958, 13
        ]

        let actual = [tokenizer.bosToken] + tokenizer.encode(prompt)
        XCTAssertEqual(actual, expected)
        XCTAssertEqual(actual.count, 39)
        XCTAssertEqual(actual[6], 13)
    }

    func testTinyLlamaProbeStringsMatchHuggingFaceReference() throws {
        let tokenizer = try loadRealTinyLlamaTokenizerOrSkip()

        // Captured with HuggingFace tokenizers 0.22.2 on 2026-07-03.
        let cases: [(name: String, text: String, expected: [Int])] = [
            ("bare_newline", "\n", [1, 29871, 13]),
            ("emoji_brain", "🧠", [1, 29871, 243, 162, 170, 163]),
            ("eos_mid_text", "alpha</s>beta", [1, 15595, 2, 21762]),
            ("beyond_bmp", "𐍈", [1, 29871, 243, 147, 144, 139]),
            ("empty", "", [1]),
            ("plain_ascii", "Hello world", [1, 15043, 3186]),
            ("unicode_cafe", "café", [1, 274, 28059]),
            ("tabs_newlines", "a\tb\nc", [1, 263, 12, 29890, 13, 29883]),
            ("chat_marker_plain", "<|user|>", [1, 529, 29989, 1792, 29989, 29958])
        ]

        for testCase in cases {
            let actual = [tokenizer.bosToken] + tokenizer.encode(testCase.text)
            XCTAssertEqual(actual, testCase.expected, "Mismatch for \(testCase.name)")
        }
    }

    private func loadRealTinyLlamaTokenizerOrSkip() throws -> BPETokenizer {
        let path = resolveProjectPath("Models/tinyllama-raw/tokenizer.json")
        guard FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("Models/tinyllama-raw/tokenizer.json not available")
        }
        return try TokenizerLoader.loadHuggingFace(from: path)
    }

    private func resolveProjectPath(_ path: String) -> String {
        if FileManager.default.fileExists(atPath: path) {
            return path
        }

        var directory = FileManager.default.currentDirectoryPath
        for _ in 0..<10 {
            let packagePath = (directory as NSString).appendingPathComponent("Package.swift")
            if FileManager.default.fileExists(atPath: packagePath) {
                let resolved = (directory as NSString).appendingPathComponent(path)
                if FileManager.default.fileExists(atPath: resolved) {
                    return resolved
                }
            }

            let parent = (directory as NSString).deletingLastPathComponent
            if parent == directory || parent == "/" {
                break
            }
            directory = parent
        }

        return path
    }
}
