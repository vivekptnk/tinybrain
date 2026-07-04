/// HuggingFace Tokenizer Adapter
///
/// **TB-009:** Converts HuggingFace tokenizer.json format to BPETokenizer
///
/// Handles:
/// - Complex nested JSON structure
/// - Added tokens, special tokens
/// - Byte-level BPE (like GPT-2, Llama)
/// - Pre-tokenizers and normalizers

import Foundation

/// Adapter for HuggingFace tokenizer.json format
public enum HuggingFaceAdapter {
    
    /// Load HuggingFace tokenizer.json and convert to BPETokenizer
    ///
    /// - Parameter path: Path to tokenizer.json file
    /// - Returns: BPETokenizer configured from HF format
    /// - Throws: TokenizerError if parsing fails
    public static func load(from path: String) throws -> BPETokenizer {
        guard FileManager.default.fileExists(atPath: path) else {
            throw TokenizerError.fileNotFound(path)
        }
        
        let url = URL(fileURLWithPath: path)
        let data = try Data(contentsOf: url)
        
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw TokenizerError.invalidJSON
        }
        
        // Parse HuggingFace structure
        guard let model = json["model"] as? [String: Any] else {
            throw TokenizerError.missingRequiredField("model")
        }

        let tokenizerConfig = loadSiblingTokenizerConfig(for: url)
        
        // Extract vocabulary
        let vocab = try extractVocabulary(from: model, addedTokens: json["added_tokens"] as? [[String: Any]])
        
        // Extract merges
        let merges = try extractMerges(from: model)
        
        // Extract special tokens
        let specialTokens = extractSpecialTokens(
            from: json,
            tokenizerConfig: tokenizerConfig,
            model: model,
            vocab: vocab
        )

        // Extract HF model behavior that affects parity with SentencePiece BPE.
        let byteFallback = (model["byte_fallback"] as? Bool) ?? false
        let preTokenizedTokens = extractPreTokenizedTokens(
            from: json["added_tokens"] as? [[String: Any]],
            vocab: vocab
        )
        let usesSentencePieceWhitespace = extractSentencePieceWhitespaceNormalizer(from: json)
        let byteLevelConfiguration = extractByteLevelConfiguration(from: json, model: model)
        let appliesNFC = extractAppliesNFCNormalizer(from: json)
        let addsBosToken = extractAddsBosToken(
            from: tokenizerConfig,
            json: json,
            bosToken: specialTokens.bos_token
        )
        
        #if DEBUG
        print("📖 Loaded HuggingFace tokenizer:")
        print("   Vocabulary size: \(vocab.count)")
        print("   Merge rules: \(merges.count)")
        print("   Special tokens: BOS=\(specialTokens.bos_token ?? "none"), EOS=\(specialTokens.eos_token ?? "none")")
        print("   Byte fallback: \(byteFallback)")
        print("   ByteLevel: \(byteLevelConfiguration.enabled)")
        print("   NFC normalizer: \(appliesNFC)")
        print("   Pre-tokenized added tokens: \(preTokenizedTokens.count)")
        #endif
        
        return BPETokenizer(
            vocab: vocab,
            merges: merges,
            specialTokens: specialTokens,
            byteFallback: byteFallback,
            preTokenizedTokens: preTokenizedTokens,
            usesSentencePieceWhitespace: usesSentencePieceWhitespace,
            byteLevel: byteLevelConfiguration.enabled,
            byteLevelPattern: byteLevelConfiguration.splitRegexPattern,
            addsBosToken: addsBosToken,
            appliesNFC: appliesNFC
        )
    }
    
    // MARK: - Vocabulary Extraction
    
    private static func extractVocabulary(
        from model: [String: Any],
        addedTokens: [[String: Any]]?
    ) throws -> [String: Int] {
        var vocab: [String: Int] = [:]
        
        // Extract main vocabulary
        // Note: Swift's JSON parsing may drop some entries due to encoding issues
        // We'll handle this by trying multiple approaches
        if let vocabDict = model["vocab"] as? [String: Int] {
            vocab = vocabDict
        } else if let vocabDict = model["vocab"] as? [String: Any] {
            // Some formats have nested structure
            var skipped = 0
            for (token, value) in vocabDict {
                if let id = value as? Int {
                    vocab[token] = id
                } else {
                    skipped += 1
                }
            }
            #if DEBUG
            if skipped > 0 {
                print("⚠️ Skipped \(skipped) non-integer vocab entries")
            }
            #endif
        } else if let vocabDict = model["vocab"] as? [AnyHashable: Any] {
            // Handle cases where keys might not be strings
            var skipped = 0
            for (key, value) in vocabDict {
                if let token = key as? String, let id = value as? Int {
                    vocab[token] = id
                } else if let id = value as? Int {
                    let token = String(describing: key)
                    vocab[token] = id
                } else {
                    skipped += 1
                }
            }
            #if DEBUG
            if skipped > 0 {
                print("⚠️ Skipped \(skipped) invalid vocab entries")
            }
            #endif
        } else {
            throw TokenizerError.missingRequiredField("model.vocab")
        }
        
        // If we're missing tokens, pad with <unk> variants to match model vocab size
        let expectedVocabSize = 32000
        if vocab.count < expectedVocabSize {
            let missing = expectedVocabSize - vocab.count
            #if DEBUG
            print("⚠️ Missing \(missing) tokens from vocab, padding with placeholders")
            #endif

            // Add placeholder tokens for missing IDs
            var usedIds = Set(vocab.values)
            for id in 0..<expectedVocabSize {
                if !usedIds.contains(id) {
                    vocab["<unk_\(id)>"] = id
                    usedIds.insert(id)
                }
            }
        }
        
        #if DEBUG
        print("🔍 Loaded \(vocab.count) vocabulary entries from main vocab")
        #endif
        
        // Add added_tokens (special tokens added post-training)
        if let added = addedTokens {
            var addedCount = 0
            for tokenInfo in added {
                if let content = tokenInfo["content"] as? String,
                   let id = tokenInfo["id"] as? Int {
                    vocab[content] = id
                    addedCount += 1
                }
            }
            #if DEBUG
            print("🔍 Added \(addedCount) special tokens")
            #endif
        }

        #if DEBUG
        print("🔍 Final vocabulary size: \(vocab.count) tokens")
        #endif
        
        return vocab
    }
    
    // MARK: - Merge Rules Extraction
    
    private static func extractMerges(from model: [String: Any]) throws -> [[String]] {
        guard let mergesList = model["merges"] as? [String] else {
            // Some models might not have merges (character-level)
            return []
        }
        
        // Convert "a b" string format to [["a", "b"]]
        var merges: [[String]] = []
        for mergeStr in mergesList {
            let parts = mergeStr.split(separator: " ").map(String.init)
            if parts.count == 2 {
                merges.append(parts)
            }
        }
        
        return merges
    }
    
    // MARK: - Special Tokens Extraction
    
    private static func extractSpecialTokens(
        from json: [String: Any],
        tokenizerConfig: [String: Any]?,
        model: [String: Any],
        vocab: [String: Int]
    ) -> BPEVocabulary.SpecialTokens {
        // Look in multiple possible locations
        let bosTokenConfigured = tokenizerConfig?.keys.contains("bos_token") == true
        let shouldInferBosToken = !bosTokenConfigured || extractTokenContent(tokenizerConfig?["bos_token"]) != nil

        var bosToken = extractTokenContent(tokenizerConfig?["bos_token"])
        var eosToken = extractTokenContent(tokenizerConfig?["eos_token"])
        var unkToken = extractTokenContent(tokenizerConfig?["unk_token"]) ?? (model["unk_token"] as? String)
        var padToken = extractTokenContent(tokenizerConfig?["pad_token"])
        
        // Check added_tokens array
        if let added = json["added_tokens"] as? [[String: Any]] {
            for tokenInfo in added {
                if let content = tokenInfo["content"] as? String,
                   let special = tokenInfo["special"] as? Bool,
                   special {
                    // Match by name patterns
                    let lower = content.lowercased()
                    if shouldInferBosToken,
                       lower.contains("bos") || lower.contains("<s>") || lower == "<|begin_of_text|>" {
                        bosToken = bosToken ?? content
                    }
                    if lower == "<|im_end|>" {
                        eosToken = content
                    }
                    if lower.contains("eos") || lower.contains("</s>") || lower == "<|end_of_text|>" {
                        eosToken = eosToken ?? content
                    }
                    if lower.contains("unk") || lower == "<unk>" {
                        unkToken = unkToken ?? content
                    }
                    if lower.contains("pad") || lower == "<pad>" {
                        padToken = padToken ?? content
                    }
                    if lower == "<|endoftext|>" {
                        padToken = padToken ?? content
                        eosToken = eosToken ?? content
                    }
                }
            }
        }
        
        // Check post_processor (another common location)
        if let postProcessor = json["post_processor"] as? [String: Any] {
            if let single = postProcessor["single"] as? [[String: Any]] {
                for item in single {
                    if let specialToken = item["SpecialToken"] as? [String: Any],
                       let id = specialToken["id"] as? String {
                        if shouldInferBosToken, id.contains("bos") || id == "<s>" {
                            bosToken = id
                        }
                        if id.contains("eos") || id == "</s>" {
                            eosToken = id
                        }
                    }
                }
            }
        }
        
        // Fallback: Look for common patterns in vocab
        if bosToken == nil, shouldInferBosToken {
            if vocab["<s>"] != nil {
                bosToken = "<s>"
            } else if vocab["<BOS>"] != nil {
                bosToken = "<BOS>"
            } else if vocab["<|begin_of_text|>"] != nil {
                bosToken = "<|begin_of_text|>"
            }
        }
        
        if eosToken == nil {
            if vocab["</s>"] != nil {
                eosToken = "</s>"
            } else if vocab["<EOS>"] != nil {
                eosToken = "<EOS>"
            } else if vocab["<|end_of_text|>"] != nil {
                eosToken = "<|end_of_text|>"
            }
        }
        
        if unkToken == nil {
            if vocab["<unk>"] != nil {
                unkToken = "<unk>"
            } else if vocab["<UNK>"] != nil {
                unkToken = "<UNK>"
            }
        }
        
        if padToken == nil {
            if vocab["<pad>"] != nil {
                padToken = "<pad>"
            } else if vocab["<PAD>"] != nil {
                padToken = "<PAD>"
            }
        }
        
        return BPEVocabulary.SpecialTokens(
            bos_token: bosToken,
            eos_token: eosToken,
            unk_token: unkToken,
            pad_token: padToken
        )
    }

    private static func loadSiblingTokenizerConfig(for tokenizerURL: URL) -> [String: Any]? {
        let configURL = tokenizerURL
            .deletingLastPathComponent()
            .appendingPathComponent("tokenizer_config.json")

        guard FileManager.default.fileExists(atPath: configURL.path),
              let data = try? Data(contentsOf: configURL),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return nil
        }

        return json
    }

    private static func extractTokenContent(_ value: Any?) -> String? {
        if value == nil || value is NSNull {
            return nil
        }

        if let string = value as? String {
            return string
        }

        if let dictionary = value as? [String: Any] {
            return dictionary["content"] as? String
        }

        return nil
    }

    private static func extractAddsBosToken(
        from tokenizerConfig: [String: Any]?,
        json: [String: Any],
        bosToken: String?
    ) -> Bool {
        if let addBosToken = tokenizerConfig?["add_bos_token"] as? Bool {
            return addBosToken
        }

        guard let bosToken,
              let postProcessor = json["post_processor"] as? [String: Any],
              let single = postProcessor["single"] as? [[String: Any]] else {
            return false
        }

        return single.contains { item in
            guard let specialToken = item["SpecialToken"] as? [String: Any],
                  let id = specialToken["id"] as? String else {
                return false
            }
            return id == bosToken
        }
    }

    // MARK: - ByteLevel Extraction

    private struct ByteLevelConfiguration {
        let enabled: Bool
        let splitRegexPattern: String?
    }

    private static func extractByteLevelConfiguration(
        from json: [String: Any],
        model: [String: Any]
    ) -> ByteLevelConfiguration {
        guard (model["type"] as? String) == "BPE",
              tokenizerComponent(json["pre_tokenizer"], containsType: "ByteLevel"),
              (
                tokenizerComponent(json["decoder"], containsType: "ByteLevel") ||
                    tokenizerComponent(json["post_processor"], containsType: "ByteLevel")
              ) else {
            return ByteLevelConfiguration(enabled: false, splitRegexPattern: nil)
        }

        return ByteLevelConfiguration(
            enabled: true,
            splitRegexPattern: firstSplitRegexPattern(in: json["pre_tokenizer"])
        )
    }

    private static func tokenizerComponent(_ value: Any?, containsType expectedType: String) -> Bool {
        if value == nil || value is NSNull {
            return false
        }

        if let dictionary = value as? [String: Any] {
            if dictionary["type"] as? String == expectedType {
                return true
            }

            for key in ["pretokenizers", "decoders", "normalizers", "processors"] {
                if tokenizerComponent(dictionary[key], containsType: expectedType) {
                    return true
                }
            }
        }

        if let array = value as? [[String: Any]] {
            return array.contains { tokenizerComponent($0, containsType: expectedType) }
        }

        return false
    }

    private static func firstSplitRegexPattern(in component: Any?) -> String? {
        if component == nil || component is NSNull {
            return nil
        }

        if let dictionary = component as? [String: Any] {
            if dictionary["type"] as? String == "Split" {
                return regexPatternString(from: dictionary["pattern"])
            }

            for key in ["pretokenizers", "decoders", "normalizers", "processors"] {
                if let pattern = firstSplitRegexPattern(in: dictionary[key]) {
                    return pattern
                }
            }
        }

        if let array = component as? [[String: Any]] {
            for child in array {
                if let pattern = firstSplitRegexPattern(in: child) {
                    return pattern
                }
            }
        }

        return nil
    }

    private static func regexPatternString(from pattern: Any?) -> String? {
        if let string = pattern as? String {
            return string
        }

        if let dictionary = pattern as? [String: Any] {
            return dictionary["Regex"] as? String
        }

        return nil
    }

    // MARK: - Added Tokens / Normalizer Extraction

    /// Extract added tokens that HuggingFace matches before model tokenization.
    ///
    /// Added tokens are not merge candidates; they are split out first and mapped
    /// directly to their declared IDs. This is what keeps `</s>` at id 2 instead
    /// of letting the text `</s>` flow through BPE merges.
    private static func extractPreTokenizedTokens(
        from addedTokens: [[String: Any]]?,
        vocab: [String: Int]
    ) -> Set<String> {
        guard let addedTokens else {
            return []
        }

        var tokens: Set<String> = []
        for tokenInfo in addedTokens {
            guard let content = tokenInfo["content"] as? String,
                  let id = tokenInfo["id"] as? Int,
                  vocab[content] == id else {
                continue
            }

            tokens.insert(content)
        }

        return tokens
    }

    /// Detect the common LLaMA/TinyLlama normalizer:
    /// `Prepend("▁")` plus replacing spaces with `▁`.
    ///
    /// A nil return lets BPETokenizer fall back to vocabulary-based detection for
    /// non-HF adapters and small TinyBrain fixtures.
    private static func extractSentencePieceWhitespaceNormalizer(from json: [String: Any]) -> Bool? {
        guard let normalizer = json["normalizer"] as? [String: Any] else {
            return nil
        }

        return normalizerUsesSentencePieceWhitespace(normalizer)
    }

    private static func normalizerUsesSentencePieceWhitespace(_ normalizer: [String: Any]) -> Bool {
        if let type = normalizer["type"] as? String {
            if type == "Prepend", normalizer["prepend"] as? String == "\u{2581}" {
                return true
            }

            if type == "Replace",
               let pattern = normalizer["pattern"] as? [String: Any],
               pattern["String"] as? String == " ",
               normalizer["content"] as? String == "\u{2581}" {
                return true
            }
        }

        if let normalizers = normalizer["normalizers"] as? [[String: Any]] {
            return normalizers.contains { normalizerUsesSentencePieceWhitespace($0) }
        }

        return false
    }

    private static func extractAppliesNFCNormalizer(from json: [String: Any]) -> Bool {
        guard let normalizer = json["normalizer"], !(normalizer is NSNull) else {
            return false
        }

        return normalizerAppliesNFC(normalizer)
    }

    private static func normalizerAppliesNFC(_ normalizer: Any?) -> Bool {
        if normalizer == nil || normalizer is NSNull {
            return false
        }

        if let dictionary = normalizer as? [String: Any] {
            if dictionary["type"] as? String == "NFC" {
                return true
            }

            if let normalizers = dictionary["normalizers"] as? [[String: Any]] {
                return normalizers.contains { normalizerAppliesNFC($0) }
            }
        }

        if let normalizers = normalizer as? [[String: Any]] {
            return normalizers.contains { normalizerAppliesNFC($0) }
        }

        return false
    }
}
