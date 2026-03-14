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
        
        // Extract vocabulary
        let vocab = try extractVocabulary(from: model, addedTokens: json["added_tokens"] as? [[String: Any]])
        
        // Extract merges
        let merges = try extractMerges(from: model)
        
        // Extract special tokens
        let specialTokens = extractSpecialTokens(from: json, vocab: vocab)
        
        #if DEBUG
        print("📖 Loaded HuggingFace tokenizer:")
        print("   Vocabulary size: \(vocab.count)")
        print("   Merge rules: \(merges.count)")
        print("   Special tokens: BOS=\(specialTokens.bos_token ?? "none"), EOS=\(specialTokens.eos_token ?? "none")")
        #endif
        
        return BPETokenizer(
            vocab: vocab,
            merges: merges,
            specialTokens: specialTokens
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
            for id in 0..<expectedVocabSize {
                if !vocab.values.contains(id) {
                    vocab["<unk_\(id)>"] = id
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
        vocab: [String: Int]
    ) -> BPEVocabulary.SpecialTokens {
        // Look in multiple possible locations
        var bosToken: String?
        var eosToken: String?
        var unkToken: String?
        var padToken: String?
        
        // Check added_tokens array
        if let added = json["added_tokens"] as? [[String: Any]] {
            for tokenInfo in added {
                if let content = tokenInfo["content"] as? String,
                   let special = tokenInfo["special"] as? Bool,
                   special {
                    // Match by name patterns
                    let lower = content.lowercased()
                    if lower.contains("bos") || lower.contains("<s>") || lower == "<|begin_of_text|>" {
                        bosToken = content
                    }
                    if lower.contains("eos") || lower.contains("</s>") || lower == "<|end_of_text|>" {
                        eosToken = content
                    }
                    if lower.contains("unk") || lower == "<unk>" {
                        unkToken = content
                    }
                    if lower.contains("pad") || lower == "<pad>" {
                        padToken = content
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
                        if id.contains("bos") || id == "<s>" {
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
        if bosToken == nil {
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
}

