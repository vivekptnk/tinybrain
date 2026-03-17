/// TikToken Tokenizer Adapter
///
/// Parses OpenAI's TikToken `.tiktoken` file format and converts to BPETokenizer.
///
/// **TikToken format:**
/// Each line is a base64-encoded BPE token followed by a space and its integer rank:
/// ```
/// IQ==  0
/// Ig==  1
/// Iw==  2
/// SGVsbG8=  15496
/// ```
/// The rank is the merge priority (lower rank = merged earlier / higher priority).
///
/// **Usage:**
/// ```swift
/// let tokenizer = try TikTokenAdapter.load(from: "cl100k_base.tiktoken")
/// ```
///
/// Used by GPT-2, GPT-3.5, GPT-4 (cl100k_base), and many open models that
/// adopt the OpenAI BPE vocabulary.

import Foundation

/// Adapter for OpenAI's TikToken `.tiktoken` file format
public enum TikTokenAdapter {

    // MARK: - Public API

    /// Load a TikToken `.tiktoken` file and convert to BPETokenizer
    ///
    /// - Parameter path: Path to the `.tiktoken` file
    /// - Returns: BPETokenizer configured from the TikToken vocabulary
    /// - Throws: TokenizerError if the file is missing or malformed
    public static func load(from path: String) throws -> BPETokenizer {
        guard FileManager.default.fileExists(atPath: path) else {
            throw TokenizerError.fileNotFound(path)
        }

        let content = try String(contentsOfFile: path, encoding: .utf8)
        return try parse(tikTokenContent: content)
    }

    /// Parse TikToken file content directly (useful for testing)
    ///
    /// - Parameter tikTokenContent: Raw text content of a `.tiktoken` file
    /// - Returns: BPETokenizer
    /// - Throws: TokenizerError if content is malformed or contains no valid entries
    public static func parse(tikTokenContent: String) throws -> BPETokenizer {
        let lines = tikTokenContent.components(separatedBy: .newlines)

        // vocab: decoded token bytes represented as string → rank (= token ID)
        var vocab: [String: Int] = [:]
        // All entries as (tokenString, rank) for merge reconstruction
        var entries: [(token: String, rank: Int)] = []

        var skipped = 0
        for rawLine in lines {
            let line = rawLine.trimmingCharacters(in: .whitespaces)
            guard !line.isEmpty else { continue }

            // Split into base64 part and rank part
            let parts = line.split(separator: " ", maxSplits: 1)
            guard parts.count == 2,
                  let rank = Int(parts[1].trimmingCharacters(in: .whitespaces)),
                  let tokenData = Data(base64Encoded: String(parts[0])) else {
                skipped += 1
                continue
            }

            // Represent token bytes as a displayable string:
            // - If it's valid UTF-8, store as-is
            // - Otherwise, store each byte as "<0xNN>" (same convention as BPETokenizer.decode)
            let tokenString = tokenStringFromData(tokenData)
            vocab[tokenString] = rank
            entries.append((token: tokenString, rank: rank))
        }

        guard !vocab.isEmpty else {
            throw TokenizerError.invalidVocabularyFormat("TikToken file contains no valid entries (skipped \(skipped))")
        }

        #if DEBUG
        if skipped > 0 {
            print("⚠️ TikToken: skipped \(skipped) malformed lines")
        }
        #endif

        // Sort entries by rank to reconstruct BPE merge order
        entries.sort { $0.rank < $1.rank }

        // Reconstruct BPE merges: a token of length > 1 was formed by merging
        // its two component halves. We find the best split for each multi-byte token.
        let merges = buildMerges(from: entries, vocab: vocab)

        // Add standard GPT-style special tokens to vocab if not present
        var finalVocab = vocab
        let specialTokens = addSpecialTokensIfNeeded(&finalVocab)

        #if DEBUG
        print("📖 Loaded TikToken tokenizer:")
        print("   Vocabulary size: \(finalVocab.count)")
        print("   Merge rules: \(merges.count)")
        print("   Special tokens: BOS=\(specialTokens.bos_token ?? "none"), EOS=\(specialTokens.eos_token ?? "none")")
        #endif

        return BPETokenizer(vocab: finalVocab, merges: merges, specialTokens: specialTokens)
    }

    // MARK: - Internal Helpers

    /// Convert raw token bytes to a string representation
    ///
    /// Valid UTF-8 sequences are stored as their natural string form.
    /// Invalid UTF-8 bytes are stored as `<0xNN>` hex escapes (compatible with BPETokenizer.decode).
    static func tokenStringFromData(_ data: Data) -> String {
        // Try as valid UTF-8 first
        if let str = String(data: data, encoding: .utf8) {
            return str
        }

        // Fall back to byte-escape notation
        return data.map { String(format: "<0x%02X>", $0) }.joined()
    }

    /// Reconstruct BPE merge rules from a sorted (by rank) token list.
    ///
    /// For each token of length > 1 character (or > 1 byte escape), we try to split
    /// it into two parts that are both present in the vocabulary, preferring the split
    /// where the left part has the smallest rank (i.e., was merged first).
    ///
    /// - Parameters:
    ///   - entries: Sorted by rank ascending
    ///   - vocab: tokenString → rank mapping
    /// - Returns: BPE merge rules [[left, right]] in priority order
    static func buildMerges(from entries: [(token: String, rank: Int)], vocab: [String: Int]) -> [[String]] {
        var merges: [[String]] = []

        for entry in entries {
            let token = entry.token

            // Skip single characters / single byte-escape tokens
            let parts = tokenParts(token)
            guard parts.count > 1 else { continue }

            // Try to find best split: prefer split where left part has lowest rank
            var bestSplit: (left: String, right: String, leftRank: Int)? = nil

            for splitAt in 1..<parts.count {
                let left = parts[0..<splitAt].joined()
                let right = parts[splitAt...].joined()

                if let leftRank = vocab[left], vocab[right] != nil {
                    if bestSplit == nil || leftRank < bestSplit!.leftRank {
                        bestSplit = (left: left, right: right, leftRank: leftRank)
                    }
                }
            }

            if let split = bestSplit {
                merges.append([split.left, split.right])
            }
        }

        return merges
    }

    /// Split a token string into its atomic parts (single chars or `<0xNN>` escapes)
    static func tokenParts(_ token: String) -> [String] {
        var parts: [String] = []
        var remaining = token[...]

        while !remaining.isEmpty {
            if remaining.hasPrefix("<0x") {
                // Find closing ">" — range(of:) returns a Range whose upperBound is past ">"
                if let closeRange = remaining.range(of: ">") {
                    // Slice from start up to (not including) the upperBound index to get "<0xNN>"
                    let escapeStr = String(remaining[remaining.startIndex..<closeRange.upperBound])
                    parts.append(escapeStr)
                    remaining = remaining[closeRange.upperBound...]
                } else {
                    // Malformed escape, treat rest as one part
                    parts.append(String(remaining))
                    break
                }
            } else {
                // Single Unicode character
                let char = remaining.removeFirst()
                parts.append(String(char))
            }
        }

        return parts
    }

    /// Add standard GPT-2/GPT-4 special tokens if not already present in the vocab.
    ///
    /// TikToken files typically don't include special tokens inline; they're added
    /// by the tiktoken library at runtime. We replicate the common defaults here.
    ///
    /// - Parameter vocab: The vocabulary to augment (modified in place)
    /// - Returns: Detected special token configuration
    @discardableResult
    static func addSpecialTokensIfNeeded(_ vocab: inout [String: Int]) -> BPEVocabulary.SpecialTokens {
        // GPT-2 uses byte-level BPE with no traditional BOS/EOS in the file itself;
        // GPT-4 (cl100k) adds "<|endoftext|>" as both BOS and EOS.
        // We'll use the highest existing rank + offset for any missing tokens.
        let maxExistingRank = vocab.values.max() ?? 0

        var nextId = maxExistingRank + 1

        let bosKey = "<|endoftext|>"
        let eosKey = "<|endoftext|>"

        // BOS and EOS share the same token in GPT-4 style
        if vocab[bosKey] == nil {
            vocab[bosKey] = nextId
            nextId += 1
        }

        let unkKey = "<|unk|>"
        if vocab[unkKey] == nil {
            vocab[unkKey] = nextId
            nextId += 1
        }

        return BPEVocabulary.SpecialTokens(
            bos_token: bosKey,
            eos_token: eosKey,
            unk_token: unkKey,
            pad_token: nil
        )
    }
}
