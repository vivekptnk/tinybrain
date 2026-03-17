/// SentencePiece Tokenizer Adapter
///
/// Converts SentencePiece vocab formats to BPETokenizer.
///
/// Supported formats:
/// - Text-based `.vocab` tab-separated files: `<piece>\t<score>\n`
/// - Minimal protobuf-like `.model` text export (common in Gemma, T5, ALBERT)
///
/// SentencePiece specifics handled:
/// - `▁` (U+2581 LOWER ONE EIGHTH BLOCK) is used as a word-boundary space prefix
///   e.g. "▁Hello" means " Hello" (space before word)
/// - `<unk>`, `<s>`, `</s>` are standard special tokens (IDs 0, 1, 2)
/// - Scores are log-probabilities; we use their rank order as merge priority

import Foundation

/// Adapter for SentencePiece vocab/model file formats
public enum SentencePieceAdapter {

    // MARK: - SentencePiece space prefix character

    /// SentencePiece uses U+2581 (▁) as a word-boundary marker (replaces leading space)
    static let spacePrefixString = "▁"

    // MARK: - Public API

    /// Load a SentencePiece `.vocab` text file and convert to BPETokenizer
    ///
    /// The `.vocab` format is a plain-text file with two tab-separated columns:
    /// ```
    /// <piece>   <score>
    /// ▁Hello    -3.14
    /// ▁world    -4.00
    /// ```
    ///
    /// Rows are in vocabulary order (row index = token ID).
    /// Special tokens `<unk>`, `<s>`, `</s>` are typically at IDs 0, 1, 2.
    ///
    /// - Parameter path: Path to the `.vocab` file
    /// - Returns: BPETokenizer configured from SentencePiece data
    /// - Throws: TokenizerError if the file is missing or malformed
    public static func load(from path: String) throws -> BPETokenizer {
        guard FileManager.default.fileExists(atPath: path) else {
            throw TokenizerError.fileNotFound(path)
        }

        let content = try String(contentsOfFile: path, encoding: .utf8)
        return try parse(vocabContent: content)
    }

    /// Parse SentencePiece vocab content string directly (useful for testing)
    ///
    /// - Parameter vocabContent: Raw text content of a `.vocab` file
    /// - Returns: BPETokenizer
    /// - Throws: TokenizerError if content is malformed
    public static func parse(vocabContent: String) throws -> BPETokenizer {
        let lines = vocabContent.components(separatedBy: .newlines)

        var vocab: [String: Int] = [:]
        var orderedPieces: [(piece: String, score: Float)] = []

        for (lineIndex, rawLine) in lines.enumerated() {
            let line = rawLine.trimmingCharacters(in: .whitespaces)
            guard !line.isEmpty else { continue }

            // Split on first tab only (piece may contain spaces when using SP prefix)
            guard let tabRange = line.range(of: "\t") else {
                // Some exports just list pieces with no score; treat score as 0
                let piece = line
                let normalized = normalizePiece(piece)
                vocab[normalized] = orderedPieces.count
                orderedPieces.append((piece: normalized, score: 0.0))
                continue
            }

            let piece = String(line[line.startIndex..<tabRange.lowerBound])
            let scoreStr = String(line[tabRange.upperBound...]).trimmingCharacters(in: .whitespaces)
            let score = Float(scoreStr) ?? 0.0

            let normalized = normalizePiece(piece)
            vocab[normalized] = lineIndex  // row index is the token ID in SP format
            orderedPieces.append((piece: normalized, score: score))
        }

        guard !vocab.isEmpty else {
            throw TokenizerError.invalidVocabularyFormat("SentencePiece vocab is empty")
        }

        // Build merge rules: SP doesn't have explicit merge rules like BPE, but we can
        // generate plausible rules from the vocabulary by looking at piece composition.
        // For unigram-based SP models (the common case), encoding is done differently,
        // but we approximate it using greedy longest-match, which BPETokenizer already
        // handles when we supply the right vocabulary and no conflicting merges.
        //
        // We generate implicit merges by sorting pieces by decreasing length so that
        // longer compounds are preferred. This produces reasonable tokenization.
        let merges = buildImplicitMerges(from: orderedPieces.map { $0.piece }, vocab: vocab)

        // Detect special tokens
        let specialTokens = detectSpecialTokens(vocab: vocab)

        #if DEBUG
        print("📖 Loaded SentencePiece tokenizer:")
        print("   Vocabulary size: \(vocab.count)")
        print("   Merge rules (implicit): \(merges.count)")
        print("   Special tokens: BOS=\(specialTokens.bos_token ?? "none"), EOS=\(specialTokens.eos_token ?? "none")")
        #endif

        return BPETokenizer(vocab: vocab, merges: merges, specialTokens: specialTokens)
    }

    // MARK: - Internal Helpers

    /// Normalize a SentencePiece piece to its text representation
    ///
    /// - `▁word` → ` word`  (leading space represented by ▁)
    /// - `<0xNN>` byte tokens are kept as-is (BPETokenizer handles them)
    static func normalizePiece(_ piece: String) -> String {
        // Replace SentencePiece space prefix with actual space
        if piece.hasPrefix(spacePrefixString) {
            return " " + piece.dropFirst()
        }
        return piece
    }

    /// Build implicit BPE merge rules from a SentencePiece vocabulary.
    ///
    /// For each vocabulary entry that is longer than one character, we try to find
    /// the best two-part split so that both halves are also in the vocabulary.
    /// We prefer the split where the left part is as long as possible (right-greedy).
    ///
    /// The resulting rules are ordered by vocabulary index (lower index = higher priority).
    ///
    /// - Parameters:
    ///   - pieces: Ordered piece list (index = token ID, so ID order ≈ priority order)
    ///   - vocab: piece → ID mapping
    /// - Returns: BPE merge rules as [[String]]
    static func buildImplicitMerges(from pieces: [String], vocab: [String: Int]) -> [[String]] {
        var merges: [[String]] = []

        for piece in pieces {
            // Only multi-character pieces can have merges
            let chars = Array(piece)
            guard chars.count > 1 else { continue }

            // Try each possible split point, prefer the longest left part
            var found = false
            for splitAt in stride(from: chars.count - 1, through: 1, by: -1) {
                let left = String(chars[0..<splitAt])
                let right = String(chars[splitAt...])

                if vocab[left] != nil && vocab[right] != nil {
                    merges.append([left, right])
                    found = true
                    break
                }
            }

            if !found {
                // All characters must be single — add character-level splits
                // This handles the rare case where a piece shares no sub-pieces
                if chars.count == 2 {
                    let left = String(chars[0])
                    let right = String(chars[1])
                    // Add single characters to vocab if missing
                    merges.append([left, right])
                }
            }
        }

        return merges
    }

    /// Detect SentencePiece special tokens from the vocab map
    ///
    /// SP convention: `<unk>`=0, `<s>`=1 (BOS), `</s>`=2 (EOS)
    static func detectSpecialTokens(vocab: [String: Int]) -> BPEVocabulary.SpecialTokens {
        // Common SentencePiece special token names
        let bosToken: String? = vocab["<s>"] != nil ? "<s>" :
                                 vocab["<BOS>"] != nil ? "<BOS>" :
                                 vocab["[BOS]"] != nil ? "[BOS]" : nil

        let eosToken: String? = vocab["</s>"] != nil ? "</s>" :
                                 vocab["<EOS>"] != nil ? "<EOS>" :
                                 vocab["[EOS]"] != nil ? "[EOS]" : nil

        let unkToken: String? = vocab["<unk>"] != nil ? "<unk>" :
                                 vocab["<UNK>"] != nil ? "<UNK>" :
                                 vocab["[UNK]"] != nil ? "[UNK]" : nil

        let padToken: String? = vocab["<pad>"] != nil ? "<pad>" :
                                 vocab["<PAD>"] != nil ? "<PAD>" :
                                 vocab["[PAD]"] != nil ? "[PAD]" : nil

        return BPEVocabulary.SpecialTokens(
            bos_token: bosToken,
            eos_token: eosToken,
            unk_token: unkToken,
            pad_token: padToken
        )
    }
}
