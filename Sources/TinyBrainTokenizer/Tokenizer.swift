/// Tokenization support for TinyBrain
///
/// Provides BPE and SentencePiece tokenization for converting text to/from token IDs.
///
/// **TB-005 Implementation**
///
/// ## How BPE (Byte Pair Encoding) Works
///
/// BPE is a data compression algorithm adapted for tokenization:
///
/// 1. **Start with characters**: "Hello" → ['H', 'e', 'l', 'l', 'o']
/// 2. **Learn merges**: Most frequent pairs get merged:
///    - 'l' + 'l' → 'll'
///    - 'He' + 'll' → 'Hell'
///    - 'Hell' + 'o' → 'Hello'
/// 3. **Result**: Fewer tokens, handles unknowns via subwords
///
/// **Why BPE?**
/// - Balance between character-level (flexible but long) and word-level (compact but rigid)
/// - Handles unknown words by breaking into known subwords
/// - Standard for GPT, LLaMA, and most modern LLMs

import Foundation

/// Protocol for text tokenization
public protocol Tokenizer {
    /// Encode text into token IDs
    func encode(_ text: String) -> [Int]
    
    /// Decode token IDs back into text
    func decode(_ tokens: [Int]) -> String
    
    /// Vocabulary size
    var vocabularySize: Int { get }
}

// MARK: - Vocabulary Data Structures

/// BPE vocabulary loaded from file
///
/// **Format:**
/// ```json
/// {
///   "vocab": { "token": id, ... },
///   "merges": [["a", "b"], ...],
///   "special_tokens": { "bos_token": "<BOS>", ... }
/// }
/// ```
public struct BPEVocabulary: Codable {
    /// Maps token string → ID
    public let vocab: [String: Int]
    
    /// Ordered list of BPE merge rules (applied in order)
    public let merges: [[String]]
    
    /// Special tokens configuration
    public let special_tokens: SpecialTokens?
    
    public struct SpecialTokens: Codable {
        public let bos_token: String?
        public let eos_token: String?
        public let unk_token: String?
        public let pad_token: String?
        
        public init(bos_token: String?, eos_token: String?, unk_token: String?, pad_token: String?) {
            self.bos_token = bos_token
            self.eos_token = eos_token
            self.unk_token = unk_token
            self.pad_token = pad_token
        }
    }
}

// MARK: - BPE Tokenizer Implementation

/// Byte Pair Encoding tokenizer
///
/// **TB-005:** Full BPE implementation with:
/// - Unicode normalization (NFC)
/// - Special token handling (BOS, EOS, UNK, PAD)
/// - Graceful unknown character handling
/// - Educational transparency
public struct BPETokenizer: Tokenizer {
    // MARK: - Public Properties
    
    public let vocabularySize: Int
    
    /// Beginning-of-sequence token ID
    public let bosToken: Int
    
    /// End-of-sequence token ID
    public let eosToken: Int
    
    /// Unknown token ID (fallback for out-of-vocab characters)
    public let unkToken: Int
    
    /// Padding token ID
    public let padToken: Int
    
    // MARK: - Private Properties
    
    /// Token string → ID mapping
    private let tokenToId: [String: Int]
    
    /// ID → token string mapping (inverse of tokenToId)
    private let idToToken: [Int: String]
    
    /// BPE merge rules in priority order
    /// Each rule is a pair of strings to merge
    private let mergeRules: [(String, String)]
    
    /// Priority map for efficient merge lookups
    /// Maps (token1, token2) → merge priority (lower = higher priority)
    private let mergePriority: [String: [String: Int]]

    /// Whether unknown BPE pieces should be decomposed into `<0xNN>` byte tokens.
    private let byteFallbackEnabled: Bool

    /// Tokens declared by the source tokenizer as added/special tokens.
    ///
    /// These are matched before BPE so literals such as `</s>` map to their
    /// exact vocabulary IDs and never merge through normal text.
    private let preTokenizedTokens: [String]

    /// Whether text segments use SentencePiece/HF metaspace normalization.
    private let usesSentencePieceWhitespace: Bool
    
    // MARK: - Initialization
    
    /// Initialize BPE tokenizer with raw vocabulary data
    ///
    /// **TB-009:** Direct initialization for adapter pattern
    /// Used by TokenizerLoader to support multiple formats
    ///
    /// - Parameters:
    ///   - vocab: Token string → ID mapping
    ///   - merges: BPE merge rules (ordered)
    ///   - specialTokens: Special token configuration
    ///   - byteFallback: When true, unknown pieces encode as UTF-8 `<0xNN>` tokens.
    ///   - preTokenizedTokens: Literal tokens to split out before BPE.
    ///   - usesSentencePieceWhitespace: Overrides automatic `▁` whitespace detection.
    public init(vocab: [String: Int],
                merges: [[String]],
                specialTokens: BPEVocabulary.SpecialTokens,
                byteFallback: Bool = false,
                preTokenizedTokens: Set<String> = [],
                usesSentencePieceWhitespace: Bool? = nil) {
        // Build token maps
        self.tokenToId = vocab
        self.vocabularySize = vocab.count
        
        // Build inverse mapping
        var idToTokenMap: [Int: String] = [:]
        for (token, id) in vocab {
            idToTokenMap[id] = token
        }
        self.idToToken = idToTokenMap
        
        // Parse merge rules
        var rules: [(String, String)] = []
        var priorityMap: [String: [String: Int]] = [:]
        
        for (priority, mergePair) in merges.enumerated() {
            guard mergePair.count == 2 else { continue }
            let first = mergePair[0]
            let second = mergePair[1]
            rules.append((first, second))
            
            // Build priority map for O(1) lookup
            if priorityMap[first] == nil {
                priorityMap[first] = [:]
            }
            priorityMap[first]![second] = priority
        }
        
        self.mergeRules = rules
        self.mergePriority = priorityMap
        self.byteFallbackEnabled = byteFallback
        self.preTokenizedTokens = preTokenizedTokens
            .filter { !$0.isEmpty && vocab[$0] != nil }
            .sorted {
                if $0.count == $1.count {
                    return $0 < $1
                }
                return $0.count > $1.count
            }

        let spaceMarker = "\u{2581}"
        self.usesSentencePieceWhitespace = usesSentencePieceWhitespace ??
            (vocab[spaceMarker] != nil || vocab[spaceMarker + "a"] != nil)
        
        // Extract special tokens with smart fallback to actual vocab entries
        // Use first available valid token if special tokens not defined
        let validIds = Array(vocab.values).sorted()
        let firstValidId = validIds.first ?? 0
        
        self.bosToken = (specialTokens.bos_token.flatMap { vocab[$0] }) ?? firstValidId
        self.eosToken = (specialTokens.eos_token.flatMap { vocab[$0] }) ?? (validIds.dropFirst().first ?? firstValidId)
        self.unkToken = (specialTokens.unk_token.flatMap { vocab[$0] }) ?? (validIds.dropFirst(2).first ?? firstValidId)
        self.padToken = (specialTokens.pad_token.flatMap { vocab[$0] }) ?? (validIds.dropFirst(3).first ?? firstValidId)
    }
    
    /// Initialize BPE tokenizer from vocabulary file (TinyBrain JSON format)
    ///
    /// **Educational:**
    /// 1. Load JSON vocabulary
    /// 2. Delegate to raw init
    ///
    /// - Parameter vocabularyPath: Path to JSON vocab file
    /// - Throws: If file not found or JSON invalid
    public init(vocabularyPath: String) throws {
        // Load and parse JSON
        let url = URL(fileURLWithPath: vocabularyPath)
        
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw TokenizerError.vocabularyNotFound(vocabularyPath)
        }
        
        let data = try Data(contentsOf: url)
        let vocabulary = try JSONDecoder().decode(BPEVocabulary.self, from: data)
        
        // Use raw init (DRY principle)
        self.init(
            vocab: vocabulary.vocab,
            merges: vocabulary.merges,
            specialTokens: vocabulary.special_tokens ?? BPEVocabulary.SpecialTokens(
                bos_token: "<BOS>",
                eos_token: "<EOS>",
                unk_token: "<UNK>",
                pad_token: "<PAD>"
            )
        )
    }
    
    // MARK: - Encoding
    
    /// Encode text into token IDs using BPE algorithm
    ///
    /// **Educational BPE Algorithm:**
    ///
    /// ```
    /// Input: "Hello"
    /// Step 1: Split to characters: ['H', 'e', 'l', 'l', 'o']
    /// Step 2: Apply merges in priority order:
    ///   - Merge 'l'+'l' → 'll': ['H', 'e', 'll', 'o']
    ///   - Merge 'He'+'ll' → 'Hell': ['Hell', 'o']
    ///   - Merge 'Hell'+'o' → 'Hello': ['Hello']
    /// Step 3: Convert to IDs: [102]
    /// ```
    ///
    /// - Parameter text: Input text to tokenize
    /// - Returns: Array of token IDs
    public func encode(_ text: String) -> [Int] {
        // Step 1: Handle empty string
        if text.isEmpty {
            return []
        }

        // Step 2: Split declared added/special tokens before BPE. HuggingFace
        // tokenizers normalize each surrounding text span independently, which
        // matters for SentencePiece-style leading metaspace after `</s>`.
        if preTokenizedTokens.isEmpty {
            return encodeTextSegment(text)
        }

        var encoded: [Int] = []
        for segment in splitPreTokenizedSegments(text) {
            switch segment {
            case .text(let textSegment):
                encoded.append(contentsOf: encodeTextSegment(textSegment))
            case .preTokenized(let token):
                encoded.append(tokenToId[token] ?? unkToken)
            }
        }

        return encoded
    }

    private enum EncodingSegment {
        case text(String)
        case preTokenized(String)
    }

    /// Encode one ordinary text segment after any added/special tokens are removed.
    private func encodeTextSegment(_ text: String) -> [Int] {
        // Unicode normalization (NFC - canonical composition) applies to text,
        // not to already-split special tokens whose HF config marks normalized=false.
        let normalized = text.precomposedStringWithCanonicalMapping
        guard !normalized.isEmpty else {
            return []
        }

        // SentencePiece-style preprocessing:
        // Replace spaces with ▁ (U+2581) and prepend ▁ at the start.
        let spaceMarker = "\u{2581}"  // ▁
        let processed: String
        if usesSentencePieceWhitespace {
            processed = spaceMarker + normalized.replacingOccurrences(of: " ", with: spaceMarker)
        } else {
            processed = normalized
        }

        // Split into characters (initial tokens)
        var tokens = processed.map { String($0) }

        // Apply BPE merges until no more merges possible
        tokens = applyBPEMerges(tokens)

        // Convert tokens to IDs, using byte fallback only when the source
        // tokenizer explicitly declares model.byte_fallback.
        var ids: [Int] = []
        for token in tokens {
            ids.append(contentsOf: tokenIds(for: token))
        }
        return ids
    }

    /// Split text into normal spans and declared added/special token spans.
    private func splitPreTokenizedSegments(_ text: String) -> [EncodingSegment] {
        var segments: [EncodingSegment] = []
        var textStart = text.startIndex
        var index = text.startIndex

        while index < text.endIndex {
            if let token = matchingPreTokenizedToken(in: text, at: index) {
                if textStart < index {
                    segments.append(.text(String(text[textStart..<index])))
                }

                segments.append(.preTokenized(token))
                index = text.index(index, offsetBy: token.count)
                textStart = index
            } else {
                index = text.index(after: index)
            }
        }

        if textStart < text.endIndex {
            segments.append(.text(String(text[textStart..<text.endIndex])))
        }

        return segments
    }

    /// Return the longest declared added/special token that starts at `index`.
    private func matchingPreTokenizedToken(in text: String, at index: String.Index) -> String? {
        let suffix = text[index...]
        return preTokenizedTokens.first { suffix.hasPrefix($0) }
    }

    /// Convert one BPE piece to ids, applying SentencePiece byte fallback if enabled.
    private func tokenIds(for token: String) -> [Int] {
        if let id = tokenToId[token] {
            return [id]
        }

        guard byteFallbackEnabled else {
            return [unkToken]
        }

        var byteIds: [Int] = []
        byteIds.reserveCapacity(token.utf8.count)

        for byte in token.utf8 {
            let byteToken = String(format: "<0x%02X>", byte)
            guard let id = tokenToId[byteToken] else {
                return [unkToken]
            }
            byteIds.append(id)
        }

        return byteIds.isEmpty ? [unkToken] : byteIds
    }
    
    /// Apply BPE merge rules to token sequence
    ///
    /// **Algorithm:**
    /// 1. Find the highest-priority merge currently present
    /// 2. Apply that merge to every non-overlapping occurrence
    /// 3. Repeat until no merges left
    ///
    /// Batched merging keeps the same BPE rank order as repeatedly merging the
    /// best pair, but avoids rescanning and reallocating the token array once
    /// per pair occurrence in repeated text.
    ///
    /// - Parameter tokens: Initial token sequence
    /// - Returns: Merged token sequence
    private func applyBPEMerges(_ tokens: [String]) -> [String] {
        guard tokens.count > 1 else {
            return tokens
        }
        
        var currentTokens = tokens

        // Keep merging until no more merges are present.
        while true {
            // Find best (lowest priority number = highest priority) merge
            var bestMerge: (first: String, second: String, priority: Int)? = nil
            
            for i in 0..<(currentTokens.count - 1) {
                let first = currentTokens[i]
                let second = currentTokens[i + 1]
                
                // Check if this pair has a merge rule
                if let priority = mergePriority[first]?[second] {
                    if bestMerge == nil || priority < bestMerge!.priority {
                        bestMerge = (first: first, second: second, priority: priority)
                    }
                }
            }
            
            // No more merges available
            guard let merge = bestMerge else {
                break
            }
            
            // Apply the best merge to all non-overlapping occurrences in one pass.
            let mergedToken = merge.first + merge.second
            var mergedTokens: [String] = []
            mergedTokens.reserveCapacity(currentTokens.count)

            var index = 0
            while index < currentTokens.count {
                if index < currentTokens.count - 1,
                   currentTokens[index] == merge.first,
                   currentTokens[index + 1] == merge.second {
                    mergedTokens.append(mergedToken)
                    index += 2
                } else {
                    mergedTokens.append(currentTokens[index])
                    index += 1
                }
            }

            currentTokens = mergedTokens
        }
        
        return currentTokens
    }
    
    // MARK: - Decoding
    
    /// Decode token IDs back into text
    ///
    /// **Educational:**
    /// Decoding is simpler than encoding - just lookup and concatenate!
    ///
    /// ```
    /// Input: [102, 8, 9, 105]
    /// Step 1: Lookup: ['Hello', ',', ' ', 'world']
    /// Step 2: Concatenate: "Hello, world"
    /// ```
    ///
    /// - Parameter tokens: Token IDs to decode
    /// - Returns: Reconstructed text
    public func decode(_ tokens: [Int]) -> String {
        let tokenStrings = tokens.compactMap { idToToken[$0] }
        
        // Handle byte-level BPE (used by GPT-2, Llama, etc.)
        // Tokens like "<0x20>" represent bytes
        var bytes: [UInt8] = []
        
        for tokenStr in tokenStrings {
            // Check if token is a byte representation like "<0x20>"
            if tokenStr.hasPrefix("<0x") && tokenStr.hasSuffix(">") {
                // Extract hex value
                let hexStr = tokenStr.dropFirst(3).dropLast()
                if let byte = UInt8(hexStr, radix: 16) {
                    bytes.append(byte)
                }
            } else {
                // Regular token - convert SentencePiece ▁ to space, then to UTF-8 bytes
                let decoded = tokenStr.replacingOccurrences(of: "\u{2581}", with: " ")
                bytes.append(contentsOf: Array(decoded.utf8))
            }
        }

        // Convert bytes to string and strip leading space (SentencePiece artifact)
        var result = String(decoding: bytes, as: UTF8.self)
        if usesSentencePieceWhitespace && result.hasPrefix(" ") {
            result = String(result.dropFirst())
        }
        return result
    }
    
    // MARK: - Helper Functions
    
    /// Resolve special token ID from vocabulary
    ///
    /// **REVIEW HITLER FIX:** Don't hard-code IDs - find them in vocab
    ///
    /// - Parameters:
    ///   - tokenString: Optional token string from special_tokens section
    ///   - fallbackKey: Fallback key to search in vocab (e.g., "<BOS>")
    ///   - vocab: Token → ID mapping
    /// - Returns: Resolved token ID
    /// - Throws: If no valid token found and vocab is empty
    private static func resolveSpecialToken(
        tokenString: String?,
        fallbackKey: String,
        vocab: [String: Int]
    ) throws -> Int {
        // If specified in special_tokens, look it up
        if let tokenStr = tokenString, let id = vocab[tokenStr] {
            return id
        }
        
        // Fallback: try to find by key name in vocab (e.g., "<BOS>")
        if let id = vocab[fallbackKey] {
            return id
        }
        
        // Last resort: use first token in vocab (better than non-existent ID)
        if let firstId = vocab.values.min() {
            return firstId
        }
        
        throw TokenizerError.invalidVocabularyFormat("No special token found for \(fallbackKey) and vocab is empty")
    }
}

// MARK: - Errors

public enum TokenizerError: Error, CustomStringConvertible, LocalizedError {
    case vocabularyNotFound(String)
    case invalidVocabularyFormat(String)
    case unsupportedFormat(String)
    case invalidJSON
    case missingRequiredField(String)
    case fileNotFound(String)
    case matchingTokenizerNotFound(model: String, expectedPath: String)
    
    public var description: String {
        switch self {
        case .vocabularyNotFound(let path):
            return "Vocabulary file not found: \(path)"
        case .invalidVocabularyFormat(let message):
            return "Invalid vocabulary format: \(message)"
        case .unsupportedFormat(let format):
            return "Unsupported tokenizer format: \(format)"
        case .invalidJSON:
            return "Invalid JSON in tokenizer file"
        case .missingRequiredField(let field):
            return "Missing required field: \(field)"
        case .fileNotFound(let path):
            return "File not found: \(path)"
        case .matchingTokenizerNotFound(let model, let expectedPath):
            return "No tokenizer found for \(model) — expected \(expectedPath). Decoding with a mismatched tokenizer would produce garbage."
        }
    }

    public var errorDescription: String? {
        description
    }
}
