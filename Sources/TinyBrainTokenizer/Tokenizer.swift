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
    let vocab: [String: Int]
    
    /// Ordered list of BPE merge rules (applied in order)
    let merges: [[String]]
    
    /// Special tokens configuration
    let special_tokens: SpecialTokens?
    
    struct SpecialTokens: Codable {
        let bos_token: String?
        let eos_token: String?
        let unk_token: String?
        let pad_token: String?
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
    
    // MARK: - Initialization
    
    /// Initialize BPE tokenizer from vocabulary file
    ///
    /// **Educational:**
    /// 1. Load JSON vocabulary
    /// 2. Build bidirectional token↔ID maps
    /// 3. Parse merge rules
    /// 4. Extract special tokens
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
        
        // Build token maps
        self.tokenToId = vocabulary.vocab
        self.vocabularySize = vocabulary.vocab.count
        
        // Build inverse mapping (ID → token)
        var idToTokenMap: [Int: String] = [:]
        for (token, id) in vocabulary.vocab {
            idToTokenMap[id] = token
        }
        self.idToToken = idToTokenMap
        
        // Parse merge rules
        var rules: [(String, String)] = []
        var priorityMap: [String: [String: Int]] = [:]
        
        for (priority, mergePair) in vocabulary.merges.enumerated() {
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
        
        // Extract special tokens
        let specialTokens = vocabulary.special_tokens
        
        // Look up special token IDs from vocabulary (avoiding closure capture issues)
        if let bosString = specialTokens?.bos_token, let bosId = tokenToId[bosString] {
            self.bosToken = bosId
        } else {
            self.bosToken = 0
        }
        
        if let eosString = specialTokens?.eos_token, let eosId = tokenToId[eosString] {
            self.eosToken = eosId
        } else {
            self.eosToken = 1
        }
        
        if let unkString = specialTokens?.unk_token, let unkId = tokenToId[unkString] {
            self.unkToken = unkId
        } else {
            self.unkToken = 2
        }
        
        if let padString = specialTokens?.pad_token, let padId = tokenToId[padString] {
            self.padToken = padId
        } else {
            self.padToken = 3
        }
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
        // Step 1: Unicode normalization (NFC - canonical composition)
        // This ensures "café" (composed) and "cafe\u{0301}" (decomposed) are identical
        let normalized = text.precomposedStringWithCanonicalMapping
        
        // Step 2: Handle empty string
        if normalized.isEmpty {
            return []
        }
        
        // Step 3: Split into characters (initial tokens)
        var tokens = normalized.map { String($0) }
        
        // Step 4: Apply BPE merges until no more merges possible
        tokens = applyBPEMerges(tokens)
        
        // Step 5: Convert tokens to IDs
        return tokens.map { token in
            tokenToId[token] ?? unkToken  // Unknown tokens → UNK
        }
    }
    
    /// Apply BPE merge rules to token sequence
    ///
    /// **Algorithm:**
    /// 1. Find highest-priority merge in current tokens
    /// 2. Apply that merge
    /// 3. Repeat until no merges left
    ///
    /// **Time complexity:** O(n² × m) where n = tokens, m = merges
    /// This is acceptable for educational purposes; production would optimize.
    ///
    /// - Parameter tokens: Initial token sequence
    /// - Returns: Merged token sequence
    private func applyBPEMerges(_ tokens: [String]) -> [String] {
        var currentTokens = tokens
        
        // Keep merging until no more merges possible
        while true {
            // Find best (lowest priority number = highest priority) merge
            var bestMerge: (index: Int, priority: Int)? = nil
            
            for i in 0..<(currentTokens.count - 1) {
                let first = currentTokens[i]
                let second = currentTokens[i + 1]
                
                // Check if this pair has a merge rule
                if let priority = mergePriority[first]?[second] {
                    if bestMerge == nil || priority < bestMerge!.priority {
                        bestMerge = (index: i, priority: priority)
                    }
                }
            }
            
            // No more merges available
            guard let merge = bestMerge else {
                break
            }
            
            // Apply the merge
            let merged = currentTokens[merge.index] + currentTokens[merge.index + 1]
            currentTokens.replaceSubrange(merge.index...merge.index + 1, with: [merged])
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
        return tokens
            .compactMap { idToToken[$0] }  // Lookup each ID, skip invalid
            .joined()  // Concatenate strings
    }
}

// MARK: - Errors

enum TokenizerError: Error, CustomStringConvertible {
    case vocabularyNotFound(String)
    case invalidVocabularyFormat(String)
    
    var description: String {
        switch self {
        case .vocabularyNotFound(let path):
            return "Vocabulary file not found: \(path)"
        case .invalidVocabularyFormat(let message):
            return "Invalid vocabulary format: \(message)"
        }
    }
}

