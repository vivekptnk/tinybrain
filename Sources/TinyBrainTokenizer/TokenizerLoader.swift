/// Tokenizer Loading Infrastructure
///
/// **TB-009 / TB-010:** Format-agnostic tokenizer loading for studio/pipeline use
///
/// Supports:
/// - HuggingFace tokenizer.json (Llama, Phi, Gemma, etc.)
/// - TinyBrain simple JSON format
/// - SentencePiece `.vocab` / `.model` text-based files (Gemma, T5, ALBERT)
/// - TikToken base64 BPE format (GPT-2/3/4 style)
///
/// Priority when auto-detecting in a directory:
/// 1. HuggingFace  (`tokenizer.json` with version+model keys)
/// 2. TinyBrain    (`*.json` with vocab+merges at top level)
/// 3. SentencePiece (`tokenizer.model`, `spiece.model`, `*.vocab`)
/// 4. TikToken     (`*.tiktoken`)
///
/// Design: Adapter pattern — each format has an adapter that converts to BPETokenizer

import Foundation

// MARK: - Format Detection

/// Supported tokenizer formats
public enum TokenizerFormat: Equatable {
    case huggingFace   // tokenizer.json with "version", "model" keys
    case tinyBrain     // Simple JSON with "vocab", "merges" keys
    case sentencePiece // SentencePiece .vocab or .model text file
    case tiktoken      // OpenAI TikToken base64 BPE format

    /// Detect tokenizer format by inspecting the file at `path`.
    ///
    /// Detection order (to avoid false positives):
    /// 1. `.tiktoken` extension → tiktoken
    /// 2. `.model` or `.vocab` extension → sentencePiece
    /// 3. `.json` content inspection → huggingFace or tinyBrain
    /// 4. Plain-text content heuristic → sentencePiece (tab-separated vocab)
    public static func detect(at path: String) -> TokenizerFormat? {
        guard FileManager.default.fileExists(atPath: path) else {
            return nil
        }

        let url = URL(fileURLWithPath: path)
        let ext = url.pathExtension.lowercased()
        let filename = url.lastPathComponent.lowercased()

        // Explicit extension matches
        if ext == "tiktoken" {
            return .tiktoken
        }

        if ext == "model" || filename == "tokenizer.model" || filename == "spiece.model" {
            return .sentencePiece
        }

        if ext == "vocab" {
            return .sentencePiece
        }

        // JSON files: inspect content
        if ext == "json" {
            guard let data = try? Data(contentsOf: url),
                  let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                return nil
            }

            // HuggingFace format has "version" and "model" keys
            if json["version"] != nil && json["model"] != nil {
                return .huggingFace
            }

            // TinyBrain format has "vocab" and "merges" at top level
            if json["vocab"] != nil && json["merges"] != nil {
                return .tinyBrain
            }

            return nil
        }

        // For extensionless or unknown-extension files, peek at content
        // to check if it looks like a SentencePiece vocab (tab-separated)
        if let data = try? Data(contentsOf: url),
           let preview = String(data: data.prefix(512), encoding: .utf8) {
            // SentencePiece vocab has lines like: "<piece>\t<score>"
            let looksLikeSPVocab = preview
                .components(separatedBy: .newlines)
                .prefix(5)
                .filter { !$0.isEmpty }
                .allSatisfy { $0.contains("\t") }
            if looksLikeSPVocab {
                return .sentencePiece
            }
        }

        return nil
    }
}

// MARK: - Main Loader

/// Format-agnostic tokenizer loader
///
/// **Studio API:** Users call this, don't need to know about formats
public enum TokenizerLoader {
    
    /// Load tokenizer from any supported format (auto-detect)
    ///
    /// - Parameter path: Path to tokenizer file
    /// - Returns: Tokenizer instance
    /// - Throws: TokenizerError if format unsupported or parsing fails
    public static func load(from path: String) throws -> any Tokenizer {
        guard let format = TokenizerFormat.detect(at: path) else {
            throw TokenizerError.unsupportedFormat(path)
        }
        
        switch format {
        case .huggingFace:
            return try loadHuggingFace(from: path)

        case .tinyBrain:
            return try BPETokenizer(vocabularyPath: path)

        case .sentencePiece:
            return try loadSentencePiece(from: path)

        case .tiktoken:
            return try loadTikToken(from: path)
        }
    }

    /// Load the tokenizer that belongs to a specific `.tbf` model file.
    ///
    /// This intentionally fails instead of falling back to an arbitrary
    /// tokenizer. A model/tokenizer mismatch can decode valid token IDs into
    /// convincing garbage, so callers should surface the error to the user and
    /// keep the previously active model.
    public static func loadTokenizer(forModelAt path: String) throws -> any Tokenizer {
        let resolvedPath = resolvePath(path) ?? path
        return try loadTokenizer(forModelAt: URL(fileURLWithPath: resolvedPath))
    }

    /// Load the tokenizer that belongs to a specific `.tbf` model file.
    public static func loadTokenizer(forModelAt modelURL: URL) throws -> any Tokenizer {
        let candidates = tokenizerCandidateURLs(forModelAt: modelURL)

        for candidate in candidates {
            if FileManager.default.fileExists(atPath: candidate.path) {
                return try load(from: candidate.path)
            }
        }

        let expectedPath = candidates.first?.path ?? modelURL
            .deletingLastPathComponent()
            .appendingPathComponent("tokenizer.json")
            .path
        throw TokenizerError.matchingTokenizerNotFound(
            model: modelURL.lastPathComponent,
            expectedPath: expectedPath
        )
    }
    
    /// Load HuggingFace tokenizer.json format
    ///
    /// Parses HF format and adapts to BPETokenizer
    ///
    /// - Parameter path: Path to tokenizer.json
    /// - Returns: BPETokenizer configured from HF format
    /// - Throws: TokenizerError if parsing fails
    public static func loadHuggingFace(from path: String) throws -> BPETokenizer {
        return try HuggingFaceAdapter.load(from: path)
    }

    /// Load SentencePiece `.vocab` or `.model` text format
    ///
    /// Parses the tab-separated vocab file and converts to BPETokenizer.
    ///
    /// - Parameter path: Path to `.vocab` / `.model` file
    /// - Returns: BPETokenizer configured from SentencePiece data
    /// - Throws: TokenizerError if parsing fails
    public static func loadSentencePiece(from path: String) throws -> BPETokenizer {
        return try SentencePieceAdapter.load(from: path)
    }

    /// Load TikToken base64 BPE format
    ///
    /// Parses OpenAI's `.tiktoken` file format and converts to BPETokenizer.
    ///
    /// - Parameter path: Path to `.tiktoken` file
    /// - Returns: BPETokenizer configured from TikToken data
    /// - Throws: TokenizerError if parsing fails
    public static func loadTikToken(from path: String) throws -> BPETokenizer {
        return try TikTokenAdapter.load(from: path)
    }
    
    /// Discover and load best available tokenizer
    ///
    /// Search strategy (priority order):
    /// 1. HuggingFace `tokenizer.json` in `Models/tinyllama-raw/`
    /// 2. HuggingFace `tokenizer.json` anywhere in `Models/`
    /// 3. SentencePiece `tokenizer.model` / `spiece.model` in `Models/`
    /// 4. Any `.tiktoken` file in `Models/`
    /// 5. Fallback minimal vocab (demo mode)
    ///
    /// - Returns: Tokenizer instance (always succeeds with fallback)
    public static func loadBestAvailable() -> any Tokenizer {
        // Priority 1: TinyLlama HuggingFace tokenizer
        let tinyLlamaPath = "Models/tinyllama-raw/tokenizer.json"
        if let resolvedPath = resolvePath(tinyLlamaPath),
           FileManager.default.fileExists(atPath: resolvedPath) {
            if let tokenizer = try? load(from: resolvedPath) {
                return tokenizer
            }
        }

        let modelsPath = resolvePath("Models") ?? "Models"
        return loadBestAvailable(in: modelsPath, includeTinyLlamaPriority: false)
    }

    /// Load the best tokenizer from an explicit Models directory.
    ///
    /// Internal for hermetic tests; the public `loadBestAvailable()` keeps the
    /// repository-relative discovery behavior used by the demo.
    static func loadBestAvailable(in modelsPath: String) -> any Tokenizer {
        return loadBestAvailable(in: modelsPath, includeTinyLlamaPriority: true)
    }

    private static func loadBestAvailable(
        in modelsPath: String,
        includeTinyLlamaPriority: Bool
    ) -> any Tokenizer {
        let fm = FileManager.default

        // Priority 1: TinyLlama HuggingFace tokenizer
        if includeTinyLlamaPriority {
            let tinyLlamaPath = (modelsPath as NSString).appendingPathComponent("tinyllama-raw/tokenizer.json")
            if fm.fileExists(atPath: tinyLlamaPath), let tokenizer = try? load(from: tinyLlamaPath) {
                return tokenizer
            }
        }

        // Priority 2: Any HuggingFace tokenizer.json in Models/
        if let files = try? fm.contentsOfDirectory(atPath: modelsPath) {
            for file in files where file.hasSuffix(".json") && file.lowercased().contains("tokenizer") {
                let fullPath = (modelsPath as NSString).appendingPathComponent(file)
                if let tokenizer = try? load(from: fullPath) {
                    return tokenizer
                }
            }
        }

        // Priority 3: SentencePiece model files in Models/ (recursive one level)
        let spNames = ["tokenizer.model", "spiece.model"]
        for spName in spNames {
            let fullPath = (modelsPath as NSString).appendingPathComponent(spName)
            if fm.fileExists(atPath: fullPath), let tokenizer = try? load(from: fullPath) {
                return tokenizer
            }
        }
        // Also scan immediate subdirectories for SP files
        if let subdirs = try? fm.contentsOfDirectory(atPath: modelsPath) {
            for subdir in subdirs {
                let subdirPath = (modelsPath as NSString).appendingPathComponent(subdir)
                for spName in spNames {
                    let fullPath = (subdirPath as NSString).appendingPathComponent(spName)
                    if fm.fileExists(atPath: fullPath), let tokenizer = try? load(from: fullPath) {
                        return tokenizer
                    }
                }
            }
        }

        // Priority 4: Any .tiktoken file in Models/
        if let files = try? fm.contentsOfDirectory(atPath: modelsPath) {
            for file in files where file.hasSuffix(".tiktoken") {
                let fullPath = (modelsPath as NSString).appendingPathComponent(file)
                if let tokenizer = try? load(from: fullPath) {
                    return tokenizer
                }
            }
        }

        // Priority 5: Fallback minimal vocab (demo / test mode)
        print("ℹ️ No tokenizer found, using minimal fallback vocabulary")

        let vocab = ["<BOS>": 0, "<EOS>": 1, "<UNK>": 2, "<PAD>": 3]
        let merges: [[String]] = []
        let specialTokens = BPEVocabulary.SpecialTokens(
            bos_token: "<BOS>",
            eos_token: "<EOS>",
            unk_token: "<UNK>",
            pad_token: "<PAD>"
        )

        return BPETokenizer(
            vocab: vocab,
            merges: merges,
            specialTokens: specialTokens
        )
    }
    
    // MARK: - Path Resolution
    
    private static func resolvePath(_ path: String) -> String? {
        // If absolute, use as-is
        if path.hasPrefix("/") {
            return path
        }
        
        // Try current directory
        if FileManager.default.fileExists(atPath: path) {
            return path
        }
        
        // Try to find project root
        var currentPath = FileManager.default.currentDirectoryPath
        for _ in 0..<10 {
            let packagePath = (currentPath as NSString).appendingPathComponent("Package.swift")
            if FileManager.default.fileExists(atPath: packagePath) {
                let fullPath = (currentPath as NSString).appendingPathComponent(path)
                if FileManager.default.fileExists(atPath: fullPath) {
                    return fullPath
                }
            }
            currentPath = (currentPath as NSString).deletingLastPathComponent
            if currentPath == "/" {
                break
            }
        }
        
        return nil
    }

    private static func tokenizerCandidateURLs(forModelAt modelURL: URL) -> [URL] {
        let modelDirectory = modelURL.deletingLastPathComponent()
        let stem = modelURL.deletingPathExtension().lastPathComponent

        var roots: [URL] = [modelDirectory]
        let parent = modelDirectory.deletingLastPathComponent()
        if parent.path != modelDirectory.path {
            roots.append(parent)
        }

        let rawDirectoryNames = rawDirectoryCandidates(forModelStem: stem)
        var candidates: [URL] = []

        for root in roots {
            for rawDirectoryName in rawDirectoryNames {
                candidates.append(
                    root
                        .appendingPathComponent(rawDirectoryName)
                        .appendingPathComponent("tokenizer.json")
                )
            }
        }

        let colocatedNames = ["tokenizer.json", "spiece.model", "tokenizer.model"]
        for name in colocatedNames {
            candidates.append(modelDirectory.appendingPathComponent(name))
        }

        return uniqueURLs(candidates)
    }

    private static func rawDirectoryCandidates(forModelStem stem: String) -> [String] {
        let normalizedStem = stem.lowercased()
        let strippedQuantization = stripQuantizationSuffix(from: normalizedStem)

        var candidates: [String] = []

        candidates.append(contentsOf: fallbackRawDirectories(for: normalizedStem))

        if !strippedQuantization.isEmpty {
            candidates.append("\(strippedQuantization)-raw")
        }

        if let family = familyPrefixBeforeVersionOrQuantization(in: normalizedStem),
           !family.isEmpty {
            candidates.append("\(family)-raw")
        }

        return uniqueStrings(candidates)
    }

    private static func stripQuantizationSuffix(from stem: String) -> String {
        let quantizationTokens = [
            "int8", "int4", "fp16", "fp32", "q8", "q4", "f16", "f32"
        ]

        var parts = stem.split(separator: "-", omittingEmptySubsequences: true).map(String.init)
        while let last = parts.last,
              quantizationTokens.contains(last) || last == "quantized" {
            parts.removeLast()
        }

        return parts.isEmpty ? stem : parts.joined(separator: "-")
    }

    private static func familyPrefixBeforeVersionOrQuantization(in stem: String) -> String? {
        let quantizationTokens: Set<String> = [
            "int8", "int4", "fp16", "fp32", "q8", "q4", "f16", "f32", "quantized"
        ]
        let separators = CharacterSet(charactersIn: "-_.")
        let parts = stem
            .components(separatedBy: separators)
            .filter { !$0.isEmpty }

        var familyParts: [String] = []
        for part in parts {
            if quantizationTokens.contains(part) || part.first?.isNumber == true {
                break
            }
            familyParts.append(part)
        }

        return familyParts.isEmpty ? nil : familyParts.joined(separator: "-")
    }

    private static func fallbackRawDirectories(for stem: String) -> [String] {
        if stem.hasPrefix("tinyllama") {
            return ["tinyllama-raw"]
        }
        if stem.hasPrefix("gemma") {
            return ["gemma-2b-raw", "gemma-raw"]
        }
        if stem.hasPrefix("sarvam") {
            return ["sarvam-1-raw", "sarvam-raw"]
        }
        return []
    }

    private static func uniqueURLs(_ urls: [URL]) -> [URL] {
        var seen: Set<String> = []
        var result: [URL] = []
        for url in urls where !seen.contains(url.path) {
            seen.insert(url.path)
            result.append(url)
        }
        return result
    }

    private static func uniqueStrings(_ values: [String]) -> [String] {
        var seen: Set<String> = []
        var result: [String] = []
        for value in values where !seen.contains(value) {
            seen.insert(value)
            result.append(value)
        }
        return result
    }
}
