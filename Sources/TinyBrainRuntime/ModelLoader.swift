/// Model Loading Utilities
///
/// Handles loading models from TBF files with fallback strategies.
/// TB-008: Separates model loading concerns from UI layer.

import Foundation

/// Model loading utility with automatic fallback to toy model
public enum ModelLoader {
    
    /// Find the project root directory (where Package.swift lives)
    private static func findProjectRoot() -> String? {
        var currentPath = FileManager.default.currentDirectoryPath
        
        // Check current directory and parents
        for _ in 0..<10 {  // Max 10 levels up
            let packageSwiftPath = (currentPath as NSString).appendingPathComponent("Package.swift")
            if FileManager.default.fileExists(atPath: packageSwiftPath) {
                return currentPath
            }
            
            // Go up one level
            currentPath = (currentPath as NSString).deletingLastPathComponent
            
            // Stop at root
            if currentPath == "/" {
                break
            }
        }
        
        // Try common locations (for Xcode builds)
        let possibleRoots = [
            "/Users/vivekesque/Desktop/CreativeSpace/CodingProjects/tinybrain",  // Absolute path
            NSString(string: #file).deletingLastPathComponent  // Relative to this source file
        ]
        
        for root in possibleRoots {
            let packagePath = (root as NSString).appendingPathComponent("Package.swift")
            if FileManager.default.fileExists(atPath: packagePath) {
                return root
            }
        }
        
        return nil
    }
    
    /// Resolve model path (handles both absolute and project-relative paths)
    private static func resolvePath(_ path: String) -> String {
        // If absolute path, use as-is
        if path.hasPrefix("/") {
            return path
        }
        
        // Try current directory first
        if FileManager.default.fileExists(atPath: path) {
            return path
        }
        
        // Try project root
        if let projectRoot = findProjectRoot() {
            let fullPath = (projectRoot as NSString).appendingPathComponent(path)
            if FileManager.default.fileExists(atPath: fullPath) {
                return fullPath
            }
        }
        
        // Return original path (will fail downstream)
        return path
    }
    
    /// Load model with automatic fallback strategy
    ///
    /// Attempts to load from the specified path, falls back to toy model if:
    /// - File doesn't exist
    /// - File is corrupt
    /// - Loading fails for any reason
    ///
    /// - Parameters:
    ///   - path: Path to .tbf model file
    ///   - fallbackConfig: Configuration for toy model if loading fails
    /// - Returns: Loaded ModelWeights (real or toy)
    public static func loadWithFallback(
        from path: String,
        fallbackConfig: ModelConfig? = nil
    ) -> ModelWeights {
        let resolvedPath = resolvePath(path)
        
        if FileManager.default.fileExists(atPath: resolvedPath) {
            do {
                let weights = try ModelWeights.load(from: resolvedPath)
                return weights
            } catch {
                #if DEBUG
                print("⚠️ Failed to load model: \(error). Falling back to toy model.")
                #endif
            }
        }
        
        // Fallback to toy model
        let config = fallbackConfig ?? ModelConfig(
            numLayers: 2,
            hiddenDim: 128,
            numHeads: 4,
            vocabSize: 100,
            maxSeqLen: 256
        )
        
        return ModelWeights.makeToyModel(config: config, seed: 42)
    }
    
    /// Load model strictly (throws on failure, no fallback)
    ///
    /// Use this when you want to ensure a real model is loaded
    /// and fail fast if it's not available.
    ///
    /// - Parameter path: Path to .tbf model file
    /// - Returns: Loaded ModelWeights
    /// - Throws: TBFError or IO errors
    public static func load(from path: String) throws -> ModelWeights {
        let resolvedPath = resolvePath(path)
        
        guard FileManager.default.fileExists(atPath: resolvedPath) else {
            throw CocoaError(.fileNoSuchFile)
        }
        
        print("🧠 Loading model from: \(resolvedPath)")
        let weights = try ModelWeights.load(from: resolvedPath)
        print("✅ Model loaded! (\(weights.config.numLayers) layers, \(weights.config.hiddenDim) dims)")
        
        return weights
    }
    
    /// Discover and load the best available model
    ///
    /// Search strategy:
    /// 1. Look for TinyLlama in Models/
    /// 2. Look for any .tbf file in Models/
    /// 3. Fallback to toy model
    ///
    /// - Returns: Loaded ModelWeights
    public static func loadBestAvailable() -> ModelWeights {
        // Priority 1: TinyLlama
        let tinyLlamaPath = "Models/tinyllama-1.1b-int8.tbf"
        let resolvedPath = resolvePath(tinyLlamaPath)
        if FileManager.default.fileExists(atPath: resolvedPath) {
            return loadWithFallback(from: tinyLlamaPath)
        }
        
        // Priority 2: Any .tbf file in Models/
        if let modelsURL = FileManager.default.urls(
            for: .documentDirectory,
            in: .userDomainMask
        ).first?.appendingPathComponent("Models"),
           let files = try? FileManager.default.contentsOfDirectory(
            at: modelsURL,
            includingPropertiesForKeys: nil
           ) {
            let tbfFiles = files.filter { $0.pathExtension == "tbf" }
            if let firstModel = tbfFiles.first {
                return loadWithFallback(from: firstModel.path)
            }
        }
        
        // Priority 3: Toy model
        print("ℹ️ No real models found, using toy model")
        let config = ModelConfig(
            numLayers: 2,
            hiddenDim: 128,
            numHeads: 4,
            vocabSize: 100,
            maxSeqLen: 256
        )
        return ModelWeights.makeToyModel(config: config, seed: 42)
    }
}

