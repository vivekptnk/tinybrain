/// Smoke test: load the real TinyLlama model and check inference output
///
/// This test requires Models/tinyllama-1.1b-int8.tbf to exist.
/// It verifies that the full pipeline (load → tokenize → infer → decode) produces coherent output.

import XCTest
@testable import TinyBrainRuntime
import TinyBrainTokenizer
import TinyBrainMetal

final class RealModelSmokeTest: XCTestCase {

    func testRealModelInference() throws {
        // Load model
        let modelPath = "Models/tinyllama-1.1b-int8.tbf"
        let weights: ModelWeights
        do {
            weights = try ModelLoader.load(from: modelPath)
        } catch {
            print("⚠️ Skipping: real model not found at \(modelPath)")
            throw XCTSkip("Real model not available")
        }

        print("✅ Model loaded: \(weights.config.numLayers) layers, \(weights.config.hiddenDim) dims, vocab \(weights.config.vocabSize)")
        print("   numHeads=\(weights.config.numHeads), numKVHeads=\(weights.config.numKVHeads)")
        print("   intermediateDim=\(weights.config.intermediateDim)")

        // Verify config matches TinyLlama
        XCTAssertEqual(weights.config.numLayers, 22)
        XCTAssertEqual(weights.config.hiddenDim, 2048)
        XCTAssertEqual(weights.config.numHeads, 32)
        XCTAssertEqual(weights.config.numKVHeads, 4)  // GQA
        XCTAssertEqual(weights.config.vocabSize, 32000)

        // Load tokenizer
        let tokenizerPath = "Models/tinyllama-raw/tokenizer.json"
        let tokenizer: any Tokenizer
        do {
            let resolvedPath = resolveProjectPath(tokenizerPath)
            tokenizer = try TokenizerLoader.load(from: resolvedPath)
        } catch {
            print("⚠️ Skipping: tokenizer not found at \(tokenizerPath)")
            throw XCTSkip("Tokenizer not available")
        }

        print("✅ Tokenizer loaded")

        // Init Metal
        if MetalBackend.isAvailable {
            do {
                TinyBrainBackend.metalBackend = try MetalBackend()
                print("✅ Metal GPU initialized")
            } catch {
                print("⚠️ Metal init failed: \(error), using CPU")
            }
        }

        // Create runner
        let runner = ModelRunner(weights: weights)

        // Test 1: Simple text completion (matches ChatViewModel format)
        let prompt = "The capital of France is"
        let tokens = [1] + tokenizer.encode(prompt)  // BOS + prompt
        print("✅ Encoded \(tokens.count) tokens")
        print("   First 10: \(Array(tokens.prefix(10)))")

        // Test 2: Run inference for 30 tokens
        let config = GenerationConfig(
            maxTokens: 30,
            sampler: SamplerConfig(temperature: 0.1, topK: 1),  // Near-greedy for deterministic validation
            stopTokens: [2]  // EOS
        )

        var generatedTokens: [Int] = []
        let startTime = Date()

        let expectation = self.expectation(description: "Generation completes")

        Task {
            for try await output in runner.generateStream(prompt: tokens, config: config) {
                generatedTokens.append(output.tokenId)
                if generatedTokens.count <= 10 {
                    let text = tokenizer.decode([output.tokenId])
                    print("   Token \(generatedTokens.count): id=\(output.tokenId) prob=\(String(format: "%.4f", output.probability)) text=\"\(text)\"")
                }
            }
            expectation.fulfill()
        }

        wait(for: [expectation], timeout: 300)  // 5 min timeout for large model

        let elapsed = Date().timeIntervalSince(startTime)
        let decoded = tokenizer.decode(generatedTokens)

        print("\n📊 Results:")
        print("   Generated \(generatedTokens.count) tokens in \(String(format: "%.1f", elapsed))s")
        print("   Speed: \(String(format: "%.1f", Double(generatedTokens.count) / elapsed)) tokens/sec")
        print("   Output: \"\(decoded)\"")
        print("   Token IDs: \(generatedTokens)")

        // Verify we got some output
        XCTAssertGreaterThan(generatedTokens.count, 0, "Should generate at least one token")

        // Verify output contains "Paris" (the capital of France)
        let outputLower = decoded.lowercased()
        let containsAnswer = outputLower.contains("paris")
        print("   Contains 'Paris': \(containsAnswer)")
        XCTAssertTrue(containsAnswer, "Model should know the capital of France is Paris")

        // Check for gibberish indicators
        let uniqueTokens = Set(generatedTokens)
        let uniqueRatio = Float(uniqueTokens.count) / Float(max(generatedTokens.count, 1))
        print("   Unique token ratio: \(String(format: "%.2f", uniqueRatio)) (\(uniqueTokens.count)/\(generatedTokens.count))")

        // A model producing gibberish tends to repeat or have very low uniqueness
        // Real output should have reasonable variety
        XCTAssertGreaterThan(uniqueRatio, 0.1, "Token uniqueness too low — likely gibberish")
    }

    /// Test with simple text completion (no chat template) to isolate template issues
    func testPlainTextCompletion() throws {
        let modelPath = "Models/tinyllama-1.1b-int8.tbf"
        let weights: ModelWeights
        do {
            weights = try ModelLoader.load(from: modelPath)
        } catch {
            throw XCTSkip("Real model not available")
        }

        let tokenizerPath = resolveProjectPath("Models/tinyllama-raw/tokenizer.json")
        let tokenizer: any Tokenizer
        do {
            tokenizer = try TokenizerLoader.load(from: tokenizerPath)
        } catch {
            throw XCTSkip("Tokenizer not available")
        }

        // Init Metal
        if MetalBackend.isAvailable {
            TinyBrainBackend.metalBackend = try? MetalBackend()
        }

        let runner = ModelRunner(weights: weights)

        // Test 1: Simple factual completion (no chat template)
        let prompts = [
            "The capital of France is",
            "1 + 1 =",
            "Once upon a time",
        ]

        for prompt in prompts {
            runner.reset()
            let tokens = [1] + tokenizer.encode(prompt)  // BOS + prompt
            print("\n--- Prompt: \"\(prompt)\" (\(tokens.count) tokens) ---")
            print("   Token IDs: \(tokens)")

            let config = GenerationConfig(
                maxTokens: 20,
                sampler: SamplerConfig(temperature: 0.1, topK: 1),
                stopTokens: [2]
            )

            var generatedTokens: [Int] = []
            let expectation = self.expectation(description: "Gen \(prompt)")

            Task {
                for try await output in runner.generateStream(prompt: tokens, config: config) {
                    generatedTokens.append(output.tokenId)
                }
                expectation.fulfill()
            }

            wait(for: [expectation], timeout: 300)

            let decoded = tokenizer.decode(generatedTokens)
            print("   Output: \"\(decoded)\"")
            print("   Token IDs: \(generatedTokens)")

            XCTAssertGreaterThan(generatedTokens.count, 0)
        }
    }

    /// Test chat template with "What is 2+2?" question
    func testChatTemplateInference() throws {
        let modelPath = "Models/tinyllama-1.1b-int8.tbf"
        let weights: ModelWeights
        do {
            weights = try ModelLoader.load(from: modelPath)
        } catch {
            throw XCTSkip("Real model not available")
        }

        let tokenizerPath = resolveProjectPath("Models/tinyllama-raw/tokenizer.json")
        let tokenizer: any Tokenizer
        do {
            tokenizer = try TokenizerLoader.load(from: tokenizerPath)
        } catch {
            throw XCTSkip("Tokenizer not available")
        }

        if MetalBackend.isAvailable {
            TinyBrainBackend.metalBackend = try? MetalBackend()
        }

        let runner = ModelRunner(weights: weights)

        // Use TinyLlama-Chat's Zephyr format
        let prompt = "<|system|>\nYou are a helpful assistant.</s>\n<|user|>\nWhat is 2+2?</s>\n<|assistant|>\n"
        let tokens = [1] + tokenizer.encode(prompt)
        print("Chat prompt encoded to \(tokens.count) tokens: \(tokens)")

        let config = GenerationConfig(
            maxTokens: 30,
            sampler: SamplerConfig(temperature: 0.7, topK: 40),
            stopTokens: [2]
        )

        var generatedTokens: [Int] = []
        let expectation = self.expectation(description: "Chat generation")
        Task {
            for try await output in runner.generateStream(prompt: tokens, config: config) {
                generatedTokens.append(output.tokenId)
            }
            expectation.fulfill()
        }
        wait(for: [expectation], timeout: 300)

        let decoded = tokenizer.decode(generatedTokens)
        print("Chat output: \"\(decoded)\"")
        print("Token IDs: \(generatedTokens)")

        // Verify encoding matches HuggingFace reference exactly
        // HF produces: [1, 529, 29989, 5205, 29989, 29958, 13, 3492, 526, 263, ...]
        XCTAssertEqual(tokens[6], 13, "Newline should encode as <0x0A> (id=13), not UNK")
        XCTAssertEqual(tokens.count, 39, "Should match HuggingFace reference encoding length")
    }

    private func resolveProjectPath(_ path: String) -> String {
        // Try current directory
        if FileManager.default.fileExists(atPath: path) { return path }

        // Try project root
        var dir = FileManager.default.currentDirectoryPath
        for _ in 0..<10 {
            let pkg = (dir as NSString).appendingPathComponent("Package.swift")
            if FileManager.default.fileExists(atPath: pkg) {
                let full = (dir as NSString).appendingPathComponent(path)
                if FileManager.default.fileExists(atPath: full) { return full }
            }
            dir = (dir as NSString).deletingLastPathComponent
            if dir == "/" { break }
        }
        return path
    }
}
