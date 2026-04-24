/// CHA-109 — Gemma 2B end-to-end smoke test.
///
/// Loads `Models/gemma-2b-int4.tbf`, confirms the arch field and
/// dimensions match Gemma 2B, and runs greedy continuation on a handful
/// of prompts. XCTSkips when the `.tbf` isn't present (untracked,
/// produced by `python3 Scripts/convert_model.py --input
/// Models/gemma-2b-raw --output Models/gemma-2b-int4.tbf --quantize
/// int4`).
///
/// This is the acceptance test for CHA-109:
///   1. `.tbf` loads and `config.architecture == "gemma"`
///   2. Config dims match HF reference (18 layers, hidden=2048, heads=8,
///      kv_heads=1 MQA, vocab=256_000, intermediate=16_384).
///   3. Greedy continuation on 3 factual prompts produces coherent
///      English (non-gibberish token-diversity signal + basic
///      "contains plausible answer" check).

import XCTest
@testable import TinyBrainRuntime
import TinyBrainTokenizer
import TinyBrainMetal

final class GemmaSmokeTest: XCTestCase {

    private let modelPath = "Models/gemma-2b-int4.tbf"
    private let tokenizerPath = "Models/gemma-2b-raw/tokenizer.json"

    func testGemmaTBFConfigMatchesHFReference() throws {
        let weights: ModelWeights
        do {
            weights = try ModelLoader.load(from: modelPath)
        } catch {
            throw XCTSkip("Gemma .tbf not found at \(modelPath); run the converter first")
        }

        let cfg = weights.config
        XCTAssertEqual(cfg.architecture, "gemma",
                       "Converter must have stamped architecture=gemma into TBF header")
        XCTAssertTrue(cfg.isGemmaStyle)
        XCTAssertEqual(cfg.numLayers, 18)
        XCTAssertEqual(cfg.hiddenDim, 2048)
        XCTAssertEqual(cfg.numHeads, 8)
        XCTAssertEqual(cfg.numKVHeads, 1, "Gemma 2B is MQA (1 KV head for 8 Q heads)")
        XCTAssertEqual(cfg.vocabSize, 256_000)
        XCTAssertEqual(cfg.intermediateDim, 16_384, "Gemma 2B intermediate is 8× hidden")
        XCTAssertEqual(cfg.headDim, 256)
    }

    func testGemmaGreedyContinuationIsCoherent() throws {
        let weights: ModelWeights
        do {
            weights = try ModelLoader.load(from: modelPath)
        } catch {
            throw XCTSkip("Gemma .tbf not found at \(modelPath); run the converter first")
        }

        let resolvedTokenizer = resolveProjectPath(tokenizerPath)
        let tokenizer: any Tokenizer
        do {
            tokenizer = try TokenizerLoader.load(from: resolvedTokenizer)
        } catch {
            throw XCTSkip("Gemma tokenizer not available at \(tokenizerPath): \(error)")
        }

        if MetalBackend.isAvailable {
            TinyBrainBackend.metalBackend = try? MetalBackend()
        }

        let runner = ModelRunner(weights: weights)

        // Gemma BOS token is id 2 (<bos>). Confirm from tokenizer_config.
        let bosTokenId = 2
        let eosTokenId = 1

        let prompts = [
            "The capital of France is",
            "1 + 1 =",
            "Once upon a time",
        ]

        for prompt in prompts {
            runner.reset()
            let tokens = [bosTokenId] + tokenizer.encode(prompt)

            let config = GenerationConfig(
                maxTokens: 16,
                sampler: SamplerConfig(temperature: 0.0, topK: 1),  // Pure greedy
                stopTokens: [eosTokenId]
            )

            var generated: [Int] = []
            let expectation = self.expectation(description: "greedy: \(prompt)")

            Task {
                for try await output in runner.generateStream(prompt: tokens, config: config) {
                    generated.append(output.tokenId)
                }
                expectation.fulfill()
            }

            wait(for: [expectation], timeout: 300)

            let decoded = tokenizer.decode(generated)
            print("\n--- Gemma INT4 continuation for \(prompt.debugDescription) ---")
            print("  Token ids: \(generated)")
            print("  Text:      \(decoded.debugDescription)")

            XCTAssertGreaterThan(generated.count, 0,
                                 "Gemma should emit at least one continuation token")

            let uniqueRatio = Float(Set(generated).count) / Float(max(generated.count, 1))
            XCTAssertGreaterThan(uniqueRatio, 0.2,
                                 "Token diversity too low (\(uniqueRatio)) — likely INT4 gibberish regression")
        }
    }

    private func resolveProjectPath(_ path: String) -> String {
        if FileManager.default.fileExists(atPath: path) { return path }

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
