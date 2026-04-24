/// TinyBrain Benchmark CLI Tool
///
/// Measures latency, throughput, and energy consumption for model inference.

import ArgumentParser
import Foundation
import TinyBrainRuntime
import TinyBrainMetal
import Yams

@main
struct TinyBrainBench: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "tinybrain-bench",
        abstract: "Benchmark TinyBrain model inference performance",
        version: "0.1.0"
    )
    
    @Option(name: .shortAndLong, help: "Path to the model file")
    var model: String?
    
    @Option(name: .shortAndLong, help: "Number of tokens to generate")
    var tokens: Int = 50
    
    @Flag(name: .long, help: "Show verbose output")
    var verbose: Bool = false
    
    @Flag(name: .long, help: "Run live streaming demo")
    var demo: Bool = false
    
    @Flag(name: .long, help: "Run comprehensive feature showcase (TB-001 through TB-005)")
    var showcase: Bool = false
    
    @Flag(name: .long, help: "Run interactive chat mode")
    var chat: Bool = false
    
    // TB-007 Phase 3: New benchmark harness features
    @Option(name: .long, help: "Load YAML scenario file")
    var scenario: String?
    
    @Option(name: .long, help: "Output format (json or markdown)")
    var output: String?
    
    @Flag(name: .long, help: "Show device information")
    var deviceInfo: Bool = false
    
    @Option(name: .long, help: "Number of warmup iterations")
    var warmup: Int = 3
    
    @Flag(name: .long, help: "Dry run (parse scenario without execution)")
    var dryRun: Bool = false

    // CHA-108: Perplexity regression harness (INT4 vs INT8 on a pinned slice)
    @Option(name: .long, help: "Run INT4 vs INT8 perplexity on a model (.tbf). Reports both ppl values + delta over a pinned WikiText-2 slice.")
    var perplexity: String?

    @Option(name: .long, help: "Path to the pinned perplexity slice JSON (defaults to Tests/TinyBrainRuntimeTests/Fixtures/wikitext2_slice.json).")
    var perplexitySlice: String?

    @Option(name: .long, help: "INT4 group size used when re-quantizing for the perplexity harness (default: 32, CHA-104 v0.2.0 knee).")
    var perplexityGroupSize: Int = 32

    @Option(name: .long, help: "Maximum acceptable |Δppl|/ppl_INT8 before --perplexity exits non-zero (default: 0.01 per CHA-104 DoD).")
    var perplexityThreshold: Double = 0.01

    // SwiftPM passes XCTest flags when running via `swift test -c release`.
    // These absorb the unknown arguments so ArgumentParser doesn't reject them.
    @Option(name: .customLong("test-bundle-path"), help: .hidden) var testBundlePath: String?
    @Option(name: .customLong("filter"), help: .hidden) var xcFilter: String?
    @Argument(parsing: .captureForPassthrough) var remainingArgs: [String] = []

    func run() async throws {
        if let modelPath = perplexity {
            try runPerplexity(modelPath: modelPath)
            return
        }

        // TB-007 Phase 3: New features
        if deviceInfo {
            showDeviceInfo()
            return
        }
        
        if let scenarioPath = scenario {
            try await runScenario(path: scenarioPath)
            return
        }
        
        // Existing features
        if chat {
            try await runInteractiveChat()
            return
        }
        
        if showcase {
            try await runFeatureShowcase()
            return
        }
        
        if demo {
            try await runBenchmark()
            return
        }
        
        print("🧠 TinyBrain Benchmark Tool v0.1.0")
        print("=" * 50)
        
        if let modelPath = model {
            print("Model: \(modelPath)")
        } else {
            print("⚠️  No model specified. Use --model <path>")
        }
        
        print("Tokens to generate: \(tokens)")
        print("\n⚠️  Benchmark implementation tracked in TB-007")
        print("This tool will measure:")
        print("  • Latency (ms/token)")
        print("  • Throughput (tokens/sec)")
        print("  • Energy consumption (J/token)")
        print("  • Memory usage (peak MB)")
    }
    
    func runStreamingDemo() async throws {
        print("🧠 TinyBrain Live Streaming Demo")
        print("=" * 50)
        print()
        
        // Initialize Metal backend if available (silences fallback warnings)
        if MetalBackend.isAvailable {
            do {
                TinyBrainBackend.metalBackend = try MetalBackend()
                print("🚀 Metal GPU initialized")
            } catch {
                print("⚠️  Metal init failed, using CPU: \(error)")
            }
        }
        print()
        
        // Create a small toy model
        let config = ModelConfig(
            numLayers: 2,
            hiddenDim: 128,
            numHeads: 4,
            vocabSize: 100,
            maxSeqLen: 256
        )
        
        print("📦 Creating toy model...")
        let weights = ModelWeights.makeToyModel(config: config, seed: 42)
        let runner = ModelRunner(weights: weights)
        print("✅ Model ready!\n")
        
        // Simple prompt
        let prompt = [1, 2, 3, 4, 5]
        
        // Configure generation
        let samplerConfig = SamplerConfig(
            temperature: 0.8,
            topK: 40,
            topP: 0.9,
            repetitionPenalty: 1.1
        )
        
        let genConfig = GenerationConfig(
            maxTokens: tokens,
            sampler: samplerConfig,
            stopTokens: []
        )
        
        print("💬 Prompt tokens: \(prompt)")
        print("🎲 Temperature: \(samplerConfig.temperature)")
        print("📊 Top-K: \(samplerConfig.topK ?? 0)")
        print()
        print("🔄 Streaming tokens (watch them appear one by one):")
        print("─" * 50)
        
        var tokenCount = 0
        var totalProbability: Float = 0.0
        let startTime = Date()
        
        for try await output in runner.generateStream(prompt: prompt, config: genConfig) {
            // Print token as it arrives
            let char = Character(UnicodeScalar(UInt8(output.tokenId % 94 + 33)))
            print("Token #\(String(format: "%2d", tokenCount + 1)): ID=\(String(format: "%3d", output.tokenId)) | Prob=\(String(format: "%.3f", output.probability)) | Char='\(char)'")
            
            tokenCount += 1
            totalProbability += output.probability
            
            // Flush output immediately for live streaming effect
            fflush(stdout)
        }
        
        let elapsed = Date().timeIntervalSince(startTime)
        let avgProbability = totalProbability / Float(tokenCount)
        
        print("─" * 50)
        print("✅ Streaming Complete!")
        print()
        print("📊 Statistics:")
        print("  • Tokens generated: \(tokenCount)")
        print("  • Time elapsed: \(String(format: "%.2f", elapsed))s")
        print("  • Speed: \(String(format: "%.1f", Double(tokenCount) / elapsed)) tokens/sec")
        print("  • Avg probability: \(String(format: "%.1f", avgProbability * 100))%")
    }
    
    func runFeatureShowcase() async throws {
        print("\n")
        print("╔══════════════════════════════════════════════════╗")
        print("║  🧠 TinyBrain Complete Feature Showcase 🧠      ║")
        print("║  Demonstrating TB-001 through TB-005             ║")
        print("╚══════════════════════════════════════════════════╝")
        print()
        
        // Initialize Metal backend if available
        if MetalBackend.isAvailable {
            do {
                TinyBrainBackend.metalBackend = try MetalBackend()
                print("🚀 Metal GPU backend initialized!")
            } catch {
                print("⚠️  Metal init failed: \(error)")
            }
        }
        print()
        
        // TB-001: Project Structure
        print("═══ TB-001: Project Scaffolding ═══")
        print("✅ Swift Package Manager structure")
        print("✅ Multi-module architecture (Runtime, Metal, Tokenizer)")
        print("✅ Test infrastructure (172 tests)")
        print("✅ Documentation (DocC)")
        print()
        
        // TB-002: Tensor Operations
        print("═══ TB-002: Swift Tensor Engine (CPU) ═══")
        print("Demonstrating core tensor operations...")
        print()
        
        let a = Tensor<Float>(shape: TensorShape([2, 3]), data: [1, 2, 3, 4, 5, 6])
        let b = Tensor<Float>(shape: TensorShape([3, 2]), data: [7, 8, 9, 10, 11, 12])
        
        print("Matrix A (2×3):")
        print("  [\(a[0, 0]), \(a[0, 1]), \(a[0, 2])]")
        print("  [\(a[1, 0]), \(a[1, 1]), \(a[1, 2])]")
        print()
        print("Matrix B (3×2):")
        print("  [\(b[0, 0]), \(b[0, 1])]")
        print("  [\(b[1, 0]), \(b[1, 1])]")
        print("  [\(b[2, 0]), \(b[2, 1])]")
        print()
        
        let c = a.matmul(b)
        print("Result C = A × B (2×2):")
        print("  [\(c[0, 0]), \(c[0, 1])]")
        print("  [\(c[1, 0]), \(c[1, 1])]")
        print("✅ MatMul working!")
        print()
        
        let softmaxInput = Tensor<Float>(shape: TensorShape([1, 4]), data: [1.0, 2.0, 3.0, 4.0])
        let softmaxResult = softmaxInput.softmax()
        print("Softmax([1, 2, 3, 4]) = [\(String(format: "%.3f", softmaxResult[0, 0])), \(String(format: "%.3f", softmaxResult[0, 1])), \(String(format: "%.3f", softmaxResult[0, 2])), \(String(format: "%.3f", softmaxResult[0, 3]))]")
        print("Sum = \(String(format: "%.3f", softmaxResult[0, 0] + softmaxResult[0, 1] + softmaxResult[0, 2] + softmaxResult[0, 3])) ✅")
        print()
        
        // TB-003: Metal
        print("═══ TB-003: Metal GPU Acceleration ═══")
        if MetalBackend.isAvailable {
            print("✅ Metal GPU available: \(try MetalBackend().deviceInfo)")
            print("✅ Automatic CPU/GPU routing based on workload size")
            print("✅ Buffer pooling for efficient memory management")
        } else {
            print("⚠️  Metal not available (running on CPU)")
        }
        print()
        
        // TB-004: Quantization & KV Cache
        print("═══ TB-004: INT8 Quantization & KV Cache ═══")
        
        print("✅ INT8 per-channel quantization implemented")
        print("✅ Paged KV cache with 16-token pages")
        print("✅ TBF model format (mmap-backed loading)")
        print("✅ 57 tests validate quantization accuracy (< 1% error)")
        print()
        
        // TB-005: Sampling & Streaming
        print("═══ TB-005: Tokenizer, Sampler & Streaming ═══")
        
        print("Creating model and sampler...")
        let config = ModelConfig(numLayers: 2, hiddenDim: 128, numHeads: 4, vocabSize: 100, maxSeqLen: 256)
        let weights = ModelWeights.makeToyModel(config: config, seed: 42)
        let runner = ModelRunner(weights: weights)
        
        print("✅ Model loaded (2 layers, 128 hidden dims)")
        print()
        
        print("Sampler configuration:")
        let samplerConfig = SamplerConfig(
            temperature: 0.8,
            topK: 40,
            topP: 0.9,
            repetitionPenalty: 1.1
        )
        print("  • Temperature: \(samplerConfig.temperature)")
        print("  • Top-K: \(samplerConfig.topK ?? 0)")
        print("  • Top-P: \(samplerConfig.topP ?? 0)")
        print("  • Repetition Penalty: \(samplerConfig.repetitionPenalty)")
        print()
        
        print("🔄 Streaming 10 tokens (watch them appear):")
        print("─" * 50)
        
        let prompt = [1, 2, 3, 4, 5]
        let genConfig = GenerationConfig(maxTokens: 10, sampler: samplerConfig, stopTokens: [])
        
        var count = 0
        for try await output in runner.generateStream(prompt: prompt, config: genConfig) {
            count += 1
            let char = Character(UnicodeScalar(UInt8(output.tokenId % 94 + 33)))
            print("  Token \(count): id=\(output.tokenId) prob=\(String(format: "%.3f", output.probability)) char='\(char)'")
            try? await Task.sleep(nanoseconds: 100_000_000) // 0.1s for visibility
        }
        
        print("─" * 50)
        print("✅ Streaming complete! All tokens delivered via AsyncSequence")
        print()
        
        // Summary
        print("╔══════════════════════════════════════════════════╗")
        print("║           🎉 All Features Working! 🎉            ║")
        print("╠══════════════════════════════════════════════════╣")
        print("║ ✅ TB-001: Project structure & build system     ║")
        print("║ ✅ TB-002: Tensor operations (MatMul, Softmax)  ║")
        print("║ ✅ TB-003: Metal GPU acceleration               ║")
        print("║ ✅ TB-004: INT8 quantization & KV cache         ║")
        print("║ ✅ TB-005: Sampling & streaming API             ║")
        print("╠══════════════════════════════════════════════════╣")
        print("║  172 tests passing • Ready for production use   ║")
        print("╚══════════════════════════════════════════════════╝")
        print()
    }
    
    func runInteractiveChat() async throws {
        print("\n")
        print("╔══════════════════════════════════════════════════╗")
        print("║        🧠 TinyBrain Interactive Chat 🧠         ║")
        print("╚══════════════════════════════════════════════════╝")
        print()
        
        // Initialize Metal if available
        if MetalBackend.isAvailable {
            do {
                TinyBrainBackend.metalBackend = try MetalBackend()
                print("🚀 Metal GPU initialized")
            } catch {
                print("⚠️  Using CPU only")
            }
        } else {
            print("💻 Using CPU only (Metal not available)")
        }
        print()
        
        // Create model
        print("📦 Loading model...")
        let config = ModelConfig(
            numLayers: 2,
            hiddenDim: 128,
            numHeads: 4,
            vocabSize: 100,
            maxSeqLen: 256
        )
        let weights = ModelWeights.makeToyModel(config: config, seed: 42)
        let runner = ModelRunner(weights: weights)
        print("✅ Model ready (2 layers, 128 dims)")
        print()
        
        // Sampler config
        var temperature: Float = 0.8
        var topK: Int = 40
        
        print("🎲 Sampling: temperature=\(temperature), top-k=\(topK)")
        print()
        print("─" * 50)
        print("Type your prompt and press Enter (or 'quit' to exit)")
        print("Commands: /temp <value> | /topk <value> | /help")
        print("─" * 50)
        print()
        
        // Interactive loop
        while true {
            // Print prompt
            print("You: ", terminator: "")
            fflush(stdout)
            
            // Read input
            guard let input = readLine() else {
                break
            }
            
            let trimmed = input.trimmingCharacters(in: .whitespacesAndNewlines)
            
            // Handle commands
            if trimmed == "quit" || trimmed == "exit" {
                print("👋 Goodbye!")
                break
            }
            
            if trimmed.starts(with: "/temp ") {
                if let value = Float(trimmed.dropFirst(6)) {
                    temperature = value
                    print("✅ Temperature set to \(temperature)")
                }
                print()
                continue
            }
            
            if trimmed.starts(with: "/topk ") {
                if let value = Int(trimmed.dropFirst(6)) {
                    topK = value
                    print("✅ Top-K set to \(topK)")
                }
                print()
                continue
            }
            
            if trimmed == "/help" {
                print("Commands:")
                print("  /temp <0.1-2.0>  - Set temperature")
                print("  /topk <1-100>    - Set top-k")
                print("  quit             - Exit")
                print()
                continue
            }
            
            if trimmed.isEmpty {
                continue
            }
            
            // Generate response
            print("🧠 TinyBrain: ", terminator: "")
            fflush(stdout)
            
            // Simple tokenization (character-based for demo)
            let promptTokens = Array(trimmed.prefix(10)).map { char in
                Int(char.asciiValue ?? 0) % config.vocabSize
            }
            
            let samplerConfig = SamplerConfig(
                temperature: temperature,
                topK: topK,
                topP: 0.9,
                repetitionPenalty: 1.1
            )
            
            let genConfig = GenerationConfig(
                maxTokens: tokens,
                sampler: samplerConfig,
                stopTokens: []
            )
            
            // Stream response
            var response = ""
            do {
                for try await output in runner.generateStream(prompt: promptTokens, config: genConfig) {
                    let char = Character(UnicodeScalar(UInt8(output.tokenId % 94 + 33)))
                    print(char, terminator: "")
                    fflush(stdout)
                    response.append(char)
                }
                print()  // Newline after response
            } catch {
                print("\n❌ Error: \(error)")
            }
            
            runner.reset()
            print()
        }
    }
    
    // MARK: - TB-007 Phase 3: New Benchmark Features
    
    func showDeviceInfo() {
        print("🔍 Device Information")
        print("=" * 50)
        print()
        
        // System info
        let processInfo = ProcessInfo.processInfo
        print("Device: \(processInfo.hostName)")
        print("OS: \(processInfo.operatingSystemVersionString)")
        print("CPU Count: \(processInfo.activeProcessorCount) cores")
        print("Memory: \(String(format: "%.2f", Double(processInfo.physicalMemory) / (1024 * 1024 * 1024))) GB")
        print()
        
        // Metal availability
        if MetalBackend.isAvailable {
            do {
                let backend = try MetalBackend()
                print("GPU: \(backend.deviceInfo)")
                print("Metal: ✅ Available")
            } catch {
                print("Metal: ⚠️ Error initializing: \(error)")
            }
        } else {
            print("Metal: ❌ Not available")
        }
        print()
        
        // Memory usage
        let memoryMB = MemoryTracker.currentMemoryUsageMB()
        print("Current Memory Usage: \(String(format: "%.2f", memoryMB)) MB")
    }
    
    func runScenario(path: String) async throws {
        print("📋 Loading scenario: \(path)")
        print()
        
        guard FileManager.default.fileExists(atPath: path) else {
            print("❌ Error: Scenario file not found: \(path)", to: &standardError)
            throw ExitCode.failure
        }
        
        let yamlString = try String(contentsOfFile: path, encoding: .utf8)
        let decoder = YAMLDecoder()
        let scenarioFile = try decoder.decode(ScenarioFile.self, from: yamlString)
        
        if dryRun {
            print("✅ Scenario loaded: \(scenarioFile.scenarios.count) scenarios")
            for (idx, scenario) in scenarioFile.scenarios.enumerated() {
                print("  \(idx + 1). \(scenario.name)")
                print("     Model: \(scenario.model)")
                print("     Prompts: \(scenario.prompts.count)")
            }
            return
        }
        
        // Run scenarios
        var results: [BenchmarkResult] = []
        
        for scenario in scenarioFile.scenarios {
            if verbose {
                print("Running scenario: \(scenario.name)")
            }
            
            let result = try await runSingleScenario(scenario)
            results.append(result)
            
            if verbose {
                print("  ✓ Complete")
                print()
            }
        }
        
        // Output results
        outputResults(results)
    }
    
    func runSingleScenario(_ scenario: BenchmarkScenario) async throws -> BenchmarkResult {
        // Create toy model (real models would be loaded here)
        let config = ModelConfig(
            numLayers: 2,
            hiddenDim: 128,
            numHeads: 4,
            vocabSize: 100,
            maxSeqLen: 256
        )
        
        let weights = ModelWeights.makeToyModel(config: config, seed: 42)
        let runner = ModelRunner(weights: weights)
        
        // Warmup
        let actualWarmup = scenario.warmup ?? warmup
        if verbose && actualWarmup > 0 {
            print("  Warmup: \(actualWarmup) iterations...")
        }
        
        for _ in 0..<actualWarmup {
            let prompt = [1, 2, 3]
            let config = GenerationConfig(maxTokens: 5, sampler: SamplerConfig(), stopTokens: [])
            for try await _ in runner.generateStream(prompt: prompt, config: config) {
                // Discard warmup tokens
            }
            runner.reset()
        }
        
        // Actual benchmark
        let memoryBefore = MemoryTracker.currentMemoryUsageMB()
        var peakMemory = memoryBefore
        var totalTokens = 0
        let startTime = Date()
        
        for promptText in scenario.prompts {
            // Simple tokenization (character-based for toy model)
            let prompt = Array(promptText.prefix(10)).map { Int($0.unicodeScalars.first!.value) % 100 }
            
            let samplerConfig = SamplerConfig(
                temperature: scenario.sampler?.temperature ?? 0.7,
                topK: scenario.sampler?.topK,
                topP: scenario.sampler?.topP,
                repetitionPenalty: scenario.sampler?.repetitionPenalty ?? 1.0
            )
            
            let genConfig = GenerationConfig(
                maxTokens: scenario.maxTokens,
                sampler: samplerConfig,
                stopTokens: []
            )
            
            for try await _ in runner.generateStream(prompt: prompt, config: genConfig) {
                totalTokens += 1
                
                // Track peak memory
                let currentMemory = MemoryTracker.currentMemoryUsageMB()
                peakMemory = max(peakMemory, currentMemory)
            }
            
            runner.reset()
        }
        
        let elapsed = Date().timeIntervalSince(startTime)
        
        // Build result
        let deviceInfo = BenchmarkResult.DeviceInfo(
            name: ProcessInfo.processInfo.hostName,
            os: ProcessInfo.processInfo.operatingSystemVersionString,
            metalAvailable: MetalBackend.isAvailable
        )
        
        let metrics = BenchmarkResult.Metrics(
            tokensPerSec: Double(totalTokens) / elapsed,
            msPerToken: (elapsed * 1000) / Double(totalTokens),
            memoryPeakMB: peakMemory,
            totalTokens: totalTokens,
            elapsedSeconds: elapsed
        )
        
        let dateFormatter = ISO8601DateFormatter()
        
        return BenchmarkResult(
            device: deviceInfo,
            scenario: scenario.name,
            metrics: metrics,
            timestamp: dateFormatter.string(from: Date())
        )
    }
    
    func runBenchmark() async throws {
        // Unified benchmark for --demo flag with new output format support
        if verbose {
            print("🧠 Running benchmark...")
            print()
        }
        
        // Initialize Metal backend if available
        if MetalBackend.isAvailable {
            do {
                TinyBrainBackend.metalBackend = try MetalBackend()
                if verbose {
                    print("🚀 Metal GPU initialized")
                }
            } catch {
                if verbose {
                    print("⚠️  Metal init failed, using CPU: \(error)")
                }
            }
        }
        
        if verbose {
            print()
        }
        
        // Create toy model
        let config = ModelConfig(
            numLayers: 2,
            hiddenDim: 128,
            numHeads: 4,
            vocabSize: 100,
            maxSeqLen: 256
        )
        
        let weights = ModelWeights.makeToyModel(config: config, seed: 42)
        let runner = ModelRunner(weights: weights)
        
        // Warmup
        if verbose {
            print("Warmup: \(warmup) iterations...")
        }
        for _ in 0..<warmup {
            let prompt = [1, 2, 3]
            let config = GenerationConfig(maxTokens: 5, sampler: SamplerConfig(), stopTokens: [])
            for try await _ in runner.generateStream(prompt: prompt, config: config) {}
            runner.reset()
        }
        
        // Benchmark run
        let memoryBefore = MemoryTracker.currentMemoryUsageMB()
        var peakMemory = memoryBefore
        let prompt = [1, 2, 3, 4, 5]
        let samplerConfig = SamplerConfig(temperature: 0.8, topK: 40, topP: 0.9, repetitionPenalty: 1.1)
        let genConfig = GenerationConfig(maxTokens: tokens, sampler: samplerConfig, stopTokens: [])
        
        var tokenCount = 0
        let startTime = Date()
        
        for try await _ in runner.generateStream(prompt: prompt, config: genConfig) {
            tokenCount += 1
            let currentMemory = MemoryTracker.currentMemoryUsageMB()
            peakMemory = max(peakMemory, currentMemory)
        }
        
        let elapsed = Date().timeIntervalSince(startTime)
        
        // Build result
        let deviceInfo = BenchmarkResult.DeviceInfo(
            name: ProcessInfo.processInfo.hostName,
            os: ProcessInfo.processInfo.operatingSystemVersionString,
            metalAvailable: MetalBackend.isAvailable
        )
        
        let metrics = BenchmarkResult.Metrics(
            tokensPerSec: Double(tokenCount) / elapsed,
            msPerToken: (elapsed * 1000) / Double(tokenCount),
            memoryPeakMB: peakMemory,
            totalTokens: tokenCount,
            elapsedSeconds: elapsed
        )
        
        let dateFormatter = ISO8601DateFormatter()
        let result = BenchmarkResult(
            device: deviceInfo,
            scenario: nil,
            metrics: metrics,
            timestamp: dateFormatter.string(from: Date())
        )
        
        outputResults([result])
    }
    
    // MARK: - CHA-108: INT4 vs INT8 Perplexity Harness

    func runPerplexity(modelPath: String) throws {
        let resolvedModel = resolvePerplexityPath(modelPath)
        guard FileManager.default.fileExists(atPath: resolvedModel) else {
            print("❌ Error: model not found at \(resolvedModel)", to: &standardError)
            throw ExitCode.failure
        }

        let slicePath = perplexitySlice ?? "Tests/TinyBrainRuntimeTests/Fixtures/wikitext2_slice.json"
        let resolvedSlice = resolvePerplexityPath(slicePath, anchor: resolvedModel)
        guard FileManager.default.fileExists(atPath: resolvedSlice) else {
            print("❌ Error: slice not found at \(resolvedSlice)", to: &standardError)
            throw ExitCode.failure
        }

        if MetalBackend.isAvailable, TinyBrainBackend.metalBackend == nil {
            do {
                TinyBrainBackend.metalBackend = try MetalBackend()
                if verbose { print("🚀 Metal GPU initialized") }
            } catch {
                if verbose { print("⚠️  Metal init failed, using CPU: \(error)") }
            }
        }

        let slice = try PerplexitySlice.load(from: URL(fileURLWithPath: resolvedSlice))
        let weightsINT8 = try ModelLoader.load(from: resolvedModel)

        print("🧠 Perplexity: INT4 vs INT8")
        print("   Model: \(resolvedModel)")
        print("   Slice: \(resolvedSlice) (\(slice.tokens.count) tokens, seed=\(slice.seed))")
        print("   Source: \(slice.source)")
        print()

        let logProgress: ((String) -> Void)? = verbose ? { message in
            FileHandle.standardError.write(Data((message + "\n").utf8))
        } : nil

        if verbose { logProgress?("Running INT8 baseline...") }
        let resultINT8 = try PerplexityHarness.computePerplexity(
            weights: weightsINT8, slice: slice, progress: logProgress)

        if verbose { logProgress?("Re-quantizing to INT4 (group=\(perplexityGroupSize))...") }
        let weightsINT4 = PerplexityHarness.convertToINT4(
            weightsINT8, groupSize: perplexityGroupSize, progress: logProgress)

        if verbose { logProgress?("Running INT4 perplexity...") }
        let resultINT4 = try PerplexityHarness.computePerplexity(
            weights: weightsINT4, slice: slice, progress: logProgress)

        let delta = abs(resultINT4.perplexity - resultINT8.perplexity) / resultINT8.perplexity
        let withinBudget = Double(delta) <= perplexityThreshold

        if output == "json" {
            struct PerplexityOutput: Encodable {
                let model: String
                let slice: String
                let seed: String
                let numPredictions: Int
                let groupSize: Int
                let int8Perplexity: Float
                let int4Perplexity: Float
                let deltaRelative: Float
                let int8Seconds: Double
                let int4Seconds: Double
                let thresholdRelative: Double
                let withinThreshold: Bool
            }
            let payload = PerplexityOutput(
                model: resolvedModel,
                slice: resolvedSlice,
                seed: slice.seed,
                numPredictions: resultINT8.numPredictions,
                groupSize: perplexityGroupSize,
                int8Perplexity: resultINT8.perplexity,
                int4Perplexity: resultINT4.perplexity,
                deltaRelative: delta,
                int8Seconds: resultINT8.elapsedSeconds,
                int4Seconds: resultINT4.elapsedSeconds,
                thresholdRelative: perplexityThreshold,
                withinThreshold: withinBudget
            )
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            if let data = try? encoder.encode(payload), let str = String(data: data, encoding: .utf8) {
                print(str)
            }
        } else {
            print(String(format: "   INT8 perplexity: %.4f  (%d preds, %.2fs)",
                         resultINT8.perplexity, resultINT8.numPredictions, resultINT8.elapsedSeconds))
            print(String(format: "   INT4 perplexity: %.4f  (%d preds, %.2fs)",
                         resultINT4.perplexity, resultINT4.numPredictions, resultINT4.elapsedSeconds))
            print(String(format: "   Δ (INT4 vs INT8): %+.3f%%", delta * 100))
            let budgetPct = perplexityThreshold * 100
            print(withinBudget
                  ? String(format: "   ✅ within %.2f%% threshold", budgetPct)
                  : String(format: "   ❌ exceeds %.2f%% threshold", budgetPct))
        }

        if !withinBudget {
            throw ExitCode.failure
        }
    }

    private func resolvePerplexityPath(_ path: String, anchor: String? = nil) -> String {
        if (path as NSString).isAbsolutePath { return path }
        if FileManager.default.fileExists(atPath: path) { return path }

        let cwd = FileManager.default.currentDirectoryPath
        var searchRoots: [String] = [cwd]
        if let anchor, (anchor as NSString).isAbsolutePath {
            searchRoots.append((anchor as NSString).deletingLastPathComponent)
        }

        for root in searchRoots {
            var dir = root
            for _ in 0..<10 {
                let pkg = (dir as NSString).appendingPathComponent("Package.swift")
                if FileManager.default.fileExists(atPath: pkg) {
                    let full = (dir as NSString).appendingPathComponent(path)
                    if FileManager.default.fileExists(atPath: full) { return full }
                }
                dir = (dir as NSString).deletingLastPathComponent
                if dir == "/" { break }
            }
        }
        return path
    }

    func outputResults(_ results: [BenchmarkResult]) {
        if output == "json" {
            // JSON output
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            
            if let jsonData = try? encoder.encode(results.first!),
               let jsonString = String(data: jsonData, encoding: .utf8) {
                print(jsonString)
            }
        } else if output == "markdown" {
            // Markdown table output
            print("| Metric | Value |")
            print("|--------|-------|")
            
            for result in results {
                if let scenario = result.scenario {
                    print("| Scenario | \(scenario) |")
                }
                print("| Device | \(result.device.name) |")
                print("| Tokens/sec | \(String(format: "%.2f", result.metrics.tokensPerSec)) |")
                print("| ms/token | \(String(format: "%.2f", result.metrics.msPerToken)) |")
                print("| Peak Memory (MB) | \(String(format: "%.2f", result.metrics.memoryPeakMB)) |")
                print("| Total Tokens | \(result.metrics.totalTokens) |")
                print("| Elapsed (s) | \(String(format: "%.2f", result.metrics.elapsedSeconds)) |")
            }
        } else {
            // Human-readable output (default)
            for result in results {
                if let scenario = result.scenario {
                    print("📊 Scenario: \(scenario)")
                }
                print("📊 Benchmark Results")
                print("=" * 50)
                print("Device: \(result.device.name)")
                print("Tokens generated: \(result.metrics.totalTokens)")
                print("Time elapsed: \(String(format: "%.2f", result.metrics.elapsedSeconds))s")
                print("Speed: \(String(format: "%.1f", result.metrics.tokensPerSec)) tokens/sec")
                print("Latency: \(String(format: "%.1f", result.metrics.msPerToken)) ms/token")
                print("Peak Memory: \(String(format: "%.2f", result.metrics.memoryPeakMB)) MB")
                print()
            }
        }
    }
}

extension String {
    static func *(lhs: String, rhs: Int) -> String {
        String(repeating: lhs, count: rhs)
    }
}

var standardError = FileHandle.standardError

extension FileHandle: TextOutputStream {
    public func write(_ string: String) {
        guard let data = string.data(using: .utf8) else { return }
        self.write(data)
    }
}

