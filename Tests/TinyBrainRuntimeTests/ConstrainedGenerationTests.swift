import Foundation
import XCTest
@testable import TinyBrainRuntime
import TinyBrainMetal
import TinyBrainTokenizer

final class ConstrainedGenerationTests: XCTestCase {
    func testGenerateStreamAppliesOutputSchemaConstraint() async throws {
        // WHAT: Strict schema constraints must beat the model's invalid raw argmax.
        // WHY: `GenerationConfig.outputSchema` is user-facing policy, not metadata.
        // HOW: A zero-layer toy model always prefers token 0 (`not-json`), yet
        // constrained decoding must emit a schema-valid JSON object.
        let pieces = ["not-json", "{", "\"answer\"", ":", "\"yes\"", "}"]
        let lookup = ArrayTokenLookup(pieces)
        let runner = ModelRunner(weights: biasedToyWeights(vocabSize: pieces.count))
        let schema = answerSchema
        let config = GenerationConfig(
            maxTokens: 16,
            sampler: SamplerConfig(temperature: 0.0, topK: 1),
            outputSchema: schema,
            constraintMode: .strict
        )

        let outputs = try await collectOutputs(
            runner: runner,
            config: config,
            tokenLookup: lookup
        )
        let text = outputs.map { pieces[$0.tokenId] }.joined()

        XCTAssertEqual(outputs.first?.tokenId, 1, "Strict masking should reject the invalid raw argmax")
        XCTAssertEqual(text, "{\"answer\":\"yes\"}")
        try assertJSON(text, validatesAgainst: schema)
        XCTAssertTrue(outputs.allSatisfy { $0.constraintState != nil })
    }

    func testStrictSchemaConstraintIsHermeticAcrossSeedsAndTemperatures() async throws {
        // WHAT: Strict mode should be deterministic when the schema leaves only
        // one valid token at each state.
        // WHY: Sampling randomness must not leak invalid tokens through the mask.
        // HOW: Run several seeds and temperatures against the same invalid-argmax
        // model and validate every completed JSON object.
        let pieces = ["not-json", "{", "\"answer\"", ":", "\"yes\"", "}"]
        let lookup = ArrayTokenLookup(pieces)

        for temperature in [Float(0.0), 0.2, 1.0, 2.0] {
            for seed in UInt64(1)...UInt64(8) {
                let config = GenerationConfig(
                    maxTokens: 16,
                    sampler: SamplerConfig(temperature: temperature, seed: seed),
                    outputSchema: answerSchema,
                    constraintMode: .strict
                )
                let outputs = try await collectOutputs(
                    runner: ModelRunner(weights: biasedToyWeights(vocabSize: pieces.count)),
                    config: config,
                    tokenLookup: lookup
                )
                let text = outputs.map { pieces[$0.tokenId] }.joined()
                XCTAssertEqual(text, "{\"answer\":\"yes\"}", "temperature=\(temperature) seed=\(seed)")
                try assertJSON(text, validatesAgainst: answerSchema)
            }
        }
    }

    func testConstrainedGenerationStopsWhenSchemaComplete() async throws {
        // WHAT: Once the schema is complete, generation stops even if maxTokens
        // leaves room for unconstrained text.
        // WHY: Structured output must not be followed by free-form tail text.
        // HOW: The toy model still prefers `not-json`, but the stream should end
        // exactly on the object close.
        let pieces = ["not-json", "{", "\"answer\"", ":", "\"yes\"", "}", "tail"]
        let lookup = ArrayTokenLookup(pieces)
        let config = GenerationConfig(
            maxTokens: 64,
            sampler: SamplerConfig(temperature: 0.0, topK: 1),
            outputSchema: answerSchema,
            constraintMode: .strict
        )

        let outputs = try await collectOutputs(
            runner: ModelRunner(weights: biasedToyWeights(vocabSize: pieces.count)),
            config: config,
            tokenLookup: lookup
        )
        let text = outputs.map { pieces[$0.tokenId] }.joined()

        XCTAssertEqual(outputs.map(\.tokenId), [1, 2, 3, 4, 5])
        XCTAssertEqual(text, "{\"answer\":\"yes\"}")
    }

    func testConstrainedGenerationRequiresTokenLookup() async throws {
        // WHAT: Strict schema generation without token text lookup fails loudly.
        // WHY: Falling back to unconstrained output would silently violate policy.
        // HOW: Consume the stream and assert the explicit generation error.
        let config = GenerationConfig(
            maxTokens: 4,
            sampler: SamplerConfig(temperature: 0.0, topK: 1),
            outputSchema: answerSchema,
            constraintMode: .strict
        )

        do {
            for try await _ in ModelRunner(weights: biasedToyWeights(vocabSize: 4))
                .generateStream(prompt: [], config: config) {}
            XCTFail("Expected constrained generation to require a token lookup")
        } catch let error as GenerationError {
            XCTAssertEqual(error, .constrainedGenerationRequiresTokenLookup)
        } catch {
            XCTFail("Unexpected error: \(error)")
        }
    }

    func testConstrainedGenerationThrowsWhenMaxTokensCutsOffIncompleteJSON() async throws {
        // WHAT: A maxTokens cutoff in the middle of a constrained object is an error.
        // WHY: Returning a truncated prefix such as `{"a":` would masquerade as success.
        // HOW: The stream has enough budget to reach the string value, but not finish it.
        let pieces = ["not-json", "{", "\"a\"", ":", "\"unfinished"]
        let lookup = ArrayTokenLookup(pieces)
        let schema: JSONSchema = .object(properties: [
            JSONSchemaProperty(name: "a", schema: .string, required: true)
        ], required: ["a"])
        let config = GenerationConfig(
            maxTokens: 3,
            sampler: SamplerConfig(temperature: 0.0, topK: 1),
            outputSchema: schema,
            constraintMode: .strict
        )

        do {
            _ = try await collectOutputs(
                runner: ModelRunner(weights: biasedToyWeights(vocabSize: pieces.count)),
                config: config,
                tokenLookup: lookup
            )
            XCTFail("Expected constrained generation to throw before returning truncated JSON")
        } catch let error as GenerationError {
            XCTAssertEqual(error, .constrainedGenerationIncomplete)
        } catch {
            XCTFail("Unexpected error: \(error)")
        }
    }

    func testUnconstrainedGenerationSequenceUnchanged() async throws {
        // WHAT: Without an effective schema, the generation path must remain byte-identical.
        // WHY: F13 may not add token lookup, logit copies, or behavior changes to default generation.
        // HOW: A zero-layer model with a fixed invalid argmax emits the same ids with
        // and without a lookup object because the lookup is ignored when unconstrained.
        let pieces = ["not-json", "{", "\"answer\"", ":", "\"yes\"", "}"]
        let lookup = CountingTokenLookup(pieces)
        let config = GenerationConfig(
            maxTokens: 4,
            sampler: SamplerConfig(temperature: 0.0, topK: 1)
        )

        let plain = try await collectOutputs(
            runner: ModelRunner(weights: biasedToyWeights(vocabSize: pieces.count)),
            config: config,
            tokenLookup: nil
        ).map(\.tokenId)
        let withLookup = try await collectOutputs(
            runner: ModelRunner(weights: biasedToyWeights(vocabSize: pieces.count)),
            config: config,
            tokenLookup: lookup
        ).map(\.tokenId)

        XCTAssertEqual(plain, [0, 0, 0, 0])
        XCTAssertEqual(withLookup, plain)
        XCTAssertEqual(lookup.decodeCount, 0, "Unconstrained generation must not touch TokenLookup")
    }

    func testToolCallingConfigSynthesizesSingleToolSchema() async throws {
        // WHAT: A single required tool can synthesize the tool-call schema.
        // WHY: Tool-call generation should not require callers to duplicate a
        // schema that is already unambiguous in ToolCallingConfig.
        // HOW: No explicit outputSchema is set; strict decoding must still emit
        // a valid retrieve call and reject the invalid raw argmax.
        let tool = ToolDefinition(
            name: "retrieve",
            description: "Retrieve passages",
            parameters: .object(properties: [
                JSONSchemaProperty(name: "query", schema: .enum(values: ["yes"]), required: true)
            ], required: ["query"])
        )
        let pieces = [
            "not-json", "{", "\"name\"", ":", "\"retrieve\"", ",",
            "\"arguments\"", "{\"query\"", "\"query\"", "\"yes\"", "}", "\"other\""
        ]
        let lookup = ArrayTokenLookup(pieces)
        let config = GenerationConfig(
            maxTokens: 32,
            sampler: SamplerConfig(temperature: 0.0, topK: 1),
            constraintMode: .strict,
            toolCallingConfig: ToolCallingConfig(tools: [tool], toolChoice: .required)
        )

        let outputs = try await collectOutputs(
            runner: ModelRunner(weights: biasedToyWeights(vocabSize: pieces.count)),
            config: config,
            tokenLookup: lookup
        )
        let text = outputs.map { pieces[$0.tokenId] }.joined()
        var parser = ToolCallParser(tools: [tool])
        parser.feed(text)

        guard case .success(let call) = parser.extractToolCall() else {
            return XCTFail("Expected a valid retrieve tool call, got \(text)")
        }
        XCTAssertEqual(call.name, "retrieve")
        XCTAssertEqual(call.arguments["query"] as? String, "yes")
    }

    func testConstrainedGenerationDoesNotDecodeVocabularyPerGeneratedToken() async throws {
        // WHAT: Token strings are materialized once per constrained generation.
        // WHY: Large model vocabularies cannot afford O(vocab) token decodes per output token.
        // HOW: CountingTokenLookup would expose repeated table construction immediately.
        let pieces = ["not-json", "{", "\"answer\"", ":", "\"yes\"", "}"]
        let lookup = CountingTokenLookup(pieces)
        let config = GenerationConfig(
            maxTokens: 32,
            sampler: SamplerConfig(temperature: 0.0, topK: 1),
            outputSchema: answerSchema,
            constraintMode: .strict
        )

        let outputs = try await collectOutputs(
            runner: ModelRunner(weights: biasedToyWeights(vocabSize: pieces.count)),
            config: config,
            tokenLookup: lookup
        )

        XCTAssertEqual(outputs.count, 5)
        XCTAssertEqual(lookup.decodeCount, pieces.count)
        print("TINYBRAIN_CONSTRAINT_PERF generated_tokens=\(outputs.count) vocab=\(pieces.count) token_decode_calls=\(lookup.decodeCount)")
    }

    func testQwenConstrainedToolCallSmoke() async throws {
        guard ProcessInfo.processInfo.environment["TINYBRAIN_RUN_QWEN_SMOKE"] == "1" else {
            throw XCTSkip("Set TINYBRAIN_RUN_QWEN_SMOKE=1 to run constrained Qwen tool-call smoke")
        }

        let modelPath = "Models/qwen2.5-1.5b-int8.tbf"
        let tokenizerPath = "Models/qwen2.5-1.5b-raw/tokenizer.json"
        guard FileManager.default.fileExists(atPath: resolveProjectPath(modelPath)) else {
            throw XCTSkip("\(modelPath) not available")
        }
        guard FileManager.default.fileExists(atPath: resolveProjectPath(tokenizerPath)) else {
            throw XCTSkip("\(tokenizerPath) not available")
        }

        if MetalBackend.isAvailable {
            TinyBrainBackend.metalBackend = try? MetalBackend()
        }

        let requiredOnlyTool = ToolDefinition(
            name: "retrieve",
            description: "Retrieve local passages for a search query.",
            parameters: .object(properties: [
                JSONSchemaProperty(name: "query", schema: .string, required: true)
            ], required: ["query"])
        )
        let optionalKTool = ToolDefinition(
            name: "retrieve",
            description: "Retrieve local passages for a search query.",
            parameters: .object(properties: [
                JSONSchemaProperty(name: "query", schema: .string, required: true),
                JSONSchemaProperty(name: "k", schema: .integer, required: false)
            ], required: ["query"])
        )
        let weights = try ModelLoader.load(from: modelPath)
        let tokenizer = try TokenizerLoader.loadHuggingFace(from: resolveProjectPath(tokenizerPath))
        let lookup = TokenizerLookup(tokenizer: tokenizer)

        try await runQwenToolSmoke(
            weights: weights,
            tokenizer: tokenizer,
            lookup: lookup,
            tool: requiredOnlyTool,
            userRequest: "Find passages about TinyBrain constrained decoding.",
            requireOptionalK: false
        )
        try await runQwenToolSmoke(
            weights: weights,
            tokenizer: tokenizer,
            lookup: lookup,
            tool: optionalKTool,
            userRequest: "Find 3 passages about TinyBrain constrained decoding. Include k as 3.",
            requireOptionalK: true
        )
    }

    private func runQwenToolSmoke(
        weights: ModelWeights,
        tokenizer: BPETokenizer,
        lookup: TokenizerLookup,
        tool: ToolDefinition,
        userRequest: String,
        requireOptionalK: Bool
    ) async throws {
        let toolConfig = ToolCallingConfig(tools: [tool], toolChoice: .required)
        let prompt = """
        <|im_start|>system
        You are a tool-calling assistant. \(toolConfig.buildSystemPrompt())
        Return only the JSON tool call.<|im_end|>
        <|im_start|>user
        \(userRequest)<|im_end|>
        <|im_start|>assistant

        """
        let promptIds = tokenizer.encode(prompt)
        let config = GenerationConfig(
            maxTokens: 128,
            sampler: SamplerConfig(temperature: 0.0, topK: 1),
            constraintMode: .strict,
            toolCallingConfig: toolConfig
        )
        var generated: [Int] = []
        for try await output in ModelRunner(weights: weights).generateStream(
            prompt: promptIds,
            config: config,
            tokenLookup: lookup
        ) {
            generated.append(output.tokenId)
        }
        let text = tokenizer.decode(generated)
        print("TINYBRAIN_QWEN_CONSTRAINED_OUTPUT optional_k=\(requireOptionalK) \(text)")

        try StrictJSONOracle.validate(text)
        var parser = ToolCallParser(tools: [tool])
        parser.feed(text)
        guard case .success(let call) = parser.extractToolCall() else {
            return XCTFail("Qwen constrained output did not parse as a tool call: \(text)")
        }
        XCTAssertEqual(call.name, "retrieve")
        XCTAssertNotNil(call.arguments["query"] as? String)
        if requireOptionalK {
            XCTAssertNotNil(call.arguments["k"] as? NSNumber, "Optional k variant should exercise the optional argument path")
        }
    }

    private var answerSchema: JSONSchema {
        .object(properties: [
            JSONSchemaProperty(name: "answer", schema: .enum(values: ["yes"]), required: true)
        ], required: ["answer"])
    }

    private func collectOutputs(
        runner: ModelRunner,
        config: GenerationConfig,
        tokenLookup: (any TokenLookup)?
    ) async throws -> [TokenOutput] {
        var outputs: [TokenOutput] = []
        for try await output in runner.generateStream(
            prompt: [],
            config: config,
            tokenLookup: tokenLookup
        ) {
            outputs.append(output)
        }
        return outputs
    }

    private func assertJSON(_ text: String, validatesAgainst schema: JSONSchema) throws {
        try StrictJSONOracle.validate(text)
        let data = try XCTUnwrap(text.data(using: .utf8))
        let value = try JSONSerialization.jsonObject(with: data)
        switch SchemaValidator.validate(value, against: schema) {
        case .success:
            break
        case .failure(let error):
            XCTFail("Expected valid JSON for schema, got \(error): \(text)")
        }
    }

    private func biasedToyWeights(vocabSize: Int) -> ModelWeights {
        let config = ModelConfig(
            numLayers: 0,
            hiddenDim: 1,
            numHeads: 1,
            vocabSize: vocabSize,
            maxSeqLen: 64,
            numKVHeads: 1,
            intermediateDim: 4
        )
        var logits = [Float](repeating: 0, count: vocabSize)
        if vocabSize > 0 { logits[0] = 100 }
        let tokenEmbeddings = Tensor<Float>(
            shape: TensorShape(vocabSize, 1),
            data: [Float](repeating: 0, count: vocabSize)
        )
        let output = LinearLayerWeights(
            floatWeights: Tensor<Float>.zeros(shape: TensorShape(1, vocabSize)),
            bias: Tensor<Float>(shape: TensorShape(vocabSize), data: logits)
        )
        return ModelWeights(
            config: config,
            tokenEmbeddings: tokenEmbeddings,
            layers: [],
            output: output
        )
    }

    private func resolveProjectPath(_ path: String) -> String {
        if FileManager.default.fileExists(atPath: path) { return path }

        var dir = FileManager.default.currentDirectoryPath
        for _ in 0..<10 {
            let packagePath = (dir as NSString).appendingPathComponent("Package.swift")
            if FileManager.default.fileExists(atPath: packagePath) {
                let fullPath = (dir as NSString).appendingPathComponent(path)
                if FileManager.default.fileExists(atPath: fullPath) { return fullPath }
            }
            dir = (dir as NSString).deletingLastPathComponent
            if dir == "/" { break }
        }
        return path
    }
}

final class ConstrainedSamplerGranularityTests: XCTestCase {
    func testConstrainedSamplerConsumesMultiCharacterToken() {
        // WHAT: A single token can contain multiple JSON characters.
        // WHY: Prefix-only checks corrupt parser state when tokens like `{"answer"`
        // are accepted but only treated as `{`.
        // HOW: Advancing one compound token must land at the colon state.
        let schema: JSONSchema = .object(properties: [
            JSONSchemaProperty(name: "answer", schema: .enum(values: ["yes"]), required: true)
        ], required: ["answer"])
        var sampler = ConstrainedSampler(schema: schema, mode: .strict)
        let lookup = ArrayTokenLookup(["{\"answer\"", ":", "\"yes\"", "}"])
        var logits = Tensor<Float>(shape: TensorShape(4), data: [1, 1, 1, 1])

        sampler.maskLogits(&logits, tokenizer: lookup)
        XCTAssertEqual(logits.data[0], 1)

        sampler.advance(token: "{\"answer\"")
        XCTAssertTrue(sampler.isValidPrefix)
        XCTAssertEqual(sampler.stateDescription, "expecting ':'")
    }

    func testConstrainedSamplerAllowsSplitStringValue() {
        // WHAT: String values may arrive split across token boundaries.
        // WHY: Real tokenizers often produce fragments such as `"hel` then `lo"`.
        // HOW: The first fragment stays a valid prefix; the second closes the value.
        let schema: JSONSchema = .object(properties: [
            JSONSchemaProperty(name: "answer", schema: .string, required: true)
        ], required: ["answer"])
        var sampler = ConstrainedSampler(schema: schema, mode: .strict)

        sampler.advance(token: "{")
        sampler.advance(token: "\"answer\"")
        sampler.advance(token: ":")
        sampler.advance(token: "\"hel")
        XCTAssertTrue(sampler.isValidPrefix)
        XCTAssertEqual(sampler.stateDescription, "expecting value(string)")

        sampler.advance(token: "lo\"")
        XCTAssertEqual(sampler.stateDescription, "expecting ',' or '}'")
    }

    func testConstrainedSamplerAllowsPartialBooleanAndNullLiterals() {
        // WHAT: Literal values can span several tokens.
        // WHY: Strict masking must permit continuations after `t`, `tr`, or `nu`.
        // HOW: The state remains valid until the literal completes.
        var booleanSampler = ConstrainedSampler(schema: .boolean, mode: .strict)
        let booleanLookup = ArrayTokenLookup(["r", "x"])
        var booleanLogits = Tensor<Float>(shape: TensorShape(2), data: [1, 1])

        booleanSampler.advance(token: "t")
        booleanSampler.maskLogits(&booleanLogits, tokenizer: booleanLookup)
        XCTAssertEqual(booleanLogits.data[0], 1)
        XCTAssertEqual(booleanLogits.data[1], -Float.infinity)
        booleanSampler.advance(token: "r")
        booleanSampler.advance(token: "u")
        booleanSampler.advance(token: "e")
        XCTAssertTrue(booleanSampler.isComplete)

        var nullSampler = ConstrainedSampler(schema: .null, mode: .strict)
        nullSampler.advance(token: "n")
        nullSampler.advance(token: "u")
        XCTAssertTrue(nullSampler.isValidPrefix)
        nullSampler.advance(token: "l")
        nullSampler.advance(token: "l")
        XCTAssertTrue(nullSampler.isComplete)
    }

    func testRequiredObjectDoesNotAllowCloseBeforeRequiredField() {
        // WHAT: `}` is invalid while required object properties are missing.
        // WHY: Post-parse validation is too late; strict decoding must prevent
        // incomplete objects from being sampled.
        // HOW: After `{`, the close token is hard-masked until `x` is emitted.
        let schema: JSONSchema = .object(properties: [
            JSONSchemaProperty(name: "x", schema: .string, required: true)
        ], required: ["x"])
        var sampler = ConstrainedSampler(schema: schema, mode: .strict)
        let lookup = ArrayTokenLookup(["}", "\"x\""])
        var logits = Tensor<Float>(shape: TensorShape(2), data: [1, 1])

        sampler.advance(token: "{")
        sampler.maskLogits(&logits, tokenizer: lookup)

        XCTAssertEqual(logits.data[0], -Float.infinity)
        XCTAssertEqual(logits.data[1], 1)
    }

    func testObjectCommaRequiresAnotherKeyBeforeClose() {
        // WHAT: After an object comma, a key is mandatory before `}`.
        // WHY: RFC 8259 forbids trailing commas, even though JSONSerialization
        // accepts them on Darwin.
        // HOW: Once `{"a":"x",` is accepted, `}` must be hard-masked and only
        // the remaining optional key can continue the object.
        let schema: JSONSchema = .object(properties: [
            JSONSchemaProperty(name: "a", schema: .string, required: true),
            JSONSchemaProperty(name: "b", schema: .string, required: false)
        ], required: ["a"])
        var sampler = ConstrainedSampler(schema: schema, mode: .strict)
        let lookup = ArrayTokenLookup(["}", "\"b\""])
        var logits = Tensor<Float>(shape: TensorShape(2), data: [1, 1])

        sampler.advance(token: "{\"a\":\"x\",")
        sampler.maskLogits(&logits, tokenizer: lookup)

        XCTAssertEqual(logits.data[0], -Float.infinity)
        XCTAssertEqual(logits.data[1], 1)
    }

    func testObjectOptionalPropertyRolloutsRejectTrailingCommasAcrossSeeds() throws {
        // WHAT: Optional object properties may be skipped or emitted, but never
        // reached through a trailing comma.
        // WHY: The fuzz oracle must reject `{"a":"x",}` so grammar regressions
        // cannot hide behind JSONSerialization's permissive parser.
        // HOW: 300 seeded rollouts sample between close/comma after the required
        // field, then validate every result with the strict RFC oracle.
        let schema: JSONSchema = .object(properties: [
            JSONSchemaProperty(name: "a", schema: .string, required: true),
            JSONSchemaProperty(name: "b", schema: .string, required: false)
        ], required: ["a"])
        let pieces = ["{", "}", "\"a\"", "\"b\"", ":", ",", "\"x\""]
        var completed = 0

        for seed in UInt64(1)...UInt64(300) {
            let text = try constrainedRollout(
                schema: schema,
                pieces: pieces,
                seed: seed,
                maxTokens: 16
            )
            try StrictJSONOracle.validate(text)
            let data = try XCTUnwrap(text.data(using: .utf8))
            let value = try JSONSerialization.jsonObject(with: data)
            try SchemaValidator.validate(value, against: schema).get()
            completed += 1
        }

        print("TINYBRAIN_CONSTRAINT_FUZZ optional_object_rollouts=\(completed) invalid=0")
    }

    func testNumberRejectsFractionOrExponentUntilIntegerDigitExists() {
        // WHAT: After `-`, only integer digits can continue a JSON number.
        // WHY: Prefixes such as `-.` are impossible to complete as RFC 8259
        // numbers and caused max-token wedges in fuzzing.
        // HOW: The mask must reject `.` and `e`, while keeping digit tokens.
        let schema: JSONSchema = .object(properties: [
            JSONSchemaProperty(name: "n", schema: .number, required: true)
        ], required: ["n"])
        var sampler = ConstrainedSampler(schema: schema, mode: .strict)
        let lookup = ArrayTokenLookup([".", "e", "0", "5"])
        var logits = Tensor<Float>(shape: TensorShape(4), data: [1, 1, 1, 1])

        sampler.advance(token: "{\"n\":-")
        sampler.maskLogits(&logits, tokenizer: lookup)

        XCTAssertEqual(logits.data[0], -Float.infinity)
        XCTAssertEqual(logits.data[1], -Float.infinity)
        XCTAssertEqual(logits.data[2], 1)
        XCTAssertEqual(logits.data[3], 1)
    }

    func testNumberRolloutsCompleteAndRemainStrictJSONAcrossSeeds() throws {
        // WHAT: Number-constrained rollouts must either complete as valid JSON
        // numbers or be masked away before invalid prefixes are accepted.
        // WHY: The grammar previously accepted `-.513...` prefixes that could
        // never finish as JSON numbers.
        // HOW: 250 seeded rollouts include tempting fraction/exponent fragments
        // and validate every completed object with the strict oracle.
        let schema: JSONSchema = .object(properties: [
            JSONSchemaProperty(name: "n", schema: .number, required: true)
        ], required: ["n"])
        let pieces = ["{", "}", "\"n\"", ":", "-", ".", "5", "0", "e", "E", "+"]
        var completed = 0

        for seed in UInt64(1)...UInt64(250) {
            let text = try constrainedRollout(
                schema: schema,
                pieces: pieces,
                seed: seed,
                maxTokens: 64
            )
            try StrictJSONOracle.validate(text)
            let data = try XCTUnwrap(text.data(using: .utf8))
            let value = try JSONSerialization.jsonObject(with: data)
            try SchemaValidator.validate(value, against: schema).get()
            completed += 1
        }

        print("TINYBRAIN_CONSTRAINT_FUZZ number_rollouts=\(completed) invalid=0 wedged=0")
    }

    func testConstrainedSamplerReusesTokenTableAndStateMask() {
        // WHAT: Repeated masks for the same parser state should be cache hits.
        // WHY: Large vocabularies make full decode and grammar passes expensive.
        // HOW: The lookup decodes each token once, and the allowed-mask cache has
        // one miss despite several mask applications.
        let pieces = ["{", "}", "\"x\"", "noise"]
        let lookup = CountingTokenLookup(pieces)
        var sampler = ConstrainedSampler(
            schema: .object(properties: [
                JSONSchemaProperty(name: "x", schema: .string, required: true)
            ], required: ["x"]),
            mode: .strict
        )

        for _ in 0..<5 {
            var logits = Tensor<Float>(shape: TensorShape(pieces.count), data: [1, 1, 1, 1])
            sampler.maskLogits(&logits, tokenizer: lookup)
        }

        XCTAssertEqual(lookup.decodeCount, pieces.count)
        XCTAssertEqual(sampler.allowedMaskCacheMisses, 1)
        print("TINYBRAIN_CONSTRAINT_MASK_CACHE repeated_masks=5 vocab=\(pieces.count) token_decode_calls=\(lookup.decodeCount) cache_misses=\(sampler.allowedMaskCacheMisses)")
    }
}

private func constrainedRollout(
    schema: JSONSchema,
    pieces: [String],
    seed: UInt64,
    maxTokens: Int
) throws -> String {
    var sampler = ConstrainedSampler(schema: schema, mode: .strict)
    var samplerConfig = SamplerConfig(temperature: 1.0, seed: seed)
    var text = ""
    var history: [Int] = []

    for _ in 0..<maxTokens {
        var logits = Tensor<Float>(
            shape: TensorShape(pieces.count),
            data: [Float](repeating: 0, count: pieces.count)
        )
        sampler.maskLogits(&logits, tokenizer: ArrayTokenLookup(pieces))
        guard sampler.lastAllowedTokenCount.map({ $0 > 0 }) == true else {
            throw ConstrainedRolloutError.noValidTokens(text: text, state: sampler.stateDescription)
        }

        let sampled = Sampler.sampleDetailed(
            logits: logits,
            config: &samplerConfig,
            history: history
        )
        let token = pieces[sampled.tokenId]
        text += token
        sampler.advance(token: token)
        guard sampler.isValidPrefix else {
            throw ConstrainedRolloutError.invalidPrefix(text: text)
        }
        if sampler.isComplete {
            return text
        }
        history.append(sampled.tokenId)
    }

    throw ConstrainedRolloutError.incomplete(text: text, state: sampler.stateDescription)
}

private enum ConstrainedRolloutError: Error, CustomStringConvertible {
    case noValidTokens(text: String, state: String)
    case invalidPrefix(text: String)
    case incomplete(text: String, state: String)

    var description: String {
        switch self {
        case .noValidTokens(let text, let state):
            return "No valid constrained tokens from state \(state): \(text)"
        case .invalidPrefix(let text):
            return "Constrained rollout accepted invalid prefix: \(text)"
        case .incomplete(let text, let state):
            return "Constrained rollout did not complete from state \(state): \(text)"
        }
    }
}

private enum StrictJSONOracle {
    static func validate(_ text: String) throws {
        try rejectTrailingCommasAndInvalidNumbers(in: text)
    }

    private static func rejectTrailingCommasAndInvalidNumbers(in text: String) throws {
        let scalars = text.unicodeScalars
        var index = scalars.startIndex
        var inString = false
        var escaping = false
        var unicodeEscapeDigitsRemaining = 0
        var previousNonWhitespace: UnicodeScalar?

        while index < scalars.endIndex {
            let scalar = scalars[index]

            if inString {
                if unicodeEscapeDigitsRemaining > 0 {
                    unicodeEscapeDigitsRemaining -= 1
                } else if escaping {
                    if scalar == "u" {
                        unicodeEscapeDigitsRemaining = 4
                    }
                    escaping = false
                } else if scalar == "\\" {
                    escaping = true
                } else if scalar == "\"" {
                    inString = false
                }
                index = scalars.index(after: index)
                continue
            }

            if scalar == "\"" {
                inString = true
                previousNonWhitespace = scalar
                index = scalars.index(after: index)
                continue
            }

            if scalar == "}" || scalar == "]" {
                if previousNonWhitespace == "," {
                    throw StrictJSONError.trailingCommaBeforeClose(text: text)
                }
            }

            if scalar == "-" || scalar.isStrictJSONDigit {
                let start = index
                index = scalars.index(after: index)
                while index < scalars.endIndex, scalars[index].isStrictJSONNumberBody {
                    index = scalars.index(after: index)
                }
                let number = String(scalars[start..<index])
                guard isStrictJSONNumber(number) else {
                    throw StrictJSONError.invalidNumber(number: number, text: text)
                }
                previousNonWhitespace = "0"
                continue
            }

            if !scalar.isStrictJSONWhitespace {
                previousNonWhitespace = scalar
            }
            index = scalars.index(after: index)
        }
    }

    private static func isStrictJSONNumber(_ text: String) -> Bool {
        enum State {
            case start
            case afterMinus
            case zero
            case intDigits
            case afterDot
            case fracDigits
            case afterExp
            case afterExpSign
            case expDigits
        }

        var state = State.start
        for scalar in text.unicodeScalars {
            switch state {
            case .start:
                if scalar == "-" {
                    state = .afterMinus
                } else if scalar == "0" {
                    state = .zero
                } else if scalar.isStrictJSONNonZeroDigit {
                    state = .intDigits
                } else {
                    return false
                }
            case .afterMinus:
                if scalar == "0" {
                    state = .zero
                } else if scalar.isStrictJSONNonZeroDigit {
                    state = .intDigits
                } else {
                    return false
                }
            case .zero:
                if scalar == "." {
                    state = .afterDot
                } else if scalar == "e" || scalar == "E" {
                    state = .afterExp
                } else {
                    return false
                }
            case .intDigits:
                if scalar.isStrictJSONDigit {
                    state = .intDigits
                } else if scalar == "." {
                    state = .afterDot
                } else if scalar == "e" || scalar == "E" {
                    state = .afterExp
                } else {
                    return false
                }
            case .afterDot:
                if scalar.isStrictJSONDigit {
                    state = .fracDigits
                } else {
                    return false
                }
            case .fracDigits:
                if scalar.isStrictJSONDigit {
                    state = .fracDigits
                } else if scalar == "e" || scalar == "E" {
                    state = .afterExp
                } else {
                    return false
                }
            case .afterExp:
                if scalar == "+" || scalar == "-" {
                    state = .afterExpSign
                } else if scalar.isStrictJSONDigit {
                    state = .expDigits
                } else {
                    return false
                }
            case .afterExpSign:
                if scalar.isStrictJSONDigit {
                    state = .expDigits
                } else {
                    return false
                }
            case .expDigits:
                if scalar.isStrictJSONDigit {
                    state = .expDigits
                } else {
                    return false
                }
            }
        }

        switch state {
        case .zero, .intDigits, .fracDigits, .expDigits:
            return true
        case .start, .afterMinus, .afterDot, .afterExp, .afterExpSign:
            return false
        }
    }
}

private enum StrictJSONError: Error, CustomStringConvertible {
    case trailingCommaBeforeClose(text: String)
    case invalidNumber(number: String, text: String)

    var description: String {
        switch self {
        case .trailingCommaBeforeClose(let text):
            return "Strict JSON rejects trailing comma before close: \(text)"
        case .invalidNumber(let number, let text):
            return "Strict JSON rejects invalid number \(number): \(text)"
        }
    }
}

private extension UnicodeScalar {
    var isStrictJSONWhitespace: Bool {
        self == " " || self == "\n" || self == "\r" || self == "\t"
    }

    var isStrictJSONDigit: Bool {
        value >= 48 && value <= 57
    }

    var isStrictJSONNonZeroDigit: Bool {
        value >= 49 && value <= 57
    }

    var isStrictJSONNumberBody: Bool {
        isStrictJSONDigit || self == "." || self == "e" || self == "E" || self == "+" || self == "-"
    }
}

private struct ArrayTokenLookup: TokenLookup {
    private let pieces: [String]

    init(_ pieces: [String]) {
        self.pieces = pieces
    }

    var vocabularySize: Int { pieces.count }

    func decode(tokenId: Int) -> String {
        guard tokenId >= 0, tokenId < pieces.count else { return "" }
        return pieces[tokenId]
    }
}

private final class CountingTokenLookup: TokenLookup {
    private let pieces: [String]
    private(set) var decodeCount = 0

    init(_ pieces: [String]) {
        self.pieces = pieces
    }

    var vocabularySize: Int { pieces.count }

    func decode(tokenId: Int) -> String {
        decodeCount += 1
        guard tokenId >= 0, tokenId < pieces.count else { return "" }
        return pieces[tokenId]
    }
}

private struct TokenizerLookup: TokenLookup {
    let tokenizer: BPETokenizer

    var vocabularySize: Int { tokenizer.vocabularySize }

    func decode(tokenId: Int) -> String {
        tokenizer.decode([tokenId])
    }
}
