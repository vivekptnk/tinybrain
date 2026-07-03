import XCTest
@testable import TinyBrainRAG

final class RAGPromptBuilderTests: XCTestCase {
    func testBuildIncludesNumberedPassagesAndCitationInstructions() throws {
        let tokenizer = try RAGTestSupport.tokenizer()
        let builder = RAGPromptBuilder(
            tokenizer: tokenizer,
            budget: PromptBudget(contextWindow: 1_200, generationHeadroom: 128)
        )
        let passages = [
            RAGTestSupport.passage("Hello world!", rank: 0),
            RAGTestSupport.passage("TinyBrain runs local inference.", rank: 1)
        ]

        let result = builder.build(question: "What runs locally?", passages: passages)

        XCTAssertEqual(result.included.count, 2)
        XCTAssertTrue(result.prompt.contains("[1] Hello world!"))
        XCTAssertTrue(result.prompt.contains("[2] TinyBrain runs local inference."))
        XCTAssertTrue(result.prompt.contains("cite every claim"))
        XCTAssertTrue(result.prompt.contains("What runs locally?"))
    }

    func testBuildReturnsIncludedPassagesInRankOrder() throws {
        let tokenizer = try RAGTestSupport.tokenizer()
        let builder = RAGPromptBuilder(
            tokenizer: tokenizer,
            budget: PromptBudget(contextWindow: 1_200, generationHeadroom: 128)
        )
        let passages = [
            RAGTestSupport.passage("third", rank: 2),
            RAGTestSupport.passage("first", rank: 0),
            RAGTestSupport.passage("second", rank: 1)
        ]

        let result = builder.build(question: "Order?", passages: passages)

        XCTAssertEqual(result.included.map(\.rank), [0, 1, 2])
        XCTAssertLessThan(
            result.prompt.range(of: "[1] first")!.lowerBound,
            result.prompt.range(of: "[2] second")!.lowerBound
        )
    }

    func testBuildDropsLowestRankedPassagesToFitBudget() throws {
        let tokenizer = try RAGTestSupport.tokenizer()
        let passages = [
            RAGTestSupport.passage("Hello world!", rank: 0),
            RAGTestSupport.passage("TinyBrain TinyBrain TinyBrain.", rank: 1),
            RAGTestSupport.passage("café café café café café.", rank: 2)
        ]
        let roomy = RAGPromptBuilder(
            tokenizer: tokenizer,
            budget: PromptBudget(contextWindow: 2_000, generationHeadroom: 128)
        )
        let full = roomy.build(question: "What is relevant?", passages: passages)
        let tightBudget = PromptBudget(
            contextWindow: tokenizer.encode(full.prompt).count - 1 + 128,
            generationHeadroom: 128
        )
        let tight = RAGPromptBuilder(tokenizer: tokenizer, budget: tightBudget)

        let result = tight.build(question: "What is relevant?", passages: passages)

        XCTAssertEqual(result.included.map(\.rank), [0, 1])
        XCTAssertFalse(result.prompt.contains("[3] café café café café café."))
        XCTAssertLessThanOrEqual(
            tokenizer.encode(result.prompt).count,
            tightBudget.contextWindow - tightBudget.generationHeadroom
        )
    }

    func testBuildNeverTruncatesIncludedPassageText() throws {
        let tokenizer = try RAGTestSupport.tokenizer()
        let passage = RAGTestSupport.passage("TinyBrain keeps documents on device.", rank: 0)
        let roomy = RAGPromptBuilder(
            tokenizer: tokenizer,
            budget: PromptBudget(contextWindow: 1_200, generationHeadroom: 128)
        )

        let result = roomy.build(question: "Where are documents?", passages: [passage])

        XCTAssertEqual(result.included, [passage])
        XCTAssertTrue(result.prompt.contains(passage.chunk.text))
    }

    func testBuildDropsEveryPassageWhenBudgetTooSmall() throws {
        let tokenizer = try RAGTestSupport.tokenizer()
        let noPassageBuilder = RAGPromptBuilder(
            tokenizer: tokenizer,
            budget: PromptBudget(contextWindow: 1_200, generationHeadroom: 64)
        )
        let noPassage = noPassageBuilder.build(question: "What?", passages: [])
        let tinyBudget = PromptBudget(
            contextWindow: tokenizer.encode(noPassage.prompt).count + 64,
            generationHeadroom: 64
        )
        let builder = RAGPromptBuilder(tokenizer: tokenizer, budget: tinyBudget)

        let result = builder.build(
            question: "What?",
            passages: [RAGTestSupport.passage("Hello world!", rank: 0)]
        )

        XCTAssertEqual(result.included, [])
        XCTAssertFalse(result.prompt.contains("[1] Hello world!"))
    }

    func testBuildReservesGenerationHeadroom() throws {
        let tokenizer = try RAGTestSupport.tokenizer()
        let budget = PromptBudget(contextWindow: 900, generationHeadroom: 200)
        let builder = RAGPromptBuilder(tokenizer: tokenizer, budget: budget)

        let result = builder.build(
            question: "What runs locally?",
            passages: [
                RAGTestSupport.passage("TinyBrain runs local inference.", rank: 0),
                RAGTestSupport.passage("The index stores metadata.", rank: 1)
            ]
        )

        XCTAssertLessThanOrEqual(
            tokenizer.encode(result.prompt).count,
            budget.contextWindow - budget.generationHeadroom
        )
    }

    func testBuildNumberingMatchesIncludedPassages() throws {
        let tokenizer = try RAGTestSupport.tokenizer()
        let builder = RAGPromptBuilder(
            tokenizer: tokenizer,
            budget: PromptBudget(contextWindow: 1_200, generationHeadroom: 128)
        )
        let passages = [
            RAGTestSupport.passage("alpha", rank: 0),
            RAGTestSupport.passage("beta", rank: 1)
        ]

        let result = builder.build(question: "Letters?", passages: passages)

        XCTAssertEqual(result.included, passages)
        XCTAssertTrue(result.prompt.contains("[1] alpha"))
        XCTAssertTrue(result.prompt.contains("[2] beta"))
    }

    func testBuildIsDeterministic() throws {
        let tokenizer = try RAGTestSupport.tokenizer()
        let builder = RAGPromptBuilder(
            tokenizer: tokenizer,
            budget: PromptBudget(contextWindow: 1_200, generationHeadroom: 128)
        )
        let passages = [
            RAGTestSupport.passage("Hello world!", rank: 0),
            RAGTestSupport.passage("TinyBrain runs local inference.", rank: 1)
        ]

        let first = builder.build(question: "What?", passages: passages)
        let second = builder.build(question: "What?", passages: passages)

        XCTAssertEqual(first.prompt, second.prompt)
        XCTAssertEqual(first.included, second.included)
    }
}
