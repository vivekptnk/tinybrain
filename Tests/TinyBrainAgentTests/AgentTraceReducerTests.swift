import XCTest
@testable import TinyBrainAgent

final class AgentTraceReducerTests: XCTestCase {
    func testReducerMapsScriptedPlanActObserveBudgetFinalSequence() {
        var reducer = AgentTraceReducer()
        let start = Date(timeIntervalSince1970: 10)

        var snapshot = reducer.reset(maxSteps: 3, now: start)
        XCTAssertTrue(snapshot.isRunning)
        XCTAssertEqual(snapshot.maxSteps, 3)

        snapshot = reducer.reduce(
            .stepStarted(index: 0, promptTokens: 128),
            now: start.addingTimeInterval(0.1)
        )
        XCTAssertEqual(snapshot.steps.map(\.state), [.planning])
        XCTAssertEqual(snapshot.activeStepIndex, 0)

        snapshot = reducer.reduce(
            .toolCallProposed(
                index: 0,
                toolName: "retrieve",
                argumentsJSON: #"{"k":3,"query":"Project Atlas timing"}"#,
                query: "Project Atlas timing",
                k: 3,
                rawOutput: #"{"name":"retrieve","arguments":{"query":"Project Atlas timing","k":3}}"#
            ),
            now: start.addingTimeInterval(0.2)
        )
        XCTAssertEqual(snapshot.steps[0].state, .calling)
        XCTAssertEqual(snapshot.steps[0].toolName, "retrieve")
        XCTAssertEqual(snapshot.steps[0].liftedQuery, "Project Atlas timing")
        XCTAssertEqual(snapshot.steps[0].liftedK, 3)

        let result = """
        [1] Project Atlas review lock is August 14, 2026, and the owner is Mira Chen. (source: ops/project-atlas.md, distance: 0.124)
        [2] RAG retrieval returns ranked passages with lower-is-better distances. (source: tinybrain/rag-retrieval.md, distance: 0.642)
        """
        snapshot = reducer.reduce(
            .toolExecuted(
                index: 0,
                toolName: "retrieve",
                resultContent: result,
                isError: false,
                elapsedMs: 18.4,
                resultTokens: 42
            ),
            now: start.addingTimeInterval(0.4)
        )
        XCTAssertEqual(snapshot.steps[0].state, .observed)
        XCTAssertEqual(snapshot.steps[0].passages.count, 2)
        XCTAssertEqual(snapshot.steps[0].passages[0].source, "ops/project-atlas.md")
        XCTAssertEqual(snapshot.runMetrics.promptTokens, 128)
        XCTAssertEqual(snapshot.runMetrics.resultTokens, 42)

        snapshot = reducer.reduce(
            .budgetExhausted(maxSteps: 3),
            now: start.addingTimeInterval(0.5)
        )
        XCTAssertTrue(snapshot.isBudgetExhausted)
        XCTAssertTrue(snapshot.isRunning)
        XCTAssertEqual(snapshot.terminationReason, "budget exhausted")

        snapshot = reducer.reduce(
            .finalAnswer(
                answer: "Project Atlas locks on August 14, 2026, with Mira Chen as owner.",
                terminationReason: "budget exhausted",
                completedStepCount: 1
            ),
            now: start.addingTimeInterval(0.7)
        )
        XCTAssertFalse(snapshot.isRunning)
        XCTAssertEqual(snapshot.finalAnswer, "Project Atlas locks on August 14, 2026, with Mira Chen as owner.")
        XCTAssertEqual(snapshot.steps.map(\.state), [.done])
        XCTAssertEqual(snapshot.runMetrics.steps, 1)
    }
}
