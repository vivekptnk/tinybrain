// Contract tests for TinyBrainSmartService.
//
// Mirrors Cartographer's `SmartAnnotationServiceTests` where the semantics
// transfer (e.g. empty query ⇒ invalidQuery), and adds bridge-specific
// behavior: token-budget enforcement, lazy prewarm, memory-pressure
// eviction, cancellation, and TBF header validation.

import Cartographer
import XCTest
@testable import TinyBrainCartographerBridge

final class TinyBrainSmartServiceTests: XCTestCase {

    let projectID = UUID()

    // MARK: - Fixtures

    private func annotation(
        title: String = "",
        body: String = "",
        type: AnnotationType = .pin,
        metadata: [String: String] = [:],
        updatedAt: Date = Date()
    ) -> Annotation {
        Annotation(
            type: type,
            coordinate: GeoCoordinate(latitude: 0, longitude: 0),
            title: title,
            body: body,
            metadata: metadata,
            updatedAt: updatedAt,
            projectID: projectID
        )
    }

    private func makeService(
        handler: @escaping @Sendable (String, Int) async throws -> String = { _, _ in "[]" },
        configuration: TinyBrainSmartService.Configuration = .default
    ) -> (TinyBrainSmartService, FakeInferenceBackend) {
        let backend = FakeInferenceBackend(handler: handler)
        var config = configuration
        config.enableMemoryPressureEviction = false
        let service = TinyBrainSmartService(backend: backend, configuration: config)
        return (service, backend)
    }

    // MARK: - search: parity with mock contract

    func testSearchReturnsIndicesInModelOrder() async throws {
        let a = annotation(title: "Coffee at Blue Bottle")
        let b = annotation(title: "Best coffee in town")
        let c = annotation(title: "Tea house")
        let (service, _) = makeService(handler: { _, _ in "[1, 0]" })

        let result = try await service.search(query: "coffee", in: [a, b, c])
        XCTAssertEqual(result, [b.id, a.id])
    }

    func testSearchEmptyQueryThrowsInvalidQuery() async {
        let (service, _) = makeService()
        do {
            _ = try await service.search(query: "   ", in: [])
            XCTFail("expected invalidQuery")
        } catch SmartAnnotationServiceError.invalidQuery {
            // expected
        } catch {
            XCTFail("unexpected error: \(error)")
        }
    }

    func testSearchEmptyCorpusReturnsEmptyWithoutCallingBackend() async throws {
        let (service, backend) = makeService()
        let result = try await service.search(query: "coffee", in: [])
        XCTAssertTrue(result.isEmpty)
        XCTAssertTrue(backend.calls.isEmpty)
    }

    func testSearchDedupesAndClampsToCorpus() async throws {
        let a = annotation(title: "A")
        let b = annotation(title: "B")
        // Model returns duplicates and out-of-range indices.
        let (service, _) = makeService(handler: { _, _ in "[0, 0, 5, 1]" })
        let result = try await service.search(query: "x", in: [a, b])
        XCTAssertEqual(result, [a.id, b.id])
    }

    func testSearchToleratesChatteredOutput() async throws {
        let a = annotation(title: "A")
        let b = annotation(title: "B")
        let (service, _) = makeService(handler: { _, _ in
            "Sure! The relevant indices are [1, 0]. Hope that helps."
        })
        let result = try await service.search(query: "x", in: [a, b])
        XCTAssertEqual(result, [b.id, a.id])
    }

    // MARK: - summarize: parity

    func testSummarizeReturnsTrimmedModelOutput() async throws {
        let pins = (0..<3).map { annotation(title: "Pin \($0)") }
        let (service, _) = makeService(handler: { _, _ in
            "  Three pins near the visitor center, all recently updated.  "
        })
        let result = try await service.summarize(annotations: pins)
        XCTAssertEqual(result, "Three pins near the visitor center, all recently updated.")
    }

    func testSummarizeEmptyCorpusReturnsEmptyWithoutCallingBackend() async throws {
        let (service, backend) = makeService()
        let result = try await service.summarize(annotations: [])
        XCTAssertEqual(result, "")
        XCTAssertTrue(backend.calls.isEmpty)
    }

    // MARK: - Token budget enforcement (§4.1)

    func testSearchRejectsPromptOverInputBudget() async {
        var config = TinyBrainSmartService.Configuration.default
        config.searchInputBudget = 10
        config.enableMemoryPressureEviction = false
        let backend = FakeInferenceBackend(handler: { _, _ in "[]" })
        let service = TinyBrainSmartService(backend: backend, configuration: config)
        let corpus = (0..<50).map { annotation(title: "Annotation \($0)") }

        do {
            _ = try await service.search(query: "needle", in: corpus)
            XCTFail("expected invalidQuery")
        } catch SmartAnnotationServiceError.invalidQuery(let message) {
            XCTAssertTrue(message.contains("input budget"))
        } catch {
            XCTFail("unexpected error: \(error)")
        }
        XCTAssertTrue(backend.calls.isEmpty)
    }

    func testSearchPassesOutputBudgetToBackend() async throws {
        let a = annotation(title: "A")
        let (service, backend) = makeService(handler: { _, _ in "[0]" })
        _ = try await service.search(query: "x", in: [a])
        XCTAssertEqual(backend.calls.first?.maxOutputTokens, 256)
    }

    func testSummarizePassesOutputBudgetToBackend() async throws {
        let a = annotation(title: "A")
        let (service, backend) = makeService(handler: { _, _ in "a summary" })
        _ = try await service.summarize(annotations: [a])
        XCTAssertEqual(backend.calls.first?.maxOutputTokens, 384)
    }

    func testConfigurationDefaultsMatchContract() {
        let defaults = TinyBrainSmartService.Configuration.default
        XCTAssertEqual(defaults.searchInputBudget, 1_536)
        XCTAssertEqual(defaults.searchOutputBudget, 256)
        XCTAssertEqual(defaults.summarizeInputBudget, 1_536)
        XCTAssertEqual(defaults.summarizeOutputBudget, 384)
    }

    // MARK: - Lazy prewarm (§2.2)

    func testFirstCallTriggersPrewarm() async throws {
        let a = annotation(title: "A")
        let (service, backend) = makeService(handler: { _, _ in "[0]" })
        XCTAssertEqual(backend.prewarmCount, 0)
        _ = try await service.search(query: "x", in: [a])
        XCTAssertEqual(backend.prewarmCount, 1)
    }

    func testSecondCallPrewarmsAgainButBackendIsIdempotent() async throws {
        let a = annotation(title: "A")
        let (service, backend) = makeService(handler: { _, _ in "[0]" })
        _ = try await service.search(query: "x", in: [a])
        _ = try await service.summarize(annotations: [a])
        // The actor calls prewarm before every call; the real backend
        // dedupes internally. We only assert the count is ≥ 2 here.
        XCTAssertGreaterThanOrEqual(backend.prewarmCount, 2)
    }

    // MARK: - Memory-pressure eviction (§2.2)

    func testEvictForMemoryPressureCausesModelUnavailable() async {
        let a = annotation(title: "A")
        let (service, _) = makeService(handler: { _, _ in "[0]" })
        await service.evictForMemoryPressure()
        do {
            _ = try await service.search(query: "x", in: [a])
            XCTFail("expected modelUnavailable")
        } catch SmartAnnotationServiceError.modelUnavailable {
            // expected
        } catch {
            XCTFail("unexpected error: \(error)")
        }
    }

    func testEvictionIsStickyAcrossSearchAndSummarize() async {
        let a = annotation(title: "A")
        let (service, _) = makeService(handler: { _, _ in "done" })
        await service.evictForMemoryPressure()
        var searchFailed = false
        do { _ = try await service.search(query: "x", in: [a]) }
        catch SmartAnnotationServiceError.modelUnavailable { searchFailed = true }
        catch { XCTFail("unexpected: \(error)") }
        var summarizeFailed = false
        do { _ = try await service.summarize(annotations: [a]) }
        catch SmartAnnotationServiceError.modelUnavailable { summarizeFailed = true }
        catch { XCTFail("unexpected: \(error)") }
        XCTAssertTrue(searchFailed && summarizeFailed)
    }

    // MARK: - Cancellation (§3)

    func testCancellationPropagatesAsCancellationError() async {
        let a = annotation(title: "A")
        let (service, _) = makeService(handler: { _, _ in
            try await Task.sleep(nanoseconds: 2_000_000_000)
            return "[0]"
        })

        let task = Task {
            try await service.search(query: "x", in: [a])
        }
        // Cancel immediately.
        task.cancel()
        do {
            _ = try await task.value
            XCTFail("expected cancellation")
        } catch is CancellationError {
            // expected
        } catch {
            XCTFail("unexpected error: \(error)")
        }
    }

    // MARK: - Backend failure surfaces as inferenceFailed

    func testBackendThrowSurfacesAsInferenceFailed() async {
        struct BoomError: Error {}
        let a = annotation(title: "A")
        let (service, _) = makeService(handler: { _, _ in throw BoomError() })
        do {
            _ = try await service.search(query: "x", in: [a])
            XCTFail("expected inferenceFailed")
        } catch SmartAnnotationServiceError.inferenceFailed {
            // expected
        } catch {
            XCTFail("unexpected error: \(error)")
        }
    }

    // MARK: - estimatedTokenCount is nonisolated

    func testEstimatedTokenCountIsNonisolatedAndUsable() {
        let (service, _) = makeService()
        // Call from a sync context — would not compile if the method
        // required actor isolation.
        let count = service.estimatedTokenCount("hello world")
        XCTAssertEqual(count, 2)
    }

    // MARK: - TBF header validation

    func testInitRejectsMissingFile() throws {
        let url = URL(fileURLWithPath: "/tmp/tinybrain-bridge-tests-does-not-exist.tbf")
        XCTAssertThrowsError(try TinyBrainSmartService(modelURL: url)) { error in
            guard case SmartAnnotationServiceError.modelUnavailable = error else {
                XCTFail("expected modelUnavailable, got \(error)")
                return
            }
        }
    }

    func testInitRejectsNonTBFMagic() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("bridge-test-\(UUID()).tbf")
        try Data("JUNK_not_a_tbf_at_all".utf8).write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }

        XCTAssertThrowsError(try TinyBrainSmartService(modelURL: url)) { error in
            guard case SmartAnnotationServiceError.modelUnavailable(let message) = error else {
                XCTFail("expected modelUnavailable, got \(error)")
                return
            }
            XCTAssertTrue(message.contains("TBF"))
        }
    }

    func testInitAcceptsTBFMagic() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("bridge-test-\(UUID()).tbf")
        var header = Data("TBFM".utf8)
        header.append(Data(count: 4_096)) // padding to look plausible
        try header.write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }

        XCTAssertNoThrow(try TinyBrainSmartService(modelURL: url,
                                                   configuration: .init(enableMemoryPressureEviction: false)))
    }
}
