// SmartServiceBenchmark — CHA-164
//
// Latency + footprint measurement harness for TinyBrainSmartService.
// Two execution paths:
//
//   1. Instrumented fake (always runs): validates harness plumbing and
//      service-layer overhead using FakeInferenceBackend with simulated
//      per-call delay.
//
//   2. Real-model path (skipped in CI): activated when the environment
//      variable TINYBRAIN_BENCH_MODEL points to a TinyLlama INT4 .tbf.
//      Records cold-start, per-call p50/p95, peak resident memory, disk
//      footprint, and pressure-eviction correctness. Numbers are printed
//      in a format that can be pasted directly into CHA-153.
//
// Run on device:
//   TINYBRAIN_BENCH_MODEL=/path/to/tinyllama-1.1b-int4.tbf \
//     swift test --filter SmartServiceBenchmark

import Darwin.Mach
import XCTest
import Cartographer
@testable import TinyBrainCartographerBridge

final class SmartServiceBenchmark: XCTestCase {

    // MARK: - Fixtures

    private let projectID = UUID()

    private func annotation(_ i: Int) -> Annotation {
        Annotation(
            type: .pin,
            coordinate: GeoCoordinate(latitude: Double(i) * 0.001, longitude: 0),
            title: "Location \(i): \(["café","park","trail","viewpoint","museum"][i % 5])",
            body: "Detailed description for annotation \(i). Tags: outdoor, nearby.",
            metadata: [:],
            updatedAt: Date(),
            projectID: projectID
        )
    }

    private func corpus(count: Int = 50) -> [Annotation] {
        (0..<count).map { annotation($0) }
    }

    private func makeServiceWithDelay(
        _ delayNs: UInt64 = 0
    ) -> (TinyBrainSmartService, FakeInferenceBackend) {
        let backend = FakeInferenceBackend { _, _ in
            if delayNs > 0 {
                try await Task.sleep(nanoseconds: delayNs)
            }
            return "[0, 1, 2]"
        }
        var cfg = TinyBrainSmartService.Configuration.default
        cfg.enableMemoryPressureEviction = false
        return (TinyBrainSmartService(backend: backend, configuration: cfg), backend)
    }

    // MARK: - Resident memory helper

    private static func physicalFootprintMB() -> Double {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size
        )
        let kr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        guard kr == KERN_SUCCESS else { return 0 }
        return Double(info.phys_footprint) / 1_048_576
    }

    // MARK: - ① Harness smoke (fake backend, no model required)

    func testHarnessColdsStartWithFakeBackend() async throws {
        let (service, backend) = makeServiceWithDelay()
        let c = corpus()

        let t0 = ContinuousClock.now
        _ = try await service.search(query: "coffee", in: c)
        let elapsed = ContinuousClock.now - t0

        XCTAssertEqual(backend.prewarmCount, 1, "prewarm must fire exactly once on first call")
        let ms = elapsed.asMilliseconds
        print("harness cold-start (fake): \(String(format: "%.2f", ms)) ms")
    }

    func testHarnessPerCallLatencyWithFakeBackend() async throws {
        // 5 ms simulated model latency per call — validates p50/p95 plumbing.
        let (service, _) = makeServiceWithDelay(5_000_000)
        let c = corpus()

        // prewarm
        _ = try await service.search(query: "warmup", in: c)

        var latencies = [Double]()
        for _ in 0..<20 {
            let t0 = ContinuousClock.now
            _ = try await service.search(query: "trail near water", in: c)
            latencies.append((ContinuousClock.now - t0).asMilliseconds)
        }

        let (p50, p95) = percentiles(latencies)
        print("harness search p50 (fake+5ms): \(String(format: "%.2f", p50)) ms")
        print("harness search p95 (fake+5ms): \(String(format: "%.2f", p95)) ms")

        // Service overhead alone must stay below 50 ms on any machine.
        XCTAssertLessThan(p95 - 5, 50, "service layer overhead p95 exceeds 50 ms")
    }

    func testHarnessMemoryPressureEviction() async throws {
        let (service, _) = makeServiceWithDelay()
        let c = corpus()

        _ = try await service.search(query: "park", in: c)

        await service.evictForMemoryPressure()

        do {
            _ = try await service.search(query: "coffee", in: c)
            XCTFail("expected modelUnavailable after eviction")
        } catch SmartAnnotationServiceError.modelUnavailable {
            // expected
        }
    }

    // MARK: - ② Real-model path (requires TINYBRAIN_BENCH_MODEL env var)

    /// Full measurement pass: cold-start, per-call p50/p95, disk footprint,
    /// resident memory, and eviction. Results printed in CHA-164 table format.
    func testRealModelMeasurementPass() async throws {
        guard let modelPath = ProcessInfo.processInfo.environment["TINYBRAIN_BENCH_MODEL"] else {
            throw XCTSkip("Set TINYBRAIN_BENCH_MODEL=/path/to/model.tbf to run real-hardware pass")
        }
        let modelURL = URL(fileURLWithPath: modelPath)
        guard FileManager.default.fileExists(atPath: modelPath) else {
            throw XCTSkip("Model file not found: \(modelPath)")
        }

        // ── Disk footprint ──────────────────────────────────────────────
        let attrs = try FileManager.default.attributesOfItem(atPath: modelPath)
        let diskMB = (attrs[.size] as? Int).map { Double($0) / 1_048_576 } ?? 0
        let diskGB = diskMB / 1024

        // ── Cold-start ──────────────────────────────────────────────────
        var cfg = TinyBrainSmartService.Configuration.default
        cfg.enableMemoryPressureEviction = false
        let service = try TinyBrainSmartService(modelURL: modelURL, configuration: cfg)
        let c = corpus()

        let baselineMemMB = Self.physicalFootprintMB()
        let coldT0 = ContinuousClock.now
        _ = try await service.search(query: "coffee", in: c)
        let coldStartMs = (ContinuousClock.now - coldT0).asMilliseconds
        let postWarmMemMB = Self.physicalFootprintMB()

        // ── Per-call latency (20 iterations) ───────────────────────────
        var latencies = [Double]()
        var peakMemMB = postWarmMemMB
        for _ in 0..<20 {
            let t0 = ContinuousClock.now
            _ = try await service.search(query: "trail near water", in: c)
            latencies.append((ContinuousClock.now - t0).asMilliseconds)
            peakMemMB = max(peakMemMB, Self.physicalFootprintMB())
        }

        let (p50, p95) = percentiles(latencies)
        let residentMB = peakMemMB - baselineMemMB

        // ── Eviction ────────────────────────────────────────────────────
        await service.evictForMemoryPressure()
        var evictionWorks = false
        do {
            _ = try await service.search(query: "post-evict", in: c)
        } catch SmartAnnotationServiceError.modelUnavailable {
            evictionWorks = true
        }

        // ── Budget checks ───────────────────────────────────────────────
        let a14ColdBudgetMs = 1500.0  // A14 is the looser bound; assertion uses it
        let callBudgetMs    = 400.0
        let memBudgetMB     = 400.0
        let diskBudgetGB    = 1.0

        // ── Print table for CHA-153 ─────────────────────────────────────
        print("""

        ┌─────────────────────────────────────────────────────────┐
        │  CHA-164 Measurement Results                            │
        ├──────────────────────────┬──────────────┬───────────────┤
        │  Metric                  │  Measured    │  Budget       │
        ├──────────────────────────┼──────────────┼───────────────┤
        │  Model disk              │  \(fmt(diskGB, "%.3f")) GB   │  ≤ 1.000 GB   │
        │  Peak resident memory    │  \(fmt(residentMB, "%.1f")) MB  │  ≤ 400 MB     │
        │  Cold-start (this chip)  │  \(fmt(coldStartMs, "%.0f")) ms  │  ≤1500/800 ms │
        │  Search p50 (50-ann)     │  \(fmt(p50, "%.0f")) ms  │  ≤ 400 ms     │
        │  Search p95 (50-ann)     │  \(fmt(p95, "%.0f")) ms  │  ≤ 400 ms     │
        │  Eviction → unavailable  │  \(evictionWorks ? "✅ yes" : "❌ no")          │  required     │
        └──────────────────────────┴──────────────┴───────────────┘
        """)

        // ── Assertions ──────────────────────────────────────────────────
        XCTAssertLessThanOrEqual(diskGB, diskBudgetGB,
            "disk footprint \(fmt(diskGB, "%.3f")) GB exceeds \(diskBudgetGB) GB")
        XCTAssertLessThanOrEqual(residentMB, memBudgetMB,
            "resident memory \(fmt(residentMB, "%.1f")) MB exceeds \(memBudgetMB) MB")
        XCTAssertLessThanOrEqual(coldStartMs, a14ColdBudgetMs,
            "cold-start \(fmt(coldStartMs, "%.0f")) ms exceeds A14 budget \(a14ColdBudgetMs) ms")
        XCTAssertLessThanOrEqual(p50, callBudgetMs,
            "search p50 \(fmt(p50, "%.0f")) ms exceeds \(callBudgetMs) ms")
        XCTAssertLessThanOrEqual(p95, callBudgetMs,
            "search p95 \(fmt(p95, "%.0f")) ms exceeds \(callBudgetMs) ms")
        XCTAssertTrue(evictionWorks, "eviction must throw modelUnavailable")
    }

    // MARK: - Helpers

    private func percentiles(_ values: [Double]) -> (p50: Double, p95: Double) {
        let sorted = values.sorted()
        let p50 = sorted[values.count / 2]
        let p95 = sorted[Int(Double(values.count) * 0.95)]
        return (p50, p95)
    }

    private func fmt(_ value: Double, _ spec: String) -> String {
        String(format: spec, value)
    }
}

// MARK: - Duration convenience

private extension Duration {
    var asMilliseconds: Double {
        Double(components.seconds) * 1000
            + Double(components.attoseconds) / 1_000_000_000_000_000
    }
}
