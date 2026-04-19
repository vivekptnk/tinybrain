// MemoryPressureMonitor — thin wrapper around
// `DispatchSource.makeMemoryPressureSource(.critical)` used by
// `TinyBrainSmartService` to evict the model on critical pressure.
//
// The source is registered on a dedicated utility queue so firing it does
// not land on `@MainActor` or the caller's cooperative pool. The actor
// polls `didFire()` lazily on each call (see INTEGRATION-TINYBRAIN.md §2.2);
// this keeps the Sendable contract clean without needing to hop back into
// the actor from the handler.

import Foundation

#if canImport(Darwin)
import Darwin
#endif

/// Monitors `DispatchSource` critical memory-pressure events. Thread-safe.
/// Conforms to `Sendable` via an internal NSLock guarding the flag.
final class MemoryPressureMonitor: @unchecked Sendable {

    private let source: DispatchSourceMemoryPressure
    private let lock = NSLock()
    private var fired: Bool = false

    init() {
        let queue = DispatchQueue(
            label: "com.tinybrain.cartographer-bridge.memory-pressure",
            qos: .utility
        )
        source = DispatchSource.makeMemoryPressureSource(
            eventMask: .critical,
            queue: queue
        )
        source.setEventHandler { [weak self] in
            guard let self else { return }
            self.lock.lock()
            self.fired = true
            self.lock.unlock()
        }
        source.activate()
    }

    deinit {
        source.cancel()
    }

    /// Returns `true` if a critical memory-pressure event has fired since
    /// the last call. Resets the flag.
    func didFire() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        let value = fired
        fired = false
        return value
    }
}
