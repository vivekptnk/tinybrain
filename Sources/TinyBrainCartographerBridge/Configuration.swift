// Configuration for TinyBrainSmartService.
//
// Encodes the per-call token budgets in INTEGRATION-TINYBRAIN.md §4.1. All
// budgets are exposed as public so tests can lower them, but production
// consumers should stick to `.default`.

import Foundation

extension TinyBrainSmartService {

    /// Tuning knobs for `TinyBrainSmartService`.
    ///
    /// Fields mirror the budgets named in the Cartographer integration
    /// contract. Construct `.default` unless a test is validating budget
    /// enforcement.
    public struct Configuration: Sendable {

        // MARK: - Token budgets (§4.1)

        /// Maximum number of input tokens `search` will accept before
        /// throwing `SmartAnnotationServiceError.invalidQuery`.
        public var searchInputBudget: Int

        /// Maximum number of output tokens `search` will generate. The IDs
        /// we return are small, so 256 is plenty for typical chunk sizes.
        public var searchOutputBudget: Int

        /// Maximum number of input tokens `summarize` will accept before
        /// throwing `SmartAnnotationServiceError.invalidQuery`.
        public var summarizeInputBudget: Int

        /// Maximum number of output tokens `summarize` will generate.
        public var summarizeOutputBudget: Int

        // MARK: - Memory pressure (§2.2)

        /// When `true`, the service subscribes to critical memory-pressure
        /// events and evicts the session on fire. Default `true`. Disable
        /// in tests that construct many services on a constrained host.
        public var enableMemoryPressureEviction: Bool

        // MARK: - Init

        public init(
            searchInputBudget: Int = 1536,
            searchOutputBudget: Int = 256,
            summarizeInputBudget: Int = 1536,
            summarizeOutputBudget: Int = 384,
            enableMemoryPressureEviction: Bool = true
        ) {
            self.searchInputBudget = searchInputBudget
            self.searchOutputBudget = searchOutputBudget
            self.summarizeInputBudget = summarizeInputBudget
            self.summarizeOutputBudget = summarizeOutputBudget
            self.enableMemoryPressureEviction = enableMemoryPressureEviction
        }

        /// Defaults that match INTEGRATION-TINYBRAIN.md §4.1 exactly.
        public static let `default` = Configuration()
    }
}
