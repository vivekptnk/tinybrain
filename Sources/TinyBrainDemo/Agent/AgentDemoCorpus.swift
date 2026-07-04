/// Bundled notes used by the in-app TinyBrain Agent demo.
///
/// The corpus is intentionally small, local, and varied so retrieval behavior is
/// visible without a document picker or network access.

import Foundation
import TinyBrainRAG

/// A concise on-device note available to the demo agent.
public struct AgentDemoNote: Identifiable, Equatable, Sendable {
    /// Stable note identifier.
    public let id: String

    /// Display title used in source labels.
    public let title: String

    /// Source path stored in the RAG index.
    public let sourcePath: String

    /// Note body indexed for retrieval.
    public let text: String

    /// Creates a demo note.
    public init(id: String, title: String, sourcePath: String, text: String) {
        self.id = id
        self.title = title
        self.sourcePath = sourcePath
        self.text = text
    }
}

/// A suggested agent task tuned to one fact in the bundled corpus.
public struct AgentPromptChip: Identifiable, Equatable, Sendable {
    /// Stable chip identifier.
    public let id: String

    /// Short chip label.
    public let label: String

    /// Prompt inserted into the composer.
    public let prompt: String

    /// Human-readable fact the chip is expected to retrieve.
    public let targetFact: String

    /// Creates a prompt chip.
    public init(id: String, label: String, prompt: String, targetFact: String) {
        self.id = id
        self.label = label
        self.prompt = prompt
        self.targetFact = targetFact
    }
}

/// Static demo corpus for Phase P1 of the Agent Trace UI.
public enum AgentDemoCorpus {
    /// The P1 bundled corpus.
    public static let notes: [AgentDemoNote] = [
        AgentDemoNote(
            id: "agent-loop",
            title: "AgentLoop",
            sourcePath: "tinybrain/agent-loop.md",
            text: """
            TinyBrain AgentLoop runs a plan-act-observe cycle on device. A planning turn may emit one JSON tool call, the app executes that tool, and the observation is appended before the model answers. For P1 the only registered tool is retrieve.
            """
        ),
        AgentDemoNote(
            id: "agent-tools",
            title: "Agent Tools",
            sourcePath: "tinybrain/tools.md",
            text: """
            TinyBrain tools are registered in a ToolRegistry with a schema and async handler. The retrieve tool accepts query and k arguments, then returns numbered passages with source paths and lower-is-better distances.
            """
        ),
        AgentDemoNote(
            id: "xray",
            title: "X-Ray Panel",
            sourcePath: "tinybrain/xray.md",
            text: """
            X-Ray visualizes attention, top token probabilities, layer activation norms, and KV cache pages while the model generates. It uses fill-quaternary tiles, 0.5 hairlines, monospaced numerics, and a LIVE pill.
            """
        ),
        AgentDemoNote(
            id: "telemetry",
            title: "Telemetry",
            sourcePath: "tinybrain/telemetry.md",
            text: """
            TinyBrain telemetry tracks tokens per second, milliseconds per token, estimated energy, average probability, and KV cache utilization. Metrics update on each streamed token.
            """
        ),
        AgentDemoNote(
            id: "kv-cache",
            title: "KV Cache",
            sourcePath: "tinybrain/kv-cache.md",
            text: """
            The KV cache stores past key and value vectors so decoding reuses previous context. TinyBrain shows cache pages as a grid and plans paged reusable cache storage for longer streams.
            """
        ),
        AgentDemoNote(
            id: "quantization",
            title: "Quantization",
            sourcePath: "tinybrain/quantization.md",
            text: """
            TinyBrain implements per-channel INT8 quantization first, with INT4 planned later. Quantized paths are checked against FP16 references with tight numerical tolerances.
            """
        ),
        AgentDemoNote(
            id: "rag-retrieval",
            title: "RAG Retrieval",
            sourcePath: "tinybrain/rag-retrieval.md",
            text: """
            RAG retrieval chunks notes, embeds each chunk, searches the HNSW index, and returns ranked passages. Smaller vector distance means a stronger match for the query.
            """
        ),
        AgentDemoNote(
            id: "prompt-budget",
            title: "Prompt Budget",
            sourcePath: "tinybrain/prompt-budget.md",
            text: """
            TinyBrain RAG keeps retrieved evidence inside a prompt budget. The agent loop also has a step budget; when exhausted, it must answer from gathered observations instead of calling more tools.
            """
        ),
        AgentDemoNote(
            id: "metal",
            title: "Metal Backend",
            sourcePath: "tinybrain/metal.md",
            text: """
            TinyBrain's Metal backend accelerates critical kernels on Apple Silicon while keeping CPU fallbacks for CI. Kernel tuning documents threadgroup sizes and tile dimensions.
            """
        ),
        AgentDemoNote(
            id: "project-atlas",
            title: "Project Atlas",
            sourcePath: "ops/project-atlas.md",
            text: """
            Project Atlas is the internal map annotation sprint. The timing fact is specific: Atlas review lock is August 14, 2026, and the owner is Mira Chen.
            """
        ),
        AgentDemoNote(
            id: "coffee",
            title: "Coffee Recipe",
            sourcePath: "lab/coffee.md",
            text: """
            The lab coffee recipe uses 18 grams of coffee, 288 grams of water, a 94 C kettle, and a 2 minute 45 second total brew. The keyword is courier coffee.
            """
        ),
        AgentDemoNote(
            id: "thermostat",
            title: "Thermostat",
            sourcePath: "lab/thermostat.md",
            text: """
            The west lab thermostat target is 21 C after 18:00. During benchmark runs, keep the room below 23 C so fan noise does not contaminate recordings.
            """
        ),
        AgentDemoNote(
            id: "travel",
            title: "Travel Memo",
            sourcePath: "ops/travel.md",
            text: """
            The July field trip meets at Gate B7 at 07:35. Pack the USB-C power meter, the thermal camera, and the small calibration stand.
            """
        ),
        AgentDemoNote(
            id: "release",
            title: "Release Checklist",
            sourcePath: "ops/release.md",
            text: """
            Before a TinyBrain demo release, run swift build, swift test, and the tinybrain-chat smoke launch. Do not commit model binaries; models stay under Models with gitignore coverage.
            """
        ),
        AgentDemoNote(
            id: "energy-overlay",
            title: "Energy Overlay",
            sourcePath: "tinybrain/energy-overlay.md",
            text: """
            The Energy Overlay label is user-facing copy for the demo. It should stay consistent across UI and docs when showing estimated joules or thermal behavior.
            """
        )
    ]

    /// RAG documents derived from the bundled notes.
    public static var documents: [RAGDocument] {
        notes.map { RAGDocument(text: $0.text, sourcePath: $0.sourcePath) }
    }

    /// Suggested prompts whose wording shares terms with the corpus for both
    /// NaturalLanguage and deterministic stub retrieval.
    public static let promptChips: [AgentPromptChip] = [
        AgentPromptChip(
            id: "agentloop",
            label: "AgentLoop",
            prompt: "Explain the TinyBrain AgentLoop plan-act-observe cycle and the retrieve tool.",
            targetFact: "AgentLoop uses a plan-act-observe cycle and P1 registers retrieve."
        ),
        AgentPromptChip(
            id: "atlas",
            label: "Atlas Timing",
            prompt: "Find the Project Atlas review lock timing and owner.",
            targetFact: "Atlas review lock is August 14, 2026; owner Mira Chen."
        ),
        AgentPromptChip(
            id: "distance",
            label: "Distances",
            prompt: "How does RAG retrieval use ranked passages and vector distances?",
            targetFact: "Smaller vector distance means a stronger retrieval match."
        ),
        AgentPromptChip(
            id: "coffee",
            label: "Coffee",
            prompt: "What is the courier coffee recipe in the lab note?",
            targetFact: "18 g coffee, 288 g water, 94 C kettle, 2:45 brew."
        )
    ]
}
