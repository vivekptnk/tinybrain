// swift-tools-version: 5.10
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "TinyBrain",
    platforms: [
        .iOS(.v17),
        .macOS(.v14)
    ],
    products: [
        // Main umbrella library (import this in your app)
        .library(
            name: "TinyBrain",
            targets: ["TinyBrain"]
        ),
        // Individual modules (for advanced users)
        .library(
            name: "TinyBrainRuntime",
            targets: ["TinyBrainRuntime"]
        ),
        .library(
            name: "TinyBrainMetal",
            targets: ["TinyBrainMetal"]
        ),
        .library(
            name: "TinyBrainTokenizer",
            targets: ["TinyBrainTokenizer"]
        ),
        .library(
            name: "TinyBrainDemo",
            targets: ["TinyBrainDemo"]
        ),
        // ProximaKit bridge (optional — brings in ProximaKit dependency)
        .library(
            name: "TinyBrainProximaKit",
            targets: ["TinyBrainProximaKit"]
        ),
        .library(
            name: "TinyBrainRAG",
            targets: ["TinyBrainRAG"]
        ),
        // Cartographer bridge (optional — conforms to Cartographer's
        // SmartAnnotationService protocol). Cartographer is pinned by SHA
        // on feat/cg-003-demo-app-wiring (see dependencies below). Re-pin
        // to a tag or main once that branch lands upstream (CHA-175 follow-up).
        .library(
            name: "TinyBrainCartographerBridge",
            targets: ["TinyBrainCartographerBridge"]
        ),
        // Demo chat app
        .executable(
            name: "tinybrain-chat",
            targets: ["ChatDemo"]
        ),
        // Benchmark executable
        .executable(
            name: "tinybrain-bench",
            targets: ["TinyBrainBench"]
        ),
        // Retrieval-augmented generation CLI
        .executable(
            name: "tinybrain-rag",
            targets: ["RAGDemo"]
        )
    ],
    dependencies: [
        // Swift Argument Parser for CLI tools
        .package(url: "https://github.com/apple/swift-argument-parser", from: "1.3.0"),
        // YAML parsing for benchmark scenarios
        .package(url: "https://github.com/jpsim/Yams.git", from: "5.0.0"),
        // ProximaKit — vector similarity search (used by TinyBrainProximaKit bridge)
        // ProximaKit pinned by revision to the commit that became the v1.6.0 tag; re-pin with from: "1.6.0" in a future bump.
        .package(url: "https://github.com/vivekptnk/ProximaKit.git", revision: "dcc55ca6b44d50707421b1da404c2e49117596e9"),
        // Cartographer — map annotation engine (used by TinyBrainCartographerBridge).
        // Pinned to 1b15048 (fix(concurrency): constrain retryOnTransient to Sendable for Swift 6 — required for the CI runner's toolchain; the previous pin 81a3047 failed to compile on GitHub's macos-15 Swift with a non-sendable-result error in SyncEngine).
        // Pinning by revision (not branch:) keeps resolution stable if the branch is
        // ever force-pushed. Re-pin to a tag once the branch lands upstream.
        .package(
            url: "https://github.com/vivekptnk/cartographer.git",
            revision: "1b150485860679e227bd37799e5f3357dcc3a7fe"
        )
    ],
    targets: [
        // MARK: - Umbrella Module
        .target(
            name: "TinyBrain",
            dependencies: [
                "TinyBrainRuntime",
                "TinyBrainMetal",
                "TinyBrainTokenizer"
            ],
            path: "Sources/TinyBrain"
        ),
        
        // MARK: - Core Runtime
        .target(
            name: "TinyBrainRuntime",
            dependencies: [],
            path: "Sources/TinyBrainRuntime"
        ),
        .testTarget(
            name: "TinyBrainRuntimeTests",
            dependencies: ["TinyBrainRuntime", "TinyBrainMetal"],
            path: "Tests/TinyBrainRuntimeTests",
            resources: [
                .process("Fixtures")
            ]
        ),
        
        // MARK: - Metal Backend
        .target(
            name: "TinyBrainMetal",
            dependencies: ["TinyBrainRuntime"],
            path: "Sources/TinyBrainMetal",
            resources: [
                .process("Shaders")
            ]
        ),
        .testTarget(
            name: "TinyBrainMetalTests",
            dependencies: ["TinyBrainMetal"],
            path: "Tests/TinyBrainMetalTests"
        ),
        
        // MARK: - Tokenizer
        .target(
            name: "TinyBrainTokenizer",
            dependencies: ["TinyBrainRuntime"],
            path: "Sources/TinyBrainTokenizer"
        ),
        .testTarget(
            name: "TinyBrainTokenizerTests",
            dependencies: ["TinyBrainTokenizer"],
            path: "Tests/TinyBrainTokenizerTests",
            resources: [
                .process("Fixtures")
            ]
        ),
        
        // MARK: - Demo App Library
        .target(
            name: "TinyBrainDemo",
            dependencies: [
                "TinyBrainRuntime",
                "TinyBrainMetal",
                "TinyBrainTokenizer"
            ],
            path: "Sources/TinyBrainDemo"
        ),
        .testTarget(
            name: "TinyBrainDemoTests",
            dependencies: ["TinyBrainDemo"],
            path: "Tests/TinyBrainDemoTests"
        ),
        
        // MARK: - ProximaKit Bridge
        .target(
            name: "TinyBrainProximaKit",
            dependencies: [
                "TinyBrainRuntime",
                "TinyBrainTokenizer",
                .product(name: "ProximaKit", package: "ProximaKit")
            ],
            path: "Sources/TinyBrainProximaKit"
        ),
        .testTarget(
            name: "TinyBrainProximaKitTests",
            dependencies: ["TinyBrainProximaKit"],
            path: "Tests/TinyBrainProximaKitTests"
        ),

        // MARK: - Retrieval-Augmented Generation
        .target(
            name: "TinyBrainRAG",
            dependencies: [
                "TinyBrainRuntime",
                "TinyBrainTokenizer",
                .product(name: "ProximaKit", package: "ProximaKit")
            ],
            path: "Sources/TinyBrainRAG"
        ),
        .testTarget(
            name: "TinyBrainRAGTests",
            dependencies: ["TinyBrainRAG"],
            path: "Tests/TinyBrainRAGTests",
            resources: [
                .process("Fixtures")
            ]
        ),

        // MARK: - Cartographer Bridge
        .target(
            name: "TinyBrainCartographerBridge",
            dependencies: [
                "TinyBrainRuntime",
                "TinyBrainTokenizer",
                .product(name: "Cartographer", package: "cartographer")
            ],
            path: "Sources/TinyBrainCartographerBridge",
            exclude: ["README.md"]
        ),
        .testTarget(
            name: "TinyBrainCartographerBridgeTests",
            dependencies: [
                "TinyBrainCartographerBridge",
                .product(name: "Cartographer", package: "cartographer")
            ],
            path: "Tests/TinyBrainCartographerBridgeTests"
        ),

        // MARK: - Chat Demo Executable
        .executableTarget(
            name: "ChatDemo",
            dependencies: ["TinyBrainDemo"],
            path: "Examples/ChatDemo",
            exclude: ["Info.plist"]
        ),

        // MARK: - RAG Demo Executable
        .executableTarget(
            name: "RAGDemo",
            dependencies: [
                "TinyBrainRAG",
                "TinyBrainRuntime",
                "TinyBrainTokenizer",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
                .product(name: "ProximaEmbeddings", package: "ProximaKit")
            ],
            path: "Examples/RAGDemo"
        ),
        
        // MARK: - Benchmark Tool
        .executableTarget(
            name: "TinyBrainBench",
            dependencies: [
                "TinyBrainRuntime",
                "TinyBrainMetal",
                "TinyBrainTokenizer",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
                .product(name: "Yams", package: "Yams")
            ],
            path: "Sources/TinyBrainBench"
        ),
        .testTarget(
            name: "TinyBrainBenchTests",
            dependencies: ["TinyBrainBench"],
            path: "Tests/TinyBrainBenchTests",
            resources: [
                .process("Fixtures")
            ]
        )
    ]
)
