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
        // TinyBrainCartographerBridge product removed from the default build
        // because the upstream `vivekptnk/cartographer` repo is private and
        // CI has no credentials to fetch it. Source files remain in
        // `Sources/TinyBrainCartographerBridge/` and will be re-enabled once
        // cartographer is public OR CI auth is provisioned (tracked in CHA-177).
        // Demo chat app
        .executable(
            name: "tinybrain-chat",
            targets: ["ChatDemo"]
        ),
        // Benchmark executable
        .executable(
            name: "tinybrain-bench",
            targets: ["TinyBrainBench"]
        )
    ],
    dependencies: [
        // Swift Argument Parser for CLI tools
        .package(url: "https://github.com/apple/swift-argument-parser", from: "1.3.0"),
        // YAML parsing for benchmark scenarios
        .package(url: "https://github.com/jpsim/Yams.git", from: "5.0.0"),
        // ProximaKit — vector similarity search (used by TinyBrainProximaKit bridge)
        .package(url: "https://github.com/vivekptnk/ProximaKit.git", branch: "main")
        // Cartographer dep removed — repo is private and CI can't auth.
        // Re-add via pinned git URL once CHA-177 provisions CI creds or
        // the repo is made public.
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

        // MARK: - Cartographer Bridge (disabled — see CHA-177)
        // Target declarations removed to unbreak CI while cartographer repo
        // is private. Re-enable by restoring this block and the matching
        // dependency/product entries above.

        // MARK: - Chat Demo Executable
        .executableTarget(
            name: "ChatDemo",
            dependencies: ["TinyBrainDemo"],
            path: "Examples/ChatDemo",
            exclude: ["Info.plist"]
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

