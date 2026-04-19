# TinyBrainCartographerBridge

Bridge module that conforms a TinyBrain-backed inference actor to
Cartographer's `SmartAnnotationService` protocol.

The contract this module implements is defined in the Cartographer repo at
`docs/INTEGRATION-TINYBRAIN.md` — that file is the source of truth for
lifecycle, threading, token budgets, fallback, and footprint.

## Importing

```swift
import TinyBrainCartographerBridge

let modelURL = Bundle.main.url(forResource: "tinyllama-int4", withExtension: "tbf")!
let smart = try TinyBrainSmartService(modelURL: modelURL)
// Pass `smart` wherever Cartographer expects `any SmartAnnotationService`.
```

## Local dependency note

`Cartographer` is declared in this repo's `Package.swift` as a local path
dependency at `Dependencies/cartographer`. Until the Cartographer v0.2
protocol branch publishes to `origin/main` (tracked on CHA-138 / CHA-153),
contributors need a local checkout at that path — e.g.:

```bash
git clone --branch feat/cg-003-demo-app-wiring \
  <cartographer-source> \
  Dependencies/cartographer
```

The `Dependencies/` directory is `.gitignore`d. When Cartographer tags a
protocol preview on origin, flip the dep in `Package.swift` to a git URL
with an explicit revision and delete the local checkout.
