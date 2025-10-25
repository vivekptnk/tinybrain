/// TinyBrain – Umbrella Module
///
/// This is the main entry point for TinyBrain. Import this module in your app:
///
/// ```swift
/// import TinyBrain
/// ```
///
/// This automatically gives you access to all TinyBrain functionality:
/// - Core runtime (`Tensor`, `ModelRunner`)
/// - Metal acceleration (`MetalBackend`)
/// - Tokenization (`Tokenizer`, `BPETokenizer`)
///
/// You don't need to import the individual submodules unless you want fine-grained control.

@_exported import TinyBrainRuntime
@_exported import TinyBrainMetal
@_exported import TinyBrainTokenizer

