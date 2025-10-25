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
/// - Metal acceleration (`MetalBackend`) - **automatically initialized!**
/// - Tokenization (`Tokenizer`, `BPETokenizer`)
///
/// Metal GPU acceleration is **automatically enabled** when you import TinyBrain.
///
/// You don't need to import the individual submodules unless you want fine-grained control.

@_exported import TinyBrainRuntime
@_exported import TinyBrainMetal
@_exported import TinyBrainTokenizer

// MARK: - Metal Configuration

/// **REVIEW HITLER FIX:** Metal initialization happens when TinyBrainBackend is first accessed
///
/// Since TinyBrainBackend extension is defined below, it will run when the class is loaded.
/// The enableMetalAcceleration() override ensures Metal is available.

/// Implementation of enableMetal() for the umbrella module
///
/// This allows TinyBrainRuntime to remain independent of Metal
/// while the umbrella module provides the integration.
extension TinyBrainBackend {
    /// **REVIEW HITLER FIX:** Auto-initialize Metal on first access
    private static let autoInit: Bool = {
        if MetalBackend.isAvailable {
            do {
                let backend = try MetalBackend()
                metalBackend = backend
                print("[TinyBrain] ✅ Metal automatically enabled on first use")
                return true
            } catch {
                print("[TinyBrain] ⚠️ Metal init failed: \(error)")
                return false
            }
        }
        return false
    }()
    
    /// Enable Metal GPU acceleration (actual implementation)
    ///
    /// **REVIEW HITLER FIX:** Now triggers auto-init if not already done
    ///
    /// Call this once at app startup (or let auto-init handle it):
    /// ```swift
    /// import TinyBrain  // Auto-inits Metal!
    ///
    /// // Optional explicit call:
    /// TinyBrainBackend.enableMetal()
    /// ```
    @discardableResult
    public static func enableMetalAcceleration() -> Bool {
        // Trigger auto-init
        _ = autoInit
        
        guard metalBackend == nil else {
            return true  // Already enabled
        }
        
        // Try to create Metal backend
        guard MetalBackend.isAvailable else {
            if debugLogging {
                print("[TinyBrain] Metal not available on this device")
            }
            return false
        }
        
        do {
            let backend = try MetalBackend()
            metalBackend = backend
            
            if debugLogging {
                print("[TinyBrain] ✅ Metal GPU acceleration enabled (\(backend.deviceInfo))")
                print("[TinyBrain] GPU will be used for matrices ≥\(metalSizeThreshold)×\(metalSizeThreshold)")
            }
            
            return true
        } catch {
            if debugLogging {
                print("[TinyBrain] ❌ Metal initialization failed: \(error)")
            }
            return false
        }
    }
    
    /// **TB-004:** Alias for enableMetalAcceleration() to match Backend API
    @discardableResult
    public static func enableMetal() -> Bool {
        enableMetalAcceleration()
    }
}

