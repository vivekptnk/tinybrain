/// Key-Value cache for transformer attention
///
/// **TB-004 Phase 4:** Paged KV cache enabling efficient streaming inference
///
/// ## How Transformer Attention Works
///
/// For token at position `t`, attention computes:
/// ```
/// Q_t = query for current token
/// K_0...t = keys for all previous tokens (NEED TO CACHE!)
/// V_0...t = values for all previous tokens (NEED TO CACHE!)
///
/// attention_scores = Q_t × [K_0, K_1, ..., K_t]ᵀ
/// output = softmax(attention_scores) × [V_0, V_1, ..., V_t]
/// ```
///
/// **Without cache:** Recompute K₀...K_{t-1} every time → O(t²) compute!  
/// **With cache:** Store K₀...K_{t-1}, only compute K_t → O(t) compute!
///
/// ## Paging Strategy
///
/// Instead of one giant buffer, use **pages**:
/// ```
/// Page 0: tokens 0-15
/// Page 1: tokens 16-31
/// Page 2: tokens 32-47
/// ...
/// ```
///
/// Benefits:
/// - Pre-allocated on GPU (zero overhead during inference)
/// - Easy eviction (free oldest page)
/// - Efficient for variable-length sequences

import Foundation
#if canImport(Metal)
import Metal
#endif

/// Backing storage for a page of cached keys/values
private final class KVPage {
    var data: [Float]
    
    init(size: Int) {
        self.data = [Float](repeating: 0.0, count: size)
    }
    
    func reset() {
        for i in 0..<data.count {
            data[i] = 0
        }
    }
}

private struct PooledPage {
    let poolIndex: Int
    let page: KVPage
}

/// KV cache manager for transformer layers
public final class KVCache {
    /// Number of transformer layers
    public let numLayers: Int
    
    /// Hidden dimension size
    public let hiddenDim: Int
    
    /// Maximum tokens to cache
    public let maxTokens: Int
    
    /// Tokens per page
    public let pageSize: Int
    
    /// Maximum number of pages per layer
    private let maxPages: Int
    
    /// Page allocators per layer
    private var allocators: [PageAllocator]
    
    /// Storage for keys: [layer][logicalPageId] → pooled page binding
    private var keyPages: [[Int: PooledPage]]
    
    /// Storage for values: [layer][logicalPageId] → pooled page binding
    private var valuePages: [[Int: PooledPage]]
    
    /// Pre-allocated key buffers per layer
    private let keyPools: [[KVPage]]
    
    /// Pre-allocated value buffers per layer
    private let valuePools: [[KVPage]]
    
    /// Current sequence length per layer
    private var lengths: [Int]
    
    /// Minimum cached position per layer (for eviction tracking)
    private var minPositions: [Int]
    
    /// Lock for thread safety
    private let lock = NSLock()
    
    /// Initialize KV cache
    ///
    /// - Parameters:
    ///   - numLayers: Number of transformer layers
    ///   - hiddenDim: Hidden dimension size (e.g., 768)
    ///   - maxTokens: Maximum tokens to cache (e.g., 2048)
    ///   - pageSize: Tokens per page (default: 16)
    public init(numLayers: Int, hiddenDim: Int, maxTokens: Int, pageSize: Int = 16) {
        self.numLayers = numLayers
        self.hiddenDim = hiddenDim
        self.maxTokens = maxTokens
        self.pageSize = pageSize
        
        self.maxPages = (maxTokens + pageSize - 1) / pageSize
        // Initialize simple properties first
        self.keyPages = Array(repeating: [:], count: numLayers)
        self.valuePages = Array(repeating: [:], count: numLayers)
        self.lengths = Array(repeating: 0, count: numLayers)
        self.minPositions = Array(repeating: 0, count: numLayers)
        
        // Initialize arrays that need self
        let elementsPerPage = pageSize * hiddenDim
        var allocators: [PageAllocator] = []
        var keyPools: [[KVPage]] = []
        var valuePools: [[KVPage]] = []
        
        for _ in 0..<numLayers {
            allocators.append(PageAllocator(pageSize: pageSize, maxPages: maxPages))
            keyPools.append((0..<maxPages).map { _ in KVPage(size: elementsPerPage) })
            valuePools.append((0..<maxPages).map { _ in KVPage(size: elementsPerPage) })
        }
        
        self.allocators = allocators
        self.keyPools = keyPools
        self.valuePools = valuePools
    }
    
    /// Current cache length (max across all layers)
    public var length: Int {
        lock.lock()
        defer { lock.unlock() }
        
        return lengths.max() ?? 0
    }
    
    /// Append key/value for a new token
    ///
    /// **TB-004:** Core API for building up context during inference
    ///
    /// Automatically evicts oldest tokens if cache exceeds maxTokens.
    ///
    /// - Parameters:
    ///   - layer: Layer index [0, numLayers)
    ///   - key: Key tensor [hiddenDim]
    ///   - value: Value tensor [hiddenDim]
    ///   - position: Token position in sequence
    public func append(layer: Int, key: Tensor<Float>, value: Tensor<Float>, position: Int) {
        lock.lock()
        defer { lock.unlock() }
        
        precondition(layer >= 0 && layer < numLayers, "Layer index out of bounds")
        precondition(key.shape == TensorShape(hiddenDim), "Key shape mismatch")
        precondition(value.shape == TensorShape(hiddenDim), "Value shape mismatch")
        
        let allocator = allocators[layer]
        
        // Evict until we have room for another token
        while lengths[layer] >= maxTokens {
            evictOldestPage(layer: layer)
            lengths[layer] = max(0, lengths[layer] - pageSize)
        }
        
        // Determine which page this position belongs to
        let pageId = position / pageSize
        let offsetInPage = position % pageSize
        
        // Get or create pooled page
        if keyPages[layer][pageId] == nil {
            var poolIndex = allocator.allocatePage()
            
            if poolIndex == nil {
                evictOldestPage(layer: layer)
                poolIndex = allocator.allocatePage()
            }
            
            guard let resolvedIndex = poolIndex else {
                fatalError("Failed to allocate KV cache page for layer \(layer)")
            }
            
            let keyPage = keyPools[layer][resolvedIndex]
            keyPage.reset()
            let valuePage = valuePools[layer][resolvedIndex]
            valuePage.reset()
            
            keyPages[layer][pageId] = PooledPage(poolIndex: resolvedIndex, page: keyPage)
            valuePages[layer][pageId] = PooledPage(poolIndex: resolvedIndex, page: valuePage)
        }
        
        // Write key/value into page (FAST: direct mutation, no copying!)
        if let keyBinding = keyPages[layer][pageId],
           let valueBinding = valuePages[layer][pageId] {
            
            // **REVIEW HITLER FIX:** Direct mutation (reference semantics)
            // No copying! We mutate the class's internal array directly
            let rowStartIdx = offsetInPage * hiddenDim
            let keyInput = key.rawData
            let valueInput = value.rawData
            
            // Bulk write directly to page's data array
            for i in 0..<hiddenDim {
                keyBinding.page.data[rowStartIdx + i] = keyInput[i]
                valueBinding.page.data[rowStartIdx + i] = valueInput[i]
            }
            // No need to store back - it's a class reference!
        }
        
        // Update length - track how many tokens we have
        lengths[layer] = max(lengths[layer], position + 1)
        
        // Cap at maxTokens
        if lengths[layer] > maxTokens {
            lengths[layer] = maxTokens
        }
    }
    
    /// Get keys for a range of tokens
    ///
    /// - Parameters:
    ///   - layer: Layer index
    ///   - range: Token positions to retrieve
    /// - Returns: Tensor of shape [range.count, hiddenDim]
    public func getKeys(layer: Int, range: Range<Int>) -> Tensor<Float> {
        lock.lock()
        defer { lock.unlock() }
        
        precondition(layer >= 0 && layer < numLayers, "Layer out of bounds")
        
        let count = range.count
        var result = Tensor<Float>.zeros(shape: TensorShape(count, hiddenDim))
        
        // Collect keys from pages (convert back to tensor)
        for (idx, pos) in range.enumerated() {
            let pageId = pos / pageSize
            let offsetInPage = pos % pageSize
            
            if let binding = keyPages[layer][pageId] {
                // Copy row from page to result
                let rowStartIdx = offsetInPage * hiddenDim
                for i in 0..<hiddenDim {
                    result[idx, i] = binding.page.data[rowStartIdx + i]
                }
            }
        }
        
        return result
    }
    
    /// Get values for a range of tokens
    ///
    /// - Parameters:
    ///   - layer: Layer index
    ///   - range: Token positions to retrieve
    /// - Returns: Tensor of shape [range.count, hiddenDim]
    public func getValues(layer: Int, range: Range<Int>) -> Tensor<Float> {
        lock.lock()
        defer { lock.unlock() }
        
        precondition(layer >= 0 && layer < numLayers, "Layer out of bounds")
        
        let count = range.count
        var result = Tensor<Float>.zeros(shape: TensorShape(count, hiddenDim))
        
        // Collect values from pages (convert back to tensor)
        for (idx, pos) in range.enumerated() {
            let pageId = pos / pageSize
            let offsetInPage = pos % pageSize
            
            if let binding = valuePages[layer][pageId] {
                // Copy row from page to result
                let rowStartIdx = offsetInPage * hiddenDim
                for i in 0..<hiddenDim {
                    result[idx, i] = binding.page.data[rowStartIdx + i]
                }
            }
        }
        
        return result
    }
    
    /// Evict oldest page to make room
    ///
    /// **Strategy:** Remove lowest page ID (FIFO eviction)
    private func evictOldestPage(layer: Int) {
        // Find the oldest page (lowest logical page ID still allocated)
        guard let oldestLogicalPageId = keyPages[layer].keys.min(),
              let binding = keyPages[layer][oldestLogicalPageId] else {
            return
        }
        
        allocators[layer].freePage(binding.poolIndex)
        
        // Remove pages from cache
        keyPages[layer].removeValue(forKey: oldestLogicalPageId)
        valuePages[layer].removeValue(forKey: oldestLogicalPageId)
        
        // Update min position
        minPositions[layer] = (oldestLogicalPageId + 1) * pageSize
    }
    
    /// Clear all cached data
    ///
    /// Useful for starting a new sequence/conversation
    public func clear() {
        lock.lock()
        defer { lock.unlock() }
        
        // Free all pages
        for layer in 0..<numLayers {
            allocators[layer].reset()
            keyPages[layer].removeAll()
            valuePages[layer].removeAll()
            lengths[layer] = 0
            minPositions[layer] = 0
        }
    }
    
    /// Get cache statistics
    public var stats: String {
        lock.lock()
        defer { lock.unlock() }
        
        let totalPages = keyPages.flatMap { $0.keys }.count
        let utilizationPct = Double(length) / Double(maxTokens) * 100
        
        return """
        KVCache Stats:
          Layers: \(numLayers)
          Cached tokens: \(length) / \(maxTokens)
          Utilization: \(String(format: "%.1f%%", utilizationPct))
          Pages used: \(totalPages)
          Page size: \(pageSize) tokens/page
        """
    }
}
