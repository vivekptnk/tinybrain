/// Page allocator for KV cache memory management
///
/// **TB-004 Phase 4:** Implements paging strategy for efficient attention cache
///
/// ## What is a Page Allocator?
///
/// Like OS memory management, but for transformer KV cache!
///
/// **The Problem:**
/// - Need to cache 2048 tokens × 768 dimensions × 2 (K and V) = ~12MB per layer
/// - Allocating/deallocating during inference is SLOW
/// - Solution: Pre-allocate pages, hand them out as needed!
///
/// **Page Strategy:**
/// ```
/// Pre-allocate:
/// ┌────┬────┬────┬────┬────┬────┬────┬────┐
/// │ P0 │ P1 │ P2 │ P3 │ P4 │ P5 │ P6 │ P7 │ ... (128 pages total)
/// └────┴────┴────┴────┴────┴────┴────┴────┘
///  16    16   16   16   16   16   16   16  tokens per page
///
/// As tokens arrive:
/// Token 0-15:  Use P0
/// Token 16-31: Allocate P1
/// Token 32-47: Allocate P2
/// ...
///
/// When context fills:
/// Evict P0 (oldest), reuse for new tokens!
/// ```

import Foundation

/// Page allocator for KV cache
public final class PageAllocator {
    /// Number of tokens per page
    public let pageSize: Int
    
    /// Maximum number of pages in pool
    public let maxPages: Int
    
    /// Free page IDs available for allocation
    private var freePages: [Int]
    
    /// Currently allocated page IDs
    private var allocatedPages: Set<Int>
    
    /// Lock for thread safety
    private let lock = NSLock()
    
    /// Initialize page allocator
    ///
    /// - Parameters:
    ///   - pageSize: Tokens per page (default: 16)
    ///   - maxPages: Maximum pages in pool (default: 128 = 2048 tokens)
    public init(pageSize: Int = 16, maxPages: Int = 128) {
        self.pageSize = pageSize
        self.maxPages = maxPages
        
        // Initialize free list with all page IDs
        self.freePages = Array(0..<maxPages).reversed()  // Stack: pop from end
        self.allocatedPages = []
    }
    
    /// Allocate a page from the free list
    ///
    /// **Zero allocation!** Just hands out pre-existing page ID.
    ///
    /// - Returns: Page ID, or nil if pool exhausted
    public func allocatePage() -> Int? {
        lock.lock()
        defer { lock.unlock() }
        
        guard !freePages.isEmpty else {
            return nil  // Pool exhausted
        }
        
        let pageId = freePages.removeLast()
        allocatedPages.insert(pageId)
        
        return pageId
    }
    
    /// Free a page back to the pool
    ///
    /// **Makes it available for reuse** - no deallocation!
    ///
    /// - Parameter pageId: Page ID to free
    public func freePage(_ pageId: Int) {
        lock.lock()
        defer { lock.unlock() }
        
        guard allocatedPages.contains(pageId) else {
            return  // Already freed or invalid
        }
        
        allocatedPages.remove(pageId)
        freePages.append(pageId)
    }
    
    /// Get number of currently allocated pages
    public var allocatedCount: Int {
        lock.lock()
        defer { lock.unlock() }
        
        return allocatedPages.count
    }
    
    /// Get number of free pages available
    public var freeCount: Int {
        lock.lock()
        defer { lock.unlock() }
        
        return freePages.count
    }
    
    /// Compact the free list (defragmentation)
    ///
    /// Sorts free pages for better cache locality.
    public func compact() {
        lock.lock()
        defer { lock.unlock() }
        
        // Sort free pages for sequential allocation
        freePages.sort(by: >)  // Reverse order for stack-like pop
    }
    
    /// Reset allocator to initial state
    public func reset() {
        lock.lock()
        defer { lock.unlock() }
        
        allocatedPages.removeAll()
        freePages = Array(0..<maxPages).reversed()
    }
    
    /// Debug description
    public var description: String {
        lock.lock()
        defer { lock.unlock() }
        
        let utilizationPct = Double(allocatedPages.count) / Double(maxPages) * 100
        
        return """
        PageAllocator:
          Page size: \(pageSize) tokens
          Max pages: \(maxPages)
          Allocated: \(allocatedPages.count) (\(String(format: "%.1f%%", utilizationPct)))
          Free: \(freePages.count)
        """
    }
}

