import Foundation

/// Filesystem operation audited by ``SandboxPolicy``.
public enum SandboxOperation: String, Codable, Equatable {
    /// File read operation.
    case readFile
    /// Directory listing operation.
    case listDirectory
    /// File write operation.
    case writeFile
}

/// Sandbox decision recorded for an attempted effect.
public enum SandboxDecision: String, Codable, Equatable {
    /// The operation was allowed by policy.
    case allowed
    /// The operation was denied by policy.
    case denied
    /// A write was authorized but skipped because dry-run mode is enabled.
    case dryRun
}

/// Audit record emitted by every allowed and denied filesystem attempt.
public struct AuditEvent: Codable, Equatable {
    /// Operation being attempted.
    public let operation: SandboxOperation

    /// Original path supplied by the tool call.
    public let requestedPath: String

    /// Canonical path after standardization and symlink resolution.
    public let resolvedPath: String

    /// Policy decision.
    public let decision: SandboxDecision

    /// Optional byte count associated with the operation.
    public let byteCount: Int?

    /// Human-readable reason for denials or dry-runs.
    public let reason: String?

    /// Creates an audit event.
    public init(
        operation: SandboxOperation,
        requestedPath: String,
        resolvedPath: String,
        decision: SandboxDecision,
        byteCount: Int? = nil,
        reason: String? = nil
    ) {
        self.operation = operation
        self.requestedPath = requestedPath
        self.resolvedPath = resolvedPath
        self.decision = decision
        self.byteCount = byteCount
        self.reason = reason
    }
}

/// Errors returned by sandbox-gated tools.
public enum SandboxPolicyError: Error, Equatable, LocalizedError {
    /// The path is outside the readable roots.
    case outsideReadableSandbox(String)
    /// The path is outside the writable roots.
    case outsideWritableSandbox(String)
    /// The path is not a regular file.
    case notRegularFile(String)
    /// The path is not a directory.
    case notDirectory(String)
    /// Dry-run mode prevented an otherwise authorized write.
    case dryRunWrite(String)
    /// A required string argument was missing.
    case missingArgument(String)

    public var errorDescription: String? {
        switch self {
        case .outsideReadableSandbox(let path):
            return "\(path) is outside the readable sandbox"
        case .outsideWritableSandbox(let path):
            return "\(path) is outside the writable sandbox"
        case .notRegularFile(let path):
            return "\(path) is not a regular file"
        case .notDirectory(let path):
            return "\(path) is not a directory"
        case .dryRunWrite(let path):
            return "dry-run enabled; write_file did not write \(path)"
        case .missingArgument(let name):
            return "missing required argument '\(name)'"
        }
    }
}

/// Security boundary for agent filesystem tools.
///
/// Paths are canonicalized and prefix-checked before the filesystem effect. This
/// policy does not defend against a time-of-check/time-of-use race where another
/// process swaps a symlink between the check and the write. TinyBrain's threat
/// model is a trusted single-user local app, not hostile concurrent processes.
public struct SandboxPolicy {
    /// Explicit roots from which tools may read.
    public let readableRoots: [URL]

    /// Explicit roots to which tools may write. Empty means deny all writes.
    public let writableRoots: [URL]

    /// When true, authorized writes are audited but not performed.
    public let dryRun: Bool

    /// Maximum bytes returned by `read_file`.
    public let maxReadBytes: Int

    /// Maximum entries returned by `list_dir`.
    public let maxDirectoryEntries: Int

    private let auditSink: (AuditEvent) -> Void

    /// Creates a sandbox policy.
    ///
    /// - Precondition: `maxReadBytes` and `maxDirectoryEntries` must be positive.
    public init(
        readableRoots: [URL] = [],
        writableRoots: [URL] = [],
        dryRun: Bool = false,
        maxReadBytes: Int = 64 * 1024,
        maxDirectoryEntries: Int = 200,
        audit: @escaping (AuditEvent) -> Void = { _ in }
    ) {
        precondition(maxReadBytes > 0, "maxReadBytes must be positive")
        precondition(maxDirectoryEntries > 0, "maxDirectoryEntries must be positive")
        self.readableRoots = readableRoots.map(Self.canonicalRoot)
        self.writableRoots = writableRoots.map(Self.canonicalRoot)
        self.dryRun = dryRun
        self.maxReadBytes = maxReadBytes
        self.maxDirectoryEntries = maxDirectoryEntries
        self.auditSink = audit
    }

    /// Reads a UTF-8 file under a readable root, truncating to `maxReadBytes`.
    public func readFile(atPath path: String) throws -> String {
        let resolved = resolve(path, preferredRoots: readableRoots)
        guard isAllowed(resolved, roots: readableRoots) else {
            deny(.readFile, requestedPath: path, resolved: resolved, reason: "outside the readable sandbox")
            throw SandboxPolicyError.outsideReadableSandbox(resolved.path)
        }

        guard isRegularFile(resolved) else {
            audit(.readFile, requestedPath: path, resolved: resolved, decision: .allowed)
            throw SandboxPolicyError.notRegularFile(resolved.path)
        }

        let data = try readDataPrefix(from: resolved)
        let capped = data.prefix(maxReadBytes)
        let content = String(data: capped, encoding: .utf8) ?? String(decoding: capped, as: UTF8.self)
        let truncated = data.count > maxReadBytes
        audit(
            .readFile,
            requestedPath: path,
            resolved: resolved,
            decision: .allowed,
            byteCount: min(data.count, maxReadBytes)
        )
        if truncated {
            return content + "\n[truncated at \(maxReadBytes) bytes]"
        }
        return content
    }

    /// Lists entries under a readable root.
    public func listDirectory(atPath path: String) throws -> String {
        let resolved = resolve(path, preferredRoots: readableRoots)
        guard isAllowed(resolved, roots: readableRoots) else {
            deny(.listDirectory, requestedPath: path, resolved: resolved, reason: "outside the readable sandbox")
            throw SandboxPolicyError.outsideReadableSandbox(resolved.path)
        }

        guard isDirectory(resolved) else {
            audit(.listDirectory, requestedPath: path, resolved: resolved, decision: .allowed)
            throw SandboxPolicyError.notDirectory(resolved.path)
        }

        let urls = try FileManager.default.contentsOfDirectory(
            at: resolved,
            includingPropertiesForKeys: [.isDirectoryKey, .fileSizeKey],
            options: [.skipsHiddenFiles]
        )
        let sorted = urls.sorted { $0.lastPathComponent < $1.lastPathComponent }
        let lines = try sorted.prefix(maxDirectoryEntries).map { url in
            let values = try url.resourceValues(forKeys: [.isDirectoryKey, .fileSizeKey])
            let isDirectory = values.isDirectory == true
            let suffix = isDirectory ? "/" : ""
            let size = values.fileSize ?? 0
            return "\(url.lastPathComponent)\(suffix)\t\(isDirectory ? "directory" : "file")\t\(size) bytes"
        }
        let truncated = sorted.count > maxDirectoryEntries
        audit(
            .listDirectory,
            requestedPath: path,
            resolved: resolved,
            decision: .allowed,
            byteCount: sorted.count
        )
        if truncated {
            return (lines + ["[truncated at \(maxDirectoryEntries) entries]"]).joined(separator: "\n")
        }
        return lines.joined(separator: "\n")
    }

    /// Writes a UTF-8 file under a writable root.
    public func writeFile(atPath path: String, content: String) throws -> String {
        let resolved = resolve(path, preferredRoots: writableRoots)
        guard isAllowed(resolved, roots: writableRoots) else {
            deny(.writeFile, requestedPath: path, resolved: resolved, reason: "outside the writable sandbox")
            throw SandboxPolicyError.outsideWritableSandbox(resolved.path)
        }

        if dryRun {
            audit(
                .writeFile,
                requestedPath: path,
                resolved: resolved,
                decision: .dryRun,
                byteCount: content.utf8.count,
                reason: "dry-run enabled"
            )
            throw SandboxPolicyError.dryRunWrite(resolved.path)
        }

        do {
            try FileManager.default.createDirectory(
                at: resolved.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
            try content.write(to: resolved, atomically: true, encoding: .utf8)
        } catch {
            audit(
                .writeFile,
                requestedPath: path,
                resolved: resolved,
                decision: .denied,
                byteCount: content.utf8.count,
                reason: error.localizedDescription
            )
            throw error
        }
        audit(
            .writeFile,
            requestedPath: path,
            resolved: resolved,
            decision: .allowed,
            byteCount: content.utf8.count
        )
        return "Wrote \(content.utf8.count) bytes to \(resolved.path)"
    }

    private func deny(
        _ operation: SandboxOperation,
        requestedPath: String,
        resolved: URL,
        reason: String
    ) {
        audit(operation, requestedPath: requestedPath, resolved: resolved, decision: .denied, reason: reason)
    }

    private func audit(
        _ operation: SandboxOperation,
        requestedPath: String,
        resolved: URL,
        decision: SandboxDecision,
        byteCount: Int? = nil,
        reason: String? = nil
    ) {
        auditSink(
            AuditEvent(
                operation: operation,
                requestedPath: requestedPath,
                resolvedPath: resolved.path,
                decision: decision,
                byteCount: byteCount,
                reason: reason
            )
        )
    }

    private func resolve(_ path: String, preferredRoots: [URL]) -> URL {
        let rawURL: URL
        if NSString(string: path).isAbsolutePath {
            rawURL = URL(fileURLWithPath: path)
        } else if let root = preferredRoots.first ?? readableRoots.first {
            rawURL = root.appendingPathComponent(path)
        } else {
            rawURL = URL(fileURLWithPath: path)
        }
        return Self.canonicalizeByResolvingExistingAncestor(rawURL)
    }

    private func isAllowed(_ url: URL, roots: [URL]) -> Bool {
        roots.contains { root in
            let rootPath = root.path
            let path = url.path
            return path == rootPath || path.hasPrefix(rootPath + "/")
        }
    }

    private func isRegularFile(_ url: URL) -> Bool {
        (try? url.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile) == true
    }

    private func isDirectory(_ url: URL) -> Bool {
        (try? url.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true
    }

    private func readDataPrefix(from url: URL) throws -> Data {
        let handle = try FileHandle(forReadingFrom: url)
        defer {
            try? handle.close()
        }
        return try handle.read(upToCount: maxReadBytes + 1) ?? Data()
    }

    private static func canonicalRoot(_ url: URL) -> URL {
        canonicalizeByResolvingExistingAncestor(url)
    }

    private static func canonicalizeByResolvingExistingAncestor(_ url: URL) -> URL {
        var ancestor = url.standardizedFileURL
        var unresolvedComponents: [String] = []

        while !FileManager.default.fileExists(atPath: ancestor.path) {
            let parent = ancestor.deletingLastPathComponent()
            if parent.path == ancestor.path {
                break
            }
            unresolvedComponents.insert(ancestor.lastPathComponent, at: 0)
            ancestor = parent
        }

        let resolvedAncestor = ancestor.standardizedFileURL.resolvingSymlinksInPath()
        let resolved = unresolvedComponents.reduce(resolvedAncestor) { partial, component in
            partial.appendingPathComponent(component)
        }
        return resolved.standardizedFileURL
    }
}
