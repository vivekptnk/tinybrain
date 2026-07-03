import Foundation
import ProximaKit

/// Deterministic bag-of-words embedder used by hermetic RAG tests.
///
/// The projection hashes normalized word tokens into a fixed vector with a
/// caller-provided seed. Texts sharing words land in similar directions, which
/// is enough to assert retrieval ordering without model files or network calls.
public struct DeterministicStubEmbedder: TextEmbedder {
    public let dimension: Int
    public let seed: UInt64

    public init(dimension: Int = 64, seed: UInt64 = 0) {
        precondition(dimension > 0, "dimension must be positive")
        self.dimension = dimension
        self.seed = seed
    }

    public func embed(_ text: String) async throws -> Vector {
        var components = [Float](repeating: 0, count: dimension)
        let words = normalizedWords(in: text)

        if words.isEmpty {
            return Vector(components)
        }

        for word in words {
            let hash = fnv1a64(word, seed: seed)
            let index = Int(hash % UInt64(dimension))
            let sign: Float = ((hash >> 63) == 0) ? 1 : -1
            components[index] += sign
        }

        let magnitude = sqrt(components.reduce(Float(0)) { $0 + $1 * $1 })
        if magnitude > 0 {
            components = components.map { $0 / magnitude }
        }
        return Vector(components)
    }

    private func normalizedWords(in text: String) -> [String] {
        text.lowercased()
            .split { !$0.isLetter && !$0.isNumber }
            .map(String.init)
    }

    private func fnv1a64(_ string: String, seed: UInt64) -> UInt64 {
        var hash: UInt64 = 0xcbf2_9ce4_8422_2325 ^ seed
        for byte in string.utf8 {
            hash ^= UInt64(byte)
            hash &*= 0x0000_0100_0000_01b3
        }
        return hash
    }
}
