/// Message Model
///
/// **TDD Phase: GREEN**
/// Represents a chat message in the conversation.
///
/// Features:
/// - Role identification (user/assistant/system)
/// - Unique identification
/// - Timestamp tracking
/// - Content storage

import Foundation

/// Represents a single message in the chat conversation
public struct Message: Identifiable {
    
    /// Unique identifier for the message
    public let id: UUID
    
    /// Role of the message sender
    public let role: MessageRole
    
    /// Message content text
    public let content: String
    
    /// When the message was created
    public let timestamp: Date
    
    /// Initialize a new message
    public init(role: MessageRole, content: String, timestamp: Date = Date()) {
        self.id = UUID()
        self.role = role
        self.content = content
        self.timestamp = timestamp
    }
    
    // MARK: - Convenience Properties
    
    /// Whether this is a user message
    public var isUser: Bool {
        role == .user
    }
    
    /// Whether this is an assistant message
    public var isAssistant: Bool {
        role == .assistant
    }
    
    /// Whether this is a system message
    public var isSystem: Bool {
        role == .system
    }
}

/// Message sender role
public enum MessageRole: String, Codable {
    case user
    case assistant
    case system
}

// MARK: - Codable Conformance

extension Message: Codable {}

// MARK: - Equatable (for testing)

extension Message: Equatable {
    public static func == (lhs: Message, rhs: Message) -> Bool {
        lhs.id == rhs.id &&
        lhs.role == rhs.role &&
        lhs.content == rhs.content &&
        lhs.timestamp == rhs.timestamp
    }
}

