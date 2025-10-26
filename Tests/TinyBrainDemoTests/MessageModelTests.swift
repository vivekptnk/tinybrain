/// Message Model Tests
///
/// **TDD Phase: RED**
/// Tests define requirements for chat message representation.
///
/// Tests cover:
/// - Message creation
/// - Role identification
/// - Timestamp tracking
/// - Content storage

import XCTest
@testable import TinyBrainDemo

final class MessageModelTests: XCTestCase {
    
    // MARK: - Initialization Tests
    
    func testMessageInitialization() {
        let message = Message(role: .user, content: "Hello, TinyBrain!")
        
        XCTAssertEqual(message.role, .user, "Role should be user")
        XCTAssertEqual(message.content, "Hello, TinyBrain!", "Content should match")
        XCTAssertNotNil(message.id, "ID should be generated")
        XCTAssertNotNil(message.timestamp, "Timestamp should be set")
    }
    
    func testAssistantMessage() {
        let message = Message(role: .assistant, content: "Hello! How can I help?")
        
        XCTAssertEqual(message.role, .assistant, "Role should be assistant")
    }
    
    func testSystemMessage() {
        let message = Message(role: .system, content: "System initialized")
        
        XCTAssertEqual(message.role, .system, "Role should be system")
    }
    
    // MARK: - Identifiable Tests
    
    func testMessageIsIdentifiable() {
        let message1 = Message(role: .user, content: "First")
        let message2 = Message(role: .user, content: "Second")
        
        // Each message should have unique ID
        XCTAssertNotEqual(message1.id, message2.id, "Messages should have unique IDs")
    }
    
    // MARK: - Timestamp Tests
    
    func testTimestampIsRecent() {
        let beforeCreation = Date()
        let message = Message(role: .user, content: "Test")
        let afterCreation = Date()
        
        // Timestamp should be between before and after
        XCTAssertGreaterThanOrEqual(message.timestamp, beforeCreation, "Timestamp should be after creation started")
        XCTAssertLessThanOrEqual(message.timestamp, afterCreation, "Timestamp should be before creation ended")
    }
    
    // MARK: - Content Tests
    
    func testEmptyContent() {
        let message = Message(role: .user, content: "")
        
        XCTAssertEqual(message.content, "", "Empty content should be allowed")
    }
    
    func testLongContent() {
        let longText = String(repeating: "A", count: 10000)
        let message = Message(role: .assistant, content: longText)
        
        XCTAssertEqual(message.content.count, 10000, "Long content should be preserved")
    }
    
    func testMultilineContent() {
        let multiline = """
        Line 1
        Line 2
        Line 3
        """
        let message = Message(role: .assistant, content: multiline)
        
        XCTAssertTrue(message.content.contains("\n"), "Multiline content should preserve newlines")
    }
    
    // MARK: - Convenience Tests
    
    func testIsUserMessage() {
        let userMessage = Message(role: .user, content: "Test")
        let assistantMessage = Message(role: .assistant, content: "Test")
        
        XCTAssertTrue(userMessage.isUser, "User message should identify as user")
        XCTAssertFalse(assistantMessage.isUser, "Assistant message should not identify as user")
    }
    
    func testIsAssistantMessage() {
        let userMessage = Message(role: .user, content: "Test")
        let assistantMessage = Message(role: .assistant, content: "Test")
        
        XCTAssertFalse(userMessage.isAssistant, "User message should not identify as assistant")
        XCTAssertTrue(assistantMessage.isAssistant, "Assistant message should identify as assistant")
    }
}

