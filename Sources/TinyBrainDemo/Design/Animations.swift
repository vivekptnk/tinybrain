/// Animation Utilities for TinyBrain Chat
///
/// Provides smooth, polished animations for:
/// - Typewriter text streaming effect
/// - Message appearance
/// - Panel transitions
/// - Interactive feedback
///
/// **Performance:** All animations are optimized for 60 FPS on iOS/macOS

import SwiftUI

// MARK: - Typewriter Effect

/// Manages typewriter effect for streaming text
@MainActor
public final class TypewriterEffect: ObservableObject {
    @Published public private(set) var displayedText: String = ""
    @Published public private(set) var isAnimating: Bool = false
    
    private var fullText: String = ""
    private var currentIndex: Int = 0
    private var timer: Timer?
    
    /// Characters per second for typewriter effect
    public var speed: Double = 30.0 // ~33ms per character
    
    public init() {}
    
    /// Start animating the given text
    public func animate(_ text: String) {
        stop()
        fullText = text
        currentIndex = 0
        displayedText = ""
        isAnimating = true
        
        startTimer()
    }
    
    /// Append new text and continue animation
    public func append(_ text: String) {
        fullText += text
        if !isAnimating && !fullText.isEmpty {
            isAnimating = true
            startTimer()
        }
    }
    
    /// Skip animation and show all text immediately
    public func skipAnimation() {
        stop()
        displayedText = fullText
    }
    
    /// Stop animation
    public func stop() {
        timer?.invalidate()
        timer = nil
        isAnimating = false
    }
    
    /// Reset to empty state
    public func reset() {
        stop()
        fullText = ""
        currentIndex = 0
        displayedText = ""
    }
    
    private func startTimer() {
        let interval = 1.0 / speed
        
        // Use MainActor.run to ensure thread-safety for @MainActor properties
        timer = Timer.scheduledTimer(withTimeInterval: interval, repeats: true) { [weak self] _ in
            Task { @MainActor [weak self] in
                guard let self = self else { return }
                
                if self.currentIndex < self.fullText.count {
                    let index = self.fullText.index(
                        self.fullText.startIndex,
                        offsetBy: self.currentIndex
                    )
                    self.displayedText.append(self.fullText[index])
                    self.currentIndex += 1
                } else {
                    self.stop()
                }
            }
        }
    }
}

// MARK: - Transition Extensions

public extension AnyTransition {
    /// Slide and fade transition for messages
    static var messageAppear: AnyTransition {
        .asymmetric(
            insertion: .move(edge: .bottom).combined(with: .opacity),
            removal: .opacity
        )
    }
    
    /// Scale and fade for panels
    static var panelExpand: AnyTransition {
        .scale(scale: 0.95).combined(with: .opacity)
    }
    
    /// Smooth slide for sidebars
    static func slideFromEdge(_ edge: Edge) -> AnyTransition {
        .move(edge: edge).combined(with: .opacity)
    }
}

// MARK: - View Modifiers

/// Pulsing animation for active generation indicator
public struct PulsingModifier: ViewModifier {
    @State private var isPulsing = false
    let minOpacity: Double
    let maxOpacity: Double
    let duration: Double
    
    public init(minOpacity: Double = 0.4, maxOpacity: Double = 1.0, duration: Double = 1.0) {
        self.minOpacity = minOpacity
        self.maxOpacity = maxOpacity
        self.duration = duration
    }
    
    public func body(content: Content) -> some View {
        content
            .opacity(isPulsing ? maxOpacity : minOpacity)
            .animation(
                .easeInOut(duration: duration).repeatForever(autoreverses: true),
                value: isPulsing
            )
            .onAppear {
                isPulsing = true
            }
    }
}

/// Shimmer effect for loading states
public struct ShimmerModifier: ViewModifier {
    @State private var phase: CGFloat = 0
    
    public func body(content: Content) -> some View {
        content
            .overlay(
                LinearGradient(
                    colors: [
                        .clear,
                        .white.opacity(0.3),
                        .clear
                    ],
                    startPoint: .leading,
                    endPoint: .trailing
                )
                .offset(x: phase)
                .mask(content)
            )
            .onAppear {
                withAnimation(.linear(duration: 1.5).repeatForever(autoreverses: false)) {
                    phase = 300
                }
            }
    }
}

/// Bounce effect for button presses
public struct BounceModifier: ViewModifier {
    @State private var isPressed = false
    
    public func body(content: Content) -> some View {
        content
            .scaleEffect(isPressed ? 0.95 : 1.0)
            .animation(.spring(response: 0.3, dampingFraction: 0.6), value: isPressed)
            .simultaneousGesture(
                DragGesture(minimumDistance: 0)
                    .onChanged { _ in
                        if !isPressed {
                            isPressed = true
                        }
                    }
                    .onEnded { _ in
                        isPressed = false
                    }
            )
    }
}

public extension View {
    /// Apply pulsing animation
    func pulsing(minOpacity: Double = 0.4, maxOpacity: Double = 1.0, duration: Double = 1.0) -> some View {
        modifier(PulsingModifier(minOpacity: minOpacity, maxOpacity: maxOpacity, duration: duration))
    }
    
    /// Apply shimmer loading effect
    func shimmer() -> some View {
        modifier(ShimmerModifier())
    }
    
    /// Apply bounce press effect
    func bounceOnPress() -> some View {
        modifier(BounceModifier())
    }
    
    /// Animate appearance with message transition
    func messageTransition() -> some View {
        transition(.messageAppear)
    }
}

// MARK: - Custom Shapes

/// Rounded message bubble shape with tail (optional)
public struct MessageBubbleShape: Shape {
    let hastail: Bool
    let tailPosition: Edge
    let cornerRadius: CGFloat
    
    public init(hasTail: Bool = false, tailPosition: Edge = .leading, cornerRadius: CGFloat = 12) {
        self.hastail = hasTail
        self.tailPosition = tailPosition
        self.cornerRadius = cornerRadius
    }
    
    public func path(in rect: CGRect) -> Path {
        // Simple rounded rectangle for now
        // Can be enhanced with actual tail in refinement phase
        let path = Path(roundedRect: rect, cornerRadius: cornerRadius)
        return path
    }
}

// MARK: - Haptic Feedback (iOS)

#if os(iOS)
import UIKit

public enum HapticFeedback {
    case light
    case medium
    case heavy
    case success
    case warning
    case error
    case selection
    
    public func trigger() {
        switch self {
        case .light:
            UIImpactFeedbackGenerator(style: .light).impactOccurred()
        case .medium:
            UIImpactFeedbackGenerator(style: .medium).impactOccurred()
        case .heavy:
            UIImpactFeedbackGenerator(style: .heavy).impactOccurred()
        case .success:
            UINotificationFeedbackGenerator().notificationOccurred(.success)
        case .warning:
            UINotificationFeedbackGenerator().notificationOccurred(.warning)
        case .error:
            UINotificationFeedbackGenerator().notificationOccurred(.error)
        case .selection:
            UISelectionFeedbackGenerator().selectionChanged()
        }
    }
}
#else
// Placeholder for macOS (no haptics)
public enum HapticFeedback {
    case light, medium, heavy, success, warning, error, selection
    
    public func trigger() {
        // No-op on macOS
    }
}
#endif

