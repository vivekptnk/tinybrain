/// TinyBrain Design System
///
/// **TDD Phase: GREEN**
/// Implementation to satisfy design system tests.
///
/// A comprehensive design system providing:
/// - Semantic color palette with accessibility compliance
/// - Consistent spacing scale
/// - Typography hierarchy
/// - Platform-adaptive layouts
/// - Shadow styles and corner radii
///
/// **Philosophy:**
/// Clean, modern aesthetic that rivals top AI chat interfaces while maintaining
/// educational clarity and Apple platform conventions.

import SwiftUI

/// Centralized design system for TinyBrain Chat
public final class TinyBrainTheme {
    
    /// Shared singleton instance
    public static let shared = TinyBrainTheme()
    
    private init() {}
    
    // MARK: - Color System
    
    public var colors: Colors {
        Colors()
    }
    
    public struct Colors {
        // MARK: Semantic Colors
        
        /// Primary brand color - used for key UI elements
        public let primary = Color.primary
        
        /// Secondary text and icons
        public let secondary = Color.secondary
        
        /// Accent color for interactive elements and highlights
        public let accent = Color.accentColor
        
        /// Main background color
        public let background: Color = {
            #if os(iOS)
            return Color(uiColor: .systemBackground)
            #else
            return Color(nsColor: .windowBackgroundColor)
            #endif
        }()
        
        /// Surface color for cards and panels
        public let surface: Color = {
            #if os(iOS)
            return Color(uiColor: .secondarySystemBackground)
            #else
            return Color(nsColor: .controlBackgroundColor)
            #endif
        }()
        
        /// Error state color
        public let error = Color.red
        
        /// Success state color
        public let success = Color.green
        
        // MARK: Message Colors
        
        /// User message bubble background
        public let userMessageBackground = Color.accentColor.opacity(0.15)
        
        /// Assistant message bubble background
        public let assistantMessageBackground: Color = {
            #if os(iOS)
            return Color(uiColor: .tertiarySystemBackground)
            #else
            return Color(nsColor: .controlBackgroundColor).opacity(0.5)
            #endif
        }()
        
        // MARK: Telemetry Colors
        
        /// High confidence indicator (>80%)
        public let highConfidence = Color.green
        
        /// Medium confidence indicator (50-80%)
        public let mediumConfidence = Color.orange
        
        /// Low confidence indicator (<50%)
        public let lowConfidence = Color.red
    }
    
    // MARK: - Gradients
    
    public var gradients: Gradients {
        Gradients()
    }
    
    public struct Gradients {
        /// Header gradient for app branding
        public let header = LinearGradient(
            colors: [
                Color.accentColor.opacity(0.6),
                Color.accentColor.opacity(0.3)
            ],
            startPoint: .topLeading,
            endPoint: .bottomTrailing
        )
        
        /// Subtle background gradient
        public let background = LinearGradient(
            colors: [
                Color.clear,
                Color.accentColor.opacity(0.03)
            ],
            startPoint: .top,
            endPoint: .bottom
        )
        
        /// Glassmorphic effect for panels
        public let glass = LinearGradient(
            colors: [
                Color.white.opacity(0.1),
                Color.white.opacity(0.05)
            ],
            startPoint: .topLeading,
            endPoint: .bottomTrailing
        )
    }
    
    // MARK: - Spacing Scale
    
    public var spacing: Spacing {
        Spacing()
    }
    
    public struct Spacing {
        /// Extra small spacing (4pt)
        public let xs: CGFloat = 4
        
        /// Small spacing (8pt)
        public let sm: CGFloat = 8
        
        /// Medium spacing (16pt) - default
        public let md: CGFloat = 16
        
        /// Large spacing (24pt)
        public let lg: CGFloat = 24
        
        /// Extra large spacing (32pt)
        public let xl: CGFloat = 32
        
        /// Extra extra large spacing (48pt)
        public let xxl: CGFloat = 48
    }
    
    // MARK: - Typography
    
    public var typography: Typography {
        Typography()
    }
    
    public struct Typography {
        /// Display text style (32pt, bold)
        public let display: Font = .system(size: 32, weight: .bold, design: .rounded)
        
        /// Title text style (24pt, semibold)
        public let title: Font = .system(size: 24, weight: .semibold, design: .rounded)
        
        /// Headline text style (18pt, semibold)
        public let headline: Font = .system(size: 18, weight: .semibold, design: .default)
        
        /// Body text style (16pt, regular)
        public let body: Font = .system(size: 16, weight: .regular, design: .default)
        
        /// Caption text style (12pt, regular)
        public let caption: Font = .system(size: 12, weight: .regular, design: .default)
        
        /// Monospaced font for code/metrics
        public let monospace: Font = .system(size: 14, weight: .regular, design: .monospaced)
        
        // MARK: Size Accessors (for testing)
        
        public var displaySize: CGFloat { 32 }
        public var titleSize: CGFloat { 24 }
        public var headlineSize: CGFloat { 18 }
        public var bodySize: CGFloat { 16 }
        public var captionSize: CGFloat { 12 }
    }
    
    // MARK: - Corner Radii
    
    public var corners: CornerRadii {
        CornerRadii()
    }
    
    public struct CornerRadii {
        /// Small corner radius (6pt) - buttons, tags
        public let small: CGFloat = 6
        
        /// Medium corner radius (12pt) - cards, inputs
        public let medium: CGFloat = 12
        
        /// Large corner radius (20pt) - panels, modals
        public let large: CGFloat = 20
        
        /// Extra large corner radius (28pt) - special elements
        public let xlarge: CGFloat = 28
    }
    
    // MARK: - Shadows
    
    public var shadows: Shadows {
        Shadows()
    }
    
    public struct Shadows {
        /// Small shadow for subtle elevation
        public let small = ShadowStyle(
            color: Color.black.opacity(0.1),
            radius: 4,
            x: 0,
            y: 2
        )
        
        /// Medium shadow for cards
        public let medium = ShadowStyle(
            color: Color.black.opacity(0.15),
            radius: 8,
            x: 0,
            y: 4
        )
        
        /// Large shadow for modals
        public let large = ShadowStyle(
            color: Color.black.opacity(0.2),
            radius: 16,
            x: 0,
            y: 8
        )
    }
    
    public struct ShadowStyle {
        public let color: Color
        public let radius: CGFloat
        public let x: CGFloat
        public let y: CGFloat
    }
    
    // MARK: - Layout Constants
    
    public var layout: Layout {
        Layout()
    }
    
    public struct Layout {
        /// Minimum touch target size (platform-adaptive)
        public let minTouchTarget: CGFloat = {
            #if os(iOS)
            return 44 // Apple HIG for iOS
            #else
            return 32 // Reasonable for macOS
            #endif
        }()
        
        /// Maximum content width for readability
        public let maxContentWidth: CGFloat = 800
        
        /// Standard sidebar width
        public let sidebarWidth: CGFloat = 280
        
        /// Message bubble max width (as fraction of container)
        public let messageBubbleMaxWidthFraction: CGFloat = 0.75
        
        /// Standard animation duration
        public let animationDuration: Double = 0.3
        
        /// Quick animation duration
        public let quickAnimationDuration: Double = 0.15
        
        /// Slow animation duration
        public let slowAnimationDuration: Double = 0.6
    }
    
    // MARK: - Animation Curves
    
    public var animations: Animations {
        Animations()
    }
    
    public struct Animations {
        /// Standard spring animation
        public let spring = Animation.spring(response: 0.3, dampingFraction: 0.7, blendDuration: 0)
        
        /// Bouncy spring for playful interactions
        public let bouncy = Animation.spring(response: 0.4, dampingFraction: 0.6, blendDuration: 0)
        
        /// Smooth ease in/out
        public let smooth = Animation.easeInOut(duration: 0.3)
        
        /// Quick ease for micro-interactions
        public let quick = Animation.easeOut(duration: 0.15)
        
        /// Gentle ease for large movements
        public let gentle = Animation.easeInOut(duration: 0.6)
    }
}

// MARK: - View Modifiers

public extension View {
    /// Apply TinyBrain card style
    func tinyBrainCard() -> some View {
        let theme = TinyBrainTheme.shared
        return self
            .background(theme.colors.surface)
            .cornerRadius(theme.corners.medium)
            .shadow(
                color: theme.shadows.small.color,
                radius: theme.shadows.small.radius,
                x: theme.shadows.small.x,
                y: theme.shadows.small.y
            )
    }
    
    /// Apply glassmorphic panel style
    func glassmorphicPanel() -> some View {
        let theme = TinyBrainTheme.shared
        return self
            .background(
                theme.gradients.glass
                    .background(.ultraThinMaterial)
            )
            .cornerRadius(theme.corners.medium)
    }
}

