/// TinyBrain Design System
///
/// Instrument-grade native macOS tokens for the TinyBrain demo.

import SwiftUI

#if os(macOS)
import AppKit
#elseif os(iOS)
import UIKit
#endif

/// Centralized design system for TinyBrain Chat.
public final class TinyBrainTheme {

    /// Shared singleton instance.
    public static let shared = TinyBrainTheme()

    private init() {}

    // MARK: - Color System

    public var colors: Colors {
        Colors()
    }

    public struct Colors {
        public let accent: Color
        public let accentHover: Color
        public let accentQuiet: Color
        public let accentHairline: Color
        public let positive: Color
        public let warning: Color
        public let critical: Color
        public let canvas: Color
        public let chromeFallback: Color
        public let fillQuaternary: Color
        public let fillTertiary: Color
        public let hairline: Color
        public let textPrimary: Color
        public let textSecondary: Color
        public let textTertiary: Color
        public let textQuaternary: Color

        public init() {
            accent = .tbAdaptive(light: 0x2F54EB, dark: 0x6E8BFF)
            accentHover = .tbAdaptive(light: 0x1E40E0, dark: 0x8AA0FF)
            accentQuiet = .tbAdaptive(light: 0x2F54EB, lightAlpha: 0.12, dark: 0x6E8BFF, darkAlpha: 0.16)
            accentHairline = .tbAdaptive(light: 0x2F54EB, lightAlpha: 0.24, dark: 0x6E8BFF, darkAlpha: 0.30)
            positive = .tbAdaptive(light: 0x059669, dark: 0x34D399)
            warning = .tbAdaptive(light: 0xB45309, dark: 0xFBBF24)
            critical = .tbAdaptive(light: 0xDC2626, dark: 0xFF6B6B)
            canvas = .tbAdaptive(light: 0xFFFFFF, dark: 0x16171A)
            chromeFallback = .tbAdaptive(light: 0xF2F2F4, dark: 0x1D1E22)
            fillQuaternary = .tbAdaptive(light: 0x000000, lightAlpha: 0.04, dark: 0xFFFFFF, darkAlpha: 0.05)
            fillTertiary = .tbAdaptive(light: 0x000000, lightAlpha: 0.06, dark: 0xFFFFFF, darkAlpha: 0.08)
            hairline = .tbAdaptive(light: 0x000000, lightAlpha: 0.12, dark: 0xFFFFFF, darkAlpha: 0.10)
            textPrimary = .primary
            textSecondary = .secondary
            textTertiary = .tbTertiaryLabel
            textQuaternary = .tbQuaternaryLabel
        }

        /// Badge tint map from the redesign spec.
        public func badgeTint(_ badge: QuantBadge) -> Color {
            switch badge {
            case .int4:
                return positive
            case .int8:
                return accent
            case .fp16:
                return warning
            case .fp32:
                return critical
            case .toy, .unknown:
                return textSecondary
            }
        }

        // MARK: Compatibility aliases for existing demo tests.

        public var primary: Color { textPrimary }
        public var secondary: Color { textSecondary }
        public var background: Color { canvas }
        public var surface: Color { fillQuaternary }
        public var error: Color { critical }
        public var success: Color { positive }
        public var userMessageBackground: Color { accentQuiet }
        public var assistantMessageBackground: Color { fillQuaternary }
        public var highConfidence: Color { positive }
        public var mediumConfidence: Color { warning }
        public var lowConfidence: Color { critical }
    }

    // MARK: - Gradients

    public var gradients: Gradients {
        Gradients()
    }

    public struct Gradients {
        private let colors = TinyBrainTheme.shared.colors

        public var header: LinearGradient {
            LinearGradient(
                colors: [colors.accent.opacity(0.20), colors.accent.opacity(0.08)],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )
        }

        public var background: LinearGradient {
            LinearGradient(
                colors: [Color.clear, colors.accent.opacity(0.03)],
                startPoint: .top,
                endPoint: .bottom
            )
        }

        public var glass: LinearGradient {
            LinearGradient(
                colors: [Color.white.opacity(0.10), Color.white.opacity(0.05)],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )
        }
    }

    // MARK: - Spacing Scale

    public var spacing: Spacing {
        Spacing()
    }

    public struct Spacing {
        public let two: CGFloat = 2
        public let four: CGFloat = 4
        public let eight: CGFloat = 8
        public let twelve: CGFloat = 12
        public let sixteen: CGFloat = 16
        public let twenty: CGFloat = 20
        public let twentyFour: CGFloat = 24
        public let thirtyTwo: CGFloat = 32

        public let windowPadding: CGFloat = 16
        public let sidebarPadding: CGFloat = 16

        // Compatibility aliases.
        public var xxs: CGFloat { two }
        public var xs: CGFloat { four }
        public var sm: CGFloat { eight }
        public var md: CGFloat { sixteen }
        public var lg: CGFloat { twentyFour }
        public var xl: CGFloat { thirtyTwo }
        public var xxl: CGFloat { thirtyTwo }
    }

    // MARK: - Typography

    public var typography: Typography {
        Typography()
    }

    public struct Typography {
        public let wordmark: Font = .system(size: 15, weight: .semibold, design: .rounded)
        public let title1: Font = .system(size: 22, weight: .bold, design: .default)
        public let title2: Font = .system(size: 17, weight: .semibold, design: .default)
        public let body: Font = .system(size: 14, weight: .regular, design: .default)
        public let callout: Font = .system(size: 13, weight: .regular, design: .default)
        public let label: Font = .system(size: 12, weight: .medium, design: .default)
        public let caption: Font = .system(size: 11, weight: .regular, design: .default)
        public let overline: Font = .system(size: 10.5, weight: .semibold, design: .default)
        public let metricValue: Font = .system(size: 20, weight: .medium, design: .monospaced)
        public let metricUnit: Font = .system(size: 11, weight: .regular, design: .monospaced)
        public let monoSM: Font = .system(size: 11, weight: .regular, design: .monospaced)

        // Compatibility aliases.
        public var display: Font { title1 }
        public var title: Font { title2 }
        public var headline: Font { title2 }
        public var monospace: Font { monoSM }

        public var displaySize: CGFloat { 22 }
        public var titleSize: CGFloat { 17 }
        public var headlineSize: CGFloat { 17 }
        public var bodySize: CGFloat { 14 }
        public var captionSize: CGFloat { 11 }
    }

    // MARK: - Corner Radii

    public var corners: CornerRadii {
        CornerRadii()
    }

    public struct CornerRadii {
        public let small: CGFloat = 6
        public let medium: CGFloat = 10
        public let large: CGFloat = 14
        public let pill: CGFloat = 999

        // Compatibility aliases.
        public var xlarge: CGFloat { pill }
    }

    // MARK: - Shadows

    public var shadows: Shadows {
        Shadows()
    }

    public struct Shadows {
        public let small = ShadowStyle(color: Color.black.opacity(0.10), radius: 4, x: 0, y: 2)
        public let medium = ShadowStyle(color: Color.black.opacity(0.15), radius: 8, x: 0, y: 4)
        public let large = ShadowStyle(color: Color.black.opacity(0.20), radius: 16, x: 0, y: 8)
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
        public let minTouchTarget: CGFloat = {
            #if os(iOS)
            return 44
            #else
            return 32
            #endif
        }()

        public let maxContentWidth: CGFloat = 800
        public let sidebarWidth: CGFloat = 248
        public let xRayPanelWidth: CGFloat = 320
        public let headerHeight: CGFloat = 48
        public let animationDuration: Double = 0.3
        public let quickAnimationDuration: Double = 0.15
        public let slowAnimationDuration: Double = 0.6
        public let messageBubbleMaxWidthFraction: CGFloat = 0.72
    }

    // MARK: - Animation Curves

    public var animations: Animations {
        Animations()
    }

    public struct Animations {
        public let spring = Animation.spring(response: 0.3, dampingFraction: 0.7, blendDuration: 0)
        public let bouncy = Animation.spring(response: 0.4, dampingFraction: 0.6, blendDuration: 0)
        public let smooth = Animation.easeInOut(duration: 0.3)
        public let quick = Animation.easeOut(duration: 0.15)
        public let gentle = Animation.easeInOut(duration: 0.6)
    }
}

// MARK: - View Modifiers

public extension View {
    /// Apply TinyBrain tile style.
    func tinyBrainCard() -> some View {
        let theme = TinyBrainTheme.shared
        return self
            .background(theme.colors.fillQuaternary)
            .clipShape(RoundedRectangle(cornerRadius: theme.corners.medium, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: theme.corners.medium, style: .continuous)
                    .stroke(theme.colors.hairline, lineWidth: 0.5)
            )
    }

    /// Apply native material panel style.
    func glassmorphicPanel() -> some View {
        let theme = TinyBrainTheme.shared
        return self
            .background(.regularMaterial)
            .clipShape(RoundedRectangle(cornerRadius: theme.corners.large, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: theme.corners.large, style: .continuous)
                    .stroke(theme.colors.hairline, lineWidth: 0.5)
            )
    }
}

// MARK: - Adaptive Color Helpers

private extension Color {
    static func tbAdaptive(
        light: UInt32,
        lightAlpha: Double = 1,
        dark: UInt32,
        darkAlpha: Double = 1
    ) -> Color {
        #if os(macOS)
        return Color(nsColor: NSColor(name: nil) { appearance in
            let isDark = appearance.bestMatch(from: [.darkAqua, .aqua]) == .darkAqua
            return NSColor.tbHex(isDark ? dark : light, alpha: isDark ? darkAlpha : lightAlpha)
        })
        #elseif os(iOS)
        return Color(uiColor: UIColor { traits in
            let isDark = traits.userInterfaceStyle == .dark
            return UIColor.tbHex(isDark ? dark : light, alpha: isDark ? darkAlpha : lightAlpha)
        })
        #else
        return Color(red: 1, green: 1, blue: 1, opacity: darkAlpha)
        #endif
    }

    static var tbTertiaryLabel: Color {
        #if os(macOS)
        return Color(nsColor: .tertiaryLabelColor)
        #elseif os(iOS)
        return Color(uiColor: .tertiaryLabel)
        #else
        return .secondary
        #endif
    }

    static var tbQuaternaryLabel: Color {
        #if os(macOS)
        return Color(nsColor: .quaternaryLabelColor)
        #elseif os(iOS)
        return Color(uiColor: .quaternaryLabel)
        #else
        return .secondary.opacity(0.6)
        #endif
    }
}

#if os(macOS)
private extension NSColor {
    static func tbHex(_ value: UInt32, alpha: Double) -> NSColor {
        NSColor(
            red: CGFloat((value >> 16) & 0xFF) / 255.0,
            green: CGFloat((value >> 8) & 0xFF) / 255.0,
            blue: CGFloat(value & 0xFF) / 255.0,
            alpha: CGFloat(alpha)
        )
    }
}
#elseif os(iOS)
private extension UIColor {
    static func tbHex(_ value: UInt32, alpha: Double) -> UIColor {
        UIColor(
            red: CGFloat((value >> 16) & 0xFF) / 255.0,
            green: CGFloat((value >> 8) & 0xFF) / 255.0,
            blue: CGFloat(value & 0xFF) / 255.0,
            alpha: CGFloat(alpha)
        )
    }
}
#endif
