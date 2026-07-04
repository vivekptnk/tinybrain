/// Timeline tile for one TinyBrain Agent step.

import SwiftUI

/// Renders one plan, tool-call, observe, or terminal agent step.
public struct AgentStepTile: View {
    private let step: AgentVisibleStep
    private let isActive: Bool
    private let theme = TinyBrainTheme.shared

    @State private var showRawOutput = false
    @State private var showFullResult = false

    /// Creates a step tile.
    public init(step: AgentVisibleStep, isActive: Bool) {
        self.step = step
        self.isActive = isActive
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: theme.spacing.ten) {
            tileHeader

            stepBody
        }
        .padding(theme.spacing.twelve)
        .background(theme.colors.fillQuaternary)
        .clipShape(RoundedRectangle(cornerRadius: theme.corners.medium, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: theme.corners.medium, style: .continuous)
                .stroke(borderColor, lineWidth: 0.5)
        )
        .animation(theme.animations.quick, value: step.state)
        .animation(theme.animations.spring, value: step.toolName)
        .animation(theme.animations.quick, value: step.passages)
    }

    private var tileHeader: some View {
        HStack(spacing: theme.spacing.eight) {
            Text(String(format: "STEP %02d", step.index + 1))
                .font(theme.typography.overline)
                .tracking(0.6)
                .foregroundStyle(theme.colors.textSecondary)

            stateBadge

            Spacer(minLength: theme.spacing.four)

            HStack(spacing: theme.spacing.six) {
                Text("\(step.promptTokens)p")
                if let generatedTokens = step.generatedTokens {
                    Text("\(generatedTokens)g")
                }
                if let resultTokens = step.resultTokens {
                    Text("\(resultTokens)r")
                }
                elapsedText
            }
            .font(theme.typography.monoSM)
            .foregroundStyle(theme.colors.textTertiary)
        }
    }

    @ViewBuilder
    private var elapsedText: some View {
        if isActive && (step.state == .planning || step.state == .calling) {
            TimelineView(.periodic(from: .now, by: 0.5)) { context in
                Text(formatLiveElapsed(now: context.date))
                    .contentTransition(.numericText())
            }
        } else if let elapsedMs = step.elapsedMs {
            Text(formatMs(elapsedMs))
        }
    }

    private var stateBadge: some View {
        Text(step.state.badge)
            .font(theme.typography.monoSM)
            .foregroundStyle(stateColor)
            .padding(.horizontal, theme.spacing.eight)
            .padding(.vertical, theme.spacing.four)
            .background(stateColor.opacity(0.12))
            .clipShape(Capsule())
    }

    @ViewBuilder
    private var stepBody: some View {
        switch step.state {
        case .planning:
            planningBody
        case .calling, .observed, .error, .done:
            toolBody
        case .cancelled:
            Text(step.resultContent ?? "Cancelled before the step completed.")
                .font(theme.typography.caption)
                .foregroundStyle(theme.colors.textTertiary)
        }
    }

    private var planningBody: some View {
        VStack(alignment: .leading, spacing: theme.spacing.eight) {
            HStack(spacing: theme.spacing.eight) {
                Image(systemName: "sparkles")
                    .font(.system(size: 12, weight: .medium))
                    .foregroundStyle(theme.colors.accent)
                Text("Planning next action")
                    .font(theme.typography.callout)
                    .foregroundStyle(theme.colors.textSecondary)
            }

            shimmerLine(width: 0.88)
            shimmerLine(width: 0.62)
        }
    }

    private var toolBody: some View {
        VStack(alignment: .leading, spacing: theme.spacing.ten) {
            if let toolName = step.toolName {
                toolCapsule(toolName)
            }

            if step.toolName == "retrieve" {
                retrieveArgumentLabels
            }

            if let argumentsJSON = step.argumentsJSON {
                jsonBlock(argumentsJSON)
            }

            if let rawOutput = step.rawOutput, !rawOutput.isEmpty {
                DisclosureGroup(isExpanded: $showRawOutput) {
                    rawBlock(rawOutput)
                } label: {
                    Text("Raw")
                        .font(theme.typography.monoSM)
                        .foregroundStyle(theme.colors.textTertiary)
                }
            }

            if step.state == .calling {
                HStack(spacing: theme.spacing.eight) {
                    Circle()
                        .fill(theme.colors.accent)
                        .frame(width: 5, height: 5)
                        .pulsing(minOpacity: 0.35, maxOpacity: 1.0, duration: 0.8)
                    Text("Calling tool")
                        .font(theme.typography.caption)
                        .foregroundStyle(theme.colors.textTertiary)
                }
            }

            if step.isError, let content = step.resultContent {
                Text(content)
                    .font(theme.typography.caption)
                    .foregroundStyle(theme.colors.critical)
                    .fixedSize(horizontal: false, vertical: true)
            } else if !step.passages.isEmpty {
                passageRows
            } else if let content = step.resultContent, !content.isEmpty {
                DisclosureGroup(isExpanded: $showFullResult) {
                    rawBlock(content)
                } label: {
                    Text("Result")
                        .font(theme.typography.monoSM)
                        .foregroundStyle(theme.colors.textTertiary)
                }
            } else if step.state == .done {
                Text("Final answer assembled from gathered evidence.")
                    .font(theme.typography.caption)
                    .foregroundStyle(theme.colors.textTertiary)
            }
        }
    }

    @ViewBuilder
    private var retrieveArgumentLabels: some View {
        HStack(spacing: theme.spacing.eight) {
            if let query = step.liftedQuery {
                VStack(alignment: .leading, spacing: theme.spacing.two) {
                    Text("query")
                        .font(theme.typography.overline)
                        .tracking(0.6)
                        .foregroundStyle(theme.colors.textTertiary)
                    Text(query)
                        .font(theme.typography.caption)
                        .foregroundStyle(theme.colors.textSecondary)
                        .lineLimit(2)
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            }

            if let k = step.liftedK {
                VStack(alignment: .trailing, spacing: theme.spacing.two) {
                    Text("k")
                        .font(theme.typography.overline)
                        .tracking(0.6)
                        .foregroundStyle(theme.colors.textTertiary)
                    Text("\(k)")
                        .font(theme.typography.metricValue)
                        .foregroundStyle(theme.colors.textPrimary)
                }
                .frame(width: 32)
            }
        }
    }

    private var passageRows: some View {
        VStack(alignment: .leading, spacing: theme.spacing.eight) {
            ForEach(Array(step.passages.enumerated()), id: \.element.id) { offset, passage in
                RetrievedPassageRow(passage: passage)
                    .transition(.move(edge: .bottom).combined(with: .opacity))
                    .animation(theme.animations.spring.delay(Double(offset) * 0.05), value: step.passages)
            }
        }
    }

    private var borderColor: Color {
        switch step.state {
        case .error:
            return theme.colors.critical.opacity(0.38)
        case .calling, .planning:
            return theme.colors.accentHairline
        case .observed:
            return theme.colors.positive.opacity(0.28)
        case .done:
            return theme.colors.hairline
        case .cancelled:
            return theme.colors.warning.opacity(0.30)
        }
    }

    private var stateColor: Color {
        switch step.state {
        case .planning, .calling:
            return theme.colors.accent
        case .observed:
            return theme.colors.positive
        case .error:
            return theme.colors.critical
        case .done:
            return theme.colors.textSecondary
        case .cancelled:
            return theme.colors.warning
        }
    }

    private func toolCapsule(_ name: String) -> some View {
        HStack(spacing: theme.spacing.six) {
            Image(systemName: "wrench.and.screwdriver")
                .font(.system(size: 11, weight: .medium))
            Text(name)
                .font(theme.typography.monoSM)
        }
        .foregroundStyle(theme.colors.accent)
        .padding(.horizontal, theme.spacing.eight)
        .padding(.vertical, theme.spacing.four)
        .background(theme.colors.accentQuiet)
        .clipShape(Capsule())
    }

    private func jsonBlock(_ text: String) -> some View {
        Text(text)
            .font(theme.typography.monoSM)
            .foregroundStyle(theme.colors.textSecondary)
            .lineLimit(5)
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(theme.spacing.eight)
            .background(theme.colors.fillTertiary)
            .clipShape(RoundedRectangle(cornerRadius: theme.corners.small, style: .continuous))
            .textSelection(.enabled)
    }

    private func rawBlock(_ text: String) -> some View {
        Text(text)
            .font(theme.typography.monoSM)
            .foregroundStyle(theme.colors.textSecondary)
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.top, theme.spacing.four)
            .textSelection(.enabled)
    }

    private func shimmerLine(width: CGFloat) -> some View {
        GeometryReader { geometry in
            RoundedRectangle(cornerRadius: 3, style: .continuous)
                .fill(theme.colors.fillTertiary)
                .frame(width: geometry.size.width * width, height: 8)
                .shimmer()
        }
        .frame(height: 8)
    }

    private func formatMs(_ ms: Double) -> String {
        if ms >= 1_000 {
            return String(format: "%.1fs", ms / 1_000)
        }
        return String(format: "%.0fms", ms)
    }

    private func formatLiveElapsed(now: Date) -> String {
        String(format: "%.1fs", max(0, now.timeIntervalSince(step.startedAt)))
    }
}

/// Timeline row with the vertical accent rail node.
public struct AgentTimelineStepRow: View {
    private let step: AgentVisibleStep
    private let isActive: Bool
    private let theme = TinyBrainTheme.shared

    /// Creates a timeline row.
    public init(step: AgentVisibleStep, isActive: Bool) {
        self.step = step
        self.isActive = isActive
    }

    public var body: some View {
        HStack(alignment: .top, spacing: theme.spacing.eight) {
            VStack(spacing: 0) {
                node
                Rectangle()
                    .fill(theme.colors.accentHairline)
                    .frame(width: 1)
                    .frame(maxHeight: .infinity)
            }
            .frame(width: 24)

            AgentStepTile(step: step, isActive: isActive)
        }
    }

    private var node: some View {
        ZStack {
            Circle()
                .fill(nodeColor.opacity(isActive ? 0.18 : 0.10))
                .frame(width: 24, height: 24)
            Circle()
                .fill(nodeColor)
                .frame(width: 14, height: 14)
                .modifierIf(isActive) { view in
                    view.pulsing(minOpacity: 0.45, maxOpacity: 1.0, duration: 0.8)
                }
            Text("\(step.index + 1)")
                .font(.system(size: 8, weight: .semibold, design: .monospaced))
                .foregroundStyle(.white)
        }
    }

    private var nodeColor: Color {
        switch step.state {
        case .error:
            return theme.colors.critical
        case .observed:
            return theme.colors.positive
        case .cancelled:
            return theme.colors.warning
        case .done:
            return theme.colors.textSecondary
        case .planning, .calling:
            return theme.colors.accent
        }
    }
}

private extension TinyBrainTheme.Spacing {
    var six: CGFloat { 6 }
    var ten: CGFloat { 10 }
}

private extension View {
    @ViewBuilder
    func modifierIf<Content: View>(
        _ condition: Bool,
        transform: (Self) -> Content
    ) -> some View {
        if condition {
            transform(self)
        } else {
            self
        }
    }
}
