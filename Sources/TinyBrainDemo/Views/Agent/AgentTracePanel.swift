/// Live trace panel for the TinyBrain Agent plan-act-observe loop.

import SwiftUI

/// Sidebar-style Agent Trace panel, visually paired with `XRayPanel`.
public struct AgentTracePanel: View {
    @ObservedObject private var trace: AgentTraceViewModel
    private let preferredWidth: CGFloat?
    private let theme = TinyBrainTheme.shared

    /// Creates an Agent Trace panel.
    public init(trace: AgentTraceViewModel, preferredWidth: CGFloat? = 390) {
        self.trace = trace
        self.preferredWidth = preferredWidth
    }

    public var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: theme.spacing.sixteen) {
                panelHeader

                separator

                AgentTraceMetricStrip(metrics: trace.runMetrics, maxSteps: trace.maxSteps)

                if trace.isBudgetExhausted {
                    AgentBudgetBand()
                        .transition(.move(edge: .top).combined(with: .opacity))
                }

                separator

                timeline

                if let answer = trace.finalAnswer, !answer.isEmpty {
                    separator
                    finalAnswerTile(answer)
                        .transition(.move(edge: .bottom).combined(with: .opacity))
                }

                Spacer(minLength: 0)
            }
            .padding(theme.spacing.twelve)
        }
        .frame(width: preferredWidth)
        .background(.bar)
        .overlay(alignment: .leading) {
            Rectangle()
                .fill(theme.colors.hairline)
                .frame(width: 0.5)
        }
        .animation(theme.animations.quick, value: trace.steps)
        .animation(theme.animations.quick, value: trace.isBudgetExhausted)
        .animation(theme.animations.quick, value: trace.finalAnswer)
    }

    private var panelHeader: some View {
        HStack(spacing: theme.spacing.eight) {
            Image(systemName: "point.topleft.down.curvedto.point.bottomright.up")
                .font(.system(size: 15, weight: .medium))
                .foregroundStyle(theme.colors.accent)

            Text("Agent Trace")
                .font(theme.typography.title2)
                .foregroundStyle(theme.colors.textPrimary)

            Spacer()

            headerReadout

            AgentLivePill(isLive: trace.isRunning)
        }
    }

    @ViewBuilder
    private var headerReadout: some View {
        if trace.isRunning {
            TimelineView(.periodic(from: .now, by: 0.5)) { context in
                Text(runningReadout(now: context.date))
                    .font(theme.typography.monoSM)
                    .foregroundStyle(theme.colors.textSecondary)
                    .contentTransition(.numericText())
            }
        } else {
            Text(trace.finalAnswer == nil ? "idle" : "done")
                .font(theme.typography.monoSM)
                .foregroundStyle(theme.colors.textSecondary)
        }
    }

    private var separator: some View {
        Rectangle()
            .fill(theme.colors.hairline)
            .frame(height: 0.5)
    }

    @ViewBuilder
    private var timeline: some View {
        if trace.steps.isEmpty {
            AgentTraceEmptyState(isRunning: trace.isRunning)
        } else {
            VStack(alignment: .leading, spacing: theme.spacing.twelve) {
                sectionHeader("Timeline")

                VStack(alignment: .leading, spacing: theme.spacing.ten) {
                    ForEach(trace.steps) { step in
                        AgentTimelineStepRow(
                            step: step,
                            isActive: trace.isRunning && trace.activeStepIndex == step.index
                        )
                        .transition(.move(edge: .bottom).combined(with: .opacity))
                    }
                }
            }
        }
    }

    private func finalAnswerTile(_ answer: String) -> some View {
        VStack(alignment: .leading, spacing: theme.spacing.eight) {
            HStack {
                Text("FINAL")
                    .font(theme.typography.overline)
                    .tracking(0.6)
                    .foregroundStyle(theme.colors.textSecondary)
                Spacer()
                Text((trace.terminationReason ?? "complete").uppercased())
                    .font(theme.typography.monoSM)
                    .foregroundStyle(theme.colors.textTertiary)
            }

            Text(answer)
                .font(theme.typography.callout)
                .foregroundStyle(theme.colors.textPrimary)
                .lineLimit(4)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(theme.spacing.twelve)
        .background(theme.colors.fillQuaternary)
        .clipShape(RoundedRectangle(cornerRadius: theme.corners.medium, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: theme.corners.medium, style: .continuous)
                .stroke(theme.colors.accentHairline, lineWidth: 0.5)
        )
    }

    private func runningReadout(now: Date) -> String {
        let step = (trace.activeStepIndex ?? 0) + 1
        let elapsed: TimeInterval
        if let startedAt = trace.startedAt {
            elapsed = max(0, now.timeIntervalSince(startedAt))
        } else {
            elapsed = 0
        }
        return "step \(step)/\(trace.maxSteps) · \(String(format: "%.1fs", elapsed))"
    }
}

private struct AgentTraceMetricStrip: View {
    let metrics: AgentRunMetrics
    let maxSteps: Int
    private let theme = TinyBrainTheme.shared

    var body: some View {
        LazyVGrid(
            columns: [
                GridItem(.flexible(), spacing: theme.spacing.eight),
                GridItem(.flexible(), spacing: theme.spacing.eight)
            ],
            spacing: theme.spacing.eight
        ) {
            metricTile(label: "Steps", value: "\(metrics.steps)/\(maxSteps)", unit: "", icon: "point.3.connected.trianglepath.dotted")
            metricTile(label: "Prompt tok", value: "\(metrics.promptTokens)", unit: "tok", icon: "text.badge.checkmark")
            metricTile(label: "Tool ms", value: String(format: "%.0f", metrics.toolElapsedMs), unit: "ms", icon: "timer")
            metricTile(label: "Result tok", value: "\(metrics.resultTokens)", unit: "tok", icon: "arrow.down.doc")
        }
    }

    private func metricTile(label: String, value: String, unit: String, icon: String) -> some View {
        VStack(alignment: .leading, spacing: theme.spacing.six) {
            HStack(spacing: theme.spacing.six) {
                Image(systemName: icon)
                    .font(.system(size: 11, weight: .medium))
                    .foregroundStyle(theme.colors.textSecondary)
                    .frame(width: 14)
                Text(label)
                    .font(theme.typography.caption)
                    .foregroundStyle(theme.colors.textSecondary)
                    .lineLimit(1)
                Spacer(minLength: 0)
            }

            HStack(alignment: .firstTextBaseline, spacing: theme.spacing.four) {
                Text(value)
                    .font(theme.typography.metricValue)
                    .foregroundStyle(theme.colors.textPrimary)
                    .minimumScaleFactor(0.75)
                    .lineLimit(1)
                if !unit.isEmpty {
                    Text(unit)
                        .font(theme.typography.metricUnit)
                        .foregroundStyle(theme.colors.textSecondary)
                }
            }
        }
        .padding(theme.spacing.eight)
        .background(theme.colors.fillQuaternary)
        .clipShape(RoundedRectangle(cornerRadius: theme.corners.medium, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: theme.corners.medium, style: .continuous)
                .stroke(theme.colors.hairline, lineWidth: 0.5)
        )
    }
}

private struct AgentLivePill: View {
    let isLive: Bool
    private let theme = TinyBrainTheme.shared

    var body: some View {
        HStack(spacing: 5) {
            if isLive {
                Circle()
                    .fill(theme.colors.accent)
                    .frame(width: 5, height: 5)
                    .pulsing(minOpacity: 0.35, maxOpacity: 1.0, duration: 0.8)
            }

            Text(isLive ? "LIVE" : "IDLE")
                .font(theme.typography.monoSM)
                .foregroundStyle(isLive ? theme.colors.accent : theme.colors.textTertiary)
        }
        .padding(.horizontal, theme.spacing.eight)
        .padding(.vertical, theme.spacing.four)
        .background(theme.colors.fillQuaternary)
        .clipShape(Capsule())
    }
}

private struct AgentBudgetBand: View {
    private let theme = TinyBrainTheme.shared

    var body: some View {
        HStack(alignment: .center, spacing: theme.spacing.eight) {
            Image(systemName: "exclamationmark.triangle.fill")
                .font(.system(size: 12, weight: .medium))
                .foregroundStyle(theme.colors.warning)

            Text("Step budget reached - answering from gathered evidence.")
                .font(theme.typography.caption)
                .foregroundStyle(theme.colors.textSecondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(.horizontal, theme.spacing.ten)
        .padding(.vertical, theme.spacing.eight)
        .background(theme.colors.warning.opacity(0.12))
        .clipShape(RoundedRectangle(cornerRadius: theme.corners.small, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: theme.corners.small, style: .continuous)
                .stroke(theme.colors.warning.opacity(0.30), lineWidth: 0.5)
        )
    }
}

private struct AgentTraceEmptyState: View {
    let isRunning: Bool
    private let theme = TinyBrainTheme.shared

    var body: some View {
        VStack(alignment: .leading, spacing: theme.spacing.eight) {
            sectionHeader("Timeline")

            HStack(alignment: .center, spacing: theme.spacing.eight) {
                Circle()
                    .fill(isRunning ? theme.colors.accent : theme.colors.textQuaternary)
                    .frame(width: 7, height: 7)
                    .modifierIf(isRunning) { view in
                        view.pulsing(minOpacity: 0.35, maxOpacity: 1.0, duration: 0.8)
                    }

                Text(isRunning ? "Waiting for first planning event" : "Idle until an agent run starts.")
                    .font(theme.typography.caption)
                    .foregroundStyle(theme.colors.textTertiary)
            }
            .padding(theme.spacing.twelve)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(theme.colors.fillQuaternary)
            .clipShape(RoundedRectangle(cornerRadius: theme.corners.medium, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: theme.corners.medium, style: .continuous)
                    .stroke(theme.colors.hairline, lineWidth: 0.5)
            )
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
