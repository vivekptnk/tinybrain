/// Model Picker View
///
/// Native popover for switching between the built-in toy model and discovered
/// `.tbf` files. The header chip is driven by the active runner identity, not
/// by a default picker selection.

import SwiftUI

// MARK: - ModelPickerView

/// Header picker component for switching between available models.
public struct ModelPickerView: View {
    @ObservedObject var pickerVM: ModelPickerViewModel

    let activeModelName: String
    let activeQuant: QuantBadge
    let activeModelPath: String?
    let switchingModelPath: String?
    let isSwitching: Bool

    /// Called when the user confirms a model selection.
    var onSelect: (ModelInfo?) -> Void

    @State private var isPresented = false
    private let theme = TinyBrainTheme.shared

    public init(
        pickerVM: ModelPickerViewModel,
        activeModelName: String,
        activeQuant: QuantBadge,
        activeModelPath: String?,
        switchingModelPath: String?,
        isSwitching: Bool,
        initialShowPicker: Bool = false,
        onSelect: @escaping (ModelInfo?) -> Void
    ) {
        self._pickerVM = ObservedObject(wrappedValue: pickerVM)
        self.activeModelName = activeModelName
        self.activeQuant = activeQuant
        self.activeModelPath = activeModelPath
        self.switchingModelPath = switchingModelPath
        self.isSwitching = isSwitching
        self.onSelect = onSelect
        _isPresented = State(initialValue: initialShowPicker)
    }

    public var body: some View {
        Button {
            pickerVM.refresh()
            isPresented.toggle()
        } label: {
            chipLabel
        }
        .buttonStyle(.plain)
        .disabled(isSwitching)
        .popover(isPresented: $isPresented, arrowEdge: .bottom) {
            popoverContent
        }
        .onAppear {
            pickerVM.refresh()
        }
    }

    // MARK: - Chip

    private var chipLabel: some View {
        HStack(spacing: theme.spacing.eight) {
            Image(systemName: "cpu")
                .font(.system(size: 12, weight: .medium))
                .foregroundStyle(theme.colors.textSecondary)

            Text(activeModelName)
                .font(theme.typography.label)
                .foregroundStyle(theme.colors.textPrimary)
                .lineLimit(1)
                .truncationMode(.middle)

            quantBadge(activeQuant)

            Image(systemName: "chevron.down")
                .font(.system(size: 9, weight: .medium))
                .foregroundStyle(theme.colors.textSecondary)
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 5)
        .frame(minHeight: 26)
        .background(theme.colors.fillQuaternary)
        .clipShape(RoundedRectangle(cornerRadius: theme.corners.small, style: .continuous))
    }

    // MARK: - Popover

    private var popoverContent: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text("Models")
                    .font(theme.typography.title2)
                    .foregroundStyle(theme.colors.textPrimary)

                Spacer()

                Button {
                    pickerVM.refresh()
                } label: {
                    Image(systemName: "arrow.clockwise")
                        .font(.system(size: 13, weight: .medium))
                        .foregroundStyle(theme.colors.textSecondary)
                        .frame(width: 24, height: 24)
                }
                .buttonStyle(.plain)
                .help("Refresh models")
            }

            VStack(spacing: 4) {
                modelRow(
                    title: "Toy Model",
                    subtitle: "Built-in · untrained",
                    badge: .toy,
                    isActive: activeModelPath == nil,
                    isLoading: isSwitching && switchingModelPath == nil
                ) {
                    choose(nil)
                }

                if pickerVM.availableModels.isEmpty {
                    Text("No `.tbf` models in `Models/`. Add one and Refresh.")
                        .font(theme.typography.caption)
                        .foregroundStyle(theme.colors.textSecondary)
                        .fixedSize(horizontal: false, vertical: true)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 8)
                } else {
                    ForEach(pickerVM.availableModels) { model in
                        let badge = QuantBadge(hint: model.quantization)
                        modelRow(
                            title: model.displayName,
                            subtitle: modelSubtitle(for: model, badge: badge),
                            badge: badge,
                            isActive: activeModelPath == model.path,
                            isLoading: isSwitching && switchingModelPath == model.path
                        ) {
                            choose(model)
                        }
                    }
                }
            }
        }
        .padding(12)
        .frame(width: 300)
        .background(.regularMaterial)
    }

    @ViewBuilder
    private func modelRow(
        title: String,
        subtitle: String,
        badge: QuantBadge,
        isActive: Bool,
        isLoading: Bool,
        action: @escaping () -> Void
    ) -> some View {
        Button(action: action) {
            HStack(spacing: 8) {
                quantBadge(badge)

                VStack(alignment: .leading, spacing: 2) {
                    Text(title)
                        .font(theme.typography.label)
                        .foregroundStyle(theme.colors.textPrimary)
                        .lineLimit(1)
                        .truncationMode(.middle)

                    Text(subtitle)
                        .font(theme.typography.caption)
                        .foregroundStyle(theme.colors.textSecondary)
                        .lineLimit(1)
                }

                Spacer()

                if isLoading {
                    ProgressView()
                        .scaleEffect(0.62)
                        .frame(width: 16, height: 16)
                } else if isActive {
                    Image(systemName: "checkmark")
                        .font(.system(size: 12, weight: .semibold))
                        .foregroundStyle(theme.colors.accent)
                }
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 8)
            .background(isActive ? theme.colors.accentQuiet : Color.clear)
            .clipShape(RoundedRectangle(cornerRadius: theme.corners.small, style: .continuous))
        }
        .buttonStyle(.plain)
        .disabled(isSwitching)
    }

    private func choose(_ model: ModelInfo?) {
        isPresented = false
        onSelect(model)
    }

    private func modelSubtitle(for model: ModelInfo, badge: QuantBadge) -> String {
        [badge.rawValue, model.formattedSize, model.interactionHint]
            .compactMap { $0 }
            .joined(separator: " · ")
    }

    private func quantBadge(_ badge: QuantBadge) -> some View {
        let tint = theme.colors.badgeTint(badge)
        return Text(badge.rawValue)
            .font(theme.typography.monoSM)
            .foregroundStyle(tint)
            .padding(.horizontal, 4)
            .padding(.vertical, 2)
            .background(tint.opacity(0.14))
            .clipShape(RoundedRectangle(cornerRadius: 3, style: .continuous))
    }
}

// MARK: - Preview

#if DEBUG
struct ModelPickerView_Previews: PreviewProvider {
    static var previews: some View {
        let vm = ModelPickerViewModel()
        ModelPickerView(
            pickerVM: vm,
            activeModelName: "Toy Model",
            activeQuant: .toy,
            activeModelPath: nil,
            switchingModelPath: nil,
            isSwitching: false,
            initialShowPicker: false
        ) { _ in }
        .padding()
    }
}
#endif
