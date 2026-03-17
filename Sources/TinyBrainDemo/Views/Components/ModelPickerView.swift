/// Model Picker View
///
/// A dropdown menu that lists available `.tbf` models found in the `Models/`
/// directory. Placed in the ChatView header so users can switch models
/// without restarting the app.
///
/// When the user selects a model:
/// 1. The picker calls back to the ChatViewModel's `switchModel(_:)` method
/// 2. ChatViewModel loads the new weights + tokenizer and resets the runner
///
/// If the `Models/` directory is empty or unavailable, the picker shows
/// "Toy Model" and is non-interactive (greyed out).

import SwiftUI

// MARK: - ModelPickerView

/// Header picker component for switching between available models
public struct ModelPickerView: View {

    // MARK: - Properties

    @ObservedObject var pickerVM: ModelPickerViewModel

    /// Called when the user confirms a model selection
    var onSelect: (ModelInfo?) -> Void

    // MARK: - Private State

    @State private var isExpanded = false

    // MARK: - Body

    public var body: some View {
        HStack(spacing: 6) {
            Image(systemName: "cube.box")
                .font(.system(size: 12, weight: .medium))
                .foregroundStyle(.secondary)

            Menu {
                // Toy model option
                Button {
                    onSelect(nil)
                } label: {
                    Label("Toy Model", systemImage: "testtube.2")
                }

                if !pickerVM.availableModels.isEmpty {
                    Divider()

                    ForEach(pickerVM.availableModels) { model in
                        Button {
                            onSelect(model)
                        } label: {
                            VStack(alignment: .leading, spacing: 2) {
                                Text(model.displayName)
                                Text("\(model.quantization.rawValue) · \(model.formattedSize)")
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }
                        }
                    }

                    Divider()
                }

                Button {
                    pickerVM.refresh()
                } label: {
                    Label("Refresh Model List", systemImage: "arrow.clockwise")
                }
            } label: {
                modelLabel
            }
            .menuStyle(.borderlessButton)
            .fixedSize()

            if pickerVM.isSwitching {
                ProgressView()
                    .scaleEffect(0.6)
                    .frame(width: 12, height: 12)
            }
        }
        .onAppear {
            pickerVM.refresh()
        }
    }

    // MARK: - Subviews

    private var modelLabel: some View {
        HStack(spacing: 4) {
            Text(pickerVM.selectedDisplayName)
                .font(.system(size: 12, weight: .medium))
                .foregroundStyle(.primary)
                .lineLimit(1)
                .truncationMode(.middle)

            if let model = pickerVM.selectedModel {
                Text(model.quantization.rawValue)
                    .font(.system(size: 9, weight: .semibold))
                    .padding(.horizontal, 4)
                    .padding(.vertical, 2)
                    .background(quantizationColor(model.quantization).opacity(0.15))
                    .foregroundStyle(quantizationColor(model.quantization))
                    .clipShape(RoundedRectangle(cornerRadius: 3))
            }

            Image(systemName: "chevron.up.chevron.down")
                .font(.system(size: 9, weight: .medium))
                .foregroundStyle(.secondary)
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background(Color.primary.opacity(0.05))
        .clipShape(RoundedRectangle(cornerRadius: 6))
    }

    private func quantizationColor(_ q: QuantizationHint) -> Color {
        switch q {
        case .int4:    return .green
        case .int8:    return .blue
        case .fp16:    return .orange
        case .fp32:    return .red
        case .unknown: return .secondary
        }
    }
}

// MARK: - Preview

#if DEBUG
struct ModelPickerView_Previews: PreviewProvider {
    static var previews: some View {
        let vm = ModelPickerViewModel()
        ModelPickerView(pickerVM: vm) { _ in }
            .padding()
    }
}
#endif
