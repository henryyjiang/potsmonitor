import SwiftUI

struct StatsView: View {
    @EnvironmentObject var tracker: PredictionTracker
    @State private var showClearConfirm = false

    var body: some View {
        NavigationStack {
            List {
                confusionMatrixSection
                metricsSection
                recentPredictionsSection
                clearSection
            }
            .navigationTitle("Prediction Stats")
            .alert("Clear Stats?", isPresented: $showClearConfirm) {
                Button("Clear", role: .destructive) { tracker.clearStats() }
                Button("Cancel", role: .cancel) {}
            } message: {
                Text("Resets all prediction history and accuracy statistics.")
            }
        }
    }

    // MARK: - Sections

    private var confusionMatrixSection: some View {
        Section {
            VStack(spacing: 10) {
                HStack(spacing: 8) {
                    Text("").frame(maxWidth: .infinity, alignment: .leading)
                    Text("Flareup\nOccurred")
                        .font(.caption.bold()).multilineTextAlignment(.center)
                        .frame(maxWidth: .infinity)
                    Text("No\nFlareup")
                        .font(.caption.bold()).multilineTextAlignment(.center)
                        .frame(maxWidth: .infinity)
                }
                HStack(spacing: 8) {
                    Text("Warning\nSent")
                        .font(.caption.bold()).multilineTextAlignment(.leading)
                        .frame(maxWidth: .infinity, alignment: .leading)
                    matrixCell(tracker.tp, label: "TP", color: .green)
                    matrixCell(tracker.fp, label: "FP", color: .red)
                }
                HStack(spacing: 8) {
                    Text("No\nWarning")
                        .font(.caption.bold()).multilineTextAlignment(.leading)
                        .frame(maxWidth: .infinity, alignment: .leading)
                    matrixCell(tracker.fn, label: "FN", color: .orange)
                    matrixCell(tracker.tn, label: "TN", color: .green)
                }
            }
            .padding(.vertical, 4)
        } header: {
            Text("Confusion Matrix")
        } footer: {
            Text("TN estimated from tracked days with no events. Pending predictions resolve after 15 minutes.")
        }
    }

    private var metricsSection: some View {
        Section("Accuracy Metrics") {
            metricRow("Precision", value: tracker.precision,
                      detail: "Of warnings sent, % correct")
            metricRow("Recall",    value: tracker.recall,
                      detail: "Of flareups, % predicted in advance")
            metricRow("F1 Score",  value: tracker.f1,
                      detail: "Harmonic mean of precision and recall")
        }
    }

    @ViewBuilder
    private var recentPredictionsSection: some View {
        let recent = Array(tracker.records.suffix(10).reversed())
        if !recent.isEmpty {
            Section("Recent Predictions") {
                ForEach(recent) { record in
                    HStack(spacing: 10) {
                        outcomeIcon(record.outcome)
                        VStack(alignment: .leading, spacing: 2) {
                            Text(record.timestamp.formatted(date: .abbreviated, time: .shortened))
                                .font(.body)
                            Text("\(Int(record.probability * 100))% probability")
                                .font(.caption).foregroundColor(.secondary)
                        }
                        Spacer()
                        Text(outcomeName(record.outcome))
                            .font(.caption.bold())
                            .foregroundColor(outcomeColor(record.outcome))
                    }
                }
            }
        }
    }

    private var clearSection: some View {
        Section {
            Button(role: .destructive) {
                showClearConfirm = true
            } label: {
                Label("Clear Prediction History", systemImage: "trash")
            }
        }
    }

    // MARK: - Helper Views

    private func matrixCell(_ value: Int, label: String, color: Color) -> some View {
        VStack(spacing: 3) {
            Text("\(value)")
                .font(.title2.bold())
                .foregroundColor(value == 0 ? .secondary : color)
            Text(label)
                .font(.caption2.bold())
                .foregroundColor(.secondary)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 10)
        .background((value == 0 ? Color.secondary : color).opacity(0.08))
        .cornerRadius(8)
    }

    private func metricRow(_ name: String, value: Double?, detail: String) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack {
                Text(name)
                Spacer()
                if let v = value {
                    Text(String(format: "%.1f%%", v * 100))
                        .foregroundColor(.secondary)
                } else {
                    Text("—")
                        .foregroundColor(.secondary)
                }
            }
            Text(detail)
                .font(.caption).foregroundColor(.secondary)
        }
    }

    private func outcomeIcon(_ outcome: PredictionRecord.Outcome) -> some View {
        let (name, color): (String, Color) = switch outcome {
        case .truePositive:  ("checkmark.circle.fill", .green)
        case .falsePositive: ("xmark.circle.fill",     .red)
        case .pending:       ("clock.circle.fill",     .orange)
        }
        return Image(systemName: name)
            .foregroundColor(color)
            .font(.body)
    }

    private func outcomeName(_ outcome: PredictionRecord.Outcome) -> String {
        switch outcome {
        case .truePositive:  "TP"
        case .falsePositive: "FP"
        case .pending:       "Pending"
        }
    }

    private func outcomeColor(_ outcome: PredictionRecord.Outcome) -> Color {
        switch outcome {
        case .truePositive:  .green
        case .falsePositive: .red
        case .pending:       .orange
        }
    }
}
