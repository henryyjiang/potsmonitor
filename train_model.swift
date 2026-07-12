#!/usr/bin/env swift
import Foundation
import CreateML

// Trains the bundled POTSFlareupModel on the 14 features the app actually
// computes at runtime (see FeatureEngine.swift / POTSPredictor.featureDict).
//
// Two methodology rules, both to keep the reported metric honest:
//   • Split by DAY. Sliding windows overlap (60 s wide, 10 s slide), so a
//     random split leaks near-duplicate rows across train/test. Holding out
//     whole days keeps each flareup episode entirely on one side.
//   • Oversample the minority class on the TRAIN side only — never on test.

let featuresURL = URL(fileURLWithPath: "/Users/paff/Desktop/Projects/POTSMonitor/features.csv")
let modelOut    = URL(fileURLWithPath: "/Users/paff/Desktop/Projects/POTSMonitor/POTSFlareupModel.mlmodel")

// 15 model features, matching FeatureEngine.swift / POTSPredictor.featureDict.
let FEATURES = ["meanHR","maxHR","minHR","hrDelta","rmssd","sdnn","meanRR",
                "accMagMean","accMagStd","accVertDelta","postureJerkPeak",
                "hrRiseFromBaseline","rmssdPctChange","hrSlope","pNN50"]

// Held-out days — positive-days spread across the recording timeline so the
// test set contains real pre-flareup episodes the model never saw in training.
let TEST_DAYS: Set<String> = ["2026-04-30", "2026-05-07", "2026-05-13"]

let OVERSAMPLE_RATIO = 2   // train up to negative:positive == 2:1

// MARK: - Load CSV manually (full control over the day-based split)

print("Loading \(featuresURL.lastPathComponent) …")
let raw = try String(contentsOf: featuresURL, encoding: .utf8)
var lines = raw.split(separator: "\n").map(String.init)
let header = lines.removeFirst().split(separator: ",").map(String.init)
let colIdx = Dictionary(uniqueKeysWithValues: header.enumerated().map { ($1, $0) })

struct Row { var feats: [Double]; var label: Int; var day: String }
var trainRows: [Row] = []
var testRows:  [Row] = []

for line in lines {
    let c = line.split(separator: ",", omittingEmptySubsequences: false).map(String.init)
    guard c.count == header.count else { continue }
    let feats = FEATURES.map { Double(c[colIdx[$0]!]) ?? 0 }
    let label = Int(c[colIdx["label"]!]) ?? 0
    let day   = c[colIdx["date"]!]
    let row = Row(feats: feats, label: label, day: day)
    if TEST_DAYS.contains(day) { testRows.append(row) } else { trainRows.append(row) }
}

func counts(_ rows: [Row]) -> (pos: Int, neg: Int) {
    (rows.filter { $0.label == 1 }.count, rows.filter { $0.label == 0 }.count)
}
let (trPos, trNeg) = counts(trainRows)
let (tePos, teNeg) = counts(testRows)
print("  Train: \(trainRows.count) rows (\(trPos) flareup / \(trNeg) normal)")
print("  Test:  \(testRows.count) rows (\(tePos) flareup / \(teNeg) normal)  days=\(TEST_DAYS.sorted())")
guard tePos > 0 else { fatalError("Test set has no positive windows — pick different TEST_DAYS.") }

// MARK: - Oversample the minority class on the TRAIN split only

if trPos > 0 && trNeg / trPos > OVERSAMPLE_RATIO {
    let target = trNeg / OVERSAMPLE_RATIO
    let positives = trainRows.filter { $0.label == 1 }
    while counts(trainRows).pos < target {
        trainRows.append(contentsOf: positives)
    }
    trainRows.shuffle()
    let after = counts(trainRows)
    print("  After train-only oversampling: \(trainRows.count) rows (\(after.pos) flareup / \(after.neg) normal)")
}

// MARK: - Build MLDataTables (features + label only — NO date column)

func makeTable(_ rows: [Row]) throws -> MLDataTable {
    var dict: [String: MLDataValueConvertible] = [:]
    for (i, name) in FEATURES.enumerated() { dict[name] = rows.map { $0.feats[i] } }
    dict["label"] = rows.map { $0.label }
    return try MLDataTable(dictionary: dict)
}
let trainTable = try makeTable(trainRows)
let testTable  = try makeTable(testRows)

// MARK: - Train (shallow trees — only ~9 independent positive-days exist,
// so deep/many-iteration boosting just memorises).

print("\nTraining MLBoostedTreeClassifier …")
let params = MLBoostedTreeClassifier.ModelParameters(
    maxDepth: 4,
    maxIterations: 150,
    minLossReduction: 0.0,
    minChildWeight: 10.0,
    stepSize: 0.1
)
let classifier = try MLBoostedTreeClassifier(
    trainingData: trainTable, targetColumn: "label", parameters: params)

// MARK: - Honest metrics on the held-out days (F1 on the flareup class)

let preds = try classifier.predictions(from: testTable)
let predLabels = (0..<preds.count).map { preds[$0].intValue ?? 0 }
let trueLabels = testRows.map { $0.label }
var tp = 0, fp = 0, fn = 0, tn = 0
for (p, t) in zip(predLabels, trueLabels) {
    if t == 1 && p == 1 { tp += 1 }
    else if t == 0 && p == 1 { fp += 1 }
    else if t == 1 && p == 0 { fn += 1 }
    else { tn += 1 }
}
let precision = tp + fp > 0 ? Double(tp) / Double(tp + fp) : 0
let recall    = tp + fn > 0 ? Double(tp) / Double(tp + fn) : 0
let f1        = precision + recall > 0 ? 2 * precision * recall / (precision + recall) : 0
print(String(format: "\nHeld-out days — F1=%.3f  precision=%.3f  recall=%.3f", f1, precision, recall))
print("  TP=\(tp)  FP=\(fp)  FN=\(fn)  TN=\(tn)")
print("  (accuracy would be \(String(format: "%.1f%%", Double(tp+tn)/Double(max(1,tp+fp+fn+tn))*100)) — ignore it; the classes are ~3% positive.)")

// MARK: - Retrain on ALL data for the shipped model, then save

print("\nRetraining on all data for the shipped model …")
let allTable = try makeTable(trainRows + testRows)   // trainRows already oversampled; fine for final fit
let finalModel = try MLBoostedTreeClassifier(
    trainingData: allTable, targetColumn: "label", parameters: params)

let meta = MLModelMetadata(
    author: "POTS Monitor",
    shortDescription: "Flareup predictor (14 features, day-grouped eval, trained \(ISO8601DateFormatter().string(from: Date())))",
    version: "2.0"
)
try? FileManager.default.removeItem(at: modelOut)
try finalModel.write(to: modelOut, metadata: meta)
print("Saved model to \(modelOut.path)")
print("Model features (\(FEATURES.count)): \(FEATURES.joined(separator: ", "))")
