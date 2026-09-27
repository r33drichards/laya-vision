import LayaCore
import PhotosUI
import SwiftUI

private let defaultQuestions = """
{
  "damage": {"type": "score", "instructions": "How much damage does the item show?",
             "criteria": ["none", "cosmetic: scratches or dents", "functional: parts broken or missing", "destroyed"]},
  "category": {"type": "choice", "instructions": "What kind of item is this?",
               "criteria": ["electronics", "clothing", "furniture", "food", "other"]},
  "outdoors": {"type": "noul", "instructions": "Was the photo taken outdoors?"}
}
"""

@MainActor
final class Model: ObservableObject {
    @Published var folder = ModelFolder.find()
    @Published var variant = "q8"
    @Published var vision = Compute.cpu
    @Published var text = Compute.cpu
    @Published var status = ""
    @Published var busy = false
    @Published var loaded: LoadInfo?

    @Published var uiImage: UIImage? = Bundle.main.url(forResource: "example", withExtension: "jpg").flatMap { UIImage(contentsOfFile: $0.path) }
    @Published var context = "customer says it arrived broken"
    @Published var questionsText = defaultQuestions
    @Published var orders = 1
    @Published var result: Prediction?
    @Published var runError = ""

    @Published var runs = 10
    @Published var bench: BenchStats?
    @Published var benchProgress = ""

    private var engine: Engine?

    init() {
        if let f = folder, !f.variants.contains(variant), let first = f.variants.last { variant = first }
    }

    var state: JSON? { context.isEmpty ? nil : .object([("context", .string(context))]) }

    func rgba() -> RGBAImage? { uiImage.flatMap(RGBAImage.init) }

    func load() {
        guard let folder = folder else { return }
        busy = true
        loaded = nil
        result = nil
        bench = nil
        status = "Loading \(variant) (vision: \(vision.rawValue), text: \(text.rawValue))… Core ML compiles on the first load, which can take minutes."
        let (variant, vision, text, img) = (variant, vision, text, rgba())
        Task.detached {
            do {
                let engine = try await MainActor.run { try self.engine ?? Engine(folder: folder) }
                let info = try engine.load(variant: variant, vision: vision, text: text, warmup: img)
                await MainActor.run {
                    self.engine = engine
                    self.loaded = info
                    self.status = String(format: "Loaded in %.1f s, warm-up run %.0f ms, memory %.0f MB.", info.loadSeconds, info.warmupMs, info.footprintMB)
                    self.busy = false
                }
            } catch {
                await MainActor.run {
                    self.status = "Load failed: \(error)"
                    self.busy = false
                }
            }
        }
    }

    func run() {
        guard let engine = engine, let img = rgba() else { return }
        busy = true
        runError = ""
        let (state, text, orders) = (state, questionsText, orders)
        Task.detached {
            do {
                let qs = try Predictor.questions(JSON.parse(text))
                let p = try engine.predict(img, state: state, questions: qs, orders: orders)
                await MainActor.run { self.result = p; self.busy = false }
            } catch {
                await MainActor.run { self.runError = "\(error)"; self.busy = false }
            }
        }
    }

    func benchmark() {
        guard let engine = engine, let img = rgba() else { return }
        busy = true
        bench = nil
        runError = ""
        let (state, text, orders, runs) = (state, questionsText, orders, runs)
        Task.detached {
            do {
                let qs = try Predictor.questions(JSON.parse(text))
                let b = try engine.benchmark(img, state: state, questions: qs, orders: orders, runs: runs) { i in
                    Task { @MainActor in self.benchProgress = "run \(i)/\(runs)" }
                }
                await MainActor.run { self.bench = b; self.benchProgress = ""; self.busy = false }
            } catch {
                await MainActor.run { self.runError = "\(error)"; self.busy = false }
            }
        }
    }

    /// Everything about this run as JSON, to paste into an issue or a chat.
    var report: String {
        let enc = JSONEncoder()
        enc.outputFormatting = [.prettyPrinted, .sortedKeys]
        func j<T: Encodable>(_ v: T?) -> String { v.flatMap { try? enc.encode($0) }.map { String(decoding: $0, as: UTF8.self) } ?? "null" }
        let img = uiImage.map { "\(Int($0.size.width))x\(Int($0.size.height))" } ?? "none"
        return """
        {"device": \(JSON.string(Device.model).pyDumps()), "os": \(JSON.string(Device.os).pyDumps()), \
        "memory_gb": \(String(format: "%.1f", Device.memoryGB)), "cores": \(Device.cores), \
        "onnxruntime": \(JSON.string(Engine.ortVersion).pyDumps()), "checkpoint": \(JSON.string(folder?.checkpoint ?? "?").pyDumps()), \
        "image": "\(img)", "option_orders": \(orders),
        "load": \(j(loaded)),
        "last_run": \(j(result?.times)),
        "answers": \(result?.json.pyDumps() ?? "null"),
        "benchmark": \(j(bench))}
        """
    }
}

struct ContentView: View {
    @StateObject private var m = Model()
    @State private var pick: PhotosPickerItem?

    var body: some View {
        NavigationStack {
            Form {
                modelSection
                imageSection
                questionSection
                if let r = m.result { answersSection(r) }
                benchSection
                deviceSection
            }
            .navigationTitle("Laya Vision")
            .disabled(m.busy)
            .overlay { if m.busy { ProgressView().controlSize(.large) } }
        }
        .onChange(of: pick) { _, item in
            Task {
                if let data = try? await item?.loadTransferable(type: Data.self), let img = UIImage(data: data) {
                    m.uiImage = img
                    m.result = nil
                }
            }
        }
    }

    var modelSection: some View {
        Section("Model") {
            if let f = m.folder {
                Text(f.checkpoint).font(.caption).foregroundStyle(.secondary)
                Picker("Precision", selection: $m.variant) { ForEach(f.variants, id: \.self) { Text($0) } }
                Picker("Vision tower", selection: $m.vision) { ForEach(Compute.allCases) { Text($0.rawValue).tag($0) } }
                Picker("Language model", selection: $m.text) { ForEach(Compute.allCases) { Text($0.rawValue).tag($0) } }
                Button(m.loaded == nil ? "Load" : "Reload") { m.load() }
                if !m.status.isEmpty { Text(m.status).font(.footnote) }
            } else {
                Text("No model files. Run ios/scripts/export_models.sh on your Mac and rebuild, or copy an export's folder into the app's Documents as \"Models\" with Finder.")
                    .font(.footnote)
            }
        }
    }

    var imageSection: some View {
        Section("Image and context") {
            if let img = m.uiImage {
                Image(uiImage: img).resizable().scaledToFit().frame(maxHeight: 220)
            }
            PhotosPicker("Choose a photo", selection: $pick, matching: .images)
            TextField("Context (optional)", text: $m.context, axis: .vertical)
        }
    }

    var questionSection: some View {
        Section {
            TextEditor(text: $m.questionsText)
                .font(.system(.caption, design: .monospaced))
                .frame(minHeight: 160)
                .autocorrectionDisabled()
                .textInputAutocapitalization(.never)
            Stepper("Option orders: \(m.orders)", value: $m.orders, in: 1...2)
            Button("Run") { m.run() }.disabled(m.loaded == nil || m.uiImage == nil)
            if !m.runError.isEmpty { Text(m.runError).foregroundStyle(.red).font(.footnote) }
        } header: {
            Text("Questions (JSON, as predict takes them)")
        }
    }

    func answersSection(_ r: Prediction) -> some View {
        Section("Answers") {
            ForEach(r.answers.indices, id: \.self) { qi in
                answerView(qid: r.answers[qi].0, a: r.answers[qi].1)
            }
            let t = r.times
            Text(String(format: "%.0f ms: resize %.0f, vision %.0f, language model %.0f, head %.0f · %d tokens, %d passes",
                        t.totalMs, t.preprocessMs, t.visionMs, t.textMs, t.headMs, t.tokens, t.forwardPasses))
                .font(.footnote.monospacedDigit())
            if r.stateTruncated > 0 { Text("Context cut by \(r.stateTruncated) tokens to fit.").font(.footnote) }
        }
    }

    func answerView(qid: String, a: Answer) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(headline(qid, a)).font(.headline)
            ForEach(a.probabilities.indices, id: \.self) { i in
                probabilityRow(label: a.type == .score && i < a.legend.count
                                   ? "\(a.probabilities[i].0): \(a.legend[i])" : a.probabilities[i].0,
                               p: a.probabilities[i].1)
            }
        }
    }

    func probabilityRow(label: String, p: Double) -> some View {
        HStack {
            Text(label).font(.caption).lineLimit(1).frame(width: 130, alignment: .leading)
            GeometryReader { g in
                Capsule().fill(Color.accentColor.opacity(0.25))
                    .overlay(alignment: .leading) { Capsule().fill(Color.accentColor).frame(width: g.size.width * p) }
            }
            .frame(height: 8)
            Text(String(format: "%.1f%%", 100 * p)).font(.caption.monospacedDigit()).frame(width: 52, alignment: .trailing)
        }
    }

    func headline(_ qid: String, _ a: Answer) -> String {
        switch a.type {
        case .choice: return "\(qid): \(a.choice ?? "")"
        case .score: return String(format: "%@: level %.2f", qid, a.score ?? 0)
        case .noul: return String(format: "%@: P(true) = %.3f", qid, a.noul ?? 0)
        }
    }

    var benchSection: some View {
        Section("Benchmark") {
            Stepper("Runs: \(m.runs)", value: $m.runs, in: 3...50)
            Button("Run benchmark") { m.benchmark() }.disabled(m.loaded == nil || m.uiImage == nil)
            if !m.benchProgress.isEmpty { Text(m.benchProgress).font(.footnote) }
            if let b = m.bench {
                ForEach(["total", "preprocess", "vision", "text", "head"], id: \.self) { k in
                    HStack {
                        Text(k == "text" ? "language model" : k)
                        Spacer()
                        Text(String(format: "%.0f ms (%.0f–%.0f)", b.medianMs[k] ?? 0, b.minMs[k] ?? 0, b.maxMs[k] ?? 0))
                            .font(.body.monospacedDigit())
                    }
                }
                Text("Median of \(b.runs) runs (min–max). Thermal state \(b.thermalBefore) → \(b.thermalAfter), peak memory \(Int(b.peakFootprintMB)) MB.")
                    .font(.footnote)
            }
            Button("Copy report") { UIPasteboard.general.string = m.report }
        }
    }

    var deviceSection: some View {
        Section("Device") {
            LabeledContent("Model", value: Device.model)
            LabeledContent("System", value: Device.os)
            LabeledContent("Memory", value: String(format: "%.1f GB", Device.memoryGB))
            LabeledContent("ONNX Runtime", value: "\(Engine.ortVersion)\(Engine.coreMLAvailable ? ", Core ML EP" : "")")
        }
    }
}
