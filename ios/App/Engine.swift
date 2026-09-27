import Foundation
import LayaCore
import OnnxRuntimeBindings
import UIKit

/// The model files from scripts/export_onnx.py: `laya_web.json`, `tokenizer/`, and `{vision,text,head}{,_fp16,_q8}.onnx`.
/// Looked up in the app's Documents folder first (so a new export can be copied in with Finder, without a rebuild),
/// then in the app bundle (ios/Models, copied in at build time).
struct ModelFolder {
    let url: URL
    let variants: [String]  // "fp32", "fp16", "q8" that have all three graphs
    let checkpoint: String

    static func find() -> ModelFolder? {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first?.appendingPathComponent("Models")
        for dir in [docs, Bundle.main.url(forResource: "Models", withExtension: nil)].compactMap({ $0 }) {
            let cfg = dir.appendingPathComponent("laya_web.json")
            guard FileManager.default.fileExists(atPath: cfg.path) else { continue }
            let variants = ["fp32", "fp16", "q8"].filter { v in
                let suffix = v == "fp32" ? "" : "_" + v
                return ["vision", "text", "head"].allSatisfy {
                    FileManager.default.fileExists(atPath: dir.appendingPathComponent("\($0)\(suffix).onnx").path)
                }
            }
            var checkpoint = "?"
            if let j = try? JSON.parse(data: Data(contentsOf: cfg)) {
                let src = j["source"]?.stringValue ?? "?"
                let rev = j["checkpoint"]?["revision"]?.stringValue.map { "@" + $0.prefix(7) } ?? ""
                checkpoint = src + rev
            }
            return ModelFolder(url: dir, variants: variants, checkpoint: checkpoint)
        }
        return nil
    }
}

struct LoadInfo: Codable {
    var variant: String
    var vision: Compute
    var text: Compute
    var loadSeconds: Double
    var sessionSeconds: [String: Double]
    var warmupMs: Double
    var footprintMB: Double
}

struct BenchStats: Codable {
    var runs: Int
    var medianMs: [String: Double]
    var minMs: [String: Double]
    var maxMs: [String: Double]
    var thermalBefore: String
    var thermalAfter: String
    var peakFootprintMB: Double
}

/// Device facts for a benchmark report.
enum Device {
    static var model: String {
        var u = utsname()
        uname(&u)
        return withUnsafeBytes(of: &u.machine) { String(decoding: $0.prefix { $0 != 0 }, as: UTF8.self) }
    }

    static var os: String { "\(UIDevice.current.systemName) \(UIDevice.current.systemVersion)" }
    static var memoryGB: Double { Double(ProcessInfo.processInfo.physicalMemory) / 1e9 }
    static var cores: Int { ProcessInfo.processInfo.activeProcessorCount }

    static var thermal: String {
        switch ProcessInfo.processInfo.thermalState {
        case .nominal: return "nominal"
        case .fair: return "fair"
        case .serious: return "serious"
        case .critical: return "critical"
        @unknown default: return "unknown"
        }
    }

    /// The process's memory footprint, what iOS counts against the app's limit.
    static var footprintMB: Double {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count) }
        }
        return kr == KERN_SUCCESS ? Double(info.phys_footprint) / 1e6 : -1
    }
}

/// An image as the model's resize wants it: RGBA bytes with the photo's orientation applied, in sRGB.
struct RGBAImage {
    let bytes: [UInt8]
    let width: Int
    let height: Int

    init?(_ image: UIImage) {
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1
        format.opaque = true
        format.preferredRange = .standard  // sRGB, 8 bits per channel
        let size = image.size  // in points with scale 1 = pixels, orientation applied
        let upright = UIGraphicsImageRenderer(size: size, format: format).image { _ in image.draw(in: CGRect(origin: .zero, size: size)) }
        guard let cg = upright.cgImage else { return nil }
        let w = cg.width, h = cg.height
        var px = [UInt8](repeating: 0, count: w * h * 4)
        let ok = px.withUnsafeMutableBytes { buf -> Bool in
            guard let ctx = CGContext(data: buf.baseAddress, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                                      space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                      bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else { return false }
            ctx.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))
            return true
        }
        guard ok else { return nil }
        bytes = px
        width = w
        height = h
    }
}

/// Owns the loaded model; everything here runs off the main thread (see ContentView).
final class Engine: @unchecked Sendable {
    let folder: ModelFolder
    let cfg: LayaConfig
    let tokenizer: BPETokenizer
    private(set) var predictor: Predictor?
    private(set) var info: LoadInfo?

    init(folder: ModelFolder) throws {
        self.folder = folder
        cfg = try LayaConfig(json: JSON.parse(data: Data(contentsOf: folder.url.appendingPathComponent("laya_web.json"))))
        tokenizer = try BPETokenizer(contentsOf: folder.url.appendingPathComponent("tokenizer/tokenizer.json"))
    }

    /// Create the sessions, then one warm-up prediction (the first run pays for kernel setup and Core ML
    /// specialisation; timing it separately keeps the benchmark honest).
    func load(variant: String, vision: Compute, text: Compute, warmup image: RGBAImage?) throws -> LoadInfo {
        predictor = nil
        info = nil
        let t0 = Date()
        let cache = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0].appendingPathComponent("coreml")
        let backend = try ORTBackend(dir: folder.url, suffix: variant == "fp32" ? "" : "_" + variant, vision: vision, text: text,
                                     cfg: cfg, cacheDir: cache)
        let p = Predictor(cfg: cfg, tokenizer: tokenizer, backend: backend)
        let loadSeconds = Date().timeIntervalSince(t0)
        var warm = 0.0
        if let img = image {
            let q = try Predictor.questions(JSON.parse(#"{"w": {"type": "noul", "instructions": "Warm-up."}}"#))
            warm = try p.predict(rgb: img.bytes, channels: 4, height: img.height, width: img.width, state: nil, questions: q).times.totalMs
        }
        predictor = p
        let i = LoadInfo(variant: variant, vision: vision, text: text, loadSeconds: loadSeconds, sessionSeconds: backend.sessionSeconds,
                         warmupMs: warm, footprintMB: Device.footprintMB)
        info = i
        return i
    }

    func predict(_ img: RGBAImage, state: JSON?, questions: [(String, Question)], orders: Int) throws -> Prediction {
        guard let p = predictor else { throw LayaError.badQuestion("load a model first") }
        return try p.predict(rgb: img.bytes, channels: 4, height: img.height, width: img.width, state: state,
                             questions: questions, nPermutations: orders)
    }

    func benchmark(_ img: RGBAImage, state: JSON?, questions: [(String, Question)], orders: Int, runs: Int,
                   progress: (Int) -> Void) throws -> BenchStats {
        let before = Device.thermal
        var samples: [String: [Double]] = [:]
        var peak = Device.footprintMB
        for r in 0..<runs {
            let t = try predict(img, state: state, questions: questions, orders: orders).times
            for (k, v) in [("preprocess", t.preprocessMs), ("vision", t.visionMs), ("text", t.textMs), ("head", t.headMs),
                           ("total", t.totalMs)] {
                samples[k, default: []].append(v)
            }
            peak = max(peak, Device.footprintMB)
            progress(r + 1)
        }
        func median(_ a: [Double]) -> Double {
            let s = a.sorted()
            return s.count % 2 == 1 ? s[s.count / 2] : (s[s.count / 2 - 1] + s[s.count / 2]) / 2
        }
        return BenchStats(runs: runs, medianMs: samples.mapValues(median), minMs: samples.mapValues { $0.min() ?? 0 },
                          maxMs: samples.mapValues { $0.max() ?? 0 }, thermalBefore: before, thermalAfter: Device.thermal,
                          peakFootprintMB: peak)
    }

    static var ortVersion: String { ORTVersion() ?? "?" }
    static var coreMLAvailable: Bool { ORTIsCoreMLExecutionProviderAvailable() }
}
