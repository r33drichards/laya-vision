import Foundation
import LayaCore
import OnnxRuntimeBindings

/// Where a graph runs. `cpu` is ONNX Runtime's own kernels; the others hand the graph to Core ML (ONNX Runtime's
/// Core ML execution provider, ML Program format), restricted to the given compute units. Operators Core ML
/// cannot take stay on the CPU.
enum Compute: String, CaseIterable, Identifiable, Codable {
    case cpu = "CPU"
    case coreMLAll = "Core ML: all"
    case coreMLGPU = "Core ML: CPU + GPU"
    case coreMLANE = "Core ML: CPU + Neural Engine"

    var id: String { rawValue }

    var coreMLUnits: String? {
        switch self {
        case .cpu: return nil
        case .coreMLAll: return "ALL"
        case .coreMLGPU: return "CPUAndGPU"
        case .coreMLANE: return "CPUAndNeuralEngine"
        }
    }
}

/// The exported graphs (scripts/export_onnx.py) in ONNX Runtime. The head is a few million parameters and always
/// runs on the CPU; the vision tower and the language model run where `vision` and `text` say.
final class ORTBackend: LayaBackend, @unchecked Sendable {
    let env: ORTEnv
    let visionSession: ORTSession
    let textSession: ORTSession
    let headSession: ORTSession
    let cfg: LayaConfig
    /// seconds to create each session (Core ML compiles its part of the graph here)
    let sessionSeconds: [String: Double]

    init(dir: URL, suffix: String, vision: Compute, text: Compute, cfg: LayaConfig, cacheDir: URL) throws {
        let env = try ORTEnv(loggingLevel: .warning)
        var seconds: [String: Double] = [:]
        func session(_ name: String, _ compute: Compute) throws -> ORTSession {
            let opts = try ORTSessionOptions()
            try opts.setGraphOptimizationLevel(.all)
            if let units = compute.coreMLUnits {
                let cache = cacheDir.appendingPathComponent("\(name)\(suffix)-\(units)")
                try FileManager.default.createDirectory(at: cache, withIntermediateDirectories: true)
                try opts.appendCoreMLExecutionProvider(withOptionsV2: [
                    "ModelFormat": "MLProgram",
                    "MLComputeUnits": units,
                    "ModelCacheDirectory": cache.path,
                ])
            }
            let path = dir.appendingPathComponent("\(name)\(suffix).onnx").path
            let t0 = Date()
            let s = try ORTSession(env: env, modelPath: path, sessionOptions: opts)
            seconds[name] = Date().timeIntervalSince(t0)
            return s
        }
        visionSession = try session("vision", vision)
        textSession = try session("text", text)
        headSession = try session("head", .cpu)
        self.env = env
        self.cfg = cfg
        sessionSeconds = seconds
    }

    static func tensor(_ a: [Float], _ shape: [Int]) throws -> ORTValue {
        let data = a.withUnsafeBufferPointer { NSMutableData(bytes: $0.baseAddress, length: $0.count * MemoryLayout<Float>.size) }
        return try ORTValue(tensorData: data, elementType: .float, shape: shape.map { NSNumber(value: $0) })
    }

    static func tensor(_ a: [Int64], _ shape: [Int]) throws -> ORTValue {
        let data = a.withUnsafeBufferPointer { NSMutableData(bytes: $0.baseAddress, length: $0.count * MemoryLayout<Int64>.size) }
        return try ORTValue(tensorData: data, elementType: .int64, shape: shape.map { NSNumber(value: $0) })
    }

    static func floats(_ v: ORTValue?) throws -> [Float] {
        guard let v = v else { throw LayaError.badQuestion("a graph output is missing") }
        let d = try v.tensorData()
        return Array(UnsafeBufferPointer(start: d.bytes.assumingMemoryBound(to: Float.self), count: d.length / MemoryLayout<Float>.size))
    }

    func vision(pixels: [Float]) throws -> [Float] {
        let s = cfg.imageSize
        let out = try visionSession.run(withInputs: ["pixel_values": ORTBackend.tensor(pixels, [1, 3, s, s])],
                                        outputNames: ["image_features"], runOptions: nil)
        return try ORTBackend.floats(out["image_features"])
    }

    func text(ids: [Int64], imageFeatures: [Float], nImages: Int, optionSpan: [Int64]) throws -> [Float] {
        let out = try textSession.run(withInputs: [
            "input_ids": ORTBackend.tensor(ids, [1, ids.count]),
            "image_features": ORTBackend.tensor(imageFeatures, [nImages, cfg.imageSeqLen, cfg.hiddenSize]),
            "option_span": ORTBackend.tensor(optionSpan, [2]),
        ], outputNames: ["last_hidden_state"], runOptions: nil)
        return try ORTBackend.floats(out["last_hidden_state"])
    }

    func head(hidden: [Float], length: Int, markers: [Int64], qtype: Int64) throws -> (logits: [Float], act: [Float]) {
        let out = try headSession.run(withInputs: [
            "hidden": ORTBackend.tensor(hidden, [1, length, cfg.hiddenSize]),
            "marker_pos": ORTBackend.tensor(markers, [markers.count]),
            "qtype": ORTBackend.tensor([qtype], [1]),
        ], outputNames: ["logits", "act_logits"], runOptions: nil)
        return (try ORTBackend.floats(out["logits"]), try ORTBackend.floats(out["act_logits"]))
    }
}
