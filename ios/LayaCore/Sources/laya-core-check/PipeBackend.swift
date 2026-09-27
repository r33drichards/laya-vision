import Foundation
import LayaCore

/// `LayaBackend` over ios/tools/ort_server.py: onnxruntime (CPU) in a Python subprocess, for checking `Predictor`
/// end to end on Linux. See that file for the framing.
final class PipeBackend: LayaBackend {
    let proc = Process()
    let toServer = Pipe(), fromServer = Pipe()
    let cfg: LayaConfig

    init(python: String, server: String, exportDir: String, suffix: String, cfg: LayaConfig) throws {
        self.cfg = cfg
        proc.executableURL = URL(fileURLWithPath: python)
        proc.arguments = [server, exportDir, suffix]
        proc.standardInput = toServer
        proc.standardOutput = fromServer
        try proc.run()
    }

    deinit {
        try? toServer.fileHandleForWriting.close()
        proc.waitUntilExit()
    }

    enum Input { case f(String, [Float], [Int]); case i(String, [Int64], [Int]) }

    func call(_ graph: String, _ inputs: [Input]) throws -> [[Float]] {
        var d = Data(graph.utf8)
        func i32(_ v: Int) { var x = Int32(v).littleEndian; d.append(Data(bytes: &x, count: 4)) }
        func dims(_ s: [Int]) { i32(s.count); for v in s { var x = Int64(v).littleEndian; d.append(Data(bytes: &x, count: 8)) } }
        i32(inputs.count)
        for inp in inputs {
            switch inp {
            case let .f(name, v, s):
                i32(name.utf8.count); d.append(Data(name.utf8)); d.append(Data("f".utf8)); dims(s)
                v.withUnsafeBufferPointer { d.append(Data(buffer: $0)) }
            case let .i(name, v, s):
                i32(name.utf8.count); d.append(Data(name.utf8)); d.append(Data("i".utf8)); dims(s)
                v.withUnsafeBufferPointer { d.append(Data(buffer: $0)) }
            }
        }
        try toServer.fileHandleForWriting.write(contentsOf: d)
        let r = fromServer.fileHandleForReading
        func read(_ n: Int) throws -> Data {
            var out = Data()
            while out.count < n {
                guard let chunk = try r.read(upToCount: n - out.count), !chunk.isEmpty else { throw LayaError.tooLong("ort_server closed") }
                out.append(chunk)
            }
            return out
        }
        func int32() throws -> Int { Int(try read(4).withUnsafeBytes { $0.loadUnaligned(as: Int32.self) }) }
        let n = try int32()
        if n < 0 { throw LayaError.badQuestion("onnxruntime: " + String(decoding: try read(try int32()), as: UTF8.self)) }
        var outs: [[Float]] = []
        for _ in 0..<n {
            let rank = try int32()
            let shape = try (0..<rank).map { _ in Int(try read(8).withUnsafeBytes { $0.loadUnaligned(as: Int64.self) }) }
            let count = shape.reduce(1, *)
            let raw = try read(4 * count)
            outs.append(raw.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) })
        }
        return outs
    }

    func vision(pixels: [Float]) throws -> [Float] {
        let s = cfg.imageSize
        return try call("v", [.f("pixel_values", pixels, [1, 3, s, s])])[0]
    }

    func text(ids: [Int64], imageFeatures: [Float], nImages: Int, optionSpan: [Int64]) throws -> [Float] {
        try call("t", [.i("input_ids", ids, [1, ids.count]),
                       .f("image_features", imageFeatures, [nImages, cfg.imageSeqLen, cfg.hiddenSize]),
                       .i("option_span", optionSpan, [2])])[0]
    }

    func head(hidden: [Float], length: Int, markers: [Int64], qtype: Int64) throws -> (logits: [Float], act: [Float]) {
        let o = try call("h", [.f("hidden", hidden, [1, length, cfg.hiddenSize]), .i("marker_pos", markers, [markers.count]),
                               .i("qtype", [qtype], [1])])
        return (o[0], o[1])
    }
}
