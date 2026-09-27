// Checks for LayaCore, driven by tests/test_ios_core.py (mirrors web-demo/test_parity.mjs):
//   laya-core-check ids <parity.json> <tokenizer.json>
//   laya-core-check pixels <parity.json> <raw HxWx3 uint8 file> <H> <W> <out float32 file>
//   laya-core-check encode <texts.json> <tokenizer.json>     (prints one JSON array of ids per text)
//   laya-core-check predict <export dir> <raw HxWx3 uint8 file> <H> <W> <questions.json> <state.json> <suffix> <python>
//       (the whole Predictor on the exported graphs, onnxruntime via ios/tools/ort_server.py; prints predict's JSON)
import Foundation
import LayaCore

func fail(_ msg: String) -> Never {
    FileHandle.standardError.write((msg + "\n").data(using: .utf8)!)
    exit(2)
}

let args = CommandLine.arguments
guard args.count >= 3 else { fail("usage: laya-core-check ids <parity.json> <tokenizer.json> | pixels <parity.json> <rgb> <H> <W> <out>") }
if args[1] == "encode" {
    let tok = try BPETokenizer(contentsOf: URL(fileURLWithPath: args[3]))
    for t in try JSON.parse(data: Data(contentsOf: URL(fileURLWithPath: args[2]))).arrayValue ?? [] {
        print("[" + tok.encode(t.stringValue ?? "").map(String.init).joined(separator: ",") + "]")
    }
    exit(0)
}
if args[1] == "predict" {
    guard args.count == 10, let h = Int(args[4]), let w = Int(args[5]) else { fail("predict: see the usage at the top of main.swift") }
    let dir = URL(fileURLWithPath: args[2])
    let cfg = try LayaConfig(json: JSON.parse(data: Data(contentsOf: dir.appendingPathComponent("laya_web.json"))))
    let tok = try BPETokenizer(contentsOf: dir.appendingPathComponent("tokenizer/tokenizer.json"))
    let server = URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("../../../tools/ort_server.py").standardized.path
    let backend = try PipeBackend(python: args[9], server: server, exportDir: args[2], suffix: args[8], cfg: cfg)
    let predictor = Predictor(cfg: cfg, tokenizer: tok, backend: backend)
    let rgb = [UInt8](try Data(contentsOf: URL(fileURLWithPath: args[3])))
    let questions = try Predictor.questions(JSON.parse(data: Data(contentsOf: URL(fileURLWithPath: args[6]))))
    let state = try JSON.parse(data: Data(contentsOf: URL(fileURLWithPath: args[7])))
    for n in [1, 2] {
        // LAYA_PIXELS: float32 [3, S, S] pixel values to use instead of the Swift resize (isolates the resize)
        let p: Prediction
        if let f = ProcessInfo.processInfo.environment["LAYA_PIXELS"] {
            let px = try Data(contentsOf: URL(fileURLWithPath: f)).withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
            p = try predictor.predict(pixels: px, state: state, questions: questions, nPermutations: n)
        } else {
            p = try predictor.predict(rgb: rgb, channels: 3, height: h, width: w, state: state, questions: questions, nPermutations: n)
        }
        print(p.json.pyDumps(ensureAscii: false))
        let t = p.times
        FileHandle.standardError.write(String(format: "orders=%d preprocess %.0f ms, vision %.0f ms, text %.0f ms, head %.0f ms, total %.0f ms\n",
                                              n, t.preprocessMs, t.visionMs, t.textMs, t.headMs, t.totalMs).data(using: .utf8)!)
    }
    exit(0)
}
let fixture = try JSON.parse(data: Data(contentsOf: URL(fileURLWithPath: args[2])))
guard let cfgJSON = fixture["cfg"] else { fail("fixture has no cfg") }
let cfg = try LayaConfig(json: cfgJSON)

func ints(_ j: JSON?) -> [Int] { (j?.arrayValue ?? []).compactMap { $0.intValue } }

switch args[1] {
case "ids":
    let tok = try BPETokenizer(contentsOf: URL(fileURLWithPath: args[3]))
    let sb = SequenceBuilder(cfg: cfg, tokenizer: tok)
    var n = 0, bad = 0
    for c in fixture["cases"]?.arrayValue ?? [] {
        let name = c["name"]?.stringValue ?? "?"
        let prefix = sb.prefixIds(nImages: c["n_images"]?.intValue ?? 0)
        if prefix != ints(c["prefix_ids"]) { print("\(name): prefix differs"); bad += 1 }
        let text = SequenceBuilder.stateText(c["state"])
        for r in c["rows"]?.arrayValue ?? [] {
            let qid = r["qid"]?.stringValue ?? ""
            let q = try Question(c["questions"]![qid]!)
            let order = ints(r["order"])
            let got = try sb.build(prefix: prefix, text: text, question: q,
                                   order: permutations(q.options.count, 2)[order.first == 0 ? 0 : 1])
            n += 1
            for (key, g) in [("ids", got.ids), ("markers", got.markers), ("option_span", got.optionSpan)] {
                let want = ints(r[key])
                if g != want {
                    bad += 1
                    let i = zip(g, want).enumerated().first { $0.element.0 != $0.element.1 }?.offset ?? min(g.count, want.count)
                    print("\(name)/\(qid)/\(order): \(key) differ at \(i) (swift \(g.count), py \(want.count))")
                    print("  swift", Array(g[max(0, i - 3)..<min(g.count, i + 5)]), "py", Array(want[max(0, i - 3)..<min(want.count, i + 5)]))
                }
            }
        }
    }
    if bad > 0 { print("\(bad) mismatches over \(n) rows"); exit(1) }
    print("\(n) rows match")
case "pixels":
    guard args.count == 7, let h = Int(args[4]), let w = Int(args[5]) else { fail("pixels <parity.json> <rgb> <H> <W> <out>") }
    let rgb = [UInt8](try Data(contentsOf: URL(fileURLWithPath: args[3])))
    let px = ImagePrep.pixelValues(interleaved: rgb, channels: 3, h: h, w: w, cfg: cfg)
    try px.withUnsafeBufferPointer { Data(buffer: $0) }.write(to: URL(fileURLWithPath: args[6]))
default:
    fail("unknown mode \(args[1])")
}
