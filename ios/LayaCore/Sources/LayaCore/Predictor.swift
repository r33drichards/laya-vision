import Foundation

/// The three exported graphs (scripts/export_onnx.py). The iOS app implements this with ONNX Runtime; the Linux
/// check tool with onnxruntime in a Python subprocess, so the loop below is tested against PyTorch end to end.
public protocol LayaBackend: AnyObject {
    /// `pixel_values [1, 3, S, S]` -> `image_features [1, image_seq_len, d]`, flattened
    func vision(pixels: [Float]) throws -> [Float]
    /// `input_ids [1, L]`, `image_features [n, image_seq_len, d]`, `option_span [2]` -> `last_hidden_state [1, L, d]`
    func text(ids: [Int64], imageFeatures: [Float], nImages: Int, optionSpan: [Int64]) throws -> [Float]
    /// `hidden [1, L, d]`, `marker_pos [K]`, `qtype [1]` -> (`logits [K]`, `act_logits [n_act]`)
    func head(hidden: [Float], length: Int, markers: [Int64], qtype: Int64) throws -> (logits: [Float], act: [Float])
}

public struct StageTimes: Codable {
    public var preprocessMs = 0.0
    public var visionMs = 0.0
    public var textMs = 0.0
    public var headMs = 0.0
    public var totalMs = 0.0
    public var tokens = 0
    public var forwardPasses = 0
    public init() {}
}

public struct Prediction {
    public let answers: [(String, Answer)]
    public let times: StageTimes
    public let stateTruncated: Int

    /// `VLMAgent.predict`'s output schema.
    public var json: JSON {
        .object([
            ("model", .string("laya-vlm")),
            ("answers", .object(answers.map { ($0.0, $0.1.json) })),
            ("usage", .object([("input_tokens", .number(String(times.tokens))), ("output_tokens", .number("0")),
                               ("images", .number("1"))])),
        ])
    }
}

@inline(__always) func nowMs() -> Double { Double(DispatchTime.now().uptimeNanoseconds) / 1e6 }

public final class Predictor {
    public let cfg: LayaConfig
    public let builder: SequenceBuilder
    public let backend: LayaBackend

    public init(cfg: LayaConfig, tokenizer: BPETokenizer, backend: LayaBackend) {
        self.cfg = cfg
        builder = SequenceBuilder(cfg: cfg, tokenizer: tokenizer)
        self.backend = backend
    }

    /// Parse `{"qid": {type, instructions, criteria}, ...}` keeping its order.
    public static func questions(_ json: JSON) throws -> [(String, Question)] {
        guard let o = json.objectValue, !o.isEmpty else { throw LayaError.badQuestion("questions must be a non-empty JSON object") }
        return try o.map { ($0.0, try Question($0.1)) }
    }

    /// One image (as interleaved RGB/RGBA bytes), a state, the questions: `VLMAgent.predict` for one image, with
    /// one or two option orders (identity, reversed) averaged, as web-demo/ did.
    public func predict(rgb: [UInt8], channels: Int, height: Int, width: Int, state: JSON?,
                        questions: [(String, Question)], nPermutations: Int = 1) throws -> Prediction {
        var t = StageTimes()
        let t0 = nowMs()
        let px = ImagePrep.pixelValues(interleaved: rgb, channels: channels, h: height, w: width, cfg: cfg)
        t.preprocessMs = nowMs() - t0
        return try predict(pixels: px, state: state, questions: questions, nPermutations: nPermutations, times: t, start: t0)
    }

    public func predict(pixels: [Float], state: JSON?, questions: [(String, Question)], nPermutations: Int = 1,
                        times: StageTimes = StageTimes(), start: Double? = nil) throws -> Prediction {
        var t = times
        let t0 = start ?? nowMs()
        var s = nowMs()
        let feats = try backend.vision(pixels: pixels)
        t.visionMs = nowMs() - s
        let prefix = builder.prefixIds(nImages: 1)
        let text = SequenceBuilder.stateText(state)
        var answers: [(String, Answer)] = []
        var truncated = 0
        for (qid, q) in questions {
            let k = q.options.count
            if k < 2 { throw LayaError.badQuestion("question \(qid) needs at least two options") }
            var rows: [HeadRow] = []
            for order in permutations(k, nPermutations) {
                let it = try builder.build(prefix: prefix, text: text, question: q, order: order)
                if it.markers.count != k { throw LayaError.tooLong("question \(qid): options exceed head_max_len") }
                truncated = max(truncated, it.stateTruncated)
                t.tokens += it.ids.count
                t.forwardPasses += 1
                s = nowMs()
                let hidden = try backend.text(ids: it.ids.map(Int64.init), imageFeatures: feats, nImages: 1,
                                              optionSpan: it.optionSpan.map(Int64.init))
                t.textMs += nowMs() - s
                s = nowMs()
                let out = try backend.head(hidden: hidden, length: it.ids.count, markers: it.markers.map(Int64.init),
                                           qtype: Int64(q.type.index))
                t.headMs += nowMs() - s
                let act = softmax(out.act.map(Double.init))
                rows.append(HeadRow(order: order, logits: Array(out.logits.prefix(k)), actProb: act.first ?? 0))
            }
            answers.append((qid, answer(cfg: cfg, question: q, rows: rows)))
        }
        t.totalMs = nowMs() - t0
        return Prediction(answers: answers, times: t, stateTruncated: truncated)
    }
}
