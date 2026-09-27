import Foundation

/// One forward pass of the head for one option order: the logits in that order and P(act).
public struct HeadRow {
    public let order: [Int]
    public let logits: [Float]
    public let actProb: Double
    public init(order: [Int], logits: [Float], actProb: Double) {
        self.order = order
        self.logits = logits
        self.actProb = actProb
    }
}

/// An answer in `VLMAgent.predict`'s schema.
public struct Answer {
    public let type: QType
    /// choice: option key; score: expected level; noul: P(true)
    public let choice: String?
    public let score: Double?
    public let noul: Double?
    /// (label, probability) in option order; for score the labels are the level numbers, for noul false/true
    public let probabilities: [(String, Double)]
    public let legend: [String]
    public let confidence: Double
    public let actProbability: Double

    public var json: JSON {
        func num(_ x: Double) -> JSON { .number(String(round4(x))) }
        var kv: [(String, JSON)] = [("type", .string(type.rawValue))]
        switch type {
        case .choice:
            kv.append(("choice", .string(choice ?? "")))
            kv.append(("probabilities", .object(probabilities.map { ($0.0, num($0.1)) })))
        case .score:
            kv.append(("score", num(score ?? 0)))
            kv.append(("legend", .object(legend.enumerated().map { (String($0.offset), .string($0.element)) })))
            kv.append(("probabilities", .object(probabilities.map { ($0.0, num($0.1)) })))
        case .noul:
            kv.append(("noul", num(noul ?? 0)))
        }
        kv.append(("confidence", num(confidence)))
        kv.append(("action", .object([("act_probability", num(actProbability))])))
        return .object(kv)
    }
}

func round4(_ x: Double) -> Double { (x * 1e4).rounded() / 1e4 }

public func softmax(_ z: [Double]) -> [Double] {
    let m = z.max() ?? 0
    let e = z.map { exp($0 - m) }
    let s = e.reduce(0, +)
    return e.map { $0 / s }
}

func tempBucket(_ qt: QType, _ k: Int) -> String {
    let size = k <= 2 ? "2" : k <= 5 ? "3-5" : k <= 10 ? "6-10" : "11+"
    return "\(qt.rawValue):\(size)"
}

func confidence(_ p: [Double], _ k: Int) -> Double {
    if k < 2 { return 1 }
    var ent = 0.0
    for v in p.prefix(k) { ent -= v * log(min(1, max(1e-12, v))) }
    return min(1, max(0, 1 - ent / log(Double(k))))
}

/// The tail of `VLMAgent.predict`: average the logits over option orders, apply the checkpoint's temperature for
/// this question type and option count, softmax, and build the answer.
public func answer(cfg: LayaConfig, question q: Question, rows: [HeadRow]) -> Answer {
    let k = q.options.count
    var zSum = [Double](repeating: 0, count: k)
    var act = 0.0
    for r in rows {
        for (j, opt) in r.order.enumerated() { zSum[opt] += Double(r.logits[j]) }
        act += r.actProb
    }
    let n = Double(max(1, rows.count))
    let t = cfg.temperatureByOptions[tempBucket(q.type, k)] ?? (q.type.index < cfg.temperature.count ? cfg.temperature[q.type.index] : 1)
    let p = softmax(zSum.map { $0 / n / max(1e-3, t) })
    let conf = round4(confidence(p, k))
    let actP = act / n
    switch q.type {
    case .choice:
        let best = p.indices.max { p[$0] < p[$1] } ?? 0
        return Answer(type: .choice, choice: q.choices[best].0, score: nil, noul: nil,
                      probabilities: zip(q.choices.map { $0.0 }, p).map { ($0, $1) }, legend: [],
                      confidence: conf, actProbability: actP)
    case .score:
        let s = p.enumerated().reduce(0.0) { $0 + Double($1.offset) * $1.element }
        return Answer(type: .score, choice: nil, score: s, noul: nil,
                      probabilities: p.enumerated().map { (String($0.offset), $0.element) }, legend: q.levels,
                      confidence: conf, actProbability: actP)
    case .noul:
        return Answer(type: .noul, choice: nil, score: nil, noul: p[1],
                      probabilities: [("false", p[0]), ("true", p[1])], legend: [],
                      confidence: round4(max(p[1], 1 - p[1])), actProbability: actP)
    }
}
