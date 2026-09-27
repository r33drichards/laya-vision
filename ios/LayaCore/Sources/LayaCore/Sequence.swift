import Foundation

/// What the app needs from the exporter's `laya_web.json` (scripts/export_onnx.py `write_config`).
public struct LayaConfig {
    public let maxLen: Int
    public let headMaxLen: Int
    public let imageSize: Int
    public let imageSeqLen: Int
    public let stage1LongestEdge: Int
    public let mean: Double
    public let std: Double
    public let hiddenSize: Int
    public let optionEndId: Int
    public let imageTokenId: Int
    public let prefix: String
    public let questionTemplate: String
    public let optionBullet: String
    public let optionEnd: String
    public let fakeImageToken: String
    public let globalImageToken: String
    public let imageToken: String
    public let stripFromInstructions: String
    public let temperature: [Double]
    public let temperatureByOptions: [String: Double]
    public let raw: JSON

    public init(json j: JSON) throws {
        func need<T>(_ v: T?, _ what: String) throws -> T {
            guard let v = v else { throw JSONError.syntax("laya_web.json: missing \(what)") }
            return v
        }
        if let fv = j["format_version"]?.intValue, fv != 1 { throw JSONError.syntax("laya_web.json: format_version \(fv)") }
        if let r = j["readout"]?.stringValue, r != "terminator" { throw JSONError.syntax("laya_web.json: readout \(r)") }
        let img = try need(j["image"], "image"), tx = try need(j["text"], "text"), ids = try need(j["token_ids"], "token_ids")
        maxLen = try need(j["max_len"]?.intValue, "max_len")
        headMaxLen = try need(j["head_max_len"]?.intValue, "head_max_len")
        imageSize = try need(img["size"]?.intValue, "image.size")
        imageSeqLen = try need(img["seq_len"]?.intValue, "image.seq_len")
        stage1LongestEdge = try need(img["stage1_longest_edge"]?.intValue, "image.stage1_longest_edge")
        mean = try need(img["mean"]?.doubleValue, "image.mean")
        std = try need(img["std"]?.doubleValue, "image.std")
        hiddenSize = j["hidden_size"]?.intValue ?? 576
        optionEndId = try need(ids["option_end"]?.intValue, "token_ids.option_end")
        imageTokenId = try need(ids["image"]?.intValue, "token_ids.image")
        prefix = try need(tx["prefix"]?.stringValue, "text.prefix")
        questionTemplate = try need(tx["question"]?.stringValue, "text.question")
        optionBullet = try need(tx["option_bullet"]?.stringValue, "text.option_bullet")
        optionEnd = try need(tx["option_end"]?.stringValue, "text.option_end")
        fakeImageToken = try need(tx["fake_image_token"]?.stringValue, "text.fake_image_token")
        globalImageToken = try need(tx["global_image_token"]?.stringValue, "text.global_image_token")
        imageToken = try need(tx["image_token"]?.stringValue, "text.image_token")
        stripFromInstructions = try need(tx["strip_from_instructions"]?.stringValue, "text.strip_from_instructions")
        temperature = (j["temperature"]?.arrayValue ?? []).compactMap { $0.doubleValue }
        var tbo: [String: Double] = [:]
        for (k, v) in j["temperature_by_options"]?.objectValue ?? [] { if let d = v.doubleValue { tbo[k] = d } }
        temperatureByOptions = tbo
        raw = j
    }
}

public enum QType: String, CaseIterable {
    case choice, score, noul
    public var index: Int { self == .choice ? 0 : self == .score ? 1 : 2 }
}

public enum LayaError: Error, CustomStringConvertible {
    case badQuestion(String)
    case tooLong(String)
    public var description: String {
        switch self {
        case let .badQuestion(m): return m
        case let .tooLong(m): return m
        }
    }
}

/// `VLMAgent._to_internal`: a question as the sequence builder sees it.
public struct Question {
    public let type: QType
    public let instructions: String
    /// choice: ordered (key, description or nil); score: levels; noul: optional true/false descriptions
    public let choices: [(String, String?)]
    public let levels: [String]
    public let noulTrue: String?
    public let noulFalse: String?

    public init(_ def: JSON) throws {
        guard let t = def["type"]?.stringValue, let qt = QType(rawValue: t) else {
            throw LayaError.badQuestion("question type must be choice, score or noul")
        }
        type = qt
        guard let ins = def["instructions"] else { throw LayaError.badQuestion("every question needs instructions") }
        instructions = ins.stringValue ?? ins.pyDumps(ensureAscii: true)
        let crit = def["criteria"] ?? .null
        var choices: [(String, String?)] = [], levels: [String] = []
        var tDesc: String? = nil, fDesc: String? = nil
        switch qt {
        case .choice:
            if let a = crit.arrayValue {
                choices = a.map { ($0.stringValue ?? $0.pyDumps(ensureAscii: true), nil) }
            } else if let o = crit.objectValue {
                choices = o.map { k, v in (k, v.isNull ? nil : (v.stringValue ?? v.pyDumps(ensureAscii: true))) }
            } else {
                throw LayaError.badQuestion("a choice question needs criteria")
            }
        case .score:
            guard let a = crit.arrayValue else { throw LayaError.badQuestion("a score question needs a list of criteria, one per level") }
            levels = a.map { $0.stringValue ?? $0.pyDumps(ensureAscii: true) }
        case .noul:
            tDesc = crit["true"]?.stringValue
            fDesc = crit["false"]?.stringValue
        }
        self.choices = choices
        self.levels = levels
        noulTrue = tDesc
        noulFalse = fDesc
    }

    /// `common.render_options`
    public var options: [String] {
        switch type {
        case .choice: return choices.map { k, v in (v?.isEmpty ?? true) ? k : "\(k): \(v!)" }
        case .score: return levels.enumerated().map { "level \($0.offset): \($0.element)" }
        case .noul:
            return ["false: " + ((noulFalse?.isEmpty ?? true) ? "no, the statement does not hold" : noulFalse!),
                    "true: " + ((noulTrue?.isEmpty ?? true) ? "yes, the statement holds" : noulTrue!)]
        }
    }
}

/// The first two of `laya.vlm._permutations` (identity, reversed); later ones use Python's seeded shuffle.
public func permutations(_ k: Int, _ n: Int) -> [[Int]] {
    var p = [Array(0..<k)]
    if n > 1 && k > 1 { p.append(Array((0..<k).reversed())) }
    return Array(p.prefix(max(1, min(n, 2))))
}

public struct Inputs {
    public let ids: [Int]
    public let markers: [Int]
    public let optionSpan: [Int]
    public let stateTokens: Int
    public let stateTruncated: Int
}

public final class SequenceBuilder {
    public let cfg: LayaConfig
    public let tok: BPETokenizer

    public init(cfg: LayaConfig, tokenizer: BPETokenizer) {
        self.cfg = cfg
        tok = tokenizer
    }

    /// The prefix text with each image replaced by its `<image>` run, tokenized as one string.
    public func prefixIds(nImages: Int) -> [Int] {
        let run = cfg.fakeImageToken + cfg.globalImageToken + String(repeating: cfg.imageToken, count: cfg.imageSeqLen)
            + cfg.fakeImageToken
        return tok.encode(cfg.prefix + String(repeating: run, count: nImages))
    }

    /// `laya.vlm.split_state`'s text half: a string as is, anything else as `json.dumps` without the images.
    public static func stateText(_ state: JSON?) -> String {
        guard let state = state else { return "" }
        switch state {
        case .null: return ""
        case let .string(s): return s
        case .array: return state.pyDumps(ensureAscii: false)
        case let .object(kv):
            let rest = kv.filter { $0.0 != "image" && $0.0 != "images" }
            return rest.isEmpty ? "" : JSON.object(rest).pyDumps(ensureAscii: false)
        default: return state.pyDumps(ensureAscii: false)
        }
    }

    /// `build_vlm_inputs` with the "terminator" readout: prefix, state text cut to fit, question, options each
    /// followed by the option-end token whose position is that option's marker.
    public func build(prefix: [Int], text: String, question q: Question, order: [Int]? = nil) throws -> Inputs {
        let opts = q.options
        let order = order ?? Array(0..<opts.count)
        var optIds = order.map { Array(tok.encode(cfg.optionBullet + opts[$0].replacingOccurrences(of: cfg.optionEnd, with: " ")).prefix(48)) }
        let ins = q.instructions.replacingOccurrences(of: cfg.stripFromInstructions, with: " ")
        let question = replaceFirst(replaceFirst(cfg.questionTemplate, "%s", q.type.rawValue), "%s", ins)
        var headIds = tok.encode(question)
        func used() -> Int { optIds.reduce(0) { $0 + $1.count + 1 } }
        var optBudget = cfg.headMaxLen - used()
        if optBudget < 16 {
            let per = max(4, (cfg.headMaxLen - 16) / max(1, optIds.count) - 1)
            optIds = optIds.map { Array($0.prefix(per)) }
            optBudget = cfg.headMaxLen - used()
        }
        if headIds.count > max(8, optBudget) {
            let keep = max(8, optBudget)
            let front = keep / 2
            headIds = Array(headIds.prefix(front)) + Array(headIds.suffix(keep - front))
        }
        var tail = headIds
        var markers: [Int] = []
        let spanStart = tail.count
        for o in optIds {
            tail += o
            tail.append(cfg.optionEndId)
            markers.append(tail.count - 1)
        }
        if prefix.count + tail.count > cfg.maxLen {
            throw LayaError.tooLong("question + options + images exceed max_len=\(cfg.maxLen)")
        }
        let room = max(0, cfg.maxLen - prefix.count - tail.count)
        let full = text.isEmpty ? [] : tok.encode(text)
        let st = Array(full.prefix(room))
        let off = prefix.count + st.count
        return Inputs(ids: prefix + st + tail, markers: markers.map { $0 + off }, optionSpan: [spanStart + off, tail.count + off],
                      stateTokens: st.count, stateTruncated: full.count - st.count)
    }
}

func replaceFirst(_ s: String, _ a: String, _ b: String) -> String {
    guard let r = s.range(of: a) else { return s }
    return s.replacingCharacters(in: r, with: b)
}
