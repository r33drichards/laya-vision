import Foundation

/// A JSON value that keeps object key order. Order matters here: a choice question's options, the questions of a
/// request and the keys of the state text all follow the order they were written in, as Python dicts do, and
/// Foundation's JSONSerialization does not keep it.
public indirect enum JSON: Equatable {
    case null
    case bool(Bool)
    /// The literal as written, so an integer prints as an integer and `1.0` stays a float, like Python's json.
    case number(String)
    case string(String)
    case array([JSON])
    case object([(String, JSON)])

    public static func == (a: JSON, b: JSON) -> Bool {
        switch (a, b) {
        case (.null, .null): return true
        case let (.bool(x), .bool(y)): return x == y
        case let (.number(x), .number(y)): return x == y
        case let (.string(x), .string(y)): return x == y
        case let (.array(x), .array(y)): return x == y
        case let (.object(x), .object(y)):
            return x.count == y.count && zip(x, y).allSatisfy { $0.0 == $1.0 && $0.1 == $1.1 }
        default: return false
        }
    }

    public subscript(key: String) -> JSON? {
        if case let .object(kv) = self { return kv.first { $0.0 == key }?.1 }
        return nil
    }

    public var stringValue: String? { if case let .string(s) = self { return s }; return nil }
    public var arrayValue: [JSON]? { if case let .array(a) = self { return a }; return nil }
    public var objectValue: [(String, JSON)]? { if case let .object(o) = self { return o }; return nil }
    public var doubleValue: Double? { if case let .number(n) = self { return Double(n) }; return nil }
    public var intValue: Int? {
        if case let .number(n) = self { return Int(n) ?? Double(n).map { Int($0) } }
        return nil
    }
    public var isNull: Bool { if case .null = self { return true }; return false }

    public static func parse(_ text: String) throws -> JSON {
        var p = Parser(Array(text.unicodeScalars))
        p.skipSpace()
        let v = try p.value()
        p.skipSpace()
        guard p.i == p.s.count else { throw JSONError.syntax("trailing characters at \(p.i)") }
        return v
    }

    public static func parse(data: Data) throws -> JSON {
        guard let s = String(data: data, encoding: .utf8) else { throw JSONError.syntax("not UTF-8") }
        return try parse(s)
    }

    /// `json.dumps(value, ensure_ascii=ensureAscii)` with Python's default `", "` and `": "` separators.
    public func pyDumps(ensureAscii: Bool = true) -> String {
        switch self {
        case .null: return "null"
        case let .bool(b): return b ? "true" : "false"
        case let .number(n): return pyNumber(n)
        case let .string(s): return pyString(s, ensureAscii: ensureAscii)
        case let .array(a): return "[" + a.map { $0.pyDumps(ensureAscii: ensureAscii) }.joined(separator: ", ") + "]"
        case let .object(o):
            return "{" + o.map { pyString($0.0, ensureAscii: ensureAscii) + ": " + $0.1.pyDumps(ensureAscii: ensureAscii) }
                .joined(separator: ", ") + "}"
        }
    }
}

public enum JSONError: Error, CustomStringConvertible {
    case syntax(String)
    public var description: String { if case let .syntax(m) = self { return "invalid JSON: " + m }; return "" }
}

/// Python's repr for a JSON number literal: integers as written, floats in Python's shortest round-trip form
/// (`1.0`, `1e-05`, `1e+16`).
func pyNumber(_ literal: String) -> String {
    let isFloat = literal.contains(where: { ".eE".contains($0) })
    if !isFloat {
        // Python keeps big integers exact; strip a leading "-0" oddity only
        return literal == "-0" ? "0" : literal
    }
    guard let d = Double(literal) else { return literal }
    if d.isNaN { return "NaN" }
    if d.isInfinite { return d > 0 ? "Infinity" : "-Infinity" }
    // Python switches to exponent form below 1e-4 and at or above 1e16; Swift's description is also the shortest
    // round-trip digits but switches at different magnitudes, so rebuild the form from the digits.
    let a = abs(d)
    if a != 0 && (a < 1e-4 || a >= 1e16) {
        var s = String(format: "%.17g", d)
        // shortest digits that round-trip
        for p in 1...17 {
            let t = String(format: "%.\(p - 1)e", d)
            if Double(t) == d { s = t; break }
        }
        // "1.5e-05" style: Python drops a trailing ".0" mantissa and pads the exponent to two digits
        let parts = s.split(separator: "e", maxSplits: 1).map(String.init)
        var mant = parts[0]
        if mant.contains(".") {
            while mant.hasSuffix("0") { mant.removeLast() }
            if mant.hasSuffix(".") { mant.removeLast() }
        }
        var exp = parts[1]
        let sign = exp.hasPrefix("-") ? "-" : "+"
        exp = String(exp.drop(while: { $0 == "+" || $0 == "-" }))
        while exp.count > 2 && exp.hasPrefix("0") { exp.removeFirst() }
        if exp.count < 2 { exp = "0" + exp }
        return mant + "e" + sign + exp
    }
    var s = "\(d)"  // shortest round-trip, positional in this range
    if s.contains("e") {  // Swift uses exponent form for some values Python prints positionally
        s = String(format: "%.17f", d)
        for p in 0...17 {
            let t = String(format: "%.\(p)f", d)
            if Double(t) == d { s = t; break }
        }
    }
    if !s.contains(".") { s += ".0" }
    return s
}

func pyString(_ s: String, ensureAscii: Bool) -> String {
    var out = "\""
    for u in s.unicodeScalars {
        let c = u.value
        switch u {
        case "\"": out += "\\\""
        case "\\": out += "\\\\"
        case "\n": out += "\\n"
        case "\r": out += "\\r"
        case "\t": out += "\\t"
        case "\u{08}": out += "\\b"
        case "\u{0C}": out += "\\f"
        default:
            if c < 0x20 {
                out += String(format: "\\u%04x", c)
            } else if ensureAscii && c > 0x7E {
                if c > 0xFFFF {
                    let v = c - 0x10000
                    out += String(format: "\\u%04x\\u%04x", 0xD800 + (v >> 10), 0xDC00 + (v & 0x3FF))
                } else {
                    out += String(format: "\\u%04x", c)
                }
            } else {
                out.unicodeScalars.append(u)
            }
        }
    }
    return out + "\""
}

private struct Parser {
    let s: [Unicode.Scalar]
    var i = 0
    init(_ s: [Unicode.Scalar]) { self.s = s }

    mutating func skipSpace() {
        while i < s.count, " \t\n\r".unicodeScalars.contains(s[i]) { i += 1 }
    }

    mutating func expect(_ word: String) throws {
        for u in word.unicodeScalars {
            guard i < s.count, s[i] == u else { throw JSONError.syntax("expected \(word) at \(i)") }
            i += 1
        }
    }

    mutating func value() throws -> JSON {
        guard i < s.count else { throw JSONError.syntax("unexpected end") }
        switch s[i] {
        case "{":
            i += 1
            var kv: [(String, JSON)] = []
            skipSpace()
            if i < s.count, s[i] == "}" { i += 1; return .object(kv) }
            while true {
                skipSpace()
                guard i < s.count, s[i] == "\"" else { throw JSONError.syntax("expected a key at \(i)") }
                let k = try string()
                skipSpace()
                try expect(":")
                skipSpace()
                let v = try value()
                if let j = kv.firstIndex(where: { $0.0 == k }) { kv[j].1 = v } else { kv.append((k, v)) }  // Python: last wins
                skipSpace()
                guard i < s.count else { throw JSONError.syntax("unterminated object") }
                if s[i] == "," { i += 1; continue }
                if s[i] == "}" { i += 1; return .object(kv) }
                throw JSONError.syntax("expected , or } at \(i)")
            }
        case "[":
            i += 1
            var a: [JSON] = []
            skipSpace()
            if i < s.count, s[i] == "]" { i += 1; return .array(a) }
            while true {
                skipSpace()
                a.append(try value())
                skipSpace()
                guard i < s.count else { throw JSONError.syntax("unterminated array") }
                if s[i] == "," { i += 1; continue }
                if s[i] == "]" { i += 1; return .array(a) }
                throw JSONError.syntax("expected , or ] at \(i)")
            }
        case "\"": return .string(try string())
        case "t": try expect("true"); return .bool(true)
        case "f": try expect("false"); return .bool(false)
        case "n": try expect("null"); return .null
        default:
            let start = i
            while i < s.count, "+-0123456789.eE".unicodeScalars.contains(s[i]) { i += 1 }
            var lit = ""
            lit.unicodeScalars.append(contentsOf: s[start..<i])
            guard !lit.isEmpty, Double(lit) != nil else { throw JSONError.syntax("bad value at \(start)") }
            return .number(lit)
        }
    }

    mutating func hex4() throws -> UInt32 {
        guard i + 4 <= s.count else { throw JSONError.syntax("bad \\u escape") }
        var t = ""
        t.unicodeScalars.append(contentsOf: s[i..<i + 4])
        guard let v = UInt32(t, radix: 16) else { throw JSONError.syntax("bad \\u escape") }
        i += 4
        return v
    }

    mutating func string() throws -> String {
        i += 1  // opening quote
        var out = String.UnicodeScalarView()
        while true {
            guard i < s.count else { throw JSONError.syntax("unterminated string") }
            let c = s[i]
            i += 1
            if c == "\"" { return String(out) }
            if c != "\\" { out.append(c); continue }
            guard i < s.count else { throw JSONError.syntax("bad escape") }
            let e = s[i]
            i += 1
            switch e {
            case "\"": out.append("\"")
            case "\\": out.append("\\")
            case "/": out.append("/")
            case "b": out.append("\u{08}")
            case "f": out.append("\u{0C}")
            case "n": out.append("\n")
            case "r": out.append("\r")
            case "t": out.append("\t")
            case "u":
                var v = try hex4()
                if (0xD800..<0xDC00).contains(v), i + 1 < s.count, s[i] == "\\", s[i + 1] == "u" {
                    let save = i
                    i += 2
                    let lo = try hex4()
                    if (0xDC00..<0xE000).contains(lo) { v = 0x10000 + ((v - 0xD800) << 10) + (lo - 0xDC00) } else { i = save }
                }
                out.append(Unicode.Scalar(v) ?? "\u{FFFD}")
            default: throw JSONError.syntax("bad escape \\\(e)")
            }
        }
    }
}
