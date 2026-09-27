import Foundation

/// The byte-level BPE tokenizer of SmolVLM's `tokenizer.json`: no normalizer, the GPT-2 pre-tokenizer
/// (`ByteLevel`, `use_regex`, no prefix space), BPE merges, and added tokens matched exactly before anything else.
/// `encode` is `tokenizer(text, add_special_tokens=False).input_ids`, which is all laya's sequence builder calls.
/// tests/test_ios_core.py checks it against Python on web-demo/fixtures/parity.json.
public final class BPETokenizer {
    let vocab: [String: Int]
    let ranks: [Pair: Int]
    let added: [(scalars: [Unicode.Scalar], id: Int)]  // longest first
    let byteChar: [Character]
    let splitDigits: Bool
    var cache: [String: [Int]] = [:]

    struct Pair: Hashable { let a: String; let b: String }

    public enum LoadError: Error, CustomStringConvertible {
        case unsupported(String)
        public var description: String { if case let .unsupported(m) = self { return "unsupported tokenizer.json: " + m }; return "" }
    }

    public convenience init(contentsOf url: URL) throws {
        try self.init(data: Data(contentsOf: url))
    }

    public init(data: Data) throws {
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let model = root["model"] as? [String: Any], (model["type"] as? String) == "BPE",
              let vocab = model["vocab"] as? [String: Int], let merges = model["merges"] as? [Any]
        else { throw LoadError.unsupported("expected a BPE model with vocab and merges") }
        if let n = root["normalizer"], !(n is NSNull) { throw LoadError.unsupported("normalizer") }
        // ByteLevel alone, or Sequence[Digits(individual_digits), ByteLevel] (SmolVLM's own tokenizer.json)
        var digits = false
        if let pre = root["pre_tokenizer"] as? [String: Any] {
            var steps = [pre]
            if (pre["type"] as? String) == "Sequence" { steps = (pre["pretokenizers"] as? [[String: Any]]) ?? [] }
            if let first = steps.first, (first["type"] as? String) == "Digits" {
                guard (first["individual_digits"] as? Bool) == true else { throw LoadError.unsupported("Digits without individual_digits") }
                digits = true
                steps.removeFirst()
            }
            guard steps.count == 1, (steps[0]["type"] as? String) == "ByteLevel", (steps[0]["add_prefix_space"] as? Bool) != true
            else { throw LoadError.unsupported("pre_tokenizer must be [Digits(individual)] + ByteLevel without a prefix space") }
        }
        splitDigits = digits
        self.vocab = vocab
        var ranks: [Pair: Int] = [:]
        ranks.reserveCapacity(merges.count)
        for (r, m) in merges.enumerated() {
            let pair: [String]
            if let s = m as? String {
                pair = s.split(separator: " ", maxSplits: 1, omittingEmptySubsequences: false).map(String.init)
            } else if let a = m as? [String] {
                pair = a
            } else { continue }
            if pair.count == 2 { ranks[Pair(a: pair[0], b: pair[1])] = r }
        }
        self.ranks = ranks
        var added: [([Unicode.Scalar], Int)] = []
        for case let t as [String: Any] in (root["added_tokens"] as? [Any]) ?? [] {
            if let c = t["content"] as? String, let id = t["id"] as? Int, !c.isEmpty {
                added.append((Array(c.unicodeScalars), id))
            }
        }
        self.added = added.sorted { $0.0.count > $1.0.count }.map { (scalars: $0.0, id: $0.1) }
        self.byteChar = BPETokenizer.bytesToUnicode()
    }

    /// GPT-2's reversible byte -> printable character map.
    static func bytesToUnicode() -> [Character] {
        var bs: [Int] = Array(33...126) + Array(161...172) + Array(174...255)
        var cs = bs
        var n = 0
        for b in 0..<256 where !bs.contains(b) {
            bs.append(b)
            cs.append(256 + n)
            n += 1
        }
        var out = [Character](repeating: " ", count: 256)
        for (b, c) in zip(bs, cs) { out[b] = Character(Unicode.Scalar(UInt32(c))!) }
        return out
    }

    public func encode(_ text: String) -> [Int] {
        var ids: [Int] = []
        let s = Array(text.unicodeScalars)
        var start = 0
        var i = 0
        // added tokens first: leftmost, then longest at that position
        while i < s.count {
            if let hit = added.first(where: { tok in
                tok.scalars.count <= s.count - i && tok.scalars[0] == s[i] && Array(s[i..<i + tok.scalars.count]) == tok.scalars
            }) {
                if start < i { encodeChunk(s[start..<i], into: &ids) }
                ids.append(hit.id)
                i += hit.scalars.count
                start = i
            } else {
                i += 1
            }
        }
        if start < s.count { encodeChunk(s[start..<s.count], into: &ids) }
        return ids
    }

    func encodeChunk(_ chunk: ArraySlice<Unicode.Scalar>, into ids: inout [Int]) {
        guard splitDigits else { return encodePieces(Array(chunk), into: &ids) }
        // Digits(individual_digits): each ASCII digit is its own piece before the byte-level split (the Rust
        // pre-tokenizer tests `is_ascii_digit`, so ½ or ² stay with their neighbours)
        var start = chunk.startIndex
        for i in chunk.indices where ("0"..."9").contains(chunk[i]) {
            if start < i { encodePieces(Array(chunk[start..<i]), into: &ids) }
            encodePieces([chunk[i]], into: &ids)
            start = i + 1
        }
        if start < chunk.endIndex { encodePieces(Array(chunk[start..<chunk.endIndex]), into: &ids) }
    }

    func encodePieces(_ scalars: [Unicode.Scalar], into ids: inout [Int]) {
        for piece in BPETokenizer.preTokenize(scalars) {
            var word = ""
            word.unicodeScalars.append(contentsOf: piece)
            if let hit = cache[word] { ids += hit; continue }
            let mapped = word.utf8.map { String(byteChar[Int($0)]) }
            let out = bpe(mapped).map { vocab[$0] ?? 0 }
            cache[word] = out
            ids += out
        }
    }

    func bpe(_ symbols: [String]) -> [String] {
        var w = symbols
        while w.count > 1 {
            var best = Int.max
            var at = -1
            for j in 0..<(w.count - 1) {
                if let r = ranks[Pair(a: w[j], b: w[j + 1])], r < best { best = r; at = j }
            }
            if at < 0 { break }
            let a = w[at], b = w[at + 1]
            var merged: [String] = []
            merged.reserveCapacity(w.count)
            var j = 0
            while j < w.count {
                if j < w.count - 1 && w[j] == a && w[j + 1] == b {
                    merged.append(a + b)
                    j += 2
                } else {
                    merged.append(w[j])
                    j += 1
                }
            }
            w = merged
        }
        return w
    }

    // The GPT-2 pattern, as a scanner over Unicode scalars:
    //   's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
    static func isLetter(_ u: Unicode.Scalar) -> Bool {
        switch u.properties.generalCategory {
        case .uppercaseLetter, .lowercaseLetter, .titlecaseLetter, .modifierLetter, .otherLetter: return true
        default: return false
        }
    }

    static func isNumber(_ u: Unicode.Scalar) -> Bool {
        switch u.properties.generalCategory {
        case .decimalNumber, .letterNumber, .otherNumber: return true
        default: return false
        }
    }

    static func isSpace(_ u: Unicode.Scalar) -> Bool { u.properties.isWhitespace }
    static func isOther(_ u: Unicode.Scalar) -> Bool { !isSpace(u) && !isLetter(u) && !isNumber(u) }

    static func preTokenize(_ s: [Unicode.Scalar]) -> [ArraySlice<Unicode.Scalar>] {
        var out: [ArraySlice<Unicode.Scalar>] = []
        let n = s.count
        var i = 0
        func run(from j: Int, while pred: (Unicode.Scalar) -> Bool) -> Int {
            var k = j
            while k < n && pred(s[k]) { k += 1 }
            return k
        }
        while i < n {
            let c = s[i]
            // contractions
            if c == "'" && i + 1 < n {
                var matched = 0
                for suffix in ["s", "t", "re", "ve", "m", "ll", "d"] {
                    let u = Array(suffix.unicodeScalars)
                    if i + 1 + u.count <= n && Array(s[(i + 1)..<(i + 1 + u.count)]) == u { matched = 1 + u.count; break }
                }
                if matched > 0 { out.append(s[i..<i + matched]); i += matched; continue }
            }
            // ` ?\p{L}+`, ` ?\p{N}+`, ` ?[^\s\p{L}\p{N}]+`: the optional leading space is a literal U+0020
            let classes: [(Unicode.Scalar) -> Bool] = [isLetter, isNumber, isOther]
            if let k = classes.firstIndex(where: { $0(c) }) {
                let end = run(from: i, while: classes[k])
                out.append(s[i..<end]); i = end; continue
            }
            if c == " " && i + 1 < n, let k = classes.firstIndex(where: { $0(s[i + 1]) }) {
                let end = run(from: i + 1, while: classes[k])
                out.append(s[i..<end]); i = end; continue
            }
            // `\s+(?!\S)`, else `\s+`: a whitespace run leaves its last character to the next token when a
            // non-space follows it
            let end = run(from: i, while: isSpace)
            if end < n && end - 1 > i {
                out.append(s[i..<(end - 1)]); i = end - 1
            } else {
                out.append(s[i..<max(end, i + 1)]); i = max(end, i + 1)
            }
        }
        return out
    }
}
