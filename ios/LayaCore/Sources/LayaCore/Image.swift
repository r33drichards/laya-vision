import Foundation

/// The processor's resize, as laya.preprocess and web-demo/laya.js do it: the longest edge to
/// `stage1_longest_edge` (even sizes), then to `size` x `size`, each a separable antialiased LANCZOS-3 resize with
/// the horizontal pass rounded to uint8 before the vertical one (torch's uint8 path), then `(v/255 - mean) / std`.
public enum ImagePrep {
    struct Row { let lo: Int; let w: [Double] }

    static func lanczos(_ x: Double, _ a: Double = 3) -> Double {
        let x = abs(x)
        if x < 1e-12 { return 1 }
        if x >= a { return 0 }
        let px = Double.pi * x, pxa = px / a
        return (sin(px) / px) * (sin(pxa) / pxa)
    }

    static func axisWeights(_ nIn: Int, _ nOut: Int) -> [Row] {
        if nIn == nOut { return (0..<nOut).map { Row(lo: $0, w: [1]) } }
        let scale = Double(nIn) / Double(nOut)
        let stretch = max(1, scale)
        let support = 3 * stretch
        let span = Int(support.rounded(.up)) * 2 + 2
        return (0..<nOut).map { i in
            let centre = (Double(i) + 0.5) * scale
            let lo = max(0, Int((centre - support + 0.5).rounded(.down)))
            let n = max(0, min(span, nIn - lo))
            var w = (0..<n).map { lanczos((Double(lo + $0) + 0.5 - centre) / stretch) }
            let sum = w.reduce(0, +)
            for j in 0..<n { w[j] /= sum }
            return Row(lo: lo, w: w)
        }
    }

    public static func stage1Size(h: Int, w: Int, longest: Int) -> (Int, Int) {
        var h = h, w = w
        if w >= h {
            h = (longest * h) / w
            h += h % 2
            w = longest
        } else {
            w = (longest * w) / h
            w += w % 2
            h = longest
        }
        return (max(h, 1), max(w, 1))
    }

    /// Round half to even, as torch does, then clamp to uint8.
    @inline(__always) static func toByte(_ v: Double) -> UInt8 {
        var r = v.rounded()  // half away from zero
        if abs(v.truncatingRemainder(dividingBy: 1)) == 0.5 && Int(r) % 2 != 0 { r -= 1 }
        return r < 0 ? 0 : r > 255 ? 255 : UInt8(r)
    }

    /// Planar uint8 `[3, h, w]` -> planar uint8 `[3, oh, ow]`.
    public static func resizePlanar(_ src: [UInt8], h: Int, w: Int, oh: Int, ow: Int) -> [UInt8] {
        let wx = axisWeights(w, ow), wy = axisWeights(h, oh)
        var tmp = [UInt8](repeating: 0, count: h * ow)
        var out = [UInt8](repeating: 0, count: 3 * oh * ow)
        src.withUnsafeBufferPointer { s in
            for c in 0..<3 {
                let base = c * h * w
                for y in 0..<h {
                    let row = base + y * w
                    for x in 0..<ow {
                        let r = wx[x]
                        var acc = 0.0
                        for j in 0..<r.w.count { acc += r.w[j] * Double(s[row + r.lo + j]) }
                        tmp[y * ow + x] = toByte(acc)
                    }
                }
                let ob = c * oh * ow
                for y in 0..<oh {
                    let r = wy[y]
                    for x in 0..<ow {
                        var acc = 0.0
                        for j in 0..<r.w.count { acc += r.w[j] * Double(tmp[(r.lo + j) * ow + x]) }
                        out[ob + y * ow + x] = toByte(acc)
                    }
                }
            }
        }
        return out
    }

    /// Interleaved RGBA or RGB bytes, `h` x `w` -> Float32 `[3, size, size]` pixel values.
    public static func pixelValues(interleaved px: [UInt8], channels: Int, h: Int, w: Int, cfg: LayaConfig) -> [Float] {
        let n = h * w
        var planar = [UInt8](repeating: 0, count: 3 * n)
        for i in 0..<n {
            planar[i] = px[channels * i]
            planar[n + i] = px[channels * i + 1]
            planar[2 * n + i] = px[channels * i + 2]
        }
        let (mh, mw) = stage1Size(h: h, w: w, longest: cfg.stage1LongestEdge)
        let mid = resizePlanar(planar, h: h, w: w, oh: mh, ow: mw)
        let s = cfg.imageSize
        let out = resizePlanar(mid, h: mh, w: mw, oh: s, ow: s)
        let mean = cfg.mean, std = cfg.std
        return out.map { Float((Double($0) / 255 - mean) / std) }
    }
}
