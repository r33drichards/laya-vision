// Pure JavaScript ports of the parts of laya/vlm.py and laya/common.py the browser needs: question
// normalisation, option rendering, the "terminator" token sequence (build_vlm_inputs), the image resize, and
// turning logits into answers. No DOM, no model runtime: the tokenizer is passed in, so the same module runs in
// the worker and in Node for the parity test (web-demo/test_parity.mjs).

export const QTYPES = { choice: 0, score: 1, noul: 2 };
const QTYPE_NAMES = ["choice", "score", "noul"];

// ---------------------------------------------------------------------------------------------------------
// Python json.dumps, for instructions given as objects and for the state text
// ---------------------------------------------------------------------------------------------------------

function pyString(s, ensureAscii) {
  let out = '"';
  for (const ch of s) {
    const c = ch.codePointAt(0);
    if (ch === '"') out += '\\"';
    else if (ch === "\\") out += "\\\\";
    else if (ch === "\n") out += "\\n";
    else if (ch === "\r") out += "\\r";
    else if (ch === "\t") out += "\\t";
    else if (ch === "\b") out += "\\b";
    else if (ch === "\f") out += "\\f";
    else if (c < 0x20) out += "\\u" + c.toString(16).padStart(4, "0");
    else if (ensureAscii && c > 0x7e) {
      if (c > 0xffff) {
        const v = c - 0x10000;
        out += "\\u" + (0xd800 + (v >> 10)).toString(16).padStart(4, "0");
        out += "\\u" + (0xdc00 + (v & 0x3ff)).toString(16).padStart(4, "0");
      } else out += "\\u" + c.toString(16).padStart(4, "0");
    } else out += ch;
  }
  return out + '"';
}

function pyNumber(x) {
  if (Number.isInteger(x) && !Object.is(x, -0) && Math.abs(x) < 1e16) return String(x);
  if (!Number.isFinite(x)) return Number.isNaN(x) ? "NaN" : x > 0 ? "Infinity" : "-Infinity";
  // JSON numbers parsed from text lose the int/float distinction; a float with an integral value prints as "1.0"
  let s = String(x);
  if (/e/.test(s)) s = s.replace(/e([+-])(\d)$/, "e$10$2");
  return s;
}

/** ``json.dumps(value, ensure_ascii=ensureAscii)`` with Python's default ``", "`` / ``": "`` separators. */
export function pyJsonDumps(value, ensureAscii = true) {
  if (value === null || value === undefined) return "null";
  if (value === true) return "true";
  if (value === false) return "false";
  if (typeof value === "number") return pyNumber(value);
  if (typeof value === "string") return pyString(value, ensureAscii);
  if (Array.isArray(value)) return "[" + value.map((v) => pyJsonDumps(v, ensureAscii)).join(", ") + "]";
  return "{" + Object.entries(value).map(([k, v]) => pyString(k, ensureAscii) + ": " + pyJsonDumps(v, ensureAscii)).join(", ") + "}";
}

// ---------------------------------------------------------------------------------------------------------
// Questions (VLMAgent._to_internal, common.render_options)
// ---------------------------------------------------------------------------------------------------------

export function toInternal(qdef) {
  const t = qdef.type;
  if (!(t in QTYPES)) throw new Error(`question type must be choice, score or noul, got ${JSON.stringify(t)}`);
  let crit = qdef.criteria ?? null;
  if (t === "choice" && Array.isArray(crit)) crit = Object.fromEntries(crit.map((c) => [String(c), null]));
  let ins = qdef.instructions;
  if (ins === undefined) throw new Error("every question needs instructions");
  if (typeof ins !== "string") ins = pyJsonDumps(ins, true);
  return { t, ins, crit };
}

export function renderOptions(q) {
  const { t, crit } = q;
  if (t === "choice") {
    if (!crit || typeof crit !== "object") throw new Error("a choice question needs criteria");
    return Object.entries(crit).map(([k, v]) => (v ? `${k}: ${v}` : k));
  }
  if (t === "score") {
    if (!Array.isArray(crit)) throw new Error("a score question needs a list of criteria, one per level");
    return crit.map((c, i) => `level ${i}: ${c}`);
  }
  const c = crit || {};
  return [
    "false: " + (c.false || "no, the statement does not hold"),
    "true: " + (c.true || "yes, the statement holds"),
  ];
}

/** Deterministic option orders. Only the first two of ``laya.vlm._permutations`` (identity, reversed) are
 * reproduced; later ones come from Python's seeded ``random.shuffle`` and are not ported. */
export function permutations(k, n) {
  const perms = [[...Array(k).keys()]];
  if (n > 1 && k > 1) perms.push([...Array(k).keys()].reverse());
  return perms.slice(0, Math.max(1, Math.min(n, 2)));
}

// ---------------------------------------------------------------------------------------------------------
// Sequence (laya.vlm.vlm_prefix + build_vlm_inputs, readout="terminator", truncate_left=False)
// ---------------------------------------------------------------------------------------------------------

const replaceAll = (s, a, b) => s.split(a).join(b);

/** ``prefix_ids``: the prefix text with each image replaced by its <image> run, tokenized as one string. */
export function prefixIds(encode, cfg, nImages) {
  const tx = cfg.text;
  const run = tx.fake_image_token + tx.global_image_token + tx.image_token.repeat(cfg.image.seq_len) + tx.fake_image_token;
  return encode(tx.prefix + run.repeat(nImages));
}

/** Split a state object like ``laya.vlm.split_state``: images are handled by the caller, the rest becomes text. */
export function stateText(state) {
  if (state === null || state === undefined) return "";
  if (typeof state === "string") return state;
  if (Array.isArray(state)) return pyJsonDumps(state, false);
  const rest = Object.fromEntries(Object.entries(state).filter(([k]) => k !== "image" && k !== "images"));
  return Object.keys(rest).length ? pyJsonDumps(rest, false) : "";
}

/**
 * One sequence for internal question ``q``: ``{ids, markers, option_span}``. ``encode(text)`` must tokenize
 * without special tokens. ``prefix`` is ``prefixIds(...)``.
 */
export function buildInputs(encode, cfg, prefix, text, q, order = null) {
  const tx = cfg.text;
  const maxLen = cfg.max_len, headMaxLen = cfg.head_max_len;
  const endId = cfg.token_ids.option_end;
  const opts = renderOptions(q);
  order = order ?? [...opts.keys()];
  let optIds = order.map((i) => encode(tx.option_bullet + replaceAll(opts[i], tx.option_end, " ")).slice(0, 48));
  const question = tx.question.replace("%s", q.t).replace("%s", replaceAll(String(q.ins), tx.strip_from_instructions, " "));
  let headIds = encode(question);
  const used = () => optIds.reduce((a, o) => a + o.length + 1, 0);
  let optBudget = headMaxLen - used();
  if (optBudget < 16) {
    const per = Math.max(4, Math.floor((headMaxLen - 16) / Math.max(1, optIds.length)) - 1);
    optIds = optIds.map((o) => o.slice(0, per));
    optBudget = headMaxLen - used();
  }
  if (headIds.length > Math.max(8, optBudget)) {
    const keep = Math.max(8, optBudget);
    const front = Math.floor(keep / 2);
    headIds = headIds.slice(0, front).concat(headIds.slice(headIds.length - (keep - front)));
  }
  const tail = [...headIds];
  const markers = [];
  const spanStart = tail.length;
  for (const o of optIds) {
    tail.push(...o, endId);
    markers.push(tail.length - 1);
  }
  const room = Math.max(0, maxLen - prefix.length - tail.length);
  const st = text ? encode(text).slice(0, room) : [];
  const off = prefix.length + st.length;
  if (prefix.length + tail.length > maxLen) throw new Error(`question + options + images exceed max_len=${maxLen}`);
  return {
    ids: [...prefix, ...st, ...tail],
    markers: markers.map((m) => m + off),
    option_span: [spanStart + off, tail.length + off],
    state_tokens: st.length,
    state_truncated: text ? encode(text).length - st.length : 0,
  };
}

// ---------------------------------------------------------------------------------------------------------
// Image: the processor's two LANCZOS hops (laya.preprocess._axis_weights / stage1_size)
// ---------------------------------------------------------------------------------------------------------

function sinc(x) {
  if (x === 0) return 1;
  const px = Math.PI * x;
  return Math.sin(px) / px;
}

function lanczos(x, a = 3) {
  x = Math.abs(x);
  if (x < 1e-12) return 1;
  return x < a ? sinc(x) * sinc(x / a) : 0;
}

/** Sparse ``_axis_weights``: for each output pixel, the first input index and its normalised weights. */
export function axisWeights(nIn, nOut) {
  const rows = [];
  if (nIn === nOut) {
    for (let i = 0; i < nOut; i++) rows.push({ lo: i, w: new Float64Array([1]) });
    return rows;
  }
  const scale = nIn / nOut;
  const stretch = Math.max(1, scale);
  const support = 3 * stretch;
  const span = Math.ceil(support) * 2 + 2;
  for (let i = 0; i < nOut; i++) {
    const centre = (i + 0.5) * scale;
    const lo = Math.max(0, Math.floor(centre - support + 0.5));
    const n = Math.max(0, Math.min(span, nIn - lo));
    const w = new Float64Array(n);
    let sum = 0;
    for (let j = 0; j < n; j++) {
      w[j] = lanczos((lo + j + 0.5 - centre) / stretch);
      sum += w[j];
    }
    for (let j = 0; j < n; j++) w[j] /= sum;
    rows.push({ lo, w });
  }
  return rows;
}

export function stage1Size(h, w, longest = 2048) {
  if (w >= h) {
    h = Math.trunc((longest * h) / w);
    h += h % 2;
    w = longest;
  } else {
    w = Math.trunc((longest * w) / h);
    w += w % 2;
    h = longest;
  }
  return [Math.max(h, 1), Math.max(w, 1)];
}

// round half to even, as torch does, then clamp to uint8
function toByte(v) {
  let r = Math.round(v);
  if (Math.abs(v % 1) === 0.5 && r % 2 !== 0) r -= 1;
  return r < 0 ? 0 : r > 255 ? 255 : r;
}

/** Planar uint8 ``[3, h, w]`` -> planar uint8 ``[3, oh, ow]``: the horizontal pass, rounded to uint8, then the
 * vertical pass, rounded again. That is the order and the rounding of torch's uint8 antialiased resize, which the
 * processor calls; keeping the intermediate in float instead is ~0.5 grey levels further off on noisy images. */
export function resizePlanar(src, h, w, oh, ow) {
  const wx = axisWeights(w, ow), wy = axisWeights(h, oh);
  const tmp = new Uint8Array(h * ow);
  const out = new Uint8Array(3 * oh * ow);
  for (let c = 0; c < 3; c++) {
    const base = c * h * w;
    for (let y = 0; y < h; y++) {
      const row = base + y * w;
      for (let x = 0; x < ow; x++) {
        const { lo, w: k } = wx[x];
        let s = 0;
        for (let j = 0; j < k.length; j++) s += k[j] * src[row + lo + j];
        tmp[y * ow + x] = toByte(s);
      }
    }
    const ob = c * oh * ow;
    for (let y = 0; y < oh; y++) {
      const { lo, w: k } = wy[y];
      for (let x = 0; x < ow; x++) {
        let s = 0;
        for (let j = 0; j < k.length; j++) s += k[j] * tmp[(lo + j) * ow + x];
        out[ob + y * ow + x] = toByte(s);
      }
    }
  }
  return out;
}

/** RGBA bytes (canvas ImageData) -> Float32 ``[3, S, S]`` pixel values normalised like the processor. */
export function pixelValues(rgba, h, w, cfg) {
  const planar = new Uint8Array(3 * h * w);
  for (let i = 0, n = h * w; i < n; i++) {
    planar[i] = rgba[4 * i];
    planar[n + i] = rgba[4 * i + 1];
    planar[2 * n + i] = rgba[4 * i + 2];
  }
  const [mh, mw] = stage1Size(h, w, cfg.image.stage1_longest_edge);
  const mid = resizePlanar(planar, h, w, mh, mw);
  const s = cfg.image.size;
  const px = resizePlanar(mid, mh, mw, s, s);
  const out = new Float32Array(px.length);
  const { mean, std } = cfg.image;
  for (let i = 0; i < px.length; i++) out[i] = (px[i] / 255 - mean) / std;
  return { data: out, bytes: px };
}

// ---------------------------------------------------------------------------------------------------------
// Answers (the tail of VLMAgent.predict)
// ---------------------------------------------------------------------------------------------------------

const round4 = (x) => Math.round(x * 1e4) / 1e4;

export function tempBucket(qt, k) {
  const size = k <= 2 ? "2" : k <= 5 ? "3-5" : k <= 10 ? "6-10" : "11+";
  return `${QTYPE_NAMES[qt]}:${size}`;
}

export function softmax(z) {
  const m = Math.max(...z);
  const e = z.map((v) => Math.exp(v - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
}

export function confidenceFromProbs(p, k) {
  if (k < 2) return 1;
  let ent = 0;
  for (const v of p.slice(0, k)) ent -= v * Math.log(Math.min(1, Math.max(1e-12, v)));
  return Math.min(1, Math.max(0, 1 - ent / Math.log(k)));
}

/** ``rows``: ``[{order, logits, actProb}]`` for one question. Returns the ``predict`` answer dict. */
export function answer(cfg, q, rows) {
  const k = renderOptions(q).length;
  const zSum = new Array(k).fill(0);
  let actSum = 0;
  for (const r of rows) {
    r.order.forEach((opt, j) => { zSum[opt] += r.logits[j]; });
    actSum += r.actProb;
  }
  const qt = QTYPES[q.t];
  const tScale = cfg.temperature_by_options[tempBucket(qt, k)] ?? cfg.temperature[qt];
  const p = softmax(zSum.map((z) => z / rows.length / Math.max(1e-3, tScale)));
  const conf = round4(confidenceFromProbs(p, k));
  const ext = { act_probability: round4(actSum / rows.length) };
  if (q.t === "choice") {
    const keys = Object.keys(q.crit);
    const best = p.indexOf(Math.max(...p));
    return { type: "choice", choice: keys[best], probabilities: Object.fromEntries(keys.map((kk, i) => [kk, round4(p[i])])),
             confidence: conf, action: ext };
  }
  if (q.t === "score") {
    return { type: "score", score: round4(p.reduce((a, v, i) => a + i * v, 0)),
             legend: Object.fromEntries(q.crit.map((c, i) => [String(i), c])),
             probabilities: Object.fromEntries(p.map((v, i) => [String(i), round4(v)])), confidence: conf, action: ext };
  }
  return { type: "noul", noul: round4(p[1]), confidence: round4(Math.max(p[1], 1 - p[1])), action: ext };
}
