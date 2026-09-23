// Inference worker: loads the exported ONNX graphs with onnxruntime-web (WebGPU, else WASM) and answers
// {type: "run"} messages with the same answer schema as VLMAgent.predict. All model work happens here so the
// page stays responsive. Nothing leaves the browser: the only network requests are the pinned runtime files from
// jsDelivr and the model files from the URL the page gives us.
import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.webgpu.min.mjs";
import { Tokenizer } from "https://cdn.jsdelivr.net/npm/@huggingface/tokenizers@0.2.0/dist/tokenizers.min.mjs";
import * as laya from "./laya.js";

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/";
ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;

const CACHE = "laya-vision-models-v1";
let state = null; // {cfg, tok, sessions, backend, variant}

const send = (type, data = {}) => self.postMessage({ type, ...data });

async function cached(key) {
  try {
    const cache = await caches.open(CACHE);
    return { cache, hit: await cache.match(key) };
  } catch {
    return { cache: null, hit: null };
  }
}

/** GET ``url`` as an ArrayBuffer with progress messages, through the Cache Storage API when it is available. The
 * cache key carries the file's SHA-256 from laya_web.json, so a re-export under the same URL is fetched again instead
 * of served stale (the first fp16 export was broken on real GPUs and replaced in place). */
async function fetchBytes(url, label, sha256) {
  const key = sha256 ? `${url}?sha256=${sha256}` : url;
  const { cache, hit } = await cached(key);
  let res = hit;
  if (!res) {
    res = await fetch(url);
    if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  }
  const total = Number(res.headers.get("content-length")) || 0;
  const reader = res.clone().body.getReader();
  const parts = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value);
    got += value.length;
    send("progress", { phase: "load", label, loaded: got, total, cached: !!hit });
  }
  const buf = new Uint8Array(got);
  let o = 0;
  for (const p of parts) { buf.set(p, o); o += p.length; }
  if (cache && !hit) {
    try { await cache.put(key, res); } catch { /* quota: fine, the HTTP cache may still help */ }
  }
  return buf;
}

/** ``{ep, f16, adapter}``: WebGPU when asked for (or "auto") and an adapter exists, else WASM. ``f16`` says whether
 * the adapter can run the fp16 graphs (the ``shader-f16`` feature); software adapters usually cannot. */
async function pickBackend(requested) {
  if (requested === "wasm") return { ep: "wasm", f16: true, adapter: "" };
  let adapter = null;
  try { adapter = navigator.gpu ? await navigator.gpu.requestAdapter() : null; } catch { adapter = null; }
  if (!adapter) {
    if (requested === "webgpu") throw new Error("WebGPU was requested but no adapter is available in this browser.");
    return { ep: "wasm", f16: true, adapter: "" };
  }
  const info = adapter.info || {};
  const name = [info.vendor, info.architecture, info.description].filter(Boolean).join(" ") || "unknown adapter";
  return { ep: "webgpu", f16: adapter.features.has("shader-f16"), adapter: name + (adapter.isFallbackAdapter ? " (fallback)" : "") };
}

async function load({ baseUrl, variant, backend }) {
  const t0 = performance.now();
  const base = new URL(baseUrl, self.location.href);
  if (!base.pathname.endsWith("/")) base.pathname += "/";
  const cfg = await (await fetch(new URL("laya_web.json", base), { cache: "no-cache" })).json();
  if (cfg.format_version !== 1 || cfg.readout !== "terminator") throw new Error("unsupported laya_web.json (format or readout)");
  const [tj, tc] = await Promise.all(cfg.tokenizer.map((p) => fetch(new URL(p, base)).then((r) => r.json())));
  const tok = new Tokenizer(tj, tc);
  const { ep, f16, adapter } = await pickBackend(backend);
  if (variant === "auto") variant = ep === "webgpu" && f16 ? "fp16" : "q8";
  if (variant === "fp16" && !f16) throw new Error("this WebGPU adapter has no shader-f16 support; pick fp32, q8 or q4, or WASM");
  const suffix = variant === "fp32" ? "" : "_" + variant;
  const sessions = {};
  const timings = {};
  for (const name of ["vision", "text", "head"]) {
    const file = `${name}${suffix}.onnx`;
    if (!cfg.files[file]) throw new Error(`${file} is not in laya_web.json; export it with --quantize ${variant}`);
    const t = performance.now();
    const bytes = await fetchBytes(new URL(file, base).href, file, cfg.files[file].sha256);
    timings[`download ${file}`] = performance.now() - t;
    const t2 = performance.now();
    // the tiny head graph runs on WASM: a WebGPU dispatch per op costs more than the arithmetic
    const eps = name === "head" ? ["wasm"] : [ep];
    sessions[name] = await ort.InferenceSession.create(bytes, { executionProviders: eps, graphOptimizationLevel: "all" });
    timings[`session ${file}`] = performance.now() - t2;
  }
  state = { cfg, tok, sessions, backend: ep, variant };
  send("loaded", { backend: ep, adapter, variant, files: cfg.files, suffix, seconds: (performance.now() - t0) / 1000, timings,
                   source: cfg.source, temperature: cfg.temperature, isolated: self.crossOriginIsolated });
}

const i64 = (arr) => new ort.Tensor("int64", BigInt64Array.from(arr.map((v) => BigInt(v))), [arr.length]);

async function run({ images, stateObj, questions, nPermutations }) {
  if (!state) throw new Error("load a model first");
  const { cfg, tok, sessions } = state;
  const encode = (s) => tok.encode(s, { add_special_tokens: false }).ids;
  const t0 = performance.now();
  const d = cfg.hidden_size, L = cfg.image.seq_len, S = cfg.image.size;

  // images -> features [n, L, d]
  let feats = new ort.Tensor("float32", new Float32Array(0), [0, L, d]);
  const tImg = performance.now();
  if (images.length) {
    const chunks = [];
    for (const im of images) {
      const px = laya.pixelValues(im.rgba, im.height, im.width, cfg);
      const out = await sessions.vision.run({ pixel_values: new ort.Tensor("float32", px.data, [1, 3, S, S]) });
      chunks.push(out.image_features.data.slice());
    }
    const all = new Float32Array(chunks.reduce((a, c) => a + c.length, 0));
    let o = 0;
    for (const c of chunks) { all.set(c, o); o += c.length; }
    feats = new ort.Tensor("float32", all, [images.length, L, d]);
  }
  const visionMs = performance.now() - tImg;

  const prefix = laya.prefixIds(encode, cfg, images.length);
  const text = laya.stateText(stateObj);
  const answers = {};
  const details = {};
  let nTokens = 0;
  const qids = Object.keys(questions);
  for (const [qi, qid] of qids.entries()) {
    const q = laya.toInternal(questions[qid]);
    const k = laya.renderOptions(q).length;
    if (k < 2) throw new Error(`question ${qid} needs at least two options`);
    const rows = [];
    for (const order of laya.permutations(k, nPermutations)) {
      const it = laya.buildInputs(encode, cfg, prefix, text, q, order);
      if (it.markers.length !== k) throw new Error(`question ${qid}: options exceed head_max_len`);
      nTokens += it.ids.length;
      const ids = new ort.Tensor("int64", BigInt64Array.from(it.ids.map(BigInt)), [1, it.ids.length]);
      const h = await sessions.text.run({ input_ids: ids, image_features: feats, option_span: i64(it.option_span) });
      const out = await sessions.head.run({ hidden: h.last_hidden_state, marker_pos: i64(it.markers),
                                            qtype: i64([laya.QTYPES[q.t]]) });
      const act = laya.softmax(Array.from(out.act_logits.data));
      rows.push({ order, logits: Array.from(out.logits.data), actProb: act[0] });
      details[qid] = { tokens: it.ids.length, state_tokens: it.state_tokens, state_truncated: it.state_truncated };
      send("progress", { phase: "run", label: `question ${qi + 1}/${qids.length}`, loaded: qi + 1, total: qids.length });
    }
    answers[qid] = laya.answer(cfg, q, rows);
  }
  send("result", {
    result: { model: "laya-vlm", answers, usage: { input_tokens: nTokens, output_tokens: 0, images: images.length } },
    details,
    timing: { vision_ms: visionMs, total_ms: performance.now() - t0 },
  });
}

self.onmessage = async ({ data }) => {
  try {
    if (data.type === "load") await load(data);
    else if (data.type === "run") await run(data);
  } catch (err) {
    send("error", { message: String(err?.message || err) });
  }
};
