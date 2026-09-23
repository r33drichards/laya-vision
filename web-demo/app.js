// Page logic: reads the inputs, decodes the image to RGBA on a canvas, and talks to worker.js.
const $ = (id) => document.getElementById(id);
const worker = new Worker("worker.js", { type: "module" });

const DEFAULT_QUESTIONS = {
  damage: { type: "score", instructions: "How much damage does the item show?",
            criteria: ["none", "cosmetic: scratches or dents", "functional: parts broken or missing", "destroyed"] },
  category: { type: "choice", instructions: "What kind of item is this?",
              criteria: ["electronics", "clothing", "furniture", "food", "other"] },
  outdoors: { type: "noul", instructions: "Was the photo taken outdoors?" },
};
$("questions").value = JSON.stringify(DEFAULT_QUESTIONS, null, 2);

const params = new URLSearchParams(location.search);
// ?models=<url> wins; otherwise a local export in ./models/laya-vision/ if there is one (local development),
// else the published copy on the Hub (the GitHub Pages site, where the files are too big to host)
const HUB_MODELS = "https://huggingface.co/thaitea/laya-vision-web/resolve/main/";
if (params.get("models")) $("base-url").value = params.get("models");
else fetch(new URL("laya_web.json", new URL($("base-url").value, location.href)), { method: "HEAD" })
  .then((r) => { if (!r.ok) $("base-url").value = HUB_MODELS; })
  .catch(() => { $("base-url").value = HUB_MODELS; });
if (params.get("variant")) $("variant").value = params.get("variant");
if (params.get("backend")) $("backend").value = params.get("backend");

let image = null; // {rgba, width, height}
let loadedBaseUrl = null;
let ready = false; // a model is loaded in the worker

function setStatus(el, text, kind = "") {
  $(el).textContent = text;
  $(el).className = "status " + kind;
}

const mb = (n) => (n / 1e6).toFixed(1) + " MB";

/** Decode ``blob`` into the preview canvas and the RGBA buffer the worker resizes. The browser decodes (and applies
 * EXIF orientation); the model's own resize runs in the worker. */
async function showImage(blob) {
  const bmp = await createImageBitmap(blob);
  const c = $("preview");
  c.width = bmp.width;
  c.height = bmp.height;
  const ctx = c.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(bmp, 0, 0);
  image = { rgba: ctx.getImageData(0, 0, bmp.width, bmp.height).data, width: bmp.width, height: bmp.height };
}

$("image").addEventListener("change", async (ev) => {
  const file = ev.target.files[0];
  if (file) await showImage(file);
});

// preload the example photo (two turntables and a mixer) so a first visit can press Run straight away
fetch("example.jpg").then((r) => (r.ok ? r.blob() : null)).then((b) => { if (b && !image) return showImage(b); })
  .catch(() => {});

$("clear-image").addEventListener("click", () => {
  image = null;
  $("image").value = "";
  const c = $("preview");
  c.width = c.height = 0;
});

$("load").addEventListener("click", () => {
  $("load").disabled = $("run").disabled = true;
  ready = false;
  setStatus("model-status", "Loading…");
  $("progress").hidden = false;
  loadedBaseUrl = $("base-url").value.trim();
  worker.postMessage({ type: "load", baseUrl: loadedBaseUrl, variant: $("variant").value, backend: $("backend").value });
});

function parseState(text) {
  const t = text.trim();
  if (!t) return null;
  try { return JSON.parse(t); } catch { return t; } // not JSON: a plain text state, as predict accepts
}

$("run").addEventListener("click", () => {
  let questions;
  try {
    questions = JSON.parse($("questions").value);
    if (!questions || typeof questions !== "object" || Array.isArray(questions)) throw new Error("expected an object of questions");
  } catch (err) {
    setStatus("run-status", "Questions are not valid JSON: " + err.message, "error");
    return;
  }
  $("run").disabled = $("load").disabled = true;
  setStatus("run-status", "Running…");
  const images = image ? [{ rgba: image.rgba, width: image.width, height: image.height }] : [];
  worker.postMessage({ type: "run", images, stateObj: parseState($("state").value), questions,
                       nPermutations: Number($("perms").value) });
});

function bar(label, p, best) {
  const row = document.createElement("div");
  row.className = "bar" + (best ? " best" : "");
  const name = document.createElement("span");
  name.className = "label";
  name.textContent = label;
  const track = document.createElement("span");
  track.className = "track";
  const fill = document.createElement("span");
  fill.className = "fill";
  fill.style.width = (100 * p).toFixed(1) + "%";
  track.append(fill);
  const val = document.createElement("span");
  val.className = "val";
  val.textContent = (100 * p).toFixed(1) + "%";
  row.append(name, track, val);
  return row;
}

function renderAnswers(result, details, timing) {
  const root = $("answers");
  root.replaceChildren();
  for (const [qid, a] of Object.entries(result.answers)) {
    const card = document.createElement("div");
    card.className = "card";
    const h = document.createElement("h3");
    let headline;
    if (a.type === "choice") headline = `choice: ${a.choice}`;
    else if (a.type === "score") headline = `score: ${a.score.toFixed(2)} (expected level)`;
    else headline = `noul: P(true) = ${a.noul.toFixed(3)}`;
    h.textContent = `${qid} — ${headline}`;
    card.append(h);
    if (a.type === "noul") {
      card.append(bar("false", 1 - a.noul, a.noul < 0.5), bar("true", a.noul, a.noul >= 0.5));
    } else {
      const probs = Object.entries(a.probabilities);
      const top = Math.max(...probs.map(([, p]) => p));
      for (const [k, p] of probs) card.append(bar(a.type === "score" ? `${k}: ${a.legend[k]}` : k, p, p === top));
    }
    const meta = document.createElement("p");
    meta.className = "meta";
    const d = details[qid];
    meta.textContent = `confidence ${a.confidence.toFixed(3)} (1 − normalised entropy) · ${d.tokens} tokens` +
      (d.state_truncated ? ` · state cut by ${d.state_truncated} tokens to fit max_len` : "");
    card.append(meta);
    root.append(card);
  }
  $("raw").textContent = JSON.stringify(result, null, 2);
  $("results-panel").hidden = false;
  setStatus("run-status", `Done in ${(timing.total_ms / 1000).toFixed(2)} s (vision ${(timing.vision_ms / 1000).toFixed(2)} s).`, "ok");
}

async function showValidation(msg) {
  // validation.json is written by the exporter; it is optional, the page works without it
  try {
    const base = new URL(loadedBaseUrl, location.href);
    const v = await (await fetch(new URL("validation.json", base.href.endsWith("/") ? base : base.href + "/"))).json();
    const key = msg.variant;
    const rows = v.results?.[key];
    if (!rows) return "";
    const dp = Math.max(...rows.map((r) => r.max_abs_prob_diff));
    const agree = rows.filter((r) => r.same_argmax).length;
    return ` Export check for ${key}: max |Δp| ${dp.toExponential(1)} vs PyTorch, same top answer ${agree}/${rows.length}.`;
  } catch {
    return "";
  }
}

worker.onmessage = async ({ data }) => {
  if (data.type === "progress") {
    if (data.phase === "run") {
      setStatus("run-status", `Answered ${data.label}…`);
      return;
    }
    const p = $("progress");
    p.max = data.total || 1;
    p.value = data.loaded;
    setStatus("model-status", `${data.label}: ${mb(data.loaded)}${data.total ? " / " + mb(data.total) : ""}${data.cached ? " (from cache)" : ""}`);
  } else if (data.type === "loaded") {
    $("progress").hidden = true;
    const size = Object.entries(data.files).filter(([f]) => f.endsWith(data.suffix + ".onnx") &&
      (data.suffix || !/_(fp16|q8|q4)\.onnx$/.test(f))).reduce((a, [, m]) => a + m.bytes, 0);
    const note = await showValidation(data);
    setStatus("model-status", `Loaded ${data.variant} on ${data.backend.toUpperCase()}${data.adapter ? " [" + data.adapter + "]" : ""} (${mb(size)}) in ${data.seconds.toFixed(1)} s` +
      `${data.isolated || data.backend !== "wasm" ? "" : ", single-threaded (page is not cross-origin isolated)"}.${note}`, "ok");
    ready = true;
    $("load").disabled = $("run").disabled = false;
  } else if (data.type === "result") {
    $("run").disabled = $("load").disabled = false;
    renderAnswers(data.result, data.details, data.timing);
    window.__lastResult = data.result; // for the headless smoke test
  } else if (data.type === "error") {
    $("progress").hidden = true;
    $("load").disabled = false;
    $("run").disabled = !ready;
    setStatus(ready ? "run-status" : "model-status", "Error: " + data.message, "error");
    window.__lastError = data.message;
  }
};
