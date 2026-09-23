# Laya Vision in the browser

A static page that runs the [thaitea/laya-vision](https://huggingface.co/thaitea/laya-vision) decision model on
the visitor's device with [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/): WebGPU when the browser
has it, WASM (CPU) otherwise. You pick an image, edit the state and the questions (the same JSON as
`agent.predict`'s arguments), and get the same answer schema back. There is no server, API, database or telemetry,
and no build step. The image never leaves the page.

**Status: experimental, partial.** What has been checked, and how, is in [What works](#what-works); what has not
is in [What does not, or is untested](#what-does-not-or-is-untested).

## Where the model files live

The ONNX files (250 MB to 950 MB, depending on precision) are too big for GitHub Pages, so they are published to
the Hub at [thaitea/laya-vision-web](https://huggingface.co/thaitea/laya-vision-web) by
`modal run modal_app.py::publish_web` (the export below, run on Modal, then uploaded with the `huggingface-thaitea`
secret). The page loads `?models=<url>` if given, else `./models/laya-vision/` if a local export is there, else
the Hub copy. The public site is https://r33drichards.github.io/laya-vision/, deployed by
`.github/workflows/pages.yml` on every push to `main` that touches the page.

To produce the files locally from a checkpoint, from the repository root:

```bash
pip install onnx onnxruntime onnxscript          # export and validation only; the page needs none of this
python scripts/export_onnx.py thaitea/laya-vision --out web-demo/models/laya-vision --quantize fp16,q8,q4 --validate
```

This writes, into `--out`:

| File | What | Size |
|---|---|---|
| `vision.onnx` (+ `_fp16`, `_q8`, `_q4`) | vision tower + connector: `pixel_values [n,3,512,512]` → `image_features [n,64,576]` | 374 / 187 / 110 / 64 MB |
| `text.onnx` (+ variants) | token embeddings, image merge, language model, the option-block 4D mask built in-graph from `option_span`: → `last_hidden_state [1,L,576]` | 540 / 271 / 234 / 181 MB |
| `head.onnx` (+ variants) | type embedding, 2 head transformer layers, scorer, act head: → `logits [K]`, `act_logits [2]` | 34 / 17 / 11 / 7 MB |
| `laya_web.json` | temperatures, token ids, text templates, budgets, image settings, file sizes and SHA-256 | |
| `tokenizer/` | the checkpoint's `tokenizer.json` and `tokenizer_config.json` | |
| `validation.json` | ONNX (onnxruntime, CPU) vs `VLMAgent.predict`, per variant; the page shows its summary after loading | |

Totals: fp32 948 MB, fp16 475 MB, q8 355 MB, q4 252 MB. `q8`/`q4` are weight-only `MatMulNBits` (block 32,
symmetric; activations stay float) and keep the 113 MB token-embedding table in fp32, which is why the text graph
shrinks less than the vision graph. Dynamic int8 (`quantize_dynamic`, 8-bit activations) was tried first and
dropped: it moved probabilities by up to 0.62 and flipped the top answer on 4 of the 9 validation questions.

Why export the backbone instead of reusing `HuggingFaceTB/SmolVLM-256M-Instruct`'s `onnx/` folder: the published
checkpoint was trained with the language model unfrozen. Compared tensor by tensor with the base model, all 272
text-model tensors and the connector differ (largest absolute change 0.035 and 0.016); the 197 vision-tower tensors
are identical. So the base repo's `vision_encoder.onnx` (which includes the connector) and `decoder_model_merged.onnx`
would both be wrong for this checkpoint, and the head needs its own graph anyway.

Only the `"terminator"` readout (SmolVLM, SmolVLM2 checkpoints) is exportable; ModernVBERT's `"mask"` sequence is
not ported. Image splitting (`image_split_edge`) is not supported.

## Run locally

```bash
cd web-demo
python3 -m http.server 8080        # after exporting into web-demo/models/laya-vision (git-ignored)
```

Open `http://localhost:8080`, press **Load model**, choose an image, **Run**. "Auto" precision picks fp16 on a
WebGPU adapter with `shader-f16`, else q8. Model files are kept in the browser's Cache Storage after the first load.

`python3 -m http.server` does not send cross-origin isolation headers, so WASM runs single-threaded. Any static
HTTPS host works for deployment; `_headers` (Cloudflare Pages / Netlify syntax) adds
`Cross-Origin-Opener-Policy` / `Cross-Origin-Embedder-Policy` for multi-threaded WASM and `Referrer-Policy:
no-referrer`. Under COEP, model files on another origin must be served with CORS
(`Access-Control-Allow-Origin`) or `Cross-Origin-Resource-Policy: cross-origin`.

## Pins

The page loads exactly these, from jsDelivr, with no other third-party requests:

- `onnxruntime-web@1.30.0` (`dist/ort.webgpu.min.mjs`, which includes the WASM backend, and its `.wasm` files)
- `@huggingface/tokenizers@0.2.0` (`dist/tokenizers.min.mjs`), the tokenizer library transformers.js uses

`package.json` pins the same versions for the tests; the smoke test serves those URLs from `node_modules`, so a
version drift fails there.

## How the page reproduces `predict`

- `laya.js` is a line-by-line port of `build_vlm_inputs` for the terminator readout (the `<image>` run prefix,
  state text, question line, one option per line ending in `\n`, the per-option 48-token cap, the option and
  instruction budgets, the state cut to `max_len`), of `render_options`, `_to_internal` (including Python's
  `json.dumps` for object instructions and the state), temperatures by type and option-count bucket, and the answer
  schema (`choice`, `score` expected level, `noul`, `confidence`, `action.act_probability`).
- Image preprocessing reproduces the processor's two resizes: LANCZOS-3 with antialiasing to a longest edge of
  2048, then to 512×512, each done horizontally then vertically with a round-and-clamp to uint8 after each pass,
  which is what torch's uint8 antialiased resize does.
- `worker.js` runs the vision graph once per image, then the text and head graphs once per question (and per
  option order), and returns the answers. The head always runs on WASM; it is too small to be worth a GPU dispatch.

## What works

Checked on Linux, CPU only, in this repository's test setup:

- **Export parity (fp32).** `--validate` (output committed as `fixtures/export_validation.json`) runs three states (a 640×480 drawing with a note, a 210×160 noise frame,
  a text-only state) × the three README questions through onnxruntime and through PyTorch. Largest difference:
  2.0e-6 in probability, 2.5e-5 in raw logit, same top answer 9/9.
- **Quantised variants**, same inputs, onnxruntime CPU vs PyTorch (from `validation.json`):

  | Variant | Size | max \|Δp\| | max \|Δlogit\| | same top answer |
  |---|---|---|---|---|
  | fp32 | 948 MB | 2.0e-6 | 2.5e-5 | 9/9 |
  | fp16 | 475 MB | 1.8e-3 | 1.9e-2 | 9/9 |
  | q8 | 355 MB | 1.8e-2 | 0.25 | 9/9 |
  | q4 | 252 MB | 0.17 | 1.7 | 6/9 |
  | int8 dynamic (dropped) | 329 MB | 0.62 | 5.3 | 5/9 |

- **Token ids.** `tests/test_web_demo.py` commits `fixtures/parity.json`: ids, markers and option spans from
  Python for 5 cases (README example; text-only state with unicode, object instructions, newlines in options and an
  `<end_of_utterance>` in the instructions; two images; a state long enough to be cut; 12 long options that trigger
  the option and instruction budgets), each in two option orders, 18 rows. The Node run of `laya.js` with
  `@huggingface/tokenizers` matches all 18 rows exactly, and the Python side is regenerated in the test, so a
  format change in `laya/vlm.py` fails until the fixture is refreshed.
- **Pixels.** The JavaScript resize against the Hugging Face processor on four images (smooth 640×480, noise
  210×160, noise 100×300, 2600×1300): mean difference 0.02 to 0.05 grey levels, at most 2 levels.
- **Headless browser.** `smoke_test.mjs` drives the page in the pre-installed headless Chromium: load, upload a
  PNG (the export's 640×480 validation drawing), run the README questions with the README note as state. Against
  PyTorch on the same image (P(damage level 0), P(category = other), P(outdoors)):

  | Run | Load | Run (3 questions) | damage p0 | other | outdoors |
  |---|---|---|---|---|---|
  | PyTorch `predict` | | | 0.4742 | 0.7840 | 0.7070 |
  | fp32, WASM | 25 s | 17.4 s | 0.4749 | 0.7824 | 0.7073 |
  | fp16, WASM | 24 s | 15.6 s | 0.4747 | 0.7848 | 0.7081 |
  | q8, WASM | 11 s | 13.1 s | 0.4802 | 0.7758 | 0.6917 |
  | q4, WASM | 10 s | 18.9 s | 0.3509 | 0.6158 | 0.7126 |
  | q8, WebGPU on SwiftShader (software) | 16 s | 523 s | 0.4801 | 0.7755 | 0.6916 |

  The fp32 browser run differs from PyTorch by up to 0.0016 where onnxruntime on the processor's own pixels
  differs by 2e-6; that remainder is the JavaScript image path (resize within 2 grey levels, see above). With the
  text-only state the fp32 browser answers equal PyTorch's to all four printed decimals, and `Option orders: 2`
  (as written + reversed) runs. Timings are single-threaded WASM on a shared 4-core container under heavy load
  from other jobs, served from local disk: they show that it runs, not how fast it is.

## What does not, or is untested

- **No real GPU was available.** WebGPU ran only on SwiftShader, Chromium's software adapter, which has no
  `shader-f16`: the q8 graphs gave the same answers as WASM to 1e-4, so the WebGPU kernels for these graphs work,
  but fp16 on WebGPU (the default on a desktop GPU) has not run anywhere, and there are no GPU timings.
- **fp16 and real float16 arithmetic.** The first published fp16 files gave exactly uniform probabilities on
  WebGPU (Apple Metal): the converter had put RMSNorm's `x**2` in float16, and the residual stream (~2.5e3 before the
  final norm) overflows it, so every hidden state came out 0. onnxruntime's CPU backend runs those ops in float32,
  which is why `--validate` passed. The export now keeps every normalisation in float32; checked with onnx's
  reference evaluator in true float16 on a real photo (text graph: option hidden states within 0.06 of fp32;
  vision features within 0.09 of fp32, against 57 off before) and by `tests/test_web_demo.py`. Not yet confirmed on
  a real GPU.
- **Real devices and timings.** Only the headless CPU run above; no phone, no Safari or Firefox. The fp32 files
  need about 1 GB of downloads and more than that in memory; a phone will likely not manage fp32.
- **Decoding is the browser's.** JPEG decoding, EXIF orientation (browsers apply it, PIL does not), colour
  management and transparency can make the pixels differ from what `PIL.Image.open(...).convert("RGB")` gives,
  independently of the resize, which is matched.
- **Option orders.** `n_permutations` beyond 2 uses Python's seeded `random.shuffle`, which is not ported; the page
  offers 1 (as written) or 2 (plus reversed).
- **Numbers in the state.** `json.dumps` prints floats like `1.0` and `1e-05`; JavaScript's parsed JSON prints them
  as `1` and `0.00001`, so a state with such numbers tokenises differently. Integer-like object keys are reordered
  by JavaScript. Strings, booleans, `null`, integers and nesting match.
- **No batching or prefix caching**: one text-graph run per question and order, so time grows linearly with the
  number of questions.

## Probability caveats

The page repeats these under the results:

- The probabilities are a softmax over the options you list, conditional on that list. A question whose right answer
  is not among them still gets a confident-looking distribution; add an option such as "none of these" if that
  matters.
- The per-type temperatures were fitted on the validation splits of the training data, where calibrated ECE is 0.02
  to 0.03. On other images they are scores, not calibrated probabilities, and a high value does not show that the
  answer is correct.
- Quantised files are a different model from the PyTorch one by the amounts measured above.

## Tests

```bash
python -m pytest -q tests/test_web_demo.py            # fixture parity always; Node parts after `npm install` here
cd web-demo && npm install && PLAYWRIGHT_BROWSERS_PATH=/opt/pw-browsers \
  node smoke_test.mjs --models ./models/laya-vision/ --image some.png --variant q8 --backend wasm
```

`npm install` only fetches test dependencies (`@huggingface/tokenizers`, `onnxruntime-web`, `playwright-core`);
the smoke test uses an existing Chromium and never downloads a browser.
