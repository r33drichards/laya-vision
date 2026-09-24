# Browser demo files and checks

The [browser demo](../tutorials/browser-demo.md) is the static page in
[`web-demo/`](https://github.com/r33drichards/laya-vision/tree/main/web-demo), served at
<https://r33drichards.github.io/laya-vision/demo/>. How it works: [The browser runtime](../concepts/browser-runtime.md).

## Exported files

[`scripts/export_onnx.py`](https://github.com/r33drichards/laya-vision/blob/main/scripts/export_onnx.py) writes,
into `--out`:

| File | What | Size |
|---|---|---|
| `vision.onnx` (+ `_fp16`, `_q8`, `_q4`) | vision tower + connector: `pixel_values [n,3,512,512]` → `image_features [n,64,576]` | 374 / 187 / 110 / 64 MB |
| `text.onnx` (+ variants) | token embeddings, image merge, language model, the option-block 4D mask built in-graph from `option_span`: → `last_hidden_state [1,L,576]` | 540 / 271 / 234 / 181 MB |
| `head.onnx` (+ variants) | type embedding, 2 head transformer layers, scorer, act head: → `logits [K]`, `act_logits [2]` | 34 / 17 / 11 / 7 MB |
| `laya_web.json` | temperatures, token ids, text templates, budgets, image settings, file sizes and SHA-256 | |
| `tokenizer/` | the checkpoint's `tokenizer.json` and `tokenizer_config.json` | |
| `validation.json` | ONNX (onnxruntime, CPU) vs `VLMAgent.predict`, per variant; the page shows its summary after loading | |

Totals: fp32 948 MB, fp16 475 MB, q8 355 MB, q4 252 MB.

- **fp16**: weights and activations in float16, with every normalisation kept in float32 (they square residuals of
  a few thousand, which overflows float16).
- **q8** / **q4**: weight-only `MatMulNBits` (block 32, symmetric; activations stay float). They keep the 113 MB
  token-embedding table in fp32, which is why the text graph shrinks less than the vision graph.
- Dynamic int8 (`quantize_dynamic`, 8-bit activations) was tried first and dropped: it moved probabilities by up to
  0.62 and flipped the top answer on 4 of the 9 validation questions.

## Pinned runtime

The page loads exactly these, from jsDelivr, with no other third-party requests:

- `onnxruntime-web@1.30.0` (`dist/ort.webgpu.min.mjs`, which includes the WASM backend, and its `.wasm` files)
- `@huggingface/tokenizers@0.2.0` (`dist/tokenizers.min.mjs`), the tokenizer library transformers.js uses

`web-demo/package.json` pins the same versions for the tests; the smoke test serves those URLs from `node_modules`,
so a version drift fails there.

## How close each precision is

`--validate` runs three states (a 640×480 drawing with a note, a 210×160 noise frame, a text-only state) × the three
questions of the Home page example through onnxruntime (CPU) and through PyTorch:

| Variant | Size | max \|Δp\| | max \|Δlogit\| | same top answer |
|---|---|---|---|---|
| fp32 | 948 MB | 2.0e-6 | 2.5e-5 | 9/9 |
| fp16 | 475 MB | 2.5e-3 | 3.2e-2 | 9/9 |
| q8 | 355 MB | 1.8e-2 | 0.25 | 9/9 |
| q4 | 252 MB | 0.17 | 1.7 | 6/9 |
| int8 dynamic (dropped) | 329 MB | 0.62 | 5.3 | 5/9 |

onnxruntime's CPU backend runs some float16 operations in float32, so this check alone cannot catch a float16
overflow. The fp16 graphs are also checked with onnx's reference evaluator in true float16 arithmetic on a real
photo: text-graph option hidden states within 0.06 of fp32, vision features within 0.09 (of values up to 93).

## Other checks

- **Token ids.** `tests/test_web_demo.py` commits `web-demo/fixtures/parity.json`: ids, markers and option spans from
  Python for 5 cases (the README example; a text-only state with unicode, object instructions, newlines in options
  and an `<end_of_utterance>` in the instructions; two images; a state long enough to be cut; 12 long options that
  trigger the option and instruction budgets), each in two option orders, 18 rows. The Node run of `laya.js` with
  `@huggingface/tokenizers` matches all 18 rows exactly, and the Python side is regenerated in the test, so a format
  change in `laya/vlm.py` fails until the fixture is refreshed.
- **Pixels.** The JavaScript resize against the Hugging Face processor on four images (smooth 640×480, noise
  210×160, noise 100×300, 2600×1300): mean difference 0.02 to 0.05 grey levels, at most 2 levels.
- **Headless browser.** `web-demo/smoke_test.mjs` drives the page in headless Chromium: load, upload a PNG (the
  export's 640×480 validation drawing), run the questions with the example note as state. Against PyTorch on the
  same image (P(damage level 0), P(category = other), P(outdoors)):

  | Run | Load | Run (3 questions) | damage p0 | other | outdoors |
  |---|---|---|---|---|---|
  | PyTorch `predict` | | | 0.4742 | 0.7840 | 0.7070 |
  | fp32, WASM | 25 s | 17.4 s | 0.4749 | 0.7824 | 0.7073 |
  | fp16, WASM | 24 s | 15.6 s | 0.4747 | 0.7848 | 0.7081 |
  | q8, WASM | 11 s | 13.1 s | 0.4802 | 0.7758 | 0.6917 |
  | q4, WASM | 10 s | 18.9 s | 0.3509 | 0.6158 | 0.7126 |
  | q8, WebGPU on SwiftShader (software) | 16 s | 523 s | 0.4801 | 0.7755 | 0.6916 |

  The fp32 browser run differs from PyTorch by up to 0.0016 where onnxruntime on the processor's own pixels differs
  by 2e-6; that remainder is the JavaScript image path. Timings are single-threaded WASM on a shared 4-core container
  under load: they show that it runs, not how fast it is.
- **A real GPU.** One report, Chrome on macOS with an Apple GPU, fp16 on WebGPU, the fixed export: a photo of a
  washer-dryer with the three example questions ran in 2.0 s (vision 1.6 s), with probabilities within 0.04 of
  PyTorch on the same photo (for example P(category = food) 0.442 against 0.461; the photo compared in PyTorch was a
  re-encoded copy).

## Not verified

- Other devices: no phone, no Safari or Firefox, and only the one GPU above. The fp32 files need about 1 GB of
  downloads and more than that in memory; a phone will likely not manage fp32.
- GPU timings beyond the single report above.

## Probability caveats

The page repeats these under the results:

- The probabilities are a softmax over the options you list, conditional on that list. A question whose right answer
  is not among them still gets a confident-looking distribution; add an option such as "none of these" if that
  matters.
- The per-type temperatures were fitted on the validation splits of the training data. On other images they are
  scores, not calibrated probabilities, and a high value does not show that the answer is correct.
- Quantised files are a different model from the PyTorch one by the amounts measured above.
