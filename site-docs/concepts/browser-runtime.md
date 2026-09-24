# The browser runtime

The [browser demo](../tutorials/browser-demo.md) runs
[thaitea/laya-vision](https://huggingface.co/thaitea/laya-vision) on the visitor's device with
[ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/): WebGPU when the browser has it, WASM (CPU)
otherwise. It is a static page with no server, API, database or telemetry, and no build step. Its source is
[`web-demo/`](https://github.com/r33drichards/laya-vision/tree/main/web-demo).

## Three graphs

[`scripts/export_onnx.py`](https://github.com/r33drichards/laya-vision/blob/main/scripts/export_onnx.py) exports the
checkpoint in three graphs, split where the browser wants to reuse work:

- **vision**: the vision tower and connector, run once per image.
- **text**: token embeddings, the image merge, the language model, and the option-block attention mask built inside
  the graph; run once per question (and per option order).
- **head**: the type embedding, the two head transformer layers, the scorer and the act head. It is small enough
  that it always runs on WASM, where a WebGPU dispatch per operation would cost more than the arithmetic.

The backbone is exported from the checkpoint rather than reused from `HuggingFaceTB/SmolVLM-256M-Instruct`'s own
ONNX files, because the published checkpoints are trained with the language model unfrozen (and the current one keeps
20 of its 30 layers). For the previous checkpoint, all 272 text-model tensors and the connector differed from the base
model (largest absolute change 0.035 and 0.016); the 197 vision-tower tensors were identical.

## Reproducing predict

- [`laya.js`](https://github.com/r33drichards/laya-vision/blob/main/web-demo/laya.js) is a line-by-line port of
  `build_vlm_inputs` for the causal readout: the `<image>` run prefix, the state text, the question line, one option
  per line ending in `\n`, the per-option 48-token cap, the option and instruction budgets, the state cut to
  `max_len`. It also ports `render_options`, `_to_internal` (including Python's `json.dumps` for object instructions
  and the state), the temperatures by type and option-count bucket, and the answer schema.
- Image preprocessing reproduces the processor's two resizes: LANCZOS-3 with antialiasing to a longest edge of 2048,
  then to 512×512, each done horizontally then vertically with a round-and-clamp to uint8 after each pass, which is
  what torch's uint8 antialiased resize does.
- [`worker.js`](https://github.com/r33drichards/laya-vision/blob/main/web-demo/worker.js) runs the graphs off the
  page's main thread and caches the downloaded files in Cache Storage, keyed by each file's SHA-256 from
  `laya_web.json`, so a re-exported file under the same URL is downloaded again.

## Precision, and why fp16 needed care

The export writes fp32 plus three smaller variants: fp16 (weights and activations in float16, for WebGPU), and q8 and
q4 (8- and 4-bit weights with float activations, for WASM). How far each is from PyTorch:
[Browser demo files and checks](../reference/web-demo.md#how-close-each-precision-is).

The fp16 variant keeps every normalisation in float32. SmolVLM's residual stream reaches about 2,500 before the final
norm, so RMSNorm's `x**2` is about 6 million, far past float16's maximum of 65,504. The first published fp16 files
computed it in float16: on a real GPU every hidden state came out 0 and every option got the same probability.
onnxruntime's CPU backend runs those operations in float32, which is why the export's own validation passed; the
repository's test now checks the fix with real float16 arithmetic.

## What differs from Python

- **Decoding is the browser's.** JPEG decoding, EXIF orientation (browsers apply it, PIL does not), colour
  management and transparency can make the pixels differ from what `PIL.Image.open(...).convert("RGB")` gives,
  independently of the resize, which is matched.
- **Option orders.** `n_permutations` beyond 2 uses Python's seeded `random.shuffle`, which is not ported; the page
  offers 1 (as written) or 2 (plus reversed).
- **Numbers in the state.** `json.dumps` prints floats like `1.0` and `1e-05`; JavaScript's parsed JSON prints them
  as `1` and `0.00001`, so a state with such numbers tokenises differently. Integer-like object keys are reordered by
  JavaScript. Strings, booleans, `null`, integers and nesting match.
- **No batching or prefix caching**: one text-graph run per question and order, so time grows linearly with the
  number of questions.
