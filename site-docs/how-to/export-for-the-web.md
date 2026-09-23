# Export for the browser

The [browser demo](../tutorials/browser-demo.md) runs ONNX graphs exported from a checkpoint. The published
checkpoint's export is already hosted at [thaitea/laya-vision-web](https://huggingface.co/thaitea/laya-vision-web);
this page is for re-exporting it, exporting another checkpoint, or hosting the files yourself.

Only the causal `"terminator"` readout (SmolVLM and SmolVLM2 checkpoints) exports; ModernVBERT's `"mask"` sequence
is not ported, and image splitting (`image_split_edge`) is not supported.

## Export and publish in one step (Modal)

```bash
modal run modal_app.py::publish_web                                       # thaitea/laya-vision -> thaitea/laya-vision-web
modal run modal_app.py::publish_web --checkpoint user/model --repo user/model-web --quantize fp16,q8
```

The job runs the export below on Modal, validates every precision against PyTorch, and uploads the folder to the
Hub model repo with the `huggingface-thaitea` secret. Re-publishing under the same repo is safe for visitors: the
page keys its file cache by each file's SHA-256 from `laya_web.json`, so a replaced file is downloaded again.

## Export locally

From the repository root:

```bash
pip install onnx onnxruntime onnxscript          # export and validation only; the page needs none of this
python scripts/export_onnx.py thaitea/laya-vision --out web-demo/models/laya-vision --quantize fp16,q8,q4 --validate
```

`--validate` runs a few fixed inputs through onnxruntime and through `VLMAgent.predict`, writes `validation.json`, and
exits non-zero if fp32 is further than `--tol` (default 1e-3 in probability). What each file is:
[Browser demo files and checks](../reference/web-demo.md#exported-files).

## Point the page at your files

The page loads its model folder from, in order:

1. `?models=<url>` in the page address, for example
   `https://r33drichards.github.io/laya-vision/demo/?models=https://huggingface.co/user/model-web/resolve/main/`;
2. `./models/laya-vision/` next to the page, if a local export is there;
3. the Hub copy, `thaitea/laya-vision-web`.

Files on another origin must be served with CORS (`Access-Control-Allow-Origin`); the Hub does this.

## Serve the page locally

```bash
cd web-demo
python3 -m http.server 8080        # after exporting into web-demo/models/laya-vision (git-ignored)
```

Open `http://localhost:8080`, press **Load model**, and **Run**.

`python3 -m http.server` does not send cross-origin isolation headers, so WASM runs single-threaded. Any static
HTTPS host works for deployment; `web-demo/_headers` (Cloudflare Pages / Netlify syntax) adds
`Cross-Origin-Opener-Policy` / `Cross-Origin-Embedder-Policy` for multi-threaded WASM and `Referrer-Policy:
no-referrer`. Under COEP, model files on another origin must be served with CORS or
`Cross-Origin-Resource-Policy: cross-origin`. GitHub Pages, where this site and the demo are hosted, cannot set
these headers, so WASM runs single-threaded there too.

## Test the page

```bash
python -m pytest -q tests/test_web_demo.py            # fixture parity always; Node parts after `npm install` in web-demo/
cd web-demo && npm install && PLAYWRIGHT_BROWSERS_PATH=/opt/pw-browsers \
  node smoke_test.mjs --models ./models/laya-vision/ --image some.png --variant q8 --backend wasm
```

`npm install` only fetches test dependencies (`@huggingface/tokenizers`, `onnxruntime-web`, `playwright-core`); the
smoke test uses an existing Chromium and never downloads a browser. Without `--image` it runs on the example photo
the page preloads.
