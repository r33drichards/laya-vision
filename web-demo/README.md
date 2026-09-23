# Laya Vision in the browser

A static page that runs [thaitea/laya-vision](https://huggingface.co/thaitea/laya-vision) on the visitor's device
with [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/): WebGPU when the browser has it, WASM (CPU)
otherwise. There is no server, API, database or telemetry, and no build step; the image never leaves the page.

**Live: <https://r33drichards.github.io/laya-vision/demo/>**, deployed with the documentation site
(`.github/workflows/deploy-docs.yml`, which places this folder's page under `demo/`). The model files load from
[thaitea/laya-vision-web](https://huggingface.co/thaitea/laya-vision-web).

The documentation is on the site, built from `site-docs/`:

- [Run it in your browser](https://r33drichards.github.io/laya-vision/tutorials/browser-demo/): using the page.
- [Export for the browser](https://r33drichards.github.io/laya-vision/how-to/export-for-the-web/): export a
  checkpoint with `scripts/export_onnx.py`, publish it, serve the page locally, run the tests.
- [The browser runtime](https://r33drichards.github.io/laya-vision/concepts/browser-runtime/): how the page
  reproduces `predict`, and why the fp16 export keeps its normalisations in float32.
- [Browser demo files and checks](https://r33drichards.github.io/laya-vision/reference/web-demo/): the exported
  files, the pinned runtime, and how closely each precision matches PyTorch.

Quick local run, after an export into `web-demo/models/laya-vision` (git-ignored), or with no export (the page then
loads the Hub copy):

```bash
cd web-demo && python3 -m http.server 8080          # http://localhost:8080
python -m pytest -q tests/test_web_demo.py          # from the repository root
```
