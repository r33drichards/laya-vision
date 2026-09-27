# Browser runtime (not deployed)

The static page that ran the model in the browser with [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/).
It was taken down because the browser's image decoding and the smaller exports moved the answers too far
([What didn't work](https://r33drichards.github.io/laya-vision/reference/results/what-didnt-work/)), and its model repo
on the Hub was deleted. It is kept as the JavaScript reference for the on-device ports:

- `laya.js`: `predict`'s token sequence, image resize and answers in plain JavaScript. The iOS app's
  [`ios/LayaCore`](../ios/LayaCore) is ported from it.
- `fixtures/parity.json`: token ids, markers and option spans from Python for 5 cases in two option orders.
  `tests/test_web_demo.py` regenerates it from Python and checks it; `tests/test_ios_core.py` checks the Swift port
  against it.

To run the page locally, export a checkpoint into `web-demo/models/laya-vision` (git-ignored) with
`python scripts/export_onnx.py thaitea/laya-vision --out web-demo/models/laya-vision --quantize fp16,q8 --validate`, then
`cd web-demo && python3 -m http.server 8080`.
