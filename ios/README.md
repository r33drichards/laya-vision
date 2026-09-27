# Laya Vision on iOS

A SwiftUI app that runs the model on an iPhone with ONNX Runtime (CPU or Core ML), for testing the hardware. How to
build, install and use it: [Run on an iPhone](https://r33drichards.github.io/laya-vision/how-to/run-on-iphone/).

```bash
ios/scripts/export_models.sh             # ONNX export into ios/Models, checked against PyTorch
cd ios && xcodegen && open LayaVision.xcodeproj
```

- `App/`: the app (SwiftUI, the ONNX Runtime backend, image decoding, benchmark).
- `LayaCore/`: the platform-independent port of `predict`'s input and output code (tokenizer, token sequence,
  image resize, answers, the loop over questions), from `web-demo/laya.js`. It builds on Linux too;
  `tests/test_ios_core.py` checks it against Python.
- `tools/ort_server.py`: onnxruntime behind a pipe, so the tests can run `LayaCore`'s loop on the exported graphs.
- `project.yml`: the XcodeGen spec. ONNX Runtime's Swift package is pinned to a commit (ONNX Runtime 1.24.2).
