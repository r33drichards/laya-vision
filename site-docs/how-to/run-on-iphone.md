# Run on an iPhone

[`ios/`](https://github.com/r33drichards/laya-vision/tree/main/ios) is a small SwiftUI app that runs the model on the
phone itself, with no server: pick a photo, write questions in the same JSON `predict` takes, and read the answers and
how long each stage took. It is for testing the hardware (which precision and which compute units are fastest, how
much memory it takes, how it holds up over repeated runs), not an App Store app. You build it with Xcode and install
it on your own phone.

## What you need

- A Mac with Xcode 16 or later, and [XcodeGen](https://github.com/yonaskolb/XcodeGen) (`brew install xcodegen`).
- An iPhone on iOS 17 or later, and a cable.
- An Apple ID. A free one works: the app then stays installed for 7 days, after which you run it from Xcode again.
- About 3 GB of disk on the Mac and 1.5 GB on the phone for all three precisions.

## 1. Export the model

```bash
ios/scripts/export_models.sh
```

It creates a virtualenv in `ios/.venv`, exports `thaitea/laya-vision` at revision `8b318c9` (the 201M checkpoint
the [demo Space](../tutorials/space-demo.md) runs) with
[`scripts/export_onnx.py`](https://github.com/r33drichards/laya-vision/blob/main/scripts/export_onnx.py) into
`ios/Models`, and checks each precision against PyTorch. The model is three graphs (vision tower, language model,
decision head). The check on the current export:

| Precision | Size | max \|Δp\| vs PyTorch | same top answer |
|---|---|---|---|
| fp32 | 806 MB | 0.000004 | 9/9 |
| fp16 | 404 MB | 0.004 | 9/9 |
| q8 (8-bit weights) | 314 MB | 0.017 | 9/9 |

`VARIANTS=q8 ios/scripts/export_models.sh` skips fp16 (fp32 is always written). `CHECKPOINT=` and `REVISION=`
export another SmolVLM checkpoint.

## 2. Build and install

```bash
cd ios
xcodegen
open LayaVision.xcodeproj
```

In Xcode:

1. Select the **LayaVision** target, **Signing & Capabilities**, and pick your **Team** (your Apple ID's
   "Personal Team"; add the account under Xcode → Settings → Accounts if it is not listed).
2. Plug in the iPhone, pick it as the run destination, and press **Run**. The scheme builds in Release, because
   the image resize and the tokenizer are plain Swift loops that are about ten times slower unoptimised.
3. The first time, the phone asks for **Developer Mode** (Settings → Privacy & Security → Developer Mode, then
   restart) and to trust the developer (Settings → General → VPN & Device Management).

Running `xcodegen` again rewrites the project and forgets the team; pick it again.

To try a new export without rebuilding, copy the export folder into the app's files as `Models` (Finder → the
iPhone → Files → Laya Vision). The app looks there before its bundled copy.

## 3. Use it

- **Precision** is fp32, fp16 or q8, whichever were exported.
- **Vision tower** and **Language model** each run on the **CPU** (ONNX Runtime's own kernels) or through
  **Core ML** (ONNX Runtime's Core ML execution provider) restricted to all compute units, CPU + GPU, or CPU +
  Neural Engine. Operators Core ML cannot take stay on the CPU. The decision head is small and always runs on the
  CPU.
- **Load** creates the sessions and runs one warm-up prediction. The first Core ML load compiles the model and can
  take minutes; the compiled model is cached for later loads.
- **Run** answers the questions. The time is split into the image resize, the vision tower (once per image), the
  language model and the head (once per question and option order).
- **Run benchmark** repeats the same prediction and reports the median and range of each stage, the thermal state
  before and after, and the peak memory footprint.
- **Copy report** puts the device, the settings, the last answers and the benchmark on the clipboard as JSON.

## How close it is to Python

The app does not run `predict` itself. It is a port of it, checked against Python by
[`tests/test_ios_core.py`](https://github.com/r33drichards/laya-vision/blob/main/tests/test_ios_core.py), which
builds the Swift code on Linux or macOS:

- **Token ids**: identical to `build_vlm_inputs` on every row of the parity fixture (18 rows: images, text-only
  states, unicode, truncation, long option lists), and the tokenizer is identical to Hugging Face's on 21 strings
  chosen to hit each rule of its pre-tokenizer.
- **Whole pipeline**: with the Hugging Face processor's pixel values, the Swift code on the exported fp32 graphs gives
  the same answers as `VLMAgent.predict` to the 4 decimals `predict` prints, with one or two option orders
  (`LAYA_ONNX_DIR=ios/Models python -m pytest tests/test_ios_core.py`).
- **Image resize**: the Swift resize matches the processor to 0.05 grey levels on average and 2 at most; on the
  example photo that moves probabilities by up to 0.008.

Differences that remain on the phone:

- The photo is drawn with its EXIF orientation applied and converted to sRGB. `predict` in Python does neither, so a
  rotated or Display P3 photo gives different pixels in the two.
- One image per prediction, and one or two option orders. `predict(..., n_permutations=K)` averages seeded random
  orders past the second.
- fp16 and q8 are the quantized graphs above; the Core ML execution provider may also run parts of the graph at
  lower precision on the GPU or Neural Engine.
- The language model's input length depends on the question, so Core ML may specialise the graph again for each
  new length. Compare benchmark medians, not the first run of a new question.

## See also

- [Checkpoints](../reference/checkpoints.md): what `thaitea/laya-vision` is.
- [What didn't work](../reference/results/what-didnt-work.md): the earlier in-browser demo, which ran the same export
  with ONNX Runtime Web.
