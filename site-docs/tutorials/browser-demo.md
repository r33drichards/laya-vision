# Run it in your browser (WebGPU)

In this tutorial you run the published checkpoint entirely in a web page: the model downloads to your browser and
runs on your GPU with WebGPU (or on your CPU with WASM). There is no server; the image never leaves the page.

**Open the demo: <https://r33drichards.github.io/laya-vision/demo/>**

WebGPU needs a browser that supports it; without it the page runs on the CPU with WASM, much more slowly. It
has been tested in Chrome on macOS (Apple GPU, fp16). The first load downloads 250 to 950 MB
depending on the precision (see step 1); after that the files come from the browser's cache.

## 1. Load the model

Under **1. Model**, leave **Model folder URL** as it is: the files are the ONNX export of
[thaitea/laya-vision](https://huggingface.co/thaitea/laya-vision), hosted at
[thaitea/laya-vision-web](https://huggingface.co/thaitea/laya-vision-web).

- **Precision:** *auto* picks fp16 when your GPU supports it through WebGPU (the `shader-f16` feature) and q8
  otherwise. fp32 matches PyTorch most closely but is the largest download; q4 is the smallest and the least
  faithful, and can change the top answer.
- **Backend:** *auto* uses WebGPU if the browser has it, else WASM on the CPU.

Press **Load model**. The status line says which precision and backend it picked, how long loading took, and how
far that precision's export was from PyTorch in the export's own check.

## 2. Pick an image and a state

Under **2. Image and state**, the page has already loaded an example photo, two turntables and a mixer. To use your
own image, choose a file, paste one (Ctrl/Cmd+V, for example a screenshot), or drop an image file anywhere on the
page. **No image** clears it, for a text-only state.

The **State** box is the text context that goes with the image: a JSON object (every key except the image is
serialised the way Python's `json.dumps` does) or plain text. The example's note, "customer says it arrived
broken", is part of the input: it changes the answers. Clear it or write your own.

## 3. Ask questions and run

Under **3. Questions**, edit the JSON: it is the same shape as the second argument of
[`predict`](../reference/predict.md). Each question needs a `type` (`choice`, `score` or `noul`) and
`instructions`, plus `criteria` for `choice` and `score`.

**Option orders** set to 2 scores every question twice, with the options as written and reversed, and averages
the two, which reduces the model's option-order bias at twice the cost.

Press **Run**. Each answer shows the probability of every option, the chosen one, and a confidence. The raw output
below it is the same schema `predict` returns in Python.

## Reading the answers

The probabilities are a softmax over *the options you listed*: a question whose true answer is not among them still
gets a confident-looking distribution. On a photo of a washing machine, a `choice` between electronics, clothing,
furniture, food and other can put "food" first; adding "home appliance" to the options fixes it. List the answer
you expect to be right, or keep an "other".

The **Limitations** section at the bottom of the page lists what differs from the Python model.

## What you did

You ran a 256M-parameter vision-language decision model on your own device. Next:

- How the page reproduces `predict`, and why fp16 needed care: [The browser runtime](../concepts/browser-runtime.md).
- Host your own export, or a different checkpoint: [Export for the browser](../how-to/export-for-the-web.md).
- What has been checked, and how closely each precision matches PyTorch: [Browser demo files and checks](../reference/web-demo.md).
