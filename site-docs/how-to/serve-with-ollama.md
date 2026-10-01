# Serve a checkpoint with Ollama

Ollama serves decision models at `POST /v1/systemone` (TypeSafe's Jev API, Ollama 0.35 and later). This page
turns a Laya Vision checkpoint into a model Ollama itself loads and answers with, images included. To keep the
checkpoint in Python behind the same API instead, see [Serve an Ollama-style decision API](serve-systemone.md).

Three pieces are needed, because Ollama does not run Laya's option-scoring head:

1. **A model trained for Ollama's prompt.** For each question Ollama renders one chat prompt (the state and every
   question as JSON, then `Requested field: "<name>"`) and reads the next-token probabilities of the letters `A`,
   `B`, ... that label the options. `modal_app.py::finetune_ollama` trains the checkpoint's language-model head to
   answer that prompt, on Laya's training data rendered exactly as Ollama renders a request.
2. **GGUF files.** `scripts/export_ollama.py` writes the text model, the vision projector (mmproj) and a Modelfile.
3. **An Ollama that passes images.** Upstream `/v1/systemone` takes text only. The patch in
   [`integrations/ollama/`](https://github.com/r33drichards/laya-vision/tree/main/integrations/ollama) adds an
   `images` field. Without it the model still answers text questions.

## How the model matches Ollama

The prompt and token layout are ported in
[`laya/ollama.py`](https://github.com/r33drichards/laya-vision/blob/main/laya/ollama.py):

- `compile_request` reproduces Ollama's `decision.Compile` byte for byte. `tests/test_ollama.py` checks it against
  fixtures written by Ollama's own Go code, including HTML escaping, Unicode and JSON instructions.
- `GO_TEMPLATE` and `SYSTEM_PROMPT` go into the Modelfile. The model is trained on exactly what they render.
- `input_ids` lays out each image as llama.cpp's mtmd does for SmolVLM: one 512×512 tile, then the overview, with
  the same separator tokens. `image_views` resizes the image the same way (Lanczos, to a 512 longest edge, then to
  512×512).

Before training, the untrained model was exported and served by the patched Ollama. Its letter probabilities
matched PyTorch's to within 0.005 on one image, two images and text only, and the token counts Ollama reported
matched exactly. So what the model learns in training is what Ollama serves.

## Results

Run `laya-vision-ollama-6k`: `thaitea/laya-vision` at revision `f2fe3c1`, 6000 steps of 32 rows (49 minutes on one
GPU), temperature 1.63 folded into `lm_head`. Validation used the first 300 rows of each of 26 sets (the original VQA
sets, the Cauldron sets and the score sets), each asked as a one-question request.

| On the same 6,903 validation rows | Accuracy | ECE | NLL |
|---|---|---|---|
| The checkpoint's option head (`predict`, calibrated) | 75.9% | 0.035 | 0.611 |
| Letter readout, before training | 37.1% | 0.308 | 1.649 |
| **Letter readout, trained (what Ollama serves)** | **73.7%** | **0.035** | **0.620** |

The trained letter readout is 2.2 points behind the option head, with the same calibration error. Per set it is
within a point of the head on 8 of the 26, more than a point ahead on 4 (VQAv2 yes/no 71.3% against 69.7%,
VQA-RAD 83.9% against 79.0%, 62 rows) and more than a point behind on 14. It loses most on ChartQA (56.1% against 73.2%, 41 rows), AI2D (67.4% against 77.0%),
TQA (67.4% against 74.2%) and Cauldron A-OKVQA (69.3% against 76.0%).

**Through Ollama.** The Q8_0 GGUF was served by Ollama with the image patch and sent 300 of those rows (the first 100
of VQAv2 yes/no, Cauldron A-OKVQA and AVA) by `benchmarks/ollama_systemone_eval.py`:

| The same 300 rows | Accuracy | ECE | NLL |
|---|---|---|---|
| Option head (`predict`, PyTorch) | 76.0% | 0.189 | 0.775 |
| Letter readout in PyTorch (F32) | 74.3% | | |
| **Ollama, Q8_0 GGUF, `/v1/systemone` with images** | **74.3%** | 0.206 | 0.774 |

Ollama picks the same answer as PyTorch on 298 of the 300 rows. Its probabilities differ from PyTorch's by a median
of 0.005 (90th percentile 0.025, largest 0.10). About half of that is the Q8_0 quantization: an F32 GGUF on the
A-OKVQA rows differs by a median of 0.002 (largest 0.074), the remainder being llama.cpp's image resizing and kernels.
The high ECE on these 300 rows comes from AVA, a vote-histogram set, for both readouts. On the toy images used to
check the export before training, the gap was below 0.005.

The run's `metrics.json` and the per-row Ollama results are in
[`results/ollama/`](https://github.com/r33drichards/laya-vision/tree/main/results/ollama). They are an experiment
record, not one of the published numbers `benchmarks/verify_published.py` checks.

## Train

```bash
modal run --detach modal_app.py::finetune_ollama --run-name <run>
```

The defaults train `thaitea/laya-vision` (pinned at revision `f2fe3c1`) for 6000 steps of 32 rows on the Cauldron
and score sets, about 40% of one pass. Questions about the same image are grouped into one request of up to four
questions half the time, so the model sees schemas with several fields. `choice` options are shuffled so that no
letter is favoured. Before saving, one temperature is fitted on held-out rows and folded into `lm_head`, because
Ollama always scores at temperature 1. The run writes `/ckpt/smolvlm-ollama/<run>/hf` and `metrics.json`; results
are create-only, so a finished run name cannot be reused.

## Export

You need a llama.cpp checkout at the tag your Ollama pins (`LLAMA_CPP_VERSION` in ollama/ollama; `b11232` when this
page was written):

```bash
git clone --depth 1 --branch b11232 https://github.com/ggml-org/llama.cpp ~/llama.cpp
pip install -e ~/llama.cpp/gguf-py
python scripts/export_ollama.py --run <run> --llama-cpp ~/llama.cpp --out dist/ollama
```

This writes `laya-vision-q8_0.gguf` (137 MB), `laya-vision-mmproj-f16.gguf` (190 MB) and a `Modelfile` with both
`FROM` lines, the template, the system prompt, `num_ctx 4096` and `CAPABILITY decision`. Ollama requires the
capability before it scores a model on `/v1/systemone`.

## Build Ollama with image support

```bash
git clone https://github.com/ollama/ollama && cd ollama
git checkout 1abe35e    # the commit the patch was written against
git am /path/to/laya-vision/integrations/ollama/0001-decision-accept-images-in-v1-systemone.patch
cmake -B build . && cmake --build build --parallel 4    # on a small machine, add -DOLLAMA_BUILD_PARALLEL=2
./ollama serve
```

Then, in another terminal:

```bash
cd dist/ollama && ollama create laya-vision -f Modelfile
curl http://localhost:11434/v1/systemone -d '{
  "model": "laya-vision",
  "state": {"note": "customer says it arrived broken"},
  "images": ["'"$(base64 -w0 photo.jpg)"'"],
  "questions": {
    "damaged": {"type": "noul", "instructions": "Does the item look damaged?"},
    "category": {"type": "choice", "instructions": "What kind of item is this?",
                 "criteria": {"electronics": null, "clothing": null, "furniture": null, "other": null}}
  }
}'
```

`state` must not be empty, even for a question about the image alone; send something short such as
`"See the image."`, which the model was trained on.

## Limits

- Ollama allows 2 to 26 options per question and 64 questions per request.
- Every question is a separate prompt that repeats the state, the images and the whole question set, so a request
  with many questions costs more than `predict`, which encodes the image once.
- Answers come from the letter readout, not the option head, so they differ from `predict`'s. The table above
  compares them on the same rows.
- Ollama scores each question once, in the order given: there is no `n_permutations`.
