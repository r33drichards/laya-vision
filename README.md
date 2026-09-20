# Laya Vision

Image inputs for [Laya](https://github.com/NandhaKishorM/laya): typed, calibrated decisions (`choice`, `score`, `noul`) over an **image plus optional text**, in one forward pass with no text generation.

Laya Vision swaps Laya's ModernBERT text encoder for [SmolVLM-256M-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct), which already understands images. It keeps Laya's `predict(state, questions)` API, output schema, proper-scoring-rule training and temperature calibration.

- **Model:** [thaitea/laya-vision-smolvlm-256m](https://huggingface.co/thaitea/laya-vision-smolvlm-256m)
- **Try it in the browser:** [thaitea/laya-vision-demo](https://huggingface.co/spaces/thaitea/laya-vision-demo), a Hugging Face Space on free CPU at about 3 s per image. Its source is in `space/`.
- **Status:** experimental research fork. It is not affiliated with Convai Innovations, the authors of Laya.

## Results

This is the fine-tuned checkpoint `all3-3ep/best`: 3 passes over 72k training examples, about 33 minutes on one A100. Scores are on the full validation splits.

| Dataset | Type | Chance | Accuracy | ECE (raw → calibrated) |
|---|---|---|---|---|
| A-OKVQA | 4-way `choice` | 25% | 61.8% | 0.295 → 0.123 |
| ScienceQA (image subset) | 2–5-way `choice` | ~36% | 86.6% | 0.090 → 0.034 |
| VQAv2 yes/no (re-split of official val) | `noul` | 50% | 73.4% | 0.102 → 0.041 |
| **All** | | | **75.2%** | 0.124 → **0.034** |

- **Latency:** about 71 ms for one image question on an NVIDIA L4 (bf16). The image is encoded once and reused for every question in the call.
- **Option-order sensitivity:** across 4 rotations of the A-OKVQA option order, accuracy varies by 0.7 points.
- **`score` questions are not trained yet.** There was no ordinal image data, so treat `score` outputs as meaningless.

## Usage

```python
import laya
from PIL import Image

agent = laya.load_vlm("thaitea/laya-vision-smolvlm-256m")   # downloads from the Hub
result = agent.predict(
    {"image": Image.open("photo.jpg"), "note": "customer says it arrived broken"},
    {
        "damaged":  {"type": "noul",   "instructions": "Does the item in the photo look damaged?"},
        "category": {"type": "choice", "instructions": "What kind of item is this?",
                     "criteria": ["electronics", "clothing", "furniture", "food", "other"]},
    },
)
print(result["answers"]["damaged"]["noul"], result["answers"]["category"]["choice"])
```

Install with `pip install -e .`, plus `torchvision`, which the SmolVLM image processor needs.

### Run on Modal

`modal_app.py` expects the Modal volumes `laya-hf-cache`, `laya-datasets` and `laya-checkpoints`.

```bash
modal run modal_app.py::try_model --image photo.jpg [--questions q.json] [--text "..."]   # ask a checkpoint about an image
modal run modal_app.py::test                                                          # GPU tests + latency
modal run --detach modal_app.py::finetune_long                                        # ~3-epoch A100 fine-tune
modal run modal_app.py::evaluate --run-name all3-3ep/best                             # re-score a checkpoint
```

The training data is written to `laya-datasets:/data/vqa/<name>/{train,val}.jsonl` by the data-prep job on the [`siglip-projector-experiment`](https://github.com/r33drichards/laya-vision/tree/siglip-projector-experiment) branch.

## Playing games

Laya Vision can also act as a game policy: the screen is the image and the options are the game's buttons. `examples/atari_live.py` and `examples/vizdoom_live.py` let you watch it play in a local window.

The released checkpoint doesn't know any games; in ViZDoom `basic` it only ever shoots. Trained for 7 minutes on 20,000 frames auto-labelled by a scripted expert, it plays `basic` at expert level: mean reward +75.4 and 100% kills over 50 unseen episodes, against the expert's +75.8. See [docs/game-training.md](docs/game-training.md) for the pipeline, results and training-data ideas.

## Solving CAPTCHAs

[Open CaptchaWorld](https://arxiv.org/abs/2505.24878) benchmarks browser agents on interactive CAPTCHAs. This model can't click or drag, so `laya/captcha.py` re-expresses 13 of its 20 types as typed decisions — 300 puzzles, 3,121 decisions — graded offline against the benchmark's own ground truth.

```bash
python examples/captcha_eval.py --data /path/to/OpenCaptchaWorld/captcha_data
```

The released checkpoint scores **18.3% pass@1 against a 11.3% chance baseline**, but that average hides a sharp split:

- **One image, one question: it works.** `Select_Animal` ("pick a fox" over a 2x3 grid) is **96.7%, AUC 0.992** — the training distribution, since a single cropped cell plus a yes/no question is a VQAv2 question.
- **Two images to compare: completely blind.** Every reference-and-candidate type sits at AUC 0.43–0.54 while answering "yes" to ~100% of candidates. A-OKVQA, ScienceQA and VQAv2 are all single-image QA, so this capability was never trained.

A shuffled-image control (`--control shuffle`) confirms the split: the three types with signal lose it (macro-AUC 0.590 → 0.513), while the comparison types don't move, because they were never using the images. See [docs/captcha-benchmark.md](docs/captcha-benchmark.md) for the full table, the six coordinate-click types that are out of scope, and what closing the gap would take.

## What didn't work

The branch [`siglip-projector-experiment`](https://github.com/r33drichards/laya-vision/tree/siglip-projector-experiment) tried to keep Laya's ModernBERT encoder and feed it SigLIP2 image patches through a learned projector. It kept text-only answers bit-identical, but in 5 training runs it never learned to use the image. Every run collapsed to uniform predictions, and accuracy with shuffled images matched accuracy with the real ones. Details are in that branch's `laya/vision_train.py` and commit history.

## License

- **Code:** Apache 2.0, inherited from Laya. See `LICENSE`.
- **Weights:** the published weights are trained partly on ScienceQA, which is CC BY-NC-SA 4.0, so the model card marks them as non-commercial.

---

## Credits

Laya Vision is an independent fork of [Laya](https://github.com/NandhaKishorM/laya) by Convai Innovations, Apache 2.0, and is not affiliated with them. The text decision model, its typed-question API (`choice` / `score` / `noul`), the RLCD training objective and the temperature calibration are theirs; this fork adds image input, the SmolVLM backbone and the game work. The design is described in the author's [write-up](https://dev.to/nandakishor_m_6cc0adfde9f/i-built-non-autoregressive-decision-models-a-year-ago-then-a-frontier-lab-called-it-a-18me). The original text model and its Colab/Kaggle notebooks live in the upstream repo.
