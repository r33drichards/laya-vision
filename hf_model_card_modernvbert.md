---
license: cc-by-nc-sa-4.0
base_model: ModernVBERT/modernvbert
library_name: laya
pipeline_tag: visual-question-answering
datasets:
- HuggingFaceM4/the_cauldron
language:
- en
tags:
- laya
- calibration
- decision-model
- vision
- modernvbert
---

# Laya Vision (ModernVBERT-250M)

This model makes calibrated, typed decisions about **one or more images plus optional text**. It answers `choice`, `score` and `noul` (yes/no probability) questions in one forward pass, with no text generation.

It adds image input to [Laya](https://github.com/NandhaKishorM/laya) by replacing Laya's ModernBERT text encoder with [ModernVBERT](https://huggingface.co/ModernVBERT/modernvbert), a bidirectional ModernBERT-150M plus SigLIP2 vision encoder. Because the backbone is bidirectional, Laya's original readout carries over unchanged: each option is scored from a `[MASK]` marker that sees the whole sequence. Laya's `predict(state, questions)` API, proper-scoring-rule training and temperature calibration are also unchanged. A sibling checkpoint on a causal backbone is [thaitea/laya-vision-smolvlm-256m](https://huggingface.co/thaitea/laya-vision-smolvlm-256m).

- **Code:** [github.com/r33drichards/laya-vision](https://github.com/r33drichards/laya-vision), see the "ModernVBERT experiment" and "Post-training on The Cauldron" sections and `docs/modernvbert-cauldron.md`.
- **Status:** experimental. This is an independent research fork, not affiliated with Convai Innovations, the authors of Laya, nor with the ModernVBERT authors.

## Usage

Needs `transformers >= 5.3` for the ModernVBERT classes.

```bash
git clone https://github.com/r33drichards/laya-vision && pip install -e ./laya-vision torchvision "transformers>=5.3"
```

```python
import laya
from PIL import Image

agent = laya.load_vlm("thaitea/laya-vision-modernvbert-250m")
result = agent.predict(
    {"image": Image.open("photo.jpg"), "note": "customer says it arrived broken"},
    {
        "damaged":  {"type": "noul",   "instructions": "Does the item in the photo look damaged?"},
        "category": {"type": "choice", "instructions": "What kind of item is this?",
                     "criteria": ["electronics", "clothing", "furniture", "food", "other"]},
    },
)
result["answers"]["damaged"]["noul"]        # calibrated P(true)
result["answers"]["category"]["choice"]     # top option; see ["probabilities"], ["confidence"]
```

A state may carry several images: `{"images": [left, right]}` (the model was trained on NLVR2 pairs).

## Results

Scores are on full validation splits. "Calibrated" uses the per-type temperatures stored in `vlm_agent_config.json`, which are applied automatically. The first three rows are the official splits the SmolVLM checkpoint reports on; the model saw those datasets' *train* rows only as 3 of its 19 training subsets.

| Dataset | Question type | Chance | Accuracy | ECE raw | ECE calibrated |
|---|---|---|---|---|---|
| A-OKVQA (n=1,138) | 4-way `choice` | 25% | 65.2% | 0.214 | 0.064 |
| ScienceQA, image subset (n=2,097) | 2–5-way `choice` | ~36% | 79.0% | 0.114 | 0.058 |
| VQAv2 yes/no (n=5,000)\* | `noul` | 50% | 71.8% | 0.134 | 0.037 |
| **All 22 validation sets (n=22,886)**, the three above plus a 5% holdout of each training subset | | | **71.8%** | 0.103 | **0.022** |

\* The VQAv2 train/val split is a re-split of the official VQAv2 *validation* set by image (the only official split with answers), so these numbers are not comparable to published VQAv2 results.

Per-subset holdout accuracy ranges from 93% (DVQA) and 86% (OCR-VQA) down to 40% (RAVEN, 8-way) and 27% (InterGPS geometry); the full table is in `training_metrics.json` and in the repo's `docs/modernvbert-cauldron.md`. Against the SmolVLM checkpoint, which was trained for 3 epochs on exactly the three official sets, this model is 3.4 points better on A-OKVQA and 7.6 and 1.6 points worse on ScienceQA and VQAv2 yes/no.

- **Latency:** about 32 ms for one image question on an NVIDIA L4 (bf16), 48 ms in fp32. The image is encoded once per `predict` call and shared by every question.
- **Option order:** across 4 rotations of the A-OKVQA option order, accuracy varies by 1.0 point. Averaging over orders (`n_permutations=4`) gives 66.4%.

## Training

- **Backbone:** ModernVBERT (ModernBERT-150M text encoder, SigLIP2-100M vision tower, pixel-shuffle connector, 512-pixel tiles, 64 image tokens). The vision tower is frozen; the text encoder, connector and decision head are trained.
- **How options are read:** `[CLS]User:<image tokens> <type> question: <instructions>[SEP][MASK] option 1[MASK] option 2 ...[SEP]<state>[SEP]`. Each option is scored from its `[MASK]`, as in Laya's text model; the act head pools `[CLS]`.
- **Loss:** soft cross-entropy plus a strictly proper scoring rule (log + spherical), option order shuffled at random.
- **Data:** the closed-form turns of 19 subsets of [The Cauldron](https://huggingface.co/datasets/HuggingFaceM4/the_cauldron), 269,900 questions after holdouts: lettered multiple choice (AI2D, IconQA, InterGPS, ScienceQA, TQA, Visual7W), A-OKVQA's options lists, RAVEN's lettered panels, and yes/no turns (FigureQA, Hateful Memes, NLVR2, VSR, VQA-RAD, and the yes/no share of CLEVR, DVQA, MapQA, OCR-VQA, VQAv2, ChartQA). Up to 10,000 rows per subset, at most 4 questions per row. Subsets were sampled equally and none was repeated more than 4 times.
- **Schedule:** 16,869 steps at batch 32 on one A100 (68 minutes). Head LR 1.4e-4, backbone LR 2.8e-5, 3% warmup, then cosine decay to 10%. Two epochs; the final checkpoint had the best mean validation accuracy.
- **Calibration:** per-type temperatures fitted on the last 300 training records of each subset, held out of training: choice 2.71, noul 2.16.
- `training_metrics.json` has every evaluation from the run.

## Limitations

- **`score` questions are untrained.** There was no ordinal image data, so their outputs are meaningless.
- **Overconfident before calibration.** Raw ECE on the choice sets is 0.11 to 0.21; rely on the calibrated outputs, which the loader applies by default.
- **Weak on visual reasoning.** Geometry (InterGPS 27%), abstract analogies (RAVEN 40%) and textbook diagrams (TQA 51%) are far below the chart, document and photo sets.
- **Images are 512-pixel tiles.** Fine print in documents may be lost; the image processor's tiling is turned off.
- **Text-only states work but are unevaluated.** This model is not a drop-in replacement for Laya's text checkpoint.

## License

The weights are released under **CC BY-NC-SA 4.0**, because the training data includes ScienceQA, which uses that license. The base model, ModernVBERT, is MIT. The Cauldron's subsets each carry their own licenses; consult the dataset card before commercial use of any derivative. The code is Apache 2.0.
