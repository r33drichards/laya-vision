# RF100-VL

[RF100-VL](https://github.com/roboflow/rf100-vl) ([paper](https://arxiv.org/abs/2505.20612)) is 100 object-detection
datasets from Roboflow Universe in seven domains, chosen to be unlike what vision-language models were trained on:
X-rays, PCB defects, documents, aerial and thermal imagery, game screenshots. Its metric is COCO box mAP.

**Laya Vision cannot be scored on RF100-VL's own metric.** It answers typed questions and does not draw boxes. This
page is the part of detection it can be asked: for every test image and every class of its dataset, *is at least
one of these in the image?* The numbers are **not comparable to RF100-VL mAP**.

## Setup

- **Data:** the full test split, 14,237 images from all 100 datasets, from the paper's first author's mirror
  [`probicheaux/rf100-vl`](https://huggingface.co/datasets/probicheaux/rf100-vl) at `6b59bae`. The domain map comes from
  `roboflow/rf100-vl` at `451c6dd`.
- **Questions:** one `noul` per (image, class), 85,991 in all: `Is there at least one "<class>" in the image?`. The
  state text names the dataset and its class list, because many class names ("DIP", "0", "Bamboo 1") mean nothing
  alone. The label is yes when a ground-truth box of that class is in the image.
  Converter: [`laya.evalsets.rf100vl_records`](https://github.com/r33drichards/laya-vision/blob/main/laya/evalsets.py).
  Prep: `modal run modal_app.py::prepare_rf100vl`, which writes the group `rf100vl`, one set per domain.
- **Checkpoint:** [thaitea/laya-vision](https://huggingface.co/thaitea/laya-vision) (run
  `autoresearch/full/long-sep24-b64/best`), zero-shot: it never trained on RF100-VL. Calibrated probabilities, one L4.
- **Scoring:** [`benchmarks/rf100vl_presence.py`](https://github.com/r33drichards/laya-vision/blob/main/benchmarks/rf100vl_presence.py)
  scores each dataset, then averages datasets without weights, as the paper does. The metrics:
    - **Presence AP:** for each class, rank the dataset's test images by P(present) and take the average precision;
      then take the mean over classes. It is the image-level analogue of detection AP, with no localisation.
    - **Chance AP:** what a constant score gets on the same classes.
    - **AUROC:** over all of the dataset's questions.
    - **Balanced accuracy:** accuracy at P ≥ 0.5, averaging the "present" and "absent" cases.
- **Evidence:** the rows are
  [`results/raw/smolvlm-autoresearch-full-long-sep24-b64-best.rf100vl-test.predictions.jsonl.gz`](https://github.com/r33drichards/laya-vision/blob/main/results/raw/smolvlm-autoresearch-full-long-sep24-b64-best.rf100vl-test.predictions.jsonl.gz),
  with its `.meta.json` alongside. The summary is
  [`eval-results/autoresearch-full-long-sep24-b64-rf100vl.json`](https://github.com/r33drichards/laya-vision/blob/main/eval-results/autoresearch-full-long-sep24-b64-rf100vl.json).

## Results

| Domain | Datasets | Presence AP | Chance AP | AUROC | Balanced acc | Acc | Prevalence |
|---|---:|---:|---:|---:|---:|---:|---:|
| Aerial | 11 | 0.814 | 0.779 | 0.582 | 0.443 | 0.428 | 0.779 |
| Document | 10 | 0.568 | 0.503 | 0.575 | 0.521 | 0.594 | 0.497 |
| Flora/Fauna | 23 | 0.741 | 0.679 | 0.627 | 0.512 | 0.546 | 0.676 |
| Industrial | 22 | 0.587 | 0.544 | 0.512 | 0.481 | 0.533 | 0.544 |
| Lab Imaging | 13 | 0.628 | 0.607 | 0.476 | 0.539 | 0.512 | 0.607 |
| Misc | 15 | 0.565 | 0.471 | 0.578 | 0.497 | 0.539 | 0.471 |
| Sport | 6 | 0.740 | 0.718 | 0.539 | 0.602 | 0.503 | 0.718 |
| **All 100 (macro)** | 100 | 0.657 | 0.604 | 0.554 | 0.505 | 0.527 | 0.603 |
| **Both answers occur (macro)** | 76 | 0.548 | 0.480 | 0.554 | 0.519 | 0.548 | 0.478 |

In 24 datasets every answer is yes. They are single-class sets where every test image contains the object, so any
model's presence AP there is 1. The last row leaves them out.

**Reading it:** the model is barely above chance.

- **Presence AP** is 7 points above a constant score (0.548 against 0.480) where both answers occur.
- **AUROC** is 0.554. The model beats a coin flip on 48 of the 76 datasets and is below one on the rest.
- **Lab imaging is the weakest domain** (AUROC 0.476): X-rays, MRI and microscopy.
- **It answers yes too rarely:**
    - Across all 85,991 questions, 38.6% of the true answers are yes, but the model says yes to 13.1%.
    - Its pooled accuracy, 62.5%, is barely above always saying no (61.4%).
    - Its calibrated ECE is 0.169, against 0.041 on its own 34 validation sets. Its temperatures do not carry over
      to this data.
- **Best datasets:** photographs of everyday objects.

| Dataset | Domain | Images | Classes | Presence AP | Chance AP | AUROC |
|---|---|---:|---:|---:|---:|---:|
| everdaynew | Misc | 37 | 5 | 0.805 | 0.557 | 0.594 |
| aquarium-combined | Flora/Fauna | 65 | 7 | 0.468 | 0.236 | 0.695 |
| trail-camera | Flora/Fauna | 132 | 2 | 0.724 | 0.500 | 0.709 |
| car-logo-detection | Misc | 17 | 2 | 0.714 | 0.500 | 0.753 |
| water-meter | Industrial | 70 | 10 | 0.670 | 0.474 | 0.697 |
| uavdet-small | Aerial | 100 | 7 | 0.634 | 0.636 | 0.388 |
| x-ray-id | Lab Imaging | 385 | 6 | 0.988 | 0.991 | 0.314 |
| canalstenosis | Lab Imaging | 50 | 5 | 0.485 | 0.500 | 0.386 |
| dentalai | Lab Imaging | 128 | 4 | 0.435 | 0.451 | 0.353 |
| label-printing-defect-version-2 | Document | 80 | 2 | 0.451 | 0.500 | 0.391 |

The table shows the five datasets furthest above chance AP and the five furthest below it, among those where both
answers occur. Every dataset is in the summary JSON.

## Reproduce

```bash
modal run modal_app.py::prepare_rf100vl                    # ~3 min; create-only, --prefix for a new copy
modal run modal_app.py::evidence --run autoresearch/full/long-sep24-b64/best --datasets rf100vl --val-split test \
    --name <new-name>                                      # ~40 min on an L4
python benchmarks/rf100vl_presence.py results/raw/<new-name>.predictions.jsonl.gz
```
