---
license: cc-by-nc-4.0
base_model: HuggingFaceTB/SmolVLM2-256M-Video-Instruct
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
- smolvlm
---

# Laya Vision (SmolVLM2-256M, image splitting at 1024)

This model makes calibrated, typed decisions about an **image plus optional text**. It answers `choice` and `noul` (yes/no probability) questions in one forward pass, with no text generation.

It is the **image-splitting checkpoint from a benchmark**, not a general-purpose release. It was trained on a 24k-question subset of six Cauldron datasets to measure what SmolVLM2's image splitting buys. The same recipe without splitting scores 1.4 points lower on the same data. See [Benchmark](#benchmark).

- **Code:** [github.com/r33drichards/laya-vision](https://github.com/r33drichards/laya-vision) (`modal_app.py::split_bench`)
- **Sibling checkpoint:** [thaitea/laya-vision-smolvlm-256m](https://huggingface.co/thaitea/laya-vision-smolvlm-256m) (SmolVLM v1, no splitting, different training data)
- **Status:** experimental. This is an independent research fork, not affiliated with Convai Innovations, the authors of Laya.

## Usage

SmolVLM2 and image splitting are not on the repo's `main` branch yet, so install from the branch that adds them:

```bash
git clone -b claude/token-encoder-vlm-questions-66ck4u https://github.com/r33drichards/laya-vision
pip install -e ./laya-vision torchvision
```

```python
import laya
from PIL import Image

agent = laya.load_vlm("thaitea/laya-vision-smolvlm2-256m-split1024")
result = agent.predict(
    {"image": Image.open("photo.jpg")},
    {
        "outdoors": {"type": "noul",   "instructions": "Was this photo taken outdoors?"},
        "vehicle":  {"type": "choice", "instructions": "Which vehicle is shown?",
                     "criteria": ["car", "bus", "bicycle", "none"]},
    },
)
result["answers"]["outdoors"]["noul"]      # calibrated P(true)
result["answers"]["vehicle"]["choice"]     # top option; see ["probabilities"], ["confidence"]
```

The splitting settings are stored in `vlm_agent_config.json` (`image_split_edge: 1024`, `max_len: 1536`) and applied automatically. Each image is resized so its longest edge is 1024 px (small images are upscaled), cut into 512 px tiles, and followed by one downscaled view of the whole image. That makes up to 5 views of 64 tokens each.

## Results

Validation accuracy after temperature calibration. The temperatures are stored in the config and applied automatically. Each validation set is a seeded 5% of the Cauldron rows, split by row so no image is in both train and validation.

| Dataset (Cauldron subset) | Question type | n | Accuracy | ECE raw | ECE calibrated |
|---|---|---|---|---|---|
| AI2D (science diagrams) | `choice` | 282 | 74.5% | 0.104 | 0.076 |
| A-OKVQA (photos) | `choice` | 552 | 67.6% | 0.138 | 0.063 |
| TQA (textbook figures) | `choice` | 264 | 59.8% | 0.146 | 0.055 |
| OCR-VQA (book covers, yes/no only) | `noul` | 996 | 84.0% | 0.065 | 0.025 |
| MapQA (maps, yes/no only) | `noul` | 1,598 | 61.6% | 0.014 | 0.015 |
| VQAv2 (photos, yes/no only) | `noul` | 1,019 | 73.4% | 0.117 | 0.037 |
| **All** | | **4,711** | **70.3%** | 0.072 | **0.019** |

The mean over the six sets is 70.2%. The validation sets come from the Cauldron's training rows, so these numbers are not comparable to published results on the official benchmarks.

- **Latency:** 103.6 ms median (p90 107.1 ms) for one image question on an NVIDIA L4 in bf16, including the CPU preprocessing. An image averages 4.6 views and 368 input tokens.

## Benchmark

`split_bench` trained three runs that were identical except for image splitting.

| Setting | Views per image | Mean accuracy (6 sets) | All questions | L4 median latency | Training time |
|---|---|---|---|---|---|
| No splitting | 1 | 68.7% | 69.2% | 77.0 ms | 8.6 min |
| **Split at 1024 (this model)** | 4.6 | **70.2%** | **70.3%** | 103.6 ms (+35%) | 21.0 min |

The gains were on photos and diagrams: A-OKVQA +2.5, AI2D +2.1, VQAv2 +1.8 points. OCR-VQA was flat (−0.1) and MapQA gained 0.7. Every row is a single training run, and most per-set differences are within about 1–2 standard errors. The pooled gain over all questions, +1.1 ± 0.9 points (one standard error of the difference), is the most reliable figure. Splitting at 2048 gained nothing over no splitting (−0.3 ± 1.0) at 2.5× the latency; the full write-up is in the repo's `docs/split-bench.md`.

## Training

- **Backbone:** SmolVLM2-256M-Video-Instruct. The vision tower is frozen, and the language model and decision head are trained.
- **How options are read:** the question and state come first and the options are listed last. Option attention is bidirectional (`option_attention: bidirectional`), and each option is scored from the hidden state at the end of its own line.
- **Loss:** soft cross-entropy plus a strictly proper scoring rule, with the option order shuffled at random.
- **Data:** 4,000 training rows from each of six Cauldron subsets: AI2D, A-OKVQA, TQA, OCR-VQA, MapQA and VQAv2. Only closed-form turns are kept: lettered or listed choices become `choice`, and yes/no answers become `noul`. From OCR-VQA, MapQA and VQAv2 that means only the yes/no questions. Images were stored with their longest side at most 1024 px.
- **Schedule:** 1,500 steps at batch 32 (2 epochs over 24,000 rows) on one A100 80GB, 14.6 minutes of training. Head LR 1.4e-4, backbone LR 2.8e-5, 45 warmup steps, then cosine decay. The final step had the best mean validation accuracy and was kept.
- **Calibration:** per-type temperatures fitted on 300 held-out training records per dataset: choice 1.72, noul 1.76.
- `training_metrics.json` has every evaluation from the run.

## Limitations

- **`score` questions are untrained.** None of the six datasets has ordinal questions, so their outputs are meaningless.
- **Small training set.** It is 24k questions from six datasets over 2 epochs. Expect to fine-tune on your own data.
- **Slower than the unsplit model.** Preprocessing and the forward pass handle about 4.6× as many image views. If your images have no fine detail, the unsplit recipe is faster for about 1 point less accuracy.
- **One image per state.** Text-only states work but are unevaluated.

## License

The weights are released under **CC BY-NC 4.0**, a conservative choice. The Cauldron's card states that each of its source datasets is governed by its own license, and some may restrict commercial use. Check the source datasets' licenses before using this model. The base model, SmolVLM2, is Apache 2.0. The Cauldron's prompts are CC BY 4.0. The code is Apache 2.0.
