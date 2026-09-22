---
license: cc-by-nc-sa-4.0
base_model: HuggingFaceTB/SmolVLM-256M-Instruct
library_name: laya
pipeline_tag: visual-question-answering
datasets:
- HuggingFaceM4/the_cauldron
- MMInstruction/VLFeedback
- trojblue/AVA-aesthetics-10pct-min50-10bins
- Exploration/richhf_18k_with_images
- QCRI/CrisisMMD
language:
- en
tags:
- laya
- calibration
- decision-model
- vision
- smolvlm
- rubric-scoring
---

# Laya Vision (SmolVLM-256M, Cauldron + rubric scoring)

This model makes calibrated, typed decisions about an **image plus optional text**. It answers `choice`, `score` (a graded level on a rubric you write) and `noul` (yes/no probability) questions in one forward pass, with no text generation.

It is the second SmolVLM checkpoint of [Laya Vision](https://github.com/r33drichards/laya-vision), an independent fork of [Laya](https://github.com/NandhaKishorM/laya) that replaces Laya's ModernBERT encoder with [SmolVLM-256M-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct). Compared with [thaitea/laya-vision-smolvlm-256m](https://huggingface.co/thaitea/laya-vision-smolvlm-256m) it is trained on 19 closed-form subsets of The Cauldron instead of three VQA sets, its answer options attend to each other (see Training), and **its `score` head is trained**, on four rubric-scored image datasets. Laya's `predict(state, questions)` API, proper-scoring-rule training and temperature calibration are unchanged.

- **Code:** [github.com/r33drichards/laya-vision](https://github.com/r33drichards/laya-vision); the data is described in `docs/score-data.md` and the runs in `docs/score-results.md`.
- **Status:** experimental. This is an independent research fork, not affiliated with Convai Innovations, the authors of Laya.

## Usage

```bash
git clone https://github.com/r33drichards/laya-vision && pip install -e ./laya-vision torchvision
```

```python
import laya
from PIL import Image

agent = laya.load_vlm("thaitea/laya-vision-smolvlm-256m-score")
result = agent.predict(
    {"image": Image.open("photo.jpg"), "note": "customer says it arrived broken"},
    {
        "damage": {"type": "score", "instructions": "How much damage does the item in the photo show?",
                   "criteria": ["none", "cosmetic: scratches or dents", "functional: parts broken or missing", "destroyed"]},
        "category": {"type": "choice", "instructions": "What kind of item is this?",
                     "criteria": ["electronics", "clothing", "furniture", "food", "other"]},
        "outdoors": {"type": "noul", "instructions": "Was the photo taken outdoors?"},
    },
)
result["answers"]["damage"]["score"]        # expected level, 0..3, plus ["probabilities"] per level
result["answers"]["category"]["choice"]     # top option; see ["probabilities"], ["confidence"]
result["answers"]["outdoors"]["noul"]       # calibrated P(true)
```

A `score` question's `criteria` is the rubric, level 0 first. Write each level as a short clause (the model saw 3- to 5-level rubrics such as "unhelpful: ignores or misreads the question" ... "very helpful: complete, accurate and well explained"). The answer is the probability-weighted level; the per-level probabilities are calibrated with the stored `score` temperature.

## Results

All numbers are from the final checkpoint on held-out data. "Calibrated" uses the per-type temperatures stored in `vlm_agent_config.json`, applied automatically.

### `score` questions (new)

Validation splits are upstream validation or dev splits (AVA, RichHF, CrisisMMD) or a 5% row holdout (VLFeedback). "Prior only" is a model that always predicts the validation split's mean level distribution. `Levels off` is the absolute difference between the model's expected level and the target's; `xent` is cross-entropy against the target (AVA's target is the human vote histogram).

| Dataset | Rubric | Levels | n | Accuracy | Levels off | xent | Prior only: acc / levels off / xent |
|---|---|---|---|---|---|---|---|
| VLFeedback | helpfulness or visual faithfulness of a response to a question about the image | 5 | 2,000 | 53.9% | 0.80 | 1.147 | 27.5% / 1.37 / 1.560 |
| RichHF-18K | plausibility, prompt alignment, aesthetics or overall quality of a generated image | 5 | 1,990 | 57.3% | 0.50 | 0.966 | 44.0% / 0.66 / 1.168 |
| CrisisMMD damage | little / mild / severe damage in a disaster photo | 3 | 529 | 69.0% | 0.38 | 0.914 | 62.8% / 0.64 / 0.904 |
| AVA aesthetics | photo aesthetic quality, soft targets from 1-10 votes | 5 | 1,000 | 84.7%\* | 0.25 | 1.212 | 84.8% / 0.29 / 1.233 |

\* AVA's argmax accuracy is not informative: the target is a vote histogram whose mode is "average" for 85% of photos. Levels off and xent are the numbers to read (the xent floor, the targets' own entropy, is 1.098).

### Official VQA splits

| Dataset | Question type | Chance | n | Accuracy | ECE raw → calibrated |
|---|---|---|---|---|---|
| A-OKVQA (official val) | 4-way `choice` | 25% | 1,138 | 60.0% | 0.302 → 0.161 |
| ScienceQA (official val, image subset) | 2-5-way `choice` | ~36% | 2,097 | 82.8% | 0.113 → 0.038 |
| VQAv2 yes/no (re-split of official val)\*\* | `noul` | 50% | 5,000 | 72.4% | 0.187 → 0.076 |
| **All 26 validation sets** | | | **28,405** | **74.1%** | 0.095 → **0.034** |

\*\* The VQAv2 split is a re-split of the official VQAv2 *validation* set by image, so it is not comparable to published VQAv2 numbers. The earlier checkpoint, trained 12 passes over ScienceQA alone, scores 86.6% there; this one made 3.4 passes as one of 23 sets.

### Cauldron holdouts

Five percent of each subset's rows, held out by row (no image in both splits).

| Subset | Type | n | Accuracy | ECE raw → calibrated |
|---|---|---|---|---|
| A-OKVQA (Cauldron holdout) | `choice` | 552 | 74.1% | 0.184 → 0.080 |
| AI2D | `choice` | 282 | 78.0% | 0.155 → 0.059 |
| CLEVR yes/no | `noul` | 1718 | 63.1% | 0.063 → 0.046 |
| ChartQA yes/no | `noul` | 41 | 58.5% | 0.253 → 0.147 |
| DVQA yes/no | `noul` | 1397 | 91.8% | 0.021 → 0.042 |
| FigureQA | `noul` | 1928 | 83.7% | 0.040 → 0.046 |
| Hateful Memes | `noul` | 456 | 89.9% | 0.088 → 0.054 |
| IconQA | `choice` | 606 | 93.7% | 0.045 → 0.031 |
| Inter-GPS | `choice` | 94 | 34.0% | 0.244 → 0.191 |
| MapQA yes/no | `noul` | 1598 | 61.1% | 0.028 → 0.014 |
| NLVR2 (two images) | `noul` | 906 | 78.4% | 0.151 → 0.062 |
| OCR-VQA yes/no | `noul` | 996 | 86.9% | 0.075 → 0.063 |
| RAVEN | `choice (8-way)` | 542 | 77.1% | 0.171 → 0.094 |
| ScienceQA (Cauldron holdout) | `choice` | 301 | 86.7% | 0.086 → 0.057 |
| TQA | `choice` | 264 | 73.1% | 0.191 → 0.079 |
| VQA-RAD | `noul` | 62 | 88.7% | 0.096 → 0.115 |
| VQAv2 yes/no (Cauldron holdout) | `noul` | 1019 | 78.3% | 0.144 → 0.043 |
| VSR | `noul` | 160 | 86.9% | 0.099 → 0.040 |
| Visual7W | `choice` | 1729 | 87.2% | 0.054 → 0.065 |

- **Option order:** across the 4 cyclic rotations of the A-OKVQA options, accuracy is 60.0, 60.8, 60.9, 61.4 (spread 1.4 points). Averaging 4 permutations at inference gives 60.6%.
- **Latency:** the same architecture as the earlier checkpoint (about 41 ms per image question on an NVIDIA L4 in bf16 when measured side by side with ModernVBERT); the option mask adds no parameters. Not re-measured for this checkpoint.

## Training

- **Backbone:** SmolVLM-256M-Instruct. The vision tower is frozen; the language model and decision head are trained, about 150M parameters.
- **How options are read:** the image, state and question come first and the options are listed last, each scored from the hidden state at the end of its own line. New in this checkpoint: the option block attends to itself in both directions (`option_attention="bidirectional"`, a 4D attention mask), so every option's readout sees every other option; the image, state and question stay causal. The setting is stored in `vlm_agent_config.json` and applied at inference.
- **Objective:** Laya's RLCD policy gradient. Each step draws 4 Gaussian-noised copies of the option logits (sigma 0.3), scores each copy's probabilities with a strictly proper scoring rule (log score + 0.75 x spherical score, minus the ranked probability score for `score` questions) and uses the group-normalised score as the advantage; a soft cross-entropy term (weight 1) is added. The option order is shuffled per example.
- **Data:** 23 sets sampled equally, VLFeedback drawn 3x, no set repeated more than 4 times.
  - The Cauldron, 19 closed-form subsets (lettered choices, option lists, yes/no, RAVEN letters), 270k questions: AI2D, A-OKVQA, IconQA, Inter-GPS, ScienceQA, TQA, Visual7W, RAVEN, FigureQA, Hateful Memes, NLVR2, VSR, VQA-RAD, CLEVR, DVQA, MapQA, OCR-VQA, VQAv2, ChartQA.
  - VLFeedback: 158k (response, aspect) pairs rated 1-5 by GPT-4V for helpfulness or visual faithfulness; the question and response are the state text.
  - AVA: 20k photos with 1-10 human vote histograms, collapsed to 5 levels as soft targets.
  - RichHF-18K: 31k ratings of generated images (plausibility, alignment, aesthetics, overall) on a 5-point scale by three raters.
  - CrisisMMD: 2.2k disaster photos labelled little / mild / severe damage.
  - Each `score` question is asked with one of three instruction phrasings; the rubric levels are fixed. State text is capped at 1,200 characters.
- **Schedule:** 30,080 steps at batch 32 on one A100 40 GB (180 minutes), 2 epochs over 481k examples. Head LR 1.4e-4, backbone LR 2.8e-5, 3% warmup, cosine decay to 10%. The best mean validation accuracy was at the final step.
- **Calibration:** per-type temperatures fitted on the last 300 training records of each set, held out of training: choice 2.20, score 1.37, noul 2.13.
- `training_metrics.json` has every evaluation from the run.

## Limitations

- **`score` rubrics are in-distribution for response grading, image quality and damage severity.** Other rubrics work through the text, not through training: on an untrained "urgency for emergency services" rubric the model ranks a wildfire photo above a calm one, but the levels are not calibrated for it.
- **Confidence on 5-level `score` questions is low by design** (the loss rewards spreading probability over adjacent levels). Use the expected level, and the per-level probabilities, rather than `confidence`.
- **Overconfident on A-OKVQA** before calibration (raw ECE 0.30); rely on the calibrated outputs.
- **One temperature per question type.** The `score` temperature is a compromise across four sets; it slightly over-flattens VLFeedback and RichHF and under-flattens CrisisMMD.
- **Images are resized to a single 512 px tile.** NLVR2-style two-image states are supported.
- **Text-only states work but are unevaluated.**

## License

The weights are released under **CC BY-NC-SA 4.0**: the training data includes ScienceQA (via The Cauldron) and CrisisMMD, both under that license. The base model, SmolVLM, is Apache 2.0. VLFeedback and RichHF-18K images come from their upstream sources; AVA is a research dataset from DPChallenge. The code is Apache 2.0.
