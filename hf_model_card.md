---
license: cc-by-nc-sa-4.0
base_model: HuggingFaceTB/SmolVLM-256M-Instruct
library_name: laya
pipeline_tag: visual-question-answering
datasets:
- HuggingFaceM4/the_cauldron
- MMInstruction/VLFeedback
- HuggingFaceM4/A-OKVQA
- derek-thomas/ScienceQA
- lmms-lab-encoder/VQAv2
language:
- en
tags:
- laya
- calibration
- decision-model
- vision
- smolvlm
- games
---

# Laya Vision

Typed, calibrated decisions about an **image plus optional text**, in one forward pass with no text generation. Give it a picture, some context and a set of questions. It answers each one as a multiple choice (`choice`), a yes/no probability (`noul`) or a level on a rubric you write (`score`), with probabilities you can act on at face value. It also plays simple games from pixels: the screen is the image and the buttons are the options.

This checkpoint is 201M parameters: SmolVLM-256M-Instruct cut to 20 of its 30 language layers, then trained for 2 hours on question-answering data and game frames. It was found by the [autoresearch loop](https://github.com/r33drichards/laya-vision/tree/main/autoresearch) in the code repository.

It is the same model as [thaitea/laya-vision-201m](https://huggingface.co/thaitea/laya-vision-201m). The previous `thaitea/laya-vision` (237M, no game training) is still in this repo's history at revision `d1fbdc0612fbe3b3d8ec6f54d328b195d35bb338` and in [thaitea/laya-vision-smolvlm-256m-score](https://huggingface.co/thaitea/laya-vision-smolvlm-256m-score).

- **Code:** [github.com/r33drichards/laya-vision](https://github.com/r33drichards/laya-vision)
- **Status:** experimental research. An independent fork of [Laya](https://github.com/NandhaKishorM/laya), not affiliated with Convai Innovations, its authors.

## Usage

```bash
git clone https://github.com/r33drichards/laya-vision && pip install -e ./laya-vision torchvision
```

```python
import laya
from PIL import Image

agent = laya.load_vlm("thaitea/laya-vision")
result = agent.predict(
    {"image": Image.open("photo.jpg"), "note": "customer says it arrived broken"},
    {
        "damage":   {"type": "score",  "instructions": "How much damage does the item show?",
                     "criteria": ["none", "cosmetic: scratches or dents", "functional: parts broken or missing", "destroyed"]},
        "category": {"type": "choice", "instructions": "What kind of item is this?",
                     "criteria": ["electronics", "clothing", "furniture", "food", "other"]},
        "outdoors": {"type": "noul",   "instructions": "Was the photo taken outdoors?"},
    },
)
a = result["answers"]
a["damage"]["score"], a["category"]["choice"], a["outdoors"]["noul"]   # expected level 0-3, top option, P(true)
```

Pin a revision (`laya.load_vlm("thaitea/laya-vision", revision=...)`) if you need the weights not to change under you.

## Results

### Official validation splits

Calibrated answers, as `predict` returns them. These three numbers are backed by committed per-question predictions in the code repository (`results/raw/`), which `benchmarks/verify_published.py` recomputes.

| Dataset | Question type | n | Accuracy | Previous checkpoint |
|---|---|---:|---:|---:|
| A-OKVQA | 4-way `choice` | 1,138 | 59.8% | 60.0% |
| ScienceQA, image subset | 2–5-way `choice` | 2,097 | 82.4% | 82.8% |
| VQAv2 yes/no\* | `noul` | 5,000 | 71.4% | 72.4% |

\* A re-split of the official VQAv2 validation set by image, the only official split with answers, so not comparable to published VQAv2 numbers.

### Full evaluation suite

From the repository's `full_eval` run on this checkpoint ([scorecard](https://github.com/r33drichards/laya-vision/blob/ded975c10f51c57912f7c7be3cb31486807d5029/docs/evals/laya-vision-201m.md)), next to the same suite on the previous checkpoint:

| | This model | Previous checkpoint |
|---|---:|---:|
| Parameters | 201M | 237M |
| Accuracy, 59,427 questions over 34 validation sets | 69.1% | 69.3% |
| Mean accuracy over the 34 sets | 71.6% | 71.9% |
| Calibration error (ECE), all questions | 0.041 | 0.064 |
| Latency on an L4, same GPU, relative | 0.83× | 1.00× |

On 23 of the 34 sets it is within a point of the previous checkpoint; it is ahead by more than a point on 3 and behind on 8. The largest drops are on the smallest sets: VQA-RAD (62 questions) 80.6% against 88.7%, InterGPS (94) 28.7% against 35.1%.

### Games

Played greedily from pixels, one forward pass per move, on seeds never used in training. The score is normalized per game: 0 is random play, 1 is the scripted or trained expert (the autoresearch benchmark, clipped to [−0.5, 1.5]):

| Game | Normalized score |
|---|---:|
| ViZDoom basic | 0.99 |
| Atari Freeway | 0.81 |
| Acrobot | 0.76 |
| MountainCar | 0.64 |
| Maze 4×4 | 0.32 |
| Atari Breakout | 0.20 |
| Snake 10×10 | 0.17 |
| Maze 6×6 | 0.02 |
| CartPole | −0.01 |
| LunarLander | −0.44 |
| **Mean** | **0.35** |

The previous checkpoint was never trained on games and scores −0.04 on the same benchmark.

## Training

- **Start:** the previous `thaitea/laya-vision` checkpoint, cut to its first 20 language-model layers. The vision tower (12 layers) is kept and frozen.
- **Data:** every train record of 19 closed-answer subsets of The Cauldron and 4 rubric-scored sets (VLFeedback, AVA, RichHF-18K, CrisisMMD), minus a calibration tail. 45% of training draws were game frames:
  - Maze, Snake and classic control (CartPole, Acrobot, MountainCar, LunarLander), generated with every best move as a soft target;
  - Atari Freeway and Breakout (a PPO expert's action distributions) and ViZDoom basic (a scripted expert).
- **Objective:** Laya's: soft cross-entropy plus a strictly proper scoring rule, with the option order shuffled at random.
- **Schedule:** 28,414 steps at batch 64 on one H100, 120 minutes (1.82M samples). Head LR 5e-5, backbone LR 1e-5, 20 warmup steps, cosine decay to 10%.
- **Calibration:** per-type temperatures fitted on the held-out calibration tail: choice 3.86, score 2.10, noul 3.05.
- **How the recipe was chosen:** the autoresearch loop trains candidate recipes for 15 minutes and keeps the ones that extend a Pareto frontier over quality, games, parameters and latency. The 2-hour run is its batch-64 frontier point trained 8× longer.

## Limitations

- **Games are uneven.** LunarLander got worse with training, and 6×6 mazes are rarely solved.
- **Small sets dropped.** VQA-RAD and InterGPS are several points below the previous checkpoint (few questions, so noisy, but real).
- **Human vote spreads.** On sets scored against how people split their votes, it beats a predict-the-average baseline on only 3 of 9.
- **Domain.** Trained on everyday photos, diagrams, charts, rubric-graded images and simple game screens. Calibrate on your own labelled data (`agent.calibrate`) before acting on probability thresholds in another domain.
- **One image per state,** resized to a single 512 px tile.

## License

The weights are released under **CC BY-NC-SA 4.0**, because the training data includes ScienceQA and CrisisMMD, which carry that license. The base model, SmolVLM, is Apache 2.0. The code is Apache 2.0.
