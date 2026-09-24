# Laya Vision

[![Docs](https://img.shields.io/badge/docs-mkdocs-blue)](https://r33drichards.github.io/laya-vision/)

Typed, calibrated decisions about an **image plus optional text**, in one forward pass with no text generation. You give it a picture, some context and a set of questions. It answers each one as a multiple choice (`choice`), a yes/no probability (`noul`) or a level on a rubric you write (`score`), with probabilities you can act on at face value. The same model plays simple games from pixels: the screen is the image and the buttons are the options.

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

Inputs are cut to fit the checkpoint's token budgets: each option to 48 tokens (less when many options share `head_max_len`, 256), the instructions to what the options leave, the state text to what `max_len` leaves after the images. An answer whose question was cut carries a `truncated` field that says what was dropped; `predict(..., strict=True)` raises instead.

Laya Vision is an independent fork of [Laya](https://github.com/NandhaKishorM/laya) that replaces its ModernBERT text encoder with a small vision-language model. Laya's `predict(state, questions)` API, output schema, RLCD training objective and temperature calibration are unchanged. It is an experimental research project, not affiliated with Convai Innovations, the authors of Laya.

- **Try it:** [thaitea/laya-vision-demo](https://huggingface.co/spaces/thaitea/laya-vision-demo), a Space on free CPU, about 3 s per image. Source in `space/`.
- **Install:** `pip install -e .` plus `torchvision`, which the image processor needs. ModernVBERT needs `transformers >= 5.3`.
- **Documentation:** <https://r33drichards.github.io/laya-vision/>, built from [`site-docs/`](site-docs/).

## Checkpoints

| Checkpoint | Backbone | Trained on | A-OKVQA | ScienceQA | VQAv2 yes/no | `score` head | Params | Latency, L4 bf16 |
|---|---|---|---|---|---|---|---|---|
| [thaitea/laya-vision](https://huggingface.co/thaitea/laya-vision), **recommended** (same weights as [laya-vision-201m](https://huggingface.co/thaitea/laya-vision-201m)) | SmolVLM-256M cut to 20 of 30 language layers, options attend to each other | 19 Cauldron subsets + 4 rubric-scored sets + game frames | 59.8% | 82.4% | 71.4% | trained | 201M | 41 ms |
| [thaitea/laya-vision-smolvlm-256m-score](https://huggingface.co/thaitea/laya-vision-smolvlm-256m-score) (`thaitea/laya-vision` until 2026-09-24, revision `d1fbdc0`) | SmolVLM-256M, options attend to each other | 19 Cauldron subsets + 4 rubric-scored sets | 60.0% | 82.8% | 72.4% | trained | 237M | ~41 ms |
| [thaitea/laya-vision-modernvbert-250m](https://huggingface.co/thaitea/laya-vision-modernvbert-250m) | ModernVBERT-250M, bidirectional | 19 Cauldron subsets | 65.2% | 79.0% | 71.8% | untrained | 250M | 32 ms |
| [thaitea/laya-vision-smolvlm-256m](https://huggingface.co/thaitea/laya-vision-smolvlm-256m), the original | SmolVLM-256M | A-OKVQA, ScienceQA, VQAv2 yes/no | 61.8% | 86.6% | 73.4% | untrained | 237M | 41 ms |

- **Accuracies** are calibrated answers on the official validation splits. VQAv2 yes/no is a re-split of the official val set by image, so it is not comparable to published VQAv2 numbers.
- **Evidence:** the first two rows are backed by committed per-question predictions in [results/raw/](results/raw/). `python benchmarks/verify_published.py` recomputes them and checks them against this table and each checkpoint's metrics JSON ([recommended](docs/autoresearch-long-sep24-b64-metrics.json), [previous](docs/smolvlm-cauldron-score-bidir-full-metrics.json)). The rules for adding or changing numbers are in [AGENTS.md](AGENTS.md).
- **Latency** is the median `predict` call on an L4, preprocessing included. Raw milliseconds vary by about 50% between L4 hosts; timed on the same GPU, the recommended checkpoint takes 0.83× the time of the previous one.

**The recommended checkpoint against the previous one.** On the full evaluation suite (59,427 questions over 34 validation sets), it answers 69.1% correctly against 69.3%. It is within a point on 23 sets, ahead on 3 and behind on 8, mostly small ones (VQA-RAD, 62 questions, 80.6% against 88.7%). It is better calibrated (ECE 0.041 against 0.064), 15% smaller, and plays games: 0.35 on the autoresearch games benchmark (0 = random play, 1 = expert) against −0.04. Scorecards: [recommended](https://r33drichards.github.io/laya-vision/reference/evals/laya-vision-201m/), [previous](https://r33drichards.github.io/laya-vision/reference/evals/laya-vision/).

**`score` answers** mean something only on the first two rows, which were trained on rubric data. On held-out rubric data the previous checkpoint scores 54% over 5 levels on VLFeedback response grading (prior-only baseline 27.5%): [Score head results](https://r33drichards.github.io/laya-vision/reference/results/score-results/).

## Games

The recommended checkpoint on the autoresearch games benchmark (greedy play, one forward pass per move, seeds never used in training; 0 = random play, 1 = expert):

| Game | Score | Game | Score |
|---|---:|---|---:|
| ViZDoom basic | 0.99 | Atari Breakout | 0.20 |
| Atari Freeway | 0.81 | Snake 10×10 | 0.17 |
| Acrobot | 0.76 | Maze 6×6 | 0.02 |
| MountainCar | 0.64 | CartPole | −0.01 |
| Maze 4×4 | 0.32 | LunarLander | −0.44 |

These are from [the run's result](autoresearch/runs/full/long-sep24-b64.json). A variant that also trains the vision tower plays better (0.54, with 4×4 mazes 0.71 and Snake 0.80), but it loses several points on visual-reasoning sets ([its result](autoresearch/runs/full/long-sep24-vision-halflr.json)). To watch a checkpoint play, or run the longer games suite: [Play games](https://r33drichards.github.io/laya-vision/how-to/play-games/).

## Autoresearch

[autoresearch/](autoresearch/) adapts [karpathy/autoresearch](https://github.com/karpathy/autoresearch) to this model, and the recommended checkpoint came out of it. An agent edits one file (`experiment.py`), the fixed harness trains it for 15 minutes on an H100 and measures it, and a Pareto rule decides keep or discard.

- **Four objectives:** quality (accuracy over 34 eval sets minus calibration error), games, parameter count, and latency relative to a reference on the same L4. Noise margins come from repeat runs.
- **The agent's instructions** and what the tags so far found are in [autoresearch/program.md](autoresearch/program.md), with the lessons carried over from [autogo](https://github.com/r33drichards/autogo) and KataGo.
- **Long runs:** `modal run --detach autoresearch/full_run.py --commit <sha> --name <run> --minutes 120` trains a frontier recipe for longer and measures it the same way. It is resumable, and `--attach` takes over a run whose orchestrator died.

```bash
modal run autoresearch/harness.py --tag <tag>                    # one experiment: the committed experiment.py
python autoresearch/pareto.py show --tsv autoresearch/runs/<tag>/results.tsv
```

## Documentation

The documentation site, <https://r33drichards.github.io/laya-vision/>, is built from [`site-docs/`](site-docs/) with MkDocs:

- [Install](https://r33drichards.github.io/laya-vision/install/overview/) and [Tutorials](https://r33drichards.github.io/laya-vision/tutorials/overview/): the demo Space, and a first prediction in Python.
- [How-to guides](https://r33drichards.github.io/laya-vision/how-to/overview/): [calibrate on your own data](https://r33drichards.github.io/laya-vision/how-to/calibrate/), [run jobs on Modal](https://r33drichards.github.io/laya-vision/how-to/run-on-modal/), [evaluate a checkpoint](https://r33drichards.github.io/laya-vision/how-to/evaluate/), [play games](https://r33drichards.github.io/laya-vision/how-to/play-games/).
- [Concepts](https://r33drichards.github.io/laya-vision/concepts/overview/): [how it works](https://r33drichards.github.io/laya-vision/concepts/how-it-works/), the [architecture](https://r33drichards.github.io/laya-vision/concepts/architecture/), [calibration](https://r33drichards.github.io/laya-vision/concepts/calibration/), the [training data](https://r33drichards.github.io/laya-vision/concepts/data/).
- [Reference](https://r33drichards.github.io/laya-vision/reference/overview/): [`predict()` and the answer schema](https://r33drichards.github.io/laya-vision/reference/predict/), [checkpoints](https://r33drichards.github.io/laya-vision/reference/checkpoints/), and every result, from [typed answers vs generated JSON](https://r33drichards.github.io/laya-vision/reference/results/decision-vs-generation/) to [what didn't work](https://r33drichards.github.io/laya-vision/reference/results/what-didnt-work/).

To build it: `nix build .#docs` (the site lands in `./result`), or
`pip install mkdocs mkdocs-mermaid2-plugin && mkdocs build --strict`. Pull requests run the build, a link check and a
headless-browser check (`docs-check.yml`); pushes to `main` deploy it to GitHub Pages (`deploy-docs.yml`).

## License

- **Code:** Apache 2.0, inherited from Laya. See `LICENSE`.
- **Weights:** CC BY-NC-SA 4.0, because the training data includes ScienceQA and CrisisMMD, which carry that license.

## Credits

Laya Vision is an independent fork of [Laya](https://github.com/NandhaKishorM/laya) by Convai Innovations, Apache 2.0, and is not affiliated with them. The text decision model, its typed-question API (`choice` / `score` / `noul`), the RLCD training objective and the temperature calibration are theirs. This fork adds image input, the SmolVLM and ModernVBERT backbones, the rubric data, the game work and the autoresearch loop. The design is described in the author's [write-up](https://dev.to/nandakishor_m_6cc0adfde9f/i-built-non-autoregressive-decision-models-a-year-ago-then-a-frontier-lab-called-it-a-18me). The original text model and its Colab/Kaggle notebooks live in the upstream repo. ModernVBERT is by its authors under MIT; SmolVLM by Hugging Face under Apache 2.0.
