# Laya Vision

[![Docs](https://img.shields.io/badge/docs-mkdocs-blue)](https://r33drichards.github.io/laya-vision/)

Typed, calibrated decisions about an **image plus optional text**, in one forward pass with no text generation. You give it a picture, some context and a set of questions; it answers each one as a multiple choice (`choice`), a yes/no probability (`noul`) or a graded level on a rubric you write (`score`), with probabilities you can act on at face value.

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

Inputs are cut to fit the checkpoint's token budgets: each option to 48 tokens (and shorter when many options must share `head_max_len`, 256), the instructions to what the options leave, the state's text to what `max_len` leaves after the images. An answer whose question was cut carries a `truncated` field, absent otherwise: `{"options": [labels cut], "indistinguishable": [[label, label], ...], "instructions": bool, "instructions_tokens_dropped": n, "state_tokens_dropped": n}`, where `indistinguishable` lists options that are the same tokens once cut, which the model cannot tell apart. `predict(..., strict=True)` raises `ValueError` instead, naming the question and what would be cut.

Laya Vision is an independent fork of [Laya](https://github.com/NandhaKishorM/laya) that replaces its ModernBERT text encoder with a small vision-language model. Laya's `predict(state, questions)` API, output schema, RLCD training objective and temperature calibration are unchanged. It is an experimental research project, not affiliated with Convai Innovations, the authors of Laya.

- **In your browser:** <https://r33drichards.github.io/laya-vision/demo/> runs the model on your own GPU with WebGPU (or WASM on the CPU); no server, and the image never leaves the page.
- **Try it:** [thaitea/laya-vision-demo](https://huggingface.co/spaces/thaitea/laya-vision-demo), a Space on free CPU, about 3 s per image. Source in `space/`.
- **Install:** `pip install -e .` plus `torchvision`, which the image processor needs. ModernVBERT needs `transformers >= 5.3`.
- **Documentation:** <https://r33drichards.github.io/laya-vision/>, built from [`site-docs/`](site-docs/).

## Checkpoints

| Checkpoint | Backbone | Trained on | A-OKVQA | ScienceQA | VQAv2 yes/no | `score` head | Latency, L4 bf16 |
|---|---|---|---|---|---|---|---|
| [thaitea/laya-vision](https://huggingface.co/thaitea/laya-vision), **recommended** (same weights as [laya-vision-smolvlm-256m-score](https://huggingface.co/thaitea/laya-vision-smolvlm-256m-score)) | SmolVLM-256M, options attend to each other | 19 Cauldron subsets + 4 rubric-scored sets | 60.0% | 82.8% | 72.4% | trained | ~41 ms |


Accuracies are on the official validation splits (VQAv2 yes/no is a re-split of the official val set by image, so not comparable to published VQAv2 numbers). Calibrated ECE pooled over all of a checkpoint's validation sets is 0.02 to 0.035, but it varies by set: for the recommended checkpoint it is 0.16 on A-OKVQA, 0.035 on ScienceQA and 0.077 on VQAv2 yes/no ([row-level evidence](results/raw/README.md)). The original checkpoint is ahead on ScienceQA because it made 12 passes over that one train split; the others made 3 to 4 as one of 19 to 23 sets, and are far broader: the recommended one averages 75% over 26 validation sets, and 93.7% on IconQA, 91.8% on DVQA, 89.9% on Hateful Memes.

The full scorecard for the recommended checkpoint covers 34 validation sets, human-vote calibration, the games suite and latency: [Scorecard](https://r33drichards.github.io/laya-vision/reference/evals/laya-vision/).

The recommended row's accuracies are backed by committed per-row predictions in [results/raw/](results/raw/); `python benchmarks/verify_published.py` recomputes them (and calibrated ECE) and checks them against this table and [its metrics](docs/smolvlm-cauldron-score-bidir-full-metrics.json). Rules for adding or changing numbers: [AGENTS.md](AGENTS.md).

The recommended checkpoint is the only one whose `score` answers mean anything. On held-out rubric data it scores 54% over 5 levels on VLFeedback response grading (prior-only baseline 27.5%), is 0.8 levels off on average against 1.4 for the baseline, and 0.38 levels off on 3-level damage severity. Full tables, the ordinal metrics and what each run changed are in [Score head results](https://r33drichards.github.io/laya-vision/reference/results/score-results/).

## Documentation

The documentation site, <https://r33drichards.github.io/laya-vision/>, is built from [`site-docs/`](site-docs/) with MkDocs:

- [Install](https://r33drichards.github.io/laya-vision/install/overview/) and [Tutorials](https://r33drichards.github.io/laya-vision/tutorials/overview/): a first prediction in Python, and the model in your browser.
- [How-to guides](https://r33drichards.github.io/laya-vision/how-to/overview/): [calibrate on your own data](https://r33drichards.github.io/laya-vision/how-to/calibrate/), [run jobs on Modal](https://r33drichards.github.io/laya-vision/how-to/run-on-modal/), [evaluate a checkpoint](https://r33drichards.github.io/laya-vision/how-to/evaluate/), [export for the browser](https://r33drichards.github.io/laya-vision/how-to/export-for-the-web/), [play games](https://r33drichards.github.io/laya-vision/how-to/play-games/).
- [Concepts](https://r33drichards.github.io/laya-vision/concepts/overview/): [how it works](https://r33drichards.github.io/laya-vision/concepts/how-it-works/), the [architecture](https://r33drichards.github.io/laya-vision/concepts/architecture/), [calibration](https://r33drichards.github.io/laya-vision/concepts/calibration/), the [training data](https://r33drichards.github.io/laya-vision/concepts/data/).
- [Reference](https://r33drichards.github.io/laya-vision/reference/overview/): [`predict()` and the answer schema](https://r33drichards.github.io/laya-vision/reference/predict/), [checkpoints](https://r33drichards.github.io/laya-vision/reference/checkpoints/), and every result, from [typed answers vs generated JSON](https://r33drichards.github.io/laya-vision/reference/results/decision-vs-generation/) to [what didn't work](https://r33drichards.github.io/laya-vision/reference/results/what-didnt-work/).

To build it: `nix build .#docs` (the site, with the browser demo under `demo/`, lands in `./result`), or
`pip install mkdocs mkdocs-mermaid2-plugin && mkdocs build --strict`. Pull requests run the build, a link check and a
headless-browser check (`docs-check.yml`); pushes to `main` deploy it to GitHub Pages (`deploy-docs.yml`).

## License

- **Code:** Apache 2.0, inherited from Laya. See `LICENSE`.
- **Weights:** CC BY-NC-SA 4.0, because the training data includes ScienceQA and CrisisMMD, which carry that license.

## Credits

Laya Vision is an independent fork of [Laya](https://github.com/NandhaKishorM/laya) by Convai Innovations, Apache 2.0, and is not affiliated with them. The text decision model, its typed-question API (`choice` / `score` / `noul`), the RLCD training objective and the temperature calibration are theirs; this fork adds image input, the SmolVLM and ModernVBERT backbones, the rubric data and the game work. The design is described in the author's [write-up](https://dev.to/nandakishor_m_6cc0adfde9f/i-built-non-autoregressive-decision-models-a-year-ago-then-a-frontier-lab-called-it-a-18me). The original text model and its Colab/Kaggle notebooks live in the upstream repo. ModernVBERT is by its authors under MIT; SmolVLM by Hugging Face under Apache 2.0.
