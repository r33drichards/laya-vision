# Laya Vision

**Laya Vision** makes typed, calibrated decisions about an **image plus optional text**, in one forward pass with
no text generation. You give it a picture, some context and a set of questions; it answers each one as a multiple
choice (`choice`), a yes/no probability (`noul`) or a graded level on a rubric you write (`score`), with
probabilities you can act on at face value.

**[Try the demo](https://huggingface.co/spaces/thaitea/laya-vision-demo)**: a Hugging Face Space that runs the published checkpoint on a free CPU, a few seconds per
image. [How to use it](tutorials/space-demo.md).

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

Laya Vision is an independent fork of [Laya](https://github.com/NandhaKishorM/laya) that replaces its ModernBERT
text encoder with a small vision-language model. Laya's `predict(state, questions)` API, output schema, RLCD
training objective and temperature calibration are unchanged. It is an experimental research project, not
affiliated with Convai Innovations, the authors of Laya.

## Start here

- [Install](install/overview.md): the Python package and what it needs.
- [Tutorials](tutorials/overview.md): the demo Space, and a first prediction in Python.
- [How-to guides](how-to/overview.md): calibrate on your data, run jobs on Modal, evaluate, play games.
- [Concepts](concepts/overview.md): how the model reads an answer, its architecture, calibration and training data.
- [Reference](reference/overview.md): the `predict` API and answer schema, checkpoints, file formats and every published result.

## At a glance

| Area | What it covers |
|------|----------------|
| [Question types](reference/predict.md#question-types) | `choice`, `score` (a rubric you write) and `noul` (yes/no), several per call |
| [How it works](concepts/how-it-works.md) | One forward pass per image; a logit per option read from a marker token; nothing generated |
| [Calibration](concepts/calibration.md) | Per-type temperatures fitted after training, and [refitting them on your data](how-to/calibrate.md) |
| [Checkpoints](reference/checkpoints.md) | The recommended `thaitea/laya-vision` and what its numbers are backed by |
| [Demo](tutorials/space-demo.md) | The published checkpoint in a Hugging Face Space, no install |
| [Results](reference/evals/laya-vision.md) | The scorecard over 34 validation sets, games and latency, plus every experiment report |

## License and credits

- **Code:** Apache 2.0, inherited from Laya
  ([`LICENSE`](https://github.com/r33drichards/laya-vision/blob/main/LICENSE)).
- **Weights:** CC BY-NC-SA 4.0, because the training data includes ScienceQA and CrisisMMD, which carry that
  license.

Laya Vision is an independent fork of [Laya](https://github.com/NandhaKishorM/laya) by Convai Innovations, Apache
2.0, and is not affiliated with them. The text decision model, its typed-question API (`choice` / `score` /
`noul`), the RLCD training objective and the temperature calibration are theirs; this fork adds image input, the
SmolVLM and ModernVBERT backbones, the rubric data and the game work. The design is described in the author's
[write-up](https://dev.to/nandakishor_m_6cc0adfde9f/i-built-non-autoregressive-decision-models-a-year-ago-then-a-frontier-lab-called-it-a-18me).
The original text model and its Colab/Kaggle notebooks live in the upstream repo. ModernVBERT is by its authors
under MIT; SmolVLM by Hugging Face under Apache 2.0.
