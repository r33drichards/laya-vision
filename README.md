# Laya Vision

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

Laya Vision is an independent fork of [Laya](https://github.com/NandhaKishorM/laya) that replaces its ModernBERT text encoder with a small vision-language model. Laya's `predict(state, questions)` API, output schema, RLCD training objective and temperature calibration are unchanged. It is an experimental research project, not affiliated with Convai Innovations, the authors of Laya.

- **Try it:** [thaitea/laya-vision-demo](https://huggingface.co/spaces/thaitea/laya-vision-demo), a Space on free CPU, about 3 s per image. Source in `space/`.
- **Install:** `pip install -e .` plus `torchvision`, which the image processor needs. ModernVBERT needs `transformers >= 5.3`.

## Checkpoints

| Checkpoint | Backbone | Trained on | A-OKVQA | ScienceQA | VQAv2 yes/no | `score` head | Latency, L4 bf16 |
|---|---|---|---|---|---|---|---|
| [thaitea/laya-vision](https://huggingface.co/thaitea/laya-vision), **recommended** (same weights as [laya-vision-smolvlm-256m-score](https://huggingface.co/thaitea/laya-vision-smolvlm-256m-score)) | SmolVLM-256M, options attend to each other | 19 Cauldron subsets + 4 rubric-scored sets | 60.0% | 82.8% | 72.4% | trained | ~41 ms |
| [thaitea/laya-vision-modernvbert-250m](https://huggingface.co/thaitea/laya-vision-modernvbert-250m) | ModernVBERT-250M, bidirectional | 19 Cauldron subsets | 65.2% | 79.0% | 71.8% | untrained | 32 ms |
| [thaitea/laya-vision-smolvlm-256m](https://huggingface.co/thaitea/laya-vision-smolvlm-256m), the original | SmolVLM-256M | A-OKVQA, ScienceQA, VQAv2 yes/no | 61.8% | 86.6% | 73.4% | untrained | 41 ms |

Accuracies are on the official validation splits (VQAv2 yes/no is a re-split of the official val set by image, so not comparable to published VQAv2 numbers). Calibrated ECE is 0.02 to 0.03 for all three over their full validation sets. The original checkpoint is ahead on ScienceQA because it made 12 passes over that one train split; the others made 3 to 4 as one of 19 to 23 sets, and are far broader: the recommended one averages 75% over 26 validation sets, and 93.7% on IconQA, 91.8% on DVQA, 89.9% on Hateful Memes.

The recommended checkpoint is the only one whose `score` answers mean anything. On held-out rubric data it scores 54% over 5 levels on VLFeedback response grading (prior-only baseline 27.5%), is 0.8 levels off on average against 1.4 for the baseline, and 0.38 levels off on 3-level damage severity. Full tables, the ordinal metrics and what each run changed are in [docs/score-results.md](docs/score-results.md).

## How it works

A `score` question with 4 levels, a `choice` with 5 options and a `noul` are each rendered as a question followed by their options, one per line, after the image and the state text. The backbone encodes the whole sequence once per image, and a small head reads one logit per option from the hidden state at each option's marker. Softmax over the options, divided by a per-type temperature fitted after training, is the answer. Nothing is generated.

- **Causal backbones (SmolVLM)** can only read an option after everything before it, so the options go last and each is read at its line terminator. That leaves an option-order bias of about a point of accuracy, which random orders in training and permutation averaging at inference reduce. The recommended checkpoint adds a 4D attention mask that lets the option block attend to itself in both directions (`option_attention="bidirectional"`), which is worth about a point on the reasoning-heavy sets.
- **Bidirectional backbones (ModernVBERT)** use Laya's original format unchanged: a `[MASK]` in front of each option, read from a marker that sees the whole sequence. No order bias to fix, and the fastest at inference; it trails SmolVLM by about 7 points on the Cauldron holdouts, mostly on reasoning sets like RAVEN and TQA.
- **Training** is Laya's RLCD objective: Gaussian noise is added to the option logits, several noisy copies are scored with a strictly proper scoring rule (log plus spherical, plus a ranked probability score for `score` questions), and the group-normalised score is the policy-gradient advantage, with a soft cross-entropy term added. The vision tower stays frozen.

Diagrams of every variant and the shared head are in [docs/architecture.md](docs/architecture.md); the model code is `laya/vlm.py` and the training loop `laya/vlm_train.py`.

## Data

Everything is a prepared dataset on the `laya-datasets` Modal volume: `/data/vqa/<name>/{train,val}.jsonl` plus `images/`, one record per question with its type, instructions, criteria and label, optionally a soft target.

- **The Cauldron** (`laya/cauldron.py`, `prepare_cauldron`): the 19 subsets of [HuggingFaceM4/the_cauldron](https://huggingface.co/datasets/HuggingFaceM4/the_cauldron) whose answers are closed. Lettered choices and option lists become `choice`, yes/no turns become `noul`, RAVEN's letters become an 8-way `choice`; numbers, captions and free text are skipped. 270k questions.
- **Rubric-scored sets** (`laya/rubric.py`, `prepare_score`): VLFeedback (response helpfulness and visual faithfulness, 1 to 5), AVA (photo aesthetics, human vote histograms as soft targets), RichHF-18K (generated-image plausibility, alignment, aesthetics, overall) and CrisisMMD (damage severity). Each level is a short rubric clause in the style you would write for `predict`, with several instruction phrasings per question. How they were cleaned, and why, is in [docs/score-data.md](docs/score-data.md).
- **The original three** (`aokvqa`, `scienceqa`, `vqav2_yesno`): the official train splits, prepared on the [`siglip-projector-experiment`](https://github.com/r33drichards/laya-vision/tree/siglip-projector-experiment) branch.
- **Games**: frames auto-labelled by a scripted expert or a trained agent; see below.

## Running it on Modal

`modal_app.py` expects the volumes `laya-hf-cache`, `laya-datasets` and `laya-checkpoints`, and a `huggingface-thaitea` secret for the publish jobs. Dataset arguments take names or the groups `vqa`, `cauldron` and `score`.

```bash
modal run modal_app.py::try_model --image photo.jpg --questions q.json --run cauldron-score-2ep-bidir-full/best
modal run modal_app.py::test                                              # GPU tests + latency, both backbones
modal run modal_app.py::prepare_cauldron                                  # -> /data/vqa/cauldron_<subset>
modal run modal_app.py::prepare_score                                     # -> /data/vqa/score_<name>
modal run --detach modal_app.py::finetune_long --run-name my-run --epochs 2 --max-passes 4 --max-minutes 240 \
    --option-attention bidirectional --mix score_vlfeedback=3 --datasets cauldron,score --val-datasets vqa,cauldron,score
modal run --detach modal_app.py::finetune_long --backbone ModernVBERT/modernvbert --run-name my-run --datasets cauldron
modal run modal_app.py::evaluate --run-name my-run/best                   # every prepared val set, raw and calibrated
modal run --detach modal_app.py::split_bench                              # SmolVLM2, image splitting off / 1024 / 2048
modal run modal_app.py::publish --repo user/name --run my-run/best --card hf_model_card_score.md
modal run modal_app.py::publish_space                                     # push space/ to the demo Space
```

`finetune_long` keeps data, objective, schedule and evaluation identical across backbones, which is what makes the checkpoints above comparable. It also takes `--backbone HuggingFaceTB/SmolVLM2-256M-Video-Instruct` (same size and code path as SmolVLM), `--split-edge` to turn on the processor's image splitting (the image is resized to that longest edge and cut into 512 tiles plus a global view: up to 5 views at 1024, 17 at 2048, against 1 without) and `--max-len` for the sequence cap, which is 1024 by default and raised to fit the tiles when splitting. `split_bench` trains one SmolVLM2 run per split setting on a six-set subset and writes accuracy per set, tokens per question and L4 latency to `/ckpt/smolvlm2/split-bench/results.md`. Runs save under `/ckpt/smolvlm/` or `/ckpt/modernvbert/` with a `best/` and `last/` checkpoint and a `metrics.json` of every evaluation. Locally, `python -m laya.vlm_train --synthetic --steps 3` is a smoke run.

## Playing games

The screen is the image and the options are the game's buttons. `examples/atari_live.py` and `examples/vizdoom_live.py` let you watch a checkpoint play in a local window. Trained for 7 minutes on 20,000 auto-labelled frames, it plays ViZDoom `basic` at expert level (mean reward +75.4 against the expert's +75.8 over 50 unseen episodes); the Atari work, with a two-frame input and DAgger rounds, is in [docs/game-training.md](docs/game-training.md).

## What didn't work

- **A SigLIP projector into Laya's text encoder** ([`siglip-projector-experiment`](https://github.com/r33drichards/laya-vision/tree/siglip-projector-experiment)): kept text answers bit-identical but never learned to use the image in 5 runs; accuracy with shuffled images matched accuracy with real ones.
- **Annealing the cross-entropy weight to zero** so training ends on the proper scoring rule alone: tied on accuracy and made the raw model more overconfident ([docs/modernvbert-cauldron.md](docs/modernvbert-cauldron.md)).
- **Balancing AVA's levels on the train split only**: the head learned a flatter vote histogram than the val voters produce and lost to a prior-only baseline until the balancing was removed.
- **A third epoch over the same rubric data**: half a point on VLFeedback, nothing elsewhere. More passes are flat; the next gains need new rubric data.

## License

- **Code:** Apache 2.0, inherited from Laya. See `LICENSE`.
- **Weights:** CC BY-NC-SA 4.0, because the training data includes ScienceQA and CrisisMMD, which carry that license.

## Credits

Laya Vision is an independent fork of [Laya](https://github.com/NandhaKishorM/laya) by Convai Innovations, Apache 2.0, and is not affiliated with them. The text decision model, its typed-question API (`choice` / `score` / `noul`), the RLCD training objective and the temperature calibration are theirs; this fork adds image input, the SmolVLM and ModernVBERT backbones, the rubric data and the game work. The design is described in the author's [write-up](https://dev.to/nandakishor_m_6cc0adfde9f/i-built-non-autoregressive-decision-models-a-year-ago-then-a-frontier-lab-called-it-a-18me). The original text model and its Colab/Kaggle notebooks live in the upstream repo. ModernVBERT is by its authors under MIT; SmolVLM by Hugging Face under Apache 2.0.
