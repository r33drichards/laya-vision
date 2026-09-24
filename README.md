# Laya Vision

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

- **Install:** `pip install -e .` plus `torchvision`, which the image processor needs. ModernVBERT needs `transformers >= 5.3`.
- **Try it:** [thaitea/laya-vision-demo](https://huggingface.co/spaces/thaitea/laya-vision-demo), a Space on free CPU, about 3 s per image. Source in `space/`.
- **In the browser, no server (experimental):** `web-demo/` runs a model with ONNX Runtime Web from files exported with `scripts/export_onnx.py`; see [web-demo/README.md](web-demo/README.md).

Laya Vision is an independent fork of [Laya](https://github.com/NandhaKishorM/laya) that replaces its ModernBERT text encoder with a small vision-language model. Laya's `predict(state, questions)` API, output schema, RLCD training objective and temperature calibration are unchanged. It is an experimental research project, not affiliated with Convai Innovations, the authors of Laya.

## Checkpoints

| Checkpoint | Backbone | Trained on | A-OKVQA | ScienceQA | VQAv2 yes/no | `score` head | Params | Latency, L4 bf16 |
|---|---|---|---|---|---|---|---|---|
| [thaitea/laya-vision](https://huggingface.co/thaitea/laya-vision), **recommended** (same weights as [laya-vision-201m](https://huggingface.co/thaitea/laya-vision-201m)) | SmolVLM-256M cut to 20 of 30 language layers, options attend to each other | 19 Cauldron subsets + 4 rubric-scored sets + game frames | {AOK} | {SQA} | {VQA} | trained | 201M | 41 ms |
| [thaitea/laya-vision-smolvlm-256m-score](https://huggingface.co/thaitea/laya-vision-smolvlm-256m-score) (`thaitea/laya-vision` until 2026-09-24, revision `d1fbdc0`) | SmolVLM-256M, options attend to each other | 19 Cauldron subsets + 4 rubric-scored sets | 60.0% | 82.8% | 72.4% | trained | 237M | ~41 ms |
| [thaitea/laya-vision-modernvbert-250m](https://huggingface.co/thaitea/laya-vision-modernvbert-250m) | ModernVBERT-250M, bidirectional | 19 Cauldron subsets | 65.2% | 79.0% | 71.8% | untrained | 250M | 32 ms |
| [thaitea/laya-vision-smolvlm-256m](https://huggingface.co/thaitea/laya-vision-smolvlm-256m), the original | SmolVLM-256M | A-OKVQA, ScienceQA, VQAv2 yes/no | 61.8% | 86.6% | 73.4% | untrained | 237M | 41 ms |

- **Accuracies** are calibrated answers on the official validation splits. VQAv2 yes/no is a re-split of the official val set by image, so it is not comparable to published VQAv2 numbers.
- **Evidence:** the first two rows are backed by committed per-question predictions in [results/raw/](results/raw/). `python benchmarks/verify_published.py` recomputes them and checks them against this table and each checkpoint's metrics JSON. The rules for adding or changing numbers are in [AGENTS.md](AGENTS.md).
- **Latency** is the median `predict` call on an L4, preprocessing included. Raw milliseconds vary by about 50% between L4 hosts; timed on the same GPU, the recommended checkpoint takes 0.83× the time of the previous one.

**The recommended checkpoint versus the previous one.** On the full evaluation suite (59,427 questions over 34 validation sets), it answers 69.1% correctly against 69.3%. It is within a point on 23 sets, ahead on 3 and behind on 8, mostly small ones (VQA-RAD, 62 questions, 80.6% against 88.7%). It is better calibrated (ECE 0.041 against 0.064), 15% smaller, and plays games: 0.35 on the autoresearch games benchmark (0 = random play, 1 = expert) against −0.04. Scorecards: [recommended](docs/evals/laya-vision-201m.md), [previous](docs/evals/laya-vision.md).

**`score` questions** are only meaningful on the first two rows, which were trained on rubric data. On held-out rubric data the previous checkpoint scores 54% over 5 levels on VLFeedback response grading (prior-only baseline 27.5%); tables and ordinal metrics are in [docs/score-results.md](docs/score-results.md).

## How it works

A question is rendered as its instructions followed by its options, one per line, after the image and the state text. The backbone encodes the whole sequence once per image, and a small head reads one logit per option from the hidden state at that option's marker. The answer is the softmax over the options, divided by a per-type temperature fitted after training. Nothing is generated.

- **Causal backbones (SmolVLM)** can only read an option after everything before it, so the options go last, each read at its line terminator. A 4D attention mask lets the option block attend to itself in both directions (`option_attention="block"`), which is worth about a point on reasoning-heavy sets. Random option orders in training and permutation averaging at inference reduce the remaining order bias.
- **Bidirectional backbones (ModernVBERT)** use Laya's original format: a `[MASK]` in front of each option, read from a marker that sees the whole sequence. They are the fastest, and trail SmolVLM by about 7 points on the Cauldron holdouts.
- **Training** is Laya's RLCD objective: Gaussian noise on the option logits, several noisy copies scored with a strictly proper scoring rule (log plus spherical, plus a ranked probability score for `score` questions), the group-normalized score as the policy-gradient advantage, and a soft cross-entropy term. The vision tower stays frozen.
- **Budgets:** each option is cut to 48 tokens (less when many options share `head_max_len`, 256), the instructions to what the options leave, and the state text to what `max_len` leaves after the images. An answer whose question was cut carries a `truncated` field describing what was dropped, including `indistinguishable` options that become the same tokens once cut. `predict(..., strict=True)` raises instead.

Diagrams of every variant are in [docs/architecture.md](docs/architecture.md). The model is `laya/vlm.py` and the training loop `laya/vlm_train.py`.

## Calibrating on your own data

Probabilities come from option scores divided by a **temperature**: above 1 flattens them, below 1 sharpens them. The checkpoint's temperatures were fitted on its own held-out data. On your images and questions the model may be over- or under-confident, so if you act on a threshold (auto-approve above 0.9, send to a human below), calibrate on a few hundred of your own labelled questions first.

```python
cal = agent.calibrate(
    [
        {"state": {"image": img, "note": note}, "image_id": "a17",
         "questions": {"damage": damage_q, "outdoors": outdoors_q},
         "labels": {"damage": 2, "outdoors": False}},            # level index for score, option name for choice, bool for noul
        ...
    ],
    group_key="image_id",                                        # questions about one image stay together in every split
)
print(cal.summary())                                             # fitted temperatures; ECE raw / checkpoint / fitted, with 95% intervals
cal.save("my-calibration.json")

result = agent.predict(state, questions, calibration=laya.Calibration.load("my-calibration.json"))
result = agent.predict(state, questions, temperature={"noul": 1.4})   # or set one by hand, for this call only
```

`calibrate` fits one temperature per question type (types with fewer than 30 labelled questions share one) and scores it on rows it was not fitted on (5 folds that never split a group), next to the raw and checkpoint temperatures, each with a 95% interval. If `cal.evidence["all"]["ece_improvement"]` includes 0, keep the checkpoint's temperatures. Temperatures never change the chosen answer, only how sure the model says it is. A calibration records the checkpoint it was fitted for and warns if used with another. The code is plain numpy in `laya/calibration.py`.

## Playing games

The screen is the image and the options are the game's buttons. `examples/atari_live.py` and `examples/vizdoom_live.py` let you watch a checkpoint play in a local window.

The recommended checkpoint on the autoresearch games benchmark (greedy play, one forward pass per move, seeds never used in training; 0 = random play, 1 = expert):

| Game | Score | Game | Score |
|---|---:|---|---:|
| ViZDoom basic | 0.99 | Snake 10×10 | 0.17 |
| Atari Freeway | 0.81 | Maze 6×6 | 0.02 |
| Acrobot | 0.76 | CartPole | −0.01 |
| MountainCar | 0.64 | LunarLander | −0.44 |
| Maze 4×4 | 0.32 | Atari Breakout | 0.20 |

Result: [autoresearch/runs/full/long-sep24-b64.json](autoresearch/runs/full/long-sep24-b64.json). A variant that also trains the vision tower plays better (0.54: 4×4 mazes 0.71, Snake 0.80) but loses several points on visual-reasoning sets, so it is not the recommended checkpoint ([its result](autoresearch/runs/full/long-sep24-vision-halflr.json)).

`modal run modal_app.py::games_eval --model <run>/best` runs the longer games suite:
- **Atari** Freeway, Breakout and Galaxian, against random play and the expert where expert data exists.
- **ViZDoom** `basic`, against the scripted expert, random and always-attack.
- **Maze** at 4×4, 6×6 and 8×8, and **Snake** on 10×10: small seeded games in `laya/gridgames.py`.
- **Classic control** from Gymnasium (CartPole, Acrobot, MountainCar, LunarLander). A single frame hides velocity, so the previous frame is ghosted under the current one (`laya/controlgames.py`).

Earlier game-only training, with a two-frame input and DAgger rounds, is in [docs/game-training.md](docs/game-training.md).

## Autoresearch

[autoresearch/](autoresearch/) adapts [karpathy/autoresearch](https://github.com/karpathy/autoresearch) to this model: an agent edits one file (`experiment.py`), the fixed harness trains it for 15 minutes on an H100 and measures it, and a Pareto rule decides keep or discard. The recommended checkpoint came out of it.

- **Four objectives,** kept on a Pareto frontier with noise margins measured from repeat runs:
  - quality: accuracy over 34 eval sets minus calibration error;
  - games: the benchmark above;
  - parameter count;
  - latency relative to a reference, timed on the same L4.
- **Instructions** for the agent, and what the tags so far found, are in [autoresearch/program.md](autoresearch/program.md). The lessons carried over from [autogo](https://github.com/r33drichards/autogo) and from KataGo are there too.
- **Long runs:** `modal run --detach autoresearch/full_run.py --commit <sha> --name <run> --minutes 120` trains a frontier recipe for longer and measures it the same way. It is resumable, and `--attach` takes over a run whose orchestrator died.

```bash
modal run autoresearch/harness.py --tag <tag>                    # one experiment: the committed experiment.py
python autoresearch/pareto.py show --tsv autoresearch/runs/<tag>/results.tsv
```

## Speed: typed answers versus generated JSON

Fifteen questions (6 `choice`, 5 `noul`, 4 `score`) about one image, on one L4 in bf16, median of 5 runs, measured on the previous checkpoint (`laya-vision-smolvlm-256m-score`); the recommended one is 0.83× its time:

| Path | Time | Output tokens | Result |
|---|---:|---:|---|
| `predict`, default `batch_size=8` (2 forward passes) | **0.144 s** | **0** | 15 typed answers with probabilities |
| `predict`, `batch_size=15` (1 forward pass) | **0.098 s** | **0** | the same answers |
| Base SmolVLM-256M-Instruct asked for one compact JSON array | 3.216 s (22x) | 89 | prose, no JSON array; 0/15 usable |
| The same with the base model's image splitting (17 views) | 0.613 s (4.3x) | 9 | 0/15 usable |
| The base model asked one question per `generate` call, 15 calls | 4.489 s (31x) | 105 | 1/15 strictly valid |

A systems comparison, not a quality one: the base model cannot follow the compact-array instruction at this size, so its time is the time to say whatever it said. Rerun with `modal run modal_app.py::decision_vs_generation --output results/raw/<new>.json`. The [raw report](results/raw/decision-vs-generation-l4.json) has the prompts, timings, generated text and pinned revisions.

## Data

Everything is a prepared dataset on the `laya-datasets` Modal volume: `/data/vqa/<name>/{train,val}.jsonl` plus `images/`, one record per question with its type, instructions, criteria and label, optionally a soft target.

- **The Cauldron** (`laya/cauldron.py`): the 19 subsets of [HuggingFaceM4/the_cauldron](https://huggingface.co/datasets/HuggingFaceM4/the_cauldron) with closed answers. 270k questions.
- **Rubric-scored sets** (`laya/rubric.py`): VLFeedback, AVA, RichHF-18K and CrisisMMD as `score` questions, each level a short rubric clause. Cleaning notes: [docs/score-data.md](docs/score-data.md).
- **Held-out evaluation sets** (`laya/evalsets.py`): KonIQ-10k, EvalMuse-40K, CIFAR-10H, FER+, VizWiz and POPE. Most keep each image's human vote histogram, so `evaluate` also reports cross-entropy against how people split.
- **The original three:** A-OKVQA, ScienceQA and VQAv2 yes/no.
- **Games:**
  - Atari frames labelled by a PPO expert, and ViZDoom frames by a scripted one;
  - Maze, Snake and classic-control frames, generated on the fly by `autoresearch/toolkit.py`, with every best move as a soft target.

## Running it on Modal

`modal_app.py` expects the volumes `laya-hf-cache`, `laya-datasets` and `laya-checkpoints`, and a `huggingface-thaitea` secret for the publish jobs. From a Claude Code on the web session, run `bash .claude/skills/modal/check.sh` first; the [modal skill](.claude/skills/modal/SKILL.md) covers setup and detached runs, and the [evals skill](.claude/skills/evals/SKILL.md) covers evaluation.

```bash
modal run modal_app.py::try_model --image photo.jpg --questions q.json --run <run>/best
modal run modal_app.py::test                                              # GPU tests + latency
modal run modal_app.py::prepare_cauldron                                  # also prepare_score, prepare_eval
modal run --detach modal_app.py::finetune_long --run-name my-run --epochs 2 --max-passes 4 --max-minutes 240 \
    --option-attention block --mix score_vlfeedback=3 --datasets cauldron,score --val-datasets vqa,cauldron,score
modal run modal_app.py::full_eval --model my-run/best                    # datasets, games suite and latency, one results JSON
modal run modal_app.py::evidence --run my-run/best --datasets vqa         # row-level predictions for published numbers
modal run modal_app.py::publish --repo user/name --run my-run/best --card hf_model_card.md
```

`full_eval` writes `eval-results/<run>-<commit>.json`; `python scripts/eval_report.py <result files> --doc docs/evals/<name>.md` turns it into a Markdown scorecard. The same suite runs from GitHub Actions (Actions → eval → Run workflow) and comments the report on the branch's pull request; it needs the `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` secrets.

## What didn't work

- **A SigLIP projector into Laya's text encoder:** never learned to use the image in 5 runs; shuffled images scored the same as real ones.
- **Annealing the cross-entropy weight to zero:** tied on accuracy and made the raw model more overconfident.
- **Balancing AVA's levels on the train split only:** the head learned a flatter histogram than the val voters produce.
- **A third epoch over the same rubric data:** half a point on VLFeedback, nothing elsewhere.
- **In autoresearch** (15-minute runs; details in `program.md`):
  - a value head as an auxiliary loss;
  - distillation from the full model within the budget;
  - keeping every other layer instead of the first N;
  - cutting vision-tower layers.

  None of them helped. Test-time search is excluded by design: the benchmark scores what the model learned, one forward pass per move.

## License

- **Code:** Apache 2.0, inherited from Laya. See `LICENSE`.
- **Weights:** CC BY-NC-SA 4.0, because the training data includes ScienceQA and CrisisMMD, which carry that license.

## Credits

Laya Vision is an independent fork of [Laya](https://github.com/NandhaKishorM/laya) by Convai Innovations, Apache 2.0, and is not affiliated with them. The text decision model, its typed-question API (`choice` / `score` / `noul`), the RLCD training objective and the temperature calibration are theirs. This fork adds image input, the SmolVLM and ModernVBERT backbones, the rubric data, the game work and the autoresearch loop. The design is described in the author's [write-up](https://dev.to/nandakishor_m_6cc0adfde9f/i-built-non-autoregressive-decision-models-a-year-ago-then-a-frontier-lab-called-it-a-18me). ModernVBERT is by its authors under MIT; SmolVLM by Hugging Face under Apache 2.0.
