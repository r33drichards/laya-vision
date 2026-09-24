# Run jobs on Modal

Data preparation, training, evaluation, the benchmarks and publishing run on [Modal](https://modal.com) from
[`modal_app.py`](https://github.com/r33drichards/laya-vision/blob/main/modal_app.py) (app `laya-smolvlm`); the Atari
jobs are in `modal_atari_*.py`.

## What the app expects

- The volumes `laya-hf-cache` (the Hugging Face cache), `laya-datasets` (prepared datasets) and `laya-checkpoints`
  (training runs).
- A `huggingface-thaitea` secret, for the jobs that publish to the Hub.

Dataset arguments take names or the groups `vqa`, `cauldron`, `score` and `eval`. Run every command from the
repository root: the app uploads the local `laya/` code, so the jobs run the code of the checkout you launch from.

From a Claude Code on the web session, run `bash .claude/skills/modal/check.sh` first: the CLI needs
`modal[api-proxy-support]` to get through the session's proxy. The `modal` skill in
[`.claude/skills/modal`](https://github.com/r33drichards/laya-vision/blob/main/.claude/skills/modal/SKILL.md) covers
setup, a smoke test, detached runs and reading results off the volumes.

## Commands

```bash
modal run modal_app.py::try_model --image photo.jpg --questions q.json --run cauldron-score-2ep-bidir-full/best
modal run modal_app.py::test                                              # GPU tests + latency, both backbones
modal run modal_app.py::prepare_cauldron                                  # -> /data/vqa/cauldron_<subset>
modal run modal_app.py::prepare_score                                     # -> /data/vqa/score_<name>
modal run modal_app.py::prepare_eval                                      # -> /data/vqa/eval_<name>
modal run --detach modal_app.py::finetune_long --run-name my-run --epochs 2 --max-passes 4 --max-minutes 240 \
    --option-attention block --mix score_vlfeedback=3 --datasets cauldron,score --val-datasets vqa,cauldron,score
modal run --detach modal_app.py::finetune_long --backbone ModernVBERT/modernvbert --run-name my-run --datasets cauldron
modal run modal_app.py::full_eval --model my-run/best                   # every eval at once, one results JSON (no --detach: results are collected locally)
modal run modal_app.py::evaluate --run-name my-run/best                   # every prepared val set, raw and calibrated
modal run modal_app.py::evaluate --run-name my-run/best --datasets eval   # only the held-out evaluation sets
modal run --detach modal_app.py::split_bench                              # SmolVLM2, image splitting off / 1024 / 2048
modal run modal_app.py::games_eval --model my-run/best --out games.json   # Atari, ViZDoom, Maze, Snake, classic control + baselines
modal run modal_app.py::publish --repo user/name --run my-run/best --card hf_model_card_score.md
modal run modal_app.py::publish_space                                     # push space/ to the demo Space
```

What each dataset is: [Training data](../concepts/data.md). Evaluating a checkpoint end to end:
[Evaluate a checkpoint](evaluate.md).

## Training

`finetune_long` keeps data, objective, schedule and evaluation identical across backbones, which is what makes the
checkpoints comparable.

- `--backbone HuggingFaceTB/SmolVLM2-256M-Video-Instruct` trains on SmolVLM2 (same size and code path as SmolVLM).
- `--split-edge` turns on the processor's image splitting: the image is resized to that longest edge and cut into
  512 px tiles plus a global view, up to 5 views at 1024 and 17 at 2048, against 1 without. What that buys:
  [Image splitting on SmolVLM2](../reference/results/split-bench.md).
- `--max-len` sets the sequence cap, 1024 by default and raised to fit the tiles when splitting.

Runs save under `/ckpt/smolvlm/` or `/ckpt/modernvbert/` with a `best/` and `last/` checkpoint and a `metrics.json`
of every evaluation. Locally, `python -m laya.vlm_train --synthetic --steps 3` is a smoke run.

## Rules for results

Evaluation and evidence jobs are create-only: they write to new paths (a new run name, a new `--name` for
`modal run modal_app.py::evidence`), and never overwrite a file under `results/raw/`, a checkpoint or a prepared
dataset. A published number needs committed row-level evidence; the rules are in
[`AGENTS.md`](https://github.com/r33drichards/laya-vision/blob/main/AGENTS.md).
