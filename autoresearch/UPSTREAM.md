# Upstream

This directory adapts [karpathy/autoresearch](https://github.com/karpathy/autoresearch) at commit
`228791fb499afffb54b46200aca536f79142f117` (2026-03-25), MIT licensed ("License: MIT" in its README).

## What was taken, and what changed

| upstream | here | change |
|---|---|---|
| `program.md`: the agent's instructions and loop | `program.md` | Same loop: edit, commit, run, keep or reset, log, never stop. Rewritten for Modal and for three objectives. |
| `train.py`: the one file the agent edits | `experiment.py` | Instead of a nanochat GPT, a recipe for the Laya Vision model: starting checkpoint, what to cut, image size, data mix, training. `build(ctx)` and `train(agent, ctx)` are called by the harness. |
| `prepare.py`: fixed constants, data and `evaluate_bpb` | `harness.py` | The fixed 5-minute budget is kept. The metric is our eval suite (34 sets, sampled), plus parameter count and L4 latency. Runs on Modal: an H100 trains, an L4 times. |
| keep if `val_bpb` improves | `pareto.py` | Keep if the result is not dominated on (quality, params, latency) within noise margins; progress is the frontier's hypervolume. |
| `results.tsv`, untracked | `runs/<tag>/results.tsv` and one JSON per experiment | Committed on the run branch, because cloud sessions are ephemeral. |

Upstream's `train.py`, `prepare.py`, `analysis.ipynb` and dependencies (nanochat LLM pretraining, a BPE tokenizer,
FineWeb shards) are not copied, since none of it applies to this model.
