# autoresearch for Laya Vision

Autonomous research on the Laya Vision decision model: you edit one file, the harness trains it for 5 minutes on an
H100 and measures it, and you keep what pushes the Pareto frontier out. Adapted from
[karpathy/autoresearch](https://github.com/karpathy/autoresearch) (see `UPSTREAM.md`).

The goal: **the best eval quality and game play for the smallest, fastest model.** There is no single number to
minimize. Four objectives are measured for every experiment, and a result is kept when no earlier kept result is at
least as good on all four:

| objective | what | better |
|---|---|---|
| `quality` | macro accuracy over 34 eval sets (300 fixed questions each) minus the calibration error (ECE) on the questions with one right answer | higher |
| `games` | mean normalized score over the games suite (Maze, Snake, CartPole, Acrobot, MountainCar, LunarLander, Atari Freeway and Breakout, ViZDoom basic): per game (model - random) / (expert - random), clipped to [-0.5, 1.5], fixed seeds (`autoresearch/games_eval.py`) | higher |
| `params_m` | parameters of the saved model, millions | lower |
| `latency_x` | median `predict` time on an L4 in bf16, preprocessing included, divided by the base checkpoint's timed in the same container (1.0 = as fast as the released model) | lower |

Progress is the frontier's **hypervolume**: how much of the quality × games × size × latency space it covers. A
smaller model that is only slightly worse is a win, as is a better player at the same size. `pareto.py` makes the
keep / discard call, not you.

## Setup

1. **Agree on a run tag** with the human, e.g. today's date (`sep23`). The branch `autoresearch/<tag>` must not exist.
2. **Branch**: `git checkout -b autoresearch/<tag>` from `main`.
3. **Read the in-scope files**:
   - `autoresearch/harness.py`: the fixed harness: budget, eval sets, calibration, metrics. Do not modify.
   - `autoresearch/pareto.py`: the keep / discard rule. Do not modify.
   - `autoresearch/experiment.py`: the file you modify.
   - `laya/vlm.py` and `laya/vlm_train.py`: the model and the training loop your experiment calls. You may read them
     for ideas; change behaviour by writing it in `experiment.py`, not by editing them.
4. **Modal**: `modal volume ls laya-checkpoints` must work. In a Claude Code cloud sandbox, set it up as
   `.claude/skills/evals/SKILL.md` section 1 describes (the proxy extra and CA bundle, in a scratch venv).
5. **Data**: `modal volume ls laya-datasets autoresearch` must show the data pool (`pool-v1`). If it is missing, build
   it once with `modal run autoresearch/harness.py --prepare-pool` (it reads the prepared `cauldron_*`, `score_*` and
   `eval_*` sets). Every image the harness uses comes from this pool: reading the datasets' small image files
   straight from the volume is too slow to keep an H100 fed.
6. **Baseline**: run the unmodified `experiment.py` first. It continues training the released checkpoint for 5
   minutes, so it should land near the released model's numbers.
7. **Noise**: run the baseline twice more (`--desc "baseline repeat"`), look at the spread of the four objectives (the games margin in `pareto.py` is provisional until this is done),
   and if it exceeds the margins at the top of `pareto.py` tell the human before going on. Those margins decide what
   counts as a real improvement.

## Running an experiment

```bash
git commit -am "<what this tries>"                      # results are keyed by commit
modal run autoresearch/harness.py --tag <tag> > autoresearch/runs/<tag>/run.log 2>&1
grep -A12 "^---" autoresearch/runs/<tag>/run.log        # the summary and the keep/discard status
```

Never pass `--detach`: the local entrypoint collects the result, decides and writes the files. The first run of a
new harness version deploys it and snapshots the loaded data (slower); later runs restore that snapshot. A run is
5 minutes of training plus loading, calibration and eval, then the L4 latency job and the four games jobs (one per
family) in parallel.
Start it as a background shell command and wait for it; do not let its output into your context.

The harness writes `autoresearch/runs/<tag>/<commit>.json` (every metric, per dataset) and appends a row to
`autoresearch/runs/<tag>/results.tsv`:

```
commit	quality	macro_acc	ece_hard	games	params_m	latency_x	status	description
```

It also prunes saved checkpoints that are no longer on the frontier from `/ckpt/autoresearch/<tag>/`.

## The loop

LOOP FOREVER:

1. Look at the frontier: `python autoresearch/pareto.py show --tsv autoresearch/runs/<tag>/results.tsv`.
2. Pick a frontier point to build on. Usually the tip, but to push a different part of the frontier (smaller,
   faster), start from that point's recipe: `git checkout <commit> -- autoresearch/experiment.py`.
3. Change `experiment.py` with one idea, and `git commit` it.
4. Run the harness as above.
5. If the grep finds no summary, the run crashed: `tail -n 60` the log. Fix and re-run if it is a slip (a typo, a
   shape mismatch); if the idea is broken, move on. A crash is recorded as `crash` automatically.
6. **keep**: leave the commit. **discard** or **crash**: `git reset --hard HEAD~1` to drop the experiment commit.
7. Commit the log files either way, so the record survives: `git add autoresearch/runs/<tag> && git commit -m
   "results: <commit> <status>"`, then `git push origin autoresearch/<tag>`.

Do not stop to ask whether to continue. The human may be asleep and expects experiments to keep running until they
interrupt. If you run out of ideas, re-read the frontier and the per-dataset metrics in the result JSONs for where
quality is lost, combine near-misses, or try a bigger change.

## What you can and cannot do

You CAN change anything in `experiment.py`: the starting checkpoint, what gets cut, image size, which sets to train on
and in what mix, the loss weights, freezing, learning rates, batch size, schedules, or a training loop of your own.

You CANNOT:
- modify `harness.py`, `pareto.py`, `games_eval.py`, `game_baselines.json`, `toolkit.py`, or the eval data;
- train on anything but `ctx.train_examples()` (the pool's 2,000 examples per Cauldron and score set),
  `ctx.game_examples()` and what `toolkit.py` generates (on seeds below 100,000; the games benchmark plays 700,000+):
  no val splits, no `eval_*` sets, no calibration tail;
- use test-time search (`agent.cfg["search"]` / `laya.search`): the model plays greedy, one forward per move;
- exceed the 5-minute training budget (the harness fails runs over budget + 60 s);
- add dependencies beyond what the harness image installs.

A change only counts if it survives `agent.save` and reload. The harness measures the checkpoint it reloads from
disk, so an architecture change must also be written into the backbone config, as the helpers in `experiment.py`
do.

## Lessons from autogo

[autogo](https://github.com/r33drichards/autogo) runs the same kind of loop on an AlphaGo-style Go player, and its
architecture plays far better than ours. What carries over, with its evidence:

- **Check the target plumbing before anything clever.** autogo's single biggest gain was a one-line label-mask fix
  (policy trained on only the winner's moves, discarding half the search labels): +0.034 on its metric, more than any
  architecture change.
- **Train the policy on soft teacher distributions, not one-hot actions.** autogo's policy learns MCTS visit counts
  (visits^(1/T)), and falls back to label smoothing (0.1) where there is no search. Our loss takes soft targets;
  `toolkit.py` builds them for Maze and Snake (every shortest-path move, not one).
- **Give the policy a value head.** autogo trains policy and value 1:1 and cutting the value weight to 0.25 hurt the
  policy (0.322 vs 0.334). `laya` models take `"value_head": true`, and game examples can carry a `value` target.
- **Variety beats per-episode strength, and don't throw away old data.** More games at 1024 simulations beat fewer at
  2048; training only on the newest data overfit (0.273 vs 0.305).
- **Augment with the game's symmetries** where they hold (mirrors in Maze; not in Atari, whose screens have text).
- **Size the LR schedule to the time budget** so it fully decays inside the 5 minutes; `laya.vlm_train.train` already
  decays on wall-clock progress, keep it that way if you write your own loop.
- **Change one thing at a time**, in this order when unsure: learning rate and schedule, then batch size and steps,
  then architecture. Simpler wins at equal score.
- **Noise compounds; gate on it.** autogo's gains stopped compounding when exploration noise accumulated across
  iterations. Here the margins in `pareto.py` are the gate: re-measure them (repeat the baseline) when the harness
  changes.

## Ideas to start from

- **Cut depth**: `KEEP_TEXT_LAYERS` (30 in SmolVLM-256M) and `KEEP_VISION_LAYERS` (12), then use the 5 minutes to
  recover. Which layers go matters: try keeping every other layer instead of the first N.
- **Fewer image tokens**: `IMAGE_SIZE` 384 or 256 (36 or 16 tokens instead of 64). This is usually the cheapest
  latency win.
- **Smaller head**: fewer head transformer layers, or none.
- **Distillation**: train the cut model toward the full checkpoint's probabilities instead of only the labels.
- **Data mix**: the eval sets reward breadth; weight the weakest groups.
- **Game data**: mix `toolkit` game examples (soft BFS targets for Maze and Snake, expert frames for classic
  control) into the training stream, and the pool's Atari and ViZDoom expert frames via `ctx.game_examples()`.
- **Value head**: `"value_head": true` with `value` targets, as an auxiliary loss.
- **Calibration**: the harness fits temperatures, but training with the proper scoring rules (`w_ce_schedule`, `w_sph`)
  changes how well a single temperature can fix things.

**Simplicity**: all else equal, simpler is better. A tiny gain that adds a lot of code is not worth it; the same
result with less code is a win.
