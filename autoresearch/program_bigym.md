# autoresearch for Laya Vision: the BiGym track

The same loop as `program.md` (one file edited, 15 minutes of H100 training, a fixed harness measures it), with a
different objective. The human's call: **BiGym control is the objective; quality and games are guard-rails (kept only
while they stay within their noise margins of the baseline); size and latency are frozen at the 20-layer model.**

## What is measured

| | what | role |
|---|---|---|
| `bigym` | mean over 6 BiGym tasks of ½ normalized dense progress + ½ normalized success rate (below) | **the objective**: keep needs `bigym` > best kept + `BIGYM_MARGIN` |
| `quality` | as in `program.md` (macro accuracy over 34 eval sets minus ECE) | guard: >= baseline - `QUALITY_GUARD` |
| `games` | as in `program.md` (10-game normalized mean) | guard: >= baseline - `GAMES_GUARD` |
| `params_m` | parameters of the saved model | guard: <= baseline x 1.01 (size frozen) |
| `latency_x` | L4 latency ratio | reported only |

The margins live at the top of `pareto.py`'s bigym section (`BIGYM_MARGIN` 0.03, `QUALITY_GUARD` 0.01,
`GAMES_GUARD` 0.05, all **provisional** until the noise runs). The baseline is the tag's first result.

**The BiGym score** (`autoresearch/bigym_eval.py`, fixed). The saved checkpoint plays ReachTarget,
ReachTargetSingle, DrawerTopOpen, DrawerTopClose, WallCupboardOpen and WallCupboardClose greedy (one forward per
decision over `bigym_question(task, frames)`, 37 primitives), 32 episodes per task on seeds 780,000-785,031,
capped at 60 decisions (reach) / 150 (cupboards). Each episode scores its best dense progress:

- reach: `1 - d / d0` clipped to [0, 1] (`d` the allowed wrist's distance to the ball, `d0` at reset);
- cupboards: the fraction of the way from the part's joint state at reset to the goal (drawer, or the mean over the
  two wall doors);
- a success counts 1.

Per task `(model - random) / (expert - random)`, clipped to [-0.5, 1.5], against `bigym_baselines.json`: random is
`random_policy` on the same episodes, expert is the privileged oracle on the reach tasks and 1.0 on the cupboards
(BiGym's human demos succeed 90-100% there; there is no primitive oracle). 0 = random play, 1 = expert. The
same normalization applies to the success rate (random's measured rate against the oracle's on reach, 1.0 on the
cupboards), and a task's score is the mean of the two: dense progress alone let a policy score 0.79 on
DrawerTopClose by shoving the drawer with its body without ever closing it. Success
rates and each task's most played primitives are reported next to it: a policy that collapses onto one move
shows there first.

What the benchmark plays is read from the checkpoint: `cfg["bigym_frames"]` (1 or 4 head frames per decision,
default 4; `experiment_bigym.py` writes it) and its `head_max_len` / `max_len` (a question that would be cut raises).

## Setup

1. Agree on a tag, e.g. `bigym-<date>`; the branch `autoresearch/<tag>`. A tag keeps one profile.
2. Read `autoresearch/harness.py`, `pareto.py` (the bigym section), `bigym_eval.py`, `experiment_bigym.py`,
   `laya/bigymgames.py`, `laya/bigymdemos.py`, `laya/vlm_train.py`.
3. Data: the pool's `bigym` kind (`pool-v5/bigym`: all train records of `bigym_v2c_bc_f4`, `bigym_v2c_bc_f1`,
   `bigym_v2c_probe`) and `bigym_demos` (`pool-v5/bigym_demos`: the cupboard tasks' train demos as waypoints).
   Build missing parts with `modal run autoresearch/harness.py --prepare-pool`.
4. Baseline: run the unmodified `experiment_bigym.py`, then repeat it twice (`--desc "baseline repeat"`) and set
   the three margins from the spread (tell the human if it is larger than the provisional values).

## Running an experiment

```bash
git commit -am "<what this tries>"
modal run autoresearch/harness.py --tag <tag> --profile bigym --experiment autoresearch/experiment_bigym.py \
    > autoresearch/runs/<tag>/run.log 2>&1
grep -A30 "^---" autoresearch/runs/<tag>/run.log     # scores, the per-task table, keep / discard and why
python autoresearch/pareto.py show --tsv autoresearch/runs/<tag>/results.tsv
```

After training, calibration and the quality eval (the H100 job), the latency job, the four games jobs and twelve
BiGym jobs (reach tasks x 2 chunks, cupboard tasks x 3, one L4 each) run in parallel. Keep / discard / crash and the git and results
hygiene are as in `program.md`. The result JSON has everything per task (every episode's progress and success,
action counts, timings) under `bigym`.

## Rules

You CAN change anything in `experiment_bigym.py`: the data mix and share of BiGym, 1 vs 4 frames
(`BIGYM_FRAMES`, and train on the matching set), losses and soft targets, freezing, LRs, heads (value, next-move),
a training loop of your own, and simulator rollouts inside `train()`.

You CANNOT:
- modify `harness.py`, `pareto.py`, `bigym_eval.py`, `bigym_baselines.json`, `games_eval.py`, `toolkit.py`, or the
  data;
- train on the BiGym sets' val splits (not in the pool; do not read them from the volume), the eval sets, or the
  calibration tail;
- **roll out on the eval's seeds.** `ctx.bigym_game(task, seed)` only takes training seeds: below 100,000, or a
  train demo's seed (`ctx.bigym_demos(task)[i]["seed"]`: that episode starts where the demo did). Seeds 780,000+
  raise. Do not build envs yourself to get around it;
- use test-time search: the benchmark plays greedy, one forward per decision. Search, lookahead and expert
  relabelling inside `train()` are fine;
- exceed the 15-minute training budget (rollouts count), or change the architecture's size (the `params_m` guard).

Rollouts in `train()`: the H100 image has MuJoCo 3.14 and BiGym (the eval's pins); `ctx.bigym_game` picks EGL.
Making an env costs a few seconds, so make a few and reuse them (`env=`). A decision is ~5 ms of physics without the
camera and ~0.1-0.15 s with an OSMesa render (EGL is much faster); the demo follower's `lookahead` tries all 37
primitives per decision (~0.2 s without camera).

## Where it starts

The first fine-tune (`eval-results/bigym-ft-20260927b.json`, 2.2 A100-hours on the same mix as
`experiment_bigym.py`) reached 20% success on ReachTarget (random 5%) and 0% on all four cupboard tasks, and its
play collapsed: ~90% of DrawerTopClose decisions were `BASE_FORWARD`, ~90% of WallCupboardClose's `LEFT_HAND_LEFT`.
Its per-decision accuracy on the demo labels was ~20%. The BC labels come from a demo follower whose moves are
often near-ties (several primitives bring the robot about as close to the next waypoint), and one-hot targets on
near-ties teach a prior, not a policy. Offline BC also never shows the states its own mistakes lead to.

## Reference points (2026-09-29)

| checkpoint | bigym | ReachTarget | ReachTargetSingle | DrawerTopOpen | DrawerTopClose | WallCupboardOpen | WallCupboardClose |
|---|---|---|---|---|---|---|---|
| zero-shot `autoresearch/full/long-sep24-b64/best` | -0.120 | -0.500 | -0.195 | 0.000 | -0.031 | 0.000 | +0.008 |
| fine-tune `bigym-ft-20260927b/best` (2.2 A100-h) | +0.143 | +0.114 | +0.002 | 0.000 | +0.741 | 0.000 | 0.000 |
| `experiment_bigym.py`, 15 min (`bigym-dev-20260929`) | +0.096 | -0.219 | -0.003 | 0.000 | +0.794 | +0.001 | 0.000 |

(`autoresearch/runs/bigym-checks/`, `autoresearch/runs/bigym-dev-20260929/`.) Random dense progress is 0.38 on
ReachTarget, 0.20 on ReachTargetSingle, 0.03 on DrawerTopClose and 0 on the other three. No model succeeds on a
cupboard task yet. Beware what the dense score rewards: the 15-minute model's DrawerTopClose 0.80 comes from walking
into the drawer (`BASE_FORWARD` on 93% of its decisions), which pushes it most of the way shut without ever meeting
BiGym's 0.1 tolerance; read the success rates and top actions next to the score.

Costs (the dev run): 15.0 min of training (1,674 steps at batch 64, ~107k samples, 36k of them 4-frame BiGym; the
loader was the bottleneck, 37% data wait); a BiGym container takes 1.5 min (reach) to 4 min (cupboards) once
running, but one of twelve took 12.4 min on a slow host, and a new deployment's first run pays the snapshots and
L4 queueing: 47 min end to end for the first run.

## Findings of `bigym-sep29` (24 experiments + 3 baselines)

Kept chain (`autoresearch/runs/bigym-sep29/results.tsv`), each on top of the last:

| commit | change | bigym | success [RT RTS DTO DTC WCO WCC] | quality |
|---|---|---|---|---|
| 837f06a | baseline (4 frames, `experiment_bigym.py` as started) | 0.036 | 0 everywhere | 0.685 |
| 5c1f85d | 1 frame (bc_f1 only), LRs x2, 26 loader workers | 0.225 | .53 .19 0 .34 0 0 | 0.676 |
| 8a3db45 | + reach-oracle rollouts in the budget at bc_f1's weight, backbone LR 1.5e-5 | 0.307 | .62 .78 0 0 0 0 | 0.680 |
| 826bf70 | + BC split by task, weights 20/10/5/5 toward DrawerTopClose | 0.351 | .59 .94 0 .25 0 0 | 0.688 |
| a17717e | + backbone LR 2e-5, batch 32 | 0.452 | .72 .97 0 1.00 0 0 | 0.678 |

Read these with the noise in mind. Baselines at the floor repeat within 0.003, but above it one recipe spreads
widely: 5c1f85d scored 0.225 then 0.259, and a17717e's repeat scored 0.340 (DrawerTopClose 0/32, quality 0.674
under the guard). The last keep is partly a lucky draw; the recipe's expected score is nearer 0.35-0.40. The keep
margin (0.03) fits the floor's noise, not this; a next tag should repeat a candidate before keeping it, or play more
episodes.

What did not help: 4-frame recipes with any data change (reach oracle, per-primitive rebalancing + smoothing, DART
with one-hot or soft lookahead-cost targets: all stayed at ~0.03); EMA of the weights; lower backbone LR; a smaller
BiGym share or game share; training only the last 10 layers; DART on DrawerTopClose only; batch 16.

**The cap makes three cupboard tasks unreachable.** Under `CAPS` (150 decisions) the demo follower itself finishes
DrawerTopClose (median 103 decisions on the val demos) but rarely WallCupboardClose (median 172) and never
DrawerTopOpen (258) or WallCupboardOpen (295). With `CUPBOARD_EXPERT = 1.0`, those three tasks cap the benchmark at
about half its range whatever the model does. A next tag should raise their caps (e.g. 350) or normalise against
the follower's success at the cap.

## Ideas to start from

- **DAgger with the lookahead follower as the expert.** Roll the current model out (greedy, or with some
  exploration) on training episodes, and label the visited states with `laya.bigymdemos.lookahead(game,
  waypoint, banned, part)`. It needs a waypoint to aim at: roll out on a train demo's seed (`ctx.bigym_demos`)
  and track the demo's waypoints as `follow` does (reached / patience / grasp gate), or, on other seeds,
  relabel with the waypoints of the nearest demo (closest initial robot / part state). Label only a sample of the
  visited states (the labelling is the slow part: 37 tried moves each). Mix the new frames in with the BC data
  (never only the newest).
- **Soft targets from lookahead costs.** `lookahead` scores every primitive; turn the costs into a distribution
  (softmax of -cost / T), so near-ties share the mass instead of one arbitrary winner. Or plain label smoothing
  on the BC targets (autogo's 0.1).
- **More BiGym share or longer effective training**: the fine-tune drew ~50% BiGym for 2.2 h; 15 minutes is much
  less. Weigh a higher share against the quality and games guards.
- **1 vs 4 frames**: `BIGYM_FRAMES = 1` with the `bc_f1` set is 4x cheaper per sample (more steps in 15 min);
  motion may or may not matter at this accuracy.
- **Class balancing**: the labels are dominated by a few moves (`BASE_FORWARD`, hand forward/up/left); reweight
  per primitive, or per task (WallCupboardOpen has 1,603 records, WallCupboardClose 6,340).
- **Value / next-move heads**: `"value_head": true` with a progress target per frame (the probe's progress level,
  or the dense progress of a rollout), `"next_head": true` with the next decision's primitive (the BC ids are
  `<task>-<seed>-<decision>`, `laya.vlm_train.with_next_targets` builds it). Both are auxiliary, never played.
- **The probe as an auxiliary**: it teaches done / progress; check whether more or less of it helps control.
- **Reach tasks** have no BC data at all: `laya.bigymgames.oracle_action` is a free expert there (on training
  seeds), for BC or DAgger.
