# Training Laya Vision to play games

Laya Vision can act as a game policy. Each step, the screen goes in as the image and a `choice` question whose options are the game's buttons picks the move. The question builders live in `laya/games.py`, shared by the live viewers and the training-data jobs so both ask exactly the same question.

- `examples/atari_live.py --game <Name>` plays any of the 104 Atari games in `ale-py` in a local window.
  `--model <checkpoint dir>` plays a trained one; it reads `atari_frames` from the checkpoint, so a two-frame
  model sends `{"images": [previous, current]}` with no extra flag (`--frames 1|2` overrides, `--sample`
  draws from the probabilities instead of taking the top action). The previous frame follows the `expert2f`
  rule used in training and in `play_atari`: a copy of the current frame on an episode's first step and
  after an auto-FIRE.
- `examples/vizdoom_live.py --scenario <name>` does the same for ViZDoom scenarios.

## What happens without game training

The released checkpoint was trained on photo and diagram questions, not games:

- **Breakout:** it never chose FIRE, so the ball never launched. The viewer now presses FIRE automatically at the start of each game and after each lost life, which is the standard Atari `FireResetEnv` trick. `--no-auto-fire` turns that off.
- **ViZDoom `basic`:** it chose ATTACK about 90% of the time wherever the monster was, and never strafed to line up the shot. It matches "monster visible → shoot" but doesn't relate the monster's position to the crosshair.

## Where training data can come from

### Best: generate it yourself with an expert

Run an expert in the emulator and save, for every frame, the full-colour screen and the expert's action. This beats public datasets because:

- The inputs match what the model sees live: same emulator settings, colours and resolution. Most public Atari data is 84×84 grayscale.
- An expert *policy* gives a probability over actions. Laya's loss (soft cross-entropy plus a proper scoring rule) accepts soft targets, so the model learns "RIGHT 70%, ATTACK 25%" and keeps meaningful confidence.
- You can make as much as you want.

Kinds of expert:

| Expert | Where to get one | Games |
|---|---|---|
| Scripted from game internals | ViZDoom's labels buffer gives every object's screen bounding box | Doom scenarios with a simple rule, such as `basic` (done: see below) |
| Pretrained RL agents | Hugging Face Hub: Stable-Baselines3 (`sb3/ppo-*`, `sb3/dqn-*`), CleanRL models, Sample Factory ViZDoom agents | Most Atari games; several Doom scenarios |
| RAM-based rules | Atari RAM holds object positions, e.g. the ball and paddle x in Breakout | Breakout, Pong, Freeway |

### Public datasets

| Dataset | Contents | Fit |
|---|---|---|
| Atari-HEAD (Zhang et al., 2019) | About 117 h of expert human play over about 20 Atari games: frames, actions, eye gaze | Good: human play at full resolution |
| JAT dataset (`jat-project/jat-dataset` on HF) | Expert trajectories for the 57 Atari games and more | Easy to load; frames are low-res grayscale |
| DQN Replay Dataset (Agarwal et al., 2020) | Millions of frames per game from DQN training, 60 games | Large; 84×84 grayscale; mixed-quality play; better for offline RL |
| CS:GO behavioural cloning (Pearce & Zhu, 2021) | Millions of Counter-Strike frames with keyboard and mouse actions | 3D FPS like Doom; needs actions mapped to a choice list |
| MineRL / OpenAI VPT contractor data | Minecraft video with keyboard and mouse | Large; long-horizon tasks |
| VideoGameBunny instruction data, GlitchBench | Question-answer pairs about game screenshots | Teaches reading game screens rather than acting; check contents and licences before use |

Check each licence; several are research-only or non-commercial.

## Practical notes

- **Motion needs two frames.** One frame can't show which way a ball, car or ghost is moving. `laya.vlm` accepts `{"images": [previous, current]}`, so training and play can both pass the last two frames.
- **Cover off-expert states.** If the data only comes from the expert, the model never sees the situations its own mistakes lead to. Mix in random actions when collecting (epsilon-expert, as below), or use DAgger: play the trained model, label what it saw with the expert, and retrain.
- **Soft labels over hard ones** whenever the expert has probabilities.
- **Watch forgetting.** Training only on one game may erode the photo-question skills. Mix in some VQA data, or score the VQA val splits after training (`modal run modal_app.py::evaluate`).
- **Calibration temperatures are per question type.** `finetune_long --init-from` keeps the starting checkpoint's temperature for any type missing from the new data. Game data is all `choice`, so the yes/no temperature is kept.

## ViZDoom `basic` with auto-labels (implemented)

`basic` is one room with one monster somewhere along the far wall. The player can only strafe left, strafe right or shoot. Reward is +106 for the kill, −5 per missed shot and −1 per tic, and the episode ends at 300 tics.

- **Expert** (`laya.games.doom_basic_expert`): if the monster's bounding box covers the crosshair column, ATTACK; otherwise strafe toward it. Over 100 episodes:
  - expert: mean reward +80.3, kills 100%
  - random: −134.6, kills 62%
  - always ATTACK: −218.2, kills 42%
- **Data** (`modal run modal_app.py::prepare_doom_basic`): 20k train and 2k val frames with disjoint seeds. Frames come from an epsilon-expert (30% random buttons), and every frame with a visible monster is labelled with the expert's button. It is written to `laya-datasets:/data/vqa/doom_basic/` in the same JSONL format as the VQA sets.
- **Training:** `modal run --detach modal_app.py::finetune_long --datasets doom_basic --init-from all3-3ep/best --run-name doom-basic --epochs 2`
- **Evaluation by playing** (`modal run modal_app.py::doom_eval --models all3-3ep/best,doom-basic/best`): 50 episodes per policy on unseen seeds. It reports mean reward, kill rate and the action mix for the expert, random, always-ATTACK, the zero-shot model and the trained model.
- **Watch it:** download `/ckpt/smolvlm/doom-basic/best` and run `python examples/vizdoom_live.py --model <that dir>`.

### Results (2026-09-19)

Data: 20,000 train frames (7,428 MOVE_LEFT, 7,124 MOVE_RIGHT, 5,448 ATTACK) and 2,000 val frames. Training: 2 passes from `all3-3ep/best` on one A100, 7.3 minutes including evaluation. The best checkpoint was step 924.

**Frame accuracy against the expert's labels (val):** 95.9% after 0.5 pass, 98.8% after 1, 99.5% after 1.5 and 2. ECE was 0.005 raw and 0.011 calibrated.

**Playing** (50 episodes on unseen seeds, `doom_eval` / `play_doom`):

| Policy | Mean reward | Kill rate | Steps per episode |
|---|---|---|---|
| Scripted expert | +75.8 | 100% | 6.8 |
| **Trained model (`doom-basic/best`)** | **+75.4** | **100%** | 6.8 |
| Random buttons | −121.9 | 68% | 39.4 |
| Always ATTACK | −325.6 | 18% | 63.0 |
| Zero-shot model (`all3-3ep/best`) | −325.6 | 18% | 63.0 (chose ATTACK on all 3,151 steps) |

The trained model plays at expert level.

**Cost: some forgetting of the photo tasks** (full VQA val splits):

| | Before | After Doom training |
|---|---|---|
| A-OKVQA acc | 61.8% | 57.9% |
| ScienceQA acc | 86.6% | 83.0% |
| VQAv2 yes/no acc | 73.4% | 71.6% |
| VQAv2 yes/no ECE (calibrated) | 0.041 | 0.105 |

The refit `choice` temperature (8.30, up from 3.33) is shared by all `choice` questions, so it now also flattens the photo multiple-choice answers. The yes/no temperature was kept at 1.69, but the model underneath changed, so yes/no calibration got worse. Mixing VQA data into the game training, or fitting temperatures per task, should fix both.

## Atari, game-only model (implemented, 2026-09-19)

A model trained on Atari only: plain SmolVLM-256M with a fresh head, and none of the photo datasets. The data format is in [atari-data-format.md](atari-data-format.md). All sources are on `laya-datasets:/data/atari/<source>/<Game>/`.

| Source | Games | Frames per game | Format | Labels | Code |
|---|---|---|---|---|---|
| `expert`: CleanRL PPO agents (best of 9 checkpoints per game) | 57 | 20k train / 1k val | RGB 210×160 | agent's action probabilities (soft) | `laya/atari_data/expert.py`, `modal_atari_expert.py` |
| `jat`: `jat-project/jat-dataset` (Apache-2.0) | 57 | 20k / 1k | gray 84×84, newest frame of each stack | agent actions (hard) | `laya/atari_data/jat.py`, `modal_atari_jat.py` |
| `atari_head`: Atari-HEAD v4, Zenodo 3451402 (CC-BY-4.0) | 20 | 20k / 1k | RGB 210×160 | human actions (hard) | `laya/atari_data/atari_head.py`, `modal_atari_head.py` |

Checks behind the data:
- **Action mapping:** JAT and the expert agents use ALE's minimal action set; Atari-HEAD uses the full 18-action enum. Each was verified against the data, not just the docs.
- **JAT KungFuMaster and MontezumaRevenge** store frame stacks in a different byte layout. The converter decodes them and checks every game's frame order before writing.
- **Expert baselines** are in each expert `meta.json`, measured with sticky actions: uncapped, and capped at 4,500 decisions (`*_cap4500`, which evaluation uses). Solaris's expert scores below random, so Solaris is excluded from summaries.

**Training** (`modal_atari_train.py::train_atari`; vision tower frozen, full LM and head; up to 4k frames per source per game; best checkpoint by val NLL):

| Run | Data | Steps | Val frame accuracy | Calibrated ECE |
|---|---|---|---|---|
| `atari-all-v1` | 122 source×game sets, 465k frames, 0.9 pass, 70 min A100 | 13,272 | 9.6% → 33.5% | 0.056 |
| `atari-expert-v1` | expert only, 45 games, 0.8 pass, 25 min | 4,315 | 9.2% → 30.4% | 0.087 |

**Playing** (`atari_eval`: ALE v5 defaults, auto-FIRE, 3 episodes × 4,500 decisions per game). Normalised score = (model − random) / (expert − random), with capped baselines:

| Model, action choice | Median normalised (unflagged games) | Beats random |
|---|---|---|
| `atari-expert-v1`, top action | 0.002 | 23/45 |
| `atari-expert-v1`, sampled | 0.010 | 31/45 |
| `atari-all-v1`, top action | −0.003 | 21/57 |
| `atari-all-v1`, sampled | 0.002 | 32/57 |

**Result: roughly random-level play.** A few games show real skill:
- Freeway 0.63 (always go UP)
- Centipede 0.20–0.34
- Robotank 0.23
- Krull 0.19–0.23

Adding JAT and Atari-HEAD didn't help: expert-only was at least as good on play and on expert-frame NLL. Grayscale 84×84 JAT frames don't match the RGB frames seen during play.

Flagged games, whose normalised scores aren't skill:
- **Skiing:** the expert just presses NOOP.
- **DoubleDunk, Tennis:** stalling until the cap scores well.
- **Pitfall, PrivateEye, MontezumaRevenge:** expert ≈ random.
- **Tutankham:** the greedy expert gets stuck.

With top-action play the model presses nearly one button, so the 3 episodes coincide even with different seeds.

Why it falls short, and what to try:
- **Single frame, no motion:** use two-frame input. *(tested, next section: median 0.201 vs 0.131)*
- **Under one pass at 4k frames per game:** train on expert-only data with all 20k frames and several passes.
  *(tested, next section)*
- **Compounding errors in pure imitation:** use DAgger. *(tested, "DAgger round 1" below: median 0.310)*
- **Frozen photo-trained vision tower on pixel art:** unfreeze the top vision layers.
- **57 very different games in one small model:** try specialists or small game groups.

## Two frames and more data per game, 8 games (implemented, 2026-09-19)

The two failures above (one frame, and under one pass over 4k frames per game) were tested directly on
**Breakout, Pong, Freeway, SpaceInvaders, Enduro, Boxing, Qbert and MsPacman**, with `/data/atari/expert2f/`
(every record carries `prev_image`, the frame from the previous decision step). Both runs start from
`atari-expert-v1/best`, use all 20k frames per game, the vision tower frozen, and differ only in `--frames`:

    modal run --detach modal_atari_train.py::train_atari --run-name atari-8g-2f --frames 2 --sources expert2f \
        --games Breakout,Pong,Freeway,SpaceInvaders,Enduro,Boxing,Qbert,MsPacman --passes 2 --max-minutes 55 \
        --init-from atari-expert-v1/best
    modal run modal_atari_train.py::atari_eval --model atari-8g-2f/best --episodes 10 \
        --games Breakout,Pong,Freeway,SpaceInvaders,Enduro,Boxing,Qbert,MsPacman

Two frames cost about 30% throughput (2.78 vs 3.86 steps/s at batch 32), so in 55 minutes the two-frame run
reached 1.84 passes against the one-frame run's 2.00.

**Val frame accuracy** (mean over games, calibrated): 0.31 at the start, 0.596 for two frames and 0.586 for one.
At matched passes two frames is ahead throughout, e.g. 0.575 / NLL 1.151 at 1.12 passes versus 0.554 / 1.187 at
1.20. Mean per-game val NLL: 1.098 (two frames) and 1.124 (one), both raw.

**Playing** (10 episodes per game, 4,500-decision cap, normalised with the `*_cap4500` baselines, top action):

| Game | expert | random | 2 frames | 1 frame | `atari-expert-v1` |
|---|---|---|---|---|---|
| Boxing | 92.4 | 1.6 | **0.57** | 0.37 | -0.05 |
| Freeway | 34.0 | 0.0 | **0.75** | 0.67 | 0.00 |
| Pong | 5.4 | -20.2 | **0.35** | 0.29 | -0.02 |
| Qbert | 24985 | 185 | **0.31** | 0.19 | -0.00 |
| MsPacman | 6092 | 434 | **0.10** | 0.07 | 0.00 |
| Breakout | 218.0 | 1.2 | 0.03 | 0.05 | 0.00 |
| SpaceInvaders | 8251 | 125 | 0.03 | 0.02 | 0.02 |
| Enduro | 379.6 | 0.0 | 0.02 | 0.04 | 0.00 |
| **median** | | | **0.201** | 0.131 | 0.000 |
| beats random | | | 8/8 | 8/8 | 3/8 |

Both new models beat random on every game, where the 57-game model beat it on 3 of 8. Two frames wins on 5 of 8
games and on the median (0.201 vs 0.131); one frame is marginally better on Breakout and Enduro. Sampling now
*hurts* (medians 0.091 and 0.089): once the policy is good, its own top action beats sampling from it, the
opposite of the 57-game model.

Fitted temperatures (about 1.22) made ECE worse, not better (0.094 raw to 0.118 calibrated for two frames): the
calibration holdout comes from held-out episodes of the same games, and the model is already slightly
underconfident there.

**Calibration note.** Fitting temperatures on held-out *episodes of the games being trained on* over-flattens.
Frames from the same games are near-duplicates of training frames even across episodes, so the holdout looks
easier than it is, the fit pushes the temperature above 1 and raw ECE gets worse. Either fit on games the model
was not trained on, or skip temperature fitting when raw ECE is already low (below about 0.05) and keep the
starting checkpoint's temperature. A temperature far from 1.0 on this data is a sign of the holdout, not of the
model: the 3.33 -> 8.30 `choice` temperature from the Doom run also came from fitting on one task's own data, and
it flattened the photo questions along with it.

## DAgger round 1, two frames, 8 games (implemented, 2026-09-19)

The remaining failure from the list above — **compounding errors in pure imitation** — was tested with one
DAgger round. `/data/atari/dagger1/` holds about 4.5k frames per game from **the model's own rollouts**,
labelled with the expert's action probabilities, so training now covers the states the policy actually reaches
rather than only the states the expert reaches. Both runs start from `atari-8g-2f/best` (median 0.201), mix
`expert2f` with `dagger1` 50/50 by samples (`--balance game_source`), and use the original RLCD objective:

    modal run --detach modal_atari_train.py::train_atari --run-name atari-dag2f-rlcd \
        --sources expert2f,dagger1 --frames 2 --init-from atari-8g-2f/best \
        --games Breakout,Pong,Freeway,SpaceInvaders,Enduro,Boxing,Qbert,MsPacman \
        --passes 1.5 --max-minutes 50 --n-evals 4 --n-calib 200 --max-passes 5 --select-by play \
        --group-size 8 --sigma 1.0 --sigma-end 0.3 --w-sph 0.5 --w-ce-schedule anneal --train-act \
        --balance game_source --td-lambda 0.0      # and a second run with --td-lambda 1.0

`--frames` now **defaults to 2**: two frames have beaten one on every comparison, so a single-frame run has to be
asked for explicitly. `--select-by play` keeps the checkpoint with the best in-run play check, which picked
step 4375 of 5560 for the first run and step 4475 of 5692 for the second — in both cases *not* the last step, and
not the best-NLL step either (5560 / 5692). Frame metrics still do not predict play strength.

**Playing** (10 episodes per game, 4,500-decision cap, top action, normalised with the `*_cap4500` baselines;
no episode hit the cap):

| Game | `dag2f-rlcd` (td 0) | `dag2f-rlcd-td` (td 1) | `atari-8g-2f` | `atari-8g-1f` | gated (td 0) |
|---|---|---|---|---|---|
| Boxing | **0.84** | 0.60 | 0.57 | 0.37 | 0.58 |
| Freeway | **0.79** | 0.74 | 0.75 | 0.67 | 0.79 |
| Qbert | **0.32** | 0.23 | 0.31 | 0.19 | 0.32 |
| MsPacman | **0.31** | 0.13 | 0.10 | 0.07 | −0.02 |
| Pong | 0.31 | **0.56** | 0.35 | 0.29 | 0.25 |
| Breakout | **0.09** | 0.06 | 0.03 | 0.05 | 0.03 |
| Enduro | **0.08** | 0.02 | 0.02 | 0.04 | 0.00 |
| SpaceInvaders | 0.03 | 0.02 | 0.03 | 0.02 | 0.01 |
| **median** | **0.310** | 0.182 | 0.201 | 0.131 | 0.140 |
| beats random | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 |

Sampling from the probabilities instead of taking the top action still hurts, as it did for the 8-game models:
median 0.155 against 0.310 for `dag2f-rlcd`, worse on 7 of 8 games (only Enduro improves, 0.08 to 0.10).

**One DAgger round lifts the median from 0.201 to 0.310**, better on 7 of 8 games (SpaceInvaders is level), with
the biggest gains where the policy's own mistakes matter most: Boxing 0.57 → 0.84, MsPacman 0.10 → 0.31,
Enduro 0.02 → 0.08. `atari-expert-v1`, for reference, is 0.000.

**Outcome-blended targets (`--td-lambda 1.0`) are worse overall**: median 0.182, below even the starting
checkpoint. They win big on exactly one game, Pong (0.56 against 0.31), and lose on the other seven. The in-run
play check preferred them early (0.127 / 0.130 against 0.073 / 0.110 at the first two evals) and then fell behind
(0.157 against 0.241) — a reminder that the 3-game, 3-episode, 400-step play check is a noisy selector. Blending
each target toward the action actually taken also costs frame agreement everywhere (mean val accuracy 0.611
against 0.640).

**Agreement with the expert on the model's own states** (the point of DAgger), from each run's final val pass:

| | on its own states (`dagger1` val) | on expert states (`expert2f` val) |
|---|---|---|
| `dag2f-rlcd` | 0.679 | 0.621 |
| `dag2f-rlcd-td` | 0.631 | 0.601 |

Agreement is now *higher* on the policy's own states than on the expert's, i.e. the off-expert states the first
round exposed are no longer the hard ones. Per game it ranges from 0.88 (Freeway) to 0.46 (SpaceInvaders), and
SpaceInvaders is also where play is weakest.

**The escalate-head gate does not help** (`--gate`, which repeats the previous action whenever the act head says
escalate rather than act). Median 0.140 against 0.310 ungated for `td 0`, and 0.003 against 0.182 for `td 1`.
The head escalates on 76-99% of steps in most games (mean P(act) 0.13-0.25), because `train_act`'s cost matrix
only pays for acting when P(correct) > 0.625 and this policy is rarely that sure. Holding the previous action that
often is close to a constant policy: `td 1` gated drops to 5 of 8 games above random and goes negative on Boxing.
Freeway is the exception — P(act) is 1.000 there, nothing is gated, and the score is unchanged. The act head is
a useful confidence signal, but repeating the last action is the wrong fallback for a game policy; a gate would
need an action to escalate *to*.

**Fitted temperatures are still above 1** — 1.268 (`td 0`) and 1.277 (`td 1`) for `choice`, from 1.218 in the
starting checkpoint — and calibrating still makes ECE worse, not better: 0.094 raw to 0.136 calibrated, and 0.088
to 0.118. Annealing the cross-entropy weight to 0 did **not** bring the temperature to 1.0. The cause is the
holdout, not the objective (see the calibration note above): the 200 calibration frames per (source, game) come
from held-out episodes of the games being trained on, where the model is already slightly underconfident. Raw ECE
is low enough (0.09) that the fit should simply be skipped for this data.

Caveats:
- **`dagger1` was collected by rolling out the one-frame model**, so these are approximately, not exactly, the
  two-frame model's own states. The round still helped substantially; a second round should roll out the
  two-frame policy it is now training.
- Both runs spent their 50-minute budget on 0.96-0.98 of the planned 1.5 passes (34.4 min training, 15.6 min in
  the four evals, at 2.69 steps/s with 0.8% data wait). More budget, not more data, is the next limit.
- One seed each, 10 episodes per game: game-level differences of a few hundredths are noise, the median gap of
  0.11 is not.

The DAgger runs above use the **Hugging Face processor path at 512**, inherited from `atari-8g-2f`. The
device-side path in the next section is an independent change on the same baseline (0.255 against 0.201), so the
two gains have not been combined: a DAgger round trained and played on the device-side path is untested, and the
next section's warning applies -- do not simply replay these checkpoints on the other path.

## Cheap preprocessing: the resize on the GPU, and a lower input resolution (implemented, 2026-09-19)

Turning one 210x160 Atari frame into the model's input cost more CPU than the forward pass cost GPU. The Hugging
Face `Idefics3ImageProcessor` resizes with LANCZOS **twice** -- `210x160 -> 2048x1560` (its `size.longest_edge`)
`-> 512x512` (its `max_image_size`) -- so it upscales the frame to 3.2 megapixels only to throw that away again,
in Python, per frame. Measured: **14.6 ms/frame** on an M-series CPU, **33.6 ms/frame** in a Modal container.

Both hops are linear and separable, so `laya/preprocess.py` composes the whole chain into one pair of small
matrices (`Wh [out, H]`, `Ww [out, W]`) and runs the resize as two matmuls on the model's own device:

    out[c, i, j] = sum_p sum_q  Wh[i, p] * Ww[j, q] * in[c, p, q]

`ImagePrep` holds the three values that decide the path -- `image_size`, `preprocess` (`gpu` or `processor`) and
`image_interpolation` -- and they are written to `vlm_agent_config.json`, so a checkpoint records what it was
trained with and play matches it. A config saved before these keys existed means 512 through the processor, which
is what those checkpoints were trained with, so they are unaffected.

`image_size` also sets what an image costs the language model: `(image_size / patch_size)^2 / scale_factor^2`,
which is **64 tokens at 512 and 16 at 256** for SmolVLM-256M. Nothing downstream assumes 64.

### How close is the cheaper filter?

Against the processor's own output on 50 real `expert2f/Breakout` frames, in 0-255 grey levels:

| filter | 512 max | 512 mean | 256 max | 256 mean |
|---|---|---|---|---|
| composed two-hop (`processor`, the default) | 22 | **0.049** | 19 | **0.134** |
| single-hop `lanczos` | 22 | 0.049 | 21 | 0.295 |
| `bicubic` (torchvision) | 24 | 0.407 | 27 | 0.610 |
| `bilinear` (torchvision) | 55 | 0.793 | 55 | 0.766 |

The composed operator is not bit-exact with the processor, and cannot be: the processor rounds *and clamps* its
2048-pixel intermediate back to uint8, and clamping is not linear, so LANCZOS overshoot at a hard edge survives
here where the processor cut it off. That is the whole story of the `max` column -- p99.9 is 2 grey levels at 512.

Feeding `atari-8g-2f`'s own weights through both paths at 512, its action probabilities move by **0.006 mean /
0.027 max** and the top action changes on 3 of 50 frames. `bicubic` moves them 5x further (top action changes on
15 of 50), which is why the composed LANCZOS operator is the default rather than a torchvision resize. Note
torchvision has no CUDA LANCZOS kernel, so a plain `tvF.resize` on a GPU silently drops to BICUBIC.

### Benchmark (L4, Breakout, batch 32, `play` with 4 episodes in lockstep)

| config | frames | img tokens | prep CPU ms/frame | prep GPU ms/frame | decisions/s | decisions/s cached | train steps/s |
|---|---|---|---|---|---|---|---|
| 512 processor | 1 | 64 | 33.57 | - | 17.3 | - | 0.879 |
| 512 processor | 2 | 128 | 32.88 | - | 11.2 | 8.7 | 0.595 |
| 512 gpu | 1 | 64 | 0.14 | 0.224 | 49.1 | - | 1.478 |
| 512 gpu | 2 | 128 | 0.10 | 0.243 | 38.4 | **53.1** | 0.891 |
| 256 gpu | 1 | 16 | 0.16 | 0.088 | 53.5 | - | 2.683 |
| 256 gpu | 2 | 32 | 0.11 | 0.088 | 51.0 | **54.3** | 2.112 |

* The CPU cost of one decision's image inputs falls **~240x** (33.6 -> 0.14 ms/frame); the resize reappears on the
  GPU at 0.22 ms/frame (512) or 0.088 ms/frame (256).
* **Play: 2.8-3.4x** more decisions/s from the path alone, **4.7x** two-frame with the feature cache.
* **Training: no win from the path on an A100**, and the L4 column above overstates it. Those training rows are
  40-step runs, so worker spin-up is most of their "loader wait"; the trustworthy numbers are the full 8-game
  two-frame runs on an A100 with 22 loader workers, where **512 processor and 512 gpu both do 2.78 steps/s** and
  only 256 pulls ahead, at **4.79**. With 22 workers the processor's 33 ms was already hidden behind the GPU, so
  training at 512 is GPU-bound either way and the 1.72x at 256 comes from **4x fewer image tokens**, not from
  cheaper preprocessing. The path matters for training only when the loader is actually the bottleneck (few CPUs,
  or a cheaper forward pass).
* Play is where the path pays, because a rollout loop is single-process: there are no loader workers to hide the
  33 ms behind, so it lands directly on the critical path.
* Lowering the resolution barely helps the *processor* path (11.7 ms at 256 vs 14.6 at 512): the dominant hop is
  the upscale to 2048, which does not depend on the target size. The win needs the device-side path.

### Caching the previous frame's encoder output

In two-frame play the model sees `[previous, current]` every step, so this step's current frame is next step's
previous frame and half the vision-tower work is a repeat. `FrameFeatureCache` keys on a 64-bit hash of the raw
bytes, confirmed by an exact `np.array_equal` before a hit counts, so a collision costs a re-encode and never a
wrong answer. It also catches repeats *inside* a batch (an episode's first step, and the step after an auto-FIRE,
pass the same frame twice). Hit rate in play is ~0.47.

It is not bit-exact and cannot be: a hit was computed in whatever batch its miss belonged to, and the vision
tower's reductions are not associative. In fp32 the difference is **1.7e-6**; in bf16 it is ~0.03 on a probability
but the top action is unchanged, and batch-of-1 against batch-of-2 *without* any cache already differs by the same
amount.

It is **on by default only on the device-side path**. On the processor path it is a 23% *loss* (11.2 -> 8.7
decisions/s): caching means preprocessing frames one at a time, and the processor costs ~33 ms of CPU per frame
regardless, so the lost batching outweighs halving the encoder's work.

Reproducing the two runs:

    modal run --detach modal_atari_train.py::train_atari --run-name atari-8g-2f-512gpu --frames 2 \
        --sources expert2f --games Breakout,Pong,Freeway,SpaceInvaders,Enduro,Boxing,Qbert,MsPacman \
        --passes 2 --max-minutes 55 --init-from atari-expert-v1/best --image-size 512 --preprocess gpu
    modal run --detach modal_atari_train.py::atari_eval --model atari-8g-2f-512gpu/best --episodes 10 \
        --games Breakout,Pong,Freeway,SpaceInvaders,Enduro,Boxing,Qbert,MsPacman

### Caching the fixed question

Every step asks the same question, so could its part of the sequence be cached? Not as keys/values: it comes
after the image, and at batch 1 on a GPU the language model costs the same ~24 ms for 150 or 300 tokens anyway.
What pays is capturing the whole fixed-shape decision as one CUDA graph: `--cuda-graph` in both live viewers and
`model_policy(..., cuda_graph=True)`: ~11 ms per decision at batch 1 in bf16 on an L4, 3.4-5.0x fewer, with the same
answer. The measurements and the question-first layout that was considered and rejected are in
[game-caching.md](game-caching.md).

### Is the dataset's full-size PNG decode worth avoiding?

No. The PNGs decode in **1.14 ms/frame** once the Modal volume has served them; the 72-283 ms/frame a naive timing
shows is the volume's cold first-touch latency, not PIL. And a pre-resized 256x256 frame has 65,536 pixels against
the original's 33,600, so storing pre-resized frames would make the decode *slower* as well as throwing away the
option of training at another resolution. Training at 256 is no longer loader-bound anyway (1-2% data wait).

### Does the quality hold? The path yes, 256 no

`atari-8g-2f-256` matches `atari-8g-2f`'s recipe exactly (init `atari-expert-v1/best`, source `expert2f`, the
same 8 games, two frames, vision tower frozen, same LRs, `--passes 2 --max-minutes 55`) and differs only in
`--image-size 256 --preprocess gpu`. It was **4.79 steps/s against 2.78**, so it finished all **2.00 passes in
32 min** where the 512 run reached 1.84 in 55, with the loader wait down to 0.5%.

Val frame accuracy still came out **lower**: 0.554 against 0.596, mean per-game val NLL 1.182 against 1.098 --
despite the extra passes.

Playing (10 episodes per game, 4,500-decision cap, top action, `*_cap4500` baselines):

| Game | `atari-8g-2f` 512 processor | `atari-8g-2f-512gpu` 512 gpu | `atari-8g-2f-256` 256 gpu | 8g-2f weights *played* on the gpu path |
|---|---|---|---|---|
| Boxing | **0.57** | 0.48 | 0.03 | 0.38 |
| Freeway | **0.75** | 0.72 | 0.65 | 0.68 |
| Pong | 0.35 | **0.55** | 0.16 | 0.32 |
| Qbert | 0.31 | **0.43** | 0.03 | 0.25 |
| MsPacman | **0.10** | 0.08 | 0.09 | 0.08 |
| Breakout | **0.03** | 0.02 | 0.01 | 0.03 |
| SpaceInvaders | **0.03** | 0.02 | 0.03 | 0.02 |
| Enduro | 0.02 | **0.07** | 0.03 | 0.03 |
| **median** | 0.201 | **0.255** | 0.030 | 0.165 |
| val frame acc | 0.596 | **0.602** | 0.554 | - |
| passes in 55 min | 1.84 | 1.92 | 2.00 (in 32 min) | - |
| beats random | 8/8 | 8/8 | 8/8 | 8/8 |

**512 on the device-side path is the one to keep.** `atari-8g-2f-512gpu` is the same recipe with
`--preprocess gpu`, and it matches the original on val frame accuracy (0.602 against 0.596) and comes out *ahead*
on play (median 0.255 against 0.201), winning big on Pong (0.35 -> 0.55) and Qbert (0.31 -> 0.43) and losing some
of Boxing. Per-game differences of this size are within what 10 episodes resolve on these games, so the honest
claim is **no worse, at 2.8-4.7x the decisions/s** -- not that the filter made the model better.

**256 is not usable.** A 4-point drop in frame accuracy became a 7x drop in median normalised score, concentrated
in exactly the games that need to locate something small and precisely: Boxing 0.57 -> 0.03 (punch range), Qbert
0.31 -> 0.03 (which cube), Pong 0.35 -> 0.16 (where the ball is). Freeway, whose decision is "is the lane clear",
barely moved. **Frame accuracy is a bad proxy for play strength** -- a model can keep predicting the expert's
modal action while losing the spatial precision that makes the action pay off.

The last column is the other half of the warning: taking `atari-8g-2f`'s own weights, trained through the
processor, and merely *playing* them on the device-side path at the same 512 costs 0.201 -> 0.165, and 7 of 8
games are flat or down. The filter difference is tiny (0.049 grey levels mean, the top action changing on 3 frames
in 50) and it still shows up in play. **Train and play on the same path**; do not migrate an existing checkpoint by
flipping the flag.

## Durability of long runs

Modal preempts containers, and an A100 training run is the expensive thing to lose, so `train_atari` is
resumable and `play_atari` / `train_atari` carry `retries=modal.Retries(max_retries=3, initial_delay=10.0)`:
a preempted container restarts and continues instead of dying.

- Every `--state-every-min` minutes (default 10) and after every eval, the run writes `<run>/state.pt`: weights,
  optimizer state, step, RNG states, per-group sample counts, the best-so-far record and the log. It is written
  to `state.pt.tmp` and renamed, then the volume is committed, so a torn write never replaces a good state.
- **A state write costs GPU time, so it is rate-limited** (fixed 2026-09-19). `train_atari` gives the training
  loop `eval_every=min(25, eval_every)` so its `maybe_eval` can check progress cheaply every 25 steps, but the
  loop used to write `state.pt` after *every* `eval_fn` call — including the ~99% of those calls that decided not
  to evaluate. At 5-14 s a write every 25 steps (about 14 s of training), the first DAgger runs were spending
  roughly a third of their A100 time checkpointing, with `--state-every-min` at its 10-minute default the whole
  time. Now `eval_fn` returns whether it really evaluated and only then earns a write; `save_state_every_min` is
  clamped to `vlm_train.MIN_STATE_MINUTES` (2 min) with a warning, and the periodic write also backs off to at
  most 1/`STATE_SAVE_OVERHEAD` (5%) of the wall clock when writes are slow. Any launcher, however it is called,
  now pays at most a few percent for durability.
- On start, an existing `state.pt` is resumed and logged (`resuming ... at step N`). `--restart` ignores it, but
  only for that call's first attempt (the marker records the Modal function-call id), so a retry of a restarted
  run still resumes rather than starting over.
- `--max-minutes` counts **training time across attempts**: the resumed loop backdates its clock by the elapsed
  time in `state.pt`, so a preempted run cannot spend twice its budget. `max_passes` likewise counts the samples
  already drawn per group. The sampler is an endless random stream, so a resumed run re-seeds it (seed + step)
  instead of replaying the same order.
- `--crash-at-step N` raises once at step N to exercise the path. Verified: the run trained to step 22, wrote
  `state.pt`, crashed at step 44, and the retried container resumed at step 22 with 0.7 minutes already counted
  and finished normally.
- `atari_eval --out results.json` writes each game's result as it lands and skips games already in that file
  (matching model, episodes, cap, sampling, frames and gate), so an interrupted evaluation only redoes what is
  missing; `--restart` ignores the file. `write_synthetic` skips a game that already has `_READY` unless
  `force=True`; the real data jobs live in the `modal_atari_{expert,head,jat}.py` apps.

## Next ideas

1. **DAgger round 2**, rolling out the *two-frame* policy this time (round 1's states came from the one-frame
   model) and spending more than 1.5 passes of budget, which is what limited both round-1 runs.
2. **Skip the temperature fit** on same-game holdouts, or fit on held-out *games*; raw ECE is already ~0.09 and
   the fit makes it worse.
3. **A gate with somewhere to escalate to.** The act head is well-behaved as a confidence signal but repeating
   the previous action is the wrong fallback; give it a scripted or slower policy to defer to, or drop the gate.
4. **Unfreeze the top vision layers** — the last untested item on the failure list, and pixel art is far from the
   photos the tower was trained on.
5. **RL fine-tuning** from game reward, using the same proper-scoring policy-gradient term the loss already has.
   `--td-lambda` is a first step in that direction and only helped Pong, so credit assignment needs more than a
   return-to-go percentile.
6. **Mixed multi-game training** with VQA data mixed in, then checking both play strength and VQA accuracy.
7. **DAgger rounds on Doom** `defend_the_center` (turn plus shoot), with the labels buffer as the expert.
