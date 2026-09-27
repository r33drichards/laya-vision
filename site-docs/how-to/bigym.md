# Evaluate on BiGym

[BiGym](https://github.com/chernyadev/bigym) is a MuJoCo benchmark for a mobile two-armed robot (a Unitree H1)
in a kitchen. Its policies output a 15-dimensional continuous joint action at 50 Hz, while laya answers discrete
questions about an image. [`laya/bigymgames.py`](https://github.com/r33drichards/laya-vision/blob/main/laya/bigymgames.py)
bridges the two in two ways, and
[`modal_bigym.py`](https://github.com/r33drichards/laya-vision/blob/main/modal_bigym.py) runs both on Modal:

- **Perception probe.** Frames from the robot's head camera go to the model with three questions. The simulator
  knows the correct answer to each:
    - `done` (`noul`): is the task already complete?
    - `progress` (`score`, 4 levels): how far along is the task? The level comes from how far the drawer or doors
      are open, or from the distance between the wrist and the target.
    - `side` (`choice`, reach tasks only): which hand is closer to the target?

  Each question is scored for accuracy, ECE and NLL, next to a prior-only baseline. That baseline always predicts
  the label frequencies of the same rows, so its accuracy is the share of the most common label.
- **Zero-shot control.** The model picks one of 37 named motion primitives each decision:
    - move one wrist 3 cm forward, back, left, right, up or down, in the robot's frame
    - roll one wrist 0.25 rad (about 14°) clockwise or counterclockwise, as the head camera sees it
    - tilt one gripper 15° up, down, left or right. H1's arms have no wrist pitch or yaw joint, so a tilt moves the
      shoulder and elbow and shifts the hand a few centimetres
    - open or close a gripper
    - step forward or back, sidestep left or right, turn, crouch or stand up (the base moves in X, Y, height and
      yaw, the four DOFs BiGym's demos were recorded with)
    - stay

  It sees the current head frame, or with `--frames 2` / `--frames 4` its last 2 or 4 decision frames (0.1 s
  apart, oldest first, as `{"images": [...]}`), so it can see motion. The options are short phrases ("left hand
  up", "close right gripper", "wait"), because 37 options share a 256-token budget with the instruction (253 of 256 tokens with 4 frames).

  Damped least-squares inverse kinematics on the wrist site turns a wrist move into joint deltas. Each primitive is
  applied once and then held for 0.1 s (5 env steps). Random play uses the same seeded episodes and the same
  primitives. So does a privileged oracle on the reach tasks: it greedily moves the nearer wrist toward the
  target, and solves every episode. The cupboard tasks have no primitive oracle. Their expert reference is
  BiGym's human demonstrations replayed in the simulator.

Tasks: `ReachTarget`, `ReachTargetSingle`, `DrawerTopOpen`, `DrawerTopClose`, `WallCupboardOpen`,
`WallCupboardClose`. The wall cabinet sits above BiGym's default head view, so on the two wall tasks the head
camera is tilted up by 30°.

## Run it

```bash
modal run modal_bigym.py::bigym_eval                   # all six tasks, both parts, thaitea/laya-vision (pinned)
modal run modal_bigym.py::bigym_eval --tasks ReachTarget,DrawerTopClose --parts control --episodes 3
modal run modal_bigym.py::bigym_eval --parts control --frames 1,2,4   # the model once per frame count
modal run modal_bigym.py::bigym_eval --model thaitea/laya-vision --revision <sha> --probe-n 200 --episodes 20
```

It prints a probe table and a control table, and writes one JSON to `eval-results/` (`--out` names it). The JSON
holds every probe row (the label, the model's probabilities and the simulator's state), every episode, and the
versions of MuJoCo and BiGym. The probe frames are regenerated from their seeds rather than stored. Results are
create-only: an existing `--out` is refused.

Locally, with `pip install mujoco git+https://github.com/chernyadev/bigym` and `MUJOCO_GL=egl` (or `osmesa`):

```python
from laya import bigymgames as bg
bg.play_episodes("ReachTarget", bg.oracle_policy, episodes=3)   # success_rate 1.0
bg.demo_reference("DrawerTopClose", amount=5)                   # downloads the demos (~120 MB) on first use
```

`python -m pytest tests/test_bigymgames.py` runs the checks that need no simulator anywhere, and the simulator
checks when `mujoco` and `bigym` are installed.

## Reading the numbers

- **Control is zero-shot.** The checkpoint has never seen a robot. Its game training is Atari, ViZDoom, grid
  games and classic control, so expect success near random. The oracle and the demos show that each task can be
  solved within the step cap (60 decisions for reach, 150 for the cupboard tasks).
- **The probe comes first.** A task-state reader has to work before fine-tuning on BiGym, or using laya as a
  success detector, is worth trying. Compare accuracy with `prior`, not with 50%. The labels are unbalanced
  (reach frames are mostly not done), so a model that always says "no" can look accurate.
- **The demos are replayed at 20 Hz under MuJoCo 3.14.** They were recorded under 3.1.5, and BiGym warns that
  replay can drift. A few percent of them fail for that reason, not because the task is hard.

## Results: thaitea/laya-vision

These results are for revision `f2fe3c1` (MuJoCo 3.14.0, L4 bf16). They were run with 21 primitives and a 3-DOF
base, before the wrist rolls, tilts, sidesteps and crouching were added.

- **Control** comes from a run with code `51afb36`, with every episode in
  [`eval-results/bigym-laya-vision-f2fe3c1-frames.json`](https://github.com/r33drichards/laya-vision/blob/main/eval-results/bigym-laya-vision-f2fe3c1-frames.json).
- **Probe** comes from a run with code `376e1c6`, with every probe row in
  [`eval-results/bigym-laya-vision-f2fe3c1.json`](https://github.com/r33drichards/laya-vision/blob/main/eval-results/bigym-laya-vision-f2fe3c1.json).
  AUROC and Spearman below are recomputed from those rows by `laya.bigymgames.probe_metrics`.
- **That first run's control numbers are superseded.** Its prompt listed the options as "NAME: description",
  which overflowed the 256-token head budget. The options were cut to about 12 tokens, most of the instruction
  (the task) was dropped, and "open left gripper" and "close left gripper" became identical. The probe questions
  were not truncated, so the probe results stand. `predict` now runs with `strict=True`, and a test checks that
  every question fits.

**Control** (success over 20 seeded episodes per policy; the model at 1, 2 and 4 frames; oracle and demos are
the references):

| Task | Model, 1 frame | 2 frames | 4 frames | Random | Oracle | Demos (20) |
|---|---:|---:|---:|---:|---:|---:|
| ReachTarget | 0% | 0% | 0% | 35% | 100% | – |
| ReachTargetSingle | 0% | 0% | 0% | 25% | 100% | – |
| DrawerTopOpen | 0% | 0% | 0% | 0% | – | 100% |
| DrawerTopClose | 100%\* | 100%\* | 100%\* | 0% | – | 100% |
| WallCupboardOpen | 0% | 0% | 0% | 0% | – | 90% |
| WallCupboardClose | 0% | 0% | 0% | 0% | – | 100% |

\* **DrawerTopClose is not a skill.** The model chooses "step forward" on 98–100% of decisions, and the robot's
body pushes the open drawer shut. A policy that always steps forward also succeeds on every episode, in the same
64 decisions.

- **More frames do not help.** Success is the same at 1, 2 and 4 frames on every task. The choices shift a little
  but stay collapsed.
- **The model mostly repeats one move per task.**
    - Reach tasks: gripper open or close, and "wait".
    - DrawerTopOpen and DrawerTopClose: "step forward".
    - Wall cabinet: one gripper command, up to 100% of decisions.
- **It never beats random on the reach tasks**, where random play reaches the target in 25–35% of episodes.

**Probe** (200 frames per task; `prior` is the accuracy of always giving the most common label):

| Task | `done` acc / prior | `done` AUROC | `progress` acc / prior | `progress` Spearman | `side` acc / prior |
|---|---:|---:|---:|---:|---:|
| ReachTarget | 77% / 84% | 0.76 | 18% / 41% | −0.03 | 54% / 70% |
| ReachTargetSingle | 48% / 87% | 0.51 | 22% / 60% | 0.00 | 57% / 57% |
| DrawerTopOpen | 75% / 75% | 0.69 | 26% / 25% | 0.11 | – |
| DrawerTopClose | 68% / 75% | 0.38 | 22% / 25% | −0.17 | – |
| WallCupboardOpen | 65% / 75% | 0.61 | 22% / 25% | 0.30 | – |
| WallCupboardClose | 54% / 75% | 0.23 | 28% / 25% | −0.44 | – |

- **No question beats the prior on accuracy by more than 3 points.** `progress` answers pile up on level 2,
  "mostly done", whatever the true level: 86–93% of frames on the cupboard tasks, 55–62% on the reach tasks.
- **The ranking carries some signal, with the wrong sign on the close tasks.**
    - On the open tasks, P(done) ranks the more-open frames higher (AUROC 0.69 and 0.61).
    - On the close tasks it ranks them the same way, which is backwards for "closed", so AUROC is below 0.5 and
      Spearman is negative.
    - Reading: the model registers how open the drawer or doors are, but not which direction the task asks for.
      It says "done" for open.
- **ReachTarget `done`** has the best ranking (0.76). On ReachTargetSingle, where only the left hand counts, it
  is at chance.

**What this means.** Zero-shot, this checkpoint is not a BiGym policy or a success detector. The positive AUROC
on the open tasks, together with the flipped sign on the close tasks, suggests the visual features carry
cabinet-state information that the question wording does not reach. That argues for training on BiGym frames
(the probe's labelled frames, or the demos) before trying control again.
