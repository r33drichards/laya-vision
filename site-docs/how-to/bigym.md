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
- **Zero-shot control.** The model picks one of 21 named motion primitives each decision:
    - move one wrist 3 cm forward, back, left, right, up or down, in the robot's frame
    - open or close a gripper
    - step or turn the base
    - stay

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
