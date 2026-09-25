---
license: cc-by-nc-sa-4.0
base_model: thaitea/laya-vision
library_name: laya
pipeline_tag: visual-question-answering
datasets:
- HuggingFaceM4/the_cauldron
- MMInstruction/VLFeedback
language:
- en
tags:
- laya
- calibration
- decision-model
- vision
- smolvlm
- robotics
- imitation-learning
- games
---

# Laya Vision: Microduck find-and-kick

[thaitea/laya-vision](https://huggingface.co/thaitea/laya-vision) (201M) fine-tuned to drive a simulated
[Microduck](https://github.com/pollen-robotics/microduck) from its own camera: find the ball, walk up to it, kick it.
Each step it sees one 256×256 frame from the duck's camera and answers one `choice` question over four actions,
`FORWARD`, `LEFT`, `RIGHT` and `KICK`. It plays in both of [quackd](https://github.com/r33drichards/quackd)'s
simulators: `microduck:mujoco`, MuJoCo physics with upstream's Microduck model walking on upstream's trained policy,
and `microduck:sim2d`, a flat cartoon.

- **Try it:** [thaitea/laya-vision-microduck-kick-demo](https://huggingface.co/spaces/thaitea/laya-vision-microduck-kick-demo)
- **Code:** data and evaluation in [r33drichards/quackd](https://github.com/r33drichards/quackd/tree/claude/dazzling-mendel-e6k2yb) (`scripts/collect_microduck_rollouts.py`, `scripts/eval_microduck.py`), training in [r33drichards/laya-vision](https://github.com/r33drichards/laya-vision/tree/claude/dazzling-mendel-e6k2yb) (`modal_microduck.py`)
- **Status:** experimental research. It has only seen those two simulators, rendered by quackd 0.14.0: not a real
  camera, and not the browser simulator at quackd.org, which draws its scene with a different renderer. An independent
  fork of [Laya](https://github.com/NandhaKishorM/laya), not affiliated with Convai Innovations, its authors.

## Usage

```bash
pip install "quackd[mujoco]==0.14.0"          # MuJoCo renders headless through OSMesa: MUJOCO_GL=osmesa
git clone https://github.com/r33drichards/laya-vision && pip install -e ./laya-vision torchvision
```

```python
import laya
from quackd_microduck.sim3d.world import MujocoWorld
from quackd_microduck.sim3d.render import render_headcam

agent = laya.load_vlm("thaitea/laya-vision-microduck-kick")
question = {"action": {
    "type": "choice",
    "instructions": "You are a small duck robot looking through your own camera. Somewhere in the arena is an "
                    "orange ball. Turn to find it, walk up to it until it is close and in front of you, then kick "
                    "it. Which action should you take now?",
    "criteria": {"FORWARD": "walk forward", "LEFT": "turn left", "RIGHT": "turn right",
                 "KICK": "kick the ball in front of you"},
}}
frame = render_headcam(MujocoWorld(seed=0, body="microduck"), 256)
agent.predict({"image": frame}, question)["answers"]["action"]["choice"]
```

The question must be worded as above: it is the one the model was trained on. What an action does depends on the
simulator. In MuJoCo, `FORWARD` is 1.0 s at 0.3 m/s and a turn 0.6 s at 0.8 rad/s; in the 2D simulator, every move is
0.5 s at 0.25 m/s or 0.6 rad/s. A kick is followed by 1.5 s for the ball to roll. `scripts/eval_microduck.py --sim
mujoco --policy laya --model thaitea/laya-vision-microduck-kick` runs the whole loop.

**Probabilities on this question are underconfident.** The choice temperature was fitted on the base recipe's
calibration set so that general questions stay calibrated (see below), and on duck frames it spreads probability more
than the answers deserve: calibrated ECE 0.29 on the 3D val frames and 0.19 on the 2D ones, where the top answer is
right 91.8% and 98.1% of the time. The chosen action is unaffected. Pass `temperature={"choice": 2.27}` to `predict`
for probabilities fitted to duck frames instead.

## Training

- **Duck data:** frames from `find-and-kick` episodes, each labelled with a teacher's action: the ground-truth oracle,
  except that it turns left while the ball is out of view, since an empty frame cannot say which way is shorter. The
  duck follows the teacher with a 30% chance of a random move, so the data covers off-course states and the way back;
  30% of episodes start near the ball; three quarters of the ball-out-of-view frames are dropped. 3D: 15,061 train and
  1,376 val frames from 1,000 and 100 episodes (seeds 1000–1999 and 101000–101099). 2D: 11,554 and 1,136 from 800 and
  80 episodes. The evaluation's seeds, 0–9 and 20–119, were never used.
- **Replay:** the recipe `thaitea/laya-vision` came from, so it keeps what it could do: the autoresearch pool's Cauldron
  and rubric-scored sets, its Atari and ViZDoom expert frames, and generated Maze, Snake and classic-control examples.
- **Mix:** duck frames 35% of draws (3D 21%, 2D 14%), games 30%, the recipe's question-answering sets 35%.
- **Run:** from `thaitea/laya-vision`, 4,000 steps at batch 32 on an A100 (21 minutes), vision tower frozen, head LR
  1.41e-4, backbone 2.83e-5. Action accuracy on the held-out val frames at the end: 92.0% (3D), 97.9% (2D).
  Temperatures then refitted on the recipe's calibration tail alone (`modal_microduck.py::recalibrate`).
  `training_metrics.json` has the training log.

## Results

### The task

Closed-loop, graded by the simulator's ground truth: success is a kick that connected and a ball that ended at least
0.3 m from where it started. At most 60 decisions an episode; a duck that falls over has failed.

| Policy | 3D, seeds 0–9 | 3D, seeds 20–119 | 2D, seeds 20–119 |
|---|---:|---:|---:|
| **this model** | **8/10** | **86/100** | **95/100** |
| the teacher it imitates | 10/10 | 95/100 | 95/100 |
| `thaitea/laya-vision` (before) | 0/10 | | 0/10 (seeds 0–9) |
| random actions | 0/10 | 9/100 | 6/100 |

In 3D the model never fell (random actions fell in 72 of 100 episodes). Of its 14 misses on seeds 20–119, 8 ran out
of steps without connecting a kick and 6 kicked but moved the ball less than 0.3 m. In 2D it matches its teacher.
Episodes were played with this checkpoint's weights before the temperatures were refitted, which changes
probabilities but never the chosen action. Each decision takes about 0.6 s on 4 CPU cores.

### Question answering

On the three official VQA validation splits, from committed row-level predictions
(`results/raw/smolvlm-microduck-kick-ft2-cal-best.vqa-val.*`, checked by `benchmarks/verify_published.py`):

| Checkpoint | A-OKVQA | ScienceQA | VQAv2 yes/no |
|---|---:|---:|---:|
| [thaitea/laya-vision-microduck-kick](https://huggingface.co/thaitea/laya-vision-microduck-kick) | 58.9% | 82.8% | 71.0% |
| [thaitea/laya-vision](https://huggingface.co/thaitea/laya-vision) | 59.8% | 82.4% | 71.4% |

Calibrated ECE on those three: 0.153, 0.021 and 0.092, against the base's 0.089, 0.031 and 0.065. On the full suite
of `thaitea/laya-vision` (59,427 questions over 34 validation sets) pooled accuracy is 69.1% before and after, with ECE
0.041 before and 0.047 after. 7 sets rose by more than a point and 10 fell by more than a point, the largest falls on
DVQA (92.4% → 90.4%), EvalMuse (23.3% → 21.3%), CrisisMMD (67.9% → 65.6%) and ChartQA (70.7% → 65.9%, 41 questions).

### Games

The replay kept most of them, but not all:

| Game | before | after |
|---|---:|---:|
| ViZDoom basic, mean reward | 75.8 | 75.6 |
| Atari Freeway | 23.7 | 24.7 |
| Atari Galaxian (random play: 673) | 450 | 697 |
| Atari Breakout | 41.3 | 9.7 |
| Maze 4×4, solved | 46% | 18% |
| Maze 6×6, solved | 0% | 2% |
| Snake 10×10, food per game | 4.3 | 3.25 |
| Acrobot | −138.7 | −179.2 |
| MountainCar | −145.4 | −168.0 |
| CartPole | 24.4 | 9.7 |
| LunarLander | −223.9 | −212.6 |

Breakout, the 4×4 maze and CartPole (where it now always pushes right) are clearly worse. For those, use
`thaitea/laya-vision`.
