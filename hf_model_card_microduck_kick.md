---
license: cc-by-nc-sa-4.0
base_model: thaitea/laya-vision
library_name: laya
pipeline_tag: visual-question-answering
datasets:
- HuggingFaceM4/A-OKVQA
- derek-thomas/ScienceQA
- lmms-lab-encoder/VQAv2
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
---

# Laya Vision: Microduck find-and-kick

[thaitea/laya-vision](https://huggingface.co/thaitea/laya-vision) (201M) fine-tuned to drive a simulated
[Microduck](https://github.com/pollen-robotics/microduck) from its own camera: find the ball, walk up to it, kick
it. Each step it sees one 256×256 frame from the duck's camera in [quackd](https://github.com/r33drichards/quackd)'s 2D
simulator (`microduck:sim2d`) and answers one `choice` question over four actions, `FORWARD`, `LEFT`, `RIGHT` and
`KICK`.

**It has only ever seen that 2D cartoon.** It has not been tried on the MuJoCo simulator's 3D renders or on a real
camera, and there is no reason to expect it to work on either.

- **Code:** training and eval data from [r33drichards/quackd](https://github.com/r33drichards/quackd/tree/claude/dazzling-mendel-e6k2yb) (`scripts/collect_microduck_rollouts.py`, `scripts/eval_microduck.py`), training with [r33drichards/laya-vision](https://github.com/r33drichards/laya-vision) (`modal_app.py::finetune_long`)
- **Status:** experimental research. An independent fork of [Laya](https://github.com/NandhaKishorM/laya), not affiliated with Convai Innovations, its authors.

## Usage

```python
import laya
from quackd.sim2d.world import World
from quackd.sim2d.render import render_duckcam

agent = laya.load_vlm("thaitea/laya-vision-microduck-kick")
question = {"action": {
    "type": "choice",
    "instructions": "You are a small duck robot looking through your own camera. Somewhere in the arena is an "
                    "orange ball. Turn to find it, walk up to it until it is close and in front of you, then kick "
                    "it. Which action should you take now?",
    "criteria": {"FORWARD": "walk forward", "LEFT": "turn left", "RIGHT": "turn right",
                 "KICK": "kick the ball in front of you"},
}}
frame = render_duckcam(World(seed=0), 256)
agent.predict({"image": frame}, question)["answers"]["action"]["choice"]
```

The question must be worded as above: it is the one the model was trained on. One action is 0.5 s of simulated
time (0.125 m forward, or about 17° of turn); a kick is followed by 1.5 s for the ball to roll.
`scripts/eval_microduck.py --policy laya --model thaitea/laya-vision-microduck-kick` runs the whole loop.

## Training

- **Data:** 11,554 train and 1,136 val frames from 800 and 80 episodes of `find-and-kick` on `microduck:sim2d`
  (seeds 1000–1799 and 101000–101079, never the eval's). Each frame is labelled with a teacher's action: the
  ground-truth oracle, except that it turns left while the ball is out of view, since an empty frame cannot say
  which way is shorter. The duck follows the teacher with a 30% chance of a random move, so the data covers
  off-course states and the way back; 30% of episodes start near the ball; three quarters of the ball-out-of-view
  frames are dropped. Labels: LEFT 44%, FORWARD 37%, KICK 13%, RIGHT 6%.
- **Mix:** those frames drawn half the time, and the A-OKVQA, ScienceQA and VQAv2 yes/no train splits (each capped
  at 12,000) the other half.
- **Run:** from `thaitea/laya-vision`, 1 epoch of the mix (1,287 steps, batch 32, A100, 8 minutes), full model
  unfrozen except the vision tower, `finetune_long` defaults otherwise. Best checkpoint by action accuracy on the
  duck val frames: **98.6%**, ECE 0.006. `training_metrics.json` has the log.

## Results

### The task

Closed-loop, graded by the simulator's ground truth: success is a kick that connected and a ball that ended at least
0.3 m from where it started. At most 60 decisions an episode.

| Policy | Seeds 0–9 | Seeds 20–119 |
|---|---:|---:|
| **this model** | **10/10** | **95/100** |
| the teacher it imitates | 10/10 | 95/100 |
| `thaitea/laya-vision` (before fine-tuning) | 0/10 | |
| random actions | 4/10 | 6/100 |

Before fine-tuning the model chose `KICK` on 1,799 of 1,800 decisions and never approached the ball.

### Image question answering

The full evaluation suite of `thaitea/laya-vision` (59,427 questions over 34 validation sets), calibrated answers:

| | before | after |
|---|---:|---:|
| Pooled accuracy | 69.1% | 70.0% |
| ECE | 0.041 | 0.021 |
| Cauldron sets, mean accuracy (19) | 77.3% | 77.5% |
| Rubric-scored sets, mean accuracy (4) | 65.9% | 65.1% |
| Held-out eval sets, mean accuracy (8) | 60.8% | 62.5% |

11 sets rose more than a point and 9 fell more than a point. The rises on ScienceQA (+4.8), A-OKVQA (+1.4) and their
Cauldron versions (+5.0, +5.4) come from training on those sets' train splits. VizWiz rose from 47.3% to 62.4%,
which is not understood. The largest falls: FER+ 45.7% → 41.3% (3,569 questions), AVA 84.2% → 81.5%, Cauldron VQAv2
78.9% → 76.7%, ChartQA 70.7% → 61.0% (41 questions).

### Games: this checkpoint forgot them

No game frames were in the fine-tuning mix, and most games got worse:

| Game | before | after |
|---|---:|---:|
| ViZDoom basic, mean reward | 75.8 | 64.7 |
| Atari Freeway | 23.7 | 21.3 |
| Atari Breakout | 41.3 | 5.3 |
| Atari Galaxian (random: 673) | 450 | 380 |
| Maze 4×4, solved | 46% | 8% |
| Snake 10×10, food per game | 4.3 | 1.85 |
| Acrobot | −138.7 | −277.2 |
| MountainCar | −145.4 | −178.6 |
| CartPole | 24.4 | 16.7 |
| LunarLander | −223.9 | −169.4 |

For the games, use `thaitea/laya-vision`.
