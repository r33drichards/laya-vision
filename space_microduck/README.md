---
title: Laya Vision plays find-and-kick
emoji: 🦆
colorFrom: yellow
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: cc-by-nc-sa-4.0
models:
- thaitea/laya-vision-microduck-kick
short_description: A vision model walks a simulated duck robot to a ball
---

# Laya Vision plays find-and-kick

[thaitea/laya-vision-microduck-kick](https://huggingface.co/thaitea/laya-vision-microduck-kick) driving a simulated
[Microduck](https://github.com/pollen-robotics/microduck) from its own camera, on CPU, in either of
[quackd](https://github.com/r33drichards/quackd)'s simulators (`quackd[mujoco]` 0.14.0 from PyPI): `microduck:mujoco`,
MuJoCo physics with upstream's Microduck model walking on upstream's trained policy (rendered through OSMesa), and `microduck:sim2d`, the flat cartoon. Each step the model sees the duck's camera and picks
`FORWARD`, `LEFT`, `RIGHT` or `KICK`; the view of the arena and the ground-truth numbers are for the viewer only.

The action space, timings and question are the ones quackd's `scripts/eval_microduck.py` scores. Replaying an eval
episode's actions through this Space's loop lands the duck and the ball in the same place in both simulators.
The checkpoint is pinned at revision `7505ee2` in `app.py`; `LAYA_MODEL` and `LAYA_REVISION` override it.
It is a Docker Space because the Gradio SDK installs `gradio[mcp]`, whose `mcp<2` pin cannot coexist with quackd's
`mcp>=2`; the `Dockerfile` installs Gradio without that extra. The code is in
[r33drichards/laya-vision](https://github.com/r33drichards/laya-vision) under `space_microduck/`.
