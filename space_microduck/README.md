---
title: Laya Vision plays find-and-kick
emoji: 🦆
colorFrom: yellow
colorTo: indigo
sdk: gradio
sdk_version: 6.28.0
python_version: "3.12"
app_file: app.py
suggested_hardware: zero-a10g
pinned: false
license: cc-by-nc-sa-4.0
models:
- thaitea/laya-vision-microduck-kick
short_description: A vision model walks a simulated duck robot to a ball
---

# Laya Vision plays find-and-kick

[thaitea/laya-vision-microduck-kick](https://huggingface.co/thaitea/laya-vision-microduck-kick) driving a simulated
[Microduck](https://github.com/pollen-robotics/microduck) from its own camera, in real time, in either of
[quackd](https://github.com/r33drichards/quackd)'s simulators (0.14.0 from PyPI): `microduck:mujoco`, MuJoCo physics
with upstream's Microduck model walking on upstream's trained policy (rendered through OSMesa, `packages.txt`), and
`microduck:sim2d`, the flat cartoon. Each step the model sees the duck's camera and picks `FORWARD`, `LEFT`, `RIGHT` or
`KICK`; the view of the arena and the ground-truth numbers are for the viewer only.

Each Play is one ZeroGPU call (`play_episode` in `app.py`): the GPU worker builds the world from the seed, plays the
episode at wall-clock speed with the model deciding on the GPU (about 70 ms a decision), and streams the frames back.
One call per episode rather than per decision keeps a visitor inside ZeroGPU's run limit. The action space, timings and question are the ones quackd's
`scripts/eval_microduck.py` scores, and the physics is stepped on the simulator's own clock whatever the pacing, so an
episode goes where the eval's does. The checkpoint is pinned at revision `7505ee2` in `app.py`; `LAYA_MODEL` and
`LAYA_REVISION` override it. quackd is installed at start-up without its dependencies, because the Gradio SDK's
`gradio[mcp]` pins `mcp<2` and quackd declares `mcp>=2` for a server this Space does not run. The code is in
[r33drichards/laya-vision](https://github.com/r33drichards/laya-vision) under `space_microduck/`.
