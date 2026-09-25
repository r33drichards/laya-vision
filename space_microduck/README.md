---
title: Laya Vision plays find-and-kick
emoji: 🦆
colorFrom: yellow
colorTo: indigo
sdk: gradio
sdk_version: 6.28.0
python_version: "3.12"
app_file: app.py
pinned: false
license: cc-by-nc-sa-4.0
models:
- thaitea/laya-vision-microduck-kick
short_description: A vision model drives a simulated duck robot to a ball
---

# Laya Vision plays find-and-kick

[thaitea/laya-vision-microduck-kick](https://huggingface.co/thaitea/laya-vision-microduck-kick) driving a simulated
[Microduck](https://github.com/pollen-robotics/microduck) in [quackd](https://github.com/r33drichards/quackd)'s 2D
simulator (`microduck:sim2d`, from `quackd` 0.14.0 on PyPI), on CPU. Each step the model sees the duck's camera and
picks `FORWARD`, `LEFT`, `RIGHT` or `KICK`; the top-down view and the ground-truth numbers are for the viewer only.

The action space, timings and question are the ones quackd's `scripts/eval_microduck.py` scores, and on seeds 0–9 this
Space's loop reproduces that harness's episodes step for step. `LAYA_MODEL` and `LAYA_REVISION` override the
checkpoint. The code is in [r33drichards/laya-vision](https://github.com/r33drichards/laya-vision) under
`space_microduck/`.
