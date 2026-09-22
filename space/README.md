---
title: Laya Vision
emoji: 👁️
colorFrom: indigo
colorTo: green
sdk: gradio
sdk_version: 6.28.0
python_version: "3.12"
app_file: app.py
pinned: false
license: cc-by-nc-sa-4.0
models:
- thaitea/laya-vision
short_description: Calibrated yes/no, multiple-choice and rubric-score answers about images
---

# Laya Vision demo

Ask calibrated yes/no (`noul`), multiple-choice (`choice`) and rubric-graded (`score`) questions about an image, in one forward pass with no text generation. The Space runs [thaitea/laya-vision](https://huggingface.co/thaitea/laya-vision), the latest Laya Vision checkpoint (SmolVLM-256M, trained on The Cauldron plus four rubric-scored datasets), on CPU. The code is at [r33drichards/laya-vision](https://github.com/r33drichards/laya-vision).

The example photo is "Cat November 2010-1a" by Alvesgaspar, CC BY-SA 3.0, from Wikimedia Commons.
