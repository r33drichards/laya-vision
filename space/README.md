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
short_description: Calibrated yes/no, choice and rubric answers about an image
---

# Laya Vision demo

Ask calibrated yes/no (`noul`), multiple-choice (`choice`) and rubric-graded (`score`) questions about an image, in one forward pass with no text generation. The Space runs [thaitea/laya-vision](https://huggingface.co/thaitea/laya-vision) at revision `8b318c9`, the 201M checkpoint (SmolVLM-256M cut to 20 of its 30 language layers, trained on The Cauldron, four rubric-scored datasets and game frames), on CPU. The revision is pinned in `app.py` (`LAYA_REVISION`). The code is at [r33drichards/laya-vision](https://github.com/r33drichards/laya-vision).

The example photo is "Cat November 2010-1a" by Alvesgaspar, CC BY-SA 3.0, from Wikimedia Commons.
