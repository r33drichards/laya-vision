# What didn't work

- **A SigLIP projector into Laya's text encoder**
  ([`siglip-projector-experiment`](https://github.com/r33drichards/laya-vision/tree/siglip-projector-experiment)): kept text answers bit-identical but
  never learned to use the image in 5 runs; accuracy with shuffled images matched accuracy with real ones.
- **Annealing the cross-entropy weight to zero** so training ends on the proper scoring rule alone: tied on
  accuracy and made the raw model more overconfident ([ModernVBERT on The Cauldron](modernvbert-cauldron.md)).
- **Balancing AVA's levels on the train split only**: the head learned a flatter vote histogram than the val voters
  produce and lost to a prior-only baseline until the balancing was removed.
- **A third epoch over the same rubric data**: half a point on VLFeedback, nothing elsewhere. More passes are flat;
  the next gains need new rubric data ([Score head results](score-results.md)).
- **An in-browser demo (ONNX Runtime Web, WebGPU)**: the model exported to ONNX and run on the visitor's device.
  It ran, and matched PyTorch closely at fp32, but the smaller exports it needed (fp16 on the GPU, 8- and 4-bit
  weights) and the browser's own image decoding moved the answers enough to misrepresent the model, so it was
  removed in favour of the [Hugging Face Space](https://huggingface.co/spaces/thaitea/laya-vision-demo). Dynamic int8 quantisation, tried for it, moved probabilities
  by up to 0.62 and flipped the top answer on 4 of 9 validation questions. The code is in the repository history.
