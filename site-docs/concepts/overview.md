# Concepts

These pages are **explanation**: how Laya Vision works and why it is designed the way it is. They are for
understanding, not step-by-step instructions (see the [How-to guides](../how-to/overview.md) for those).

## The big picture

Laya Vision turns "ask a vision-language model a question" into "score the options you wrote". The image, the state
text, the question and its options go through the backbone once; a small head reads one logit per option; a
per-type temperature turns those logits into probabilities. Nothing is generated, so the cost is one forward pass
per batch of questions, whatever the answers are.

## Explanations

- [How it works](how-it-works.md): the sequence, the per-option readout, and how causal and bidirectional backbones
  differ.
- [Architecture](architecture.md): every backbone variant and the shared head, with diagrams, the training
  objective, the shared-prefix cache and the checkpoint layout.
- [Calibration](calibration.md): what the temperatures do, why they are fitted after training, and what refitting
  them can and cannot change.
- [Training data](data.md): the prepared datasets, from The Cauldron to the held-out evaluation sets.
- [Training data for score questions](score-data.md): the four rubric-scored datasets behind the `score` head, and
  how they were cleaned.
- [The browser runtime](browser-runtime.md): how the web page reproduces `predict` with ONNX Runtime Web.
- [Caching the fixed question in game play](game-caching.md): what caching the constant question could save, and
  the CUDA-graph path that was built instead.
