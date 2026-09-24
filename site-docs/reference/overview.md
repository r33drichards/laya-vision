# Reference

Factual reference material: the API, its inputs and outputs, the checkpoints, file formats and the published
results. For learning and tasks, see the [Tutorials](../tutorials/overview.md) and
[How-to guides](../how-to/overview.md).

## API and checkpoints

- [predict() and the answer schema](predict.md): loading a checkpoint, the state, the three question types,
  every `predict` argument, and every field of the result.
- [Checkpoints](checkpoints.md): the recommended checkpoint, its accuracy and calibration, and the evidence behind
  them.

## Files and formats

- [Browser demo files and checks](web-demo.md): what the web export writes, the pinned runtime, and how closely each
  precision matches PyTorch.
- [Atari training data format](atari-data-format.md): the layout every Atari training source is written in.

## Results

Generated from result files by `scripts/eval_report.py`, and not edited by hand:

- [Scorecard: thaitea/laya-vision (201M)](evals/laya-vision-201m.md): the recommended checkpoint on 34 validation
  sets, human-vote calibration, the games suite and latency.
- [Scorecard: the previous checkpoint](evals/laya-vision.md): the same suite on `thaitea/laya-vision` before
  2026-09-24 (now `laya-vision-smolvlm-256m-score`).

Experiment reports:

- [Score head results](results/score-results.md): the first runs with rubric-scored data, both backbones.
- [ModernVBERT on The Cauldron](results/modernvbert-cauldron.md): the bidirectional backbone post-trained on 19
  Cauldron subsets.
- [Image splitting on SmolVLM2](results/split-bench.md): accuracy against latency at three split settings.
- [Robustness](results/robustness.md): answers under meaning-preserving perturbations, and whether the image is
  used.
- [Game training](results/game-training.md): ViZDoom and Atari policies, DAgger, and the cheaper preprocessing.
- [Typed answers vs generated JSON](results/decision-vs-generation.md): `predict` against a generative baseline on
  the same image.
- [What didn't work](results/what-didnt-work.md): approaches that were tried and dropped.
