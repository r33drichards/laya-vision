# Checkpoints

| Checkpoint | Backbone | Trained on | A-OKVQA | ScienceQA | VQAv2 yes/no | `score` head | Params | Latency, L4 bf16 |
|---|---|---|---|---|---|---|---|---|
| [thaitea/laya-vision](https://huggingface.co/thaitea/laya-vision), **recommended** (same weights as [laya-vision-201m](https://huggingface.co/thaitea/laya-vision-201m)) | SmolVLM-256M cut to 20 of 30 language layers, options attend to each other | 19 Cauldron subsets + 4 rubric-scored sets + game frames | 59.8% | 82.4% | 71.4% | trained | 201M | 41 ms |
| [thaitea/laya-vision-smolvlm-256m-score](https://huggingface.co/thaitea/laya-vision-smolvlm-256m-score) (`thaitea/laya-vision` until 2026-09-24, revision `d1fbdc0`) | SmolVLM-256M, options attend to each other | 19 Cauldron subsets + 4 rubric-scored sets | 60.0% | 82.8% | 72.4% | trained | 237M | ~41 ms |

Accuracies are calibrated answers on the official validation splits (VQAv2 yes/no is a re-split of the official
val set by image, so not comparable to published VQAv2 numbers). Latency is the median `predict` call on an L4,
preprocessing included; timed on the same GPU, the recommended checkpoint takes 0.83× the time of the previous one.

## The recommended checkpoint

SmolVLM-256M-Instruct cut to its first 20 language-model layers, continued from the previous checkpoint and
trained for 2 hours (28,414 steps at batch 64 on one H100) on the Cauldron and rubric sets plus game frames, which
were 45% of training draws. The recipe came out of the [autoresearch loop](https://github.com/r33drichards/laya-vision/tree/main/autoresearch).

- **Against the previous checkpoint:**
  - accuracy: 69.1% against 69.3% over 59,427 questions on 34 validation sets;
  - within a point on 23 sets, ahead on 3 and behind on 8, mostly small ones (VQA-RAD, 62 questions, 80.6% against 88.7%).
- **Calibration:** ECE 0.041 pooled over all questions, against 0.064 for the previous checkpoint. By set: 0.092 on A-OKVQA, 0.028 on ScienceQA and 0.064 on VQAv2 yes/no. What that means for your data: [Calibration](../concepts/calibration.md).
- **Games:** 0.35 on the autoresearch games benchmark (0 = random play, 1 = expert), against −0.04. Near expert on ViZDoom basic, Atari Freeway, Acrobot and MountainCar; LunarLander is worse than random.
- **Full scorecard:** [Scorecard: thaitea/laya-vision (201M)](evals/laya-vision-201m.md).

## The previous checkpoint

- **Breadth.** It averages 75% over 26 validation sets, with 93.7% on IconQA, 91.8% on DVQA and 89.9% on Hateful Memes. The full scorecard covers 34 validation sets, human-vote calibration, the games suite and latency: [Scorecard](evals/laya-vision.md).
- **`score` answers.** On held-out rubric data it scores 54% over 5 levels on VLFeedback response grading (prior-only baseline 27.5%). It is 0.8 levels off on average, against 1.4 for the baseline, and 0.38 levels off on 3-level damage severity: [Score head results](results/score-results.md).

## Evidence

Both rows' accuracies are backed by committed per-row predictions in
[`results/raw/`](https://github.com/r33drichards/laya-vision/tree/main/results/raw). `python benchmarks/verify_published.py` recomputes them (and
calibrated ECE) and checks them against the same table in the README and against each checkpoint's metrics
([recommended](https://github.com/r33drichards/laya-vision/blob/main/docs/autoresearch-long-sep24-b64-metrics.json),
[previous](https://github.com/r33drichards/laya-vision/blob/main/docs/smolvlm-cauldron-score-bidir-full-metrics.json)). The rules for adding or changing a
published number are in [`AGENTS.md`](https://github.com/r33drichards/laya-vision/blob/main/AGENTS.md).

## Other checkpoints

- [thaitea/laya-vision-modernvbert-250m](https://huggingface.co/thaitea/laya-vision-modernvbert-250m): ModernVBERT
  on 19 Cauldron subsets: [results](results/modernvbert-cauldron.md).
- [thaitea/laya-vision-smolvlm-256m](https://huggingface.co/thaitea/laya-vision-smolvlm-256m): the original, on
  A-OKVQA, ScienceQA and VQAv2 yes/no only.

Pin a checkpoint's revision when you load it (`load_vlm(..., revision=...)`): the id on the Hub can move, as
`thaitea/laya-vision` did on 2026-09-24.
