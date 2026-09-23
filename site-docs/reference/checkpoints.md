# Checkpoints

| Checkpoint | Backbone | Trained on | A-OKVQA | ScienceQA | VQAv2 yes/no | `score` head | Latency, L4 bf16 |
|---|---|---|---|---|---|---|---|
| [thaitea/laya-vision](https://huggingface.co/thaitea/laya-vision), **recommended** (same weights as [laya-vision-smolvlm-256m-score](https://huggingface.co/thaitea/laya-vision-smolvlm-256m-score)) | SmolVLM-256M, options attend to each other | 19 Cauldron subsets + 4 rubric-scored sets | 60.0% | 82.8% | 72.4% | trained | ~41 ms |

Accuracies are on the official validation splits (VQAv2 yes/no is a re-split of the official val set by image, so
not comparable to published VQAv2 numbers).

- **Calibration.** Calibrated ECE pooled over all of a checkpoint's validation sets is 0.02 to 0.035, but it
  varies by set: for the recommended checkpoint it is 0.16 on A-OKVQA, 0.035 on ScienceQA and 0.077 on VQAv2
  yes/no. What that means for your data: [Calibration](../concepts/calibration.md).
- **Breadth.** The recommended checkpoint averages 75% over 26 validation sets, with 93.7% on IconQA, 91.8% on
  DVQA and 89.9% on Hateful Memes. The full scorecard covers 34 validation sets, human-vote calibration, the games
  suite and latency: [Scorecard](evals/laya-vision.md).
- **`score` answers.** The recommended checkpoint is the only one whose `score` answers mean anything. On held-out
  rubric data it scores 54% over 5 levels on VLFeedback response grading (prior-only baseline 27.5%), is 0.8 levels
  off on average against 1.4 for the baseline, and 0.38 levels off on 3-level damage severity:
  [Score head results](results/score-results.md).

## Evidence

The recommended row's accuracies are backed by committed per-row predictions in
[`results/raw/`](https://github.com/r33drichards/laya-vision/tree/main/results/raw). `python benchmarks/verify_published.py` recomputes them (and
calibrated ECE) and checks them against the same table in the README and against
[its metrics](https://github.com/r33drichards/laya-vision/blob/main/docs/smolvlm-cauldron-score-bidir-full-metrics.json). The rules for adding or changing a
published number are in [`AGENTS.md`](https://github.com/r33drichards/laya-vision/blob/main/AGENTS.md).

## Other checkpoints

- [thaitea/laya-vision-modernvbert-250m](https://huggingface.co/thaitea/laya-vision-modernvbert-250m): ModernVBERT
  on 19 Cauldron subsets: [results](results/modernvbert-cauldron.md).

Pin a checkpoint's revision when you load it (`load_vlm(..., revision=...)`): the id on the Hub can move.
