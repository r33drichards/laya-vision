# Robustness under meaning-preserving perturbations

Does the published checkpoint (`smolvlm/cauldron-score-2ep-bidir-full/best`, SmolVLM-256M with block option
attention, on the Hub as `thaitea/laya-vision`) give the same answer when the question means the same thing?
And does it actually use the image? This page covers five perturbation families on 1,942 val rows from seven
sets. Each family is deterministic given a seed.

```
modal run --detach modal_app.py::robustness_eval --n 300 --tag <new-tag>   # L4, ~9 min for 40k rows, no training
python -m laya.robustness results/robustness/predictions.jsonl.gz         # re-summarise offline, no model
```

Code: [`laya/robustness.py`](https://github.com/r33drichards/laya-vision/blob/main/laya/robustness.py) (builders, scoring, summary; each rule is documented
there), `modal_app.py::robustness` / `robustness_eval`, tests in `tests/test_robustness.py`. Raw outputs:
[`results/robustness/predictions.jsonl.gz`](https://github.com/r33drichards/laya-vision/blob/main/results/robustness/predictions.jsonl.gz) (one line per scored
row: ids, family/variant, label, raw logits, calibrated probabilities, argmax) and
[`results/robustness/summary.json`](https://github.com/r33drichards/laya-vision/blob/main/results/robustness/summary.json).

## Setup

- **Source rows.** From each val split, 300 rows are drawn with seed 0 (all 282 of `cauldron_ai2d` and all 160
  of `cauldron_vsr`). The sets are `aokvqa`, `scienceqa` and `vqav2_yesno` (the official val splits) plus the
  Cauldron holdouts `cauldron_ai2d` (diagrams, lettered choice), `cauldron_visual7w` (photos, choice),
  `cauldron_vsr` (spatial yes/no) and `cauldron_mapqa` (maps, yes/no). All 1,942 sampled rows have an image.
- **Group bookkeeping** follows an output-blind stability fixture. Every perturbed row carries
  `group_id = <dataset>/<file index>` back to its source row. Variants are built from the inputs and the seed
  alone, before any model output, and labels are never re-derived. Accuracy is averaged within a group first,
  then across groups, so a row with seven rewordings counts once. **Flip rate** is the share of a group's
  variants whose argmax differs from the unperturbed row's argmax, averaged over groups.
- **Intervals** are 95% percentile bootstraps (1,000 resamples; 400 for ECE) over *image clusters*. Rows that
  share an image are resampled together, because a Cauldron image carries up to four questions and a VQAv2 image
  several. "vs orig" is the paired difference against the same groups' unperturbed accuracy, and its interval
  comes from the same resamples.
- **ECE** uses the checkpoint's own temperatures (choice 2.20, score 1.37, noul 2.13), 15 bins on the max
  probability, pooled over all rows of a family (so it is not group-weighted).
- **Balanced accuracy** is the mean per-label recall, group-weighted. On the yes/no sets that means yes-recall
  and no-recall. On the choice sets the label is the gold option's *position*, so the figure depends on rare
  positions. For example, ScienceQA has 6 rows whose gold is the 5th option, which is why its `orig` and
  `option_order` balanced accuracies differ so much. Read it for the yes/no sets.

### Families

| family | variants per source row | what changes |
|---|---|---|
| `option_order` | 1 (yes/no) to 6 | display order only: every cyclic shift, the reversal, one seeded random permutation, de-duplicated. The label stays in label order (`collect_logits` maps the logits back). |
| `text` | 6-7 | `prefix` "Question: ", `suffix` " Choose the correct option.", `double_spaces`, `first_case` (toggle; acronyms skipped), `end_punct` (drop or add the final ?/.), `noul_frame` ("Is it true that ...?" / "Decide whether the answer to this question is yes: ..."), `noul_options` (option texts "no"/"yes"), `option_case` (toggle the first letter of plain-word options), `option_period`. A rule that doesn't apply produces no row. |
| `image` | 8 | JPEG re-encode at q70/q40/q20, centre crop to 95% of the area, seeded random crop to 90%, downscale x0.5 and back (bicubic), brightness x0.9 / x1.1. |
| `image_shuffle` | 1 | the image(s) of a *different* image from the same dataset (a seeded derangement over distinct images, so no question gets its own image back through a sibling question). |
| `text_only` | 1 | images removed; any text context (e.g. a ScienceQA hint) kept. |

In total: 1,942 source rows, 5,875 order rows, 13,156 text rows, 15,536 image rows and 1,942 each for the two
controls, so 40,393 forwards. That took 8.4 minutes on one L4.

## Results

Accuracy with its 95% interval, the paired change from unperturbed, and the flip rate. Flip rate is the share of
variants whose answer changes, whether or not the new answer is right.

| dataset | orig acc | option order: acc / Δ / flip | text: acc / Δ / flip | image: acc / Δ / flip | **shuffled image** acc / Δ | **no image** acc / Δ | chance / majority |
|---|---|---|---|---|---|---|---|
| aokvqa | 0.587 [0.530, 0.640] | 0.584 / −0.002 / 0.077 | 0.586 / −0.001 / 0.064 | 0.586 / −0.000 / 0.157 | 0.313 / **−0.273** | 0.387 / **−0.200** | 0.250 / 0.267 |
| scienceqa | 0.863 [0.823, 0.900] | 0.855 / −0.008 / 0.072 | 0.855 / −0.008 / 0.037 | 0.846 / −0.017 / 0.075 | 0.610 / **−0.253** | 0.663 / **−0.200** | 0.366 / 0.370 |
| vqav2_yesno | 0.743 [0.690, 0.793] | 0.750 / +0.007 / 0.013 | 0.735 / −0.008 / 0.048 | 0.748 / +0.004 / 0.121 | 0.507 / **−0.237** | 0.557 / **−0.187** | 0.500 / 0.520 |
| cauldron_ai2d | 0.777 [0.727, 0.827] | 0.775 / −0.002 / 0.062 | 0.774 / −0.003 / 0.050 | 0.750 / **−0.026** / 0.108 | 0.550 / **−0.227** | 0.596 / **−0.181** | 0.250 / 0.266 |
| cauldron_visual7w | 0.870 [0.830, 0.908] | 0.864 / −0.006 / 0.050 | 0.852 / **−0.018** / 0.047 | 0.825 / **−0.045** / 0.116 | 0.410 / **−0.460** | 0.467 / **−0.403** | 0.250 / 0.290 |
| cauldron_vsr | 0.875 [0.819, 0.922] | 0.875 / +0.000 / 0.000 | 0.863 / −0.012 / 0.032 | 0.842 / −0.033 / 0.117 | 0.494 / **−0.381** | 0.494 / **−0.381** | 0.500 / 0.506 |
| cauldron_mapqa | 0.583 [0.527, 0.639] | 0.593 / +0.010 / 0.023 | 0.615 / **+0.031** / 0.086 | 0.623 / **+0.040** / 0.101 | 0.530 / −0.053 | 0.580 / −0.003 | 0.500 / 0.607 |
| **mean of 7** | 0.757 | 0.757 / −0.000 / 0.043 | 0.754 / −0.003 / 0.052 | 0.746 / −0.011 / 0.114 | 0.488 / −0.269 | 0.535 / −0.222 | |

**Bold Δ**: the paired 95% interval excludes 0. The full per-family intervals, ECE, balanced accuracy and
per-variant numbers are in `summary.json`, and there is a flat table in `laya.robustness.format_table`.

ECE with the checkpoint's temperatures, by family (point estimates):

| dataset | orig | order | text | image | shuffled | no image |
|---|---:|---:|---:|---:|---:|---:|
| aokvqa | 0.193 | 0.187 | 0.186 | 0.189 | 0.342 | 0.224 |
| scienceqa | 0.039 | 0.041 | 0.022 | 0.022 | 0.170 | 0.101 |
| vqav2_yesno | 0.089 | 0.088 | 0.069 | 0.079 | 0.320 | 0.110 |
| cauldron_ai2d | 0.045 | 0.043 | 0.044 | 0.060 | 0.185 | 0.128 |
| cauldron_visual7w | 0.090 | 0.081 | 0.072 | 0.052 | 0.190 | 0.114 |
| cauldron_vsr | 0.051 | 0.071 | 0.037 | 0.043 | 0.424 | 0.357 |
| cauldron_mapqa | 0.062 | 0.045 | 0.022 | 0.014 | 0.098 | 0.019 |

### What it says

1. **Option order is nearly a non-issue.** Accuracy spread across the orders every row has is at most 1.4 points
   (aokvqa 0.010, ai2d 0.014, visual7w 0.007, yes/no sets ≤ 0.010, vsr 0). Accuracy by the gold option's
   displayed position is flat on aokvqa (0.565-0.608), ai2d (0.759-0.784) and visual7w (0.854-0.876). Answers
   still move, though: 5-8% of a 4-way row's re-orderings change the argmax, and 15% of aokvqa rows (13% ai2d,
   10% visual7w) flip under at least one order. The block option attention removes the systematic position
   prior but does not make the head exactly order-invariant. Presumably this is because each option still sits
   at a different position after the image and the question. On the yes/no sets the swap almost never changes the
   answer (vsr 0/160, vqav2 1.3%).
2. **Rewording is mostly harmless. The exception is option-text case.** Averaged over rules, accuracy moves by at
   most 2 points on the sets that use the image (visual7w −1.8, significant), and 3-9% of answers flip. The
   worst single rule is `option_case`, which toggles the first letter of plain-word options ("cab" → "Cab").
   Paired against the same rows unperturbed, it costs 4.5 points on ai2d, 4.3 on visual7w and 7.6 on ScienceQA.
   On aokvqa it costs only 1 point but flips 18.5% of answers. The training copy of A-OKVQA (`cauldron_aokvqa`)
   has its options lowercased by `laya/cauldron.py`, so part of what the head learned is the surface form of the
   options, not only their meaning. `option_period` is harmless (within ±1 point). The yes/no reframing
   `noul_frame` costs vsr 3 points and vqav2 under 1; `noul_options` is harmless on both.
3. **Image corruptions cost a little accuracy but flip about 11% of answers.** The biggest drops come from the
   crops and the downscale: random 90% crop −9 points on visual7w and −11 on vsr, `rescale50` −8 on ai2d,
   `jpeg20` −8 on visual7w and −7 on vsr. On aokvqa accuracy holds but 16% of answers flip. That fits many of
   its rows sitting near a decision boundary, and aokvqa is also the worst-calibrated set here (ECE 0.19).
   Brightness ±10% and JPEG q70 are nearly free: brightness flips at most 9% of answers, and JPEG q70 at most 10% (13% on aokvqa).
4. **The image-shuffle control passes on six of seven sets.** A mismatched image drops accuracy by 23-46 points
   (mean −27). That is at or below the no-image accuracy on every set (equal on vsr). On vqav2 and vsr it
   reaches the chance/majority prior, and on aokvqa it comes within 5 points of it. So the model reads the
   image and trusts it: a wrong image misleads it more than no image does, and its calibrated confidence stays
   high when it is misled (shuffled-image ECE 0.17-0.42 on those six sets, against 0.04-0.19 unperturbed).
   This is the check that would have caught the SigLIP-projector failure (README, "What didn't work"). There,
   shuffled-image accuracy matched real-image accuracy.
5. **`cauldron_mapqa` fails the control. This is a limit of the checkpoint on MapQA, not a plumbing fault.**
   Removing the image costs nothing (−0.003, CI [−0.061, +0.058]) and a shuffled map costs 5 points (CI includes 0). Unperturbed
   accuracy (0.583) is *below* the majority-label rate (0.607), and the text and image perturbations *raise*
   accuracy (+3.1 and +4.0 points, both significant; `noul_frame` and `double_spaces` +5). On this sample the
   checkpoint answers MapQA's yes/no questions from the question text; it does not read the map at 512 px.
   Treat mapqa numbers as text-only until a split-image checkpoint (see `split_bench`) does better.

## Caveats

- **Modest samples.** 160-300 source rows per set, one seed. Intervals are ±4-6 points on accuracy, so a Δ under
  about 2 points is noise unless its paired interval says otherwise. The per-rule and per-op numbers above are
  point estimates without intervals, on at most 300 groups each.
- **"Meaning-preserving" is judged by rules, not people.** The crops can cut off a detail the question is about.
  This is most likely on ai2d/mapqa, where labels sit at the edges, and on the vsr random crop. `rescale50`
  halves the effective resolution of chart text. The drops for those ops are an upper bound on pure
  sensitivity. The text rules are deliberately conservative (acronyms, numbers and case-colliding options are
  skipped).
- **ECE intervals are biased upward.** Bootstrap resamples duplicate rows, and ECE on a resample is biased
  upward, so the reported `ece_ci` can sit above the point estimate. Compare ECEs across families by their point
  estimates.
- **Balanced accuracy on choice sets** is over gold positions (see Setup); use plain accuracy there.
- **Option order uses the same logits mapping as training evaluation** (`collect_logits`, label order). A unit
  test checks that scoring a row under display order `o` equals physically reordering its choice options (and
  remapping the label).
- **Only the processor preprocessing path is exercised**, which is what this checkpoint uses. A checkpoint with
  `preprocess="gpu"` cannot batch the crops, whose size differs from the source image (see the module
  docstring).
- The per-row file stores calibrated probabilities (5 d.p.) and raw logits (4 d.p.), so any temperature can be
  re-applied offline.
