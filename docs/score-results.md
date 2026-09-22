# First `score` results: Cauldron + rubric sets, both backbones

Three runs of `finetune_long` on the 19 closed-form Cauldron subsets plus the four rubric-scored sets from [score-data.md](score-data.md), one A100 40 GB each, launched 2026-09-22 00:30 UTC:

| Run | Backbone | Option attention | Trained | Best step | Mean val acc (26 sets) |
|---|---|---|---|---|---|
| `smolvlm/cauldron-score-2ep` | SmolVLM-256M | causal | 90 min, 16,408 steps, 1.13 epochs | 14,556 | 73.5% |
| `modernvbert/cauldron-score-2ep` | ModernVBERT-250M | n/a (bidirectional readout) | 90 min, 19,189 steps, 1.32 epochs | 19,189 | 66.2% |
| `smolvlm/cauldron-score-2ep-bidir` | SmolVLM-256M | bidirectional option block | 90 min, 16,692 steps, 1.15 epochs | 14,556 | 74.2% |

All three asked for 2 epochs (29,110 steps of batch 32 over 465,760 examples) but stopped at `finetune_long`'s default `--max-minutes 90`, so the cosine schedule was cut at about 60% of peak learning rate. Equal sampling over 23 sets, `--max-passes 4`, `w_ce_schedule="const"`, the released recipe otherwise. Raw logs: `smolvlm-cauldron-score-metrics.json`, `modernvbert-cauldron-score-metrics.json`, `smolvlm-cauldron-score-bidir-metrics.json`.

## The `score` head, first time trained

Val splits are the upstream validation or dev splits (AVA, RichHF, CrisisMMD) or a 5% row holdout (VLFeedback). "Majority" is the share of the most common level in the val split; a head that ignores the image scores that.

| Set | Levels | Majority | SmolVLM acc | SmolVLM ECE raw → cal. | ModernVBERT acc | ModernVBERT ECE raw → cal. |
|---|---|---|---|---|---|---|
| score_vlfeedback (response helpfulness / faithfulness) | 5 | 27.5% | 50.0% | 0.044 → 0.093 | 50.0% | 0.066 → 0.076 |
| score_richhf (generated-image plausibility / alignment / aesthetics / overall) | 5 | 44.0% | 57.1% | 0.043 → 0.079 | 57.5% | 0.063 → 0.066 |
| score_crisismmd (damage) | 3 | 62.8% | 70.5% | 0.154 → 0.067 | 70.5% | 0.185 → 0.097 |
| score_ava (photo aesthetics, soft vote targets) | 5 | 84.8% | 67.6% | 0.146 → 0.256 | 59.0% | 0.040 → 0.150 |

Ordinal metrics from `evaluate --datasets score` on the saved `best/` checkpoints, raw logits (temperature 1). `mae` is the absolute difference between the model's expected level and the target's, in levels; `xent` is the cross-entropy against the (soft) target. "Prior" is a model that always predicts the val split's mean level distribution, the floor any image-reading head must beat; for AVA the target's own entropy (1.098) is the lowest `xent` possible.

| Set | Prior mae | SmolVLM mae | ModernVBERT mae | Prior xent | SmolVLM xent | ModernVBERT xent |
|---|---|---|---|---|---|---|
| score_vlfeedback | 1.369 | **0.920** | **0.896** | 1.560 | **1.281** | **1.256** |
| score_richhf | 0.663 | **0.508** | **0.504** | 1.168 | **0.973** | **0.958** |
| score_crisismmd | 0.636 | **0.357** | **0.346** | 0.904 | 0.881 | 0.911 |
| score_ava | 0.293 | 0.366 | 0.463 | 1.233 | 1.364 | 1.492 |

VLFeedback, RichHF and CrisisMMD beat the prior clearly on expected level: a third of a level off on 5-level VLFeedback relative to guessing the prior, and the damage level is within 0.35 of the truth on average. **AVA does not beat the prior on either metric.** The cause is in the data prep, not the head: `balance_levels` capped the "average" level in the *train* split at 3x the median level count (17,620 → 2,109 rows), while the val split keeps AVA's natural distribution (52% "average"), so the head learned a flatter, more extreme-leaning histogram than the voters actually produce. For a soft-target set the balance should be off (`--balance 0` for AVA in `prepare_score`), or applied to val too. AVA's temperature-scaled `xent` (1.30 / 1.37) is closer to the prior for the same reason: flattening the over-spread prediction helps it.

Reading it:

- **The head learns rubrics.** Both backbones reach 50% over 5 levels on VLFeedback (majority 27.5%) and 57% on RichHF (majority 44%) after seeing VLFeedback only 0.2 times and RichHF once. VLFeedback was still climbing at every eval (42% → 50%). The two backbones are within a point of each other on every score set, unlike the Cauldron holdouts, so the rubric data is not where the backbones differ.
- **Raw calibration on the rubric sets is already good** (ECE 0.04 to 0.07) and the fitted `score` temperature of about 1.7 makes it *worse* on VLFeedback, RichHF and AVA while fixing CrisisMMD. One temperature per question type is fit on the pooled calibration holdout of all four sets; the head's confidence differs by set (and by level count), so a per-option-count temperature (`temperature_by_options` in the agent config) or a per-dataset one is the fix.
- **AVA is the one set that did not work**, and the ordinal table above says why: the train-only level balancing changed the target distribution. Argmax accuracy is also the wrong measure for a vote-histogram target; `mae` and `xent` are the ones to watch once the prep is fixed.
- **Temperatures (choice, score, noul):** SmolVLM 2.09 / 1.69 / 1.75; ModernVBERT 2.58 / 1.65 / 2.32. Both are more overconfident than the released SmolVLM checkpoint (about 1.3), as in the earlier Cauldron runs, and the truncated schedule kept the learning rate high through the end.

## Did the extra data cost the other questions?

Official splits, this run vs the earlier ModernVBERT Cauldron-only run (`cauldron-2ep`, 68 min, 2 full epochs of 270k) and the released SmolVLM model (3 epochs on the three official train splits):

| Set | SmolVLM cauldron-score | ModernVBERT cauldron-score | ModernVBERT cauldron-2ep | SmolVLM release |
|---|---|---|---|---|
| A-OKVQA (official val) | 60.0% | 62.8% | 65.2% | 61.8% |
| ScienceQA (official val, image subset) | 83.2% | 76.4% | 79.0% | 86.6% |
| VQAv2 yes/no (re-split of official val) | 72.6% | 71.5% | 71.8% | 73.4% |
| All val sets, calibrated ECE | 0.029 | 0.025 | 0.022 | 0.034 |

ModernVBERT is 2 to 3 points under its Cauldron-only run on A-OKVQA and ScienceQA. Some of that is the 17% of batches now spent on score data and some is the schedule cut at 1.3 epochs; the per-set pass counts on the capped Cauldron subsets are the same 4.0. SmolVLM has no Cauldron-only run to compare against; against the released model it trails on ScienceQA (83.2% vs 86.6%, 4 passes vs 12 over the ScienceQA train rows) and matches on A-OKVQA and VQAv2 yes/no with far broader data.

Cauldron holdouts where the backbones differ most (SmolVLM / ModernVBERT): IconQA 92.1% / 82.7%, RAVEN 80.3% / 42.3%, TQA 73.9% / 51.9%, VSR 90.0% / 65.6%, hateful memes 91.0% / 73.2%, NLVR2 79.6% / 72.3%. ModernVBERT is ahead only on A-OKVQA, DVQA (92.5% vs 90.1%) and ChartQA (41 rows). The 7-point gap in the mean is the causal, instruction-pretrained backbone winning the reasoning-heavy sets, the same pattern the earlier Cauldron doc noted for RAVEN and TQA.

## Bidirectional option attention on SmolVLM

`--option-attention bidirectional` passes a 4D mask that lets the option block attend to itself in both directions while the image, state and question stay causal (`laya.vlm.option_block_mask`). Checked on the pretrained weights before the run: changing option C's text moved option A's logit by about 0.001 under the causal mask (only through the head layers) and by about 0.05 under the bidirectional one, and the mask is strictly causal before the block, fully connected inside it and sees everything before it.

Same data, schedule and 90-minute cap as the causal SmolVLM run; both have their best checkpoint at step 14,556 (epoch 1.0), so the comparison is at equal steps.

| | Causal SmolVLM | Bidirectional option block |
|---|---|---|
| Mean val acc, 26 sets | 73.5% | **74.2%** |
| A-OKVQA (official) | 60.0% | **61.1%** |
| A-OKVQA, 4 cyclic option shifts | 60.0 / 60.6 / 61.3 / 61.0, spread 1.3 | 61.1 / 61.0 / 61.9 / 60.8, spread **1.1** |
| A-OKVQA, 4 permutations averaged | 60.5% | 61.7% |
| ScienceQA (official) | **83.2%** | 82.5% |
| VQAv2 yes/no | 72.6% | 72.3% |
| TQA holdout | 73.9% | **76.9%** |
| InterGPS holdout | 27.7% | **31.9%** |
| NLVR2 holdout | 79.6% | **81.8%** |
| RAVEN / IconQA / VSR holdouts | 80.3 / 92.1 / 90.0 | 80.4 / 91.9 / 90.6 |
| score_vlfeedback / richhf / crisismmd | 50.0 / 57.1 / 70.5 | 49.4 / 56.7 / 70.5 |
| Temperatures (choice, score, noul) | 2.09 / 1.69 / 1.75 | 2.27 / **3.77** / 1.83 |
| Calibrated ECE, all sets | 0.029 | 0.050 |

Letting the options see each other buys a consistent 0.7 points on the mean and 1 to 4 points on the sets where options must be compared (TQA, InterGPS, NLVR2, A-OKVQA), costs nothing on the score sets, and changes the order spread only from 1.3 to 1.1 points: a shared option block does not remove position bias by itself, because the terminator of each option is still at a different position and the pretrained backbone had 90 minutes to adapt to a mask it never saw. ModernVBERT's spread on the same check is 1.4, so all three sit at the same noise level on 1,138 rows. The one clear cost is the `score` head's raw confidence: its fitted temperature is 3.8 against 1.7 for the causal run, and the calibrated ECE over all sets is worse (0.050 vs 0.029). The cheapest follow-up is the same run with the schedule completed; the mask is a free win on accuracy at equal steps.

## Qualitative check

`try_model` on unseen val images: a CrisisMMD dev photo labelled severe damage (California wildfires), one labelled little or no damage (Hurricane Irma), and two AVA val photos with mean votes 7.0 and 3.5. The answer is the expected level; `conf` is the agent's confidence (1 minus normalised entropy). "Urgency" is a rubric neither model was trained on.

| Question (levels) | Image | SmolVLM | ModernVBERT |
|---|---|---|---|
| damage (0 none, 1 mild, 2 severe) | wildfire, severe | **1.89** (conf 0.67) | **1.93** (conf 0.80) |
| | hurricane, no damage | 1.36 (0.16) | **0.97** (0.41) |
| | AVA 7.0 photo | 0.34 | 0.52 |
| | AVA 3.5 photo | 0.43 | 0.65 |
| aesthetics (0 very poor .. 4 excellent) | AVA 7.0 photo | **2.48** | **2.77** |
| | AVA 3.5 photo | 1.93 | 1.97 |
| | wildfire | 1.81 | 1.94 |
| urgency for emergency services (0 .. 4), untrained rubric | wildfire | 2.64 | **3.36** |
| | hurricane, no damage | 2.66 | 1.88 |
| | AVA photos | 1.68 / 1.93 | 1.12 / 1.29 |

Both models order the pairs the right way on the trained rubrics: severe over no damage, the 7.0 photo over the 3.5 photo, and near-zero damage on the ordinary photos. The AVA gap is small (about half a level), consistent with the soft targets and the val majority sitting at "average". On the untrained urgency rubric ModernVBERT transfers the damage signal (3.4 for the wildfire vs 1.9 for the calm photo, 1.1 to 1.3 for the AVA photos) while SmolVLM gives the two disaster photos the same 2.6 and only separates them from the ordinary photos. Confidences on 5-level questions are low (0.05 to 0.33) because a spread over adjacent levels is what the ranked probability score rewards; the expected level is the number to use.

## Next

- Rerun with `--max-minutes 240` (or continue from `best/` with `--init-from`) so the schedule completes; every score set was still improving.
- Weight VLFeedback up (`--mix score_vlfeedback=3`): it is the largest and most rubric-like set and got 0.2 passes.
- AVA has been re-prepared without train-only balancing (`prepare_score --names ava --balance 0`, 20,437 train rows); the three checkpoints above were trained on the balanced version, so the next run picks it up.
- Per-option-count or per-dataset temperatures for `score`.
- Report `mae` / `xent` in the training-time evals too (they are in `metrics_from` now, so the next run's `metrics.json` will carry them).
