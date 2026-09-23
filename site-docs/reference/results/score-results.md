# First `score` results: Cauldron + rubric sets, both backbones

Three runs of `finetune_long` on the 19 closed-form Cauldron subsets plus the four rubric-scored sets from [score-data.md](score-data.md), one A100 40 GB each, launched 2026-09-22 00:30 UTC:

| Run | Backbone | Option attention | Trained | Best step | Mean val acc (26 sets) |
|---|---|---|---|---|---|
| `smolvlm/cauldron-score-2ep` | SmolVLM-256M | causal | 90 min, 16,408 steps, 1.13 epochs | 14,556 | 73.5% |
| `modernvbert/cauldron-score-2ep` | ModernVBERT-250M | n/a (bidirectional readout) | 90 min, 19,189 steps, 1.32 epochs | 19,189 | 66.2% |
| `smolvlm/cauldron-score-2ep-bidir` | SmolVLM-256M | bidirectional option block | 90 min, 16,692 steps, 1.15 epochs | 14,556 | 74.2% |
| `smolvlm/cauldron-score-2ep-bidir-full` | SmolVLM-256M | bidirectional option block | 180 min, 30,080 steps, 2.0 epochs (`--max-minutes 240 --mix score_vlfeedback=3`, unbalanced AVA) | 30,080 | **75.1%** |

The first three asked for 2 epochs (29,110 steps of batch 32 over 465,760 examples) but stopped at `finetune_long`'s default `--max-minutes 90`, so the cosine schedule was cut at about 60% of peak learning rate; the fourth run completed its schedule (see the last section). Equal sampling over 23 sets, `--max-passes 4`, `w_ce_schedule="const"`, the released recipe otherwise. Raw logs: `smolvlm-cauldron-score-metrics.json`, `modernvbert-cauldron-score-metrics.json`, `smolvlm-cauldron-score-bidir-metrics.json`, `smolvlm-cauldron-score-bidir-full-metrics.json`.

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

## The full-schedule run

`cauldron-score-2ep-bidir-full`: the bidirectional SmolVLM recipe with the cap raised to 240 minutes, VLFeedback drawn 3x as often (12% of batches, 1.07 passes instead of 0.2) and the re-prepared, unbalanced AVA (20,137 train rows, 2.8 passes). 30,080 steps in 180 minutes, the cosine schedule completed; the best checkpoint is the last one. Temperatures (choice, score, noul): 2.20 / **1.37** / 2.13.

| | Causal SmolVLM, 90 min | Bidir, 90 min | **Bidir, full** | Prior only |
|---|---|---|---|---|
| Mean of per-set val acc, 26 sets | 73.5% | 74.2% | **75.1%** | |
| Pooled val acc, 28,405 rows | 73.1% | 73.0% | **74.1%** | |
| A-OKVQA (official) | 60.0% | 61.1% | 60.0% | 25% |
| A-OKVQA cyclic-shift spread | 1.3 | 1.1 | 1.4 | |
| ScienceQA (official) | 83.2% | 82.5% | 82.8% | |
| VQAv2 yes/no | 72.6% | 72.3% | 72.4% | |
| IconQA / RAVEN / TQA / InterGPS holdouts | 92.1 / 80.3 / 73.9 / 27.7 | 91.9 / 80.4 / 76.9 / 31.9 | **93.7** / 77.1 / 73.1 / **34.0** | |
| score_vlfeedback acc / mae / xent | 50.0% / 0.920 / 1.281 | 49.4% | **53.9% / 0.796 / 1.147** | 27.5% / 1.369 / 1.560 |
| score_richhf acc / mae / xent | 57.1% / 0.508 / 0.973 | 56.7% | 57.3% / 0.497 / 0.966 | 44% / 0.663 / 1.168 |
| score_crisismmd acc / mae / xent | 70.5% / 0.357 / 0.881 | 70.5% | 69.0% / 0.376 / 0.914 | 62.8% / 0.636 / 0.904 |
| score_ava acc / mae / xent | 67.6% / 0.366 / 1.364 | 62.8% | 84.7% / **0.253** / **1.212** | 84.8% / 0.293 / 1.233 |
| Calibrated ECE, all sets | 0.029 | 0.050 | 0.034 | |

- **VLFeedback, the rubric set closest to how `predict` is used, gained the most**: 54% over 5 levels and 0.80 levels of expected error, from 50% and 0.92, with raw ECE 0.042 before any temperature. Five times the passes bought about 4 points; it was still rising at the last eval.
- **AVA now beats the prior** on expected level (0.253 vs 0.293) and on cross-entropy against the vote histogram (1.212 vs 1.233, with 1.098 the floor), which confirms the diagnosis above: the train-only balancing, not the head, was the problem. Its raw ECE looks terrible (0.31) because ECE is computed on the argmax label of a soft target; ignore it for this set.
- **RichHF and CrisisMMD are flat** within noise; CrisisMMD has 529 val rows and 4 passes over 2,168 train rows either way.
- **The `score` temperature came down to 1.37** from 3.8 in the 90-minute bidirectional run: the completed schedule, not the mask, was behind that overconfidence. Calibrated ECE over all sets (0.034) sits between the causal run and the truncated bidirectional one.
- **The official VQA splits did not move** (A-OKVQA 60.0%, ScienceQA 82.8%, VQAv2 72.4%) and the option-order spread is still 1.4 points: the bidirectional option block is worth about a point on the reasoning holdouts and nothing on order robustness. The mean gain of the full run over the causal one (+1.6) is mostly the score sets and IconQA, DVQA, FigureQA finishing their schedule.

The checkpoint is `/ckpt/smolvlm/cauldron-score-2ep-bidir-full/best` on the `laya-checkpoints` volume and is published as [thaitea/laya-vision-smolvlm-256m-score](https://huggingface.co/thaitea/laya-vision-smolvlm-256m-score); it is the first Laya Vision checkpoint whose `score` answers mean something.

## A third epoch from the published checkpoint

`cauldron-score-3ep-bidir-vlf5`: `--init-from cauldron-score-2ep-bidir-full/best --epochs 1 --max-passes 2 --mix score_vlfeedback=5`, so one more pass over the 481k examples with VLFeedback at 20% of the draws (0.77 further passes), a fresh warmup and cosine decay on top of the converged model. 15,040 steps in 89 minutes; best checkpoint at the end. Temperatures (choice, score, noul): 2.51 / 1.41 / 2.33. Log: `smolvlm-cauldron-score-3ep-vlf5-metrics.json`.

| | Published (2 epochs) | + 1 epoch, VLFeedback x5 | Prior only |
|---|---|---|---|
| Mean of per-set val acc, 26 sets | 75.1% | **75.2%** | |
| A-OKVQA / ScienceQA / VQAv2 yes-no | 60.0 / 82.8 / 72.4 | 60.6 / **83.8** / 72.1 | |
| A-OKVQA cyclic-shift spread | 1.4 | **0.3** | |
| IconQA / RAVEN / TQA / InterGPS | 93.7 / 77.1 / 73.1 / 34.0 | 93.1 / 78.2 / 72.0 / 36.2 | |
| score_vlfeedback acc / mae | 53.9% / 0.796 | **54.4% / 0.771** | 27.5% / 1.369 |
| score_richhf acc / mae | 57.3% / 0.497 | 57.9% / 0.496 | 44% / 0.663 |
| score_crisismmd acc / mae | **69.0% / 0.376** | 67.1% / 0.387 | 62.8% / 0.636 |
| score_ava mae / xent | 0.253 / 1.212 | **0.242 / 1.204** | 0.293 / 1.233 |
| Calibrated ECE, all sets | 0.034 | 0.034 | |

Half a point on VLFeedback, a point on ScienceQA, two points lost on CrisisMMD (529 rows, so within noise), and a more overconfident raw model (choice temperature 2.5 vs 2.2). The one striking number is the A-OKVQA option-order spread, 0.3 points against 1.4 for every earlier run; on 1,138 rows a single run cannot separate that from luck, but it is what the bidirectional option block was supposed to deliver once the backbone had enough steps under the new mask. The rubric sets have flattened: VLFeedback gained 4 points from the first extra 0.5 passes and 0.5 from the next 0.8. **The published checkpoint stays as is**; this one is `/ckpt/smolvlm/cauldron-score-3ep-bidir-vlf5/best` on the volume if anyone wants the order-spread result checked on more rows.

## Next

- Done above: the completed schedule, VLFeedback x3, the unbalanced AVA and a third epoch with VLFeedback x5. More passes over the same rubric data are flat now; the next gains need new rubric data (MM-RLHF's human ratings, LLaVA-Critic's 0-100 caption grades) or per-dataset `score` temperatures.
- Re-check the 0.3-point order spread of the third-epoch checkpoint on the Cauldron A-OKVQA holdout and ScienceQA (`evaluate` with cyclic orders) before drawing a conclusion from it.
- Published: `cauldron-score-2ep-bidir-full/best` is [thaitea/laya-vision-smolvlm-256m-score](https://huggingface.co/thaitea/laya-vision-smolvlm-256m-score), model card in `hf_model_card_score.md`; the same weights are also at [thaitea/laya-vision](https://huggingface.co/thaitea/laya-vision), the moving "latest recommended" repo.
- Done: `cauldron-score-3ep-bidir-vlf5`, one more epoch from that checkpoint with VLFeedback drawn 5x; see the last section. Diminishing returns, the published checkpoint stays.
- Per-option-count or per-dataset temperatures for `score`.
- Report `mae` / `xent` in the training-time evals too (they are in `metrics_from` now, so the next run's `metrics.json` will carry them).
