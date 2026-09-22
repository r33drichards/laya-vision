# First `score` results: Cauldron + rubric sets, both backbones

Three runs of `finetune_long` on the 19 closed-form Cauldron subsets plus the four rubric-scored sets from [score-data.md](score-data.md), one A100 40 GB each, launched 2026-09-22 00:30 UTC:

| Run | Backbone | Option attention | Trained | Best step | Mean val acc (26 sets) |
|---|---|---|---|---|---|
| `smolvlm/cauldron-score-2ep` | SmolVLM-256M | causal | 90 min, 16,408 steps, 1.13 epochs | 14,556 | 73.5% |
| `modernvbert/cauldron-score-2ep` | ModernVBERT-250M | n/a (bidirectional readout) | 90 min, 19,189 steps, 1.32 epochs | 19,189 | 66.2% |
| `smolvlm/cauldron-score-2ep-bidir` | SmolVLM-256M | bidirectional option block | see below | | |

All three asked for 2 epochs (29,110 steps of batch 32 over 465,760 examples) but stopped at `finetune_long`'s default `--max-minutes 90`, so the cosine schedule was cut at about 60% of peak learning rate. Equal sampling over 23 sets, `--max-passes 4`, `w_ce_schedule="const"`, the released recipe otherwise. Raw logs: `smolvlm-cauldron-score-metrics.json`, `modernvbert-cauldron-score-metrics.json`.

## The `score` head, first time trained

Val splits are the upstream validation or dev splits (AVA, RichHF, CrisisMMD) or a 5% row holdout (VLFeedback). "Majority" is the share of the most common level in the val split; a head that ignores the image scores that.

| Set | Levels | Majority | SmolVLM acc | SmolVLM ECE raw → cal. | ModernVBERT acc | ModernVBERT ECE raw → cal. |
|---|---|---|---|---|---|---|
| score_vlfeedback (response helpfulness / faithfulness) | 5 | 27.5% | 50.0% | 0.044 → 0.093 | 50.0% | 0.066 → 0.076 |
| score_richhf (generated-image plausibility / alignment / aesthetics / overall) | 5 | 44.0% | 57.1% | 0.043 → 0.079 | 57.5% | 0.063 → 0.066 |
| score_crisismmd (damage) | 3 | 62.8% | 70.5% | 0.154 → 0.067 | 70.5% | 0.185 → 0.097 |
| score_ava (photo aesthetics, soft vote targets) | 5 | 84.8% | 67.6% | 0.146 → 0.256 | 59.0% | 0.040 → 0.150 |

Ordinal metrics (`mae`: absolute difference between the model's and the target's expected level; `xent`: cross-entropy against the soft target), from `evaluate --datasets score` on the saved `best/` checkpoints:

_pending_

Reading it:

- **The head learns rubrics.** Both backbones reach 50% over 5 levels on VLFeedback (majority 27.5%) and 57% on RichHF (majority 44%) after seeing VLFeedback only 0.2 times and RichHF once. VLFeedback was still climbing at every eval (42% → 50%). The two backbones are within a point of each other on every score set, unlike the Cauldron holdouts, so the rubric data is not where the backbones differ.
- **Raw calibration on the rubric sets is already good** (ECE 0.04 to 0.07) and the fitted `score` temperature of about 1.7 makes it *worse* on VLFeedback, RichHF and AVA while fixing CrisisMMD. One temperature per question type is fit on the pooled calibration holdout of all four sets; the head's confidence differs by set (and by level count), so a per-option-count temperature (`temperature_by_options` in the agent config) or a per-dataset one is the fix.
- **AVA argmax accuracy is below the majority baseline by design.** The target is the vote histogram collapsed to 5 levels, so the head is trained to spread probability the way voters did; argmax accuracy and NLL against the argmax label then look bad. `mae` and `xent` are the right numbers for it (above).
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

_pending: results_

## Qualitative check

`try_model` with four `score` questions on unseen val images (two CrisisMMD dev photos, two AVA val photos):

_pending_

## Next

- Rerun with `--max-minutes 240` (or continue from `best/` with `--init-from`) so the schedule completes; every score set was still improving.
- Weight VLFeedback up (`--mix score_vlfeedback=3`): it is the largest and most rubric-like set and got 0.2 passes.
- Per-option-count or per-dataset temperatures for `score`.
- Report `mae` / `xent` in the training-time evals too (they are in `metrics_from` now, so the next run's `metrics.json` will carry them).
