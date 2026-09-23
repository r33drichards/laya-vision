# ModernVBERT post-trained on The Cauldron

Run `modernvbert/cauldron-2ep` (`modal run --detach modal_app.py::finetune_long --backbone ModernVBERT/modernvbert --run-name cauldron-2ep --epochs 2 --max-passes 4 --val-datasets <official 3 + all 19 Cauldron holdouts>`), one NVIDIA A100 40 GB, 68 minutes of training plus 15 of evaluation. The raw log is `modernvbert-cauldron-metrics.json` next to this file.

- **Backbone:** [ModernVBERT/modernvbert](https://huggingface.co/ModernVBERT/modernvbert), bidirectional, `[MASK]` readout (see the README's ModernVBERT section).
- **Data:** the 19 closed-form Cauldron subsets from `prepare_cauldron`, 269,900 training questions after the 300-per-subset calibration holdout, sampled equally per subset and capped at 4 passes over any one of them.
- **Recipe:** identical to the released SmolVLM run otherwise: vision tower frozen, everything else trained, batch 32, LR 1.41e-4 head / 2.83e-5 backbone, cosine to 10%, proper-scoring-rule plus soft cross-entropy objective, per-type temperature scaling on the calibration holdout.
- **Best checkpoint:** step 16868 of 16869 (the end of epoch 2); mean per-set val accuracy over the four evals: 62.8% → 66.1% → 66.6% → 66.9%.
- **Temperatures (choice, score, noul):** 2.71, 1.00, 2.16. `score` is untrained (no ordinal data), as before.

## Official validation splits

These are the sets the README's SmolVLM table uses, so the two columns are directly comparable. The SmolVLM checkpoint was trained for 3 epochs on exactly these sets' train splits; this model saw their Cauldron train rows (the same images) as 3 of 19 subsets, at most 4 passes each.

| Set | n | Accuracy | ECE raw → calibrated | SmolVLM release accuracy | SmolVLM calibrated ECE |
|---|---|---|---|---|---|
| A-OKVQA (official val, 4-way `choice`) | 1138 | 65.2% | 0.214 → 0.064 | 61.8% | 0.123 |
| ScienceQA (official val, image subset) | 2097 | 79.0% | 0.114 → 0.058 | 86.6% | 0.034 |
| VQAv2 yes/no (re-split of official val, `noul`) | 5000 | 71.8% | 0.134 → 0.037 | 73.4% | 0.041 |
| **All 22 val sets** | 22886 | **71.8%** | 0.103 → **0.022** | | |

Option-order sensitivity on A-OKVQA (accuracy under the 4 cyclic shifts of the options): 65.2%, 66.2%, 65.5%, 65.6%, spread 1.0 points (SmolVLM: 0.7). Averaging 4 permutations at inference gives 66.4%.

## Cauldron holdouts

Five percent of each subset's rows, held out by row so no image is in both splits.

| Subset | Type | n | Accuracy | ECE raw → calibrated | Passes seen |
|---|---|---|---|---|---|
| ai2d | `choice` | 282 | 67.0% | 0.227 → 0.071 | 4.00 |
| aokvqa | `choice` | 552 | 65.8% | 0.200 → 0.071 | 4.00 |
| chartqa | `noul` | 41 | 46.3% | 0.314 → 0.230 | 3.96 |
| clevr | `noul` | 1718 | 60.7% | 0.045 → 0.039 | 1.22 |
| dvqa | `noul` | 1397 | 93.0% | 0.020 → 0.036 | 1.58 |
| figureqa | `noul` | 1928 | 73.0% | 0.049 → 0.036 | 1.07 |
| hateful_memes | `noul` | 456 | 76.3% | 0.162 → 0.044 | 4.00 |
| iconqa | `choice` | 606 | 81.8% | 0.083 → 0.068 | 3.68 |
| intergps | `choice` | 94 | 26.6% | 0.434 → 0.219 | 3.99 |
| mapqa | `noul` | 1598 | 62.8% | 0.024 → 0.026 | 1.33 |
| nlvr2 | `noul` | 906 | 69.9% | 0.164 → 0.061 | 2.30 |
| ocrvqa | `noul` | 996 | 85.6% | 0.068 → 0.076 | 2.26 |
| raven | both | 542 | 39.9% | 0.078 → 0.038 | 4.00 |
| scienceqa | `choice` | 301 | 82.1% | 0.076 → 0.081 | 4.00 |
| tqa | `choice` | 264 | 51.1% | 0.329 → 0.105 | 4.00 |
| visual7w | `choice` | 1729 | 75.4% | 0.089 → 0.145 | 1.19 |
| vqarad | `noul` | 62 | 62.9% | 0.204 → 0.128 | 3.99 |
| vqav2 | `noul` | 1019 | 71.3% | 0.139 → 0.037 | 2.06 |
| vsr | `noul` | 160 | 65.0% | 0.223 → 0.090 | 3.99 |

## Reading it

- **A-OKVQA is above the released SmolVLM model** (65.2% vs 61.8%) with far broader training data and fewer passes over A-OKVQA itself; ScienceQA and VQAv2 yes/no are below it (79.0% vs 86.6%, 71.8% vs 73.4%). The SmolVLM run made 12 passes over ScienceQA's 5.9k train rows; this one made 4 over 5.2k, mixed with 18 other subsets, so the ScienceQA gap is at least partly a data-mix choice, not a backbone one.
- **Calibration lands in the same place** after temperature scaling (overall ECE 0.022 vs 0.034), but the raw model is much more overconfident than SmolVLM's (temperatures of 2.7 and 2.2 versus about 1.3), and raw ECE grew every eval as the learning rate decayed. A run that anneals the cross-entropy weight (`w_ce_schedule="anneal"` in `laya.vlm_train.train`) or stops at 1.5 epochs would likely keep the raw calibration.
- **The bidirectional readout still shows about 1 point of option-order spread**, from position, not from causal masking; permutation averaging recovers it.
- **Weak spots are the reasoning sets:** InterGPS 27% (geometry), RAVEN 40% (8-way visual analogies), TQA 51%, CLEVR yes/no 61%. Document and chart reading is strong: DVQA 93%, OCR-VQA 86%, IconQA 82%.
- Accuracy was still rising at the end of epoch 2 on most sets; a third epoch or `--max-passes 6` is the obvious next run.

## Second run: 3 epochs with the cross-entropy anneal (`cauldron-3ep-anneal`)

Same data and recipe, `--epochs 3 --max-passes 6 --w-ce-schedule anneal`: the soft cross-entropy weight is held for the first 30% of training and decayed to 0 by 80%, so the run ends on the proper scoring rule alone. 94 minutes of training, 19 of evaluation. Best checkpoint at step 21085 of 25304 (2.5 epochs); the final eval at 3 epochs was 0.2 points lower. Temperatures (choice, noul): 3.11, 3.08. Log: `modernvbert-cauldron-3ep-anneal-metrics.json`.

| Set | 2ep acc | 2ep ECE raw → cal. | 3ep-anneal acc | 3ep-anneal ECE raw → cal. |
|---|---|---|---|---|
| A-OKVQA (official val) | 65.2% | 0.214 → 0.064 | 65.0% | 0.265 → 0.046 |
| ScienceQA (official val) | 79.0% | 0.114 → 0.058 | 77.4% | 0.170 → 0.065 |
| VQAv2 yes/no (official val re-split) | 71.8% | 0.134 → 0.037 | 72.1% | 0.166 → 0.021 |
| **All 22 val sets** | 71.8% | 0.103 → 0.022 | 72.1% | 0.127 → 0.030 |

A-OKVQA cyclic-shift spread: 1.0 points (2ep) vs 0.6 points (3ep-anneal); with 4 permutations averaged, 66.4% vs 65.6%.

Cauldron holdouts, same columns:

| Subset | 2ep acc | 2ep ECE raw → cal. | 3ep-anneal acc | 3ep-anneal ECE raw → cal. |
|---|---|---|---|---|
| ai2d | 67.0% | 0.227 → 0.071 | 66.7% | 0.265 → 0.154 |
| aokvqa | 65.8% | 0.200 → 0.071 | 63.0% | 0.281 → 0.091 |
| chartqa | 46.3% | 0.314 → 0.230 | 46.3% | 0.196 → 0.099 |
| clevr | 60.7% | 0.045 → 0.039 | 60.3% | 0.035 → 0.046 |
| dvqa | 93.0% | 0.020 → 0.036 | 93.7% | 0.026 → 0.068 |
| figureqa | 73.0% | 0.049 → 0.036 | 75.1% | 0.082 → 0.048 |
| hateful_memes | 76.3% | 0.162 → 0.044 | 75.7% | 0.211 → 0.064 |
| iconqa | 81.8% | 0.083 → 0.068 | 84.5% | 0.080 → 0.108 |
| intergps | 26.6% | 0.434 → 0.219 | 28.7% | 0.501 → 0.221 |
| mapqa | 62.8% | 0.024 → 0.026 | 60.8% | 0.035 → 0.030 |
| nlvr2 | 69.9% | 0.164 → 0.061 | 71.0% | 0.185 → 0.061 |
| ocrvqa | 85.6% | 0.068 → 0.076 | 85.8% | 0.080 → 0.092 |
| raven | 39.9% | 0.078 → 0.038 | 43.5% | 0.047 → 0.052 |
| scienceqa | 82.1% | 0.076 → 0.081 | 84.7% | 0.071 → 0.089 |
| tqa | 51.1% | 0.329 → 0.105 | 50.0% | 0.386 → 0.144 |
| visual7w | 75.4% | 0.089 → 0.145 | 76.1% | 0.105 → 0.165 |
| vqarad | 62.9% | 0.204 → 0.128 | 74.2% | 0.188 → 0.044 |
| vqav2 | 71.3% | 0.139 → 0.037 | 74.3% | 0.142 → 0.046 |
| vsr | 65.0% | 0.223 → 0.090 | 62.5% | 0.310 → 0.183 |

**What it says.** The two runs tie on accuracy and on calibrated ECE. The anneal did the opposite of what it was for: as the cross-entropy weight fell, raw ECE rose at every eval (A-OKVQA 0.09 at epoch 1 to 0.27 at 2.5), and the fitted temperatures grew from 2.7 / 2.2 to 3.1 / 3.1. On this backbone the proper-scoring-rule term alone is *more* overconfident than the mixed objective, so keep `w_ce_schedule="const"`. The third epoch bought a point or two on the Cauldron holdouts (IconQA, DVQA, FigureQA) and nothing on the official splits, which had flattened by epoch 2. The next lever is the data mix (more passes over the choice-type subsets, or dropping the yes/no-heavy chart sets that dominate the samples), not the schedule.
