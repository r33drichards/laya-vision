# Training data for `score` questions

Both released checkpoints mark `score` as untrained: every post-training set was `choice` or `noul`. This page describes the four rubric-scored image datasets `laya/rubric.py` converts into `score` records, the cleanup applied, and the two runs that add them to the Cauldron mix.

## What the head needs

A `score` question is `{"type": "score", "instructions": ..., "criteria": [level 0, level 1, ...]}`; the model renders each level as `level i: <text>` and reads one logit per level. Training takes the level index as `label`, or a probability per level as `target`. The loss (`laya.common.proper_reward`) already adds a ranked probability score for `score` questions, so ordinal structure is rewarded once data exists. Nothing in the model changes for this work; only data was added.

## Sources

| Prepared set | Source | Records (approx.) | Labels | Question |
|---|---|---|---|---|
| `score_vlfeedback` | [MMInstruction/VLFeedback](https://hf.co/datasets/MMInstruction/VLFeedback) | 80k images, up to 2 records each | GPT-4V, 1–5 | Helpfulness or visual faithfulness of a model's response to a question about the image. The question and response are the state text. |
| `score_ava` | [trojblue/AVA-aesthetics-10pct-min50-10bins](https://hf.co/datasets/trojblue/AVA-aesthetics-10pct-min50-10bins) | 20k train, 1k val | Human vote histograms, 1–10, at least 50 votes | Aesthetic quality of a photo, 5 levels. The histogram (collapsed 10 → 5) is the soft `target`. |
| `score_richhf` | [Exploration/richhf_18k_with_images](https://hf.co/datasets/Exploration/richhf_18k_with_images) | 15.8k train, 1k val, up to 2 records each | Three human raters, 5-point | Plausibility, prompt alignment, aesthetics or overall quality of a generated image. The prompt is the state text for alignment and overall. |
| `score_crisismmd` | [QCRI/CrisisMMD](https://hf.co/datasets/QCRI/CrisisMMD) (`damage`) | 2.5k train, 0.5k val | Human, 3 levels | Damage severity in a disaster photo. Image only. |

The MM-RLHF and LLaVA-Critic sets from the same search were left out: MM-RLHF's images are 30 GB of zips outside the dataset viewer, and LLaVA-Critic's "pointwise" config mixes 0–100 caption grades with pairwise two-assistant prompts under one schema.

The RichHF column names are misleading: `misalignment_score` is the alignment rating (it correlates +0.75 with `overall_score` and +0.8 with the share of prompt tokens labelled as depicted), so all four RichHF columns are read as "higher is better".

## Cleanup and labelling

- **Rubrics, not numbers.** Each level is a short clause in the style of `laya.presets` ("helpful: answers the question correctly and clearly"), so the head learns to read a rubric the way `predict` users write one. Levels stay under 14 words because each option is cut to 48 tokens and the whole question head to 256.
- **Several phrasings per question.** Every question has three instruction phrasings; one is drawn per record. The criteria stay fixed.
- **State text first, clipped.** Responses are capped at 1,200 characters and prompts at 400, on a word boundary, with the question before the response. The causal builder keeps the head of the state and drops the tail, so the question always survives.
- **Level balancing.** Ratings pile up at the top (VLFeedback helpfulness, RichHF) or the middle (AVA, where the argmax of the collapsed histogram is "average" for most photos). After streaming, no level in the train split may exceed 3× the median level count (`balance_levels`); the val split is untouched.
- **At most 2 records per image** for the multi-response and multi-aspect sets, sampled, so one photo does not become 8 rows.
- **Dropped signals.** VLFeedback's "Ethical Considerations" rating (almost always 5) and CrisisMMD's tweet text (so the level has to come from the image).
- **Splits.** AVA, RichHF and CrisisMMD use their upstream validation or dev split, capped at 1,000 rows. VLFeedback has none, so 5% of rows are held out by row.

## Preparing the data

```bash
modal run modal_app.py::prepare_score                    # all four, one container each -> /data/vqa/score_<name>
modal run modal_app.py::prepare_score --names ava,crisismmd
```

Each set is written as `/data/vqa/score_<name>/{train,val}.jsonl`, `images/`, `meta.json` (row counts, level histograms before and after balancing) and `_READY`.

## The runs

Two runs, same data and recipe as the Cauldron runs in [modernvbert-cauldron.md](../reference/results/modernvbert-cauldron.md), with the four score sets added to the 19 Cauldron subsets (group names expand in `--datasets`):

```bash
# SmolVLM (causal backbone)
modal run --detach modal_app.py::finetune_long --run-name cauldron-score-2ep --epochs 2 --max-passes 4 \
    --datasets cauldron,score --val-datasets vqa,cauldron,score

# ModernVBERT (bidirectional backbone)
modal run --detach modal_app.py::finetune_long --backbone ModernVBERT/modernvbert --run-name cauldron-score-2ep \
    --epochs 2 --max-passes 4 --datasets cauldron,score --val-datasets vqa,cauldron,score
```

With equal sampling over 23 sets the score sets get about 17% of the draws; `--mix score_ava=2` and friends reweight them. The calibration holdout (the last 300 train records per set) now contains `score` examples, so the `score` temperature is fitted rather than left at 1.0. `modal run modal_app.py::evaluate --run-name <run>/best` scores every prepared set, the score ones included.

## What to look at

- Val accuracy on the score sets is argmax accuracy over 3–5 levels and is a blunt measure for ordinal labels; the NLL column and the calibrated ECE say more. A follow-up is to add mean absolute level error to `metrics_from`.
- Whether the Cauldron holdouts and the official VQA splits hold their accuracy with the score sets mixed in.
- `try_model` with a `score` question against a photo: the released checkpoints answer it at chance.
