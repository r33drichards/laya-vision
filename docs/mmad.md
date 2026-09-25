# MMAD: industrial anomaly inspection

[MMAD](https://github.com/jam-cc/MMAD) (Jiang et al., ICLR 2025) asks 39,670 multiple-choice questions about
8,366 industrial images, and grades a model as a factory quality inspector. `modal_mmad.py` runs the
SmolVLM decision model on it.

```bash
modal run --detach modal_mmad.py::prepare                           # stage 28.3 GB of images into laya-datasets
modal run modal_mmad.py::bench --run-name all3-3ep/best --limit 50  # smoke test
modal run --detach modal_mmad.py::bench --run-name all3-3ep/best    # 1-shot, Anomaly Detection
modal run modal_mmad.py::report --answers answers_1_shot_all3-3ep-best_AnomalyDetection
```

The default is the **Anomaly Detection** subtask alone — "Is there any defect in the object?", 8,297
questions, one per image. `--question-types ""` runs all nine subtasks; read
[Why detection first](#why-detection-first) before you do.

## Why this benchmark fits the model

Every MMAD question is multiple choice over 2 or 4 options, which is exactly a `choice` question. So the
benchmark maps onto `predict(state, questions)` with nothing in between:

| MMAD | Laya Vision |
|---|---|
| the query image (plus a normal template, 1-shot) | `state = {"images": [...]}` |
| `"Question"` | `instructions` |
| `"Options": {"A": ..., "B": ...}` | `criteria`, in the benchmark's own order |
| the answer letter | `argmax` over the options, mapped back by position |

That removes the part of MMAD that is usually noisy. The reference scripts prompt a chat model with
"Answer with the option's letter", then recover the letter with a regex, fall back to fuzzy-matching the
reply against the option strings, and retry whenever the reply contains "sorry" or "cannot assist". None of
that applies here: there is no text to generate and no string to parse, so a question can never be lost to a
refusal or a malformed answer. Every one of the 39,670 questions gets a real answer.

The image is encoded once per image and reused for all of its questions (at most 5), so each image is a
single forward pass.

## Why detection first

MMAD's reference scripts do not ask an image's five questions in one go. For question *i* they make a
separate API call carrying questions 1..*i*, let the model emit a numbered list of *i* answers, and keep
only the last one. So a chat model answering question 3 has just written its own answers to 1 and 2 in that
same completion and is attending to them. The follow-ups are phrased to presuppose the defect — *"There is
a defect in the object. What is the type of the defect?"* — so a model that has just written *"1. Answer:
Yes"* is primed consistently.

This model answers each question independently in one forward pass and has no mechanism for that
conditioning. Running the follow-up subtasks here would compare two different protocols.

**Anomaly Detection is exempt.** It is always question index 0, exactly one per image, present on 8,297 of
the 8,366 images. The reference call for it therefore carries a single question and no prior context, which
is exactly what this harness does. That makes the detection number directly comparable to the published
1-shot results, and it is the default.

## Setup

- **1-shot**, the benchmark's headline setting: one known-good template image of the same product is
  prepended to the query image, taken from MMAD's own `random_templates` list, as their scripts default to.
  The model is told which is which through the state text, the analogue of the reference prompt's
  "The last image is the query image".
- **Option order is the benchmark's.** MMAD balances it already: of the yes/no detection questions, 4,195
  put "Yes." at A and 4,077 put it at B. No de-biasing is applied. `--n-permutations K` averages logits over
  K option orders if you want that bias measured rather than assumed.
- **Scoring is MMAD's own** `evaluation/examples/helper/summary.py`, downloaded with the benchmark instead
  of copied into this repo (MMAD ships no licence). Accuracy is reported per sub-dataset and question type,
  with Anomaly Detection as balanced accuracy over normal and defective images, plus recall, precision and F1.
- **`tests/test_mmad.py`** covers the letter mapping and the calibration arithmetic without a GPU, Modal or
  the 28 GB download. The mapping test matters most: MMAD grades a letter, so an off-by-one between an
  option's text and its letter would score as a plausible accuracy rather than crash.

### Two details worth knowing

**Preprocessing.** In 1-shot there are two images, and a query and its template often differ in resolution.
The device-side path in `laya/preprocess.py` stacks images into one tensor and so requires a shared
resolution; the Hugging Face processor resizes each independently. `--prep-backend auto` (the default)
therefore selects the processor whenever more than one image is in play. This costs the released checkpoint
nothing, since its config predates the device-side keys and it loads on the processor path anyway.

**Calibration.** MMAD's scorer compares letters and throws the probabilities away. `report` adds the ECE and
the detection AUROC computed from them. The checkpoint's fitted choice temperature (3.32) rescales
probabilities monotonically, so it cannot change an `argmax` — accuracy is unaffected by it, and the ECE
reported is the calibrated one.

## A laptop spot-check

`examples/mmad_local.py` runs a small sample on this machine with no Modal and no 28 GB download, by
pulling individual images from Hugging Face:

```bash
uv run examples/mmad_local.py --n 200        # DS-MVTec, 1-shot, Anomaly Detection
```

It shares `laya/mmad.py` with the Modal path, so the two cannot answer a question differently. It is a
spot-check, not a benchmark result — that revision of the dataset is an incomplete upload and serves only
12.8% of the 1-shot detection questions, which is why it defaults to DS-MVTec (40.2% covered) rather than
quietly reporting a "cross-dataset" number that is mostly one dataset. The script's docstring has the
per-subset coverage.

## Results

Checkpoint `all3-3ep/best`, 1-shot, Anomaly Detection, all 8,297 questions, 43.1 min on one L4, 0 skipped.
Scored by MMAD's own `caculate_accuracy_mmad`.

| sub-dataset | balanced accuracy | overkill | miss |
|---|---|---|---|
| MVTec-AD (incl. DS-MVTec) | 55.37 | 28.17 | 61.09 |
| MVTec-LOCO | 50.26 | 4.01 | 95.46 |
| GoodsAD | 49.99 | 2.05 | 97.98 |
| VisA | 49.56 | 40.62 | 60.27 |
| **Average** | **51.29** | 18.71 | 78.70 |

Recall 21.30, precision 63.99, F1 27.67. AUROC 0.5327. ECE 0.3954.

**The model does not do this task.** Chance is 50.0 and the average is 51.29 — it ranks 22nd of 23 against
MMAD's published models, ahead only of llava-v1.5-13b, which answers "defect" to every image and therefore
scores exactly chance. On MVTec-LOCO and GoodsAD it misses 95% and 98% of defects respectively while
raising almost no false alarms, which is the signature of answering "no defect" to essentially everything.
Only the MVTec-AD group is meaningfully above chance, at 55.37 (20th of 23).

Two things are worth keeping from the run:

* **AUROC 0.5327.** A threshold-free measure, so a constant predictor is pinned to 0.5. At 0.533 the
  ranking signal is real but negligible. A DS-MVTec-only laptop sample had suggested 0.636; that did not
  survive the other three datasets, which is a caution about extrapolating from the one sub-dataset the
  Hugging Face file revision covers well.
* **ECE 0.3954**, against the 0.034 this checkpoint reaches on its own A-OKVQA/ScienceQA/VQAv2 validation
  splits. Temperature calibration fitted in-distribution does not survive the move to industrial imagery,
  and the model is confidently wrong here rather than uncertain.

Retuning the decision threshold does not rescue it: pooled balanced accuracy is 0.5156 at the 0.5 the
benchmark scores, 0.5325 at the best threshold chosen on the same data, and 0.5288 under 5-fold
cross-validation — about a point, consistent with an AUROC that close to chance.

Note the two aggregations differ on purpose. MMAD's "Average" row (51.29) is the mean of the four
sub-dataset rows; the 0.5156 in the threshold block is pooled over all 8,297 questions. The sub-datasets
differ in size and class balance, so macro and micro disagree.

`report` also prints where the run lands among MMAD's 22 published models, read from the `all_result/`
CSVs that ship with the benchmark, for both the `Average` and `DS-MVTec` rows. Read that table with the
overkill and miss columns next to it: several published baselines are effectively one-class predictors
(llava-v1.5-13b answers "defect" to everything, for 100% overkill, 0% miss and a balanced accuracy of
exactly 50.00), so a mid-table score is not on its own evidence of discrimination. AUROC is the better
check, since a constant predictor is pinned to 0.5 there.

## Files

- `laya/mmad.py` — the mapping: questions, letter recovery, calibration. No Modal, no torch at import.
- `modal_mmad.py` — `prepare` (stage), `bench` (inference), `report` (score + leaderboard).
- `examples/mmad_local.py` — the laptop sample.
- `tests/test_mmad.py` — 14 tests over the mapping and the calibration arithmetic.
- `/data/mmad/MMAD-main/` — the benchmark, laid out exactly as its own scripts expect.
- `/data/mmad/results/<tag>.jsonl` — one line per question, written as the run goes so it can resume.
- `/data/mmad/results/<tag>.json` — the same answers as the list MMAD's scorer reads.
- `/data/mmad/results/<tag>_accuracy.csv` — the accuracy table, written by MMAD's scorer.
