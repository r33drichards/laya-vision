# Best-of-N with Laya as the verifier

Can a checkpoint rerank candidate answers? For each group of candidates, Laya scores every candidate. A best-of-N
system draws N of them and keeps the one with the highest score. The metric is
[`laya.bon.best_of_n`](https://github.com/r33drichards/laya-vision/blob/main/laya/bon.py). It is the exact
expected reward over a uniformly random N-subset, with score ties broken uniformly, so no subsets are sampled.
It comes with two reference points: a random pick (the group's mean reward) and the oracle, which picks the best
of the N by reward (pass@N for 0/1 rewards). "Gap closed" is (selected − random) / (oracle − random).
[`tests/test_bon.py`](https://github.com/r33drichards/laya-vision/blob/main/tests/test_bon.py) checks the metric
against brute-force enumeration.

Run: `modal run modal_app.py::bon_verifier` on the recommended checkpoint (`autoresearch/full/long-sep24-b64/best`,
the weights of `thaitea/laya-vision`), L4, bf16, seed 0, code `30542b6d`, 6.6 min. Rows and summary:
[`results/bon/autoresearch-full-long-sep24-b64-best-30542b6d.json`](https://github.com/r33drichards/laya-vision/blob/main/results/bon/autoresearch-full-long-sep24-b64-best-30542b6d.json).

Each candidate is judged in the state `{"image", "context": "Question: …\n\nResponse: …"}`, the layout the score
head was trained on for VLFeedback. Two scores are taken from one `predict` call per candidate:

- **correct**: P(true) of the `noul` question "Is the response a correct answer to the question about the image?"
- **helpful**: the expected level (0–4) of the helpfulness `score` rubric (`laya.rubric`).

## Multiple-choice options as candidates

Each of 400 seeded val questions per set gives one group: its options are the candidates, and the labelled option
has reward 1. `choice_prob` is the checkpoint's own `choice` head on the question, shown as a reference. At N = all
options, it equals the head's accuracy. ScienceQA questions have 2–5 options. Groups with fewer than N options use
all of them: 146 groups at N=3 and 260 at N=4.

| set | N | random | oracle (pass@N) | correct | helpful | choice head |
|---|---|---|---|---|---|---|
| A-OKVQA | 2 | 0.250 | 0.500 | 0.370 | 0.363 | 0.384 |
| A-OKVQA | 4 | 0.250 | 1.000 | 0.530 | 0.520 | 0.562 |
| ScienceQA | 2 | 0.364 | 0.727 | 0.598 | 0.590 | 0.633 |
| ScienceQA | 4 | 0.364 | 0.995 | 0.775 | 0.775 | 0.834 |

Used as a verifier, the checkpoint recovers 37–65% of the oracle's gain over random picking. The dedicated
`choice` head does better (42–75%). When the options are known up front, asking the `choice` question directly is
the better use of the model.

## VLFeedback: reranking model responses

The groups are the `score_vlfeedback` val rows, which were held out of training by row. Each group has all of the
row's model responses (4 in every group here; the prepared set keeps only 2), taken from `vlfeedback_80k.jsonl` at
`137dcea9`. The reward is the GPT-4V helpfulness rating (1–5). The `top` reward is 1 for a response that shares
the group's highest rating. There are 264 groups: only val rows whose source id occurs once are used (see below).

| reward | N | random | oracle | correct | helpful |
|---|---|---|---|---|---|
| rating (1–5) | 2 | 3.137 | 3.936 | 3.729 | 3.758 |
| rating (1–5) | 4 | 3.137 | 4.663 | 4.337 | 4.311 |
| top-rated | 2 | 0.357 | 0.617 | 0.545 | 0.553 |
| top-rated | 4 | 0.357 | 1.000 | 0.795 | 0.788 |

Here both verifier scores close 67–79% of the gap. Best-of-4 by P(correct) lifts the mean rating from 3.14 to 4.34
and keeps a top-rated response 80% of the time, against 36% for a random pick.

## Caveat: VLFeedback ids are not unique

29,073 ids repeat in `MMInstruction/VLFeedback`; for example, the M3IT subsets restart their numbering.
`prepare_score_dataset` stores each row's image as `images/vlf-<id>.jpg`, so a row with a repeated id can point at
another row's image. Of the 995 prepared val rows, 731 have such an id. This run uses only the other 264. The
`score_vlfeedback` numbers in the scorecards include the affected rows. A fix would key the images by stream
index. That requires re-preparing the set to a new name, which has not been done.
