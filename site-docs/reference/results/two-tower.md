# Two-tower and late-interaction scoring

Can Laya score options without putting them in the image's sequence? The idea is a two-tower design like
[CLM](https://huggingface.co/Contrastive-LM/CLM-v0.1-8B): each option is embedded once and cached, and the frame is
embedded without the options. That would make answers order-invariant by construction, remove the 255-option
ceiling, and save the option tokens on every game step. This page reports what that costs in accuracy and
calibration against the current cross-encoder, and what it saves in latency.

**Verdict: not worth pursuing as a replacement.** Both variants are order-invariant, with 0 flips in 9,303 questions
against the cross-encoder's 4.7%. The best one (pure two-tower, 60 min) is 2.5 points of macro accuracy behind the
cross-encoder (0.684 vs 0.708) and 0.02 behind on the harness `quality`. The loss is concentrated on questions whose
options are content-bearing free text that has to be matched against the image, where it reaches 14-25 points. The
speed gain at one decision per frame on an L4 is 9-16%, because at that size the backbone pass is bound by kernel
launches, not by tokens. Late interaction (a cross-attention head over the frame's hidden states) did worse than the
pure two-tower in both runs. The one real win is **many options**: there is no ceiling, and scoring 1,024 cached
options costs the same as 4. That is a niche use case, not the default path.

## What was compared

Code: [`laya/two_tower.py`](https://github.com/r33drichards/laya-vision/blob/main/laya/two_tower.py) (not used by
default; `laya.vlm` is unchanged), Modal entrypoint
[`modal_two_tower.py`](https://github.com/r33drichards/laya-vision/blob/main/modal_two_tower.py), CPU tests
[`tests/test_two_tower.py`](https://github.com/r33drichards/laya-vision/blob/main/tests/test_two_tower.py).

- **Teacher / baseline**: `cauldron-score-2ep-bidir-full/best` (= `thaitea/laya-vision`), the SmolVLM-256M
  cross-encoder with block option attention, scored in identity option order.
- **State tower** (both students): the cross-encoder's own sequence cut just before the options
  (`<|im_start|>User:<image x64><state>\n<type> question: <ins><end_of_utterance>\nAssistant: Options:\n`) through
  the backbone. It is causal, so it never depends on the options.
- **Option tower** (both students): each option alone, text only, as `<|im_start|>Assistant: Options:\n- <option>\n`,
  through the same backbone's language model, pooled at the closing `\n` (the token the cross-encoder reads). It has no
  image, state or question, so it can be cached per option string for ever.
- **TT (pure two-tower)**: `exp(s) * cos(P_s(h_last), P_a(h_opt))`. `P_s` and `P_a` are MLP projections to 256 dims,
  `h_last` is the last state-tower token and `s` is learnable (initialised at ln 20, CLM-style).
- **LI (late interaction)**: each cached option vector becomes one token, appended to the state tower's hidden
  states. The teacher's 2-layer head transformer and scorer (copied as initialisation) run over them with a mask:
  state tokens attend to state tokens, option tokens attend to the state and to every option. The head has no
  positional encoding, so the logits are permutation-equivariant, and options can still compare themselves.
- **Training**: the student starts as a copy of the teacher. The language model is fully trainable at LR 1e-5 and
  the vision tower and connector are frozen. New heads train at 3e-4. The copied LI head trains at 5e-5 (`li-30m`)
  or 3e-4 (`li-30m-hlr`). Batch 32, cosine schedule on wall clock, bf16 autocast on one H100. The loss is
  `1.0 x CE(teacher's calibrated probabilities) + 0.5 x CE(label or vote histogram)`, with the teacher run online on
  the same batch.
- **Data**: the autoresearch pool (`/data/autoresearch/pool-v2` on `laya-datasets`), the same one
  `autoresearch/harness.py` uses. Training uses the `train` part (112,511 examples from the 19 Cauldron and 4 score
  sets, weights `score_vlfeedback: 3`, as in the teacher's run). Per-type temperatures are fit on the `calib` part
  (2,300 held-out train records) for **every** model, the teacher included. Scoring uses the `eval` part: 300 seeded
  val questions from each of the 34 eval sets, 9,303 rows.

## Results

All five models were evaluated on the same H100 in bf16, with temperatures refit on the same calibration rows.
Macro accuracy is the mean over the 34 sets. `ece_hard` and `nll_hard` cover the questions with one right answer.
`quality` = macro − ece_hard, the autoresearch objective. The flip rate is the share of the 9,303 questions whose
answer changes between identity, reversed and shift-by-one option orders.

| model | train | macro acc | pooled acc | ECE (hard) | NLL (hard) | quality | raw ECE (T=1) | flip, reversed | flip, any of 3 orders | agrees with teacher |
|---|---|---|---|---|---|---|---|---|---|---|
| cross-encoder (teacher) | - | **0.708** | **0.714** | 0.046 | **0.572** | **0.663** | 0.137 | 3.7% | 4.7% | - |
| TT | 30 min, 6,281 steps | 0.674 | 0.674 | **0.036** | 0.660 | 0.637 | **0.058** | **0** | **0** | 83.7% |
| TT | 60 min, 13,538 steps | 0.684 | 0.683 | 0.042 | 0.647 | 0.641 | 0.068 | **0** | **0** | 84.2% |
| LI | 30 min, 5,832 steps | 0.619 | 0.617 | 0.046 | 0.765 | 0.573 | 0.071 | **0** | **0** | 77.4% |
| LI, head LR 3e-4 | 30 min, 6,474 steps | 0.639 | 0.637 | 0.043 | 0.724 | 0.595 | 0.085 | **0** | **0** | 79.5% |

Averaging the teacher over the 3 orders (`n_permutations=3`, 3x the compute) gives pooled accuracy 0.717.

Macro accuracy by group:

| group | teacher | TT 30 | TT 60 | LI 30 | LI 30, head LR 3e-4 |
|---|---|---|---|---|---|
| VQA (3 sets) | 0.726 | 0.690 | 0.702 | 0.550 | 0.597 |
| Cauldron (19) | 0.769 | 0.729 | 0.740 | 0.677 | 0.694 |
| score (4) | 0.655 | 0.648 | 0.652 | 0.645 | 0.659 |
| eval, held out (8) | 0.585 | 0.550 | 0.559 | 0.497 | 0.512 |

Where the pure two-tower loses. These are all the sets that move by more than 3 points at 60 minutes:

| set | options look like | teacher | TT 60 | Δ |
|---|---|---|---|---|
| `cauldron_tqa` | textbook answer phrases | 0.739 | 0.489 | −0.250 |
| `eval_cifar10h` (never trained on) | 10 class names | 0.803 | 0.590 | −0.213 |
| `cauldron_visual7w` | free-text answers | 0.850 | 0.657 | −0.193 |
| `cauldron_aokvqa` | free-text answers | 0.713 | 0.573 | −0.140 |
| `cauldron_ai2d` | diagram labels | 0.777 | 0.681 | −0.096 |
| `aokvqa` | free-text answers | 0.633 | 0.577 | −0.057 |
| `cauldron_scienceqa` | answer phrases | 0.867 | 0.823 | −0.043 |
| `cauldron_iconqa` | short labels | 0.930 | 0.900 | −0.030 |
| `cauldron_vqarad` | yes / no | 0.887 | 0.919 | +0.032 |
| `cauldron_chartqa` | numbers | 0.561 | 0.683 | +0.122 |

Where the options are generic (yes/no, true/false, score levels, POPE, the four score sets, VQAv2), the two-tower
matches the teacher within noise: at 300 questions a set, ±3 points is sampling noise. Where the option text carries
the answer, a 256-d pooled state vector has to anticipate which answer phrase it should match before seeing any of
them, and it cannot. This is the textbook late-interaction gap. Doubling the training time recovered 1 of the
3.5 points, so time alone will not close it cheaply. LI was supposed to recover that interaction, but it did worse
than TT on the same sets and collapsed on CIFAR-10H (0.09 and 0.13, chance 0.10). The option tokens come from a
text-only pass, and 2 head layers are not enough to ground them in the frame's 100-200 state tokens. The teacher's head
transformer had only ever seen option tokens that attended to the image for 30 layers. A higher head LR helped
(+2 points) but left LI 4.5 points behind TT.

Calibration is not the cost. After temperature scaling, the students' ECE is at or below the teacher's. Before it,
they are far better (raw ECE 0.06-0.09 vs 0.14), because they were distilled from the teacher's calibrated
probabilities. NLL is worse (0.647 vs 0.572), which follows from the accuracy.

## Latency

`modal run modal_two_tower.py::latency --runs tt-30m,li-30m --n 50`. NVIDIA L4, bf16 backbone with fp32 heads (as
`VLMAgent` loads them), median of 50 after 10 warm-ups. The input is one AI2D image preprocessed by the processor
once, with the pixels already on the GPU; the vision tower runs every time. The question is the Breakout action
question with k options (up to 18: Atari action names; beyond that, synthetic `action i` options). The timing is the
model forward only, with no Python preprocessing, so it is not comparable with `bench_latency`. Student option vectors
are computed once beforehand, and that one-off cost is listed separately. Latency does not depend on the weights, so
the 30-minute checkpoints stand for all runs.

| k options | frames per call | cross-encoder tokens (options) | state tokens | cross-encoder ms | TT ms | LI ms | TT scoring only ms | LI scoring only ms | one-off option encoding ms |
|---|---|---|---|---|---|---|---|---|---|
| 2 | 1 | 137 (21) | 116 | 33.2 | 27.9 | 30.8 | 0.44 | 1.32 | 25.7 |
| 4 | 1 | 152 (36) | 116 | 30.6 | 27.8 | 30.1 | 0.46 | 1.27 | 26.5 |
| 18 | 1 | 301 (185) | 116 | 31.8 | 28.2 | 28.7 | 0.45 | 1.30 | 25.3 |
| 64 | 1 | 398 (320) | 78 | 32.5 | 28.7 | 28.0 | 0.46 | 1.27 | 24.5 |
| 180 | 1 | 978 (900) | 78 | 32.5 | 27.8 | 28.7 | 0.45 | 1.30 | 38.6 |
| 255 | 1 | does not fit `max_len` 1024 | 116 | - | 28.5 | 29.2 | 0.46 | 1.34 | 56.8 |
| 1,024 | 1 | does not fit | 116 | - | 29.0 | 32.5 | 0.46 | 5.20 | 359.7 |
| 4 | 16 | 152 (36) | 116 | 147.5 | 137.7 | 142.5 | 0.47 | 5.50 | 26.2 |
| 18 | 16 | 301 (185) | 116 | 187.1 | 138.6 | 144.9 | 0.45 | 6.04 | 25.0 |
| 64 | 16 | 398 (320) | 78 | 216.4 | 135.2 | 141.0 | 0.46 | 6.53 | 25.8 |

The vision tower alone takes 6.7 ms per frame. At 64+ options the cross-encoder's head budget (`head_max_len` 256)
cuts the question text, which is why its state is 78 tokens there. An earlier L4 container measured every number
here 40-50% higher, with the same ratios. As in the rest of these docs, compare within a run.

- **One decision per frame**, the game step: TT saves 2.8-5.3 ms of about 32 (9-16%), whatever k is. With one
  frame the 30-layer text model is bound by kernel launches, so a 137-token and a 978-token pass cost the same.
  The two-tower does not change that; a CUDA-graph step (`laya/static_step.py`) attacks it directly.
- **Batched frames** (16 environments per call): the option tokens start to count. TT saves 7% at 4 options, 26% at
  18 and 38% at 64. From there the vision tower (16 x 1,024 patches) dominates both models.
- **Option ceiling**: the cross-encoder cannot build a 255-option question with an image (5 tokens per option at the
  minimum cut, over `max_len`), so its practical ceiling is below 255. It fit 180 here. TT and LI take 1,024 options
  at the cost of 4.

### Game-step suitability

In a game the question and options are fixed and only the frame changes. The cross-encoder cannot reuse any option
compute across frames, because the options come after the image. The students cache options for ever, so a step is
the vision tower, the state tower and a dot product (TT) or two small head layers (LI). That is a clean fit, and the
decisions are exactly order-invariant, which removes the need for `n_permutations`. The measured saving is still
only 9-16% per single-frame step. Game accuracy was not measured: the teacher was never trained on games, and
Atari action names are generic options, the regime where TT matched the teacher on the dataset evals.

## Caveats

- Every student started from the cross-encoder's weights and learned from it for 30-60 minutes. A two-tower trained
  from scratch for as long as the teacher was (2 epochs) might close more of the gap. The trend from 30 to 60
  minutes (+1 point) suggests slowly.
- One seed per configuration, and 300 questions per set: per-set differences under about ±4 points are noise.
  Macro differences of 1 point between students are at the edge of it.
- The dataset numbers come from an H100 in bf16, not the L4 the published evals use. The teacher row was measured
  the same way here, so the comparison holds, but its numbers are not the published ones.
- The students are not wired into `VLMAgent.predict` (no act head, no save/load through `vlm_agent_config.json`),
  so no `bench_latency`, robustness or games run: the latency above is the model forward only.
- LI was a single design: one option token, 2 head layers, and options masked from the state tokens. Designs with
  more interaction (several tokens per option, more layers, ColBERT-style max-sim) were not tried. Each one moves
  per-decision cost back toward the cross-encoder's.
- The teacher's 3.7-4.7% flip rate is under identity, reversed and shift-by-one orders on these 9,303 rows. That is
  lower than the 5-8% quoted elsewhere, which used different sets and perturbations.

## Commands

From the repository root, on commits `e4ade75` (training and eval) and the following commit (latency):

```bash
python -m pytest -q tests/test_two_tower.py
modal run --detach modal_two_tower.py::eval_teacher --name teacher-eval
modal run --detach modal_two_tower.py::train --run tt-30m --mode tt --minutes 30
modal run --detach modal_two_tower.py::train --run tt-60m --mode tt --minutes 60
modal run --detach modal_two_tower.py::train --run li-30m --mode li --minutes 30
modal run --detach modal_two_tower.py::train --run li-30m-hlr --mode li --minutes 30 --lr-head 3e-4
modal run modal_two_tower.py::latency --runs tt-30m,li-30m --n 50
```

Each run writes `student.pt`, `train.json` (arguments, loss history), `eval.json` and `eval_records.pt` (per-row
logits under the 3 orders) to `/ckpt/two-tower/<run>/` on `laya-checkpoints`. The result files are in
[`eval-results/two-tower-*.json`](https://github.com/r33drichards/laya-vision/tree/main/eval-results). GPU time:
about 3.1 H100-hours for the four training runs, the teacher eval and the smoke tests, plus under an hour of L4.
