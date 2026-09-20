# Open CaptchaWorld as typed decisions

[Open CaptchaWorld](https://arxiv.org/abs/2505.24878) (Luo et al., 2025) benchmarks multimodal agents on
225 interactive CAPTCHAs. Its agents drive a browser: they read a screenshot, reason, then click, drag and
type until they hit submit, and are scored pass@1 on the finished puzzle. OpenAI o3 reaches 40.0% there;
humans reach 93.3%.

Laya Vision cannot play that game. It answers typed questions (`choice` / `score` / `noul`) about images in
one forward pass, with no text generation, no coordinates and no multi-step state. So the benchmark is
re-expressed as decisions and graded offline against the bundled `ground_truth.json` files.

The result is not a number to put next to o3's. It is a map of what a 256M typed-decision model can and
cannot see, and the answer turns out to be sharply bimodal.

## What is covered

13 of 20 types, 300 puzzles, 3,121 decisions. `laya/captcha.py` does the conversion,
`examples/captcha_eval.py` runs it.

```bash
python examples/captcha_eval.py --data /path/to/OpenCaptchaWorld/captcha_data --out results.json
```

The data needs its git-lfs objects resolved (~806 MiB) or every `ground_truth.json` is just a pointer file.

Three mappings, each keeping a decision small and self-contained:

| Family | Types | Decision | Graded by |
|---|---|---|---|
| Reference + options | Connect_icon, Coordinates, Dart_Count, Image_Matching, Object_Match, Path_Finder, Rotation_Match | one `noul` per option over `[reference, candidate]` | argmax of the k yes-probabilities |
| Grid select | Select_Animal, Image_Recognition, Patch_Select, Unusual_Detection, Bingo | one `noul` per cell, over that cell's crop alone | argmax, or threshold at 0.5 and compare sets |
| Count | Dice_Count | one `choice` over a candidate ladder | exact match |

The obvious encoding — one `choice` over options named "option 1".."option k" — was rejected on purpose. It
would measure whether a 256M model can bind an option label to an image position, not whether it can see.
Pairwise comparison removes that confound.

Grid cells are cut row-major over `grid_size = [rows, cols]`, matching `static/js/script.js:944` in the
benchmark. Getting that backwards silently scores 5x5 grids against transposed truth.

### Not covered

Six coordinate-click types (Geometry_Click, Pick_Area, Place_Dot, Click_Order, Slide_Puzzle,
Misleading_Click) need pixel-precise output this architecture has no head for — Slide_Puzzle wants a drag
landing inside 10px on a 512px image. Hold_Button has no perception content at all: the answer is always
`"completed"`. `laya.captcha.EXCLUDED` lists them with reasons, and a test asserts every one of the 20 types
is either loaded or explicitly excluded.

## Results

Released checkpoint `thaitea/laya-vision-smolvlm-256m`, zero-shot, on an M-series CPU/MPS. 878 s for 2,245
forward calls (391 ms/call).

`AUC` is the probability the model scores a correct option above an incorrect one: 0.5 means it ranks
correct and incorrect identically. `yes%` is how often it answers yes to a `noul`, against the `true%` base
rate.

| type | n | pass@1 | chance | dec.acc | ECE | AUC | yes% | true% |
|---|---|---|---|---|---|---|---|---|
| Select_Animal | 30 | **0.967** | 0.167 | 0.761 | 0.218 | **0.992** | 0.406 | 0.167 |
| Image_Recognition | 20 | 0.050 | 0.014 | 0.600 | 0.157 | **0.899** | 0.878 | 0.478 |
| Patch_Select | 20 | 0.000 | 0.000 | 0.328 | 0.300 | **0.700** | 0.928 | 0.288 |
| Rotation_Match | 48 | 0.146 | 0.125 | 0.125 | 0.486 | 0.542 | 1.000 | 0.125 |
| Unusual_Detection | 30 | 0.000 | 0.091 | 0.589 | 0.056 | 0.541 | 0.300 | 0.389 |
| Image_Matching | 19 | 0.211 | 0.200 | 0.305 | 0.247 | 0.529 | 0.789 | 0.200 |
| Dart_Count | 20 | 0.100 | 0.078 | 0.078 | 0.526 | 0.497 | 1.000 | 0.078 |
| Object_Match | 20 | 0.150 | 0.200 | 0.200 | 0.445 | 0.497 | 1.000 | 0.200 |
| Path_Finder | 10 | 0.200 | 0.200 | 0.260 | 0.293 | 0.497 | 0.900 | 0.200 |
| Coordinates | 18 | 0.111 | 0.145 | 0.195 | 0.340 | 0.472 | 0.930 | 0.141 |
| Bingo | 25 | 0.040 | 0.051 | 0.051 | 0.678 | 0.491 | 1.000 | 0.051 |
| Connect_icon | 20 | 0.150 | 0.137 | 0.136 | 0.564 | 0.427 | 1.000 | 0.136 |
| Dice_Count | 20 | 0.050 | 0.100 | 0.050 | 0.139 | — | — | — |
| **ALL** | **300** | **0.183** | **0.113** | 0.236 | 0.402 | — | 0.895 | 0.171 |

Overall pass@1 0.183 against a 0.113 chance baseline. Treating that as the finding would be a mistake: it
averages two completely different behaviours.

### One image, one question: it works

**Select_Animal is 96.7% — 29 of 30 — at AUC 0.992.** That is not a positional artifact. The correct cell
is spread evenly over all six positions (6/5/4/3/7/5) across 16 different target animals. `Image_Recognition`
(AUC 0.899) and `Patch_Select` (AUC 0.700) rank correctly too.

These three are exactly the training distribution. "Is this tile a fox?" over a single crop is a VQAv2
yes/no question, and VQAv2 yes/no is what the checkpoint was fine-tuned on.

### Two images to compare: completely blind

Every reference-and-candidate type sits at **AUC 0.43–0.54 with `yes%` at or near 1.000**. The model accepts
every candidate it is shown. Connect_icon, at AUC 0.427, is slightly worse than random.

This follows from the training data. A-OKVQA, ScienceQA and VQAv2 are all single-image QA; the checkpoint has
never seen a two-image comparison task, and `laya/vlm.py` encodes both images into one flat prefix with
nothing to mark which is the reference. The capability was never trained, so there is nothing to measure.

### The shuffled-image control

Re-run with each puzzle's images replaced by another puzzle's of the same type
(`--control shuffle`). This is the check that killed the `siglip-projector-experiment` branch, where
accuracy on shuffled images matched accuracy on real ones.

| | real | shuffled |
|---|---|---|
| Select_Animal pass@1 | 0.967 | 0.433 |
| Select_Animal AUC | 0.992 | 0.670 |
| Image_Recognition AUC | 0.899 | 0.544 |
| Patch_Select AUC | 0.700 | 0.423 |
| macro-AUC over 12 types | 0.590 | 0.513 |
| overall pass@1 | 0.183 | 0.133 |

The three types with signal lose it. The eight comparison types are unmoved — Connect_icon 0.427 → 0.478,
Dart_Count 0.497 → 0.545, Bingo 0.491 → 0.506 — because they were never using the images in the first place.
That is the control behaving exactly as it should: it separates real perception from a text-side artifact,
and it confirms Select_Animal's 96.7% is real.

Select_Animal does not fall all the way to its 0.167 chance level under shuffling, which is expected: the
donor grid is another Select_Animal puzzle drawn from the same 16-animal vocabulary, so it sometimes does
contain the animal being asked about.

## Reading the metrics

**pass@1 and AUC disagree on purpose.** `Image_Recognition` ranks cells well (AUC 0.899) but completes 5% of
puzzles, because pass@1 demands an exact set match over 9 cells. `Patch_Select` needs an exact match over 25
and scores 0.000 despite AUC 0.700. Per-puzzle completion and per-decision perception are different
questions, so both are reported.

**`dec.acc` is uninformative wherever `yes%` is near 1.0.** A model that accepts every option scores exactly
`1/k` per decision, which is why the k-way rows land on their chance column. The AUC and `yes%` columns exist
to make that visible rather than something you have to infer.

**The pooled AUC on the ALL row is omitted.** Pooling decisions across types with different base rates is a
Simpson's-paradox trap; the macro-average over types (0.590 real, 0.513 shuffled) is the honest summary.

**Dice_Count's option ladder is synthetic.** Counting has no natural option list, so nine distractors are
drawn deterministically per puzzle from the range the benchmark's own sums span. Chance is 1/10. At 0.050 the
model is below chance — it cannot count. The number depends on that ladder in a way the other types' numbers
do not.

## What this would take to fix

The gap is two-image comparison, not perception, and the pipeline for closing it already exists in this repo:
`docs/game-training.md` describes taking the same checkpoint from "only ever shoots" to expert level in
ViZDoom with 7 minutes of training on 20,000 auto-labelled frames.

The analogue here is a training set of `[reference, candidate] -> yes/no` pairs, which the benchmark's own
structure generates for free: every k-way puzzle yields one positive and k-1 negatives. That is ~1,100 pairs
from the 155 k-way puzzles — enough to check whether the capability is learnable, far too few to train on
and still evaluate honestly. Real training data would need generated puzzles of the same types, with the
benchmark held out.

Until then the coordinate-click half of the benchmark stays out of reach regardless, since it needs an output
head this architecture does not have.
