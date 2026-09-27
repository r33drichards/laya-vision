# Detector regions and zoom search (exploratory)

These are exploratory probes of the released checkpoint,
[thaitea/laya-vision](https://huggingface.co/thaitea/laya-vision) at commit
`f2fe3c12cb6d04c59d8a190250bf3fb40fc828dc`. They ran on a CPU, with one option order unless noted. **None of these
numbers is a published or headline number.** The per-row outputs were not committed, so `results/claims.json` has no
entries for them and `benchmarks/verify_published.py` does not check them. Each section gives the command that
reproduces it. Brackets are 95% Wilson intervals.

**Result:**

- **Numbered marks (select):** the checkpoint reads a digit, even an 18 px one, but mostly does not tie a small
  numbered tag to its box. With 2 to 8 marked boxes it scores close to chance and mostly answers "box 2".
- **Crops of true boxes (map):** a yes/no "Is this a &lt;class&gt;?" on COCO crops is 91% accurate at 0.5 and well
  calibrated out of the box. It ranks no better than zero-shot SigLIP, and it is about 7× slower.
- **Zoom:** on V*Bench's small-object attribute questions, a crop around the true target raises accuracy from 22% to
  75%. A search that asks yes/no "Is there a &lt;target&gt; here?" over quadrants finds that crop about half the time,
  and gets 48% against 22% for the full image. Asking "where is the &lt;target&gt;?" as a choice question does not work.

## Motivation

The idea was to stream a detector's boxes into Laya: OmniParser for screens, YOLO, RF-DETR or SAM for photos. There
are two ways to use them (see [Decide over a detector's boxes](../../how-to/detector-regions.md)):

- **select** one box of N, with the boxes drawn as numbered marks and each box one option;
- **map** a question over every box's crop.

A related idea is foveation: zoom only where needed, to spend image tokens only there. The checkpoint sees an image
as one 512 px tile, 64 image tokens, so a small object in a large photo is a few pixels by the time the model sees it.

Prior art:

- [Set-of-Mark prompting](https://arxiv.org/abs/2310.11441): draw numbered marks on regions and let the model answer
  with a number.
- [OmniParser](https://arxiv.org/abs/2408.00203): a screen parser that gives the boxes and captions for Set-of-Mark
  on screenshots.
- [SoM-LLaVA](https://arxiv.org/abs/2404.16375): small models need training to use marks ("list items one by one").
- [SeeAct](https://arxiv.org/abs/2401.01614): web agents; grounding the chosen element is the hard part.
- [Roboflow Workflows](https://blog.roboflow.com/tablet-defect-inspection/): detect, crop, then a VLM classifies each
  crop.
- [What does CLIP know about a red circle?](https://arxiv.org/abs/2304.06712): a drawn circle steers CLIP's attention.
- [V*](https://arxiv.org/abs/2312.14135): guided visual search for small details in high-resolution images, and the
  V*Bench benchmark used here.
- [ZoomEye](https://arxiv.org/abs/2411.16044): tree search over image crops, scored by the model's own confidence.
- [ViCrop](https://arxiv.org/abs/2502.17422): training-free cropping from the model's attention.
- [AwaRes](https://arxiv.org/html/2603.16932): deciding when to spend more resolution.

**Method.** The experiments were ordered by expected information per hour, ln(1/p)/t, where p is the chance the
experiment succeeds and t its time, after Steinhardt's
[Research as a Stochastic Decision Process](https://cs.stanford.edu/~jsteinhardt/ResearchasaStochasticDecisionProcess.html).
Cheap "cheating" ceilings, which use the true boxes, came before real components: if a ceiling fails, the component
behind it cannot help.

## B1: can it read numbered marks?

[`benchmarks/marks_probe.py`](https://github.com/r33drichards/laya-vision/blob/main/benchmarks/marks_probe.py)
draws synthetic 512 px images (no resizing), n=200 per condition. Marks are black numbered outlines drawn with
`laya.regions.draw_marks`. In the `marks_N` conditions the question is "Which box contains the &lt;colour&gt; square?", the
options are only the box numbers, and the numbers are a random permutation, so the answer has to come from reading
the tag next to the right square. `position_4` asks the same question with no marks and position words as options.

```bash
python benchmarks/marks_probe.py --n 200 --out /tmp/marks_probe.jsonl
python benchmarks/marks_probe.py --n 200 --n-permutations 4 --conditions marks_2,marks_4,marks_4_large
python benchmarks/marks_probe.py --n 200 --contextual
```

| Condition | What it tests | Accuracy | 95% interval | Chance |
|---|---|---|---|---|
| digit_large | read one 200 px digit, options 1–9 | 100% | [98.1, 100] | 11.1% |
| digit_small | the same at 18 px, the default tag size | 99.0% | [96.4, 99.7] | 11.1% |
| marks_2 | 2 marked boxes | 56.5% | [49.6, 63.2] | 50% |
| marks_4 | 4 marked boxes | 40.5% | [33.9, 47.4] | 25% |
| marks_8 | 8 marked boxes | 24.0% | [18.6, 30.4] | 12.5% |
| marks_4_large | 4 boxes, 64 px tags | 62.5% | [55.6, 68.9] | 25% |
| position_4 | 4 squares, no marks, position words | 68.5% | [61.8, 74.5] | 25% |

On marks_2 the mean confidence was 0.83 at 56.5% accuracy.

**Label bias.** With small tags the model mostly answers "box 2": 186 of 200 on marks_2 and 102 of 200 on marks_4.
Shuffling the option order (`--n-permutations 4`) did not change this:

| Condition | 1 order | 4 orders | "box 2" chosen (4 orders) |
|---|---|---|---|
| marks_2 | 56.5% | 55.5% | 186 / 200 |
| marks_4 | 40.5% | 42.5% | 98 / 200 |
| marks_4_large | 62.5% | 60.5% | |

So it is a bias toward the label "2", not toward an option position (the checkpoint's options attend to each other
both ways, so position matters less anyway).

**Contextual calibration.** Dividing by the answer distribution on a blank image
([Zhao et al., 2021](https://arxiv.org/abs/2102.09690)) did not help: marks_2 57.0%, marks_4 40.0%, marks_8 23.5%,
marks_4_large 62.5%, position_4 65.5%. The blank-image prior is mild (0.40 / 0.60 on marks_2) while the bias on the
marked images is strong, so the marked image causes the bias.

**Conclusion.** The checkpoint reads digits but does not bind a small tag to its box. Large tags help (62.5% on 4
boxes) but do not reach the no-marks position control. Select-by-marks needs large tags and training on marks.

## A: does it know what a detector crop is?

[`benchmarks/crop_probe.py`](https://github.com/r33drichards/laya-vision/blob/main/benchmarks/crop_probe.py) crops
ground-truth boxes from COCO val2017 (`detection-datasets/coco` at `cf0b22332314a937e9dc8a1957b21725430bb41d`): 8
classes × 40 crops = 320 crops, padded by 0.1 and at least 32 px, the way `map_regions` crops. The baseline is
zero-shot SigLIP, `google/siglip-base-patch16-224` at `7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed`, prompted "a photo of
a &lt;class&gt;.".

```bash
python benchmarks/crop_probe.py --n 40 --out /tmp/crop_probe.jsonl
```

**Verification** (`noul` "Is this a &lt;class&gt;?", asked with the true class and with one random other class, 640
answers). SigLIP's probability is its raw sigmoid.

| Model | Accuracy at 0.5 | 95% interval | AUROC | ECE | Mean confidence |
|---|---|---|---|---|---|
| Laya | 91.1% | [88.6, 93.1] | 0.956 | 0.084 | 0.849 |
| SigLIP | 52.7% | [48.8, 56.5] | 0.960 | 0.435 | |

The AUROC difference (Laya − SigLIP) is −0.004 [−0.027, +0.019]. SigLIP ranks as well, but its raw P(yes) on true
crops averages 0.10, so it is useless at 0.5 until calibrated.

**8-way** (`choice` over the 8 class names; SigLIP is the softmax over the 8 prompts):

| Model | Accuracy | 95% interval | ECE |
|---|---|---|---|
| Laya | 84.7% | [80.3, 88.2] | 0.077 |
| SigLIP | 89.4% | [85.5, 92.3] | 0.055 |

The difference is −4.7 points [−9.1, 0.0].

**Latency**, one CPU thread: Laya 2.55 s per crop (3 questions in one `predict` call), SigLIP 0.35 s.

**Errors.** Laya defaults to "person": 16 of 40 chairs were called person. It is better than SigLIP on the person
class itself (AUROC 0.99 against 0.79). It is slightly underconfident: in the 0.85–0.95 confidence bin, mean
confidence was 0.88 and accuracy 0.96.

**Conclusion.** There is real signal, calibrated at 0.5 without any fitting, but no ranking advantage over SigLIP,
at about 7× the latency. SigLIP would need its own calibration to be used at a threshold.

## C1: the zoom ceiling

[`benchmarks/zoom_probe.py`](https://github.com/r33drichards/laya-vision/blob/main/benchmarks/zoom_probe.py) uses
V*Bench (`craigwu/vstar_bench` at `d9ae62c903da0c98336e85c5ee89cd863b04b4da`), 191 items. The median image is
2250×1500 and the median target about 45×48 px. The oracle crop is the union of the target boxes, padded by 0.5, at
least 256 px, squared. The random crop has the same size and does not overlap the target.

```bash
python -c "from huggingface_hub import snapshot_download as s; s('craigwu/vstar_bench', repo_type='dataset', revision='d9ae62c903da0c98336e85c5ee89cd863b04b4da', local_dir='/tmp/vstar')"
python benchmarks/zoom_probe.py --data /tmp/vstar --out /tmp/zoom_probe.jsonl
```

| Condition | Image tokens | Accuracy | 95% interval | ECE | direct_attributes | relative_position |
|---|---|---|---|---|---|---|
| full | 64 | 33.5% | [27.2, 40.5] | 0.322 | 21.7% | 51.3% |
| oracle_crop | 64 | 67.0% | [60.1, 73.3] | 0.116 | 74.8% | 55.3% |
| oracle_crop_plus_full | 128 | 67.0% | | 0.145 | | |
| random_crop (n=173) | 64 | 31.8% | | | | |

Paired against `full`: the oracle crop fixed 70 answers and broke 6 (p = 6.3e-15); the random crop fixed 20 and
broke 19 (p = 1.0). Adding the full image to the crop did not help.

`relative_position` (is A left or right of B, 2 options) stays at chance even with the oracle crop: the model
answers "left" on 71 of 76 with the full image and 67 of 76 with the crop. That is a gap in the question type, not
in resolution.

## C2: can it find where to zoom?

[`benchmarks/zoom_search_probe.py`](https://github.com/r33drichards/laya-vision/blob/main/benchmarks/zoom_search_probe.py)
replaces the oracle box with the model's own search, on `direct_attributes` only (n=115). Every variant answers the
benchmark question on one crop, so the variants differ only in how that crop is chosen. "Hit" means the chosen cell
holds the target box's centre.

```bash
python benchmarks/zoom_search_probe.py --data /tmp/vstar --variant hier --c1-rows /tmp/zoom_probe.jsonl --out /tmp/c2_hier.jsonl
# the other variants: --variant oracle_tile | choice | grid | hier3 | oracle_centred; --variant oracle_tile --oracle-grid 8
```

**At a 4×4-cell zoom** (the cell grown by 25% per side):

| Variant | How the crop is chosen | Search image tokens | Accuracy | 95% interval | Hit rate |
|---|---|---|---|---|---|
| full image | no zoom | | 21.7% | | |
| choice | "Where is the &lt;target&gt;?" over 4 quadrants, twice | 192 | 30.4% | [22.8, 39.4] | 4% |
| hier | `noul` "Is there a &lt;target&gt; in this image?" on 4 quadrants, then 4 sub-quadrants | 576 | 47.8% | [38.9, 56.9] | 48% |
| grid | the same `noul` on all 16 cells | 1088 | 47.8% | | 52% |
| oracle_tile | the cell holding the target | | 47.0% | | 100% |

The choice question found the right cell 4% of the time, below the 1-in-16 of a random pick. `hier` answered
65.5% correctly when it hit and 31.7% when it missed; paired against the full image it fixed 36 answers and broke 6
(p = 2.8e-6). `hier` matches the oracle cell at half the tokens of `grid`.

**Deeper zoom.** An oracle 8×8 cell (grown by 25%) reached 56.5% [47.4, 65.2]: 62.7% when the target was near the
crop's centre (n=67) and 47.9% near its edge (n=48). So `hier3` adds a third `noul` level and centres the final
crop: a square of side 1.5 × max(W, H) / 8 on the score-weighted centroid of the last level's cells (12 crops
scored, 832 tokens). `oracle_centred` is the same crop centred on the true target.

| Variant | Accuracy | 95% interval | ECE | Find rate | Acc given hit | Acc given miss |
|---|---|---|---|---|---|---|
| hier3 | 48.7% | [39.8, 57.7] | 0.183 | 52% | 66.7% | 29.1% |
| oracle_centred | 66.1% | [57.0, 74.1] | 0.087 | 100% | | |

Paired against the full image, `hier3` fixed 36 and broke 5 (p = 7.8e-7).

Per level of `hier3`, each given that the level before was right:

| Level | Picks the cell holding the target |
|---|---|
| 1 (quadrants) | 59% (true quadrant in the top 2: 85%) |
| 2 | 81% |
| 3 | 80% |

**Conclusions.** Yes/no search finds the target where a "where is it?" choice question fails. Once found, the
search's crop answers about as well as the crop centred on the true target (66.7% against 66.1%). The bottleneck is
the first, coarsest level: in a quadrant view the target is about 20 px.

**Caveats.** V*Bench was chosen because its targets are tiny, so the gains are larger than on typical photos. n is
small (115). There is one question wording. No image-splitting baseline was compared, because the checkpoint was not
trained with splitting (see [Image splitting on SmolVLM2](split-bench.md)). CPU timings came from a shared 4-core
machine and are not reported as latency results.

## Two-branch search

<!-- BEAM2_RESULTS -->

*Placeholder: results of a still-running experiment go here.*

## Takeaways

**What worked**

- Search, then zoom: `noul` questions over quadrants localise small targets well enough to more than double
  accuracy over the full image on V*Bench attributes.
- `noul` verification of a crop: accurate and calibrated at 0.5 without fitting.

**What didn't**

- Numbered marks with the default small tags: read as digits, not bound to boxes; a strong bias to "box 2".
- "Where is the &lt;target&gt;?" as a choice question: below random at finding the cell.
- Left/right questions: at chance, even on the oracle crop.

**Open questions and next steps**

- Two-branch search: keep the top 2 cells at the first level, since the true quadrant is in the top 2 85% of the time
  (see [Two-branch search](#two-branch-search)).
- A finer first level (for example 3×3 or overlapping cells) so the target is larger at the first look.
- Detector proposals as the first level instead of fixed quadrants.
- Training on marks and on detector crops.
- An image-splitting baseline on a checkpoint trained with splitting.
- A real use case with its own labelled data, rather than V*Bench and synthetic images.
