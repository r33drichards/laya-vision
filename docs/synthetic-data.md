# Synthetic typed-question data from existing datasets

No public dataset is labelled as Laya questions: typed decisions (`choice`, `score`, `noul`) over an image plus an
optional note. The three VQA sets the released checkpoint was trained on were converted too (A-OKVQA happened to
be 4-way multiple choice, VQAv2 was filtered to its yes/no questions). `laya/synth/` generalises that conversion:
each *source* turns a labelled row of an existing dataset into typed questions, with distractors drawn from the
same dataset and soft targets wherever the source has votes.

```bash
modal run modal_app.py::prepare_synth                                   # every source, 20k train / 1k val each
modal run modal_app.py::prepare_synth --sources websight,screenqa --n-train 50000
python -m laya.synth --root ./data --sources vizwiz --n-train 2000        # locally, same layout
```

Each source becomes its own prepared dataset `/data/vqa/synth_<source>/` in the layout `finetune_long` already
reads (`train.jsonl`, `val.jsonl`, `images/`, `meta.json`, `_READY`), so they can be mixed, capped and evaluated
separately:

```bash
modal run --detach modal_app.py::finetune_long --datasets aokvqa,scienceqa,vqav2_yesno,synth_websight,synth_screenqa,synth_vizwiz --init-from all3-3ep/best --run-name synth-v1
```

## Sources

| Source | Data | Questions it makes | Split | Licence |
|---|---|---|---|---|
| `websight` | 2M synthetic web pages with their HTML (`HuggingFaceM4/WebSight` v0.2) | `noul` has a nav bar / footer / form / table / images / buttons; `noul` dark colour scheme; `choice` what kind of business the site is for (5-way, from the generation prompt); `score` how many links, how many images (4 levels). 3 sampled per page. | hash by page | CC-BY-4.0 |
| `screen2words` | 22k RICO Android screens, 5 human summaries each, Play Store category (`bevaya/RICO-Screen2Words`) | `choice` which summary describes the screen (5-way); `choice` app category (5-way over 21) | official train / test | CC-BY-4.0 |
| `screenqa` | 86k short-answer questions over RICO screens (`bevaya/RICO-ScreenQA-Short`) | `choice` the question with 3 distractors (4-way); `noul` "<question> Is the answer 'x'?" half true, half distractor | official train / test | CC-BY-4.0 |
| `vqav2` | VQAv2 (`lmms-lab/VQAv2`), yes/no questions excluded (they are the existing `vqav2_yesno` set) | `choice` question + majority answer + 3 distractors, target = the 10 annotators' votes over the options; `score` "how many" as an ordinal count (none / one / two / three / four or more), soft from votes | official train / validation | CC-BY-4.0 |
| `aokvqa` | A-OKVQA (`HuggingFaceM4/A-OKVQA`) | `noul` "<question> Is the answer 'x'?" where x is the right choice or one of the row's own wrong choices | official train / validation | CC-BY-4.0 |
| `vizwiz` | VizWiz-VQA val, photos by blind users, 10 answers each (`lmms-lab/VizWiz-VQA`) | `noul` can the question be answered from this photo (soft: share of "unanswerable" votes); `noul` yes/no questions (soft); `choice` other questions with 3 distractors, soft from votes | hash by question | CC-BY-4.0 |
| `ava` | AVA aesthetic ratings, 10% mirror with 50+ votes per photo (`trojblue/AVA-aesthetics-10pct-min50-10bins`) | `score` 5 levels (very poor … excellent), target = the vote histogram collapsed from 10 bins | hash by photo | **research only** |
| `nlvr2` | NLVR2 dev pairs: two photos and a statement (`lmms-lab/NLVR2`) | `noul` is the statement true of the pair; the record carries both images in order | hash by statement group | CC-BY-4.0 annotations, web images |

`ava` is the first ordinal (`score`) data with real rater distributions; `vqav2` counting is the second. The
released weights are already non-commercial because of ScienceQA; keep `ava` out of any checkpoint meant to be
commercial.

## How the questions are built

- **Distractors are plausible, not random.** They are sampled from a pool of other rows' answers of the same
  kind: the same `question_type` for VQAv2, the same first three words of the question for ScreenQA, other
  screens' summaries for Screen2Words, other sites' business types for WebSight. A distractor never normalises to
  an accepted answer of the row, and for the summary and business-type questions it may not share a content word
  with one ("Fashion Retailer" is not a distractor for "Fashion Brand").
- **Soft targets.** Where the source has several annotators or raters, `target` is their distribution over the
  options and `label` is its argmax. The trainer's loss (soft cross-entropy plus a proper scoring rule) uses the
  distribution; accuracy uses the label.
- **Several phrasings** of each question family, chosen at random, so the model reads the instruction.
- **Every question on an image is in one split.** Sources with an official validation split use it; the rest are
  hash-split by image key with 5% to val.
- **Balanced by construction** where the source allows it: the "is the answer x" questions are half true, half
  false; WebSight's presence questions follow the base rate of the tag (nav and images are common, tables rare).
- Images are saved once per source image (questions share it) and resized to 1024 on the longest side. The
  model itself sees 512 (`laya/preprocess.py`), so nothing is lost at train time and the files stay small.

## Verification

- `tests/test_synth.py` checks every source on fake rows (labels come out of the HTML, distractors exclude the
  answer and near-duplicates, soft targets sum to one, the argmax is the label) and runs the writer end to end
  into `load_jsonl_examples`, including two-image records. No downloads.
- A 60-train / 12-val run of every source against the live Hub streams completed on a laptop-class container.
  Memory: about 3 GB per source, dominated by one parquet row group; the run is one container per source on
  Modal. `datasets`' streaming `.shuffle` is not used because it took about 10 GB even with `buffer_size=1`, so a
  run takes the first rows of each split in file order.

## Known limits and next steps

- **Screenshots are resized to 512 squares by the model**, so text is unreadable; the screen questions here are
  layout and state questions on purpose. Reading small text needs SmolVLM's image splitting re-enabled
  (`do_image_splitting` in `laya/vlm.py`), at several times the token cost.
- **WebSight pages are synthetic** and all Tailwind. They teach web layout vocabulary, not real desktop apps.
  Real desktop screenshots with structural labels (a DOM or accessibility tree read while capturing) are the
  strongest next source and need no public dataset; `laya.synth.core` is ready for one more `Source`.
- **Label noise.** The Screen2Words category is the app's, not necessarily visible on the screen. ScreenQA
  distractors from the same question prefix are sometimes wrong-typed ("What is the status of…" can draw a date).
- **Not included yet:** OS-Atlas (13M desktop and web GUI elements, Apache-2.0, zip-packed), AgentNet desktop
  trajectories, DocVQA and ChartQA (open-ended; need distractors and image splitting), wave-ui-25k (no stated
  licence).
