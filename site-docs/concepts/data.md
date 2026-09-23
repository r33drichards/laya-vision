# Training data

Everything is a prepared dataset on the `laya-datasets` Modal volume: `/data/vqa/<name>/{train,val}.jsonl` plus
`images/`, one record per question with its type, instructions, criteria and label, optionally a soft target.
How to prepare them: [Run jobs on Modal](../how-to/run-on-modal.md).

- **The Cauldron** (`laya/cauldron.py`, `prepare_cauldron`): the 19 subsets of
  [HuggingFaceM4/the_cauldron](https://huggingface.co/datasets/HuggingFaceM4/the_cauldron) whose answers are closed.
  Lettered choices and option lists become `choice`, yes/no turns become `noul`, RAVEN's letters become an 8-way
  `choice`; numbers, captions and free text are skipped. 270k questions.
- **Rubric-scored sets** (`laya/rubric.py`, `prepare_score`): VLFeedback (response helpfulness and visual
  faithfulness, 1 to 5), AVA (photo aesthetics, human vote histograms as soft targets), RichHF-18K (generated-image
  plausibility, alignment, aesthetics, overall) and CrisisMMD (damage severity). Each level is a short rubric clause
  in the style you would write for `predict`, with several instruction phrasings per question. How they were
  cleaned, and why: [Training data for score questions](score-data.md).
- **Held-out evaluation sets** (`laya/evalsets.py`, `prepare_eval`): KonIQ-10k photo quality and EvalMuse-40K
  prompt alignment as `score` questions, CIFAR-10H and FER+ as `choice`, VizWiz answerability and POPE (random,
  popular, adversarial) as `noul`. KonIQ, EvalMuse, CIFAR-10H, FER+ and VizWiz keep each image's human vote histogram
  as a soft target, so `evaluate` also reports cross-entropy against how people actually split (`soft_xent`, `xent`)
  next to the same number for the set's average histogram (`prior_…`). Evaluation only, except KonIQ, EvalMuse and
  FER+, which have train splits. KonIQ and FER+ also get their official test split (`evaluate --val-split test`).
- **The original three** (`aokvqa`, `scienceqa`, `vqav2_yesno`): the official train splits, prepared on the
  [`siglip-projector-experiment`](https://github.com/r33drichards/laya-vision/tree/siglip-projector-experiment)
  branch.
- **Games**: frames auto-labelled by a scripted expert or a trained agent; see
  [Game training](../reference/results/game-training.md) and the
  [Atari training data format](../reference/atari-data-format.md).

Pin revisions: the prep jobs take a `revision` for their source dataset and record what they resolved in a
`manifest.json` next to the prepared data.
