# Row-level evidence

Every file here is listed in `SHA256SUMS` (check with `sha256sum -c SHA256SUMS` from this directory) and is
create-only: a re-run goes to a new name. `python benchmarks/verify_published.py` checks the sums, recomputes the
claimed numbers from the rows and compares them with the README table and `docs/*-metrics.json`
(`results/claims.json` lists which file backs which claim, and the tolerances).

## smolvlm-cauldron-score-2ep-bidir-full-best.vqa-val

`modal run modal_app.py::evidence --run cauldron-score-2ep-bidir-full/best --datasets vqa`, code commit `07b6b53`,
one L4, bf16 autocast, identity option order, batch 32 (as the training job's final eval). The full official val
splits: A-OKVQA 1,138, ScienceQA 2,097, VQAv2 yes/no 5,000 rows. The scored `model.safetensors` (sha256
`b5eb3ca6...9122f`) is the file published as `thaitea/laya-vision` at `d1fbdc0612fbe3b3d8ec6f54d328b195d35bb338`
(and `thaitea/laya-vision-smolvlm-256m-score` at `0d69aa37786bf409f467d2550a7d2dababfe8bc0`).

Each row: `dataset`, `id` (the prepared record's id), `index` (position among the split's usable records),
`qtype`, `label`, `option_order`, raw label-order `logits`, `probs_calibrated` (softmax of the logits over the
checkpoint's per-type temperature), and `input_ids_sha256` (`laya.vlm.input_ids_sha256` of the exact input ids).
The `.meta.json` records the checkpoint (weights and config sha256), temperatures, library versions, GPU, the sha256
of each val file and the prep metadata it was written with.

| Dataset | n | Accuracy, rows | Accuracy, metrics JSON / README | Calibrated ECE, rows | Calibrated ECE, metrics JSON |
|---|---|---|---|---|---|
| A-OKVQA | 1,138 | 60.37% | 60.02% / 60.0% | 0.157 | 0.161 |
| ScienceQA | 2,097 | 82.93% | 82.78% / 82.8% | 0.035 | 0.038 |
| VQAv2 yes/no | 5,000 | 72.54% | 72.44% / 72.4% | 0.077 | 0.076 |

All within tolerance (0.5 points accuracy, 0.01 ECE). The published numbers came from the training job's final
eval on an A100; this re-run on an L4 flips a few near-tied rows, all in the model's favour here (+0.1 to +0.35
points). The README numbers are left as published.

Not recorded for this checkpoint: its `vlm_agent_config.json` predates `backbone_revision`, so the SmolVLM commit
it was fine-tuned from is unknown (the full weights, including the backbone, are pinned by the sha256 above), and
the val splits predate `manifest.json`, so their upstream dataset commits are unknown; the sha256 of each val file
as scored is in the meta.

Discrepancy noted, not fixed: the README says "Calibrated ECE is 0.02 to 0.03 for all three over their full
validation sets". For the recommended checkpoint both the committed metrics JSON and these rows give 0.16
(A-OKVQA), 0.035-0.038 (ScienceQA) and 0.076-0.077 (VQAv2 yes/no); the model card
(`hf_model_card_score.md`) has the right values. The sentence likely describes an earlier checkpoint and needs
rewording by whoever owns the README.
