# Repository instructions

- Run tests from the repository root: `python -m pytest -q` (tests/test_vlm.py downloads SmolVLM-256M, ~2 min on a
  CPU). Before touching published numbers also run `(cd results/raw && sha256sum -c SHA256SUMS)` and
  `python benchmarks/verify_published.py`, which must exit 0.
- Results are create-only. Evaluation and evidence jobs write to new paths (a new `--name` for
  `modal run modal_app.py::evidence`, a new run name for training); never overwrite or delete an existing file under
  `results/raw/`, a checkpoint, or a prepared dataset on the Modal volumes.
- A headline number (the README checkpoint table, a model card, `docs/*-metrics.json`) needs committed row-level
  evidence: the `*.predictions.jsonl.gz` rows and `*.meta.json` from `modal_app.py::evidence`, `results/raw/SHA256SUMS`
  regenerated in the same commit, and a `results/claims.json` entry so `benchmarks/verify_published.py` checks it. Do
  not change a published number without that evidence, and do not edit one to match a re-run silently: record the
  discrepancy in `results/raw/README.md`.
- Pin revisions. Load Hub models and datasets at a commit (`VLMAgent(..., revision=..., backbone_revision=...)`,
  the prep jobs' `revision`) and keep the resolved commits that get recorded (`backbone_revision` in
  `vlm_agent_config.json`, `provenance` in `predict` output, `manifest.json` next to prepared datasets). Do not commit
  model weights, caches or third-party raw records.
- Bump `laya.vlm.PROMPT_FORMAT_VERSION` in the same commit as any change that can alter the input ids built for a
  (state, question): framing text, option rendering, budgets, truncation, image-token expansion.
