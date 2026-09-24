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
- Documentation follows mcp-js's layout. Public docs live in `site-docs/` (MkDocs, `mkdocs.yml`, published to
  https://r33drichards.github.io/laya-vision/ with the browser demo under `demo/`); `docs/` is for internal files and
  the metrics JSON that `results/claims.json` points at. The nav is explicit: add every new page to `mkdocs.yml`
  under Tutorials, How-to, Concepts or Reference. Link repository files by their GitHub URL (MkDocs cannot link
  outside `site-docs/`). Before a pull request, `nix build .#docs` (a strict build) must pass; CI also runs lychee
  (`.lychee.toml`) and `scripts/check_docs_browser.py` on the result.
