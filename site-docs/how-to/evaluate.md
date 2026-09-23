# Evaluate a checkpoint

`full_eval` runs the whole evaluation suite on one checkpoint in parallel and saves one file: the `evaluate` dataset
groups (`vqa,cauldron,score,eval`), the games suite and `bench_latency`.

```bash
modal run modal_app.py::full_eval --model my-run/best
```

Do not pass `--detach`: the results are collected locally.

- The results go to `eval-results/<run>-<commit>.json` locally, and to `<run>/evals/` on the checkpoint volume,
  beside the weights but outside the folder `publish` uploads.
- The file records the git commit it ran from, and each dataset's `meta.json`. To evaluate a branch's checkpoint,
  run it from that branch's checkout, since the Modal images ship the local `laya/` code.
- `--parts datasets,games,latency`, `--datasets` and `--val-split test` narrow it down.

## Write a scorecard

```bash
python scripts/eval_report.py eval-results/<file>.json --doc site-docs/reference/evals/<name>.md
```

`eval_report.py` turns result files into a Markdown report whose charts are Mermaid blocks, rendered on this site
and on GitHub; `--html` writes the same as a standalone page. A report written under `site-docs/` links its source
files on GitHub, since the site cannot link outside its own folder. Add the new page to the `nav` in `mkdocs.yml`,
under Reference › Results. The published checkpoint's scorecard is [here](../reference/evals/laya-vision.md).

## From GitHub Actions

The same suite runs from the `eval` workflow (`.github/workflows/eval.yml`). Start it from Actions → eval → Run
workflow on the branch whose code trained the checkpoint, or with
`gh workflow run eval.yml --ref <branch> -f model=<run>/best`.

- The datasets, games and latency parts run as parallel jobs, and each job's log is the live Modal output.
- The results go to the run summary and to a comment on the branch's open pull request. Re-running for the same
  checkpoint updates that comment.
- It needs the `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` repository secrets.
