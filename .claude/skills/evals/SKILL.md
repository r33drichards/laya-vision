---
name: evals
description: Run, read and report Laya Vision evals on Modal. Use whenever the task is to evaluate, benchmark or score a checkpoint (a run on the laya-checkpoints volume, or a Hugging Face model such as thaitea/laya-vision), compare checkpoints, run the dataset evals / games suite / latency benchmark, prepare eval datasets, trigger the eval GitHub workflow, write or commit an eval report (docs/evals/*.md, HTML scorecard), add a new eval set or game, or explain eval numbers (accuracy, ECE, soft_xent vs prior, normalized Atari score). Also covers getting the Modal client to connect from a Claude Code cloud sandbox.
---

# Laya Vision evals

Everything runs on Modal from `modal_app.py`, against checkpoints on the `laya-checkpoints` volume, using whatever
`laya/` code is in the local checkout (the images ship it). So: **check out the branch whose code trained the
checkpoint, then run from there.**

## 1. Connect to Modal

Check first: `modal volume ls laya-checkpoints` should list `smolvlm`, `smolvlm2`, `modernvbert`, ...

In a Claude Code cloud sandbox it usually fails with "Could not connect to the Modal server" although
`~/.modal.toml` has a token and `curl https://api.modal.com` works. Two causes, both fixed in a scratch venv
(never in the repo):

```bash
uv venv $SCRATCH/venv --python 3.11 && . $SCRATCH/venv/bin/activate
uv pip install "modal[api-proxy-support]"            # the client needs python-socks to use HTTPS_PROXY
C=$(python -c "import certifi; print(certifi.where())")
grep -q CCR-PROXY-CA "$C" || { echo "# CCR-PROXY-CA"; cat /root/.ccr/ca-bundle.crt; } >> "$C"   # proxy's TLS CA
modal volume ls laya-checkpoints
```

To see the real error instead of the generic one:
`python -c "import modal,traceback; modal.Volume.from_name('laya-checkpoints').listdir('/')"` and read the
`__cause__`.

## 2. Pick the checkpoint

Evals take a **volume run name**: `<run>/best` under `/ckpt/smolvlm`, or `<family>/<run>/best`
(`modernvbert/...`, `smolvlm2/...`). Hugging Face ids are not accepted by `evaluate` or `bench_latency`.

A Hub model is a mirror of a volume run. `thaitea/laya-vision` = `cauldron-score-2ep-bidir-full/best`
(identical `model.safetensors` sha256 `b5eb3ca6...`). For another Hub model, match its weights to a run:

```bash
curl -s https://huggingface.co/api/models/<org>/<name>/tree/main   # lfs oid of model.safetensors = sha256
modal volume get laya-checkpoints smolvlm/<run>/best/model.safetensors . && sha256sum model.safetensors
modal volume get laya-checkpoints smolvlm/<run>/metrics.json .       # args.datasets = what it trained on
```

## 3. Make sure the data exists

`modal volume ls laya-datasets vqa` should show the groups `evaluate` reads (sets without `_READY` are skipped
with "not ready", not an error):

| group | names | prepared by |
|---|---|---|
| `vqa` | `aokvqa`, `scienceqa`, `vqav2_yesno` | the `siglip-projector-experiment` branch |
| `cauldron` | `cauldron_<19 subsets>` | `modal run modal_app.py::prepare_cauldron` |
| `score` | `score_{vlfeedback,ava,richhf,crisismmd}` | `modal run modal_app.py::prepare_score` |
| `eval` | `eval_{koniq,evalmuse,cifar10h,ferplus,vizwiz,pope_random,pope_popular,pope_adversarial}` | `modal run modal_app.py::prepare_eval` (~25 min, VizWiz is the slowest) |

Atari baselines come from `/data/atari/expert/<Game>/meta.json` on `laya-datasets` (Freeway and Breakout have
one; Galaxian does not, so it is scored against random play only).

## 4. Run

One command, three parts in parallel, one results file:

```bash
modal run modal_app.py::full_eval --model <run>/best [--parts datasets,games,latency] \
    [--datasets vqa,cauldron,score,eval] [--val-split test] [--out eval-results/<name>.json]
```

- **No `--detach`.** The local entrypoint gathers the results and writes the files; a detached run whose client
  goes away loses them. For long runs start it as a background shell command and wait for it.
- Runtime on the last run: datasets part 16.5 min (59k questions), games and latency together 12 min.
- It writes `eval-results/<run>-<commit>[-dirty][-parts].json` locally and `<run>/evals/<leaf>-<time>-<commit>.json`
  on the volume (beside the checkpoint, outside `best/`, so `publish` never uploads it). Exit code is non-zero if
  a requested part produced nothing; single failed games are listed under `errors`.
- Split the parts to start early: `--parts games,latency` does not need the eval datasets, so it can run while
  `prepare_eval` is still going, then `--parts datasets` afterwards. `eval_report.py` merges the files.

Single pieces, when that is all you need: `evaluate --run-name` (prints only, saves nothing), `games_eval`,
`maze_eval` / `snake_eval` / `doom_eval --models a/best,b/best` to compare checkpoints on one game,
`modal_atari_train.py::atari_eval` for every trained Atari game, `bench_latency --run-name`.

From GitHub: Actions → **eval** → Run workflow on the checkpoint's branch, or
`gh workflow run eval.yml --ref <branch> -f model=<run>/best`. Each part is a job streaming the Modal logs; the
report lands in the run summary and as a comment on the branch's open PR (updated in place per checkpoint).
Needs the `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` repo secrets.

## 5. Report

```bash
python scripts/eval_report.py eval-results/<name>*.json                                   # PR-comment Markdown to stdout
python scripts/eval_report.py eval-results/<name>*.json --title "<Name> scorecard" --doc docs/evals/<name>.md
python scripts/eval_report.py eval-results/<name>*.json --title "<Name> scorecard" --html <scratch>/report.html
```

- `--doc` is the one to commit: deterministic (same inputs, same bytes), links its result files, and its charts
  are Mermaid `xychart-beta` blocks that GitHub renders. Commit the result JSONs in `eval-results/` with it and
  link the doc from the README if it is a released checkpoint. `docs/evals/laya-vision.md` is the example.
- `--html` is a self-contained page (no scripts) for reading; publish it as an artifact rather than committing it.
- Standard library only; no torch needed.
- If you change the charts, check every Mermaid block still parses (render them with mermaid@11 in the
  preinstalled Chromium, `/opt/pw-browsers/chromium-*/chrome-linux/chrome`, through the proxy).

## 6. Reading the numbers

- **accuracy / ECE / NLL** are on calibrated probabilities (the checkpoint's per-type temperatures), as `predict`
  returns them. ECE under ~0.03 means probabilities can be taken at face value.
- **Sets with human votes** (KonIQ, EvalMuse, CIFAR-10H, FER+, VizWiz, the `score` sets): judge them by
  `soft_xent` / `xent` against `prior_...`, the cross-entropy of always predicting the set's average vote. Lower
  than the prior = the model learned something per image. ECE there compares confidence with the single
  most-voted answer, which a model trained to spread probability fails by design (AVA's ECE 0.39 is not a bug).
- `mae` on `score` sets is levels between the model's expected level and the voters'.
- `prior_*` is absent on the pooled `all` row when sets with different option counts are mixed.
- **Atari** `normalized` = (model - random) / (expert - random). A checkpoint not trained on a game plays at
  random-level or degenerate constant-action policies (look at the top actions); ViZDoom "identical to
  always-attack" means it only shoots.
- **Latency** from `bench_latency` times the whole `predict` call, CPU preprocessing included: 79 ms median on the
  last run against ~41 ms in the README's checkpoint table. Preprocessing is the likely difference (not yet
  confirmed), so compare latencies only within `bench_latency` runs.

## 7. Adding an eval

- **Dataset**: a pure converter in `laya/evalsets.py` returning prepared-dataset records
  (`{"id","image","state_text","question","label"[,"target"]}`; `target` = vote histogram), a unit test in
  `tests/test_evalsets.py` with rows shaped like the source, a branch in `_eval_source` in `modal_app.py`, and
  the name in `EVAL_SOURCES`. Smoke-test the source on real data locally before a Modal run: iterate
  `modal_app._eval_source(name, random.Random(0), 50, 10.0, 0, {})` for a few items and pass each record through
  `laya.vlm_train.jsonl_example`.
- **Grid game**: an env class in `laya/gridgames.py` with `step`, `done`, `render`, `expert`, plus its question in
  `laya/games.py`; `play_episodes` and the Modal functions pick it up by name. Keep eval seeds at `GRID_SEED`.
- Run `python -m pytest tests/test_evalsets.py tests/test_gridgames.py tests/test_metrics.py tests/test_eval_report.py`
  (CPU, no downloads).
