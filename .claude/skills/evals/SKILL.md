---
name: evals
description: Run, read and report Laya Vision evals on Modal. Use whenever the task is to evaluate, benchmark or score a checkpoint (a run on the laya-checkpoints volume, or a Hugging Face model such as thaitea/laya-vision), compare checkpoints, run the dataset evals / games suite / latency benchmark, prepare eval datasets, trigger the eval GitHub workflow, write or commit an eval report (site-docs/reference/evals/*.md, HTML scorecard), add a new eval set or game, or explain eval numbers (accuracy, ECE, soft_xent vs prior, normalized Atari score). Also covers the robustness / perturbation evals, row-level evidence and `benchmarks/verify_published.py`, the typed-vs-generated-JSON benchmark, the prefix-cache and CUDA-graph game-step benchmarks, fitting `calibrate()` on a user's labelled data, and getting the Modal client to connect from a Claude Code cloud sandbox.
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
`maze_eval` / `snake_eval` / `control_eval` / `doom_eval --models a/best,b/best` to compare checkpoints on one game,
`modal_atari_train.py::atari_eval` for every trained Atari game, `bench_latency --run-name`. Evals outside
`full_eval` (robustness, row-level evidence, benchmarks) are in section 8.

From GitHub: Actions → **eval** → Run workflow on the checkpoint's branch, or
`gh workflow run eval.yml --ref <branch> -f model=<run>/best`. Each part is a job streaming the Modal logs; the
report lands in the run summary and as a comment on the branch's open PR (updated in place per checkpoint).
Needs the `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` repo secrets.

## 5. Report

```bash
python scripts/eval_report.py eval-results/<name>*.json                                   # PR-comment Markdown to stdout
python scripts/eval_report.py eval-results/<name>*.json --title "<Name> scorecard" --doc site-docs/reference/evals/<name>.md
python scripts/eval_report.py eval-results/<name>*.json --title "<Name> scorecard" --html <scratch>/report.html
```

- `--doc` is the one to commit: deterministic (same inputs, same bytes), links its result files, and its charts
  are Mermaid `xychart-beta` blocks that GitHub renders. Commit the result JSONs in `eval-results/` with it and
  link the doc from the README if it is a released checkpoint. `site-docs/reference/evals/laya-vision.md` is the example.
  A doc under `site-docs/` is a page of the documentation site: add it to the `nav` in `mkdocs.yml` (Reference ›
  Results) and build with `nix build .#docs` (or `mkdocs build --strict`); its result-file links point at GitHub,
  since MkDocs cannot link outside `site-docs/`.
- `--html` is a self-contained page (no scripts) for reading; publish it as an artifact rather than committing it.
- Standard library only; no torch needed.
- If you change the charts, check every Mermaid block still parses (render them with mermaid@11 in the
  preinstalled Chromium, `/opt/pw-browsers/chromium-*/chrome-linux/chrome`, through the proxy).

## 6. Reading the numbers

- **accuracy / ECE / NLL** are on calibrated probabilities (the checkpoint's per-type temperatures), as `predict`
  returns them. ECE under ~0.03 means probabilities can be taken at face value.
- **ECE floor** (`ece_floor`, `ece_floor_p95` next to each set's `ece`, and on the pooled `all`): ECE on n rows is
  biased upward, so read it against the ECE a *perfectly calibrated* model scores on the same confidences and row
  count (`laya.robustness_floor.ece_floor_fields`: correctness redrawn as Bernoulli(confidence) 200 times, same 15
  bins, seeded by the set's name). On a few hundred rows the floor is ~0.05; on the 59k pooled rows ~0.005. An
  ECE at or under `ece_floor_p95` is sampling noise, not miscalibration; the report flags a hard-label set only
  above its p95 *and* above 0.03, and falls back to the fixed 0.10 for results from before the floor (they show no
  floor column). `evaluate` computes it for every set type, because `metrics_from` scores every type's ECE the same
  way (max probability; argmax == label, the most-voted answer on vote sets), so the floor is the right null for
  the number printed; on vote sets that ECE is still not the thing to judge. Rows count as independent: sets with
  several questions per image (VQAv2, Cauldron) have a somewhat higher true floor. It is not a confidence interval
  for the ECE.
- **Sets with human votes** (KonIQ, EvalMuse, CIFAR-10H, FER+, VizWiz, the `score` sets): judge them by
  `soft_xent` / `xent` against `prior_...`, the cross-entropy of always predicting the set's average vote. Lower
  than the prior = the model learned something per image. ECE there compares confidence with the single
  most-voted answer, which a model trained to spread probability fails by design (AVA's ECE 0.39 is not a bug).
- `mae` on `score` sets is levels between the model's expected level and the voters'.
- `prior_*` is absent on the pooled `all` row when sets with different option counts are mixed.
- **Atari** `normalized` = (model - random) / (expert - random). A checkpoint not trained on a game plays at
  random-level or degenerate constant-action policies (look at the top actions); ViZDoom "identical to
  always-attack" means it only shoots.
- **Classic control** `normalized` is the same formula with a scripted controller as the expert, both baselines
  played live on the same seeds. Returns are negative per step in Acrobot and MountainCar, so a model that never
  finishes scores exactly the random baseline (-500 / -200, normalized 0). `solved` is the share of episodes at
  the environment's solved score (CartPole 475, Acrobot -100, MountainCar -110, LunarLander 200). One step costs
  one `predict`, so a good CartPole run is 5,000 calls; the games part takes longer than the 12 min above.
- **GPU type**: `evaluate` runs on any of A10G, L4 or A100, and bf16 scores shift slightly between them (up to
  about a point on a ~100-question set, 0.01 points pooled). Its result records the GPU (`datasets.gpu`, shown in
  the report's datasets heading); compare two runs' dataset numbers only on the same GPU type. Results from before
  this field have none: `thaitea/laya-vision`'s baseline in `eval-results/laya-vision-datasets.json` ran on an L4.
- **Provenance**: every `predict` output carries a `provenance` block (prompt-format version, checkpoint and
  backbone revisions, dtype, device, library versions, `input_ids_sha256`, temperatures used). Two results with
  the same `input_ids_sha256` scored the same token sequences.
- **Latency** from `bench_latency` times the whole `predict` call, CPU preprocessing included: 79 ms median on the
  last run against ~41 ms in the README's checkpoint table. Preprocessing is the likely difference (not yet
  confirmed), so compare latencies only within `bench_latency` runs. It asks one question per call, so on
  CUDA `predict`'s automatic prefix cache stays off (it turns on only past `batch_size` rows) and runs before and
  after the cache landed are comparable; `bench_prefix_cache` measures the cache itself.

## 7. Adding an eval

- **Dataset**: a pure converter in `laya/evalsets.py` returning prepared-dataset records
  (`{"id","image","state_text","question","label"[,"target"]}`; `target` = vote histogram), a unit test in
  `tests/test_evalsets.py` with rows shaped like the source, a branch in `_eval_source` in `modal_app.py`, and
  the name in `EVAL_SOURCES`. Smoke-test the source on real data locally before a Modal run: iterate
  `modal_app._eval_source(name, random.Random(0), 50, 10.0, 0, {})` for a few items and pass each record through
  `laya.vlm_train.jsonl_example`.
- **Grid game**: an env class in `laya/gridgames.py` with `step`, `done`, `render`, `expert`, plus its question in
  `laya/games.py`; `play_episodes` and the Modal functions pick it up by name. Keep eval seeds at `GRID_SEED`.
- **Classic control game** (any discrete-action Gymnasium env): an entry in `GAMES` in `laya/controlgames.py`
  (env id, action names in the env's order, solved score), a scripted expert in `_expert_action`, and its goal and
  action wording in `CONTROL_GOALS` / `CONTROL_ACTIONS` in `laya/games.py`; add it to `SUITE_CONTROL_GAMES` in
  `modal_app.py` to put it in the suite. `tests/test_controlgames.py` checks the expert beats random.
- Run `python -m pytest tests/test_evalsets.py tests/test_gridgames.py tests/test_controlgames.py tests/test_metrics.py tests/test_eval_report.py`
  (CPU, no downloads).

## 8. Robustness, row-level evidence and benchmarks

These sit outside `full_eval`. All write **create-only** outputs: each refuses an existing path, so pass a new
`--out` / `--name` / `--tag` / `--output`. `AGENTS.md` has the rules for publishing a number.

| What | Command | Output | Time on an L4 |
|---|---|---|---|
| Robustness: option order, rewording, image corruptions, shuffled-image and no-image controls | `modal run --detach modal_app.py::robustness_eval [--run <run>/best] [--n 300] [--families option_order,text,image,image_shuffle,text_only] [--tag <new>] [--out results/robustness-<name>]` | `predictions.jsonl.gz` + `summary.json` | 8.4 min for 7 sets x 300 |
| Row-level evidence behind the README table | `modal run modal_app.py::evidence [--run <run>/best] [--datasets vqa] [--name <new>]` | `results/raw/<name>.predictions.jsonl.gz` + `.meta.json`, regenerates `SHA256SUMS` | 1.9 min for the 3 official VQA splits |
| Typed readout vs the base backbone generating JSON | `modal run modal_app.py::decision_vs_generation --output results/raw/<new>.json [--run <run>/best]` | raw report (timeline, outputs, revisions) | a few min |
| Prefix cache vs full path in `predict` | `modal run modal_app.py::bench_prefix_cache [--run-name <run>/best]` | prints only | a few min |
| CUDA-graph game step (`laya/static_step.py`) | `modal run modal_game_cache.py::main [--out <file>]`, parity: `modal_game_cache.py::verify` | JSON | a few min |

- **Robustness `--detach` is the exception** to section 4's rule: the job writes its results to
  `/ckpt/smolvlm/robustness/<tag>/` on the volume, so they survive a dropped client. The entrypoint copies them
  into `--out` when it is still attached; otherwise
  `modal volume get laya-checkpoints smolvlm/robustness/<tag>/ <out>`. Re-summarise offline with
  `python -m laya.robustness <out>/predictions.jsonl.gz`.
- **Reading robustness**: per dataset and family, group-averaged accuracy (a perturbed row counts once per source
  row), the change from unperturbed with a paired 95% bootstrap interval over image clusters, and the flip rate
  (how often the answer changes). The controls are the check that the model uses the image: shuffled-image and
  no-image accuracy should fall toward the majority-label rate. On `cauldron-score-2ep-bidir-full/best` they fall
  23-46 points on 6 of 7 sets, but **`cauldron_mapqa` fails**: removing the map changes nothing and accuracy sits
  below the majority rate. A new checkpoint that still fails there has not learned MapQA.
- **After `evidence` (or any change under `results/raw/`)**: run `python benchmarks/verify_published.py`. It
  checks `SHA256SUMS` in both directions (every file under `results/raw/` except `README.md` must be listed),
  recomputes accuracy and calibrated ECE from the rows, and compares them with the metrics JSON and the README
  table cells named in `results/claims.json`. It exits non-zero on any mismatch; fix the claim or the evidence,
  never loosen the tolerance to pass. Also `(cd results/raw && sha256sum -c SHA256SUMS)`.
- **decision_vs_generation** is a systems comparison (the base 256M model cannot follow the JSON instruction),
  and its README numbers predate the prefix cache; rerun to a new `--output` before quoting a new speed.
- **Calibrating for a user's workload** is not a Modal job: `agent.calibrate(rows, group_key="image_id")` on their
  labelled rows returns a `Calibration` with raw, checkpoint-temperature and out-of-fold ECE and 95% intervals,
  and `predict(..., calibration=cal)` applies it (see the README section "Calibrating on your own data"). Keep
  `n_permutations` the same at fit and use time. Accuracy never changes with temperature, only confidence.
- Tests for these, CPU, no GPU: `python -m pytest tests/test_robustness.py tests/test_verify_published.py
  tests/test_calibration.py tests/test_decision_vs_generation.py`. The model tests (`test_vlm.py`,
  `test_smolvlm2.py`, `test_modernvbert.py`) are slow on a shared CPU; run them on an L4 with
  `modal run modal_app.py::test --backbones HuggingFaceTB/SmolVLM-256M-Instruct` (about 2 min).
