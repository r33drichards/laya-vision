---
name: modal
description: Run this repo's GPU jobs on Modal from a Claude Code on the web session - setting up the Modal CLI behind the session proxy, smoke-testing the workspace, launching modal_app.py / modal_atari_*.py jobs (test, finetune_long, split_bench, evaluate, bench_latency, publish), following detached runs, and pulling results off the laya-* volumes. Use whenever a task says "run on Modal", "modal run", "launch the benchmark/training/eval", "check the run", or needs files from laya-checkpoints / laya-datasets, and whenever a modal command fails with "Could not connect to the Modal server".
---

# Modal from Claude Code on the web

Everything here was checked in a web session (modal 1.5.5, 2026-09): the CLI works once the proxy extra is
installed, `modal run` uploads local code and mounts the volumes, `--detach` runs survive the session, and
volume reads/downloads work.

## 1. Setup (every new session)

```bash
bash .claude/skills/modal/check.sh
```

It installs `modal[api-proxy-support]` if needed, confirms credentials, prints the workspace, and checks that
the three volumes exist. What it fixes and what it can't:

| Symptom | Cause | Fix |
|---|---|---|
| `Could not connect to the Modal server` on every command | Plain `modal` ignores `HTTPS_PROXY` without `python-socks` | `uv tool install --force 'modal[api-proxy-support]'` (check.sh does this) |
| `ImportError: A proxy is configured ... 'python-socks' package is not installed` | Same, from Python | Same |
| No `~/.modal.toml` and no `MODAL_TOKEN_ID` | The environment has no Modal credentials | Ask the user to add `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` to the environment's variables. Never ask them to paste a token into chat |
| 403 for `api.modal.com` in `curl -sS "$HTTPS_PROXY/__agentproxy/status"` | The network policy blocks Modal | Report it and point the user to https://code.claude.com/docs/en/claude-code-on-the-web. Don't route around it |

Then run the cheap end-to-end probe before spending GPU time (1 CPU, about 15 s):

```bash
modal run .claude/skills/modal/smoke.py
```

It prints the uploaded file count, the checkpoint roots, and every prepared dataset (`/data/vqa/<name>/_READY`).
If a job needs a dataset that isn't listed, prepare it first (`prepare_cauldron`, `prepare_score`, ...).

## 2. CLI habits in this sandbox

- **Wrap every call in `timeout`** (`timeout 90 modal volume ls ...`). The CLI can hang on a bad connection, and
  a Bash call that runs past 120 s gets moved to the background.
- **Always run from the repo root.** `modal_app.py` uploads `tests/` and `laya/` using relative paths.
- **Non-interactive prompts need flags:** `modal app stop -y <app-id>`, `modal volume get --force ...`.
- **`modal app logs <id> -f` never returns.** Use `modal app logs <id> --tail 50` (it fetches and exits), or put
  `-f` under `timeout 30`. A brand-new app can return `Logs query hit resource limit`; retry a minute later, or read
  the run's files on the volume.
- **The workspace is shared.** Other sessions and the user run jobs here too (see `modal app list`). Only stop apps
  whose ID you launched.
- **Never `modal deploy`** these apps. They are run-only, and the volumes were created separately.

## 3. The repo's Modal surface

`modal_app.py` (app `laya-smolvlm`) lists every entrypoint in its module docstring; read it before launching.
The Atari jobs are in `modal_atari_*.py`.

| Volume | Mount | Contents |
|---|---|---|
| `laya-hf-cache` | `/cache/hf` | HF_HOME: shared model weights |
| `laya-datasets` | `/data` | `vqa/<name>/{train,val}.jsonl, images/, _READY`; also `atari/`, `mmad/` |
| `laya-checkpoints` | `/ckpt` | `smolvlm/`, `smolvlm2/`, `modernvbert/`, ... (`CKPT_ROOTS` in `modal_app.py`) |

- **Volume CLI paths are relative to the volume root, not the mount:**
  `modal volume ls laya-checkpoints smolvlm2/split-bench`, not `/ckpt/smolvlm2/...`.
- **Run names** passed to `evaluate`, `bench_latency`, `--init-from` and similar are relative to `/ckpt/smolvlm`
  or to `/ckpt`, so a SmolVLM2 run is `smolvlm2/<run>/best`.
- **Secrets:** `huggingface-thaitea` (used by `publish*` and `prepare_*`), and `laya-otel` (`LAYA_OTLP_TOKEN` for
  `laya.telemetry`, used by the training, eval and game jobs; `LAYA_OTEL_SECRET=` runs without it). Pushing to the Hub publishes; confirm with
  the user first.

Typical order for new training code:

1. `modal run modal_app.py::test`: pytest with real weights on an L4 (test_vlm, test_smolvlm2, test_modernvbert).
   Fix any failure, commit and push before running anything long.
2. A short `finetune --minutes 18` or a `finetune_long --max-train 500 --epochs 0.2` to shake out the data path.
3. The real run with `--detach`.

## 4. Long runs

```bash
timeout 300 modal run --detach modal_app.py::split_bench 2>&1 | tee "$SCRATCH/launch.log"
grep -o 'ap-[A-Za-z0-9]*' "$SCRATCH/launch.log" | head -1      # keep the app ID
```

(`$SCRATCH` is your scratchpad directory.) With `--detach`, the app keeps running after the local client exits or the
session's container is reclaimed. `timeout` only cuts off the local log stream. Without `--detach`, killing
the client stops the app.

Don't loop on `sleep` while you wait. Schedule a check-in (`send_later`, about 30–60 min) or use Monitor. On each check:

```bash
timeout 60 modal app list | grep <app-id>                    # state: ephemeral (detached) / stopped
timeout 60 modal app logs <app-id> --tail 40                  # step / loss / eval lines
timeout 60 modal volume ls laya-checkpoints <root>/<run>      # best/, last/, metrics.json, state.pt
timeout 90 modal volume get --force laya-checkpoints <root>/<run>/metrics.json "$SCRATCH/"
```

Lessons from earlier runs:

- **Preemption happens.** `finetune_long` saves `state.pt` and resumes from it, but a parent job that fans out work
  (`split_bench`) can be restarted from the top. After a preemption, its `results.md` / `results.json` may never
  be written. Rebuild the table from each run's `metrics.json` and run `bench_latency` on each `best` separately.
  This happened in the split benchmark; see docs/split-bench.md.
- **Confirm a run wasn't cut short:** compare `train_stats.steps` with `args.steps`. The `max_minutes` limit stops
  training quietly.
- **Memory:** image splitting at 2048 means up to 17 views per image. `split_bench` defaults to `A100-80GB` so
  batch 32 fits (it also avoids H100 requests landing on H200s). If you retry one setting with a smaller batch or a different GPU, say so in the write-up, because
  the comparison is no longer like-for-like. DataLoader workers can also run out of `/dev/shm`; `_loader_fit` sizes
  them to it.
- **GPU capacity:** a job pinned to one GPU type can wait in the queue. `evaluate` accepts `["A10G", "L4", "A100"]`
  for this reason. Use the same pattern (`.with_options(gpu=[...])`) for short jobs where the GPU type doesn't matter.
- **Latency numbers** in the README and docs come from `bench_latency` (L4, bf16). Use it so new numbers compare
  with the old ones.

## 5. After a run

Copy the metrics you cite into `docs/<run>-metrics.json`, write the results into `docs/`, link them from the
README, and commit. Stop any detached app you launched that is still running (`modal app stop -y <id>`) once
its outputs are on the volume.
