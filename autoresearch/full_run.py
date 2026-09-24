"""Full-length training of an autoresearch recipe, measured exactly as the 15-minute harness measures it.

    modal run --detach autoresearch/full_run.py --commit <sha> --name <run-name> [--minutes 120] [--data full|pool]
    modal run autoresearch/full_run.py --fetch --name <run-name>      # pull the result JSON if the client went away
    modal run autoresearch/full_run.py --prepare-pool                 # build the full data pool once (CPU only)

The autoresearch loop (``harness.py``) ranks recipes on 15 minutes of training. This file takes the recipe a commit
holds (``git show <sha>:autoresearch/experiment.py``) and trains it for ``--minutes`` instead, with the experiment's
own ``build`` / ``train`` and ``ctx.time_budget_s = minutes * 60``, then measures the checkpoint through the
harness's own code paths so the numbers sit on the same axes as ``autoresearch/runs/<tag>/*.json``:

1. temperatures fitted on the harness calibration set (the last ``N_CALIB`` train records of each calibration set,
   from the harness pool), exactly as ``TrainEval.run`` does;
2. the model saved to ``/ckpt/autoresearch/full/<name>/best/`` and reloaded from disk;
3. quality on the harness eval pool (``EVAL_PER_SET`` seeded val questions per ``EVAL_DATASETS`` set), with
   ``harness._summary``;
4. ``latency_x`` and the games by calling the harness's deployed ``Latency`` and ``Games`` classes (the
   ``laya-autoresearch-<hash>`` deployment of the unchanged harness, the same one the autoresearch tag uses), with
   ``tag="full/<name>"`` and ``commit="best"``, which is where they look for the checkpoint.

The result is written to ``autoresearch/runs/full/<name>.json`` in the harness's result shape (``summary`` holds
quality / macro_acc / ece_hard / games / params_m / latency_x, so ``pareto.py`` reads it), plus a ``full_run``
block (minutes, data, samples and passes per dataset, resume count, the harness version that measured it). A copy is
kept on the volume as ``/ckpt/autoresearch/full/<name>/result.json``.

Data (``--data``). A 15-minute run sees ~165k samples (5,150 steps x 32 on the 20-layer recipe), which the harness
pool (``pool-v2``: up to 6,000 per set) covers in about one pass. Two hours at that rate is ~1.3M samples: ~40k draws
per VQA set at the recipes' equal-per-set mix (~120k for ``score_vlfeedback`` at weight 3), so the 6,000-per-set pool
would be cycled 7x. Measured on the volume (``laya-datasets``, 2026-09-24):

* the trainable train splits minus the calibration tail hold 485,871 records, 509,611 image references and 225,062
  unique images, 14.5 GB of unique image bytes (30.8 GB if every record carried its own copy; ``visual7w``,
  ``clevr``, ``figureqa``, ``mapqa`` ask ~3.5 questions per image); the three expert game sets are 20,000 frames
  each (0.7 GB);
* streaming instead is not viable: ``harness._with_bytes`` from one container reads 27-34 files/s with 32 threads,
  32-42 with 128 and 36-52 with 256 (vqav2, richhf, ocrvqa samples of 2,000), against the ~100 image reads/s the
  VQA share of training needs.

So ``--data full`` (the default) trains on a second, versioned pool, ``pool-full-v1``: every train record of every
``TRAINABLE_DATASETS`` set minus the calibration tail (the same rule as the harness) and every expert frame of the
``GAME_DATASETS``, image bytes inline, one image file read once per shard and shared by the records that ask about
it (so memory holds the 14.5 GB, not 30.8 GB). Records are sharded by their image (``SHARD_RECORDS`` per shard) and
built in parallel, one CPU container per shard. At 120 minutes the big sets see about one pass (``vlfeedback`` 0.75,
``visual7w`` / ``figureqa`` / ``clevr`` / ``mapqa`` / ``richhf`` 1.0-1.3, ``vqav2`` 2.0); small sets repeat, as they
do in the harness, because the recipes draw sets equally. Built 2026-09-24: 83 shards, 545,871 records (485,871
VQA + 60,000 game frames), 285,062 unique images, 15.4 GB; ~50 min wall with all 83 containers reading the volume
at once (shards took 4-23 min each, contention), which only has to happen once. Loading it into the training
container takes ~2 min; the 5e9ef40 recipe then trains at 5.8 steps/s with 2-3% data wait, as in the harness.
``--data pool`` trains on the harness pool itself (for a
like-for-like check of this tool against a harness result at ``--minutes 15``). Calibration and eval always come
from the harness pool. Val splits, ``eval_*`` sets and the calibration tail are never in either training pool.

Durability. The training function is retried on preemption. ``laya.vlm_train.train`` is wrapped for the
experiment's call with ``finetune_long``'s mechanism: ``save_state_fn`` writes ``state.pt`` (weights, optimizer,
step, RNG, samples per dataset, elapsed training time) atomically every ``--state-every-min`` minutes, and a retry
rebuilds the experiment, loads ``state.pt`` and passes it back as ``resume``, so the LR schedule and the time budget
continue across attempts. This covers recipes that call ``laya.vlm_train.train`` once (every recipe so far); a
recipe with its own loop runs, but a preemption restarts it. The orchestrating CPU function records the training
call's id on the volume, so its own retry waits on the same call instead of starting a second trainer.
``--crash-at-step N`` raises once, after the first state write past step N, to test the resume path.

Create-only: a name whose directory exists under ``/ckpt/autoresearch/full/`` is refused, as is an existing local
result JSON; ``state.pt`` is deleted once the run has saved its checkpoint. ``--full-eval`` then runs
``modal run modal_app.py::full_eval --model autoresearch/full/<name>/best`` (the repo's full eval suite; see
``.claude/skills/evals/SKILL.md``), which writes ``eval-results/`` locally and ``<name>/evals/`` on the volume.

Do not edit ``harness.py``, ``games_eval.py``, ``game_baselines.json``, ``toolkit.py`` or ``laya/`` for this: they
are hashed into the autoresearch tag. This file imports ``harness`` for its definitions and ships it into its image.
"""
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
import uuid
from typing import Dict, List, Optional

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import harness as H  # noqa: E402  (the fixed harness: constants, volumes, image, pool, Context, _summary)

FULL_POOL_VERSION = "full-v1"
FULL_POOL_DIR = "/data/autoresearch/pool-" + FULL_POOL_VERSION
SHARD_RECORDS = 8000        # records per pool shard (one CPU container reads each shard's images, ~40 files/s)
FULL_ROOT = os.path.join(H.ROOT, "full")    # /ckpt/autoresearch/full/<name>/
MAX_MINUTES = 20 * 60       # the training function's 24 h timeout, less loading, calibration and eval
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")

app = modal.App("laya-autoresearch-full")
image = H.image.add_local_file(os.path.join(HERE, "harness.py"), "/root/harness.py")


# ------------------------------------------------------------------------------------------------ the full pool


def full_selection(kind: str, name: str) -> List[Dict]:
    """Every record of pool part ``kind`` (``train`` or ``games``) for ``name``, in file order; the calibration
    tail is held out exactly as ``harness.pool_selection`` holds it out."""
    from laya.vlm_train import load_jsonl_examples

    if kind == "games":
        root, prepared = H.GAME_DATASETS[name]
        return [dict(ex, dataset=name) for ex in load_jsonl_examples(root, prepared, "train")]
    if kind != "train":
        raise ValueError(kind)
    exs = H.load_split(name, "train")
    return exs[:-H.N_CALIB] if name in H.CALIB_DATASETS else exs


FULL_PARTS = [("train", n) for n in H.TRAINABLE_DATASETS] + [("games", n) for n in H.GAME_DATASETS]


def _image_paths(ex: Dict) -> List[str]:
    s = ex["state"]
    if not isinstance(s, dict):
        return []
    return ([s["image"]] if isinstance(s.get("image"), str) else []) + [p for p in s.get("images") or []]


def _shard_of(ex: Dict, i: int, n_shards: int) -> int:
    """Records asking about the same image land in the same shard, so its bytes are read and stored once."""
    paths = _image_paths(ex)
    key = paths[0] if paths else "record:%d" % i
    return int(hashlib.md5(key.encode()).hexdigest()[:8], 16) % n_shards


def _shard_file(kind: str, name: str, shard: int, n_shards: int) -> str:
    return os.path.join(FULL_POOL_DIR, kind, name, "%03d-of-%03d.pkl" % (shard, n_shards))


@app.function(image=image, cpu=2, memory=8192, timeout=30 * 60, volumes={"/data": H.data_vol.read_only()})
def plan_part(kind: str, name: str) -> Dict:
    exs = full_selection(kind, name)
    n_img = sum(len(_image_paths(ex)) for ex in exs)
    uniq = len({p for ex in exs for p in _image_paths(ex)})
    return {"kind": kind, "name": name, "records": len(exs), "images": n_img, "unique_images": uniq,
            "shards": max(1, math.ceil(len(exs) / SHARD_RECORDS))}


@app.function(image=image, cpu=4, memory=16384, timeout=60 * 60, volumes={"/data": H.data_vol})
def build_shard(kind: str, name: str, shard: int, n_shards: int) -> Dict:
    """One shard: its records with the image bytes inline (one ``bytes`` object per distinct file, which pickle
    stores once), pickled to the full pool directory. Create-only: an existing shard is left alone."""
    import pickle
    from concurrent.futures import ThreadPoolExecutor

    t = time.time()
    path = _shard_file(kind, name, shard, n_shards)
    if os.path.exists(path):
        return {"kind": kind, "name": name, "shard": shard, "records": -1, "mb": round(os.path.getsize(path) / 1e6, 1),
                "seconds": 0.0}
    exs = full_selection(kind, name)
    idx = [i for i, ex in enumerate(exs) if _shard_of(ex, i, n_shards) == shard]
    paths = sorted({p for i in idx for p in _image_paths(exs[i])})

    def read(p):
        with open(p, "rb") as f:
            return f.read()

    with ThreadPoolExecutor(128) as pool:  # each small-file read is ~0.4 s of latency; overlap many
        data = dict(zip(paths, pool.map(read, paths)))
    out = []
    for i in idx:
        ex = exs[i]
        state = ex["state"]
        if isinstance(state, dict):
            state = dict(state)
            if isinstance(state.get("image"), str):
                state["image"] = data[state["image"]]
            if state.get("images"):
                state["images"] = [data[p] for p in state["images"]]
            ex = dict(ex, state=state)
        out.append(ex)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "wb") as f:
        pickle.dump({"kind": kind, "name": name, "shard": shard, "n_shards": n_shards, "indices": idx,
                     "examples": out}, f, protocol=5)
    os.replace(path + ".tmp", path)
    H.data_vol.commit()
    return {"kind": kind, "name": name, "shard": shard, "records": len(out), "unique_images": len(paths),
            "mb": round(os.path.getsize(path) / 1e6, 1), "seconds": round(time.time() - t, 1)}


@app.function(image=image, cpu=1, timeout=10 * 60, volumes={"/data": H.data_vol})
def write_manifest(manifest: Dict) -> str:
    """The manifest marks the pool complete; the loader refuses a pool without one. Create-only."""
    path = os.path.join(FULL_POOL_DIR, "manifest.json")
    H.data_vol.reload()
    if os.path.exists(path):
        raise FileExistsError("%s exists; a changed pool needs a new FULL_POOL_VERSION" % path)
    missing = [_shard_file(p["kind"], p["name"], s, p["shards"]) for p in manifest["parts"] for s in range(p["shards"])
               if not os.path.exists(_shard_file(p["kind"], p["name"], s, p["shards"]))]
    if missing:
        raise FileNotFoundError("%d shards missing, e.g. %s" % (len(missing), missing[0]))
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    H.data_vol.commit()
    return path


def build_full_pool() -> None:
    """``--prepare-pool``: plan every part, build every missing shard in parallel, then write the manifest."""
    t0 = time.time()
    plans = sorted(plan_part.starmap(FULL_PARTS), key=lambda p: FULL_PARTS.index((p["kind"], p["name"])))
    jobs = [(p["kind"], p["name"], s, p["shards"]) for p in plans for s in range(p["shards"])]
    print("full pool %s: %d parts, %d shards, %d records, %d unique images" % (
        FULL_POOL_DIR, len(plans), len(jobs), sum(p["records"] for p in plans), sum(p["unique_images"] for p in plans)))
    mb, failed, per_part = 0.0, 0, {}
    for r in build_shard.starmap(jobs, order_outputs=False, return_exceptions=True):
        if isinstance(r, Exception):
            failed += 1
            print("FAILED:", repr(r)[:300])
            continue
        mb += r["mb"]
        per_part[(r["kind"], r["name"])] = per_part.get((r["kind"], r["name"]), 0.0) + r["mb"]
        print("%-6s %-28s shard %3d %6d records %8.1f MB %6.1f s" % (r["kind"], r["name"], r["shard"], r["records"],
                                                                     r["mb"], r["seconds"]))
    if failed:
        raise SystemExit("%d shards failed; re-run --prepare-pool to build the missing ones" % failed)
    manifest = {"version": FULL_POOL_VERSION, "dir": FULL_POOL_DIR, "shard_records": SHARD_RECORDS,
                "harness": {"n_calib": H.N_CALIB, "trainable": list(H.TRAINABLE_DATASETS),
                            "calib": list(H.CALIB_DATASETS), "games": {k: list(v) for k, v in H.GAME_DATASETS.items()}},
                "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "code": _git_state(),
                "parts": [dict(p, mb=round(per_part.get((p["kind"], p["name"]), 0.0), 1)) for p in plans],
                "total_mb": round(mb, 1), "build_seconds": round(time.time() - t0, 1)}
    print("wrote", write_manifest.remote(manifest))
    print("full pool: %.1f GB in %.1f min" % (mb / 1e3, (time.time() - t0) / 60))


def load_full_pool() -> Dict[str, Dict[str, List[Dict]]]:
    """``{"train": {name: examples}, "games": {name: examples}}`` from the full pool, each set in file order."""
    import pickle
    from concurrent.futures import ThreadPoolExecutor

    mpath = os.path.join(FULL_POOL_DIR, "manifest.json")
    if not os.path.exists(mpath):
        raise FileNotFoundError("the full data pool %s is missing or incomplete (no manifest.json); build it with: "
                                "modal run autoresearch/full_run.py --prepare-pool" % FULL_POOL_DIR)
    with open(mpath) as f:
        manifest = json.load(f)
    if manifest["harness"]["n_calib"] != H.N_CALIB or manifest["harness"]["trainable"] != list(H.TRAINABLE_DATASETS):
        raise RuntimeError("the full pool was built for other harness datasets / calibration tail; build a new version")
    files = [_shard_file(p["kind"], p["name"], s, p["shards"]) for p in manifest["parts"] for s in range(p["shards"])]

    def read(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    parts: Dict = {}
    with ThreadPoolExecutor(16) as pool:
        for blob in pool.map(read, files):
            parts.setdefault((blob["kind"], blob["name"]), []).extend(zip(blob["indices"], blob["examples"]))
    out: Dict[str, Dict[str, List[Dict]]] = {"train": {}, "games": {}}
    for p in manifest["parts"]:
        rows = sorted(parts.get((p["kind"], p["name"]), []), key=lambda r: r[0])
        if len(rows) != p["records"]:
            raise RuntimeError("%s/%s: %d records in the shards, manifest says %d" % (p["kind"], p["name"], len(rows),
                                                                                      p["records"]))
        out[p["kind"]][p["name"]] = [ex for _, ex in rows]
    return out


# ------------------------------------------------------------------------------------------------ training


def _run_dir(name: str) -> str:
    return os.path.join(FULL_ROOT, name)


def _log(t_start: float, msg: str) -> None:
    print("[full_run %7.1f s] %s" % (time.time() - t_start, msg), flush=True)


@app.function(image=image, cpu=1, timeout=5 * 60, volumes={"/ckpt": H.ckpt_vol})
def claim(name: str, token: str, meta: Dict) -> str:
    """Create ``/ckpt/autoresearch/full/<name>/`` for this launch, or refuse: results are create-only."""
    H.ckpt_vol.reload()
    d = _run_dir(name)
    if os.path.exists(d):
        raise FileExistsError("%s exists on laya-checkpoints; pick a new --name (runs are create-only)" % d)
    os.makedirs(d)
    with open(os.path.join(d, "launch.json"), "w") as f:
        json.dump(dict(meta, token=token), f, indent=2)
    H.ckpt_vol.commit()
    return d


@app.function(image=image, cpu=1, memory=1024, timeout=5 * 60, volumes={"/ckpt": H.ckpt_vol})
def claim_record(name: str) -> Optional[Dict]:
    """The claim a launch wrote for ``name`` (its token and metadata), for ``--attach``; None if unclaimed."""
    H.ckpt_vol.reload()
    path = os.path.join(_run_dir(name), "launch.json")
    return json.load(open(path)) if os.path.exists(path) else None


def _check_owner(name: str, token: str) -> None:
    path = os.path.join(_run_dir(name), "launch.json")
    if not os.path.exists(path) or json.load(open(path))["token"] != token:
        raise PermissionError("%s belongs to another launch (or was not claimed); refusing to write into it" % path)


def _durable_train(real_train, out_dir: str, resume: Optional[Dict], every_min: float,
                   crash_at_step: int, record: Dict):
    """``laya.vlm_train.train`` with ``finetune_long``'s resume / ``save_state_fn`` wiring added, for the experiment's
    (single) call. ``record`` receives the loop's ``stats`` and how the call was made."""
    import torch

    state_path = os.path.join(out_dir, "state.pt")
    crashed = os.path.join(out_dir, "crashed")

    def wrapped(model, processor, examples, *args, **kw):
        record["calls"] = record.get("calls", 0) + 1
        if record["calls"] > 1:
            raise RuntimeError("the recipe calls laya.vlm_train.train more than once; full_run.py's resume covers a "
                               "single call (write the phases as one call, or extend _durable_train)")
        wrote = {"n": 0}

        def save_state(step, tstate):
            ts = time.time()
            blob = {"train": tstate, "model": {k: v.detach().cpu() for k, v in model.state_dict().items()}}
            torch.save(blob, state_path + ".tmp")
            os.replace(state_path + ".tmp", state_path)  # atomic: a torn write never replaces a good state
            H.ckpt_vol.commit()
            wrote["n"] += 1
            print("  wrote state.pt at step %d (%.1f s)" % (step, time.time() - ts), flush=True)

        def probe(step):
            # simulated preemption for testing: once, after a state write past crash_at_step
            if step >= crash_at_step and wrote["n"] and not os.path.exists(crashed):
                open(crashed, "w").close()
                H.ckpt_vol.commit()
                raise RuntimeError("crash_at_step %d: simulated preemption at step %d" % (crash_at_step, step))
            return False

        injected = {}
        for k, v in (("resume", resume), ("save_state_fn", save_state), ("save_state_every_min", every_min)):
            if kw.get(k) is None:
                kw[k] = injected[k] = v
        if crash_at_step and kw.get("eval_fn") is None:
            kw.update(eval_fn=probe, eval_every=10)
        stats = kw.get("stats")
        if stats is None:
            stats = kw["stats"] = {}
        record["injected"] = sorted(k for k, v in injected.items() if v is not None)
        record["max_minutes"] = kw.get("max_minutes")
        out = real_train(model, processor, examples, *args, **kw)
        record["stats"] = stats
        return out

    return wrapped


@app.function(image=image, gpu="H100", cpu=16, memory=163840, timeout=24 * 60 * 60,
              volumes={"/cache/hf": H.hf_vol, "/data": H.data_vol.read_only(), "/ckpt": H.ckpt_vol},
              retries=modal.Retries(max_retries=5, initial_delay=10.0))
def train_eval(source: str, name: str, commit: str, token: str, minutes: float, data: str, state_every_min: float,
               crash_at_step: int) -> Dict:
    """Train the recipe for ``minutes`` (resuming from ``state.pt`` after a preemption), then calibrate, save,
    reload and score quality exactly as ``harness.TrainEval.run``. Idempotent: a finished run returns its result."""
    import random

    import torch

    import laya.vlm_train as vt
    from laya.vlm import VLMAgent
    from laya.vlm_train import collect_logits, fit_temperatures_from, metrics_from

    t_run = time.time()
    H.ckpt_vol.reload()
    _check_owner(name, token)
    run_dir = _run_dir(name)
    out_dir = os.path.join(run_dir, "best")
    done = os.path.join(run_dir, "train_result.json")
    if os.path.exists(done):
        _log(t_run, "already trained and scored; returning %s" % done)
        return json.load(open(done))
    attempts_path = os.path.join(run_dir, "attempts.json")
    attempts = json.load(open(attempts_path)) if os.path.exists(attempts_path) else []
    attempts.append({"started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                     "call": modal.current_function_call_id(), "gpu": torch.cuda.get_device_name(0)})
    with open(attempts_path, "w") as f:
        json.dump(attempts, f, indent=2)
    H.ckpt_vol.commit()
    _log(t_run, "attempt %d on %s" % (len(attempts), attempts[-1]["gpu"]))

    held = H.load_pool(("calib", "eval"))  # calibration and eval always from the harness pool
    if data == "full":
        pool = load_full_pool()
        pool_dir = FULL_POOL_DIR
    else:
        pool = H.load_pool(("train", "games"))
        pool_dir = H.POOL_DIR
    sizes = {n: len(v) for n, v in pool["train"].items()}
    sizes.update({n: len(v) for n, v in pool["games"].items()})
    _log(t_run, "data (%s): train %d examples, games %d, calib %d, eval %d" % (
        pool_dir, sum(len(v) for v in pool["train"].values()), sum(len(v) for v in pool["games"].values()),
        sum(len(v) for v in held["calib"].values()), sum(len(v) for v in held["eval"].values())))

    torch.manual_seed(H.SEED)
    random.seed(H.SEED)
    exp = H._import_experiment(source)
    ctx = H.Context(minutes * 60, pool["train"], pool["games"])
    t_setup = time.time()
    agent = exp.build(ctx)
    setup_s = time.time() - t_setup
    _log(t_run, "build %.1f s" % setup_s)

    state_path = os.path.join(run_dir, "state.pt")
    resume = None
    if os.path.exists(state_path):
        blob = torch.load(state_path, map_location="cpu", weights_only=False)
        agent.model.load_state_dict(blob["model"])
        agent.model.to(ctx.device)
        resume = blob["train"]
        del blob
        _log(t_run, "resuming from state.pt at step %d (%.1f min trained)" % (resume["step"], resume["elapsed_s"] / 60))

    record: Dict = {}
    real_train = vt.train
    vt.train = _durable_train(real_train, run_dir, resume, state_every_min, crash_at_step, record)
    try:
        t0 = time.time()
        exp.train(agent, ctx)
        attempt_train_s = time.time() - t0
    finally:
        vt.train = real_train
    prior_s = resume["elapsed_s"] if resume else 0.0
    train_s = prior_s + attempt_train_s
    if not record.get("calls"):
        print("WARNING: the recipe never called laya.vlm_train.train, so this run was not preemption-safe")
    if train_s > minutes * 60 + 60:
        print("WARNING: training took %.0f s, over the %.0f s budget (+60 s the harness allows)" % (train_s, minutes * 60))
    agent.model.eval()

    calib = [ex for n in H.CALIB_DATASETS for ex in held["calib"][n]]
    val = [ex for n in H.EVAL_DATASETS for ex in held["eval"][n]]
    temps = fit_temperatures_from(collect_logits(agent.model, agent.processor, calib, batch_size=32, num_workers=8))
    agent.temperature, agent.temperature_by_options = list(temps), {}
    agent.save(out_dir)
    H.ckpt_vol.commit()
    del agent
    torch.cuda.empty_cache()

    agent = VLMAgent(out_dir, device="cuda")
    params = sum(p.numel() for p in agent.model.parameters())
    records = collect_logits(agent.model, agent.processor, val, batch_size=32, num_workers=12)
    metrics = metrics_from(records, agent.temperature)
    hard = metrics_from([r for r in records if float(r["target"].max()) >= 0.999], agent.temperature)
    summary = H._summary(metrics, hard)
    summary["params_m"] = params / 1e6

    stats = record.get("stats") or {}
    seen = stats.get("samples_per_dataset", {})
    res = {"tag": "full", "commit": commit, "summary": summary, "metrics": metrics, "temperature": temps,
           "setup_s": round(setup_s, 1), "train_s": round(train_s, 1), "checkpoint": out_dir,
           "config": {k: v for k, v in agent.cfg.items() if isinstance(v, (str, int, float, bool)) or v is None},
           "full_run": {"name": name, "minutes": minutes, "data": data, "pool_dir": pool_dir,
                        "calib_eval_pool": H.POOL_DIR, "attempts": len(attempts), "resumed_from_step":
                        resume["step"] if resume else None, "train_loop": {k: v for k, v in record.items() if k != "stats"},
                        "steps": stats.get("steps"), "steps_per_s": stats.get("steps_per_s"),
                        "data_wait_frac": stats.get("data_wait_frac"), "samples": sum(seen.values()) or None,
                        "set_sizes": sizes, "samples_per_dataset": seen,
                        "passes": {n: round(seen[n] / sizes[n], 2) for n in seen if sizes.get(n)}}}
    with open(done, "w") as f:
        json.dump(res, f, indent=2)
    for p in (state_path, os.path.join(run_dir, "crashed")):
        if os.path.exists(p):
            os.remove(p)  # finished: the checkpoint is in best/
    H.ckpt_vol.commit()
    print(json.dumps(summary))
    _log(t_run, "trained, saved and scored")
    return res


@app.function(image=image, cpu=1, memory=4096, timeout=24 * 60 * 60, volumes={"/ckpt": H.ckpt_vol},
              retries=modal.Retries(max_retries=3, initial_delay=10.0))
def pipeline(source: str, name: str, commit: str, token: str, minutes: float, data: str, state_every_min: float,
             crash_at_step: int, deployment: str, meta: Dict) -> Dict:
    """Train (or wait on the training call a previous attempt of this function started), then the harness's
    latency and games jobs on the saved checkpoint, then write ``result.json`` next to it."""
    import games_eval

    t0 = time.time()
    H.ckpt_vol.reload()
    _check_owner(name, token)
    run_dir = _run_dir(name)
    call_path = os.path.join(run_dir, "train_call.txt")
    if os.path.exists(call_path):
        call_id = open(call_path).read().strip()
        call = modal.FunctionCall.from_id(call_id)  # not hydrated: use the id string, not call.object_id
        _log(t0, "waiting on the existing training call %s" % call_id)
    else:
        call = train_eval.spawn(source, name, commit, token, minutes, data, state_every_min, crash_at_step)
        with open(call_path, "w") as f:
            f.write(call.object_id)
        H.ckpt_vol.commit()
        _log(t0, "training call %s" % call.object_id)
    res = call.get()

    latency = modal.Cls.from_name(deployment, "Latency")
    games = modal.Cls.from_name(deployment, "Games")
    tag = "full/" + name  # the harness classes load os.path.join(harness.ROOT, tag, commit)
    lat = latency().run.spawn(tag, "best")
    fams = {f: games().run.spawn(tag, "best", f) for f in games_eval.FAMILIES}
    res["summary"].update({k: v for k, v in lat.get().items() if k.startswith("latency")})
    played = {}
    for f, c in fams.items():
        played.update(c.get())
    g = games_eval.summarize(played)
    if not g["complete"]:
        raise RuntimeError("games benchmark incomplete, missing %s" % g["missing"])
    res["summary"]["games"] = g["games"]
    res["games"] = {"per_game": g["per_game"], "results": played}
    res.update(meta)
    res["total_s"] = round(time.time() - t0, 1)
    # modal_app.bench_latency (full_eval's latency part) reads the eval dataset list from <run>/metrics.json, which a
    # training run from modal_app writes and this layout otherwise lacks
    mpath = os.path.join(run_dir, "metrics.json")
    if not os.path.exists(mpath):
        with open(mpath, "w") as f:
            json.dump({"args": {"val_datasets": "vqa,cauldron,score,eval"},
                       "note": "written by full_run.py for full_eval's latency part"}, f, indent=2)
    out = os.path.join(run_dir, "result.json")
    if not os.path.exists(out):
        with open(out, "w") as f:
            json.dump(res, f, indent=2)
        H.ckpt_vol.commit()
    return res


@app.function(image=image, cpu=1, timeout=5 * 60, volumes={"/ckpt": H.ckpt_vol.read_only()})
def fetch_result(name: str) -> Optional[Dict]:
    path = os.path.join(_run_dir(name), "result.json")
    return json.load(open(path)) if os.path.exists(path) else None


# ------------------------------------------------------------------------------------------------ local side


def _git(*args) -> str:
    return subprocess.run(["git", *args], cwd=H.REPO, capture_output=True, text=True).stdout.strip()


def _git_state() -> Dict:
    return {"commit": _git("rev-parse", "HEAD"), "dirty": bool(_git("status", "--porcelain", "--", "autoresearch", "laya"))}


def _write_local(name: str, res: Dict) -> str:
    out = os.path.join(HERE, "runs", "full", name + ".json")
    if os.path.exists(out):
        out = out[:-5] + ".fetched-%d.json" % int(time.time())  # create-only locally too
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    return out


def _print(res: Dict) -> None:
    s, fr = res["summary"], res.get("full_run", {})
    print("---")
    for k in ("quality", "macro_acc", "ece_hard", "games", "params_m", "latency_x", "latency_ms", "latency_ref_ms"):
        if k in s:
            print("%-17s %.4f" % (k + ":", s[k]))
    print("%-17s %.1f" % ("train_seconds:", res["train_s"]))
    for k in ("steps", "samples", "attempts", "resumed_from_step", "data"):
        print("%-17s %s" % (k + ":", fr.get(k)))
    if fr.get("passes"):
        print("passes:           " + ", ".join("%s %.2f" % (n, p) for n, p in sorted(fr["passes"].items())))
    if res.get("games"):
        print("games per game:   " + ", ".join("%s %.3f" % (g, v) for g, v in res["games"]["per_game"].items()))


def _full_eval_cmd(name: str) -> List[str]:
    return [sys.executable, "-m", "modal", "run", "modal_app.py::full_eval", "--model",
            "autoresearch/full/%s/best" % name, "--out", "eval-results/autoresearch-full-%s.json" % name]


@app.local_entrypoint()
def main(commit: str = "", name: str = "", minutes: float = 120.0, data: str = "full", desc: str = "",
         state_every_min: float = 10.0, crash_at_step: int = 0, full_eval: bool = False, fetch: bool = False,
         prepare_pool: bool = False, attach: bool = False):
    if prepare_pool:
        build_full_pool()
        return
    if not NAME_RE.match(name or ""):
        raise SystemExit("--name is required: letters, digits, '.', '_', '-' (it becomes /ckpt/autoresearch/full/<name>)")
    if fetch:
        res = fetch_result.remote(name)
        if res is None:
            raise SystemExit("no result.json for %s yet (still training, or it failed: modal app logs "
                             "laya-autoresearch-full)" % name)
        print("wrote", _write_local(name, res))
        _print(res)
        return
    if attach:
        # Take over orchestration of a launched run whose pipeline died (e.g. its container was preempted and the
        # retries ran out): wait on its training call, then measure and write result.json, as the pipeline does.
        rec = claim_record.remote(name)
        if rec is None:
            raise SystemExit("%s was never claimed; launch it with --commit" % name)
        token = rec.pop("token")
        sha = rec["recipe_commit"]
        source = subprocess.run(["git", "show", "%s:autoresearch/experiment.py" % sha], cwd=H.REPO,
                                capture_output=True, text=True, check=True).stdout
        meta = {k: rec[k] for k in ("harness", "description", "recipe_commit", "full_run_code",
                                    "harness_changed_since_recipe", "harness_deployment", "full_eval_command")}
        t0 = time.time()
        res = pipeline.remote(source, name, sha[:7], token, rec["minutes"], rec["data"], state_every_min, 0,
                              rec["harness_deployment"], meta)
        res["total_s"] = round(time.time() - t0, 1)
        print("wrote", _write_local(name, res))
        _print(res)
        if full_eval:
            p = subprocess.run(_full_eval_cmd(name), cwd=H.REPO)
            if p.returncode:
                raise SystemExit("full_eval failed (exit %d)" % p.returncode)
        return
    if data not in ("full", "pool"):
        raise SystemExit("--data is full or pool")
    if not 0 < minutes <= MAX_MINUTES:
        raise SystemExit("--minutes must be in (0, %d]" % MAX_MINUTES)
    if not commit:
        raise SystemExit("--commit is required (the commit whose autoresearch/experiment.py to train)")
    sha = _git("rev-parse", "--verify", "--quiet", commit + "^{commit}")
    if not sha:
        raise SystemExit("unknown commit %r" % commit)
    source = subprocess.run(["git", "show", "%s:autoresearch/experiment.py" % sha], cwd=H.REPO, capture_output=True,
                            text=True, check=True).stdout
    local_out = os.path.join(HERE, "runs", "full", name + ".json")
    if os.path.exists(local_out):
        raise SystemExit("%s exists; pick a new --name" % local_out)
    # The harness version that measures this run, and whether the fixed files moved since the recipe was scored.
    moved = _git("diff", "--name-only", sha, "HEAD", "--", "autoresearch/harness.py", "autoresearch/games_eval.py",
                 "autoresearch/game_baselines.json", "autoresearch/toolkit.py", "laya")
    if moved:
        print("WARNING: the harness files changed between %s and HEAD (%s); this run uses HEAD's" % (sha[:7], moved.replace("\n", ", ")))
    version = H._code_hash()
    H.deployed_classes()  # the harness deployment whose Latency / Games classes measure the checkpoint
    deployment = H.deployment_name()
    token = uuid.uuid4().hex
    meta = {"harness": version, "description": desc or "full run of %s: %s" % (sha[:7], _git("log", "-1", "--format=%s", sha)),
            "recipe_commit": sha, "full_run_code": _git_state(), "harness_changed_since_recipe": moved.split("\n") if moved else [],
            "harness_deployment": deployment,
            "full_eval_command": " ".join(["modal"] + _full_eval_cmd(name)[3:])}
    print("claimed", claim.remote(name, token, dict(meta, minutes=minutes, data=data)))
    t0 = time.time()
    res = pipeline.remote(source, name, sha[:7], token, minutes, data, state_every_min, crash_at_step, deployment, meta)
    res["total_s"] = round(time.time() - t0, 1)
    print("wrote", _write_local(name, res))
    _print(res)
    print("full eval suite: %s" % meta["full_eval_command"])
    if full_eval:
        p = subprocess.run(_full_eval_cmd(name), cwd=H.REPO)
        if p.returncode:
            raise SystemExit("full_eval failed (exit %d)" % p.returncode)
