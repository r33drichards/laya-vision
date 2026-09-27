"""BiGym on Modal: a perception probe and zero-shot control for a laya-vision checkpoint (``laya.bigymgames``).

    modal run modal_bigym.py::bigym_eval                                  # all six tasks, both parts
    modal run modal_bigym.py::bigym_eval --tasks ReachTarget,DrawerTopClose --parts control --episodes 3
    modal run modal_bigym.py::bigym_eval --parts control --frames 2,4       # the model sees its last 2 / 4 views
    modal run modal_bigym.py::bigym_eval --model thaitea/laya-vision --revision <sha> --out eval-results/x.json
    modal run modal_bigym.py::bigym_eval --model <run>/best --parts control --frames 4 [--head-max-len 320]
                                                     # a checkpoint on laya-checkpoints (resolved like
                                                     # modal_app._ckpt_path: under /ckpt/smolvlm, then /ckpt)

Models: ``--model`` is a run on the laya-checkpoints volume (``<run>/best`` under /ckpt/smolvlm, or
``<family>/<run>/best`` such as ``autoresearch/full/long-sep24-b64/best``, mounted read-only) when one exists there,
else a Hugging Face id loaded at ``--revision``, which a volume run ignores. ``--head-max-len`` / ``--max-len``
override the checkpoint's token budgets (0: keep them), so a model can be scored with the budgets it was trained
with; ``predict(strict=True)`` raises rather than cut a question.

Parts:
    probe    ``--probe-n`` head-camera frames per task with simulator ground truth, asked ``done`` (noul),
             ``progress`` (score, 4 levels) and on the reach tasks ``side`` (choice): accuracy, ECE, NLL and the
             prior-only baseline of the same rows. The frames are regenerated from their seeds (not stored).
    control  ``--episodes`` seeded episodes per task for the model, random play and, on the reach tasks, the
             privileged oracle, all over the same motion primitives; the cupboard tasks' expert reference is
             BiGym's human demonstrations replayed in the sim (``--demos`` of them). ``--frames 1,2,4`` plays the
             model once per count of head frames it sees (one per decision, 0.1 s apart, oldest first).

The local client writes one JSON (``--out``, default ``eval-results/bigym-<model>-<date>.json``) and prints the
tables. Nothing is written to the volumes except the shared HF cache.
"""
import json
import os
import subprocess
import sys
import time

import modal

app = modal.App("laya-bigym")

hf_vol = modal.Volume.from_name("laya-hf-cache")
ckpt_vol = modal.Volume.from_name("laya-checkpoints")
CKPT_ROOTS = ("/ckpt/smolvlm", "/ckpt")  # modal_app._ckpt_path's search order

MODEL = "thaitea/laya-vision"
REVISION = "f2fe3c12cb6d04c59d8a190250bf3fb40fc828dc"  # thaitea/laya-vision main on 2026-09-25
BIGYM = "git+https://github.com/r33drichards/bigym@14beb30318ad14c5d6723175c2ee2281129792af"
MUJOCO = "3.14.0"  # the version the primitives, oracle and demo replay were checked with
TASKS = ("ReachTarget", "ReachTargetSingle", "DrawerTopOpen", "DrawerTopClose", "WallCupboardOpen",
         "WallCupboardClose")  # laya.bigymgames.TASKS; the local client does not import laya (no torch there)
REACH = ("ReachTarget", "ReachTargetSingle")  # tasks with a primitive oracle; the rest get the demo reference

# the same torch / transformers pins as modal_app.base_image, plus MuJoCo, BiGym and headless GL
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "libegl1", "libgl1", "libosmesa6", "libglib2.0-0")
    .pip_install("torch==2.14.0", "torchvision==0.29.0", "transformers==5.17.0", "safetensors", "huggingface_hub",
                 "numpy", "pillow", "num2words", "tqdm", "mujoco==" + MUJOCO)
    # BiGym pins safetensors==0.6.2, which transformers 5 rejects; its demos load fine with 0.8, so install it
    # without its pins and add the runtime deps it imports (at the versions checked locally; no GUI / VR extras)
    .pip_install("dm_control==1.0.47", "gymnasium==1.3.0", "mojo-mujoco-wrapper==0.1.1", "mujoco-utils==0.0.6",
                 "pyquaternion==0.9.9", "numpy-quaternion==2024.0.13", "imageio", "pyyaml", "wget==3.2")
    .pip_install(BIGYM, extra_options="--no-deps")
    .env({"HF_HOME": "/cache/hf", "TOKENIZERS_PARALLELISM": "false", "MUJOCO_GL": "egl",
          "NVIDIA_DRIVER_CAPABILITIES": "all"})
    .add_local_python_source("laya")
)


def _pick_gl() -> str:
    """EGL when a context can be made (GPU containers), else OSMesa; must run before MuJoCo renders."""
    probe = "import mujoco; c = mujoco.GLContext(64, 64); c.make_current(); c.free()"
    for gl in ("egl", "osmesa"):
        env = dict(os.environ, MUJOCO_GL=gl, PYOPENGL_PLATFORM=gl)
        if subprocess.run([sys.executable, "-c", probe], env=env, capture_output=True, timeout=120).returncode == 0:
            os.environ["MUJOCO_GL"] = os.environ["PYOPENGL_PLATFORM"] = gl
            return gl
    raise RuntimeError("no headless GL backend for MuJoCo")


def _versions() -> dict:
    import mujoco

    return {"mujoco": mujoco.__version__, "bigym": BIGYM, "gl": os.environ.get("MUJOCO_GL")}


def _ckpt_run(model: str) -> str:
    """The checkpoint directory of a laya-checkpoints run name (``<run>/best``, ``<family>/<run>/best``), resolved
    like ``modal_app._ckpt_path``; ``""`` when there is none, i.e. ``model`` is a Hugging Face id."""
    if model.startswith("/") or ".." in model.split("/"):
        return ""
    for root in CKPT_ROOTS:
        path = os.path.join(root, model)
        if os.path.exists(os.path.join(path, "vlm_agent_config.json")):
            return path
    return ""


def _agent(model: str, revision: str, head_max_len: int = 0, max_len: int = 0):
    from laya.vlm import VLMAgent

    budgets = dict(({"head_max_len": head_max_len} if head_max_len else {}), **({"max_len": max_len} if max_len else {}))
    path = _ckpt_run(model)
    if path:
        agent = VLMAgent(path, device="cuda", dtype="bf16", **budgets)
    else:
        agent = VLMAgent(model, revision=revision or None, device="cuda", dtype="bf16", **budgets)
    print("loaded %s from %s (head_max_len %d, max_len %d)" % (model, path or "the Hub@%s" % revision,
                                                                agent.cfg.get("head_max_len", 256),
                                                                agent.cfg.get("max_len", 1024)), flush=True)
    return agent


def _source(model: str, revision: str) -> dict:
    """What was loaded: the volume path, or the Hub id and revision."""
    path = _ckpt_run(model)
    return {"checkpoint": path, "revision": None} if path else {"checkpoint": None, "revision": revision}


@app.function(image=image, gpu="L4", cpu=4, timeout=120 * 60, volumes={"/cache/hf": hf_vol, "/ckpt": ckpt_vol.read_only()})
def bigym_probe(task: str, model: str = MODEL, revision: str = REVISION, n: int = 200, head_max_len: int = 0,
                max_len: int = 0) -> dict:
    """The perception probe on one task: ``n`` frames from ``laya.bigymgames.probe_frames``, all of
    ``probe_questions(task)`` in one ``predict`` per frame."""
    gl = _pick_gl()
    from laya import bigymgames as bg

    t0 = time.time()
    frames = bg.probe_frames(task, n)
    qs = bg.probe_questions(task)
    agent = _agent(model, revision, head_max_len, max_len)
    rows = []
    for i, fr in enumerate(frames):
        ans = agent.predict({"image": fr["image"]}, qs, strict=True)["answers"]
        truth = {k: v for k, v in fr["truth"].items()}
        for qid, q in qs.items():
            rows.append({"task": task, "frame": i, "seed": fr["seed"], "qid": qid, "label": fr["labels"][qid],
                         "probs": [round(float(p), 5) for p in bg.answer_probs(ans[qid], q)], "truth": truth})
    out = {"task": task, "model": model, "revision": revision, "source": _source(model, revision),
           "budgets": {k: agent.cfg.get(k) for k in ("head_max_len", "max_len")}, "n": len(frames), "questions": qs,
           "metrics": bg.probe_metrics(rows), "rows": rows, "versions": _versions(), "gl": gl,
           "seconds": round(time.time() - t0, 1)}
    print(json.dumps({"task": task, "metrics": out["metrics"]}))
    return out


def _play(task: str, policy: str, episodes: int, seed: int, model: str = "", revision: str = "",
          frames: int = 1, head_max_len: int = 0, max_len: int = 0) -> dict:
    gl = _pick_gl()
    from laya import bigymgames as bg

    t0 = time.time()
    extra = {}
    if policy == "model":
        agent = _agent(model, revision, head_max_len, max_len)
        fn = bg.model_policy(agent, task, frames)
        extra = {"source": _source(model, revision),
                 "budgets": {k: agent.cfg.get(k) for k in ("head_max_len", "max_len")}}
    elif policy == "oracle":
        fn = bg.oracle_policy
    else:
        fn = bg.random_policy(seed)
    out = bg.play_episodes(task, fn, episodes, seed)
    label = "model:%s" % model + ("" if extra.get("source", {}).get("checkpoint") else "@%s" % revision[:7])
    out.update(extra, policy=label if policy == "model" else policy, frames=frames,
               versions=_versions(), gl=gl, seconds=round(time.time() - t0, 1))
    print(json.dumps({k: v for k, v in out.items() if k != "results"}))
    return out


@app.function(image=image, gpu="L4", cpu=4, timeout=180 * 60, volumes={"/cache/hf": hf_vol, "/ckpt": ckpt_vol.read_only()})
def bigym_play(task: str, model: str = MODEL, revision: str = REVISION, episodes: int = 20,
               seed: int = 300_000, frames: int = 1, head_max_len: int = 0, max_len: int = 0) -> dict:
    """The model plays ``episodes`` seeded episodes of ``task``, one ``predict`` per primitive, seeing the last
    ``frames`` decision frames."""
    return _play(task, "model", episodes, seed, model, revision, frames, head_max_len, max_len)


@app.function(image=image, cpu=4, timeout=180 * 60)
def bigym_baseline(task: str, policy: str = "random", episodes: int = 20, seed: int = 300_000) -> dict:
    """``random`` or ``oracle`` (reach tasks) on the same seeded episodes as ``bigym_play``."""
    return _play(task, policy, episodes, seed)


@app.function(image=image, cpu=4, timeout=60 * 60)
def bigym_demos(task: str, amount: int = 20) -> dict:
    """Success rate of ``amount`` of BiGym's human demonstrations of ``task`` replayed in the sim."""
    gl = _pick_gl()
    from laya import bigymgames as bg

    out = bg.demo_reference(task, amount)
    out.update(versions=_versions(), gl=gl)
    print(json.dumps(out))
    return out


_ERRORS = []


def _get(call, what: str):
    """A call's result, or ``None`` with the error printed and recorded: one broken task should not lose the rest."""
    try:
        return call.get()
    except Exception as e:
        print("%s failed: %s" % (what, repr(e)[:300]))
        _ERRORS.append({"what": what, "error": repr(e)[:1000]})
        return None


def _fmt(v, pct=False):
    if v is None:
        return "-"
    return "%.0f%%" % (100 * v) if pct else "%.3f" % v


def _print_probe(probes: list) -> None:
    print("\n== probe (accuracy / prior-only accuracy, ECE, NLL / prior NLL, ranking: AUROC or Spearman)")
    print("%-18s %-9s %5s %7s %7s %7s %7s %7s %7s" % ("task", "question", "n", "acc", "prior", "ece", "nll", "p_nll",
                                                     "rank"))
    for p in probes:
        for qid, m in p["metrics"].items():
            print("%-18s %-9s %5d %7s %7s %7s %7s %7s %7s" % (
                p["task"], qid, m["n"], _fmt(m["acc"], True), _fmt(m["prior_acc"], True), _fmt(m["ece"]),
                _fmt(m["nll"]), _fmt(m["prior_nll"]), _fmt(m.get("auroc", m.get("spearman")))))


def _model_key(frames: int) -> str:
    return "model" if frames == 1 else "model_%df" % frames


def _print_control(control: dict, frame_counts: list) -> None:
    print("\n== control (success rate over the same seeded episodes)")
    print("%-18s %6s %7s %7s %7s %7s %6s  %s" % ("task", "frames", "model", "random", "oracle", "demos", "norm",
                                                 "top model actions"))
    for task, c in control.items():
        r, o, d = (c.get(k) for k in ("random", "oracle", "demos"))
        for n in frame_counts:
            m = c.get(_model_key(n))
            top = ""
            if m:
                total = max(1, sum(m["actions"].values()))
                top = ", ".join("%s %d%%" % (a, 100 * k / total)
                                for a, k in sorted(m["actions"].items(), key=lambda kv: -kv[1])[:3])
            print("%-18s %6d %7s %7s %7s %7s %6s  %s" % (task, n, _fmt(m and m["success_rate"], True),
                                                         _fmt(r and r["success_rate"], True),
                                                         _fmt(o and o["success_rate"], True),
                                                         _fmt(d and d["success_rate"], True),
                                                         _fmt(c["normalized"].get(_model_key(n))), top))


def _git_state() -> dict:
    def git(*args):
        try:
            return subprocess.run(["git", *args], capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception:
            return ""
    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain", "--", "laya",
                                                                  "modal_bigym.py"))}


@app.local_entrypoint()
def bigym_eval(tasks: str = ",".join(TASKS), parts: str = "probe,control", model: str = MODEL,
               revision: str = REVISION, episodes: int = 20, probe_n: int = 200, demos: int = 20,
               seed: int = 300_000, frames: str = "1", out: str = "", head_max_len: int = 0, max_len: int = 0):
    """Everything in parallel on one checkpoint; see the module docstring. ``--frames 1,2,4`` plays the model once
    per frame count (results under ``model``, ``model_2f``, ``model_4f``). ``--model`` is a laya-checkpoints run
    (``<run>/best``) or a Hub id at ``--revision``; ``--head-max-len`` / ``--max-len`` override its budgets."""
    task_list = [t for t in tasks.split(",") if t]
    frame_counts = [int(n) for n in frames.split(",") if n]
    if not frame_counts or any(not 1 <= n <= 4 for n in frame_counts):
        raise SystemExit("--frames takes counts from 1 to 4, e.g. 1,2,4")
    unknown = [t for t in task_list if t not in TASKS]
    if unknown:
        raise SystemExit("unknown tasks %s (known: %s)" % (unknown, ", ".join(TASKS)))
    path = out or "eval-results/bigym-%s-%s.json" % (model.replace("/", "_"), time.strftime("%Y%m%d-%H%M%S"))
    if os.path.exists(path):
        raise SystemExit("%s exists; results are create-only, pass a new --out" % path)
    want = set(parts.split(","))
    calls = {"probe": {}, "control": {}}
    for t in task_list:
        if "probe" in want:
            calls["probe"][t] = bigym_probe.spawn(t, model, revision, probe_n, head_max_len, max_len)
        if "control" in want:
            c = {_model_key(n): bigym_play.spawn(t, model, revision, episodes, seed, n, head_max_len, max_len)
                 for n in frame_counts}
            c["random"] = bigym_baseline.spawn(t, "random", episodes, seed)
            if t in REACH:
                c["oracle"] = bigym_baseline.spawn(t, "oracle", episodes, seed)
            else:
                c["demos"] = bigym_demos.spawn(t, demos)
            calls["control"][t] = c
    probes = [r for t, c in calls["probe"].items() for r in [_get(c, "probe " + t)] if r]
    control = {}
    for t, cs in calls["control"].items():
        control[t] = {k: _get(c, "%s %s" % (k, t)) for k, c in cs.items()}
        r, o = control[t].get("random"), control[t].get("oracle")
        control[t]["normalized"] = {}
        for n in frame_counts:
            m = control[t].get(_model_key(n))
            tie = not (m and r and o) or o["success_rate"] == r["success_rate"]
            control[t]["normalized"][_model_key(n)] = None if tie else (
                (m["success_rate"] - r["success_rate"]) / (o["success_rate"] - r["success_rate"]))
    if probes:
        _print_probe(probes)
    if control:
        _print_control(control, frame_counts)
    loaded = next((r.get("source") for r in probes + [m for c in control.values() for m in c.values()
                                                      if isinstance(m, dict) and m.get("source")]), None)
    result = {"model": model, "revision": None if loaded and loaded.get("checkpoint") else revision,
              "source": loaded, "head_max_len": head_max_len or None, "max_len": max_len or None, "tasks": task_list, "parts": sorted(want), "episodes": episodes,
              "probe_n": probe_n, "demos": demos, "seed": seed, "frames": frame_counts, "code": _git_state(),
              "date": time.strftime("%Y-%m-%d"), "errors": _ERRORS, "probe": probes, "control": control}
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(result, f)
    print("\nwrote", path)


# ---------------------------------------------------------------------------------------------------------
# Behaviour-cloning data: BiGym's human demos, followed with primitives, as laya-vision training sets
# ---------------------------------------------------------------------------------------------------------
#
#     modal run modal_bigym.py::prepare_bigym_bc --prefix bigym_smoke_20260927 --tasks DrawerTopClose --max-demos 3
#     modal run --detach modal_bigym.py::prepare_bigym_bc          # all demos of the four cupboard tasks
#
# The client spawns ``bigym_bc_build`` and returns (``--wait`` blocks for the summary); run it with --detach and
# follow ``modal app logs``. A restarted build (preemption, or the same command again) reuses the finished
# per-demo runs of the same laya commit from ``/data/vqa/<prefix>_bc.tmp/runs``; ``--fresh`` starts over.
#
# Every demo of each task is followed (``laya.bigymdemos.follow``, one container per demo); each successful run is
# replayed with the head camera and written, one frame per decision, as three datasets on laya-datasets:
# ``<prefix>_bc_f1`` / ``<prefix>_bc_f4`` (the control question on 1 / 4 frames, labelled with the follower's
# primitive) and ``<prefix>_probe`` (the probe questions on every third frame, simulator labels); see
# ``laya.bigymdata``. Create-only: the job refuses if any of the three exists.

data_vol = modal.Volume.from_name("laya-datasets")
BC_TASKS = ("DrawerTopClose", "WallCupboardClose", "DrawerTopOpen", "WallCupboardOpen")
JPEG_QUALITY = 90


def _bc_names(prefix: str) -> dict:
    return {"f1": prefix + "_bc_f1", "f4": prefix + "_bc_f4", "probe": prefix + "_probe"}


@app.function(image=image, cpu=2, memory=8192, timeout=90 * 60)
def bigym_bc_waypoints(task: str, amount: int = -1) -> list:
    """``laya.bigymdemos.demo_waypoints``: ``amount`` (-1: all) of ``task``'s demos replayed to waypoints."""
    _pick_gl()
    from laya import bigymdemos

    t0 = time.time()
    demos = bigymdemos.demo_waypoints(task, amount=amount, seed=0)
    print("%s: %d demos, %d succeed in BiGym's replay, %.0f s" % (
        task, len(demos), sum(d["success_step"] is not None for d in demos), time.time() - t0))
    return demos


@app.function(image=image, cpu=2, memory=4096, timeout=3 * 60 * 60, volumes={"/data": data_vol},
              retries=modal.Retries(max_retries=2, initial_delay=10.0))
def bigym_bc_episode(task: str, demo: dict, tmp_root: str, names: dict, probe_every: int = 3,
                     code: str = "") -> dict:
    """Follow one demo in a head-camera env, recording the frame and ground truth of every decision as the
    follower sees it; if the follower succeeds, write the frames (JPEG) under ``<tmp_root>/<dataset>/images/``.
    The result (summary, primitives, probe labels) is also saved as ``<tmp_root>/runs/<task>-<seed>.json`` so a
    restarted ``bigym_bc_build`` does not redo it.

    The frames are taken during ``follow`` itself rather than by replaying its primitives afterwards: a replay
    without the lookahead's try-and-undo steps drifts from the follower's episode (seen on most WallCupboardClose
    runs), so the replayed frames would not be the states the labels were chosen in. The env renders nothing while
    the follower plays; each decision's frame is rendered once, from its real state (the first step tried there),
    by the same ``get_observation`` that ``BiGymGame.frame()`` reads."""
    import io

    import numpy as np
    from PIL import Image

    gl = _pick_gl()
    from laya import bigymdata as bd
    from laya import bigymdemos
    from laya import bigymgames as bg

    t0 = time.time()
    out = {"task": task, "seed": int(demo["seed"]), "uuid": demo["uuid"],
           "demo_success": demo["success_step"] is not None, "kept": False, "gl": gl, "code": code}
    seen, games, problems = {}, [], []
    env = bg.make_env(task, cameras=True)
    render = env.get_observation

    def no_pixels():  # the follower's observations without the head camera: its 37 lookahead tries need none
        return env._get_proprioception_obs() | env._get_task_privileged_obs()

    def capture(game):
        d = game.decisions
        if d in seen:
            return
        # rendered from the decision's real state with the env's own camera settings (== BiGymGame.frame())
        fr = np.ascontiguousarray(np.asarray(render()["rgb_head"]).transpose(1, 2, 0))
        buf = io.BytesIO()
        Image.fromarray(fr).save(buf, "JPEG", quality=JPEG_QUALITY)
        seen[d] = {"jpg": buf.getvalue(), "truth": game.ground_truth()}

    orig_step = bg.BiGymGame.step

    def step(self, name):  # every step, lookahead tries included, starts from the decision's real state
        if not games:
            games.append(self)
        capture(self)
        return orig_step(self, name)

    try:
        bg.BiGymGame.step = step
        env.get_observation = no_pixels
        try:
            run = bigymdemos.follow(task, demo, env=env)
            n = len(run["labels"])
            if games:
                capture(games[0])  # the final state (task done when the run succeeded): probe only
        finally:
            bg.BiGymGame.step = orig_step
            env.close()
        out["follow"] = {k: v for k, v in run.items() if k != "labels"}
        prims = [lab["primitive"] for lab in run["labels"]]
        if [lab["decision"] for lab in run["labels"]] != list(range(n)) or sorted(seen) != list(range(n + 1)):
            problems.append("decisions %s labelled, %d frames recorded" % (n, len(seen)))
        if run["success"] and not seen[n]["truth"]["success"]:
            problems.append("final frame is not a success")
        if problems:
            out["problem"] = problems[0]
            print("%s-%d not kept: %s" % (task, out["seed"], problems[0]))
        elif run["success"]:
            keep_probe = set(bd.probe_decisions(n, probe_every))
            for key, name in names.items():
                os.makedirs(os.path.join(tmp_root, name, "images"), exist_ok=True)
                for d in sorted(seen):
                    if key == "probe" and d not in keep_probe or key != "probe" and d == n:
                        continue
                    with open(os.path.join(tmp_root, name, bd.image_path(task, out["seed"], d)), "wb") as f:
                        f.write(seen[d]["jpg"])
            out.update(kept=True, primitives=prims, frames=n, jpeg_bytes=sum(len(v["jpg"]) for v in seen.values()),
                       probes=[{"decision": d, "truth": seen[d]["truth"],
                                "labels": bg.labels(task, seen[d]["truth"])} for d in sorted(keep_probe)])
    except Exception as e:  # one broken demo should not lose the rest
        out["error"] = repr(e)[:1000]
        print("%s-%d failed: %s" % (task, out["seed"], out["error"]))
    out["seconds"] = round(time.time() - t0, 1)
    os.makedirs(os.path.join(tmp_root, "runs"), exist_ok=True)
    with open(os.path.join(tmp_root, "runs", "%s-%d.json" % (task, out["seed"])), "w") as f:
        json.dump(out, f)
    data_vol.commit()
    print(json.dumps({k: out.get(k) for k in ("task", "seed", "kept", "frames", "seconds")}
                     | {"follow": {k: v for k, v in (out.get("follow") or {}).items()
                                   if k in ("success", "decisions", "demo_decisions", "skipped")}}))
    return out


def _file_sha256(path: str) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@app.function(image=image, cpu=2, memory=16384, timeout=24 * 60 * 60, volumes={"/data": data_vol})
def bigym_bc_build(tasks: list, prefix: str = "bigym", max_demos: int = -1, probe_every: int = 3,
                   code: dict = None, fresh: bool = False) -> dict:
    """Fan the demos out to ``bigym_bc_episode``, then write the three datasets' jsonl, meta.json and
    manifest.json in a tmp dir, check they load (``laya.vlm_train.load_jsonl_examples``), rename them into
    /data/vqa and write ``_READY`` last."""
    import datetime
    import shutil
    from collections import Counter

    from laya import bigymdata as bd
    from laya.bigymgames import FROM_WORDS, PRIMITIVES
    from laya.vlm_train import load_jsonl_examples

    t0 = time.time()
    names = _bc_names(prefix)
    finals = {k: "/data/vqa/" + n for k, n in names.items()}
    data_vol.reload()
    taken = [p for p in finals.values() if os.path.exists(p)]
    if taken:
        raise RuntimeError("refusing: %s already exist (datasets are create-only; pass a new --prefix)" % taken)
    # the tmp dir survives a restart (preemption) of this function: finished runs of the same laya commit are
    # reused from <tmp_root>/runs/, everything else is redone; fresh=True starts over
    tmp_root = "/data/vqa/%s_bc.tmp" % prefix
    commit = (code or {}).get("commit", "")
    if fresh:
        shutil.rmtree(tmp_root, ignore_errors=True)
    for n in list(names.values()) + ["runs"]:
        os.makedirs(os.path.join(tmp_root, n, "images" if n != "runs" else ""), exist_ok=True)
    done = {}
    for fn in os.listdir(os.path.join(tmp_root, "runs")):
        with open(os.path.join(tmp_root, "runs", fn)) as f:
            r = json.load(f)
        if r.get("code") == commit and "error" not in r:
            done[(r["task"], r["seed"])] = r
    data_vol.commit()

    demos = dict(zip(tasks, list(bigym_bc_waypoints.map(tasks, kwargs={"amount": max_demos}))))
    runs = [done[(t, int(d["seed"]))] for t in tasks for d in demos[t] if (t, int(d["seed"])) in done]
    args = [(task, d, tmp_root, names, probe_every, commit) for task in tasks for d in demos[task]
            if (task, int(d["seed"])) not in done]
    print("following %d demos (%d reused from an earlier attempt): %s" % (
        len(args), len(runs), {t: len(d) for t, d in demos.items()}))
    crashed = []
    for i, r in enumerate(bigym_bc_episode.starmap(args, order_outputs=False, return_exceptions=True)):
        if isinstance(r, Exception):
            crashed.append(repr(r)[:500])
            print("episode call crashed: %s" % crashed[-1])
            continue
        runs.append(r)
        if (i + 1) % 20 == 0:
            print("%d / %d episodes back, %d kept, %.0f min" % (i + 1, len(args), sum(x["kept"] for x in runs),
                                                                 (time.time() - t0) / 60))
    if crashed:  # e.g. the app was stopped: publish nothing; the same command resumes from the saved runs
        raise RuntimeError("%d episode calls crashed; rerun to resume (finished runs are kept in %s/runs)" % (
            len(crashed), tmp_root))
    if not any(r["kept"] for r in runs):
        raise RuntimeError("no follower run was kept; nothing to publish")
    data_vol.reload()

    import mujoco

    base_meta = {"source": "BiGym human demonstrations (DemoStore), followed closed-loop with laya.bigymgames "
                           "primitives by laya.bigymdemos.follow; successful follower runs replayed with the head "
                           "camera (256x256, JPEG q%d), one frame per 0.1 s decision" % JPEG_QUALITY,
                 "bigym": BIGYM, "bigym_sha": BIGYM.rsplit("@", 1)[-1], "mujoco": mujoco.__version__,
                 "laya_commit": (code or {}).get("commit"), "laya_dirty": (code or {}).get("dirty"),
                 "primitives": list(PRIMITIVES), "tasks": list(tasks), "max_demos": max_demos,
                 "probe_every": probe_every, "val_frac": bd.VAL_FRAC,
                 "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                 "episode_calls_crashed": crashed}
    per_task, recs = {}, {k: {"train": [], "val": []} for k in names}
    for task in tasks:
        tr = sorted([r for r in runs if r["task"] == task], key=lambda r: r["seed"])
        kept = [r for r in tr if r["kept"]]
        val = set(bd.val_seeds(task, [r["seed"] for r in kept]))
        follows = [r for r in tr if "follow" in r]
        per_task[task] = {
            "demos": len(demos[task]), "demo_replay_success": sum(d["success_step"] is not None for d in demos[task]),
            "followed": len(follows), "follower_successes": sum(r["follow"]["success"] for r in follows),
            "follower_success_rate": (sum(r["follow"]["success"] for r in follows) / len(follows)) if follows else None,
            "kept": len(kept), "record_problems": [r["seed"] for r in tr if r.get("problem")],
            "errors": [{"seed": r["seed"], "error": r["error"]} for r in tr if r.get("error")],
            "val_seeds": sorted(val), "train_seeds": sorted(r["seed"] for r in kept if r["seed"] not in val),
            "mean_follower_decisions": (sum(r["follow"]["decisions"] for r in follows) / len(follows)
                                        if follows else None),
            "runs": [{"seed": r["seed"], "uuid": r["uuid"], "demo_success": r["demo_success"], "kept": r["kept"],
                      **{k: v for k, v in (r.get("follow") or {}).items()
                         if k in ("success", "decisions", "demo_decisions", "reached", "waypoints", "skipped")},
                      "seconds": r.get("seconds")} for r in tr],
        }
        for r in kept:
            split = "val" if r["seed"] in val else "train"
            recs["f1"][split] += bd.control_records(task, r["seed"], r["primitives"], 1)
            recs["f4"][split] += bd.control_records(task, r["seed"], r["primitives"], 4)
            recs["probe"][split] += bd.probe_records(task, r["seed"], r["probes"])

    metas = {}
    for key, name in names.items():
        d = os.path.join(tmp_root, name)
        stats = {}
        for task in tasks:
            s = {"demos": per_task[task]["demos"], "follower_successes": per_task[task]["follower_successes"],
                 "follower_success_rate": per_task[task]["follower_success_rate"]}
            for split in ("train", "val"):
                rows = [r for r in recs[key][split] if r["id"].startswith(task + "-")]
                s[split] = len(rows)
                if key == "probe":
                    c = Counter("%s=%d" % (r["id"].split("-")[-2], r["label"]) for r in rows)
                else:
                    c = Counter(r["primitive"] for r in rows)
                s[split + "_labels"] = dict(c.most_common())
            stats[task] = s
        for split in ("train", "val"):
            with open(os.path.join(d, split + ".jsonl"), "w") as f:
                for r in recs[key][split]:
                    f.write(json.dumps(r) + "\n")
        meta = dict(base_meta, dataset=name, kind=key,
                    question=("bigym_question(task, %d)" % (1 if key == "f1" else 4)) if key != "probe"
                    else "probe_questions(task)", records={s: len(recs[key][s]) for s in ("train", "val")},
                    per_task=stats, follower=per_task)
        with open(os.path.join(d, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        files = {s + ".jsonl": {"sha256": _file_sha256(os.path.join(d, s + ".jsonl")), "records": len(recs[key][s])}
                 for s in ("train", "val")}
        with open(os.path.join(d, "manifest.json"), "w") as f:
            json.dump({"sources": {BIGYM: base_meta["bigym_sha"], "mujoco": base_meta["mujoco"]},
                       "params": {"tasks": list(tasks), "max_demos": max_demos, "probe_every": probe_every},
                       "laya_commit": base_meta["laya_commit"], "files": files,
                       "images": len(os.listdir(os.path.join(d, "images"))), "created_utc": base_meta["created_utc"]},
                      f, indent=2)
        metas[key] = meta
    data_vol.commit()

    # check before publishing: every record loads, its images exist, and control labels decode to the primitive
    for key, name in names.items():
        imgs = set(os.listdir(os.path.join(tmp_root, name, "images")))
        for split in ("train", "val"):
            exs = load_jsonl_examples(tmp_root, name, split)
            if len(exs) != len(recs[key][split]):
                raise RuntimeError("%s/%s: %d of %d records load" % (name, split, len(exs), len(recs[key][split])))
            for r in recs[key][split]:
                for p in r.get("images") or [r["image"]]:
                    if os.path.basename(p) not in imgs:
                        raise RuntimeError("%s/%s: %s is missing %s" % (name, split, r["id"], p))
                if key != "probe" and FROM_WORDS[list(r["question"]["criteria"])[r["label"]]] != r["primitive"]:
                    raise RuntimeError("%s: label %d does not decode to %s" % (r["id"], r["label"], r["primitive"]))
        print("%s: %s records load, images present" % (name, metas[key]["records"]))

    data_vol.reload()
    taken = [p for p in finals.values() if os.path.exists(p)]
    if taken:
        raise RuntimeError("refusing: %s appeared while building; data left in %s" % (taken, tmp_root))
    for key, name in names.items():
        os.rename(os.path.join(tmp_root, name), finals[key])
    shutil.rmtree(tmp_root)  # only runs/ is left: the per-demo results, summarised in meta.json
    for p in finals.values():
        open(os.path.join(p, "_READY"), "w").close()
    data_vol.commit()
    summary = {"datasets": finals, "records": {k: m["records"] for k, m in metas.items()},
               "per_task": {t: {k: v for k, v in s.items() if k != "runs"} for t, s in per_task.items()},
               "stats": {k: m["per_task"] for k, m in metas.items()}, "minutes": round((time.time() - t0) / 60, 1)}
    print(json.dumps(summary, indent=1))
    return summary


@app.local_entrypoint()
def prepare_bigym_bc(tasks: str = ",".join(BC_TASKS), prefix: str = "bigym", max_demos: int = -1,
                     probe_every: int = 3, fresh: bool = False, wait: bool = False):
    """Build ``<prefix>_bc_f1``, ``<prefix>_bc_f4`` and ``<prefix>_probe`` on laya-datasets from all (or
    ``--max-demos``) demos of ``--tasks``; see the section comment above ``bigym_bc_build``."""
    task_list = [t for t in tasks.split(",") if t]
    unknown = [t for t in task_list if t not in TASKS]
    if unknown:
        raise SystemExit("unknown tasks %s" % unknown)
    code = _git_state()
    print("datasets %s from laya %s%s" % (list(_bc_names(prefix).values()), code["commit"][:10],
                                          " (dirty)" if code["dirty"] else ""))
    # spawned, not .remote(): with --detach the build keeps running after this client exits (a client killed
    # while blocked in .remote() cancels the build and its episode calls)
    call = bigym_bc_build.spawn(task_list, prefix, max_demos, probe_every, code, fresh)
    print("spawned bigym_bc_build: call %s; follow with `modal app logs <app-id>`" % call.object_id)
    if wait:
        print(json.dumps(call.get(), indent=1))


# ---------------------------------------------------------------------------------------------------------
# Cleaned behaviour-cloning data: undo pairs and repeated STAYs dropped, probe done=1 oversampled
# ---------------------------------------------------------------------------------------------------------
#
#     modal run modal_bigym.py::clean_bigym_bc --src bigym_v2 --dst bigym_v2c
#
# Reads ``<src>_bc_f1``, ``<src>_bc_f4`` and ``<src>_probe`` and writes ``<dst>_bc_f1``, ``<dst>_bc_f4`` and
# ``<dst>_probe`` (create-only: refuses if any exists). The control sets, train and val alike, get
# ``laya.bigymclean.clean_control`` (undo chains and repeated STAYs dropped per episode, frames and windows as
# recorded); the probe train split gets ``laya.bigymclean.oversample`` (done=1 duplicated to ``--done-target`` of
# each task's done records); probe val is copied unchanged. The images the kept records use are hard-linked from
# the source dataset where the volume allows it, else copied. Data only: no simulation.


@app.function(image=image, cpu=8, memory=8192, timeout=2 * 60 * 60, volumes={"/data": data_vol})
def bigym_clean_build(src: str, dst: str, code: dict = None, done_target: float = 0.15,
                      max_drop_frac: float = 0.4) -> dict:
    """Write the cleaned datasets in ``<name>.tmp`` dirs, check they load, rename them into /data/vqa, _READY last."""
    import datetime
    import shutil
    from collections import Counter
    from concurrent.futures import ThreadPoolExecutor

    from laya import bigymclean as bc
    from laya.bigymgames import FROM_WORDS, PRIMITIVES
    from laya.vlm_train import load_jsonl_examples

    t0 = time.time()
    srcs, dsts = _bc_names(src), _bc_names(dst)
    root = "/data/vqa"
    data_vol.reload()
    missing = [n for n in srcs.values() if not os.path.exists(os.path.join(root, n, "_READY"))]
    if missing:
        raise RuntimeError("source datasets not ready: %s" % missing)
    taken = [n for n in dsts.values() if os.path.exists(os.path.join(root, n))]
    if taken:
        raise RuntimeError("refusing: %s already exist (datasets are create-only; pass a new --dst)" % taken)
    created = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    prims = list(PRIMITIVES)
    summary = {}
    for key in ("f1", "f4", "probe"):
        s_dir, name = os.path.join(root, srcs[key]), dsts[key]
        tmp = os.path.join(root, name + ".tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        os.makedirs(os.path.join(tmp, "images"))
        with open(os.path.join(s_dir, "meta.json")) as f:
            src_meta = json.load(f)
        recs, reports = {}, {}
        for split in ("train", "val"):
            with open(os.path.join(s_dir, split + ".jsonl")) as f:
                rows = [json.loads(line) for line in f if line.strip()]
            if key != "probe":
                for r in rows:
                    if prims[r["label"]] != r["primitive"]:
                        raise RuntimeError("%s: label %d is not %s in the source" % (r["id"], r["label"],
                                                                                       r["primitive"]))
                recs[split], reports[split] = bc.clean_control(rows, prims, max_drop_frac)
            elif split == "train":
                recs[split], reports[split] = bc.oversample(rows, "done", 1, done_target)
                reports[split] = {"before": len(rows), "after": len(recs[split]), "done": reports[split]}
            else:
                labels = Counter("%s %s=%d" % (*bc.probe_question(r["id"]), r["label"]) for r in rows)
                recs[split], reports[split] = rows, {"before": len(rows), "after": len(rows), "unchanged": True,
                                                     "labels": dict(sorted(labels.items()))}
            with open(os.path.join(tmp, split + ".jsonl"), "w") as f:
                for r in recs[split]:
                    f.write(json.dumps(r) + "\n")
        paths = sorted({p for split in recs for r in recs[split] for p in (r.get("images") or [r["image"]])})
        linked = Counter()

        def place(p):
            a, b = os.path.join(s_dir, p), os.path.join(tmp, p)
            try:
                os.link(a, b)
                return "hardlink"
            except OSError:
                shutil.copyfile(a, b)
                return "copy"

        with ThreadPoolExecutor(32) as ex:
            linked.update(ex.map(place, paths))
        rules = (dict(bc.CONTROL_RULES, max_drop_frac_flag=max_drop_frac)
                 if key != "probe" else
                 {"oversample": "train only: per task, done=1 records duplicated (ids <id>-dup<N>) until they are "
                                "%.2f of that task's done records (round(t*n0/(1-t)) positives, spread evenly over "
                                "the originals); progress records and val unchanged" % done_target,
                  "done_target": done_target})
        meta = {"dataset": name, "kind": key, "source_dataset": srcs[key], "source_meta": src_meta,
                "question": src_meta.get("question"), "primitives": prims, "rules": rules,
                "records": {s: len(recs[s]) for s in recs}, "report": reports, "images": len(paths),
                "image_placement": dict(linked), "laya_commit": (code or {}).get("commit"),
                "laya_dirty": (code or {}).get("dirty"), "code": "laya/bigymclean.py, modal_bigym.py::"
                                                                 "bigym_clean_build", "created_utc": created}
        with open(os.path.join(tmp, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        with open(os.path.join(tmp, "manifest.json"), "w") as f:
            json.dump({"sources": {srcs[key]: src_meta.get("laya_commit")}, "laya_commit": meta["laya_commit"],
                       "params": {"done_target": done_target, "max_drop_frac": max_drop_frac},
                       "files": {s + ".jsonl": {"sha256": _file_sha256(os.path.join(tmp, s + ".jsonl")),
                                                "records": len(recs[s])} for s in recs},
                       "images": len(paths), "created_utc": created}, f, indent=2)
        data_vol.commit()

        # check before publishing: all records load (with next targets: no duplicate game frames), images exist,
        # ids are unique and control labels still decode to the recorded primitive
        for split in ("train", "val"):
            exs = load_jsonl_examples(root, name + ".tmp", split, next_targets=True)
            if len(exs) != len(recs[split]):
                raise RuntimeError("%s/%s: %d of %d records load" % (name, split, len(exs), len(recs[split])))
            ids = [r["id"] for r in recs[split]]
            if len(set(ids)) != len(ids):
                raise RuntimeError("%s/%s: duplicate ids" % (name, split))
            for r, e in zip(recs[split], exs):
                for p in r.get("images") or [r["image"]]:
                    if not os.path.exists(os.path.join(tmp, p)):
                        raise RuntimeError("%s: missing %s" % (r["id"], p))
                if e["label"] != r["label"] or e["id"] != r["id"]:
                    raise RuntimeError("%s: example does not match its record" % r["id"])
                if key != "probe" and (FROM_WORDS[list(r["question"]["criteria"])[r["label"]]] != r["primitive"]
                                       or prims[r["label"]] != r["primitive"]):
                    raise RuntimeError("%s: label %d does not decode to %s" % (r["id"], r["label"], r["primitive"]))
        summary[key] = {"dataset": name, "records": meta["records"], "images": len(paths),
                        "placement": dict(linked), "report": reports}
        print("%s: %s records load, %d images (%s), %.0f s" % (name, meta["records"], len(paths), dict(linked),
                                                               time.time() - t0))

    data_vol.reload()
    taken = [n for n in dsts.values() if os.path.exists(os.path.join(root, n))]
    if taken:
        raise RuntimeError("refusing: %s appeared while building; data left in the .tmp dirs" % taken)
    for name in dsts.values():
        os.rename(os.path.join(root, name + ".tmp"), os.path.join(root, name))
    for name in dsts.values():
        open(os.path.join(root, name, "_READY"), "w").close()
    data_vol.commit()
    summary["minutes"] = round((time.time() - t0) / 60, 1)
    print(json.dumps(summary, indent=1))
    return summary


@app.local_entrypoint()
def clean_bigym_bc(src: str = "bigym_v2", dst: str = "bigym_v2c", done_target: float = 0.15,
                   max_drop_frac: float = 0.4, out: str = ""):
    """Build ``<dst>_bc_f1``, ``<dst>_bc_f4`` and ``<dst>_probe`` from ``<src>_*``; see the section comment above
    ``bigym_clean_build``. ``--out`` also saves the summary JSON locally."""
    code = _git_state()
    print("%s -> %s from laya %s%s" % (list(_bc_names(src).values()), list(_bc_names(dst).values()),
                                       code["commit"][:10], " (dirty)" if code["dirty"] else ""))
    summary = bigym_clean_build.remote(src, dst, code, done_target, max_drop_frac)
    if out:
        with open(out, "w") as f:
            json.dump(summary, f, indent=1)
        print("wrote", out)
