"""BiGym on Modal: a perception probe and zero-shot control for a laya-vision checkpoint (``laya.bigymgames``).

    modal run modal_bigym.py::bigym_eval                                  # all six tasks, both parts
    modal run modal_bigym.py::bigym_eval --tasks ReachTarget,DrawerTopClose --parts control --episodes 3
    modal run modal_bigym.py::bigym_eval --parts control --frames 2,4       # the model sees its last 2 / 4 views
    modal run modal_bigym.py::bigym_eval --model thaitea/laya-vision --revision <sha> --out eval-results/x.json

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


def _agent(model: str, revision: str):
    from laya.vlm import VLMAgent

    return VLMAgent(model, revision=revision or None, device="cuda", dtype="bf16")


@app.function(image=image, gpu="L4", cpu=4, timeout=120 * 60, volumes={"/cache/hf": hf_vol})
def bigym_probe(task: str, model: str = MODEL, revision: str = REVISION, n: int = 200) -> dict:
    """The perception probe on one task: ``n`` frames from ``laya.bigymgames.probe_frames``, all of
    ``probe_questions(task)`` in one ``predict`` per frame."""
    gl = _pick_gl()
    from laya import bigymgames as bg

    t0 = time.time()
    frames = bg.probe_frames(task, n)
    qs = bg.probe_questions(task)
    agent = _agent(model, revision)
    rows = []
    for i, fr in enumerate(frames):
        ans = agent.predict({"image": fr["image"]}, qs, strict=True)["answers"]
        truth = {k: v for k, v in fr["truth"].items()}
        for qid, q in qs.items():
            rows.append({"task": task, "frame": i, "seed": fr["seed"], "qid": qid, "label": fr["labels"][qid],
                         "probs": [round(float(p), 5) for p in bg.answer_probs(ans[qid], q)], "truth": truth})
    out = {"task": task, "model": model, "revision": revision, "n": len(frames), "questions": qs,
           "metrics": bg.probe_metrics(rows), "rows": rows, "versions": _versions(), "gl": gl,
           "seconds": round(time.time() - t0, 1)}
    print(json.dumps({"task": task, "metrics": out["metrics"]}))
    return out


def _play(task: str, policy: str, episodes: int, seed: int, model: str = "", revision: str = "",
          frames: int = 1) -> dict:
    gl = _pick_gl()
    from laya import bigymgames as bg

    t0 = time.time()
    if policy == "model":
        fn = bg.model_policy(_agent(model, revision), task, frames)
    elif policy == "oracle":
        fn = bg.oracle_policy
    else:
        fn = bg.random_policy(seed)
    out = bg.play_episodes(task, fn, episodes, seed)
    out.update(policy="model:%s@%s" % (model, revision[:7]) if policy == "model" else policy, frames=frames,
               versions=_versions(), gl=gl, seconds=round(time.time() - t0, 1))
    print(json.dumps({k: v for k, v in out.items() if k != "results"}))
    return out


@app.function(image=image, gpu="L4", cpu=4, timeout=180 * 60, volumes={"/cache/hf": hf_vol})
def bigym_play(task: str, model: str = MODEL, revision: str = REVISION, episodes: int = 20,
               seed: int = 300_000, frames: int = 1) -> dict:
    """The model plays ``episodes`` seeded episodes of ``task``, one ``predict`` per primitive, seeing the last
    ``frames`` decision frames."""
    return _play(task, "model", episodes, seed, model, revision, frames)


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
               seed: int = 300_000, frames: str = "1", out: str = ""):
    """Everything in parallel on one checkpoint; see the module docstring. ``--frames 1,2,4`` plays the model once
    per frame count (results under ``model``, ``model_2f``, ``model_4f``)."""
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
            calls["probe"][t] = bigym_probe.spawn(t, model, revision, probe_n)
        if "control" in want:
            c = {_model_key(n): bigym_play.spawn(t, model, revision, episodes, seed, n) for n in frame_counts}
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
    result = {"model": model, "revision": revision, "tasks": task_list, "parts": sorted(want), "episodes": episodes,
              "probe_n": probe_n, "demos": demos, "seed": seed, "frames": frame_counts, "code": _git_state(),
              "date": time.strftime("%Y-%m-%d"), "errors": _ERRORS, "probe": probes, "control": control}
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(result, f)
    print("\nwrote", path)
