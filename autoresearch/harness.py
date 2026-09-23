"""The fixed autoresearch harness for Laya Vision: train an experiment for 5 minutes, then measure it.

    modal run autoresearch/harness.py --tag <tag> [--desc "what this experiment tries"]

This file is the ground truth, like upstream's ``prepare.py``: experiments never edit it. It sends the current
``autoresearch/experiment.py`` to an H100, where the experiment builds a model and trains it for exactly
``TIME_BUDGET`` seconds of wall clock (data loading and model loading before the clock starts are free). The harness
then, identically for every experiment:

1. fits the per-type temperatures on a fixed calibration set (the last ``N_CALIB`` train records of each
   ``CALIB_DATASETS`` set, which experiments never see);
2. saves the model to ``/ckpt/autoresearch/<tag>/<commit>/`` and reloads it from disk, so only what the checkpoint
   really holds is measured (a layer an experiment drops has to be gone from the saved config too);
3. scores the reloaded model on a fixed sample of every eval set (``EVAL_PER_SET`` seeded questions from each
   ``EVAL_DATASETS`` val split);
4. counts its parameters and times ``predict`` on an L4 in bf16 on ``LATENCY_N`` fixed images.

The three objectives (see ``pareto.py``):

* **quality** = macro accuracy over the eval sets minus the ECE pooled over the questions that have a single right
  answer (not the ones scored against human vote spreads, where ECE is not meaningful). Higher is better.
* **params_m**: parameters of the saved model, in millions. Lower is better.
* **latency_ms**: median ``predict`` time on the L4, preprocessing included. Lower is better.

The result lands in ``autoresearch/runs/<tag>/<commit>.json`` and ``pareto.py`` appends it to
``autoresearch/runs/<tag>/results.tsv`` as keep / discard.
"""
import importlib.util
import json
import os
import subprocess
import sys
import time
from typing import Dict, List

import modal

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO, os.path.join(REPO, "autoresearch")):  # the laya package to ship, and pareto.py
    if _p not in sys.path:
        sys.path.insert(0, _p)

TIME_BUDGET = 300          # seconds of training, as upstream
EVAL_PER_SET = 300         # seeded questions per eval set
LATENCY_N = 100            # images timed on the L4 (after 10 warm-up calls)
N_CALIB = 100              # last train records per calibration set, held out from training
SEED = 0

VQA = ("aokvqa", "scienceqa", "vqav2_yesno")
CAULDRON = tuple("cauldron_" + s for s in (
    "ai2d", "aokvqa", "iconqa", "intergps", "scienceqa", "tqa", "visual7w", "raven", "figureqa", "hateful_memes",
    "nlvr2", "vsr", "vqarad", "clevr", "dvqa", "mapqa", "ocrvqa", "vqav2", "chartqa"))
SCORE = tuple("score_" + s for s in ("vlfeedback", "ava", "richhf", "crisismmd"))
EVAL = tuple("eval_" + s for s in ("koniq", "evalmuse", "cifar10h", "ferplus", "vizwiz", "pope_random", "pope_popular",
                                   "pope_adversarial"))
EVAL_DATASETS = VQA + CAULDRON + SCORE + EVAL
CALIB_DATASETS = CAULDRON + SCORE          # their last N_CALIB train records calibrate every experiment
# What experiments may train on. The eval_* sets stay held out entirely (even where they have a train split): they
# measure how the model does on data it was never tuned toward. The vqa sets' train splits are not prepared.
TRAINABLE_DATASETS = CAULDRON + SCORE

app = modal.App("laya-autoresearch")
hf_vol = modal.Volume.from_name("laya-hf-cache")
data_vol = modal.Volume.from_name("laya-datasets")
ckpt_vol = modal.Volume.from_name("laya-checkpoints")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.14.0", "torchvision==0.29.0", "transformers==5.17.0", "safetensors", "huggingface_hub",
                 "numpy", "pillow", "datasets", "num2words")
    .env({"HF_HOME": "/cache/hf", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_python_source("laya")
)
VOLUMES = {"/cache/hf": hf_vol, "/data": data_vol, "/ckpt": ckpt_vol}
ROOT = "/ckpt/autoresearch"


def ckpt_path(run: str) -> str:
    """A saved run on the checkpoint volume: ``<run>`` under /ckpt/smolvlm, or ``<family>/<run>`` under /ckpt."""
    for base in ("/ckpt/smolvlm", "/ckpt"):
        p = os.path.join(base, run)
        if os.path.exists(os.path.join(p, "vlm_agent_config.json")):
            return p
    raise FileNotFoundError("no checkpoint %r under /ckpt/smolvlm or /ckpt" % run)


def load_split(name: str, split: str) -> List[Dict]:
    from laya.vlm_train import load_jsonl_examples

    if not os.path.exists("/data/vqa/%s/_READY" % name):
        return []
    try:
        return load_jsonl_examples("/data/vqa", name, split)
    except FileNotFoundError:
        return []


class Context:
    """What an experiment gets: the budget, the device, checkpoint lookup and training data. Training data never
    includes the val splits or the calibration tail the harness fits temperatures on."""

    def __init__(self, time_budget_s: float):
        self.time_budget_s = time_budget_s
        self.device = "cuda"
        self.ckpt_path = ckpt_path

    def train_examples(self, names=TRAINABLE_DATASETS) -> List[Dict]:
        out = []
        for name in names:
            if name not in TRAINABLE_DATASETS:
                raise ValueError("%s is not trainable here (one of %s)" % (name, TRAINABLE_DATASETS))
            exs = load_split(name, "train")
            out += exs[:-N_CALIB] if name in CALIB_DATASETS else exs
        return out


def _import_experiment(source: str):
    path = "/tmp/experiment.py"
    with open(path, "w") as f:
        f.write(source)
    spec = importlib.util.spec_from_file_location("experiment", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _summary(metrics: Dict, hard_metrics: Dict) -> Dict:
    names = [n for n in metrics if n != "all"]
    macro = sum(metrics[n]["acc"] for n in names) / len(names)
    ece_hard = hard_metrics["all"]["ece"]
    return {"macro_acc": macro, "ece_hard": ece_hard, "quality": macro - ece_hard, "n_sets": len(names),
            "n_questions": metrics["all"]["n"]}


@app.function(image=image, gpu="H100", cpu=16, memory=65536, timeout=45 * 60, volumes=VOLUMES)
def train_and_eval(source: str, tag: str, commit: str) -> Dict:
    import random

    import torch

    from laya.vlm_train import collect_logits, fit_temperatures_from, metrics_from

    torch.manual_seed(SEED)
    random.seed(SEED)
    data_vol.reload()
    exp = _import_experiment(source)
    ctx = Context(TIME_BUDGET)
    t_setup = time.time()
    agent = exp.build(ctx)
    setup_s = time.time() - t_setup
    t0 = time.time()
    exp.train(agent, ctx)
    train_s = time.time() - t0
    if train_s > TIME_BUDGET + 60:
        raise RuntimeError("training took %.0f s, over the %d s budget" % (train_s, TIME_BUDGET))
    agent.model.eval()

    calib = []
    for name in CALIB_DATASETS:
        calib += load_split(name, "train")[-N_CALIB:]
    temps = fit_temperatures_from(collect_logits(agent.model, agent.processor, calib, batch_size=32, num_workers=8))
    agent.temperature, agent.temperature_by_options = list(temps), {}

    out_dir = os.path.join(ROOT, tag, commit)
    agent.save(out_dir)
    ckpt_vol.commit()
    del agent
    torch.cuda.empty_cache()

    from laya.vlm import VLMAgent

    agent = VLMAgent(out_dir, device="cuda")
    params = sum(p.numel() for p in agent.model.parameters())
    rng = random.Random(SEED)
    val = []
    for name in EVAL_DATASETS:
        exs = load_split(name, "val")
        val += rng.sample(exs, min(EVAL_PER_SET, len(exs)))
    records = collect_logits(agent.model, agent.processor, val, batch_size=32, num_workers=12)
    metrics = metrics_from(records, agent.temperature)
    # one right answer (one-hot target): ECE is meaningful there, not against a spread of human votes
    hard = metrics_from([r for r in records if float(r["target"].max()) >= 0.999], agent.temperature)
    summary = _summary(metrics, hard)
    summary["params_m"] = params / 1e6
    res = {"tag": tag, "commit": commit, "summary": summary, "metrics": metrics, "temperature": temps,
           "setup_s": round(setup_s, 1), "train_s": round(train_s, 1), "checkpoint": out_dir,
           "config": {k: v for k, v in agent.cfg.items() if isinstance(v, (str, int, float, bool)) or v is None}}
    print(json.dumps(summary))
    return res


@app.function(image=image, gpu="L4", cpu=4, memory=16384, timeout=20 * 60, volumes=VOLUMES)
def latency(tag: str, commit: str) -> Dict:
    import random

    import numpy as np
    import torch
    from PIL import Image

    from laya.vlm import VLMAgent

    ckpt_vol.reload()
    agent = VLMAgent(os.path.join(ROOT, tag, commit), device="cuda", dtype="bf16")
    rng = random.Random(SEED)
    per = -(-LATENCY_N // len(EVAL_DATASETS))  # ceil, then cut to LATENCY_N
    cases = []
    for name in EVAL_DATASETS:
        exs = [ex for ex in load_split(name, "val") if isinstance(ex["state"], dict) and ex["state"].get("image")]
        for ex in rng.sample(exs, min(per, len(exs))):
            state = dict(ex["state"])
            with Image.open(state["image"]) as im:
                state["image"] = im.convert("RGB")
            q = ex["q"]
            crit = list(q["crit"]) if q["t"] == "choice" else q["crit"]
            cases.append((state, {"q": {"type": q["t"], "instructions": q["ins"], "criteria": crit}}))
    cases = cases[:LATENCY_N]
    for state, qs in cases[:10]:
        agent.predict(state, qs)
    ms = []
    for state, qs in cases:
        torch.cuda.synchronize()
        t = time.perf_counter()
        agent.predict(state, qs)
        torch.cuda.synchronize()
        ms.append((time.perf_counter() - t) * 1000)
    return {"latency_ms": float(np.median(ms)), "latency_p90_ms": float(np.percentile(ms, 90)), "n": len(ms),
            "gpu": torch.cuda.get_device_name(0)}


@app.function(image=image, timeout=10 * 60, volumes={"/ckpt": ckpt_vol})
def prune_checkpoints(tag: str, keep: List[str]) -> List[str]:
    """Delete this tag's saved checkpoints that are not in ``keep`` (the frontier); returns what was removed."""
    import shutil

    base = os.path.join(ROOT, tag)
    removed = []
    for name in sorted(os.listdir(base)) if os.path.isdir(base) else []:
        if name not in keep and os.path.exists(os.path.join(base, name, "vlm_agent_config.json")):
            shutil.rmtree(os.path.join(base, name))
            removed.append(name)
    ckpt_vol.commit()
    return removed


def _git(*args) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True).stdout.strip()


@app.local_entrypoint()
def main(tag: str, desc: str = "", prune: bool = True):
    import pareto

    exp_path = os.path.join(REPO, "autoresearch", "experiment.py")
    if _git("status", "--porcelain", "--", exp_path):
        raise SystemExit("commit autoresearch/experiment.py first: results are keyed by commit")
    commit = _git("rev-parse", "--short=7", "HEAD")
    desc = desc or _git("log", "-1", "--format=%s")
    runs = os.path.join(REPO, "autoresearch", "runs", tag)
    tsv = os.path.join(runs, "results.tsv")
    os.makedirs(runs, exist_ok=True)
    with open(exp_path) as f:
        source = f.read()
    t0 = time.time()
    try:
        res = train_and_eval.remote(source, tag, commit)
        res["summary"].update({k: v for k, v in latency.remote(tag, commit).items() if k.startswith("latency")})
    except Exception as e:
        print("crash: %r" % (e,))
        pareto.append_tsv(tsv, pareto.crash_row(commit, desc))
        print("status: crash")
        raise SystemExit(1)
    res.update(description=desc, total_s=round(time.time() - t0, 1))
    out = os.path.join(runs, commit + ".json")
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    s = res["summary"]
    print("---")
    for k in ("quality", "macro_acc", "ece_hard", "params_m", "latency_ms", "latency_p90_ms"):
        print("%-17s %.4f" % (k + ":", s[k]))
    print("%-17s %.1f" % ("train_seconds:", res["train_s"]))
    print("%-17s %.1f" % ("total_seconds:", res["total_s"]))
    rows = pareto.read_tsv(tsv)
    row = pareto.row_from_result(res, commit, desc)
    status, beaten = pareto.decide(pareto.frontier(rows), row)
    row["status"] = status
    pareto.append_tsv(tsv, row)
    print("status:           %s" % status)
    if beaten:
        print("now dominates:    %s" % ", ".join(p["commit"] for p in beaten))
    print(pareto.show(pareto.read_tsv(tsv)))
    if prune:
        keep = [r["commit"] for r in pareto.frontier(pareto.read_tsv(tsv))]
        removed = prune_checkpoints.remote(tag, keep)
        if removed:
            print("pruned checkpoints off the frontier: %s" % ", ".join(removed))
