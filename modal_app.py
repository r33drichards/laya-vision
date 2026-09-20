"""Modal jobs for the SmolVLM-backed decision model (``laya.vlm``).

    modal run modal_app.py::test                     # pytest on a GPU + latency
    modal run modal_app.py::finetune --minutes 18    # short fine-tune + held-out acc / ECE
    modal run modal_app.py::evaluate --run-name <run> # re-evaluate a saved checkpoint
    modal run --detach modal_app.py::finetune_long   # ~3-epoch A100 run with per-epoch eval + best checkpoint
    modal run modal_app.py::try_model --image photo.jpg [--questions q.json] [--text "..."]  # ask a checkpoint about an image
    modal run modal_app.py::publish [--repo user/name] [--run all3-3ep/best]  # push checkpoint + hf_model_card.md to the HF Hub
    modal run modal_app.py::prepare_doom_basic       # auto-labelled ViZDoom "basic" frames -> /data/vqa/doom_basic
    modal run modal_app.py::doom_eval --models all3-3ep/best  # play "basic": expert / random / always-attack / models

Volumes (created out of band; never ``modal deploy`` this app):
    laya-hf-cache     -> /cache/hf   (HF_HOME, shared model weights)
    laya-datasets     -> /data       (read-only; /data/vqa/<name>/{<split>.jsonl, images/, _READY})
    laya-checkpoints  -> /ckpt       (this app writes only under /ckpt/smolvlm/)
"""
import json
import os
import subprocess
import sys
import time

import modal

app = modal.App("laya-smolvlm")

hf_vol = modal.Volume.from_name("laya-hf-cache")
data_vol = modal.Volume.from_name("laya-datasets")
ckpt_vol = modal.Volume.from_name("laya-checkpoints")

base_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.14.0",
        "torchvision==0.29.0",
        "transformers==5.17.0",
        "safetensors",
        "huggingface_hub",
        "numpy",
        "pillow",
        "datasets",
        "pytest",
        "num2words",
    )
    .env({"HF_HOME": "/cache/hf", "TOKENIZERS_PARALLELISM": "false"})
)


def _with_local_code(img):
    return img.add_local_dir("tests", "/root/tests").add_local_python_source("laya")


image = _with_local_code(base_image)

BACKBONE = "HuggingFaceTB/SmolVLM-256M-Instruct"
DATASETS = ("aokvqa", "scienceqa", "vqav2_yesno")
CKPT_ROOT = "/ckpt/smolvlm"


@app.function(image=image, gpu="L4", timeout=30 * 60, volumes={"/cache/hf": hf_vol})
def test():
    """Run tests/test_vlm.py on the GPU, then time predict() in fp32 and bf16."""
    import torch

    print("GPU:", torch.cuda.get_device_name(0), "| torch", torch.__version__)
    rc = subprocess.run(
        [sys.executable, "-m", "pytest", "/root/tests/test_vlm.py", "-v", "-s", "-p", "no:cacheprovider", "-W", "ignore"],
        cwd="/root",
    ).returncode
    hf_vol.commit()

    from PIL import Image

    from laya.vlm import VLMAgent

    img = Image.new("RGB", (96, 96), (255, 255, 255))
    img.paste(Image.new("RGB", (48, 48), (220, 20, 20)), (24, 24))
    one = {"is_red": {"type": "noul", "instructions": "Is the square red?"}}
    three = dict(one, color={"type": "choice", "instructions": "What color?", "criteria": ["red", "blue", "green"]},
                 size={"type": "score", "instructions": "How big?", "criteria": ["small", "medium", "large"]})
    for dtype in ("fp32", "bf16"):
        agent = VLMAgent(backbone=BACKBONE, device="cuda", dtype=dtype)
        for label, state, qs in (("image, 1 q", {"image": img}, one), ("image, 3 q", {"image": img}, three),
                                 ("text, 1 q", "Customer: I was billed twice.", one), ("text, 3 q", "Customer: I was billed twice.", three)):
            for _ in range(3):
                agent.predict(state, qs)
            ts = []
            for _ in range(20):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                agent.predict(state, qs)
                torch.cuda.synchronize()
                ts.append((time.perf_counter() - t0) * 1000)
            ts.sort()
            print("latency %-5s %-11s median %6.1f ms  p90 %6.1f ms" % (dtype, label, ts[10], ts[18]))
        del agent
    if rc != 0:
        raise SystemExit("pytest failed with exit code %d" % rc)


def _load_split(name: str, split: str, limit):
    from laya.vlm_train import load_jsonl_examples

    return load_jsonl_examples("/data/vqa", name, split, limit=limit)


def _load_data(datasets: str, train_split: str, val_split: str, n_calib: int, max_train: int, max_val: int, caps: dict):
    """(train, calib, val) examples. The LAST ``n_calib`` train records per dataset (file order) are held out of
    training for temperature fitting, as in the SigLIP-projector runs (runs before commit 2651641 held out a seeded
    random 300 instead)."""
    data_vol.reload()
    train_ex, calib_ex, val_ex = [], [], []
    for name in [d for d in datasets.split(",") if d]:
        if not os.path.exists("/data/vqa/%s/_READY" % name):
            print("dataset %s not ready (no _READY); skipping" % name)
            continue
        tr = _load_split(name, train_split, max_train + n_calib if max_train else 0)
        calib_ex += tr[-n_calib:]
        train_ex += tr[:-n_calib]
        va = _load_split(name, val_split, caps.get(name, max_val) or 0)
        val_ex += va
        print("dataset %s: %d train, %d calib, %d val" % (name, len(tr) - n_calib, min(n_calib, len(tr)), len(va)))
    if not train_ex:
        raise SystemExit("no training data (datasets not ready)")
    return train_ex, calib_ex, val_ex


@app.function(
    image=image,
    gpu="A10G",
    cpu=16,
    memory=32768,
    timeout=40 * 60,
    volumes={"/cache/hf": hf_vol, "/data": data_vol.read_only(), "/ckpt": ckpt_vol},
)
def finetune(
    datasets: str = ",".join(DATASETS),
    minutes: float = 18.0,
    freeze: str = "full",
    n_last: int = 8,
    batch_size: int = 16,
    lr_head: float = 1e-4,
    lr_backbone: float = 2e-5,
    sigma: float = 1.0,
    sigma_end: float = 0.3,
    group_size: int = 8,
    w_sph: float = 0.5,
    w_ce: float = 0.0,
    train_split: str = "train",
    val_split: str = "val",
    max_train: int = 0,
    max_val: int = 0,
    n_calib: int = 300,
    eval_every: int = 400,
    val_caps: str = "",
    num_workers: int = 14,
    synthetic: bool = False,
    run_name: str = "",
):
    """Short fine-tune on the prepared VQA sets; logs loss and held-out accuracy / ECE, saves to /ckpt/smolvlm/<run>.

    ``max_train`` / ``max_val`` = 0 means the whole split; otherwise the first N records in file order.
    ``val_caps`` overrides the val cap per dataset, e.g. ``"vqav2_yesno=1000"``.
    """
    import torch

    from laya.vlm import VLMAgent
    from laya.vlm_train import collect_logits, fit_temperatures_from, format_metrics, metrics_from, synthetic_examples, train

    t_start = time.time()
    print("GPU:", torch.cuda.get_device_name(0), "| torch", torch.__version__)
    run_name = run_name or time.strftime("run-%Y%m%d-%H%M%S")
    out_dir = os.path.join(CKPT_ROOT, run_name)

    caps = {k: int(v) for k, v in (kv.split("=") for kv in val_caps.split(",") if kv)}
    if synthetic:
        train_ex = [dict(ex, dataset="synthetic") for ex in synthetic_examples(64, seed=0)]
        calib_ex = [dict(ex, dataset="synthetic") for ex in synthetic_examples(16, seed=2)]
        val_ex = [dict(ex, dataset="synthetic") for ex in synthetic_examples(32, seed=1)]
    else:
        train_ex, calib_ex, val_ex = _load_data(datasets, train_split, val_split, n_calib, max_train, max_val, caps)

    agent = VLMAgent(backbone=BACKBONE, device="cuda")
    hf_vol.commit()
    model, proc = agent.model, agent.processor
    ev_kw = dict(batch_size=32, num_workers=num_workers)
    small_val = []
    for name in sorted({ex["dataset"] for ex in val_ex}):
        small_val += [ex for ex in val_ex if ex["dataset"] == name][:200]

    log = {"run": run_name, "args": dict(datasets=datasets, minutes=minutes, freeze=freeze, n_last=n_last, batch_size=batch_size,
                                         lr_head=lr_head, lr_backbone=lr_backbone, max_train=max_train, max_val=max_val, val_caps=caps,
                                         objective=dict(sigma=sigma, sigma_end=sigma_end, group_size=group_size, w_sph=w_sph, w_ce=w_ce)),
           "evals": []}
    base = metrics_from(collect_logits(model, proc, small_val, **ev_kw))
    print("[eval step 0, untrained head] " + format_metrics(base), flush=True)
    log["evals"].append({"step": 0, **base})

    def eval_fn(step):
        m = metrics_from(collect_logits(model, proc, small_val, **ev_kw))
        print("[eval step %d] %s" % (step, format_metrics(m)), flush=True)
        log["evals"].append({"step": step, **m})

    losses = train(
        model, proc, train_ex, steps=10**9, batch_size=batch_size, freeze=freeze, n_last=n_last,
        lr_head=lr_head, lr_backbone=lr_backbone, device="cuda", log_every=50, max_minutes=minutes,
        num_workers=num_workers, warmup=100, eval_fn=eval_fn, eval_every=eval_every,
        sigma=sigma, sigma_end=sigma_end, group_size=group_size, w_sph=w_sph, w_ce=w_ce,
    )
    log["steps"], log["examples_seen"] = len(losses), len(losses) * batch_size
    log["loss_first50"], log["loss_last50"] = sum(losses[:50]) / min(50, len(losses)), sum(losses[-50:]) / min(50, len(losses))
    print("trained %d steps (%d examples); loss first50 %.4f -> last50 %.4f"
          % (log["steps"], log["examples_seen"], log["loss_first50"], log["loss_last50"]), flush=True)

    t_eval = time.time()
    temps = fit_temperatures_from(collect_logits(model, proc, calib_ex, **ev_kw))
    val_records = collect_logits(model, proc, val_ex, **ev_kw)
    print("calib + val eval: %d examples in %.1f min" % (len(calib_ex) + len(val_ex), (time.time() - t_eval) / 60))
    log["temperature"] = temps
    log["val_raw"] = metrics_from(val_records)
    log["val_calibrated"] = metrics_from(val_records, temps)
    print("temperatures (choice, score, noul):", [round(t, 3) for t in temps])
    print("[final val, T=1]        " + format_metrics(log["val_raw"]))
    print("[final val, calibrated] " + format_metrics(log["val_calibrated"]))

    agent.temperature = temps
    agent.save(out_dir, include_backbone=freeze != "head")
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(log, f, indent=2)
    ckpt_vol.commit()
    print("saved %s (%.1f min total)" % (out_dir, (time.time() - t_start) / 60))
    return {k: log[k] for k in ("run", "steps", "loss_first50", "loss_last50", "temperature", "val_raw", "val_calibrated")}


@app.function(
    image=image,
    gpu="A100",
    cpu=24,
    memory=65536,
    timeout=170 * 60,
    volumes={"/cache/hf": hf_vol, "/data": data_vol.read_only(), "/ckpt": ckpt_vol},
)
def finetune_long(
    datasets: str = ",".join(DATASETS),
    epochs: float = 3.0,
    max_minutes: float = 90.0,
    batch_size: int = 32,
    lr_head: float = 1e-4,
    lr_backbone: float = 2e-5,
    sigma: float = 1.0,
    sigma_end: float = 0.3,
    group_size: int = 8,
    w_sph: float = 0.5,
    w_ce: float = 0.0,
    lr_ref_batch: int = 16,
    warmup_frac: float = 0.03,
    evals_per_epoch: float = 2.0,
    train_eval_n: int = 1000,
    max_passes: float = 0.0,
    n_calib: int = 300,
    num_workers: int = 22,
    run_name: str = "all3-3ep",
    init_from: str = "",
):
    """Multi-epoch fine-tune (vision tower frozen) with per-epoch train/val tracking and best-checkpoint keeping.

    * ``epochs`` counts samples over the combined train set (datasets are still sampled equally, so small sets
      repeat more; ``max_passes`` > 0 caps the passes over any one dataset).
    * LRs are given for ``lr_ref_batch`` and scaled by sqrt(batch_size / lr_ref_batch) (the usual rule for Adam).
    * Linear warmup over ``warmup_frac`` of the steps, then cosine decay to 10%; ``max_minutes`` caps wall-clock.
    * Each eval scores the full val splits and the first ``train_eval_n`` train records per dataset (seen in
      training) to expose overfitting. ``last/`` is saved every eval; ``best/`` when mean per-dataset val acc
      improves. The final model is the best one: temperatures are fitted on the calibration holdout, then the full
      val splits are scored raw and calibrated, plus an option-order-bias check on aokvqa.
    * ``init_from`` (a run under /ckpt/smolvlm, e.g. ``all3-3ep/best``) continues from a trained checkpoint
      instead of a fresh head. Question types absent from the calibration holdout keep that checkpoint's
      temperature rather than being reset to 1.0.
    """
    import math

    import torch

    from laya.common import QTYPES
    from laya.vlm import VLMAgent, _permutations
    from laya.vlm_train import collect_logits, cyclic_orders, fit_temperatures_from, format_metrics, metrics_from, train

    t_start = time.time()
    print("GPU:", torch.cuda.get_device_name(0), "| torch", torch.__version__)
    out_dir = os.path.join(CKPT_ROOT, run_name)
    train_ex, calib_ex, val_ex = _load_data(datasets, "train", "val", n_calib, 0, 0, {})
    names = sorted({ex["dataset"] for ex in train_ex})
    train_eval = []
    for name in names:
        train_eval += [ex for ex in train_ex if ex["dataset"] == name][:train_eval_n]

    scale = math.sqrt(batch_size / lr_ref_batch)
    lr_h, lr_b = lr_head * scale, lr_backbone * scale
    steps = int(math.ceil(epochs * len(train_ex) / batch_size))
    eval_every = max(1, int(round(steps / (epochs * evals_per_epoch))))
    warmup = max(1, int(warmup_frac * steps))
    sizes = {n: sum(ex["dataset"] == n for ex in train_ex) for n in names}
    per_ds = epochs * len(train_ex) / len(names)
    print("plan: %d steps x batch %d (%.1f epochs of %d), warmup %d, eval every %d, lr head %.2e backbone %.2e"
          % (steps, batch_size, epochs, len(train_ex), warmup, eval_every, lr_h, lr_b))
    print("expected passes per dataset with equal sampling (before max_passes=%s): %s"
          % (max_passes or None, {n: round(per_ds / sizes[n], 2) for n in names}))

    if init_from:
        agent = VLMAgent(os.path.join(CKPT_ROOT, init_from), device="cuda")
        print("initialised from %s (temperatures %s)" % (init_from, [round(t, 3) for t in agent.temperature]))
    else:
        agent = VLMAgent(backbone=BACKBONE, device="cuda")
    init_temps = list(agent.temperature)
    hf_vol.commit()
    model, proc = agent.model, agent.processor
    ev_kw = dict(batch_size=64, num_workers=num_workers)
    log = {"run": run_name, "args": dict(datasets=datasets, epochs=epochs, max_minutes=max_minutes, batch_size=batch_size,
                                         lr_head=lr_h, lr_backbone=lr_b, warmup=warmup, steps=steps, eval_every=eval_every,
                                         max_passes=max_passes, n_calib=n_calib, train_eval_n=train_eval_n,
                                         objective=dict(sigma=sigma, sigma_end=sigma_end, group_size=group_size, w_sph=w_sph, w_ce=w_ce)),
           "evals": []}
    best = {"score": -1.0, "step": None, "state": None}

    def write_log():
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "metrics.json"), "w") as f:
            json.dump(log, f, indent=2)

    def eval_fn(step):
        te = time.time()
        val_m = metrics_from(collect_logits(model, proc, val_ex, **ev_kw))
        tr_m = metrics_from(collect_logits(model, proc, train_eval, **ev_kw))
        score = sum(val_m[n]["acc"] for n in names) / len(names)
        row = {"step": step, "epoch": round(step * batch_size / len(train_ex), 2), "mean_val_acc": score,
               "val": val_m, "train": tr_m}
        log["evals"].append(row)
        print("[eval step %d, epoch %.2f] mean val acc %.4f | %s" % (step, row["epoch"], score, " | ".join(
            "%s train %.3f val %.3f (gap %+.3f) ece %.3f nll %.3f" % (n, tr_m[n]["acc"], val_m[n]["acc"],
                                                                     tr_m[n]["acc"] - val_m[n]["acc"], val_m[n]["ece"], val_m[n]["nll"])
            for n in names)), flush=True)
        agent.save(os.path.join(out_dir, "last"))
        if score > best["score"]:
            best.update(score=score, step=step, state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
            agent.save(os.path.join(out_dir, "best"))
            print("  new best (step %d); saved %s/best" % (step, out_dir), flush=True)
        log["best_step"], log["best_mean_val_acc"] = best["step"], best["score"]
        write_log()
        ckpt_vol.commit()
        print("  eval + save took %.1f min" % ((time.time() - te) / 60), flush=True)

    stats = {}
    losses = train(
        model, proc, train_ex, steps=steps, batch_size=batch_size, freeze="full", lr_head=lr_h, lr_backbone=lr_b,
        device="cuda", log_every=100, max_minutes=max_minutes, num_workers=num_workers, warmup=warmup,
        eval_fn=eval_fn, eval_every=eval_every, max_passes=max_passes or None, stats=stats,
        sigma=sigma, sigma_end=sigma_end, group_size=group_size, w_sph=w_sph, w_ce=w_ce,
    )
    if not log["evals"] or log["evals"][-1]["step"] != stats["steps"]:
        eval_fn(stats["steps"])
    chunk = max(1, len(losses) // 10)
    log["train_stats"] = dict(stats, passes={n: round(stats["samples_per_dataset"].get(n, 0) / sizes[n], 2) for n in names})
    log["loss_curve"] = [{"steps": "%d-%d" % (i, min(i + chunk, len(losses)) - 1), "mean_loss": sum(losses[i:i + chunk]) / len(losses[i:i + chunk])}
                         for i in range(0, len(losses), chunk)]
    print("train stats:", json.dumps(log["train_stats"]))
    print("loss curve (10 chunks):", ", ".join("%.3f" % c["mean_loss"] for c in log["loss_curve"]))

    # final model = best checkpoint by mean val acc
    model.load_state_dict(best["state"])
    model.eval()
    print("final model: best checkpoint from step %d (mean val acc %.4f)" % (best["step"], best["score"]))
    temps = fit_temperatures_from(collect_logits(model, proc, calib_ex, **ev_kw))
    n_calib_type = [sum(QTYPES[ex["q"]["t"]] == t for ex in calib_ex) for t in range(3)]
    temps = [t if n >= 10 else init_temps[i] for i, (t, n) in enumerate(zip(temps, n_calib_type))]
    val_records = collect_logits(model, proc, val_ex, **ev_kw)
    log["temperature"] = temps
    log["final"] = {"step": best["step"], "val_raw": metrics_from(val_records), "val_calibrated": metrics_from(val_records, temps)}
    print("temperatures (choice, score, noul):", [round(t, 3) for t in temps])
    print("[final val, T=1]        " + format_metrics(log["final"]["val_raw"]))
    print("[final val, calibrated] " + format_metrics(log["final"]["val_calibrated"]))

    # option-order bias on aokvqa (4-way choice)
    aok = [ex for ex in val_ex if ex["dataset"] == "aokvqa"]
    if aok:
        p4 = collect_logits(model, proc, aok, orders=lambda k: _permutations(k, 4), **ev_kw)
        cyc = collect_logits(model, proc, aok, orders=cyclic_orders(4), **ev_kw)
        p1 = [r for r in val_records if r["dataset"] == "aokvqa"]
        bias = {"n_permutations=1": {"raw": metrics_from(p1)["aokvqa"], "calibrated": metrics_from(p1, temps)["aokvqa"]},
                "n_permutations=4": {"raw": metrics_from(p4)["aokvqa"], "calibrated": metrics_from(p4, temps)["aokvqa"]}}
        cyc_acc = []
        for s_ in range(4):
            recs = [dict(r, logits=r["logits_per_order"][s_]) for r in cyc]
            cyc_acc.append(metrics_from(recs)["aokvqa"]["acc"])
        bias["cyclic_shift_acc"] = cyc_acc
        bias["cyclic_acc_spread"] = max(cyc_acc) - min(cyc_acc)
        log["order_bias_aokvqa"] = bias
        for k_ in ("n_permutations=1", "n_permutations=4"):
            print("[aokvqa %s] raw acc %.4f ece %.4f nll %.4f | calibrated ece %.4f nll %.4f" % (
                k_, bias[k_]["raw"]["acc"], bias[k_]["raw"]["ece"], bias[k_]["raw"]["nll"],
                bias[k_]["calibrated"]["ece"], bias[k_]["calibrated"]["nll"]))
        print("[aokvqa cyclic shifts 0..3] acc %s | spread %.4f" % (", ".join("%.4f" % a for a in cyc_acc), bias["cyclic_acc_spread"]))

    agent.temperature = temps
    agent.save(os.path.join(out_dir, "best"))
    write_log()
    ckpt_vol.commit()
    print("saved %s/best with temperatures (%.1f min total)" % (out_dir, (time.time() - t_start) / 60))
    return {k: log[k] for k in ("run", "best_step", "best_mean_val_acc", "temperature", "final", "train_stats")}


@app.function(
    image=image,
    gpu="A10G",
    cpu=16,
    memory=32768,
    timeout=30 * 60,
    volumes={"/cache/hf": hf_vol, "/data": data_vol.read_only(), "/ckpt": ckpt_vol.read_only()},
)
def evaluate(run_name: str, datasets: str = ",".join(DATASETS), val_split: str = "val", max_val: int = 0):
    """Evaluate a saved checkpoint (/ckpt/smolvlm/<run_name>) on the val splits, raw and with its temperatures."""
    import torch

    from laya.vlm import VLMAgent
    from laya.vlm_train import collect_logits, format_metrics, metrics_from

    print("GPU:", torch.cuda.get_device_name(0))
    data_vol.reload()
    val_ex = []
    for name in [d for d in datasets.split(",") if d]:
        if not os.path.exists("/data/vqa/%s/_READY" % name):
            print("dataset %s not ready (no _READY); skipping" % name)
            continue
        va = _load_split(name, val_split, max_val or 0)
        print("dataset %s: %d val" % (name, len(va)))
        val_ex += va
    agent = VLMAgent(os.path.join(CKPT_ROOT, run_name), device="cuda")
    records = collect_logits(agent.model, agent.processor, val_ex, batch_size=32, num_workers=14)
    raw, cal = metrics_from(records), metrics_from(records, agent.temperature)
    print("temperatures (choice, score, noul):", [round(t, 3) for t in agent.temperature])
    print("[val, T=1]        " + format_metrics(raw))
    print("[val, calibrated] " + format_metrics(cal))
    return {"val_raw": raw, "val_calibrated": cal}


DEMO_QUESTIONS = {
    "scene": {
        "type": "choice",
        "instructions": "Where was this photo most likely taken?",
        "criteria": ["indoors", "city street", "nature or countryside", "beach or water"],
    },
    "has_person": {"type": "noul", "instructions": "Is there at least one person in the image?"},
    "has_animal": {"type": "noul", "instructions": "Is there an animal in the image?"},
    "clutter": {
        "type": "score",
        "instructions": "How cluttered or busy is the image?",
        "criteria": ["minimal, one clear subject", "some objects", "very busy, many objects"],
    },
}


@app.function(image=image, gpu="L4", timeout=10 * 60, volumes={"/cache/hf": hf_vol, "/ckpt": ckpt_vol.read_only()})
def ask(image_bytes: bytes, questions: dict, state_text: str = "", run_name: str = "all3-3ep/best", n_permutations: int = 1):
    """Load a saved checkpoint and answer typed questions about one image (plus optional text state)."""
    import io

    from PIL import Image

    from laya.vlm import VLMAgent

    t0 = time.time()
    agent = VLMAgent(os.path.join(CKPT_ROOT, run_name), device="cuda")
    load_s = time.time() - t0
    state = {"image": Image.open(io.BytesIO(image_bytes)).convert("RGB")}
    if state_text:
        state["text"] = state_text
    agent.predict(state, questions)  # warm-up
    t0 = time.time()
    out = agent.predict(state, questions, n_permutations=n_permutations)
    out["timing"] = {"load_s": round(load_s, 1), "predict_ms": round((time.time() - t0) * 1000, 1)}
    return out


@app.local_entrypoint()
def try_model(image: str, questions: str = "", text: str = "", run: str = "all3-3ep/best", perms: int = 1):
    """modal run modal_app.py::try_model --image photo.jpg [--questions q.json] [--text "..."] [--run all3-18m]

    ``--image`` is a local path or an http(s) URL. ``--questions`` is a JSON file in the ``predict`` schema;
    without it a small demo set is used.
    """
    if image.startswith(("http://", "https://")):
        import urllib.request

        req = urllib.request.Request(image, headers={"User-Agent": "laya-try/1.0"})
        with urllib.request.urlopen(req) as r:
            data = r.read()
    else:
        with open(image, "rb") as f:
            data = f.read()
    qs = DEMO_QUESTIONS
    if questions:
        with open(questions) as f:
            qs = json.load(f)
    out = ask.remote(data, qs, state_text=text, run_name=run, n_permutations=perms)
    for qid, a in out["answers"].items():
        if a["type"] == "choice":
            probs = ", ".join("%s %.2f" % kv for kv in sorted(a["probabilities"].items(), key=lambda kv: -kv[1]))
            print("%-12s choice  %-22s conf %.2f   [%s]" % (qid, a["choice"], a["confidence"], probs))
        elif a["type"] == "score":
            print("%-12s score   %.2f / %d            conf %.2f   %s" % (qid, a["score"], len(a["legend"]) - 1, a["confidence"], a["legend"]))
        else:
            print("%-12s noul    P(true) = %.3f" % (qid, a["noul"]))
    print("timing:", out["timing"])


@app.function(
    image=image,
    timeout=30 * 60,
    volumes={"/ckpt": ckpt_vol.read_only()},
    secrets=[modal.Secret.from_name("huggingface-thaitea")],
)
def push_to_hub(repo_id: str, run_name: str, model_card: str, metrics_path: str = "", private: bool = False):
    """Upload /ckpt/smolvlm/<run_name> (plus a model card and training metrics) to a Hugging Face model repo."""
    import shutil
    import tempfile

    from huggingface_hub import HfApi

    src = os.path.join(CKPT_ROOT, run_name)
    if not os.path.exists(os.path.join(src, "vlm_agent_config.json")):
        raise SystemExit("no checkpoint at %s" % src)
    stage = tempfile.mkdtemp()
    shutil.copytree(src, stage, dirs_exist_ok=True)
    with open(os.path.join(stage, "README.md"), "w") as f:
        f.write(model_card)
    if metrics_path:
        shutil.copy(os.path.join(CKPT_ROOT, metrics_path), os.path.join(stage, "training_metrics.json"))
    for root, _, files in os.walk(stage):
        for name in files:
            p = os.path.join(root, name)
            print("%10d  %s" % (os.path.getsize(p), os.path.relpath(p, stage)))
    api = HfApi()
    api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    info = api.upload_folder(folder_path=stage, repo_id=repo_id, repo_type="model",
                             commit_message="Upload %s checkpoint" % run_name)
    print("uploaded:", info.commit_url)
    return info.commit_url


@app.local_entrypoint()
def publish(repo: str = "thaitea/laya-vision-smolvlm-256m", run: str = "all3-3ep/best",
            metrics: str = "all3-3ep/metrics.json", card: str = "hf_model_card.md", private: bool = False):
    """modal run modal_app.py::publish  -- push a checkpoint + hf_model_card.md to the Hugging Face Hub."""
    with open(card) as f:
        text = f.read()
    print(push_to_hub.remote(repo, run, text, metrics_path=metrics, private=private))


# ---------------------------------------------------------------------------------------------------------
# ViZDoom "basic": auto-labelled training data and closed-loop evaluation
# ---------------------------------------------------------------------------------------------------------

doom_image = _with_local_code(base_image.pip_install("vizdoom"))


def _doom_game(scenario: str = "basic", labels: bool = False):
    import vizdoom as vzd

    g = vzd.DoomGame()
    g.load_config(os.path.join(vzd.scenarios_path, scenario + ".cfg"))
    g.set_window_visible(False)
    g.set_screen_format(vzd.ScreenFormat.RGB24)
    g.set_screen_resolution(vzd.ScreenResolution.RES_320X240)
    g.set_labels_buffer_enabled(labels)
    g.init()
    return g


@app.function(image=doom_image, cpu=8, memory=16384, timeout=60 * 60, volumes={"/data": data_vol})
def prepare_doom_basic(n_train: int = 20000, n_val: int = 2000, eps: float = 0.3, tics: int = 4, seed: int = 0):
    """Write /data/vqa/doom_basic/{train,val}.jsonl + images/ from ViZDoom ``basic``, labelled by a scripted expert.

    Every frame with a visible monster becomes a ``choice`` question (the same one the live viewer asks) whose
    label is the expert's button: ATTACK if the monster covers the crosshair, else strafe toward it. The frames
    come from an epsilon-expert behaviour policy (random button with prob ``eps``), so the data also covers
    off-target states the expert alone would rarely visit. Train and val use disjoint episode seeds.
    """
    import random
    import shutil
    from collections import Counter

    from PIL import Image

    from laya.games import doom_basic_expert, doom_buttons, doom_question

    final_dir = "/data/vqa/doom_basic"
    tmp_dir = final_dir + ".tmp"
    shutil.rmtree(tmp_dir, ignore_errors=True)
    os.makedirs(os.path.join(tmp_dir, "images"))
    g = _doom_game("basic", labels=True)
    buttons = doom_buttons(g)
    one_hot = {b: [i == j for j in range(len(buttons))] for i, b in enumerate(buttons)}
    q = doom_question("basic", buttons)["action"]
    meta = {"source": "ViZDoom basic, scripted expert from the labels buffer", "buttons": buttons, "eps": eps, "tics": tics}

    for split, n_target, seed0 in (("train", n_train, seed), ("val", n_val, seed + 1_000_000)):
        rng = random.Random(seed0)
        counts, n, ep = Counter(), 0, 0
        with open(os.path.join(tmp_dir, split + ".jsonl"), "w") as f:
            while n < n_target:
                g.set_seed(seed0 + ep)
                g.new_episode()
                t = 0
                while not g.is_episode_finished() and n < n_target:
                    s = g.get_state()
                    lab = doom_basic_expert(s.labels)
                    if lab is not None:
                        rid = "%s-%06d-%03d" % (split, ep, t)
                        Image.fromarray(s.screen_buffer).save(os.path.join(tmp_dir, "images", rid + ".jpg"), quality=92)
                        f.write(json.dumps({"id": rid, "image": "images/%s.jpg" % rid, "state_text": None,
                                            "question": q, "label": buttons.index(lab)}) + "\n")
                        counts[lab] += 1
                        n += 1
                    act = lab if (lab is not None and rng.random() >= eps) else rng.choice(buttons)
                    g.make_action(one_hot[act], tics)
                    t += 1
                ep += 1
        meta[split] = {"records": n, "episodes": ep, "labels": dict(counts)}
        print("%s: %d frames from %d episodes, labels %s" % (split, n, ep, dict(counts)))
    g.close()
    with open(os.path.join(tmp_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    shutil.rmtree(final_dir, ignore_errors=True)
    os.rename(tmp_dir, final_dir)
    open(os.path.join(final_dir, "_READY"), "w").close()
    data_vol.commit()
    return meta


@app.function(image=doom_image, gpu="L4", timeout=60 * 60, volumes={"/cache/hf": hf_vol, "/ckpt": ckpt_vol.read_only()})
def play_doom(policy: str = "model", model: str = "all3-3ep/best", episodes: int = 50, tics: int = 4, seed: int = 50_000):
    """Play ``episodes`` of ViZDoom ``basic`` and report reward and kill rate.

    ``policy`` is ``model`` (``model`` = a run under /ckpt/smolvlm or a Hub id), ``expert`` (the scripted labeller),
    ``random``, or ``always_attack``. Seeds are disjoint from the training and val data.
    """
    import random
    from collections import Counter

    import numpy as np
    import vizdoom as vzd
    from PIL import Image

    from laya.games import doom_basic_expert, doom_buttons, doom_question

    g = _doom_game("basic", labels=policy == "expert")
    buttons = doom_buttons(g)
    one_hot = {b: [i == j for j in range(len(buttons))] for i, b in enumerate(buttons)}
    agent = None
    if policy == "model":
        from laya.vlm import VLMAgent

        path = os.path.join(CKPT_ROOT, model)
        agent = VLMAgent(path if os.path.exists(path) else model, device="cuda", dtype="bf16")
    qs = doom_question("basic", buttons)
    rng = random.Random(seed)
    rets, kills, lengths, counts = [], 0, [], Counter()
    t0 = time.time()
    for ep in range(episodes):
        g.set_seed(seed + ep)
        g.new_episode()
        steps = 0
        while not g.is_episode_finished():
            s = g.get_state()
            if policy == "model":
                act = agent.predict({"image": Image.fromarray(s.screen_buffer)}, qs)["answers"]["action"]["choice"]
            elif policy == "expert":
                act = doom_basic_expert(s.labels) or "ATTACK"
            elif policy == "always_attack":
                act = "ATTACK"
            else:
                act = rng.choice(buttons)
            counts[act] += 1
            g.make_action(one_hot[act], tics)
            steps += 1
        rets.append(g.get_total_reward())
        kills += g.get_game_variable(vzd.GameVariable.KILLCOUNT) > 0
        lengths.append(steps)
    g.close()
    out = {"policy": policy if policy != "model" else "model:" + model, "episodes": episodes,
           "mean_reward": float(np.mean(rets)), "std_reward": float(np.std(rets)), "kill_rate": kills / episodes,
           "mean_steps": float(np.mean(lengths)), "actions": dict(counts), "seconds": round(time.time() - t0, 1)}
    print(json.dumps(out))
    return out


@app.local_entrypoint()
def doom_eval(models: str = "all3-3ep/best", episodes: int = 50):
    """modal run modal_app.py::doom_eval --models all3-3ep/best,doom-basic/best  -- baselines + each model, in parallel."""
    calls = [play_doom.spawn(p, "", episodes) for p in ("expert", "random", "always_attack")]
    calls += [play_doom.spawn("model", m, episodes) for m in models.split(",") if m]
    print("%-32s %12s %10s %10s  %s" % ("policy", "mean reward", "kill rate", "steps/ep", "actions"))
    for c in calls:
        r = c.get()
        print("%-32s %12.1f %9.0f%% %10.1f  %s" % (r["policy"], r["mean_reward"], 100 * r["kill_rate"], r["mean_steps"], r["actions"]))
