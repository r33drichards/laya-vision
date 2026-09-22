"""Modal jobs for the VLM-backed decision model (``laya.vlm``): SmolVLM by default, ModernVBERT on request.

    modal run modal_app.py::test                     # pytest on a GPU + latency, both backbones
    modal run modal_app.py::finetune --minutes 18    # short fine-tune + held-out acc / ECE
    modal run modal_app.py::evaluate --run-name <run> # re-evaluate a saved checkpoint
    modal run --detach modal_app.py::finetune_long   # ~3-epoch A100 run with per-epoch eval + best checkpoint
    modal run --detach modal_app.py::finetune_long --backbone ModernVBERT/modernvbert --run-name mvb-3ep
                                                     # the same run on the bidirectional backbone
    modal run modal_app.py::prepare_cauldron          # The Cauldron's closed-form subsets -> /data/vqa/cauldron_<subset>
    modal run modal_app.py::prepare_score             # rubric-scored sets (score questions) -> /data/vqa/score_<name>
    modal run --detach modal_app.py::finetune_long --run-name cauldron-score-2ep --epochs 2 --max-passes 4 \
        --datasets cauldron,score --val-datasets vqa,cauldron,score
                                                     # Cauldron + the score sets (group names expand, see DATASET_GROUPS);
                                                     # add --backbone ModernVBERT/modernvbert for the bidirectional one,
                                                     # or --option-attention block to un-causal SmolVLM's option block
    modal run --detach modal_app.py::split_bench      # SmolVLM2, image splitting off vs 1024 vs 2048 on a 6-set subset:
                                                     # accuracy per set, tokens, L4 latency -> /ckpt/smolvlm2/split-bench/
    modal run modal_app.py::bench_prefix_cache       # predict latency, prefix cache off vs on, L4 bf16
    modal run modal_app.py::try_model --image photo.jpg [--questions q.json] [--text "..."]  # ask a checkpoint about an image
    modal run modal_app.py::publish [--repo user/name] [--run all3-3ep/best]  # push checkpoint + hf_model_card.md to the HF Hub
    modal run modal_app.py::publish --repo thaitea/laya-vision-modernvbert-250m --run modernvbert/cauldron-2ep/best \
        --metrics modernvbert/cauldron-2ep/metrics.json --card hf_model_card_modernvbert.md   # the ModernVBERT one
    modal run modal_app.py::publish_space            # push space/ to the thaitea/laya-vision-demo Space
    modal run modal_app.py::prepare_doom_basic       # auto-labelled ViZDoom "basic" frames -> /data/vqa/doom_basic
    modal run modal_app.py::doom_eval --models all3-3ep/best  # play "basic": expert / random / always-attack / models

Volumes (created out of band; never ``modal deploy`` this app):
    laya-hf-cache     -> /cache/hf   (HF_HOME, shared model weights)
    laya-datasets     -> /data       (read-only; /data/vqa/<name>/{<split>.jsonl, images/, _READY})
    laya-checkpoints  -> /ckpt       (this app writes only under /ckpt/smolvlm/ and /ckpt/modernvbert/)

Data: the fine-tune jobs default to The Cauldron subsets (``CAULDRON_DATASETS``, written by ``prepare_cauldron``);
two layouts exist on the volume: the capped ``cauldron_<subset>`` sets (10,000 usable rows per subset, 5% val)
and the uncapped ``cauldronfull_<subset>`` sets (``CAULDRON_FULL_DATASETS``, every usable row, 1% val, written by
``prepare_cauldron --prefix cauldronfull_ --max-rows 0 --val-pct 1``). The original three VQA sets (``VQA_DATASETS``, official val splits, the README table) stay available with
``--datasets aokvqa,scienceqa,vqav2_yesno`` and as ``--val-datasets`` for a Cauldron-trained model. The ``score``
head has its own sets (``SCORE_DATASETS``, written by ``prepare_score`` from ``laya.rubric``): rubric-graded
responses, aesthetics votes, generated-image ratings and damage levels; add them to ``--datasets`` to train it.
``--datasets`` and ``--val-datasets`` take dataset names and the group names ``vqa``, ``cauldron``, ``cauldronfull``
and ``score`` (``DATASET_GROUPS``).

Run names: ``finetune`` and ``finetune_long`` take ``--backbone`` and write under that backbone's root
(``CKPT_ROOTS``). Everywhere a job takes a saved run (``evaluate``, ``try_model``, ``doom_eval``, ``--init-from``)
the name is relative to /ckpt/smolvlm as before, or to /ckpt, so a ModernVBERT run is ``modernvbert/<run>/best``.
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
SMOLVLM2 = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"
MODERNVBERT = "ModernVBERT/modernvbert"
VQA_DATASETS = ("aokvqa", "scienceqa", "vqav2_yesno")  # the original post-training sets, official val splits
CAULDRON_SUBSETS = ("ai2d", "aokvqa", "iconqa", "intergps", "scienceqa", "tqa", "visual7w", "raven",
                    "figureqa", "hateful_memes", "nlvr2", "vsr", "vqarad",
                    "clevr", "dvqa", "mapqa", "ocrvqa", "vqav2", "chartqa")  # see laya/cauldron.py
CAULDRON_DATASETS = tuple("cauldron_" + s for s in CAULDRON_SUBSETS)
CAULDRON_FULL_DATASETS = tuple("cauldronfull_" + s for s in CAULDRON_SUBSETS)  # uncapped prep, see the docstring
SCORE_SOURCES = ("vlfeedback", "ava", "richhf", "crisismmd")  # see laya/rubric.py
SCORE_DATASETS = tuple("score_" + s for s in SCORE_SOURCES)  # rubric-scored sets for the ``score`` head, written by prepare_score
DATASETS = CAULDRON_DATASETS
CKPT_ROOTS = {BACKBONE: "/ckpt/smolvlm", SMOLVLM2: "/ckpt/smolvlm2", MODERNVBERT: "/ckpt/modernvbert"}
CKPT_ROOT = CKPT_ROOTS[BACKBONE]


def _parse_mix(mix: str) -> dict:
    """``"name=w,name=w"`` -> ``{name: float(w)}``; ``""`` -> ``{}`` (equal sampling)."""
    out = {}
    for kv in mix.split(","):
        if kv.strip():
            k, v = kv.split("=")
            out[k.strip()] = float(v)
    return out


def _ckpt_root(backbone: str) -> str:
    """Where a backbone's runs go: the two known ones by name, anything else by its Hub name."""
    return CKPT_ROOTS.get(backbone, "/ckpt/" + backbone.split("/")[-1].lower())


def _ckpt_file(rel: str) -> str:
    """A file under a run, e.g. ``all3-3ep/metrics.json`` or ``modernvbert/cauldron-2ep/metrics.json``, resolved
    like ``_ckpt_path``: under /ckpt/smolvlm first, then /ckpt."""
    for root in (CKPT_ROOT, "/ckpt"):
        path = os.path.join(root, rel)
        if os.path.exists(path):
            return path
    return os.path.join(CKPT_ROOT, rel)


def _ckpt_path(run_name: str) -> str:
    """A saved run: ``<run>`` under /ckpt/smolvlm (the original layout) or ``<family>/<run>`` under /ckpt."""
    for root in (CKPT_ROOT, "/ckpt"):
        path = os.path.join(root, run_name)
        if os.path.exists(os.path.join(path, "vlm_agent_config.json")):
            return path
    return os.path.join(CKPT_ROOT, run_name)


@app.function(image=image, gpu="L4", timeout=45 * 60, volumes={"/cache/hf": hf_vol})
def test(backbones: str = BACKBONE + "," + SMOLVLM2 + "," + MODERNVBERT):
    """Run tests/test_vlm.py, test_smolvlm2.py and test_modernvbert.py on the GPU, then time predict() in fp32 and
    bf16."""
    import torch

    print("GPU:", torch.cuda.get_device_name(0), "| torch", torch.__version__)
    rc = subprocess.run(
        [sys.executable, "-m", "pytest", "/root/tests/test_vlm.py", "/root/tests/test_smolvlm2.py",
         "/root/tests/test_modernvbert.py", "-v", "-s",
         "-p", "no:cacheprovider", "-W", "ignore"],
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
    for backbone in [b for b in backbones.split(",") if b]:
        for dtype in ("fp32", "bf16"):
            agent = VLMAgent(backbone=backbone, device="cuda", dtype=dtype)
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
                print("latency %-28s %-5s %-11s median %6.1f ms  p90 %6.1f ms" % (backbone, dtype, label, ts[10], ts[18]))
            del agent
    if rc != 0:
        raise SystemExit("pytest failed with exit code %d" % rc)


def _load_split(name: str, split: str, limit):
    from laya.vlm_train import load_jsonl_examples

    return load_jsonl_examples("/data/vqa", name, split, limit=limit)


DATASET_GROUPS = {"vqa": VQA_DATASETS, "cauldron": CAULDRON_DATASETS, "cauldronfull": CAULDRON_FULL_DATASETS,
                  "score": SCORE_DATASETS}


def _expand_datasets(names: str) -> list:
    """``"cauldron,score,aokvqa"`` -> the dataset names, with the group names in ``DATASET_GROUPS`` expanded."""
    out = []
    for name in [d.strip() for d in names.split(",") if d.strip()]:
        for n in DATASET_GROUPS.get(name, (name,)):
            if n not in out:
                out.append(n)
    return out


def _ready(names: str):
    out = []
    for name in _expand_datasets(names):
        if os.path.exists("/data/vqa/%s/_READY" % name):
            out.append(name)
        else:
            print("dataset %s not ready (no _READY); skipping" % name)
    return out


def _load_data(datasets: str, train_split: str, val_split: str, n_calib: int, max_train: int, max_val: int, caps: dict,
               val_datasets: str = ""):
    """(train, calib, val) examples. The LAST ``n_calib`` train records per dataset (file order) are held out of
    training for temperature fitting, as in the SigLIP-projector runs (runs before commit 2651641 held out a seeded
    random 300 instead). ``val_datasets`` scores different sets than were trained on (default: the same)."""
    data_vol.reload()
    train_ex, calib_ex, val_ex = [], [], []
    for name in _ready(datasets):
        tr = _load_split(name, train_split, max_train + n_calib if max_train else 0)
        calib_ex += tr[-n_calib:]
        train_ex += tr[:-n_calib]
        print("dataset %s: %d train, %d calib" % (name, len(tr) - n_calib, min(n_calib, len(tr))))
    for name in _ready(val_datasets or datasets):
        va = _load_split(name, val_split, caps.get(name, max_val) or 0)
        val_ex += va
        print("dataset %s: %d val" % (name, len(va)))
    if not train_ex:
        raise SystemExit("no training data (datasets not ready)")
    if not val_ex:
        raise SystemExit("no validation data (datasets not ready)")
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
    backbone: str = BACKBONE,
    val_datasets: str = "",
    preprocess: str = "processor",
    option_attention: str = "causal",
    w_ce_schedule: str = "const",
    mix: str = "",
    mix_alpha: float = 0.0,
):
    """Short fine-tune on the prepared VQA sets; logs loss and held-out accuracy / ECE, saves to
    ``<backbone root>/<run>`` (/ckpt/smolvlm or /ckpt/modernvbert).

    ``max_train`` / ``max_val`` = 0 means the whole split; otherwise the first N records in file order.
    ``val_caps`` overrides the val cap per dataset, e.g. ``"vqav2_yesno=1000"``. ``preprocess`` is the image
    path recorded in the checkpoint: ``"processor"`` (the Hugging Face processor, what the released model
    used) for photos of mixed sizes; the device-side ``"gpu"`` path stacks raw frames and needs them all the
    same size, so it is for game frames (``modal_atari_train.py``). ``w_ce_schedule``, ``mix`` and ``mix_alpha``:
    see ``finetune_long``.
    """
    import torch

    from laya.vlm import VLMAgent
    from laya.vlm_train import collect_logits, fit_temperatures_from, format_metrics, metrics_from, synthetic_examples, train

    t_start = time.time()
    print("GPU:", torch.cuda.get_device_name(0), "| torch", torch.__version__)
    run_name = run_name or time.strftime("run-%Y%m%d-%H%M%S")
    out_dir = os.path.join(_ckpt_root(backbone), run_name)

    caps = {k: int(v) for k, v in (kv.split("=") for kv in val_caps.split(",") if kv)}
    if synthetic:
        train_ex = [dict(ex, dataset="synthetic") for ex in synthetic_examples(64, seed=0)]
        calib_ex = [dict(ex, dataset="synthetic") for ex in synthetic_examples(16, seed=2)]
        val_ex = [dict(ex, dataset="synthetic") for ex in synthetic_examples(32, seed=1)]
    else:
        train_ex, calib_ex, val_ex = _load_data(datasets, train_split, val_split, n_calib, max_train, max_val, caps, val_datasets)

    agent = VLMAgent(backbone=backbone, device="cuda", preprocess=preprocess, option_attention=option_attention)
    print("backbone %s, readout %s, preprocess %s" % (backbone, agent.model.readout, agent.prep.backend))
    hf_vol.commit()
    model, proc = agent.model, agent.processor
    ev_kw = dict(batch_size=32, num_workers=num_workers)
    small_val = []
    for name in sorted({ex["dataset"] for ex in val_ex}):
        small_val += [ex for ex in val_ex if ex["dataset"] == name][:200]

    mix_weights = _parse_mix(mix)
    log = {"run": run_name, "args": dict(backbone=backbone, datasets=datasets, val_datasets=val_datasets or datasets,
                                         preprocess=preprocess, option_attention=agent.model.option_attention,
                                         w_ce_schedule=w_ce_schedule, mix=mix_weights, mix_alpha=mix_alpha,
                                         minutes=minutes, freeze=freeze, n_last=n_last,
                                         batch_size=batch_size, lr_head=lr_head, lr_backbone=lr_backbone, max_train=max_train,
                                         max_val=max_val, val_caps=caps),
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
        num_workers=num_workers, warmup=100, eval_fn=eval_fn, eval_every=eval_every, w_ce_schedule=w_ce_schedule,
        mix_weights=mix_weights, mix_alpha=mix_alpha,
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
    timeout=300 * 60,
    volumes={"/cache/hf": hf_vol, "/data": data_vol.read_only(), "/ckpt": ckpt_vol},
)
def finetune_long(
    datasets: str = ",".join(DATASETS),
    epochs: float = 3.0,
    max_minutes: float = 90.0,
    batch_size: int = 32,
    lr_head: float = 1e-4,
    lr_backbone: float = 2e-5,
    lr_ref_batch: int = 16,
    warmup_frac: float = 0.03,
    evals_per_epoch: float = 2.0,
    train_eval_n: int = 1000,
    max_passes: float = 0.0,
    n_calib: int = 300,
    num_workers: int = 22,
    run_name: str = "all3-3ep",
    init_from: str = "",
    backbone: str = BACKBONE,
    val_datasets: str = "",
    preprocess: str = "processor",
    option_attention: str = "causal",
    w_ce_schedule: str = "const",
    mix: str = "",
    mix_alpha: float = 0.0,
    split_edge: int = 0,
    max_len: int = 0,
    max_train: int = 0,
    max_val: int = 0,
):
    """Multi-epoch fine-tune (vision tower frozen) with per-epoch train/val tracking and best-checkpoint keeping.

    * ``epochs`` counts samples over the combined train set (datasets are sampled by the mix below, equally by
      default, so small sets repeat more; ``max_passes`` > 0 caps the passes over any one dataset).
    * LRs are given for ``lr_ref_batch`` and scaled by sqrt(batch_size / lr_ref_batch) (the usual rule for Adam).
    * Linear warmup over ``warmup_frac`` of the steps, then cosine decay to 10%; ``max_minutes`` caps wall-clock.
    * Each eval scores the full val splits and the first ``train_eval_n`` train records per dataset (seen in
      training) to expose overfitting. ``last/`` is saved every eval; ``best/`` when mean per-dataset val acc
      improves. The final model is the best one: temperatures are fitted on the calibration holdout, then the full
      val splits are scored raw and calibrated, plus an option-order-bias check on aokvqa.
    * ``init_from`` (a run under /ckpt/smolvlm, e.g. ``all3-3ep/best``, or ``modernvbert/<run>/best``) continues
      from a trained checkpoint instead of a fresh head. Question types absent from the calibration holdout keep
      that checkpoint's temperature rather than being reset to 1.0.
    * ``backbone`` picks the family: SmolVLM (causal, the released model) or ``ModernVBERT/modernvbert``
      (bidirectional, ``[MASK]`` readout). The run is saved under that backbone's root, so the two can share
      a ``run_name``. Everything else, data, objective, schedule and evaluation, is identical, which is what
      makes the two comparable.
    * ``datasets`` defaults to The Cauldron subsets; ``val_datasets`` can score other sets, e.g. the official
      A-OKVQA / ScienceQA / VQAv2 val splits of the README table, on a Cauldron-trained model. With many
      subsets of very different sizes, set ``max_passes`` (the sampler draws subsets equally by default).
    * ``mix`` and ``mix_alpha`` set the sampling mix: dataset ``k`` is drawn with probability proportional to
      ``w_k * n_k ** mix_alpha``, where ``w_k`` comes from ``mix`` (``"name=w,name=w"``, e.g.
      ``"cauldron_ai2d=2,cauldron_figureqa=0.5"``; unlisted datasets get 1) and ``n_k`` is the dataset's train
      size. The defaults (``""``, 0) are the equal sampling of the earlier runs; ``mix_alpha=1`` samples in
      proportion to size, ``0.5`` by square root. The effective probabilities are printed at the start.
    * ``preprocess``: see ``finetune``.
    * ``option_attention="block"`` (SmolVLM only; ``"bidirectional"`` is a deprecated alias) lets the option block attend to itself in both
      directions through a 4D mask (``laya.vlm.option_block_mask``), so every option's readout sees every other
      option, as ModernVBERT's ``[MASK]`` readout does; the state and question stay causal. The pretrained
      backbone never saw this pattern, so it is only meaningful with the backbone unfrozen, as here. The setting
      is saved in the checkpoint and applied at inference. Ignored (rejected) for the ``"mask"`` readout.
    * ``w_ce_schedule="anneal"`` holds the soft cross-entropy weight for the first 30% of training and decays it
      to 0 by 80%, so the run ends on the proper scoring rule alone, which keeps the raw model calibrated
      (``laya.vlm_train.train``); ``"const"`` is the released recipe.
    * ``split_edge`` > 0 turns on the processor's image splitting (``laya.preprocess``): each image is resized to
      that longest edge and cut into 512 tiles plus a global view, up to 17 views at 2048 against 1 without.
      Needs ``preprocess="processor"``. ``max_len`` (0: ``laya.vlm.default_max_len``, 1024 without splitting)
      is the sequence cap; both are saved in the checkpoint. Neither applies with ``init_from``, which keeps the
      checkpoint's.
    * ``max_train`` / ``max_val`` > 0 keep the first records per dataset (file order) for a quicker run on a
      subset; ``max_train`` does not count the ``n_calib`` holdout.
    """
    import math

    import torch

    from laya.common import QTYPES
    from laya.vlm import VLMAgent, _permutations
    from laya.vlm_train import (collect_logits, cyclic_orders, fit_temperatures_from, format_metrics, metrics_from,
                                mix_probabilities, train)

    t_start = time.time()
    print("GPU:", torch.cuda.get_device_name(0), "| torch", torch.__version__)
    out_dir = os.path.join(_ckpt_root(backbone), run_name)
    train_ex, calib_ex, val_ex = _load_data(datasets, "train", "val", n_calib, max_train, max_val, {}, val_datasets)
    names = sorted({ex["dataset"] for ex in train_ex})
    val_names = sorted({ex["dataset"] for ex in val_ex})
    train_eval = []
    for name in names:
        train_eval += [ex for ex in train_ex if ex["dataset"] == name][:train_eval_n]

    scale = math.sqrt(batch_size / lr_ref_batch)
    lr_h, lr_b = lr_head * scale, lr_backbone * scale
    steps = int(math.ceil(epochs * len(train_ex) / batch_size))
    eval_every = max(1, int(round(steps / (epochs * evals_per_epoch))))
    warmup = max(1, int(warmup_frac * steps))
    sizes = {n: sum(ex["dataset"] == n for ex in train_ex) for n in names}
    mix_weights = _parse_mix(mix)
    probs = mix_probabilities({n: range(sizes[n]) for n in names}, mix_weights, mix_alpha)
    print("plan: %d steps x batch %d (%.1f epochs of %d), warmup %d, eval every %d, lr head %.2e backbone %.2e"
          % (steps, batch_size, epochs, len(train_ex), warmup, eval_every, lr_h, lr_b))
    print("expected passes per dataset with the sampling mix (before max_passes=%s): %s"
          % (max_passes or None, {n: round(epochs * len(train_ex) * probs[n] / sizes[n], 2) for n in names}))

    if init_from:
        agent = VLMAgent(_ckpt_path(init_from), device="cuda")
        print("initialised from %s (temperatures %s)" % (init_from, [round(t, 3) for t in agent.temperature]))
    else:
        agent = VLMAgent(backbone=backbone, device="cuda", preprocess=preprocess, option_attention=option_attention,
                         image_split_edge=split_edge, **({"max_len": max_len} if max_len else {}))
    print("backbone %s, readout %s, preprocess %s, split_edge %d, max_len %d" % (
        agent.cfg["backbone"], agent.model.readout, agent.prep.backend, agent.prep.split_edge, agent.cfg["max_len"]))
    init_temps = list(agent.temperature)
    hf_vol.commit()
    model, proc = agent.model, agent.processor
    import shutil

    tr_workers, tr_prefetch = _loader_fit(agent.prep, batch_size, num_workers, 4)
    ev_workers, _ = _loader_fit(agent.prep, 64, num_workers, 2, min_prefetch=2)  # collect_logits: prefetch 2
    print("loader: /dev/shm %.1f GB; train %d workers x prefetch %d, eval %d workers" % (
        shutil.disk_usage("/dev/shm").total / 1e9, tr_workers, tr_prefetch, ev_workers), flush=True)
    ev_kw = dict(batch_size=64, num_workers=ev_workers)
    log = {"run": run_name, "args": dict(backbone=agent.cfg["backbone"], readout=agent.model.readout, datasets=datasets,
                                         val_datasets=val_datasets or datasets, preprocess=agent.prep.backend,
                                         split_edge=agent.prep.split_edge, max_len=agent.cfg["max_len"],
                                         max_train=max_train, max_val=max_val,
                                         option_attention=agent.model.option_attention,
                                         w_ce_schedule=w_ce_schedule, mix=mix_weights, mix_alpha=mix_alpha,
                                         epochs=epochs, max_minutes=max_minutes, batch_size=batch_size, lr_head=lr_h,
                                         lr_backbone=lr_b, warmup=warmup, steps=steps, eval_every=eval_every,
                                         max_passes=max_passes, n_calib=n_calib, train_eval_n=train_eval_n),
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
        score = sum(val_m[n]["acc"] for n in val_names) / len(val_names)
        row = {"step": step, "epoch": round(step * batch_size / len(train_ex), 2), "mean_val_acc": score,
               "val": val_m, "train": tr_m}
        log["evals"].append(row)
        print("[eval step %d, epoch %.2f] mean val acc %.4f | val: %s | train (seen): %s" % (
            step, row["epoch"], score,
            " | ".join("%s %.3f ece %.3f nll %.3f" % (n, val_m[n]["acc"], val_m[n]["ece"], val_m[n]["nll"]) for n in val_names),
            " | ".join("%s %.3f" % (n, tr_m[n]["acc"]) for n in names)), flush=True)
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
        device="cuda", log_every=100, max_minutes=max_minutes, num_workers=tr_workers,
        prefetch_factor=tr_prefetch, warmup=warmup,
        eval_fn=eval_fn, eval_every=eval_every, max_passes=max_passes or None, stats=stats,
        w_ce_schedule=w_ce_schedule, mix_weights=mix_weights, mix_alpha=mix_alpha,
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
    return {k: log[k] for k in ("run", "args", "best_step", "best_mean_val_acc", "temperature", "final", "train_stats")}


@app.function(
    image=image,
    gpu=["A10G", "L4", "A100"],  # any of these: an eval should not queue on one GPU type's capacity
    cpu=16,
    memory=32768,
    timeout=30 * 60,
    volumes={"/cache/hf": hf_vol, "/data": data_vol.read_only(), "/ckpt": ckpt_vol.read_only()},
)
def evaluate(run_name: str, datasets: str = ",".join(VQA_DATASETS + CAULDRON_DATASETS + SCORE_DATASETS), val_split: str = "val",
             max_val: int = 0):
    """Evaluate a saved checkpoint (``<run>`` under /ckpt/smolvlm, or ``modernvbert/<run>``) on the val splits,
    raw and with its temperatures. Defaults to every prepared set (the official VQA splits and the Cauldron
    holdouts); sets that are not prepared are skipped."""
    import torch

    from laya.vlm import VLMAgent
    from laya.vlm_train import collect_logits, format_metrics, metrics_from

    print("GPU:", torch.cuda.get_device_name(0))
    data_vol.reload()
    val_ex = []
    for name in _ready(datasets):
        va = _load_split(name, val_split, max_val or 0)
        print("dataset %s: %d val" % (name, len(va)))
        val_ex += va
    agent = VLMAgent(_ckpt_path(run_name), device="cuda")
    print("backbone %s, readout %s" % (agent.cfg["backbone"], agent.model.readout))
    records = collect_logits(agent.model, agent.processor, val_ex, batch_size=32, num_workers=14)
    raw, cal = metrics_from(records), metrics_from(records, agent.temperature)
    print("temperatures (choice, score, noul):", [round(t, 3) for t in agent.temperature])
    print("[val, T=1]        " + format_metrics(raw))
    print("[val, calibrated] " + format_metrics(cal))
    return {"val_raw": raw, "val_calibrated": cal}


def _public_question(q: dict) -> dict:
    """An internal ``{"t", "ins", "crit"}`` question back in ``predict``'s input format."""
    crit = q["crit"]
    if q["t"] == "choice":
        crit = list(crit)
    return {"type": q["t"], "instructions": q["ins"], "criteria": crit}


@app.function(image=image, gpu="L4", cpu=4, memory=16384, timeout=60 * 60,
              volumes={"/cache/hf": hf_vol, "/data": data_vol.read_only(), "/ckpt": ckpt_vol.read_only()})
def bench_latency(run_name: str, datasets: str = "", n: int = 200, dtype: str = "bf16", seed: int = 0):
    """Time ``predict`` on real val images, one question each, as a user would call it: the checkpoint's own
    preprocessing on the CPU (the processor's resize and, with ``image_split_edge``, its tiling), then the forward
    pass, bf16 on an L4 like the README's latency column. Images are decoded before the clock starts. ``n``
    examples are drawn evenly from ``datasets`` (the run's own val sets when empty)."""
    import random

    import numpy as np
    import torch
    from PIL import Image

    from laya.vlm import VLMAgent, vlm_prefix

    path = _ckpt_path(run_name)
    agent = VLMAgent(path, device="cuda", dtype=dtype)
    if not datasets:
        with open(os.path.join(os.path.dirname(path.rstrip("/")), "metrics.json")) as f:
            datasets = json.load(f)["args"]["val_datasets"]
    names = _ready(datasets)
    rng = random.Random(seed)
    per = max(1, n // max(1, len(names)))
    picked = []
    for name in names:
        exs = [ex for ex in _load_split(name, "val", 0) if isinstance(ex["state"], dict) and ex["state"].get("image")]
        picked += rng.sample(exs, min(per, len(exs)))
    cases = []
    for ex in picked:
        state = dict(ex["state"])
        with Image.open(state["image"]) as im:
            state["image"] = im.convert("RGB")
        cases.append((state, {"q": _public_question(ex["q"])}))
    for state, qs in cases[:5]:  # warm-up: kernels, allocator
        agent.predict(state, qs)
    ms, tokens, views = [], [], []
    for state, qs in cases:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        res = agent.predict(state, qs)
        torch.cuda.synchronize()
        ms.append((time.perf_counter() - t0) * 1000)
        tokens.append(res["usage"]["input_tokens"])
    for state, _ in cases[:50]:
        views.append(vlm_prefix(agent.processor, [state["image"]], agent.prep)["n_images"])
    out = {"run": run_name, "gpu": torch.cuda.get_device_name(0), "dtype": dtype, "n": len(cases),
           "split_edge": agent.prep.split_edge, "max_len": agent.cfg["max_len"],
           "median_ms": float(np.median(ms)), "p90_ms": float(np.percentile(ms, 90)),
           "mean_input_tokens": float(np.mean(tokens)), "mean_views_per_image": float(np.mean(views))}
    print(json.dumps(out))
    return out


@app.function(image=image, gpu="L4", timeout=25 * 60, volumes={"/cache/hf": hf_vol, "/ckpt": ckpt_vol.read_only()})
def bench_prefix_cache(run_name: str = "cauldron-score-2ep-bidir-full/best", dtype: str = "bf16", n: int = 20):
    """``predict`` latency with the prefix cache off (every row runs the whole sequence), forced on (image + state
    prefilled once, only the question/option suffixes per row) and automatic (``prefix_cache=None``), 3 questions
    on one image, ``n_permutations`` 1, 4 and 8, with a short and a long state text. Also reports the largest
    probability gap between the full and the cached path. Nothing is written."""
    import numpy as np
    import torch
    from PIL import Image

    from laya.vlm import VLMAgent

    agent = VLMAgent(_ckpt_path(run_name), device="cuda", dtype=dtype)
    img = Image.new("RGB", (640, 480), (255, 255, 255))
    img.paste(Image.new("RGB", (200, 160), (220, 20, 20)), (120, 100))
    states = {"image + short state": {"image": img, "context": "A product photo from the returns desk. " * 3},
              "image + long state": {"image": img, "context": "A product photo from the returns desk. " * 90}}
    qs = {"color": {"type": "choice", "instructions": "What color is the box?", "criteria": ["red", "blue", "green", "white"]},
          "size": {"type": "score", "instructions": "How much of the photo does the box fill?",
                   "criteria": ["tiny", "a quarter", "about half", "almost all"]},
          "damaged": {"type": "noul", "instructions": "Does the box look damaged?"}}
    rows = []
    for label, state in states.items():
        for perms in (1, 4, 8):
            a = agent.predict(state, qs, n_permutations=perms, prefix_cache=False)
            b = agent.predict(state, qs, n_permutations=perms, prefix_cache=True)
            gap = 0.0
            for qid in qs:
                x, y = a["answers"][qid], b["answers"][qid]
                pa = list(x.get("probabilities", {"p": x.get("noul")}).values())
                pb = list(y.get("probabilities", {"p": y.get("noul")}).values())
                gap = max([gap] + [abs(u - v) for u, v in zip(pa, pb)])
            med = {}
            for cache in (False, True, None):
                for _ in range(3):
                    agent.predict(state, qs, n_permutations=perms, prefix_cache=cache)
                ts = []
                for _ in range(n):
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    agent.predict(state, qs, n_permutations=perms, prefix_cache=cache)
                    torch.cuda.synchronize()
                    ts.append((time.perf_counter() - t0) * 1000)
                med[cache] = float(np.median(ts))
            r = {"state": label, "n_permutations": perms, "input_tokens": a["usage"]["input_tokens"],
                 "full_ms": med[False], "cached_ms": med[True], "auto_ms": med[None],
                 "speedup_cached": med[False] / med[True], "max_prob_gap": gap}
            print(json.dumps(r))
            rows.append(r)
    return {"run": run_name, "gpu": torch.cuda.get_device_name(0), "dtype": dtype,
            "option_attention": agent.model.option_attention, "rows": rows}


SPLIT_BENCH_DATASETS = ("cauldron_ai2d", "cauldron_aokvqa", "cauldron_tqa", "cauldron_ocrvqa", "cauldron_mapqa",
                        "cauldron_vqav2")  # diagrams, photos, textbook figures, book-cover text, maps, photos


def _loader_fit(prep, batch_size: int, num_workers: int, prefetch: int, min_prefetch: int = 1) -> tuple:
    """(workers, prefetch) so the batches a DataLoader keeps in flight fit in half of /dev/shm. Each one holds
    float32 pixels, up to ``ceil(split_edge / image_size)^2 + 1`` tiles per image with splitting, so 22 workers x
    prefetch 4 that fit unsplit run the split settings out of shared memory (and a lost batch hangs the loop)."""
    import shutil

    side = -(-prep.split_edge // prep.image_size) if prep.split_edge else 0
    per_batch = batch_size * (side * side + 1) * 3 * prep.image_size ** 2 * 4
    fit = int(shutil.disk_usage("/dev/shm").total * 0.5 // per_batch)
    while prefetch > min_prefetch and num_workers * prefetch > fit:
        prefetch //= 2
    return max(2, min(num_workers, fit // prefetch)), prefetch


def _split_label(edge: int) -> str:
    return "split%d" % edge if edge else "nosplit"


def _split_table(rows: list, names: list) -> str:
    """Markdown summary of ``split_bench``: one row per setting."""
    short = [n.replace("cauldron_", "") for n in names]
    head = ("| setting | views / image | tokens / question | " + " | ".join(short) +
            " | mean acc | mean ECE (cal.) | L4 bf16 median | p90 | train samples/s |")
    lines = [head, "|" + "---|" * (head.count("|") - 1)]
    for r in rows:
        cal = r["final"]["val_calibrated"]
        accs = [cal[n]["acc"] for n in names if n in cal]
        eces = [cal[n]["ece"] for n in names if n in cal]
        lat, ts = r["latency"], r["train_stats"]
        lines.append("| %s | %.1f | %.0f | %s | %.1f%% | %.3f | %.0f ms | %.0f ms | %.0f |" % (
            r["label"], lat["mean_views_per_image"], lat["mean_input_tokens"],
            " | ".join("%.1f%%" % (100 * cal[n]["acc"]) if n in cal else "-" for n in names),
            100 * sum(accs) / len(accs), sum(eces) / len(eces), lat["median_ms"], lat["p90_ms"],
            ts["steps_per_s"] * r["args"]["batch_size"]))
    return "\n".join(lines) + "\n"


@app.function(image=image, cpu=1, memory=4096, timeout=24 * 60 * 60, volumes={"/ckpt": ckpt_vol})
def split_bench(
    bench: str = "split-bench",
    split_edges: str = "0,1024,2048",
    backbone: str = SMOLVLM2,
    datasets: str = ",".join(SPLIT_BENCH_DATASETS),
    epochs: float = 2.0,
    max_train: int = 4000,
    batch_size: int = 32,
    max_minutes: float = 240.0,
    option_attention: str = "block",
    train_gpu: str = "A100-80GB",
    latency_n: int = 300,
):
    """Image splitting off vs on, with everything else equal: one ``finetune_long`` per ``split_edges`` value
    (0 = off, the released recipe; 1024 = up to 5 views per image; 2048 = the processor default, up to 17), all
    on ``backbone`` from a fresh head, the same subset, schedule, batch and GPU type, run in parallel. Then
    ``bench_latency`` on each best checkpoint. Writes ``results.json`` and ``results.md`` (accuracy per set,
    calibrated ECE, tokens and views per question, L4 latency, training throughput) to
    ``<backbone root>/<bench>/`` and returns the markdown table.

    ``max_minutes`` caps training wall-clock per run; the table's throughput column shows whether a slow setting
    finished its epochs (compare ``train_stats.steps`` in results.json with ``args.steps``).
    """
    edges = [int(e) for e in split_edges.split(",") if e.strip()]
    root = _ckpt_root(backbone)
    ft = finetune_long.with_options(gpu=train_gpu)
    calls = {}
    for e in edges:
        run = "%s/%s" % (bench, _split_label(e))
        print("spawning %s (split_edge %d)" % (run, e), flush=True)
        calls[e] = ft.spawn(datasets=datasets, epochs=epochs, max_minutes=max_minutes, batch_size=batch_size,
                            evals_per_epoch=1.0, train_eval_n=300, max_train=max_train, run_name=run,
                            backbone=backbone, option_attention=option_attention, split_edge=e)
    rows = []
    for e in edges:
        r = calls[e].get()
        print("%s done: best mean val acc %.4f" % (_split_label(e), r["best_mean_val_acc"]), flush=True)
        rows.append(dict(r, label=_split_label(e), split_edge=e))
    lat_calls = [bench_latency.spawn(os.path.relpath(os.path.join(root, bench, r["label"], "best"), "/ckpt"),
                                     datasets=datasets, n=latency_n) for r in rows]
    for r, c in zip(rows, lat_calls):
        r["latency"] = c.get()
    names = [n for n in _expand_datasets(datasets) if any(n in r["final"]["val_calibrated"] for r in rows)]
    table = _split_table(rows, names)
    out_dir = os.path.join(root, bench)
    ckpt_vol.reload()
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(rows, f, indent=2)
    with open(os.path.join(out_dir, "results.md"), "w") as f:
        f.write(table)
    ckpt_vol.commit()
    print(table)
    return table


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
    agent = VLMAgent(_ckpt_path(run_name), device="cuda")
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

    src = _ckpt_path(run_name)
    if not os.path.exists(os.path.join(src, "vlm_agent_config.json")):
        raise SystemExit("no checkpoint at %s" % src)
    stage = tempfile.mkdtemp()
    shutil.copytree(src, stage, dirs_exist_ok=True)
    with open(os.path.join(stage, "README.md"), "w") as f:
        f.write(model_card)
    if metrics_path:
        shutil.copy(_ckpt_file(metrics_path), os.path.join(stage, "training_metrics.json"))
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


@app.function(image=image, timeout=15 * 60, secrets=[modal.Secret.from_name("huggingface-thaitea")])
def push_space(repo_id: str, files: dict):
    """Upload ``files`` (``{relative path: bytes}``, the contents of ``space/``) to a Hugging Face Space."""
    import tempfile

    from huggingface_hub import HfApi

    stage = tempfile.mkdtemp()
    for rel, data in files.items():
        path = os.path.join(stage, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)
        print("%10d  %s" % (len(data), rel))
    api = HfApi()
    api.create_repo(repo_id, repo_type="space", space_sdk="gradio", exist_ok=True)
    info = api.upload_folder(folder_path=stage, repo_id=repo_id, repo_type="space", commit_message="Update Space from space/")
    print("uploaded:", info.commit_url)
    return info.commit_url


@app.local_entrypoint()
def publish_space(repo: str = "thaitea/laya-vision-demo", folder: str = "space"):
    """modal run modal_app.py::publish_space  -- push the demo's source (space/) to its Hugging Face Space."""
    files = {}
    for root, _, names in os.walk(folder):
        for name in names:
            path = os.path.join(root, name)
            with open(path, "rb") as f:
                files[os.path.relpath(path, folder)] = f.read()
    print(push_space.remote(repo, files))


@app.local_entrypoint()
def publish(repo: str = "thaitea/laya-vision-smolvlm-256m", run: str = "all3-3ep/best",
            metrics: str = "all3-3ep/metrics.json", card: str = "hf_model_card.md", private: bool = False):
    """modal run modal_app.py::publish  -- push a checkpoint + hf_model_card.md to the Hugging Face Hub."""
    with open(card) as f:
        text = f.read()
    print(push_to_hub.remote(repo, run, text, metrics_path=metrics, private=private))


# ---------------------------------------------------------------------------------------------------------
# The Cauldron: closed-form subsets -> prepared datasets
# ---------------------------------------------------------------------------------------------------------


@app.function(image=image, cpu=4, memory=16384, timeout=6 * 60 * 60, volumes={"/cache/hf": hf_vol, "/data": data_vol},
              secrets=[modal.Secret.from_name("huggingface-thaitea")])
def prepare_cauldron_subset(subset: str, max_rows: int = 10000, max_texts: int = 4, val_pct: float = 5.0,
                            max_side: int = 1024, seed: int = 0, prefix: str = "cauldron_"):
    """Stream one Cauldron subset and write /data/vqa/<prefix><subset>/{train,val}.jsonl + images/.

    Rows are taken in stream order until ``max_rows`` *usable* rows (at least one closed-form turn, see
    ``laya.cauldron``) are written (``max_rows=0``: no cap, the whole subset); each keeps at most ``max_texts`` turns. A seeded ``val_pct`` percent of rows
    go to ``val`` (by row, so an image never sits in both splits). Images are saved as JPEG with the longest side
    at most ``max_side`` (the model sees 512-pixel tiles). The Cauldron is train-only upstream, so its
    ``aokvqa`` / ``scienceqa`` / ``vqav2`` rows are the official train splits and do not overlap the official val
    splits in ``VQA_DATASETS``.
    """
    import random
    import shutil
    from collections import Counter

    from datasets import load_dataset

    from laya.cauldron import cauldron_records

    name = prefix + subset
    final_dir = os.path.join("/data/vqa", name)
    tmp_dir = final_dir + ".tmp"
    shutil.rmtree(tmp_dir, ignore_errors=True)
    os.makedirs(os.path.join(tmp_dir, "images"))
    rng = random.Random(seed)
    ds = load_dataset("HuggingFaceM4/the_cauldron", subset, split="train", streaming=True)
    t0 = time.time()
    n_rows, n_seen, counts = 0, 0, {"train": Counter(), "val": Counter()}
    files = {split: open(os.path.join(tmp_dir, split + ".jsonl"), "w") for split in ("train", "val")}
    for i, row in enumerate(ds):
        if max_rows and n_rows >= max_rows:
            break
        n_seen += 1
        n_img = len(row["images"]) if isinstance(row["images"], list) else 1  # parse before decoding any pixels
        paths = ["images/%s-%d-%d.jpg" % (subset, i, j) for j in range(n_img)]
        recs = cauldron_records(row["texts"], paths, "%s-%d" % (subset, i), max_texts=max_texts, rng=rng)
        if not recs:
            continue
        split = "val" if rng.random() < val_pct / 100 else "train"
        images = row["images"] if isinstance(row["images"], list) else [row["images"]]
        for im, path in zip(images, paths):
            im = im.convert("RGB")
            im.thumbnail((max_side, max_side))
            im.save(os.path.join(tmp_dir, path), quality=90)
        for rec in recs:
            files[split].write(json.dumps(rec, ensure_ascii=False) + "\n")
            counts[split][rec["question"]["type"]] += 1
        n_rows += 1
        if n_rows % 1000 == 0:
            print("%s: %d rows (%d seen) in %.1f min" % (subset, n_rows, n_seen, (time.time() - t0) / 60), flush=True)
    for f in files.values():
        f.close()
    meta = {"source": "HuggingFaceM4/the_cauldron", "subset": subset, "rows": n_rows, "rows_seen": n_seen,
            "max_rows": max_rows, "max_texts": max_texts, "val_pct": val_pct, "max_side": max_side, "seed": seed,
            "records": {k: dict(v) for k, v in counts.items()}, "minutes": round((time.time() - t0) / 60, 1)}
    with open(os.path.join(tmp_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    shutil.rmtree(final_dir, ignore_errors=True)
    os.rename(tmp_dir, final_dir)
    open(os.path.join(final_dir, "_READY"), "w").close()
    data_vol.commit()
    print("%s: %d usable rows of %d seen, records %s, %.1f min" % (subset, n_rows, n_seen, meta["records"], meta["minutes"]))
    return meta


@app.local_entrypoint()
def prepare_cauldron(subsets: str = ",".join(CAULDRON_SUBSETS), max_rows: int = 10000, max_texts: int = 4,
                     val_pct: float = 5.0, max_side: int = 1024, prefix: str = "cauldron_"):
    """modal run modal_app.py::prepare_cauldron [--subsets ai2d,aokvqa] -- one container per subset, in parallel.

    ``--max-rows 0`` takes every usable row of each subset; pair it with ``--prefix cauldronfull_`` (and
    ``--val-pct 1``) to write the uncapped layout next to the capped ``cauldron_<subset>`` sets instead of over them.
    """
    names = [s for s in subsets.split(",") if s]
    kw = dict(max_rows=max_rows, max_texts=max_texts, val_pct=val_pct, max_side=max_side, prefix=prefix)
    print("%-16s %8s %8s  %s" % ("subset", "rows", "seen", "records"))
    for meta in prepare_cauldron_subset.map(names, kwargs=kw, order_outputs=True, return_exceptions=True):
        if isinstance(meta, Exception):
            print("FAILED:", repr(meta)[:300])
            continue
        print("%-16s %8d %8d  %s" % (meta["subset"], meta["rows"], meta["rows_seen"], meta["records"]))


# ---------------------------------------------------------------------------------------------------------
# Rubric-scored datasets -> prepared ``score`` datasets (laya/rubric.py)
# ---------------------------------------------------------------------------------------------------------


def _score_source(name: str, split: str, rng, max_texts: int, max_chars: int):
    """Yield ``(row_id, image, records)`` for one source and split (``"train"`` / ``"val"``); ``image`` is a PIL
    image or a Hub file path to download. Sources with an upstream validation split use it for ``val``."""
    from datasets import load_dataset

    from laya.rubric import SOURCES, ava_record, crisismmd_record, richhf_records, vlfeedback_records

    repo = SOURCES[name]
    if name == "vlfeedback":
        if split == "val":
            return  # no upstream split: the job holds out val_pct of the train rows
        for i, row in enumerate(load_dataset(repo, split="train", streaming=True)):
            rid = "vlf-%s" % (row.get("id") or i)
            yield rid, row["image"], vlfeedback_records(row, rid, rng, max_texts=max_texts, max_chars=max_chars)
    elif name == "ava":
        for i, row in enumerate(load_dataset(repo, split="validation" if split == "val" else "train", streaming=True)):
            rid = "ava-%s" % (row.get("image_id") or i)
            rec = ava_record(row, rid, rng)
            yield rid, row["image"], [rec] if rec else []
    elif name == "richhf":
        for i, row in enumerate(load_dataset(repo, split="validation" if split == "val" else "train", streaming=True)):
            rid = "richhf-%d" % i
            yield rid, row["image"], richhf_records(row, rid, rng, max_texts=max_texts)
    elif name == "crisismmd":
        for i, row in enumerate(load_dataset(repo, "damage", split="dev" if split == "val" else "train")):
            rid = "crisis-%s" % (row.get("image_id") or i)
            rec = crisismmd_record(row, rid, rng)
            yield rid, row.get("image") or row["image_path"], [rec] if rec else []
    else:
        raise ValueError("unknown score source %r (one of %s)" % (name, sorted(SOURCES)))


@app.function(image=image, cpu=4, memory=16384, timeout=6 * 60 * 60, volumes={"/cache/hf": hf_vol, "/data": data_vol},
              secrets=[modal.Secret.from_name("huggingface-thaitea")])
def prepare_score_dataset(name: str, max_rows: int = 0, max_texts: int = 2, val_pct: float = 5.0, max_val: int = 1000,
                          max_side: int = 1024, seed: int = 0, balance: float = 3.0, max_chars: int = 1200,
                          prefix: str = "score_"):
    """Stream one rubric-scored source (``laya.rubric.SOURCES``) and write /data/vqa/<prefix><name>/{train,val}.jsonl + images/.

    Rows are taken in stream order until ``max_rows`` usable rows (0: all); each keeps at most ``max_texts``
    records (sampled). Sources with an upstream validation split use it for ``val`` (capped at ``max_val``
    rows); the others hold out a seeded ``val_pct`` percent of rows by row, so an image never sits in both
    splits. After streaming, the train split is level-balanced (``laya.rubric.balance_levels``: no level above
    ``balance`` x the median level count) and images no record points at are deleted. Images are JPEG with the
    longest side at most ``max_side``. Records carry ``"target"`` (AVA's vote histogram) where the source has it.
    """
    import random
    import shutil
    from collections import Counter

    from huggingface_hub import hf_hub_download
    from PIL import Image

    from laya.rubric import SOURCES, balance_levels, level_counts

    ds_name = prefix + name
    final_dir = os.path.join("/data/vqa", ds_name)
    tmp_dir = final_dir + ".tmp"
    shutil.rmtree(tmp_dir, ignore_errors=True)
    os.makedirs(os.path.join(tmp_dir, "images"))
    rng = random.Random(seed)
    t0 = time.time()
    recs = {"train": [], "val": []}
    n_rows = {"train": 0, "val": 0}
    for split in ("train", "val"):
        for rid, image, rows in _score_source(name, split, rng, max_texts, max_chars):
            if not rows:
                continue
            if split == "train" and max_rows and n_rows["train"] >= max_rows:
                break
            if split == "val" and max_val and n_rows["val"] >= max_val:
                break
            target = split
            if split == "train" and name in ("vlfeedback",):
                target = "val" if rng.random() < val_pct / 100 else "train"
                if target == "val" and max_val and n_rows["val"] >= max_val:
                    target = "train"
            path = "images/%s.jpg" % rid
            if isinstance(image, str):
                image = Image.open(hf_hub_download(SOURCES[name], image, repo_type="dataset"))
            im = image.convert("RGB")
            im.thumbnail((max_side, max_side))
            im.save(os.path.join(tmp_dir, path), quality=90)
            for rec in rows:
                rec["image"] = path
            recs[target] += rows
            n_rows[target] += 1
            if sum(n_rows.values()) % 1000 == 0:
                print("%s: %s rows in %.1f min" % (name, n_rows, (time.time() - t0) / 60), flush=True)
    before = level_counts(recs["train"])
    recs["train"] = balance_levels(recs["train"], balance, rng)
    used = {r["image"] for split in recs for r in recs[split]}
    dropped = 0
    for fn in os.listdir(os.path.join(tmp_dir, "images")):
        if "images/" + fn not in used:
            os.remove(os.path.join(tmp_dir, "images", fn))
            dropped += 1
    for split in ("train", "val"):
        with open(os.path.join(tmp_dir, split + ".jsonl"), "w") as f:
            for rec in recs[split]:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    meta = {"source": SOURCES[name], "name": name, "rows": n_rows, "max_rows": max_rows, "max_texts": max_texts,
            "val_pct": val_pct, "max_val": max_val, "max_side": max_side, "seed": seed, "balance": balance,
            "max_chars": max_chars, "records": {s: len(recs[s]) for s in recs},
            "levels": {"train_before_balance": dict(sorted(before.items())), "train": dict(sorted(level_counts(recs["train"]).items())),
                       "val": dict(sorted(level_counts(recs["val"]).items()))},
            "images_dropped_by_balance": dropped, "minutes": round((time.time() - t0) / 60, 1)}
    with open(os.path.join(tmp_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    shutil.rmtree(final_dir, ignore_errors=True)
    os.rename(tmp_dir, final_dir)
    open(os.path.join(final_dir, "_READY"), "w").close()
    data_vol.commit()
    print("%s: rows %s, records %s, levels %s, %.1f min" % (name, n_rows, meta["records"], meta["levels"], meta["minutes"]))
    return meta


@app.local_entrypoint()
def prepare_score(names: str = ",".join(SCORE_SOURCES), max_rows: int = 0, max_texts: int = 2, val_pct: float = 5.0,
                  max_val: int = 1000, max_side: int = 1024, balance: float = 3.0, max_chars: int = 1200,
                  prefix: str = "score_"):
    """modal run modal_app.py::prepare_score [--names ava,crisismmd] -- one container per source, in parallel."""
    kw = dict(max_rows=max_rows, max_texts=max_texts, val_pct=val_pct, max_val=max_val, max_side=max_side,
              balance=balance, max_chars=max_chars, prefix=prefix)
    print("%-12s %-28s %-24s  %s" % ("source", "rows", "records", "train levels"))
    for meta in prepare_score_dataset.map([n for n in names.split(",") if n], kwargs=kw, order_outputs=True, return_exceptions=True):
        if isinstance(meta, Exception):
            print("FAILED:", repr(meta)[:300])
            continue
        print("%-12s %-28s %-24s  %s" % (meta["name"], meta["rows"], meta["records"], meta["levels"]["train"]))


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

        path = _ckpt_path(model)
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
