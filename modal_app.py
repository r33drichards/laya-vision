"""Modal jobs for the VLM-backed decision model (``laya.vlm``): SmolVLM by default, ModernVBERT on request.

    modal run modal_app.py::test                     # pytest on a GPU + latency, both backbones
    modal run modal_app.py::finetune --minutes 18    # short fine-tune + held-out acc / ECE
    modal run modal_app.py::evaluate --run-name <run> # re-evaluate a saved checkpoint
    modal run modal_app.py::evidence --run cauldron-score-2ep-bidir-full/best
                                                     # row-level val predictions -> results/raw/ + SHA256SUMS
                                                     # (checked by benchmarks/verify_published.py)
    modal run --detach modal_app.py::robustness_eval [--run <run>] [--n 300]  # perturbation robustness,
                                                     # see laya/robustness.py -> results/robustness/
    modal run --detach modal_app.py::finetune_long   # ~3-epoch A100 run with per-epoch eval + best checkpoint
    modal run --detach modal_app.py::finetune_long --backbone ModernVBERT/modernvbert --run-name mvb-3ep
                                                     # the same run on the bidirectional backbone
    modal run modal_app.py::prepare_cauldron          # The Cauldron's closed-form subsets -> /data/vqa/cauldron_<subset>
    modal run modal_app.py::prepare_score             # rubric-scored sets (score questions) -> /data/vqa/score_<name>
    modal run modal_app.py::prepare_eval              # held-out eval sets (KonIQ, EvalMuse, CIFAR-10H, FER+, VizWiz,
                                                     # POPE) -> /data/vqa/eval_<name>; evaluate scores them by default
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
    modal run modal_app.py::maze_eval --models <run>/best     # Maze at 4x4 / 6x6 / 8x8 cells: BFS expert, random, models
    modal run modal_app.py::snake_eval --models <run>/best    # Snake on a 10x10 board: greedy expert, random, models
    modal run modal_app.py::control_eval --models <run>/best  # CartPole / Acrobot / MountainCar / LunarLander: expert, random, models
    modal run modal_app.py::full_eval --model <run>/best  # EVERYTHING on one checkpoint, in parallel: evaluate over
                                                     # vqa,cauldron,score,eval + games suite + latency; one JSON in
                                                     # eval-results/ and <run>/evals/ on the checkpoint volume
    modal run modal_app.py::games_eval --model <run>/best --out games.json
                                                     # the games suite on one checkpoint: Atari Freeway / Breakout /
                                                     # Galaxian, ViZDoom basic, Maze, Snake, classic control, with
                                                     # baselines, in parallel

Volumes (created out of band; never ``modal deploy`` this app):
    laya-hf-cache     -> /cache/hf   (HF_HOME, shared model weights)
    laya-datasets     -> /data       (read-only; /data/vqa/<name>/{<split>.jsonl, images/, _READY})
    laya-checkpoints  -> /ckpt       (this app writes only under /ckpt/smolvlm/ and /ckpt/modernvbert/;
                                      robustness results under /ckpt/smolvlm/robustness/)

Data: the fine-tune jobs default to The Cauldron subsets (``CAULDRON_DATASETS``, written by ``prepare_cauldron``);
two layouts exist on the volume: the capped ``cauldron_<subset>`` sets (10,000 usable rows per subset, 5% val)
and the uncapped ``cauldronfull_<subset>`` sets (``CAULDRON_FULL_DATASETS``, every usable row, 1% val, written by
``prepare_cauldron --prefix cauldronfull_ --max-rows 0 --val-pct 1``). The original three VQA sets (``VQA_DATASETS``, official val splits, the README table) stay available with
``--datasets aokvqa,scienceqa,vqav2_yesno`` and as ``--val-datasets`` for a Cauldron-trained model. The ``score``
head has its own sets (``SCORE_DATASETS``, written by ``prepare_score`` from ``laya.rubric``): rubric-graded
responses, aesthetics votes, generated-image ratings and damage levels; add them to ``--datasets`` to train it.
The held-out evaluation sets (``EVAL_DATASETS``, written by ``prepare_eval`` from ``laya.evalsets``) score
calibration against human vote histograms, abstention, hallucination and rubric scoring; ``evaluate`` includes
them, and ``--val-split test`` reads the official test split where one exists (KonIQ, FER+).
``--datasets`` and ``--val-datasets`` take dataset names and the group names ``vqa``, ``cauldron``, ``cauldronfull``,
``score`` and ``eval`` (``DATASET_GROUPS``).

Run names: ``finetune`` and ``finetune_long`` take ``--backbone`` and write under that backbone's root
(``CKPT_ROOTS``). Everywhere a job takes a saved run (``evaluate``, ``try_model``, ``doom_eval``, ``games_eval``, ``--init-from``)
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
EVAL_SOURCES = ("koniq", "evalmuse", "cifar10h", "ferplus", "vizwiz", "pope_random", "pope_popular",
                "pope_adversarial")  # see laya/evalsets.py
EVAL_DATASETS = tuple("eval_" + s for s in EVAL_SOURCES)  # held-out sets written by prepare_eval
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
                  "score": SCORE_DATASETS, "eval": EVAL_DATASETS}


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
    retries=modal.Retries(max_retries=3, initial_delay=10.0),
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
    restart: bool = False,
    state_every_min: float = 10.0,
    crash_at_step: int = 0,
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
    * Durability: every ``state_every_min`` minutes (clamped to ``vlm_train.MIN_STATE_MINUTES``, and backed off
      further when writes are slow) and after every eval the run writes ``<out>/state.pt`` atomically (weights,
      optimizer, step, RNG, per-dataset sample counts, elapsed training time, best-so-far and the log), and a
      preempted or retried container resumes from it; ``max_minutes`` counts training time across attempts. The
      state is deleted once the run finishes, so a later call with the same ``run_name`` starts fresh.
      ``restart`` ignores an unfinished run's state; ``crash_at_step`` raises once at that step to test the
      resume path (Modal retries the container).
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

    # ------------------------------------------------------------------ durability: resume from state.pt
    state_path = os.path.join(out_dir, "state.pt")
    restart_marker = os.path.join(out_dir, "restarted")
    call_id = modal.current_function_call_id() or str(t_start)
    marked = open(restart_marker).read().strip() if os.path.exists(restart_marker) else ""
    if restart and marked != call_id:
        # only this call's first attempt restarts: its retries must still resume from their own state
        for f_ in (state_path, os.path.join(out_dir, "crashed")):
            if os.path.exists(f_):
                os.remove(f_)
                print("--restart: removed %s" % os.path.basename(f_))
        os.makedirs(out_dir, exist_ok=True)
        with open(restart_marker, "w") as f_:
            f_.write(call_id)
        ckpt_vol.commit()
    resume = None
    if os.path.exists(state_path):
        blob = torch.load(state_path, map_location="cpu", weights_only=False)
        model.load_state_dict(blob["model"])
        model.to("cuda")
        resume, log = blob["train"], blob["log"]
        best.update(blob["best"])
        print("resuming %s from state.pt at step %d (%d evals so far, best mean val acc %.4f)"
              % (run_name, resume["step"], len(log["evals"]), best["score"]), flush=True)

    def save_state(step, tstate):
        ts = time.time()
        os.makedirs(out_dir, exist_ok=True)
        blob = {"train": tstate, "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "best": {k: best[k] for k in ("score", "step")}, "log": log}
        tmp = state_path + ".tmp"
        torch.save(blob, tmp)
        os.replace(tmp, state_path)  # atomic: a torn write never replaces a good state
        ckpt_vol.commit()
        print("  wrote state.pt at step %d (%.1f s)" % (step, time.time() - ts), flush=True)

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

    def maybe_eval(step):
        # Called every few steps as a cheap probe (so ``crash_at_step`` can fire between evals); evaluates on
        # multiples of ``eval_every`` and returns True only then, so the training loop writes state.pt after a
        # real eval and not on every probe.
        if crash_at_step and step >= crash_at_step and not os.path.exists(os.path.join(out_dir, "crashed")):
            os.makedirs(out_dir, exist_ok=True)
            open(os.path.join(out_dir, "crashed"), "w").close()
            ckpt_vol.commit()
            raise RuntimeError("crash_at_step %d: simulated preemption" % crash_at_step)
        if step % eval_every == 0:
            eval_fn(step)
            return True
        return False

    stats = {}
    losses = train(
        model, proc, train_ex, steps=steps, batch_size=batch_size, freeze="full", lr_head=lr_h, lr_backbone=lr_b,
        device="cuda", log_every=100, max_minutes=max_minutes, num_workers=tr_workers,
        prefetch_factor=tr_prefetch, warmup=warmup,
        eval_fn=maybe_eval, eval_every=math.gcd(25, eval_every), max_passes=max_passes or None, stats=stats,
        w_ce_schedule=w_ce_schedule, mix_weights=mix_weights, mix_alpha=mix_alpha,
        resume=resume, save_state_fn=save_state, save_state_every_min=state_every_min,
    )
    if not log["evals"] or log["evals"][-1]["step"] != stats["steps"]:
        eval_fn(stats["steps"])
    chunk = max(1, len(losses) // 10)
    log["train_stats"] = dict(stats, passes={n: round(stats["samples_per_dataset"].get(n, 0) / sizes[n], 2) for n in names})
    log["loss_curve"] = [{"steps": "%d-%d" % (i, min(i + chunk, len(losses)) - 1), "mean_loss": sum(losses[i:i + chunk]) / len(losses[i:i + chunk])}
                         for i in range(0, len(losses), chunk)]
    print("train stats:", json.dumps(log["train_stats"]))
    print("loss curve (10 chunks):", ", ".join("%.3f" % c["mean_loss"] for c in log["loss_curve"]))

    # final model = best checkpoint by mean val acc; a resumed run kept it on disk rather than in memory
    if best["state"] is not None:
        model.load_state_dict(best["state"])
    else:
        from safetensors.torch import load_file

        model.load_state_dict(load_file(os.path.join(out_dir, "best", "model.safetensors")))
        model.to("cuda")
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
    for f_ in (state_path, os.path.join(out_dir, "crashed")):
        if os.path.exists(f_):
            os.remove(f_)  # finished: a later call with this run_name starts fresh
    ckpt_vol.commit()
    print("saved %s/best with temperatures (%.1f min total)" % (out_dir, (time.time() - t_start) / 60))
    return {k: log[k] for k in ("run", "args", "best_step", "best_mean_val_acc", "temperature", "final", "train_stats")}


@app.function(
    image=image,
    gpu=["A10G", "L4", "A100"],  # any of these: an eval should not queue on one GPU type's capacity
    cpu=16,
    memory=32768,
    timeout=60 * 60,
    volumes={"/cache/hf": hf_vol, "/data": data_vol.read_only(), "/ckpt": ckpt_vol.read_only()},
)
def evaluate(run_name: str, datasets: str = ",".join(VQA_DATASETS + CAULDRON_DATASETS + SCORE_DATASETS + EVAL_DATASETS),
             val_split: str = "val",
             max_val: int = 0):
    """Evaluate a saved checkpoint (``<run>`` under /ckpt/smolvlm, or ``modernvbert/<run>``) on the val splits,
    raw and with its temperatures. Defaults to every prepared set (the official VQA splits and the Cauldron
    holdouts); sets that are not prepared are skipped. The result records the GPU it ran on: the function takes any of
    three types, and bf16 scores shift slightly between them (up to about a point on a set of ~100 questions)."""
    import torch

    from laya.vlm import VLMAgent
    from laya.vlm_train import collect_logits, format_metrics, metrics_from

    gpu = torch.cuda.get_device_name(0)
    print("GPU:", gpu)
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
    metas = {}
    for name in {ex["dataset"] for ex in val_ex}:
        try:
            with open(os.path.join("/data/vqa", name, "meta.json")) as f:
                metas[name] = json.load(f)
        except (OSError, ValueError):
            metas[name] = None
    return {"val_raw": raw, "val_calibrated": cal, "temperature": list(agent.temperature), "dataset_meta": metas,
            "gpu": gpu}


def _file_sha256(path: str) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


@app.function(image=image, gpu="L4", cpu=8, memory=32768, timeout=40 * 60,
              volumes={"/cache/hf": hf_vol, "/data": data_vol.read_only(), "/ckpt": ckpt_vol.read_only()})
def row_evidence(run_name: str, datasets: str = "vqa", val_split: str = "val", max_val: int = 0, batch_size: int = 32,
                 code_commit: str = "") -> dict:
    """Score a saved checkpoint on val splits and return one row per example: the evidence behind a headline number.

    Scored exactly as ``evaluate`` and the fine-tune jobs' final eval (``collect_logits``: identity option order,
    bf16 autocast, ``batch_size`` 32), so the rows reproduce those metrics up to GPU-type numerics. Each row has
    the dataset, the record id and its index among the split's usable records, the label, the option order used,
    the raw label-order logits, the probabilities after the checkpoint's per-type temperature (what ``metrics_from``
    calibrates with), and the sha256 of the exact input ids (``laya.vlm.input_ids_sha256``). Returns
    ``{"rows_jsonl": bytes, "meta_json": str}``; the meta records the checkpoint (weights and config sha256, backbone
    revision), the datasets (val file sha256, and ``manifest.json`` / ``meta.json`` where the prep wrote one),
    library versions, the GPU and the metrics computed here.
    """
    import datetime

    import torch
    import transformers

    from laya.vlm import PROMPT_FORMAT_VERSION, VLMAgent
    from laya.vlm_train import collect_logits, metrics_from

    gpu = torch.cuda.get_device_name(0)
    print("GPU:", gpu)
    data_vol.reload()
    val_ex, ds_meta = [], {}
    for name in _ready(datasets):
        va = _load_split(name, val_split, max_val or 0)
        base = os.path.join("/data/vqa", name)
        info = {"n": len(va), "max_val": max_val, "file": "%s.jsonl" % val_split,
                "file_sha256": _file_sha256(os.path.join(base, val_split + ".jsonl"))}
        for extra in ("manifest.json", "meta.json"):
            if os.path.exists(os.path.join(base, extra)):
                with open(os.path.join(base, extra)) as f:
                    info[extra] = json.load(f)
        ds_meta[name] = info
        print("dataset %s: %d val" % (name, len(va)))
        val_ex += [dict(ex, index=i) for i, ex in enumerate(va)]
    path = _ckpt_path(run_name)
    agent = VLMAgent(path, device="cuda")
    t0 = time.time()
    records = collect_logits(agent.model, agent.processor, val_ex, batch_size=batch_size, num_workers=7)
    minutes = (time.time() - t0) / 60
    temps = [float(t) for t in agent.temperature]
    lines = []
    for ex, r in zip(val_ex, records):
        z = r["logits"]
        p = torch.softmax(z / temps[r["qtype"]], -1)
        lines.append(json.dumps({
            "dataset": r["dataset"], "id": ex.get("id"), "index": ex["index"], "qtype": ex["q"]["t"],
            "label": r["label"], "option_order": list(range(len(z))),
            "logits": [round(float(v), 6) for v in z], "probs_calibrated": [round(float(v), 6) for v in p],
            "input_ids_sha256": r["input_ids_sha256"][0],
        }, separators=(",", ":")))
    weights = os.path.join(path, "model.safetensors")
    meta = {
        "run": run_name, "checkpoint_path": path,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "code_commit": code_commit or None, "prompt_format_version": PROMPT_FORMAT_VERSION,
        "weights_sha256": _file_sha256(weights) if os.path.exists(weights) else None,
        "config_sha256": _file_sha256(os.path.join(path, "vlm_agent_config.json")),
        "backbone": agent.cfg.get("backbone"), "backbone_revision": agent.cfg.get("backbone_revision"),
        "readout": agent.model.readout, "option_attention": agent.model.option_attention,
        "temperature": temps, "calibration": "probs_calibrated = softmax(logits / temperature[qtype]), qtype order "
                                              "(choice, score, noul)",
        "scoring": {"orders": "identity", "n_permutations": 1, "autocast": "bf16", "batch_size": batch_size,
                    "weights_dtype": str(agent.model.encoder.dtype).replace("torch.", "")},
        "gpu": gpu, "torch": torch.__version__, "transformers": transformers.__version__,
        "datasets": ds_meta, "val_split": val_split, "n_rows": len(lines), "minutes": round(minutes, 2),
        "metrics": {"val_raw": metrics_from(records), "val_calibrated": metrics_from(records, temps)},
    }
    print(json.dumps(meta["metrics"]["val_calibrated"]))
    # plain bytes and JSON only: the local side has no torch to unpickle e.g. ``torch.__version__`` (a TorchVersion)
    return {"rows_jsonl": ("\n".join(lines) + "\n").encode(), "meta_json": json.dumps(meta)}


def _write_sha256sums(raw_dir: str) -> None:
    """Rewrite ``<raw_dir>/SHA256SUMS`` over every file under it but itself and README.md (``sha256sum -c`` format)."""
    names = []
    for root, _, files in os.walk(raw_dir):
        for fn in files:
            rel = os.path.relpath(os.path.join(root, fn), raw_dir)
            if rel not in ("SHA256SUMS", "README.md"):
                names.append(rel)
    with open(os.path.join(raw_dir, "SHA256SUMS"), "w") as f:
        for rel in sorted(names):
            f.write("%s  %s\n" % (_file_sha256(os.path.join(raw_dir, rel)), rel))


@app.local_entrypoint()
def evidence(run: str = "cauldron-score-2ep-bidir-full/best", datasets: str = "vqa", val_split: str = "val",
             max_val: int = 0, out_dir: str = "results/raw", name: str = ""):
    """modal run modal_app.py::evidence [--run <run>] [--datasets vqa] -- write row-level val predictions.

    Writes ``<out_dir>/<name>.predictions.jsonl.gz`` (gzip with a zero mtime, so the bytes depend only on the rows)
    and ``<name>.meta.json``, then regenerates ``<out_dir>/SHA256SUMS``. Results are create-only: an existing file
    is never overwritten (pick a new ``--name``). ``name`` defaults to the run and datasets, e.g.
    ``smolvlm-cauldron-score-2ep-bidir-full-best.vqa-val``.
    """
    import gzip

    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True,
                           text=True).stdout
    family_run = run if run.startswith(("smolvlm", "modernvbert")) else "smolvlm/" + run
    name = name or "%s.%s-%s" % (family_run.replace("/", "-"), datasets.replace(",", "+"), val_split)
    rows_path = os.path.join(out_dir, name + ".predictions.jsonl.gz")
    meta_path = os.path.join(out_dir, name + ".meta.json")
    for path in (rows_path, meta_path):
        if os.path.exists(path):
            raise SystemExit("%s exists; results are create-only, pass a new --name" % path)
    res = row_evidence.remote(run, datasets=datasets, val_split=val_split, max_val=max_val,
                              code_commit=commit + ("-dirty" if dirty.strip() else ""))
    os.makedirs(out_dir, exist_ok=True)
    with open(rows_path, "wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as f:
        f.write(res["rows_jsonl"])
    meta = dict(json.loads(res["meta_json"]), rows_file=os.path.basename(rows_path))
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")
    _write_sha256sums(out_dir)
    print("wrote %s (%d rows) and %s; %s/SHA256SUMS regenerated" % (rows_path, meta["n_rows"], meta_path, out_dir))
    for ds, m in meta["metrics"]["val_calibrated"].items():
        print("  %-14s n=%5d acc=%.4f ece=%.4f" % (ds, m["n"], m["acc"], m["ece"]))


ROBUSTNESS_DATASETS = ("aokvqa", "scienceqa", "vqav2_yesno", "cauldron_ai2d", "cauldron_visual7w", "cauldron_vsr",
                       "cauldron_mapqa")
ROBUSTNESS_ROOT = CKPT_ROOT + "/robustness"  # <tag>/{predictions.jsonl.gz, summary.json}; never overwritten


@app.function(image=image, gpu="L4", cpu=16, memory=32768, timeout=40 * 60,
              volumes={"/cache/hf": hf_vol, "/data": data_vol.read_only(), "/ckpt": ckpt_vol})
def robustness(run_name: str = "cauldron-score-2ep-bidir-full/best", datasets: str = ",".join(ROBUSTNESS_DATASETS),
               n_per_dataset: int = 300, families: str = "", seed: int = 0, n_boot: int = 1000, val_split: str = "val",
               tag: str = ""):
    """Meaning-preserving perturbations of ``n_per_dataset`` seeded val rows per set (``laya.robustness``): option
    order, rewording, image corruptions, and the shuffled-image / no-image controls, scored with the checkpoint's
    temperatures. Writes the per-row predictions and the summary to ``ROBUSTNESS_ROOT/<tag>/`` (a new directory;
    the job refuses an existing one) so a detached run's results survive the local client, and returns the
    summary."""
    import gzip

    import torch

    from laya import robustness as R
    from laya.vlm import VLMAgent

    out_dir = os.path.join(ROBUSTNESS_ROOT, tag or "%s-n%d-s%d" % (run_name.replace("/", "_"), n_per_dataset, seed))
    ckpt_vol.reload()
    if os.path.exists(out_dir):
        raise SystemExit("%s exists; pass a new --tag" % out_dir)
    t0 = time.time()
    print("GPU:", torch.cuda.get_device_name(0))
    data_vol.reload()
    rows = []
    for name in _ready(datasets):
        src = R.source_rows(_load_split(name, val_split, 0), n=n_per_dataset, seed=seed, dataset=name)
        print("dataset %s: %d source rows" % (name, len(src)))
        rows += src
    fams = [f for f in families.split(",") if f] or list(R.FAMILIES)
    variants = R.build_variants(rows, fams, seed)
    counts = {}
    for v in variants:
        counts[v["family"]] = counts.get(v["family"], 0) + 1
    print("%d rows to score: %s" % (len(variants), counts))
    agent = VLMAgent(_ckpt_path(run_name), device="cuda")
    print("temperatures (choice, score, noul):", [round(t, 3) for t in agent.temperature])
    preds = R.score_rows(agent.model, agent.processor, variants, agent.temperature, batch_size=32, num_workers=14)
    print("scored in %.1f min" % ((time.time() - t0) / 60))
    summary = R.summarize(preds, n_boot, seed)
    print(R.format_table(summary))
    meta = {"run": run_name, "datasets": sorted({r["dataset"] for r in rows}), "n_per_dataset": n_per_dataset,
            "families": fams, "seed": seed, "n_boot": n_boot, "val_split": val_split,
            "temperature": list(agent.temperature), "gpu": torch.cuda.get_device_name(0), "row_counts": counts,
            "minutes": (time.time() - t0) / 60}
    os.makedirs(out_dir)
    with gzip.open(os.path.join(out_dir, "predictions.jsonl.gz"), "wt") as f:
        for p in preds:
            f.write(json.dumps(p, sort_keys=True) + "\n")
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump({"meta": meta, **summary}, f, indent=1)
    ckpt_vol.commit()
    print("wrote %s (%d rows)" % (out_dir, len(preds)))
    return {"meta": meta, "summary": summary, "out_dir": out_dir}


@app.local_entrypoint()
def robustness_eval(run: str = "cauldron-score-2ep-bidir-full/best", datasets: str = ",".join(ROBUSTNESS_DATASETS),
                    n: int = 300, families: str = "", seed: int = 0, tag: str = "", out: str = "results/robustness"):
    """Run ``robustness`` (``modal run --detach`` keeps it going if this client drops), then copy
    ``predictions.jsonl.gz`` and ``summary.json`` from the volume into ``out`` (refusing to overwrite). After a
    dropped client: ``modal volume get laya-checkpoints smolvlm/robustness/<tag>/ <out>``."""
    for fname in ("predictions.jsonl.gz", "summary.json"):
        if os.path.exists(os.path.join(out, fname)):
            raise SystemExit("%s already holds %s; pass a new --out" % (out, fname))
    res = robustness.remote(run, datasets, n, families, seed, tag=tag)
    rel = os.path.relpath(res["out_dir"], "/ckpt")
    os.makedirs(out, exist_ok=True)
    for fname in ("predictions.jsonl.gz", "summary.json"):
        with open(os.path.join(out, fname), "wb") as f:
            for chunk in ckpt_vol.read_file(rel + "/" + fname):
                f.write(chunk)
    print("copied %s -> %s" % (res["out_dir"], out))


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


def _dataset_revision(repo: str, revision: str = "") -> str:
    """The commit a Hub dataset branch, tag or commit resolves to now; the prep jobs load that commit and record it."""
    from huggingface_hub import HfApi

    return HfApi().dataset_info(repo, revision=revision or None).sha


def _write_manifest(tmp_dir: str, sources: dict, params: dict) -> dict:
    """``manifest.json`` next to a prepared dataset's ``{train,val}.jsonl``: the exact upstream sources
    (``{repo: commit}``), the prep parameters, and the sha256 of each jsonl file, so a val split can be checked
    against the row-level evidence that cites it. Written by every prep job from now on (older sets have none)."""
    import datetime

    files = {}
    for fn in sorted(os.listdir(tmp_dir)):
        if fn.endswith(".jsonl"):
            with open(os.path.join(tmp_dir, fn)) as f:
                n = sum(1 for line in f if line.strip())
            files[fn] = {"sha256": _file_sha256(os.path.join(tmp_dir, fn)), "records": n}
    manifest = {"sources": sources, "params": params, "files": files,
                "images": len(os.listdir(os.path.join(tmp_dir, "images"))),
                "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}
    with open(os.path.join(tmp_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


@app.function(image=image, cpu=4, memory=16384, timeout=6 * 60 * 60, volumes={"/cache/hf": hf_vol, "/data": data_vol},
              secrets=[modal.Secret.from_name("huggingface-thaitea")])
def prepare_cauldron_subset(subset: str, max_rows: int = 10000, max_texts: int = 4, val_pct: float = 5.0,
                            max_side: int = 1024, seed: int = 0, prefix: str = "cauldron_", revision: str = ""):
    """Stream one Cauldron subset and write /data/vqa/<prefix><subset>/{train,val}.jsonl + images/.

    Rows are taken in stream order until ``max_rows`` *usable* rows (at least one closed-form turn, see
    ``laya.cauldron``) are written (``max_rows=0``: no cap, the whole subset); each keeps at most ``max_texts`` turns. A seeded ``val_pct`` percent of rows
    go to ``val`` (by row, so an image never sits in both splits). Images are saved as JPEG with the longest side
    at most ``max_side`` (the model sees 512-pixel tiles). The Cauldron is train-only upstream, so its
    ``aokvqa`` / ``scienceqa`` / ``vqav2`` rows are the official train splits and do not overlap the official val
    splits in ``VQA_DATASETS``. The Cauldron is read at one commit (``revision``, default: what ``main`` is now),
    recorded in ``manifest.json``.
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
    rev = _dataset_revision("HuggingFaceM4/the_cauldron", revision)
    ds = load_dataset("HuggingFaceM4/the_cauldron", subset, split="train", streaming=True, revision=rev)
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
    _write_manifest(tmp_dir, {"HuggingFaceM4/the_cauldron": rev}, dict(
        subset=subset, max_rows=max_rows, max_texts=max_texts, val_pct=val_pct, max_side=max_side, seed=seed))
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


def _score_source(name: str, split: str, rng, max_texts: int, max_chars: int, revision=None):
    """Yield ``(row_id, image, records)`` for one source and split (``"train"`` / ``"val"``); ``image`` is a PIL
    image or a Hub file path to download. Sources with an upstream validation split use it for ``val``.
    ``revision`` pins the source repo's commit."""
    from datasets import load_dataset

    from laya.rubric import SOURCES, ava_record, crisismmd_record, richhf_records, vlfeedback_records

    repo = SOURCES[name]
    if name == "vlfeedback":
        if split == "val":
            return  # no upstream split: the job holds out val_pct of the train rows
        for i, row in enumerate(load_dataset(repo, split="train", streaming=True, revision=revision)):
            rid = "vlf-%s" % (row.get("id") or i)
            yield rid, row["image"], vlfeedback_records(row, rid, rng, max_texts=max_texts, max_chars=max_chars)
    elif name == "ava":
        for i, row in enumerate(load_dataset(repo, split="validation" if split == "val" else "train", streaming=True,
                                                 revision=revision)):
            rid = "ava-%s" % (row.get("image_id") or i)
            rec = ava_record(row, rid, rng)
            yield rid, row["image"], [rec] if rec else []
    elif name == "richhf":
        for i, row in enumerate(load_dataset(repo, split="validation" if split == "val" else "train", streaming=True,
                                                 revision=revision)):
            rid = "richhf-%d" % i
            yield rid, row["image"], richhf_records(row, rid, rng, max_texts=max_texts)
    elif name == "crisismmd":
        for i, row in enumerate(load_dataset(repo, "damage", split="dev" if split == "val" else "train",
                                             revision=revision)):
            rid = "crisis-%s" % (row.get("image_id") or i)
            rec = crisismmd_record(row, rid, rng)
            yield rid, row.get("image") or row["image_path"], [rec] if rec else []
    else:
        raise ValueError("unknown score source %r (one of %s)" % (name, sorted(SOURCES)))


@app.function(image=image, cpu=4, memory=16384, timeout=6 * 60 * 60, volumes={"/cache/hf": hf_vol, "/data": data_vol},
              secrets=[modal.Secret.from_name("huggingface-thaitea")])
def prepare_score_dataset(name: str, max_rows: int = 0, max_texts: int = 2, val_pct: float = 5.0, max_val: int = 1000,
                          max_side: int = 1024, seed: int = 0, balance: float = 3.0, max_chars: int = 1200,
                          prefix: str = "score_", revision: str = ""):
    """Stream one rubric-scored source (``laya.rubric.SOURCES``) and write /data/vqa/<prefix><name>/{train,val}.jsonl + images/.

    Rows are taken in stream order until ``max_rows`` usable rows (0: all); each keeps at most ``max_texts``
    records (sampled). Sources with an upstream validation split use it for ``val`` (capped at ``max_val``
    rows); the others hold out a seeded ``val_pct`` percent of rows by row, so an image never sits in both
    splits. After streaming, the train split is level-balanced (``laya.rubric.balance_levels``: no level above
    ``balance`` x the median level count) and images no record points at are deleted. Images are JPEG with the
    longest side at most ``max_side``. Records carry ``"target"`` (AVA's vote histogram) where the source has it.
    The source is read at one commit (``revision``, default: what ``main`` is now), recorded in ``manifest.json``.
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
    rev = _dataset_revision(SOURCES[name], revision)
    t0 = time.time()
    recs = {"train": [], "val": []}
    n_rows = {"train": 0, "val": 0}
    for split in ("train", "val"):
        for rid, image, rows in _score_source(name, split, rng, max_texts, max_chars, rev):
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
                image = Image.open(hf_hub_download(SOURCES[name], image, repo_type="dataset", revision=rev))
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
    _write_manifest(tmp_dir, {SOURCES[name]: rev}, dict(
        name=name, max_rows=max_rows, max_texts=max_texts, val_pct=val_pct, max_val=max_val, max_side=max_side,
        seed=seed, balance=balance, max_chars=max_chars))
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
# Held-out evaluation sets (laya.evalsets): human-vote calibration, abstention, hallucination, rubric scoring
# ---------------------------------------------------------------------------------------------------------

eval_image = _with_local_code(base_image.pip_install("requests"))


def _hf_file_url(repo: str, filename: str) -> str:
    from huggingface_hub import hf_hub_url

    return hf_hub_url(repo, filename, repo_type="dataset")


def _hf_headers() -> dict:
    token = os.environ.get("HF_TOKEN")
    return {"Authorization": "Bearer " + token} if token else {}


def _koniq_source(rng):
    """KonIQ-10k from the pyiqa mirror's tarball, streamed: its own label csv comes first, then the 512x384
    images, whose JPEG bytes are kept as they are (re-encoding would change the quality being rated)."""
    import csv
    import io
    import tarfile

    import requests

    from laya.evalsets import SOURCES, koniq_record

    url = _hf_file_url(SOURCES["koniq"], "koniq10k.tgz")
    rows = None
    with requests.get(url, headers=_hf_headers(), stream=True, timeout=120) as r:
        r.raise_for_status()
        with tarfile.open(fileobj=r.raw, mode="r|gz") as tar:
            for m in tar:
                if m.name.endswith("koniq10k_distributions_sets.csv"):
                    rows = {row["image_name"]: row for row in csv.DictReader(io.StringIO(tar.extractfile(m).read().decode("utf-8")))}
                    left = set(rows)
                elif "/512x384/" in m.name and m.name.endswith(".jpg"):
                    if rows is None:
                        raise RuntimeError("koniq10k.tgz: images before the label csv")
                    fn = m.name.rsplit("/", 1)[1]
                    got = koniq_record(rows[fn], rng) if fn in rows else None
                    if got is None:
                        continue
                    split, rec = got
                    yield split, rec["id"], (tar.extractfile(m).read(), ".jpg"), [rec]
                    left.discard(fn)
                    if not left:
                        return


def _evalmuse_source(rng, max_rows: int, val_pct: float, seed: int, workers: int = 16):
    """EvalMuse-40K: labels from train_list.json, val a seeded share of prompts, and each needed image read out of
    the 54 GB split zip by range requests (``laya.evalsets.MultiPartZip``) instead of downloading it."""
    import io
    import threading
    from concurrent.futures import ThreadPoolExecutor

    import requests
    from huggingface_hub import hf_hub_download
    from PIL import Image

    from laya.evalsets import SOURCES, MultiPartZip, evalmuse_record, stable_split

    repo = SOURCES["evalmuse"]
    with open(hf_hub_download(repo, "train_list.json", repo_type="dataset")) as f:
        rows = json.load(f)
    todo = {"train": [], "val": []}
    for row in rows:
        rec = evalmuse_record(row, rng)
        if rec is not None:
            todo[stable_split(str(row["prompt_id"]), val_pct, seed)].append((row["img_path"], rec))
    if max_rows and len(todo["train"]) > max_rows:
        todo["train"] = rng.sample(todo["train"], max_rows)
    parts = ["images.zip.part-a%s" % c for c in "abcdef"]
    urls = [_hf_file_url(repo, p) for p in parts]
    sess = requests.Session()
    sess.headers.update(_hf_headers())
    sizes = [int(sess.head(u, allow_redirects=True, timeout=60).headers["Content-Length"]) for u in urls]

    def fetch(i, start, end):
        for attempt in range(5):
            try:
                r = sess.get(urls[i], headers={"Range": "bytes=%d-%d" % (start, end - 1)}, timeout=120)
                r.raise_for_status()
                if len(r.content) == end - start:
                    return r.content
            except requests.RequestException:
                pass
            time.sleep(2 ** attempt)
        raise RuntimeError("range read failed: %s [%d, %d)" % (parts[i], start, end))

    archive, local = MultiPartZip(sizes, fetch), threading.local()
    names = {n.split("dataset/images/", 1)[-1]: n for n in archive.open().namelist()}

    def read(img_path):
        if not hasattr(local, "zf"):
            local.zf = archive.open()
        return local.zf.read(names[img_path])

    for split in ("val", "train"):
        with ThreadPoolExecutor(workers) as pool:
            items = todo[split]
            for (img_path, rec), data in zip(items, pool.map(lambda it: read(it[0]), items)):
                # rated for prompt alignment, not image quality, so re-encoding (JPEG, max_side) loses nothing
                yield split, rec["id"], Image.open(io.BytesIO(data)), [rec]


def _eval_source(name: str, rng, max_rows: int, val_pct: float, seed: int, info: dict):
    """Yield ``(split, image_key, image, records)`` for one evaluation source; ``image`` is a PIL image or
    ``(bytes, ext)`` kept verbatim. ``info`` collects source checks for meta.json."""
    import csv
    import io

    from datasets import load_dataset

    from laya import evalsets as E

    repo = E.SOURCES[name]
    if name == "koniq":
        yield from _koniq_source(rng)
    elif name == "evalmuse":
        yield from _evalmuse_source(rng, max_rows, val_pct, seed)
    elif name == "cifar10h":
        for i, row in enumerate(load_dataset(repo, split="train", streaming=True)):
            rec = E.cifar10h_record(row, i, rng)
            if rec:
                yield "val", rec["id"], row["image"], [rec]
    elif name == "ferplus":
        import requests

        text = requests.get(E.FERPLUS_VOTES_URL, timeout=120).text
        votes = list(csv.DictReader(io.StringIO(text)))
        for usage, hf_split in (("Training", "train"), ("PublicTest", "valid"), ("PrivateTest", "test")):
            ds = load_dataset(repo, split=hf_split)
            sub = [(i, v) for i, v in enumerate(votes) if v["Usage"] == usage]
            if len(sub) != len(ds):
                raise RuntimeError("FER+ %s: %d vote rows but %d images" % (usage, len(sub), len(ds)))
            labels = ds["label"]
            agree = E.ferplus_agreement((v, labels[k]) for k, (_, v) in enumerate(sub))
            info["fer2013_agreement_" + usage] = round(agree, 3)
            if agree < 0.45:  # ~0.65 when aligned, ~0.17 when off by one row
                raise RuntimeError("FER+ %s: votes do not line up with %s (agreement %.2f)" % (usage, repo, agree))
            for k, (i, v) in enumerate(sub):
                got = E.ferplus_record(v, i, rng)
                if got:
                    yield got[0], got[1]["id"], ds[k]["image"], [got[1]]
    elif name == "vizwiz":
        for row in load_dataset(repo, split="val", streaming=True):
            rec = E.vizwiz_record(row, rng)
            if rec:
                yield "val", rec["id"], row["image"], [rec]
    elif name.startswith("pope_"):
        for row in load_dataset(repo, "Full", split=name.split("_", 1)[1], streaming=True):
            rec = E.pope_record(row)
            if rec:
                yield "val", "pope-" + str(row["image_source"]), row["image"], [rec]
    else:
        raise ValueError("unknown eval source %r (one of %s)" % (name, sorted(E.SOURCES)))


def _save_eval_image(image, base: str, key: str, max_side: int) -> str:
    """Write one image under ``base/images``: ``(bytes, ext)`` verbatim; a PIL image as PNG when it is tiny
    (CIFAR, FER faces: JPEG would smear them), else as JPEG with the longest side at most ``max_side``."""
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in key)
    if isinstance(image, tuple):
        rel = "images/%s%s" % (safe, image[1])
        with open(os.path.join(base, rel), "wb") as f:
            f.write(image[0])
        return rel
    im = image.convert("RGB")
    if max(im.size) <= 64:
        rel = "images/%s.png" % safe
        im.save(os.path.join(base, rel))
    else:
        rel = "images/%s.jpg" % safe
        im.thumbnail((max_side, max_side))
        im.save(os.path.join(base, rel), quality=92)
    return rel


@app.function(image=eval_image, cpu=8, memory=32768, timeout=6 * 60 * 60, volumes={"/cache/hf": hf_vol, "/data": data_vol},
              secrets=[modal.Secret.from_name("huggingface-thaitea")])
def prepare_eval_dataset(name: str, max_rows: int = 10000, max_val: int = 0, val_pct: float = 10.0, max_side: int = 1024,
                         seed: int = 0, prefix: str = "eval_"):
    """Write one ``laya.evalsets`` source to /data/vqa/<prefix><name>/{train,val[,test]}.jsonl + images/.

    Splits follow the source (see ``laya.evalsets``); sets without a train split get an empty train.jsonl, so they
    are for ``evaluate`` / ``--val-datasets`` only. ``max_rows`` caps the train records (0: all), ``max_val`` the
    val and test records each (0: all: these are eval sets, so none are dropped by default); ``val_pct`` is the
    share of prompts held out where the source has no labelled val split (EvalMuse). No level balancing: an
    evaluation set keeps its natural label distribution.
    """
    import random
    import shutil
    from collections import Counter

    final_dir = os.path.join("/data/vqa", prefix + name)
    tmp_dir = final_dir + ".tmp"
    shutil.rmtree(tmp_dir, ignore_errors=True)
    os.makedirs(os.path.join(tmp_dir, "images"))
    rng = random.Random(seed)
    t0 = time.time()
    recs = {"train": [], "val": [], "test": []}
    saved, info = {}, {}
    for n_seen, (split, key, image, rows) in enumerate(_eval_source(name, rng, max_rows, val_pct, seed, info), 1):
        cap = max_rows if split == "train" else max_val
        if cap and len(recs[split]) >= cap:
            continue
        if key not in saved:
            saved[key] = _save_eval_image(image, tmp_dir, key, max_side)
        for rec in rows:
            rec["image"] = saved[key]
        recs[split] += rows
        if n_seen % 2000 == 0:
            print("%s: %s records, %d images in %.1f min" % (name, {k: len(v) for k, v in recs.items()}, len(saved),
                                                            (time.time() - t0) / 60), flush=True)
    for split in ("train", "val", "test"):
        if split == "test" and not recs["test"]:
            continue
        with open(os.path.join(tmp_dir, split + ".jsonl"), "w") as f:
            for rec in recs[split]:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    meta = {"source": name, "records": {s: len(r) for s, r in recs.items()}, "images": len(saved),
            "labels": {s: dict(sorted(Counter(int(x["label"]) for x in r).items())) for s, r in recs.items() if r},
            "soft_targets": sum("target" in x for r in recs.values() for x in r), "max_rows": max_rows, "max_val": max_val,
            "val_pct": val_pct, "max_side": max_side, "seed": seed, "minutes": round((time.time() - t0) / 60, 1), **info}
    with open(os.path.join(tmp_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    shutil.rmtree(final_dir, ignore_errors=True)
    os.rename(tmp_dir, final_dir)
    open(os.path.join(final_dir, "_READY"), "w").close()
    data_vol.commit()
    print("%s: records %s, %d images, labels %s, %.1f min" % (name, meta["records"], len(saved), meta["labels"], meta["minutes"]))
    return meta


@app.local_entrypoint()
def prepare_eval(names: str = ",".join(EVAL_SOURCES), max_rows: int = 10000, max_val: int = 0, val_pct: float = 10.0,
                 max_side: int = 1024, prefix: str = "eval_"):
    """modal run modal_app.py::prepare_eval [--names koniq,pope_adversarial] -- one container per source, in parallel."""
    kw = dict(max_rows=max_rows, max_val=max_val, val_pct=val_pct, max_side=max_side, prefix=prefix)
    print("%-18s %-40s %s" % ("source", "records", "labels (val)"))
    for meta in prepare_eval_dataset.map([n for n in names.split(",") if n], kwargs=kw, order_outputs=True,
                                         return_exceptions=True):
        if isinstance(meta, Exception):
            print("FAILED:", repr(meta)[:300])
            continue
        print("%-18s %-40s %s" % (meta["source"], meta["records"], meta["labels"].get("val")))


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
    import vizdoom

    _write_manifest(tmp_dir, {"vizdoom": vizdoom.__version__},
                    dict(n_train=n_train, n_val=n_val, eps=eps, tics=tics, seed=seed))
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


# ---------------------------------------------------------------------------------------------------------
# Typed readout versus generated JSON (benchmarks/decision_vs_generation.py)
# ---------------------------------------------------------------------------------------------------------

bench_image = image.add_local_dir("benchmarks", "/root/benchmarks")


@app.function(image=bench_image, gpu="L4", timeout=30 * 60, volumes={"/cache/hf": hf_vol, "/ckpt": ckpt_vol.read_only()},
              secrets=[modal.Secret.from_name("huggingface-thaitea")])  # the token only lifts the Hub's download rate limit
def decision_vs_generation_run(git_sha: str, git_dirty: bool, repeats: int = 5, warmup: int = 2,
                               max_new_tokens: int = 256, dtype: str = "bf16", run_name: str = "") -> str:
    """Run the benchmark on an L4 and return its report as JSON text. ``run_name`` times a checkpoint on the volume instead of
    the pinned Hub revision of thaitea/laya-vision."""
    import pathlib

    os.environ["LAYA_GIT_SHA"], os.environ["LAYA_GIT_DIRTY"] = git_sha, str(git_dirty)
    sys.path.insert(0, "/root")
    from benchmarks.decision_vs_generation import run

    out = pathlib.Path("/tmp/decision-vs-generation.json")
    report = run(out, repeats=repeats, warmup=warmup, max_new_tokens=max_new_tokens, device="cuda", dtype=dtype,
                 typed_path=_ckpt_path(run_name) if run_name else "")
    hf_vol.commit()
    if run_name:
        report["models"]["typed"]["source"] = "laya-checkpoints:" + run_name
    return json.dumps(report, ensure_ascii=False, allow_nan=False)  # plain JSON: the local side has no torch


@app.local_entrypoint()
def decision_vs_generation(output: str = "results/raw/decision-vs-generation-l4.json", repeats: int = 5, warmup: int = 2,
                           max_new_tokens: int = 256, dtype: str = "bf16", run: str = ""):
    """modal run modal_app.py::decision_vs_generation [--output results/raw/<new>.json] [--run <ckpt>/best]

    Times ``predict`` against the base backbone generating a compact JSON array on an L4 and writes the raw report
    (per-run timings, outputs, token timelines, revisions, versions and the git sha of the code measured) to
    ``--output``, which must not exist yet."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from benchmarks.decision_vs_generation import summary

    if os.path.exists(output):
        raise SystemExit("%s exists; pass a new --output" % output)
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    if dirty:
        print("warning: uncommitted changes; the report records git_sha=%s with dirty=true" % sha)
    report = json.loads(decision_vs_generation_run.remote(sha, dirty, repeats, warmup, max_new_tokens, dtype, run))
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "w") as f:
        f.write(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps(summary(report), indent=2))
    print("wrote", output)


# ---------------------------------------------------------------------------------------------------------
# Maze and Snake (laya.gridgames), and the games suite: Atari, ViZDoom, Maze, Snake and classic control on one checkpoint
# ---------------------------------------------------------------------------------------------------------

GRID_SEED = 200_000  # eval episodes use seeds GRID_SEED + i; keep training data off this range
SUITE_ATARI_GAMES = ("Freeway", "Breakout", "Galaxian")
SUITE_CONTROL_GAMES = ("CartPole", "Acrobot", "MountainCar", "LunarLander")


def _load_policy_agent(model: str):
    from laya.vlm import VLMAgent

    path = _ckpt_path(model)
    return VLMAgent(path if os.path.exists(path) else model, device="cuda", dtype="bf16")


def _play_grid(game: str, policy: str, model: str, episodes: int, size: int, seed: int, max_steps: int) -> dict:
    from laya import gridgames

    t0 = time.time()
    if policy == "model":
        fn = gridgames.model_policy(_load_policy_agent(model), game)
    elif policy == "expert":
        fn = gridgames.expert_policy
    else:
        fn = gridgames.random_policy(seed)
    out = gridgames.play_episodes(game, fn, episodes, size, seed, max_steps)
    out.update(policy="model:" + model if policy == "model" else policy, seconds=round(time.time() - t0, 1))
    print(json.dumps({k: v for k, v in out.items() if k != "results"}))
    return out


@app.function(image=image, cpu=2, timeout=30 * 60)
def play_grid_baseline(game: str, policy: str = "expert", episodes: int = 50, size: int = 0, seed: int = GRID_SEED,
                       max_steps: int = 0):
    """``expert`` or ``random`` on Maze / Snake, on the same seeded episodes as ``play_grid``."""
    return _play_grid(game, policy, "", episodes, size, seed, max_steps)


@app.function(image=image, gpu="L4", timeout=60 * 60, volumes={"/cache/hf": hf_vol, "/ckpt": ckpt_vol.read_only()})
def play_grid(game: str, model: str, episodes: int = 50, size: int = 0, seed: int = GRID_SEED, max_steps: int = 0):
    """Play ``episodes`` of Maze or Snake (``laya.gridgames``) with a checkpoint (bf16): each step the rendered
    screen and ``laya.games.maze_question`` / ``snake_question`` go to ``predict`` and its choice is the move.
    Episode i uses seed ``seed + i``, so every checkpoint and baseline plays the same levels."""
    return _play_grid(game, "model", model, episodes, size, seed, max_steps)


def _grid_row(r: dict) -> str:
    if r["game"] == "maze":
        return "%-32s %5d %10.0f%% %11.2f %10.1f" % (r["policy"], r["size"], 100 * r["solve_rate"], r["efficiency"], r["mean_steps"])
    return "%-32s %5d %10.1f %11d %10.1f  %s" % (r["policy"], r["size"], r["mean_eaten"], r["max_eaten"], r["mean_steps"], r["ends"])


def _grid_header(game: str) -> str:
    if game == "maze":
        return "%-32s %5s %11s %11s %10s" % ("policy", "size", "solved", "efficiency", "steps/ep")
    return "%-32s %5s %10s %11s %10s  %s" % ("policy", "size", "food/ep", "max food", "steps/ep", "ends")


def _grid_eval(game: str, models: str, sizes: str, episodes: int) -> list:
    calls = []
    for size in [int(s) for s in sizes.split(",") if s]:
        calls += [play_grid_baseline.spawn(game, p, episodes, size) for p in ("expert", "random")]
        calls += [play_grid.spawn(game, m, episodes, size) for m in models.split(",") if m]
    results = [c.get() for c in calls]
    print(_grid_header(game))
    for r in results:
        print(_grid_row(r))
    return results


@app.local_entrypoint()
def maze_eval(models: str = "cauldron-score-2ep-bidir-full/best", sizes: str = "4,6,8", episodes: int = 50):
    """modal run modal_app.py::maze_eval --models a/best,b/best [--sizes 4,6,8,12]  -- expert, random, each model."""
    _grid_eval("maze", models, sizes, episodes)


@app.local_entrypoint()
def snake_eval(models: str = "cauldron-score-2ep-bidir-full/best", sizes: str = "10", episodes: int = 20):
    """modal run modal_app.py::snake_eval --models a/best,b/best [--sizes 8,10]  -- expert, random, each model."""
    _grid_eval("snake", models, sizes, episodes)


atari_image = _with_local_code(base_image.pip_install("ale-py", "gymnasium"))


def _atari_baseline(game: str) -> dict:
    """Expert and random scores from the expert data's meta.json, as ``modal_atari_train.expert_baseline`` reads
    them (the 4,500-step-capped scores when present); ``None`` for a game without expert data, e.g. Galaxian."""
    try:
        with open(os.path.join("/data/atari/expert", game, "meta.json")) as f:
            meta = json.load(f)
    except (OSError, ValueError):
        return {"expert": None, "random": None, "capped": False}
    if meta.get("expert_score_cap4500") is not None and meta.get("random_score_cap4500") is not None:
        return {"expert": meta["expert_score_cap4500"], "random": meta["random_score_cap4500"], "capped": True}
    return {"expert": meta.get("expert_score"), "random": meta.get("random_score"), "capped": False}


@app.function(image=atari_image, gpu="L4", cpu=4, timeout=60 * 60,
              volumes={"/cache/hf": hf_vol, "/data": data_vol.read_only(), "/ckpt": ckpt_vol.read_only()})
def play_atari_game(game: str, model: str, episodes: int = 3, max_steps: int = 4500, seed: int = 100_000,
                    random_episodes: int = 10):
    """One Atari game with a checkpoint, greedy, at the settings of ``modal_atari_train.play_atari`` (same seeds,
    step cap, auto-FIRE and frame count from the checkpoint), so the numbers compare with ``atari_eval``'s.
    ``normalized`` is (model - random) / (expert - random) against the expert data's baseline, when there is one."""
    from laya.atari_train import game_actions, model_policy, play, random_policy

    t0 = time.time()
    actions = game_actions(game)
    rnd = play(game, random_policy(len(actions), seed), random_episodes, max_steps, seed)
    agent = _load_policy_agent(model)
    frames = int(agent.cfg.get("atari_frames", 1))
    res = play(game, model_policy(agent, game, actions, False, seed, frames), episodes, max_steps, seed)
    base = _atari_baseline(game)
    norm = None
    if base["expert"] is not None and base["random"] is not None and base["expert"] != base["random"]:
        norm = (res["mean_score"] - base["random"]) / (base["expert"] - base["random"])
    out = {"game": game, "model": model, "frames": frames, "model_score": res["mean_score"], "model_scores": res["scores"],
           "model_steps": res["steps"], "model_capped": res["capped"], "actions": res["actions"],
           "random_score": rnd["mean_score"], "expert_score": base["expert"], "baseline_random": base["random"],
           "baseline_capped": base["capped"], "normalized": norm, "seconds": round(time.time() - t0, 1)}
    print(json.dumps(out))
    return out


control_image = _with_local_code(base_image.pip_install("gymnasium[classic-control,box2d]")
                                 .env({"SDL_VIDEODRIVER": "dummy", "SDL_AUDIODRIVER": "dummy"}))


@app.function(image=control_image, gpu="L4", cpu=4, timeout=90 * 60,
              volumes={"/cache/hf": hf_vol, "/ckpt": ckpt_vol.read_only()})
def play_control(game: str, model: str, episodes: int = 10, seed: int = GRID_SEED):
    """One classic-control game (``laya.controlgames``: CartPole, Acrobot, MountainCar, LunarLander) with a
    checkpoint, greedy, plus the scripted expert and random play on the same seeded episodes. Each step the
    rendered screen (previous frame ghosted in) and ``laya.games.control_question`` go to ``predict``.
    ``normalized`` is (model - random) / (expert - random)."""
    from laya import controlgames as cg

    t0 = time.time()
    exp = cg.play_episodes(game, cg.expert_policy, episodes, seed)
    rnd = cg.play_episodes(game, cg.random_policy(seed), episodes, seed)
    res = cg.play_episodes(game, cg.model_policy(_load_policy_agent(model), game), episodes, seed)
    out = {"game": game, "model": model, "episodes": episodes, "seed": seed,
           "model_score": res["mean_score"], "model_scores": [e["score"] for e in res["results"]],
           "model_solved": res["solved_rate"], "model_steps": res["mean_steps"], "actions": res["actions"],
           "random_score": rnd["mean_score"], "random_solved": rnd["solved_rate"],
           "expert_score": exp["mean_score"], "expert_solved": exp["solved_rate"],
           "solved_at": cg.GAMES[game]["solved"],
           "normalized": cg.normalized(res["mean_score"], rnd["mean_score"], exp["mean_score"]),
           "seconds": round(time.time() - t0, 1)}
    print(json.dumps(out))
    return out


def _control_print(rows: list) -> None:
    print("%-12s %9s %9s %9s %7s %8s  %s" % ("game", "model", "random", "expert", "norm", "solved", "top actions"))
    for r in rows:
        total = max(1, sum(r["actions"].values()))
        top = ", ".join("%s %d%%" % (a, 100 * n / total) for a, n in sorted(r["actions"].items(), key=lambda kv: -kv[1])[:3])
        norm = "-" if r["normalized"] is None else "%.2f" % r["normalized"]
        print("%-12s %9.1f %9.1f %9.1f %7s %7.0f%%  %s" % (r["game"], r["model_score"], r["random_score"], r["expert_score"],
                                                            norm, 100 * r["model_solved"], top))


@app.local_entrypoint()
def control_eval(models: str = "cauldron-score-2ep-bidir-full/best", games: str = ",".join(SUITE_CONTROL_GAMES),
                 episodes: int = 10):
    """modal run modal_app.py::control_eval --models a/best,b/best [--games CartPole,LunarLander]  -- classic control."""
    calls = [(m, play_control.spawn(g, m, episodes)) for m in models.split(",") if m for g in games.split(",") if g]
    for m in dict.fromkeys(m for m, _ in calls):
        print("\n== %s" % m)
        _control_print([r for mm, c in calls if mm == m for r in [_get(c, "control")] if r])


def _git_state() -> dict:
    """The local checkout's commit and whether it has uncommitted changes: the code the Modal images carry."""
    def git(*args):
        try:
            return subprocess.run(["git", *args], capture_output=True, text=True, timeout=10).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""
    return {"commit": git("rev-parse", "HEAD"), "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(git("status", "--porcelain", "--untracked-files=no"))}


def _games_spawn(model: str, atari_games: str, atari_episodes: int, doom_episodes: int, maze_sizes: str,
                 maze_episodes: int, snake_sizes: str, snake_episodes: int, control_games: str = "",
                 control_episodes: int = 10) -> dict:
    """Start every game of the suite, with its baselines, and return the call handles."""
    calls = {"atari": [play_atari_game.spawn(g, model, atari_episodes) for g in atari_games.split(",") if g],
             "doom": {p: play_doom.spawn(p, "", doom_episodes) for p in ("expert", "random", "always_attack")},
             "grid": [], "control": [play_control.spawn(g, model, control_episodes) for g in control_games.split(",") if g]}
    calls["doom"]["model"] = play_doom.spawn("model", model, doom_episodes)
    for game, sizes, episodes in (("maze", maze_sizes, maze_episodes), ("snake", snake_sizes, snake_episodes)):
        for size in [int(s) for s in sizes.split(",") if s]:
            calls["grid"] += [play_grid_baseline.spawn(game, p, episodes, size) for p in ("expert", "random")]
            calls["grid"].append(play_grid.spawn(game, model, episodes, size))
    return calls


_ERRORS: list = []  # what ``_get`` swallowed during this local run, for the results file


def _get(call, what: str):
    """A call's result, or ``None`` with the error printed and recorded in ``_ERRORS``: one broken game or set
    should not lose the rest."""
    try:
        return call.get()
    except Exception as e:
        print("%s failed: %s" % (what, repr(e)[:300]))
        _ERRORS.append({"what": what, "error": repr(e)[:1000]})
        return None


def _games_collect(calls: dict) -> dict:
    out = {"atari": [], "doom": {}, "maze": [], "snake": [], "control": []}
    for c in calls["atari"]:
        r = _get(c, "atari game")
        if r:
            out["atari"].append(r)
    for p, c in calls["doom"].items():
        r = _get(c, "doom " + p)
        if r:
            out["doom"][p] = r
    for c in calls["grid"]:
        r = _get(c, "grid game")
        if r:
            out[r["game"]].append(r)
    for c in calls.get("control", []):
        r = _get(c, "control game")
        if r:
            out["control"].append(r)
    return out


def _games_print(results: dict) -> None:
    print("\n== Atari (greedy)")
    print("%-10s %10s %10s %10s %8s  %s" % ("game", "model", "random", "expert", "norm", "top actions"))
    for r in results["atari"]:
        top = ", ".join("%s %d%%" % (a, 100 * n / max(1, sum(r["actions"].values())))
                        for a, n in sorted(r["actions"].items(), key=lambda kv: -kv[1])[:3])
        f = lambda v, fmt="%.1f": "-" if v is None else fmt % v  # noqa: E731
        print("%-10s %10.1f %10.1f %10s %8s  %s" % (r["game"], r["model_score"], r["random_score"], f(r["expert_score"]),
                                                   f(r["normalized"], "%.2f"), top))
    print("\n== ViZDoom basic")
    print("%-32s %12s %10s %10s" % ("policy", "mean reward", "kill rate", "steps/ep"))
    for r in results["doom"].values():
        print("%-32s %12.1f %9.0f%% %10.1f" % (r["policy"], r["mean_reward"], 100 * r["kill_rate"], r["mean_steps"]))
    for game in ("maze", "snake"):
        print("\n== %s" % game.capitalize())
        print(_grid_header(game))
        for r in results[game]:
            print(_grid_row(r))
    if results.get("control"):
        print("\n== Classic control (greedy; normalized: 0 = random, 1 = expert)")
        _control_print(results["control"])


@app.local_entrypoint()
def games_eval(model: str, atari_games: str = ",".join(SUITE_ATARI_GAMES), atari_episodes: int = 3,
               doom_episodes: int = 50, maze_sizes: str = "4,6,8", maze_episodes: int = 50, snake_sizes: str = "10",
               snake_episodes: int = 20, control_games: str = ",".join(SUITE_CONTROL_GAMES), control_episodes: int = 10,
               out: str = ""):
    """modal run modal_app.py::games_eval --model <run>/best [--out games.json]  -- the games suite on one checkpoint.

    Atari (Freeway, Breakout, Galaxian by default), ViZDoom ``basic``, Maze at each size, Snake and the classic
    control games (CartPole, Acrobot, MountainCar, LunarLander), all in parallel, with each game's baselines on the
    same seeds: random (and the expert data's score) for Atari; the scripted expert, random and always-attack for
    Doom; the BFS expert and random for Maze and Snake; a scripted controller and random for classic control. ``out`` gets every
    result plus the git commit the code came from. ``full_eval`` runs this together with the dataset evals.
    """
    calls = _games_spawn(model, atari_games, atari_episodes, doom_episodes, maze_sizes, maze_episodes, snake_sizes,
                         snake_episodes, control_games, control_episodes)
    results = dict({"model": model, "code": _git_state()}, **_games_collect(calls))
    _games_print(results)
    if out:
        with open(out, "w") as f:
            json.dump(results, f, indent=2)
        print("\nwrote", out)


@app.function(image=image, timeout=10 * 60, volumes={"/ckpt": ckpt_vol})
def save_eval_results(run_name: str, filename: str, payload: dict) -> str:
    """Write a ``full_eval`` result next to the run: ``<run dir>/evals/<filename>``, where the run dir is the
    checkpoint's parent (``<run>/best`` -> ``<run>/evals/``), so ``publish`` never uploads it with the weights."""
    ckpt_vol.reload()
    ckpt = _ckpt_path(run_name)
    evals = os.path.join(os.path.dirname(ckpt.rstrip("/")), "evals")
    os.makedirs(evals, exist_ok=True)
    path = os.path.join(evals, filename)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    ckpt_vol.commit()
    return path


def _datasets_print(evals: dict) -> None:
    """One line per dataset from ``evaluate``'s calibrated metrics (the model as it would be used)."""
    cal = evals["val_calibrated"]
    print("%-30s %6s %7s %6s %6s  %s" % ("dataset", "n", "acc", "ECE", "NLL", "vs human votes (prior)"))
    for name in sorted(cal, key=lambda n: (n == "all", n)):
        m = cal[name]
        extra = []
        for key in ("xent", "soft_xent"):
            if key in m:
                extra.append("%s %.3f%s" % (key, m[key], " (%.3f)" % m["prior_" + key] if "prior_" + key in m else ""))
        if "mae" in m:
            extra.append("mae %.2f" % m["mae"])
        print("%-30s %6d %6.1f%% %6.3f %6.3f  %s" % (name, m["n"], 100 * m["acc"], m["ece"], m["nll"], ", ".join(extra)))


@app.local_entrypoint()
def full_eval(model: str, parts: str = "datasets,games,latency", datasets: str = "vqa,cauldron,score,eval",
              val_split: str = "val", atari_games: str = ",".join(SUITE_ATARI_GAMES), atari_episodes: int = 3,
              doom_episodes: int = 50, maze_sizes: str = "4,6,8", maze_episodes: int = 50, snake_sizes: str = "10",
              snake_episodes: int = 20, control_games: str = ",".join(SUITE_CONTROL_GAMES), control_episodes: int = 10,
              out: str = "", save: bool = True):
    """modal run modal_app.py::full_eval --model <run>/best  -- every eval on one checkpoint, in parallel, one file.

    ``parts`` picks from ``datasets`` (``evaluate`` over ``datasets``: accuracy, ECE, NLL, and the human-vote and
    ordinal metrics), ``games`` (the ``games_eval`` suite) and ``latency`` (``bench_latency``); they all run at
    once. The combined result, with the git commit and each dataset's meta.json, is written to ``out`` (default
    ``eval-results/<run>-<commit>.json``) and, unless ``--no-save``, to ``<run>/evals/`` on the checkpoint volume
    next to the checkpoint. ``scripts/eval_report.py`` turns result files into Markdown; the ``eval`` GitHub
    Actions workflow runs each part as its own job and posts that report on the pull request.
    """
    import datetime

    wanted = [p.strip() for p in parts.split(",") if p.strip()]
    unknown = set(wanted) - {"datasets", "games", "latency"}
    if unknown or not wanted:
        raise SystemExit("--parts takes datasets, games and latency (got %r)" % parts)
    code = _git_state()
    t0 = time.time()
    _ERRORS.clear()
    ds_call = evaluate.spawn(model, ",".join(_expand_datasets(datasets)), val_split) if "datasets" in wanted else None
    lat_call = bench_latency.spawn(model) if "latency" in wanted else None
    game_calls = _games_spawn(model, atari_games, atari_episodes, doom_episodes, maze_sizes, maze_episodes,
                              snake_sizes, snake_episodes, control_games, control_episodes) if "games" in wanted else None
    results = {"model": model, "code": code, "parts": wanted, "val_split": val_split,
               "started": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}
    results["datasets"] = _get(ds_call, "evaluate") if ds_call else None
    results["latency"] = _get(lat_call, "bench_latency") if lat_call else None
    results["games"] = _games_collect(game_calls) if game_calls else None
    results["minutes"] = round((time.time() - t0) / 60, 1)
    results["errors"] = list(_ERRORS)

    if results["datasets"]:
        print("\n== Datasets (%s split, calibrated)" % val_split)
        _datasets_print(results["datasets"])
    if results["latency"]:
        lat = results["latency"]
        print("\n== Latency: median %.1f ms, p90 %.1f ms per predict (L4, bf16)" % (lat["median_ms"], lat["p90_ms"]))
    if results["games"]:
        _games_print(results["games"])

    tag = "" if len(wanted) == 3 else "-" + "-".join(wanted)
    stem = "%s-%s%s%s" % (model.replace("/", "-"), (code["commit"] or "nocommit")[:8], "-dirty" if code["dirty"] else "", tag)
    out = out or os.path.join("eval-results", stem + ".json")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print("\nwrote", out)
    failed = [p for p in wanted if not results[p] or (p == "games" and not any(results["games"].values()))]
    if save:
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        leaf = model.rstrip("/").rsplit("/", 1)[-1]  # best / last share the run's evals/
        name = "%s-%s-%s%s%s.json" % (leaf, stamp, (code["commit"] or "nocommit")[:8], "-dirty" if code["dirty"] else "", tag)
        path = _get(save_eval_results.spawn(model, name, results), "save to volume")
        if path:
            print("saved", path, "on laya-checkpoints")
    if failed:
        raise SystemExit("full_eval: %s produced no results (see the errors above)" % ", ".join(failed))
