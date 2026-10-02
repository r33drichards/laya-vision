"""The fixed autoresearch harness for Laya Vision: train an experiment for 15 minutes, then measure it.

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
4. counts its parameters and times ``predict`` on an L4 in bf16 on ``LATENCY_N`` fixed images, and one game move
   (``LATENCY_MOVES`` decisions along a fixed CartPole frame history, in the checkpoint's own frame mode),
   alternating with the ``REFERENCE`` checkpoint in the same container.

5. plays the games benchmark (``games_eval.py``: Maze, Snake, classic control, Atari Freeway and Breakout, ViZDoom
   basic, fixed seeds) with the reloaded checkpoint, one L4 container per game family, alongside the latency job.

The four objectives (see ``pareto.py``):

* **quality** = macro accuracy over the eval sets minus the ECE pooled over the questions that have a single right
  answer (not the ones scored against human vote spreads, where ECE is not meaningful). Higher is better.
* **games** = mean normalized game score, per game (model - random) / (expert - random) clipped to [-0.5, 1.5]
  against the fixed baselines in ``game_baselines.json``. Higher is better.
* **params_m**: parameters of the saved model, in millions. Lower is better.
* **latency_x** = max(``latency_q_x``, ``game_move_x``), lower is better; the raw times are kept in the result JSON.
  ``latency_q_x`` is the median ``predict`` time on the L4 (preprocessing included) divided by the ``REFERENCE``
  checkpoint's, timed alternately in the same container: raw milliseconds swing by ~50% between L4 hosts (52 vs 76
  ms for one model). ``game_move_x`` = ``game_move_ms`` / ``game_move_ref_ms``: the median time of one game move at
  batch 1 in the checkpoint's own game frame mode (``laya.frames``; state building, e.g. a trail's blend, included)
  over the reference's in ``single``, alternating the same way. The move is ``games_eval.move_fn``, what play runs:
  for ``stack-N`` the encoder-feature cache is warm, so in the steady state only the new frame is encoded but the
  language model reads N images. A stack-4 model's extra per-move cost shows here, where a single-image
  ``predict`` would hide it.

Game frame modes. ``experiment.py``'s ``GAME_FRAMES`` (``"single"``, ``"trail-N"``, ``"stack-N"``, N <= 5; see
``laya.frames``) picks how game states are shown: it passes the mode to ``toolkit`` and ``ctx.game_examples`` and
writes it into ``agent.cfg["game_frames"]``, so the saved checkpoint carries it and the games benchmark and the
latency job play in it.

The result lands in ``autoresearch/runs/<tag>/<commit>.json`` and ``pareto.py`` appends it to
``autoresearch/runs/<tag>/results.tsv`` as keep / discard.

The data pool. Every image the harness touches comes from a fixed, versioned pool (``pool_dir(kind)`` on the
``laya-datasets`` volume), not from the prepared datasets' image folders: reading those small files from the volume
costs about 0.4 s each (measured: 2.4 files/s serially, ~28 files/s with 32 threads), which starved training of
data. ``modal run autoresearch/harness.py --prepare-pool`` builds the pool once, one container per dataset, as
pickles of examples with the encoded image bytes inline:

* ``train``: ``TRAIN_POOL_PER_SET`` seeded examples of each ``TRAINABLE_DATASETS`` train split, calibration tail
  excluded. A 15-minute experiment sees ~90k samples, under one pass over the pool (up to 6,000 per set), and every experiment trains from
  the same one;
* ``calib``: the last ``N_CALIB`` train records of each ``CALIB_DATASETS`` set;
* ``eval``: ``EVAL_PER_SET`` seeded val examples of each ``EVAL_DATASETS`` set;
* ``games``: ``TRAIN_POOL_PER_SET`` seeded frames of each ``GAME_DATASETS`` train split, each with the
  ``next_target`` of its recorded successor where there is one (``laya.vlm_train.with_next_targets``: the target of
  step s + 1 of the same episode, computed over the whole split before sampling, so a sampled frame gets it even
  when its successor is not sampled).

The pool is immutable: changing what goes in means a new ``POOL_VERSION``. A new version need not rebuild every part:
``POOL_KIND_VERSIONS`` names, per part kind, the version whose directory (``pool_dir``) holds that kind's parts, and
a kind whose selection is unchanged keeps reading the older version's files. ``POOL_VERSION`` is the newest version;
``--prepare-pool`` builds only kinds at ``POOL_VERSION`` (create-only: an existing part file is never rewritten) and
refuses to rebuild a missing part of an older version. History:

* ``v1``: 2,000 per set, which 15 minutes cycled through twice (quality fell 0.02);
* ``v2``: 6,000 per set, every kind;
* ``v3``: ``games`` rebuilt with ``next_target`` (the next-move head's auxiliary target), otherwise the same seeded
  frames as v2; ``train``, ``calib`` and ``eval`` are read unchanged from ``pool-v2``.
* ``v4``: ``games`` rebuilt with each frame's history: ``"history"``, the encoded screens of up to
  ``POOL_HISTORY`` previous decisions of its episode, oldest first (fewer at an episode's start, or after an Atari
  auto-FIRE), so ``ctx.game_examples(frames=...)`` can build any frame mode. Atari comes from
  ``/data/atari/experthist`` (the ``expert`` recording replayed with the same seeds, which reproduced all 20,000 train
  records per game exactly, plus the screens play keeps; the old recording skipped every other step in parts of
  long episodes, so 15% of Breakout and 7% of Freeway frames had no exact 4-frame history), ViZDoom's from the
  recorded steps (all present). Same seeded selection and ``next_target`` as v3; the other kinds stay on v2.

Memory snapshots. Both GPU jobs are ``@app.cls(enable_memory_snapshot=True, single_use_containers=True)`` classes
whose ``@modal.enter(snap=True)`` method does the experiment-independent CPU work, which Modal then restores from a
snapshot on later cold starts instead of redoing it: the torch / transformers / laya imports and the whole pool in
memory (``TrainEval``), or the imports and the ``LATENCY_N`` decoded latency images (``Latency``). Nothing touches
the GPU while snapshotting; CUDA starts in the method body. ``single_use_containers`` gives every run a fresh
container, so an experiment that mutates its examples cannot leak into a later run.

Modal only snapshots deployed apps, never the ephemeral app of ``modal run``. So ``main`` deploys this file as
``laya-autoresearch-<hash>`` (hash of this file and the ``laya`` package; a few seconds, skipped when that deployment
exists) and calls its classes. ``experiment.py`` is sent as an argument, so experiment commits reuse the snapshot;
changing ``laya`` or this file makes a new deployment and new snapshots. Modal snapshots the first cold start(s) of
a deployment (those runs pay the full load plus the snapshot) and restores later ones.
"""
import importlib.util
import json
import os
import subprocess
import sys
import time
from typing import Dict, List, Optional

import modal

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO, os.path.join(REPO, "autoresearch")):  # the laya package to ship, and pareto.py
    if _p not in sys.path:
        sys.path.insert(0, _p)

TIME_BUDGET = 900          # seconds of training (upstream: 300; 5 minutes starved the games, see program.md)
EVAL_PER_SET = 300         # seeded questions per eval set
LATENCY_N = 100            # images timed on the L4 (after 10 warm-up calls)
N_CALIB = 100              # last train records per calibration set, held out from training
TRAIN_POOL_PER_SET = 6000  # seeded train examples per trainable set in the pool (fewer where a set is smaller)
SEED = 0
REFERENCE = "cauldron-score-2ep-bidir-full/best"   # latency is reported relative to this checkpoint (= thaitea/laya-vision)
POOL_VERSION = "v4"         # the newest pool version (history in the module docstring)
# The version whose directory holds each kind's parts: v3 rebuilt only ``games`` (adding next_target), v4 again
# (adding each frame's history).
POOL_KIND_VERSIONS = {"train": "v2", "calib": "v2", "eval": "v2", "games": "v4"}
POOL_HISTORY = 4           # previous frames stored per game frame in the pool: states of up to 5 frames
LATENCY_MOVES = 60         # game moves timed per checkpoint (after LATENCY_MOVE_WARMUP untimed ones)
LATENCY_MOVE_WARMUP = 8
POOL_ROOT = "/data/autoresearch"
POOL_DIR = os.path.join(POOL_ROOT, "pool-" + POOL_VERSION)   # where this version's own parts go

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
# Expert game frames experiments may train on (``ctx.game_examples()``): name -> (root, prepared name). The games
# eval plays on seeds these were never recorded on.
GAME_DATASETS = {
    "game_atari_freeway": ("/data/atari/experthist", "Freeway"),
    "game_atari_breakout": ("/data/atari/experthist", "Breakout"),
    "game_doom_basic": ("/data/vqa", "doom_basic"),
}
GAME_FAMILY = {"game_atari_freeway": "atari", "game_atari_breakout": "atari", "game_doom_basic": "doom"}

app = modal.App("laya-autoresearch")
hf_vol = modal.Volume.from_name("laya-hf-cache")
data_vol = modal.Volume.from_name("laya-datasets")
ckpt_vol = modal.Volume.from_name("laya-checkpoints")
# The harness's own modules next to the laya package in every container: the games benchmark (and its fixed
# baselines) and the toolkit experiments import to generate game training data.
HARNESS_FILES = ("games_eval.py", "game_baselines.json", "toolkit.py")


def _with_code(img):
    img = img.add_local_python_source("laya")
    for f in HARNESS_FILES:
        img = img.add_local_file(os.path.join(REPO, "autoresearch", f), "/root/" + f)
    return img


_base = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.14.0", "torchvision==0.29.0", "transformers==5.17.0", "safetensors", "huggingface_hub",
                 "numpy", "pillow", "datasets", "num2words", "gymnasium[classic-control,box2d]==1.3.0")
    .env({"HF_HOME": "/cache/hf", "TOKENIZERS_PARALLELISM": "false", "SDL_VIDEODRIVER": "dummy",
          "SDL_AUDIODRIVER": "dummy"})
)
image = _with_code(_base)
games_image = _with_code(_base.pip_install("ale-py==0.12.1", "vizdoom"))
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


def _with_bytes(ex: Dict, shared: Optional[Dict[str, bytes]] = None) -> Dict:
    """An example whose image paths are replaced by the files' bytes (``laya.vlm`` loads either), including a game
    frame's ``history``. With ``shared`` (path -> bytes) a file read once is the same bytes object everywhere it
    appears, so pickling stores it once: a game frame is also in its successors' histories."""
    def read(p):
        if shared is not None and p in shared:
            return shared[p]
        with open(p, "rb") as f:
            b = f.read()
        if shared is not None:
            shared[p] = b
        return b

    state = ex["state"]
    if not isinstance(state, dict):
        return ex
    state = dict(state)
    for key in ("image",):
        if isinstance(state.get(key), str):
            state[key] = read(state[key])
    if state.get("images"):
        state["images"] = [read(p) for p in state["images"]]
    out = dict(ex, state=state)
    if ex.get("history"):  # a game frame's previous screens
        out["history"] = [read(p) for p in ex["history"]]
    return out


def pool_selection(kind: str, name: str) -> List[Dict]:
    """The examples (with image paths) that go into pool part ``kind`` for dataset ``name``, deterministically."""
    import random

    from laya.vlm_train import load_jsonl_examples

    rng = random.Random("%d:%s:%s" % (SEED, kind, name))
    if kind == "games":
        root, prepared = GAME_DATASETS[name]
        # next_target and history from the whole split, before sampling; the list, and so the seeded sample, is
        # v2's (experthist replays expert record for record)
        exs = [dict(ex, dataset=name) for ex in load_jsonl_examples(root, prepared, "train", next_targets=True,
                                                                      history=POOL_HISTORY)]
        return rng.sample(exs, min(TRAIN_POOL_PER_SET, len(exs)))
    if kind == "eval":
        exs = load_split(name, "val")
        return rng.sample(exs, min(EVAL_PER_SET, len(exs)))
    exs = load_split(name, "train")
    if kind == "calib":
        return exs[-N_CALIB:]
    exs = exs[:-N_CALIB] if name in CALIB_DATASETS else exs
    return rng.sample(exs, min(TRAIN_POOL_PER_SET, len(exs)))


def pool_dir(kind: str) -> str:
    """The pool directory that holds part kind ``kind`` (``POOL_KIND_VERSIONS``)."""
    return os.path.join(POOL_ROOT, "pool-" + POOL_KIND_VERSIONS[kind])


def _pool_file(kind: str, name: str) -> str:
    return os.path.join(pool_dir(kind), kind, name + ".pkl")


POOL_PARTS = ([("train", n) for n in TRAINABLE_DATASETS] + [("calib", n) for n in CALIB_DATASETS]
              + [("eval", n) for n in EVAL_DATASETS] + [("games", n) for n in GAME_DATASETS])


def load_pool(kinds=("train", "calib", "eval", "games")) -> Dict[str, Dict[str, List[Dict]]]:
    """``{kind: {dataset: examples}}`` from the pool; fails with the command to build it when it is missing."""
    import pickle
    from concurrent.futures import ThreadPoolExecutor

    parts = [(k, n) for k, n in POOL_PARTS if k in kinds]
    missing = [_pool_file(k, n) for k, n in parts if not os.path.exists(_pool_file(k, n))]
    if missing:
        raise FileNotFoundError("the autoresearch data pool %s is incomplete (%d parts missing, e.g. %s); build it "
                                "with: modal run autoresearch/harness.py --prepare-pool" % (POOL_VERSION, len(missing), missing[0]))

    def read(part):
        with open(_pool_file(*part), "rb") as f:
            return part, pickle.load(f)

    out: Dict[str, Dict[str, List[Dict]]] = {k: {} for k in kinds}
    with ThreadPoolExecutor(16) as pool:  # a few large sequential reads each; a handful in flight hides latency
        for (kind, name), exs in pool.map(read, parts):
            out[kind][name] = exs
    return out


def game_states(examples: List[Dict], frames: str, family: str) -> List[Dict]:
    """Pool game examples (encoded ``state["image"]`` plus ``"history"``, the previous screens) with their state in
    frame mode ``frames`` and no ``history`` key. ``single`` for Atari and Doom is the frame alone, so the examples
    are then the pool's, untouched."""
    from concurrent.futures import ThreadPoolExecutor

    from laya import frames as F

    kind, n = F.resolve(frames, family)
    if n == 1:
        return [{k: v for k, v in ex.items() if k != "history"} for ex in examples]
    if any(ex.get("history") is None for ex in examples):
        raise ValueError("these game examples carry no frame history (pool %s); %r needs it" % (pool_dir("games"),
                                                                                               frames))

    def png(arr):
        import io

        from PIL import Image

        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="PNG", compress_level=1)
        return buf.getvalue()

    def one(ex):
        st = F.state(list(ex["history"]) + [ex["state"]["image"]], frames, family,
                     encode=png if kind == "trail" else None)
        rest = {k: v for k, v in ex["state"].items() if k != "image"}
        return dict({k: v for k, v in ex.items() if k != "history"}, state=dict(rest, **st))

    if kind == "trail":
        with ThreadPoolExecutor(16) as pool:  # decode, blend and encode; PIL and numpy release the GIL
            return list(pool.map(one, examples))
    return [one(ex) for ex in examples]


class Context:
    """What an experiment gets: the budget, the device, checkpoint lookup and training data. Training data never
    includes the val splits or the calibration tail the harness fits temperatures on."""

    def __init__(self, time_budget_s: float, train: Dict[str, List[Dict]], games: Dict[str, List[Dict]]):
        self.time_budget_s = time_budget_s
        self.device = "cuda"
        self.ckpt_path = ckpt_path
        self._train = train  # name -> train records minus the calibration tail
        self._games = games  # name -> expert game frames

    def game_examples(self, names=tuple(GAME_DATASETS), frames: str = "single") -> List[Dict]:
        """Expert frames from the data pool: Atari Freeway and Breakout (the expert agents' action distributions as
        soft targets where recorded) and ViZDoom basic (the scripted labeller). Each frame whose next step of the same
        episode was recorded carries that step's target as ``next_target`` (for a ``"next_head"`` model).
        ``frames`` is the ``laya.frames`` game frame mode of the states, built from each frame's recorded history
        (a trail is blended here, once per call, and PNG-encoded). ``toolkit.py`` generates Maze, Snake and
        classic-control examples on the fly."""
        out = []
        for name in names:
            if name not in GAME_DATASETS:
                raise ValueError("%s is not a game dataset here (one of %s)" % (name, tuple(GAME_DATASETS)))
            out += game_states(self._games[name], frames, GAME_FAMILY[name])
        return out

    def train_examples(self, names=TRAINABLE_DATASETS) -> List[Dict]:
        out = []
        for name in names:
            if name not in TRAINABLE_DATASETS:
                raise ValueError("%s is not trainable here (one of %s)" % (name, TRAINABLE_DATASETS))
            out += self._train[name]  # a new list each call; the records are shared (one run per container)
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


def _log(t_start: float, msg: str) -> None:
    print("[harness %6.1f s] %s" % (time.time() - t_start, msg), flush=True)


@app.cls(image=image, gpu="H100", cpu=16, memory=65536, timeout=60 * 60, volumes=VOLUMES,
         enable_memory_snapshot=True, single_use_containers=True)
class TrainEval:
    @modal.enter(snap=True)
    def load(self):
        """Experiment-independent CPU state, kept in the memory snapshot. No CUDA here: there is no GPU yet."""
        t = time.time()
        import random  # noqa: F401

        import torch  # noqa: F401
        import transformers  # noqa: F401

        import laya.vlm  # noqa: F401
        import laya.vlm_train  # noqa: F401
        _log(t, "imports")

        self.pool = load_pool()
        _log(t, "data pool: %s" % ", ".join("%s %d examples" % (k, sum(len(v) for v in d.values()))
                                            for k, d in self.pool.items()))
        self.snap_s = time.time() - t

    @modal.enter(snap=False)
    def restore(self):
        _log(time.time(), "restored (snapshot built in %.1f s)" % self.snap_s)

    def _train_lists(self) -> Dict[str, List[Dict]]:
        return self.pool["train"]

    def _calib(self) -> List[Dict]:
        return [ex for name in CALIB_DATASETS for ex in self.pool["calib"][name]]

    def _val(self) -> List[Dict]:
        return [ex for name in EVAL_DATASETS for ex in self.pool["eval"][name]]

    @modal.method()
    def run(self, source: str, tag: str, commit: str) -> Dict:
        import random

        import torch

        from laya.vlm_train import collect_logits, fit_temperatures_from, metrics_from

        t_run = time.time()
        torch.manual_seed(SEED)
        random.seed(SEED)
        exp = _import_experiment(source)
        ctx = Context(TIME_BUDGET, self._train_lists(), self.pool["games"])
        t_setup = time.time()
        agent = exp.build(ctx)
        setup_s = time.time() - t_setup
        _log(t_run, "build %.1f s" % setup_s)
        t0 = time.time()
        exp.train(agent, ctx)
        train_s = time.time() - t0
        if train_s > TIME_BUDGET + 60:
            raise RuntimeError("training took %.0f s, over the %d s budget" % (train_s, TIME_BUDGET))
        agent.model.eval()

        temps = fit_temperatures_from(collect_logits(agent.model, agent.processor, self._calib(), batch_size=32,
                                                     num_workers=8))
        agent.temperature, agent.temperature_by_options = list(temps), {}

        out_dir = os.path.join(ROOT, tag, commit)
        agent.save(out_dir)
        ckpt_vol.commit()
        del agent
        torch.cuda.empty_cache()

        from laya.vlm import VLMAgent

        agent = VLMAgent(out_dir, device="cuda")
        params = sum(p.numel() for p in agent.model.parameters())
        records = collect_logits(agent.model, agent.processor, self._val(), batch_size=32, num_workers=12)
        metrics = metrics_from(records, agent.temperature)
        # one right answer (one-hot target): ECE is meaningful there, not against a spread of human votes
        hard = metrics_from([r for r in records if float(r["target"].max()) >= 0.999], agent.temperature)
        summary = _summary(metrics, hard)
        summary["params_m"] = params / 1e6
        res = {"tag": tag, "commit": commit, "summary": summary, "metrics": metrics, "temperature": temps,
               "setup_s": round(setup_s, 1), "train_s": round(train_s, 1), "checkpoint": out_dir,
               "config": {k: v for k, v in agent.cfg.items() if isinstance(v, (str, int, float, bool)) or v is None}}
        print(json.dumps(summary))
        _log(t_run, "done")
        return res


def _latency_cases(evals: Dict[str, List[Dict]]) -> List:
    """The first ``ceil(LATENCY_N / len(EVAL_DATASETS))`` single-image examples of each eval set's pool part, decoded,
    cut to ``LATENCY_N``: the same fixed images for every experiment."""
    import io

    from PIL import Image

    per = -(-LATENCY_N // len(EVAL_DATASETS))
    cases = []
    for name in EVAL_DATASETS:
        exs = [ex for ex in evals[name] if isinstance(ex["state"], dict) and ex["state"].get("image") is not None]
        for ex in exs[:per]:
            state = dict(ex["state"])
            with Image.open(io.BytesIO(state["image"])) as im:
                state["image"] = im.convert("RGB")
            q = ex["q"]
            crit = list(q["crit"]) if q["t"] == "choice" else q["crit"]
            cases.append((state, {"q": {"type": q["t"], "instructions": q["ins"], "criteria": crit}}))
    return cases[:LATENCY_N]


def _latency_game_frames() -> List:
    """The fixed frame history the game move is timed on: CartPole (seed 0) under its scripted expert, one frame per
    decision, ``LATENCY_MOVE_WARMUP + LATENCY_MOVES`` of them (the expert keeps the pole up, so the episode lasts)."""
    from laya.controlgames import ControlGame

    g = ControlGame("CartPole", 0)
    frames = []
    for _ in range(LATENCY_MOVE_WARMUP + LATENCY_MOVES):
        frames.append(g.frame())
        g.step(g.expert())
    g.close()
    return frames


def time_game_moves(cand, ref, frames: List) -> Dict:
    """Median time of one CartPole move at batch 1, each checkpoint in its own frame mode (the reference in
    ``single``), as play makes it (``games_eval.move_fn``: state from the frame history, e.g. a trail's blend, then
    the forward; stack-N keeps its frame cache across moves, so after the first N only the new frame is encoded).
    Move ``i`` sees the history ``frames[:i + 1]``; the two checkpoints alternate per move like the questions. The
    first ``LATENCY_MOVE_WARMUP`` moves are untimed (they also fill the cache)."""
    import numpy as np
    import torch

    import games_eval
    from laya import frames as F
    from laya.games import control_question

    moves = {}
    for name, agent in (("cand", cand), ("ref", ref)):
        mode = F.mode_for(agent.cfg, "control")
        moves[name] = (mode, games_eval.move_fn(agent, control_question("CartPole", mode), mode, "control"))

    def timed(name, i):
        torch.cuda.synchronize()
        t = time.perf_counter()
        moves[name][1](frames[max(0, i + 1 - F.MAX_FRAMES):i + 1])
        torch.cuda.synchronize()
        return (time.perf_counter() - t) * 1000

    ms, ref_ms = [], []
    for i in range(len(frames)):
        pair = [("cand", ms), ("ref", ref_ms)] if i % 2 == 0 else [("ref", ref_ms), ("cand", ms)]
        for name, out in pair:
            t = timed(name, i)
            if i >= LATENCY_MOVE_WARMUP:
                out.append(t)
    return {"game_move_x": float(np.median(ms) / np.median(ref_ms)), "game_move_ms": float(np.median(ms)),
            "game_move_ref_ms": float(np.median(ref_ms)), "game_move_p90_ms": float(np.percentile(ms, 90)),
            "game_frames": moves["cand"][0], "game_moves": len(ms)}


@app.cls(image=image, gpu="L4", cpu=4, memory=16384, timeout=20 * 60, volumes=VOLUMES,
         enable_memory_snapshot=True, single_use_containers=True)
class Latency:
    @modal.enter(snap=True)
    def load(self):
        t = time.time()
        import numpy  # noqa: F401
        import torch  # noqa: F401

        import laya.vlm  # noqa: F401
        import laya.vlm_train  # noqa: F401

        self.cases = _latency_cases(load_pool(("eval",))["eval"])
        self.game_frames = _latency_game_frames()
        _log(t, "imports, %d latency cases and %d game frames" % (len(self.cases), len(self.game_frames)))

    @modal.method()
    def run(self, tag: str, commit: str) -> Dict:
        """Median ``predict`` time of the experiment's checkpoint, as a ratio to the base checkpoint's: both are
        loaded here and timed alternately on each case, so the L4 host's speed (raw times vary by ~50% between
        hosts) cancels out. Then the same for one game move in each checkpoint's frame mode (``time_game_moves``);
        ``latency_x`` is the larger of the two ratios."""
        import numpy as np
        import torch

        from laya.vlm import VLMAgent

        ckpt_vol.reload()
        cand = VLMAgent(os.path.join(ROOT, tag, commit), device="cuda", dtype="bf16")
        ref = VLMAgent(ckpt_path(REFERENCE), device="cuda", dtype="bf16")
        cases = self.cases
        for state, qs in cases[:10]:
            cand.predict(state, qs)
            ref.predict(state, qs)

        def timed(agent, state, qs):
            torch.cuda.synchronize()
            t = time.perf_counter()
            agent.predict(state, qs)
            torch.cuda.synchronize()
            return (time.perf_counter() - t) * 1000

        ms, ref_ms = [], []
        for i, (state, qs) in enumerate(cases):
            pair = [(cand, ms), (ref, ref_ms)] if i % 2 == 0 else [(ref, ref_ms), (cand, ms)]
            for agent, out in pair:
                out.append(timed(agent, state, qs))
        q_x = float(np.median(ms) / np.median(ref_ms))
        game = time_game_moves(cand, ref, self.game_frames)
        return {"latency_x": max(q_x, game["game_move_x"]), "latency_q_x": q_x, "latency_ms": float(np.median(ms)),
                "latency_ref_ms": float(np.median(ref_ms)), "latency_p90_ms": float(np.percentile(ms, 90)),
                "n": len(ms), "gpu": torch.cuda.get_device_name(0), **game}


@app.cls(image=games_image, gpu="L4", cpu=16, memory=32768, timeout=30 * 60, volumes=VOLUMES,
         enable_memory_snapshot=True, single_use_containers=True)
class Games:
    @modal.enter(snap=True)
    def load(self):
        import torch  # noqa: F401

        import games_eval  # noqa: F401
        import laya.vlm  # noqa: F401

    @modal.method()
    def run(self, tag: str, commit: str, family: str) -> Dict:
        """``games_eval.run_family`` on the experiment's saved checkpoint: ``{game: result}`` for ``family``."""
        import games_eval

        from laya.vlm import VLMAgent

        ckpt_vol.reload()
        t = time.time()
        agent = VLMAgent(os.path.join(ROOT, tag, commit), device="cuda", dtype="bf16")
        out = games_eval.run_family(agent, family)
        _log(t, "games %s: %s" % (family, ", ".join("%s %.3f" % (g, r["normalized"]) for g, r in out.items())))
        return out


@app.function(image=image, cpu=4, memory=8192, timeout=60 * 60, volumes={"/data": data_vol})
def build_pool_part(kind: str, name: str) -> Dict:
    """One pool part: the selected examples with their image bytes inline, pickled to the pool directory."""
    import pickle
    from concurrent.futures import ThreadPoolExecutor

    t = time.time()
    path = _pool_file(kind, name)
    if os.path.exists(path):
        return {"kind": kind, "name": name, "examples": -1, "mb": round(os.path.getsize(path) / 1e6, 1), "seconds": 0.0}
    if POOL_KIND_VERSIONS[kind] != POOL_VERSION:  # an older version's part: immutable, never rebuilt by newer code
        raise FileNotFoundError("%s is missing; %s parts belong to pool-%s, which this harness (pool-%s) does not "
                                "rebuild" % (path, kind, POOL_KIND_VERSIONS[kind], POOL_VERSION))
    exs = pool_selection(kind, name)
    with ThreadPoolExecutor(64) as pool:  # each small-file read is ~0.4 s of latency; overlap many
        shared: Dict[str, bytes] = {} if kind == "games" else None
        exs = list(pool.map(lambda ex: _with_bytes(ex, shared), exs))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "wb") as f:
        pickle.dump(exs, f, protocol=5)
    os.replace(path + ".tmp", path)
    data_vol.commit()
    out = {"kind": kind, "name": name, "examples": len(exs), "mb": round(os.path.getsize(path) / 1e6, 1),
           "seconds": round(time.time() - t, 1)}
    if kind == "games":  # previous frames per example: POOL_HISTORY, fewer only near an episode's start / an auto-FIRE
        from collections import Counter

        out["history"] = dict(sorted(Counter(len(ex.get("history") or []) for ex in exs).items()))
    return out


def build_pool():
    """Build every missing pool part, all in parallel (``modal run autoresearch/harness.py --prepare-pool``)."""
    total = 0.0
    for r in build_pool_part.starmap(POOL_PARTS, order_outputs=False, return_exceptions=True):
        if isinstance(r, Exception):
            print("FAILED:", repr(r)[:300])
            continue
        total += r["mb"]
        print("%-6s %-28s %6d examples %8.1f MB %6.1f s%s" % (r["kind"], r["name"], r["examples"], r["mb"], r["seconds"],
                                                           "  history lengths %s" % r["history"] if "history" in r else ""))
    print("pool %s: %.1f GB (%s)" % (POOL_VERSION, total / 1e3, ", ".join(
        "%s from pool-%s" % kv for kv in POOL_KIND_VERSIONS.items())))


@app.function(image=image, timeout=10 * 60, volumes={"/ckpt": ckpt_vol})
def prune_checkpoints(tag: str, prunable: List[str]) -> List[str]:
    """Delete this tag's saved checkpoints whose commit is in ``prunable`` (``pareto.prunable``: finished in
    results.tsv and off the frontier); returns what was removed. Any other directory, such as a concurrent run's
    fresh checkpoint that is not in the TSV yet, is left alone."""
    import shutil

    ckpt_vol.reload()
    base = os.path.join(ROOT, tag)
    removed = []
    for name in sorted(os.listdir(base)) if os.path.isdir(base) else []:
        if name in prunable and os.path.exists(os.path.join(base, name, "vlm_agent_config.json")):
            shutil.rmtree(os.path.join(base, name))
            removed.append(name)
    ckpt_vol.commit()
    return removed


def _code_hash() -> str:
    """What the containers run: this file, the games benchmark and toolkit, and the shipped ``laya`` package."""
    import hashlib

    h = hashlib.sha256()
    files = [os.path.abspath(__file__)] + [os.path.join(REPO, "autoresearch", f) for f in HARNESS_FILES]
    for d, _, names in sorted(os.walk(os.path.join(REPO, "laya"))):
        files += [os.path.join(d, n) for n in sorted(names) if n.endswith(".py")]
    for path in files:
        h.update(os.path.relpath(path, REPO).encode() + b"\0")
        with open(path, "rb") as f:
            h.update(f.read() + b"\0")
    return h.hexdigest()[:10]


def deployment_name() -> str:
    return "%s-%s" % (app.name, _code_hash())


def follow_logs(name: str):
    """Stream the deployment's container logs into this process's output (a deployed app's logs do not reach
    ``modal run`` on their own). Returns the process to terminate when the run is over."""
    return subprocess.Popen([sys.executable, "-m", "modal", "app", "logs", name, "-f"], cwd=REPO)


def deployed_classes():
    """``(TrainEval, Latency, Games)`` from a deployment of exactly this code, deploying it first if needed.

    Modal only snapshots deployed apps, so ``modal run`` alone would rebuild everything every time. Each version of
    the code gets its own app, ``laya-autoresearch-<hash>``: redeploying unchanged code is quick and keeps its
    snapshot, and concurrent runs from different code never call each other's deployment.
    """
    name = deployment_name()
    try:
        te = modal.Cls.from_name(name, "TrainEval")
        te.hydrate()
    except modal.exception.NotFoundError:
        cmd = [sys.executable, "-m", "modal", "deploy", os.path.abspath(__file__), "--name", name]
        p = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
        if p.returncode:
            raise RuntimeError("modal deploy failed:\n" + p.stdout + p.stderr)
        te = modal.Cls.from_name(name, "TrainEval")
    return te, modal.Cls.from_name(name, "Latency"), modal.Cls.from_name(name, "Games")


def _git(*args) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True).stdout.strip()


@app.local_entrypoint()
def main(tag: str = "", desc: str = "", prune: bool = True, prepare_pool: bool = False):
    import pareto

    if prepare_pool:
        build_pool()
        return
    if not tag:
        raise SystemExit("--tag is required (the run tag, e.g. sep23)")

    exp_path = os.path.join(REPO, "autoresearch", "experiment.py")
    if _git("status", "--porcelain", "--", exp_path):
        raise SystemExit("commit autoresearch/experiment.py first: results are keyed by commit")
    commit = _git("rev-parse", "--short=7", "HEAD")
    desc = desc or _git("log", "-1", "--format=%s")
    runs = os.path.join(REPO, "autoresearch", "runs", tag)
    tsv = os.path.join(runs, "results.tsv")
    os.makedirs(runs, exist_ok=True)
    # results are only comparable under one harness: this file, the laya package and the pool it reads
    version, pinned = _code_hash(), os.path.join(runs, "harness.txt")
    if os.path.exists(pinned) and open(pinned).read().strip() != version:
        raise SystemExit("the harness (autoresearch/harness.py or laya/) changed since tag %r started (%s -> %s), so its "
                         "results would not be comparable; start a new tag" % (tag, open(pinned).read().strip(), version))
    with open(pinned, "w") as f:
        f.write(version + "\n")
    with open(exp_path) as f:
        source = f.read()
    t0 = time.time()
    train_eval, latency, games = deployed_classes()  # a failed deploy is not the experiment's crash
    logs = follow_logs(deployment_name())
    try:
        import games_eval

        res = train_eval().run.remote(source, tag, commit)
        # the latency job and one games job per family, all on the saved checkpoint, at once
        lat = latency().run.spawn(tag, commit)
        fams = {f: games().run.spawn(tag, commit, f) for f in games_eval.FAMILIES}
        res["summary"].update({k: v for k, v in lat.get().items() if k.startswith(("latency", "game_move"))})
        played = {}
        for f, call in fams.items():
            played.update(call.get())
        g = games_eval.summarize(played)
        if not g["complete"]:
            raise RuntimeError("games benchmark incomplete, missing %s" % g["missing"])
        res["summary"]["games"] = g["games"]
        res["games"] = {"per_game": g["per_game"], "results": played}
    except Exception as e:
        print("crash: %r" % (e,))
        pareto.append_tsv(tsv, pareto.crash_row(commit, desc))
        print("status: crash")
        raise SystemExit(1)
    finally:
        time.sleep(3)  # let the last lines arrive
        logs.terminate()
    res["harness"] = version
    res.update(description=desc, total_s=round(time.time() - t0, 1))
    out = os.path.join(runs, commit + ".json")
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    s = res["summary"]
    print("---")
    for k in ("quality", "macro_acc", "ece_hard", "games", "params_m", "latency_x", "latency_q_x", "latency_ms",
              "latency_ref_ms", "game_move_x", "game_move_ms", "game_move_ref_ms"):
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
        # only commits the TSV has finished with and that are off the frontier: a concurrent run's checkpoint is
        # not in the TSV until that run decides, so it is never touched
        removed = prune_checkpoints.remote(tag, pareto.prunable(pareto.read_tsv(tsv)))
        if removed:
            print("pruned checkpoints off the frontier: %s" % ", ".join(removed))
