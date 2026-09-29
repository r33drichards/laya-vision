"""The fixed autoresearch harness for Laya Vision: train an experiment for 15 minutes, then measure it.

    modal run autoresearch/harness.py --tag <tag> [--desc "what this experiment tries"]
    modal run autoresearch/harness.py --tag bigym-<date> --profile bigym --experiment autoresearch/experiment_bigym.py

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
4. counts its parameters and times ``predict`` on an L4 in bf16 on ``LATENCY_N`` fixed images, alternating with the
   ``REFERENCE`` checkpoint in the same container.

5. plays the games benchmark (``games_eval.py``: Maze, Snake, classic control, Atari Freeway and Breakout, ViZDoom
   basic, fixed seeds) with the reloaded checkpoint, one L4 container per game family, alongside the latency job.

The four objectives (see ``pareto.py``):

* **quality** = macro accuracy over the eval sets minus the ECE pooled over the questions that have a single right
  answer (not the ones scored against human vote spreads, where ECE is not meaningful). Higher is better.
* **games** = mean normalized game score, per game (model - random) / (expert - random) clipped to [-0.5, 1.5]
  against the fixed baselines in ``game_baselines.json``. Higher is better.
* **params_m**: parameters of the saved model, in millions. Lower is better.
* **latency_x**: median ``predict`` time on the L4 (preprocessing included) divided by the ``REFERENCE``
  checkpoint's, timed alternately in the same container: raw milliseconds swing by ~50% between L4 hosts (52 vs
  76 ms for one model). Lower is better; the raw times are kept in the result JSON.

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
  frames as v2; ``train``, ``calib`` and ``eval`` are read unchanged from ``pool-v2``;
* (``v4``: a ``pool-v4/games`` directory exists on the volume, written on 2026-09-24 by code that was never
  committed; no harness reads it, and the version number is skipped;)
* ``v5``: adds the BiGym kinds, read only by the ``bigym`` profile: ``bigym``, every train record of the cleaned
  BiGym behaviour-cloning and probe sets (``BIGYM_DATASETS``; ``bc_f4`` records carry four images, stored once per
  distinct frame), and ``bigym_demos``, the waypoints of BiGym's human demos per cupboard task
  (``laya.bigymdemos.demo_waypoints``, the train-split demos that succeed; the sets' val demos are left out) for
  experiments that relabel their own rollouts with the lookahead follower. The other kinds keep their versions.

Profiles (``--profile``). ``default`` is the four-objective Pareto loop above. ``bigym`` (``program_bigym.md``) makes
BiGym control the objective with quality and games as guard-rails (``pareto.decide_bigym``): the experiment trains
on an H100 image that also has MuJoCo and BiGym (``TrainEvalBigym``: the same budget, calibration, save / reload and
quality eval, plus ``ctx.bigym_examples()``, ``ctx.bigym_demos()`` and ``ctx.bigym_game()`` for rollouts on training
seeds), then the BiGym benchmark (``bigym_eval.py``, one L4 container per task chunk, ``BiGym``) runs alongside the
latency and games jobs and its ``bigym`` score is merged into the result. ``--experiment`` picks the experiment file
(default ``autoresearch/experiment.py``); the profile and the file are recorded in the result JSON (and, for the
bigym profile, in results.tsv, whose columns are ``pareto.BIGYM_COLUMNS``). A tag keeps one profile.

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
POOL_VERSION = "v5"         # the newest pool version (history in the module docstring; v4 is skipped)
# The version whose directory holds each kind's parts: v3 rebuilt only ``games`` (adding next_target), v5 added the
# BiGym kinds.
POOL_KIND_VERSIONS = {"train": "v2", "calib": "v2", "eval": "v2", "games": "v3", "bigym": "v5", "bigym_demos": "v5"}
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
    "game_atari_freeway": ("/data/atari/expert", "Freeway"),
    "game_atari_breakout": ("/data/atari/expert", "Breakout"),
    "game_doom_basic": ("/data/vqa", "doom_basic"),
}

# BiGym (the ``bigym`` profile): the cleaned behaviour-cloning / probe sets' train splits (their val splits stay
# unused), every record in a seeded order, and the cupboard tasks whose human demos' waypoints go in the pool.
BIGYM_DATASETS = ("bigym_v2c_bc_f4", "bigym_v2c_bc_f1", "bigym_v2c_probe")
BIGYM_POOL_PER_SET = 20000   # more than any of the sets has (15,206 / 15,206 / 16,943): all of them
BIGYM_DEMO_TASKS = ("DrawerTopOpen", "DrawerTopClose", "WallCupboardOpen", "WallCupboardClose")
BIGYM_VAL_SET = "bigym_v2c_bc_f1"   # its val split names the demo seeds held out (same split for all three sets)
# the same MuJoCo / BiGym pins as modal_bigym.py (tests/test_autoresearch_bigym.py checks they match)
BIGYM = "git+https://github.com/r33drichards/bigym@14beb30318ad14c5d6723175c2ee2281129792af"
MUJOCO = "3.14.0"
PROFILES = ("default", "bigym")

app = modal.App("laya-autoresearch")
hf_vol = modal.Volume.from_name("laya-hf-cache")
data_vol = modal.Volume.from_name("laya-datasets")
ckpt_vol = modal.Volume.from_name("laya-checkpoints")
# The harness's own modules next to the laya package in every container: the games benchmark (and its fixed
# baselines) and the toolkit experiments import to generate game training data.
HARNESS_FILES = ("games_eval.py", "game_baselines.json", "toolkit.py", "bigym_eval.py", "bigym_baselines.json")


def _harness_files() -> List[str]:
    """The ``HARNESS_FILES`` that exist (``bigym_baselines.json`` does not until it is measured)."""
    return [f for f in HARNESS_FILES if os.path.exists(os.path.join(REPO, "autoresearch", f))]


def _with_code(img):
    img = img.add_local_python_source("laya")
    for f in _harness_files():
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
# MuJoCo, BiGym (installed without its pins, as modal_bigym.py does: its safetensors pin clashes with transformers 5)
# and headless GL, plus BiGym's human demos (~/.bigym) so nothing downloads inside a timed run
_bigym_base = (
    _base.apt_install("git", "libegl1", "libgl1", "libosmesa6", "libglib2.0-0")
    .pip_install("mujoco==" + MUJOCO, "dm_control==1.0.47", "mojo-mujoco-wrapper==0.1.1", "mujoco-utils==0.0.6",
                 "pyquaternion==0.9.9", "numpy-quaternion==2024.0.13", "imageio", "pyyaml", "wget==3.2", "tqdm")
    .pip_install(BIGYM, extra_options="--no-deps")
    .env({"MUJOCO_GL": "osmesa", "NVIDIA_DRIVER_CAPABILITIES": "all"})
    .run_commands("python -c 'from demonstrations.demo_store import DemoStore; DemoStore().pull_demos()'")
)
bigym_image = _with_code(_bigym_base)
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


def _with_bytes(ex: Dict, files: Optional[Dict[str, bytes]] = None) -> Dict:
    """An example whose image paths are replaced by the files' bytes (``laya.vlm`` loads either): ``"image"`` and
    every entry of an ``"images"`` list. ``files`` (path -> bytes), when given, supplies them instead of reading,
    so a frame shared by several examples is one bytes object (pickled once)."""
    state = ex["state"]
    if not isinstance(state, dict):
        return ex

    def read(p):
        if files is not None and p in files:
            return files[p]
        with open(p, "rb") as f:
            return f.read()

    state = dict(state)
    if isinstance(state.get("image"), str):
        state["image"] = read(state["image"])
    if state.get("images"):
        state["images"] = [read(p) if isinstance(p, str) else p for p in state["images"]]
    return dict(ex, state=state)


def _image_paths(ex: Dict) -> List[str]:
    state = ex["state"]
    if not isinstance(state, dict):
        return []
    return ([state["image"]] if isinstance(state.get("image"), str) else []) + \
        [p for p in state.get("images") or [] if isinstance(p, str)]


def pool_selection(kind: str, name: str) -> List[Dict]:
    """The examples (with image paths) that go into pool part ``kind`` for dataset ``name``, deterministically."""
    import random

    from laya.vlm_train import load_jsonl_examples

    rng = random.Random("%d:%s:%s" % (SEED, kind, name))
    if kind == "games":
        root, prepared = GAME_DATASETS[name]
        # next_target from the whole split, before sampling; the list, and so the seeded sample, is v2's
        exs = [dict(ex, dataset=name) for ex in load_jsonl_examples(root, prepared, "train", next_targets=True)]
        return rng.sample(exs, min(TRAIN_POOL_PER_SET, len(exs)))
    if kind == "bigym":
        exs = load_split(name, "train")
        if not exs:
            raise FileNotFoundError("/data/vqa/%s is not a prepared dataset" % name)
        return rng.sample(exs, min(BIGYM_POOL_PER_SET, len(exs)))
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
              + [("eval", n) for n in EVAL_DATASETS] + [("games", n) for n in GAME_DATASETS]
              + [("bigym", n) for n in BIGYM_DATASETS] + [("bigym_demos", t) for t in BIGYM_DEMO_TASKS])
DEFAULT_KINDS = ("train", "calib", "eval", "games")
BIGYM_KINDS = DEFAULT_KINDS + ("bigym", "bigym_demos")


def load_pool(kinds=DEFAULT_KINDS) -> Dict[str, Dict[str, List[Dict]]]:
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


class Context:
    """What an experiment gets: the budget, the device, checkpoint lookup and training data. Training data never
    includes the val splits or the calibration tail the harness fits temperatures on. Under the ``bigym`` profile
    it also has the BiGym pool (``bigym_examples``, ``bigym_demos``) and the simulator (``bigym_game``)."""

    def __init__(self, time_budget_s: float, train: Dict[str, List[Dict]], games: Dict[str, List[Dict]],
                 bigym: Optional[Dict[str, List[Dict]]] = None, bigym_demos: Optional[Dict[str, List[Dict]]] = None):
        self.time_budget_s = time_budget_s
        self.device = "cuda"
        self.ckpt_path = ckpt_path
        self._train = train  # name -> train records minus the calibration tail
        self._games = games  # name -> expert game frames
        self._bigym = bigym  # name -> BiGym behaviour-cloning / probe train records (bigym profile only)
        self._bigym_demos = bigym_demos  # cupboard task -> its train demos' waypoints (bigym profile only)
        self._gl = None

    def _need_bigym(self):
        if self._bigym is None:
            raise RuntimeError("BiGym data and the simulator are only available under --profile bigym")

    def bigym_examples(self, names=BIGYM_DATASETS) -> List[Dict]:
        """The cleaned BiGym sets' train records (``BIGYM_DATASETS``): ``bigym_v2c_bc_f4`` (the control question
        over the last four head frames, ``state["images"]``), ``bigym_v2c_bc_f1`` (one frame) and
        ``bigym_v2c_probe`` (done / progress / side questions); images inline as encoded bytes, ``dataset`` =
        the set's name. Their val splits are not here and must stay unused."""
        self._need_bigym()
        out = []
        for name in names:
            if name not in BIGYM_DATASETS:
                raise ValueError("%s is not a BiGym set here (one of %s)" % (name, BIGYM_DATASETS))
            out += self._bigym[name]
        return out

    def bigym_demos(self, task: str) -> List[Dict]:
        """BiGym's human demos of a cupboard task as ``laya.bigymdemos.demo_waypoints`` gives them (``seed``,
        ``waypoints``, ``part``, ...), the train-split demos that succeed: what ``laya.bigymdemos.follow`` and
        ``lookahead`` need to label states (DAgger). Rollouts on a demo's own seed start where the demo did."""
        self._need_bigym()
        if task not in BIGYM_DEMO_TASKS:
            raise ValueError("no demos for %s (one of %s)" % (task, BIGYM_DEMO_TASKS))
        return self._bigym_demos[task]

    def bigym_seed_ok(self, seed: int) -> bool:
        """Whether an experiment may roll out on ``seed``: below ``bigym_eval.TRAIN_SEED_MAX`` or a train demo's."""
        import bigym_eval

        seed = int(seed)
        if bigym_eval.is_eval_seed(seed):
            return False
        return seed < bigym_eval.TRAIN_SEED_MAX or any(seed == d["seed"] for ds in self._bigym_demos.values()
                                                       for d in ds)

    def bigym_game(self, task: str, seed: int, cameras: bool = True, env=None):
        """A ``laya.bigymgames.BiGymGame`` on a training seed (see ``bigym_seed_ok``; the eval's seeds raise), with
        a headless GL picked on first use. Pass ``env`` (``laya.bigymgames.make_env``, after one ``bigym_game``
        call) to reuse an environment across episodes: making one takes seconds."""
        self._need_bigym()
        if not self.bigym_seed_ok(seed):
            raise ValueError("seed %d is off-limits for training rollouts (use seeds below %d or a train demo's)"
                             % (seed, __import__("bigym_eval").TRAIN_SEED_MAX))
        if self._gl is None:
            import bigym_eval

            self._gl = bigym_eval.pick_gl()
        from laya import bigymgames as bg

        return bg.BiGymGame(task, int(seed), env=env or bg.make_env(task, cameras=cameras))

    def game_examples(self, names=tuple(GAME_DATASETS)) -> List[Dict]:
        """Expert frames from the data pool: Atari Freeway and Breakout (the expert agents' action distributions as
        soft targets where recorded) and ViZDoom basic (the scripted labeller). Each frame whose next step of the same
        episode was recorded carries that step's target as ``next_target`` (for a ``"next_head"`` model).
        ``toolkit.py`` generates Maze, Snake and classic-control examples on the fly."""
        out = []
        for name in names:
            if name not in GAME_DATASETS:
                raise ValueError("%s is not a game dataset here (one of %s)" % (name, tuple(GAME_DATASETS)))
            out += self._games[name]
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


class _TrainEvalBase:
    """The training and quality job; ``TrainEval`` (default profile) and ``TrainEvalBigym`` differ only in their
    image, the pool kinds they hold and the context they hand the experiment."""
    KINDS = DEFAULT_KINDS

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

        self.pool = load_pool(self.KINDS)
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

    def _context(self) -> Context:
        return Context(TIME_BUDGET, self._train_lists(), self.pool["games"])

    @modal.method()
    def run(self, source: str, tag: str, commit: str) -> Dict:
        import random

        import torch

        from laya.vlm_train import collect_logits, fit_temperatures_from, metrics_from

        t_run = time.time()
        torch.manual_seed(SEED)
        random.seed(SEED)
        exp = _import_experiment(source)
        ctx = self._context()
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


@app.cls(image=image, gpu="H100", cpu=16, memory=65536, timeout=60 * 60, volumes=VOLUMES,
         enable_memory_snapshot=True, single_use_containers=True)
class TrainEval(_TrainEvalBase):
    """The default profile's training and quality job."""


@app.cls(image=bigym_image, gpu="H100", cpu=16, memory=98304, timeout=60 * 60, volumes=VOLUMES,
         enable_memory_snapshot=True, single_use_containers=True)
class TrainEvalBigym(_TrainEvalBase):
    """The bigym profile's: the same job with the BiGym pool kinds and MuJoCo / BiGym for rollouts in ``train``."""
    KINDS = BIGYM_KINDS

    def _context(self) -> Context:
        return Context(TIME_BUDGET, self._train_lists(), self.pool["games"], self.pool["bigym"],
                       self.pool["bigym_demos"])


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
        _log(t, "imports and %d latency cases" % len(self.cases))

    @modal.method()
    def run(self, tag: str, commit: str) -> Dict:
        """Median ``predict`` time of the experiment's checkpoint, as a ratio to the base checkpoint's: both are
        loaded here and timed alternately on each case, so the L4 host's speed (raw times vary by ~50% between
        hosts) cancels out."""
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
        return {"latency_x": float(np.median(ms) / np.median(ref_ms)), "latency_ms": float(np.median(ms)),
                "latency_ref_ms": float(np.median(ref_ms)), "latency_p90_ms": float(np.percentile(ms, 90)),
                "n": len(ms), "gpu": torch.cuda.get_device_name(0)}


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


@app.cls(image=bigym_image, gpu="L4", cpu=8, memory=32768, timeout=40 * 60, volumes=VOLUMES,
         enable_memory_snapshot=True, single_use_containers=True)
class BiGym:
    """The BiGym benchmark (``bigym_eval.py``) on one chunk of one task. MuJoCo is not imported before the GL is
    picked (EGL on the GPU), so the snapshot holds only torch and laya."""

    @modal.enter(snap=True)
    def load(self):
        import torch  # noqa: F401

        import bigym_eval  # noqa: F401
        import games_eval  # noqa: F401
        import laya.vlm  # noqa: F401

    @modal.method()
    def run(self, ckpt: str, task: str, chunk: int) -> Dict:
        """``bigym_eval.run_chunk`` with the checkpoint at ``ckpt`` (an absolute path on the checkpoint volume)."""
        import bigym_eval

        t = time.time()
        gl = bigym_eval.pick_gl()
        ckpt_vol.reload()
        from laya.vlm import VLMAgent

        agent = VLMAgent(ckpt, device="cuda", dtype="bf16")
        load_s = time.time() - t
        out = bigym_eval.run_chunk(agent, task, chunk)
        out.update(gl=gl, load_seconds=round(load_s, 1))
        _log(t, "bigym %s chunk %d: progress %.3f, success %.2f, %d forwards, %.0f s play (%.0f s envs, %.0f s "
                "policy)" % (task, chunk, sum(e["progress"] for e in out["episodes"]) / len(out["episodes"]),
                             sum(e["success"] for e in out["episodes"]) / len(out["episodes"]), out["forwards"],
                             out["seconds"], out["env_seconds"], out["policy_seconds"]))
        return out


@app.function(image=bigym_image, cpu=4, memory=16384, timeout=60 * 60)
def bigym_reference(task: str, policy: str) -> Dict:
    """``random`` or ``oracle`` (reach tasks) on the BiGym eval's episodes of ``task`` (no camera)."""
    import bigym_eval

    gl = bigym_eval.pick_gl()
    out = bigym_eval.play_reference(task, policy)
    out.update(gl=gl, policy=policy)
    return out


def run_bigym(ckpt: str, bigym_cls=None) -> Dict:
    """The whole BiGym benchmark on ``ckpt``: every (task, chunk) in its own container, in parallel; returns
    ``bigym_eval.summarize`` plus the merged per-task results. Raises when a task is incomplete."""
    import bigym_eval

    bigym_cls = bigym_cls or BiGym
    calls = {(t, c): bigym_cls().run.spawn(ckpt, t, c) for t in bigym_eval.TASKS for c in range(bigym_eval.CHUNKS[t])}
    return collect_bigym(calls)


def collect_bigym(calls: Dict) -> Dict:
    import bigym_eval

    got: Dict[str, List[Dict]] = {}
    for (t, c), call in calls.items():
        got.setdefault(t, []).append(call.get())
    merged = {t: bigym_eval.merge_chunks(cs) for t, cs in got.items()}
    summ = bigym_eval.summarize(merged)
    if not summ["complete"]:
        raise RuntimeError("BiGym benchmark incomplete, missing %s" % summ["missing"])
    summ.update(tasks=list(bigym_eval.TASKS), eval=bigym_eval.fingerprint(),
                wall_seconds=max(r.get("seconds", 0) for cs in got.values() for r in cs),
                results={t: dict(m, chunk_timings=[{k: r.get(k) for k in ("chunk", "seconds", "env_seconds",
                                                                          "policy_seconds", "load_seconds", "gl")}
                                                   for r in got[t]]) for t, m in merged.items()})
    return summ


def print_bigym(b: Dict) -> None:
    print("%-18s %8s %8s %8s  %s" % ("bigym task", "progress", "success", "norm", "top actions"))
    for t in b["tasks"]:
        top = ", ".join("%s %.0f%%" % (a, 100 * f) for a, f in b["top_actions"].get(t, [])[:3])
        print("%-18s %8.3f %8.2f %+8.3f  %s" % (t, b["progress"].get(t, float("nan")), b["success"].get(t, float("nan")),
                                                b["per_task"].get(t) if b["per_task"].get(t) is not None else float("nan"),
                                                top))
    print("%-18s %8s %8.2f %+8.3f" % ("bigym (mean)", "", b["success_mean"], b["bigym"]))


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
    if kind == "bigym_demos":
        raise ValueError("bigym_demos parts are built by build_bigym_demo_part (they need the simulator)")
    exs = pool_selection(kind, name)
    if kind == "bigym":  # bc_f4 records share frames: read each file once, one bytes object per frame
        paths = sorted({p for ex in exs for p in _image_paths(ex)})

        def read(p):
            with open(p, "rb") as f:
                return p, f.read()

        with ThreadPoolExecutor(64) as pool:
            files = dict(pool.map(read, paths))
        exs = [_with_bytes(ex, files) for ex in exs]
    else:
        with ThreadPoolExecutor(64) as pool:  # each small-file read is ~0.4 s of latency; overlap many
            exs = list(pool.map(_with_bytes, exs))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "wb") as f:
        pickle.dump(exs, f, protocol=5)
    os.replace(path + ".tmp", path)
    data_vol.commit()
    return {"kind": kind, "name": name, "examples": len(exs), "mb": round(os.path.getsize(path) / 1e6, 1),
            "seconds": round(time.time() - t, 1)}


def _val_demo_seeds(task: str) -> List[int]:
    """The demo seeds the cleaned BiGym sets hold out for ``task`` (their val split, by record id
    ``<task>-<seed>-<decision>``)."""
    seeds = set()
    with open(os.path.join("/data/vqa", BIGYM_VAL_SET, "val.jsonl")) as f:
        for line in f:
            if line.strip():
                t, seed = json.loads(line)["id"].split("-")[:2]
                if t == task:
                    seeds.add(int(seed))
    return sorted(seeds)


@app.function(image=bigym_image, cpu=2, memory=8192, timeout=3 * 60 * 60, volumes={"/data": data_vol})
def build_bigym_demo_part(task: str) -> Dict:
    """A ``bigym_demos`` part: ``laya.bigymdemos.demo_waypoints`` of every demo of ``task``, keeping those that
    succeed in BiGym's replay and are not in the cleaned sets' val split."""
    import pickle

    import bigym_eval

    t = time.time()
    path = _pool_file("bigym_demos", task)
    if os.path.exists(path):
        return {"kind": "bigym_demos", "name": task, "examples": -1, "mb": round(os.path.getsize(path) / 1e6, 1),
                "seconds": 0.0}
    bigym_eval.pick_gl()
    from laya import bigymdemos

    val = set(_val_demo_seeds(task))
    demos = bigymdemos.demo_waypoints(task, amount=-1, seed=0)
    keep = sorted((d for d in demos if d["success_step"] is not None and d["seed"] not in val),
                  key=lambda d: d["seed"])
    if any(bigym_eval.is_eval_seed(d["seed"]) for d in keep):
        raise RuntimeError("a demo seed falls in the BiGym eval's seed range")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "wb") as f:
        pickle.dump(keep, f, protocol=5)
    os.replace(path + ".tmp", path)
    data_vol.commit()
    print("%s: %d demos, %d succeed, %d held out as val, %d kept" % (
        task, len(demos), sum(d["success_step"] is not None for d in demos), len(val), len(keep)))
    return {"kind": "bigym_demos", "name": task, "examples": len(keep), "mb": round(os.path.getsize(path) / 1e6, 1),
            "seconds": round(time.time() - t, 1)}


def build_pool():
    """Build every missing pool part, all in parallel (``modal run autoresearch/harness.py --prepare-pool``)."""
    total = 0.0
    demo_calls = [build_bigym_demo_part.spawn(n) for k, n in POOL_PARTS if k == "bigym_demos"]
    plain = [(k, n) for k, n in POOL_PARTS if k != "bigym_demos"]
    results = list(build_pool_part.starmap(plain, order_outputs=False, return_exceptions=True))
    for c in demo_calls:
        try:
            results.append(c.get())
        except Exception as e:
            results.append(e)
    for r in results:
        if isinstance(r, Exception):
            print("FAILED:", repr(r)[:300])
            continue
        total += r["mb"]
        print("%-6s %-28s %6d examples %8.1f MB %6.1f s" % (r["kind"], r["name"], r["examples"], r["mb"], r["seconds"]))
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
    files = [os.path.abspath(__file__)] + [os.path.join(REPO, "autoresearch", f) for f in _harness_files()]
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
    """``(TrainEval, Latency, Games, TrainEvalBigym, BiGym)`` from a deployment of exactly this code, deploying it
    first if needed.

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
    return (te, modal.Cls.from_name(name, "Latency"), modal.Cls.from_name(name, "Games"),
            modal.Cls.from_name(name, "TrainEvalBigym"), modal.Cls.from_name(name, "BiGym"))


def _git(*args) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True).stdout.strip()


@app.local_entrypoint()
def main(tag: str = "", desc: str = "", prune: bool = True, prepare_pool: bool = False, profile: str = "default",
         experiment: str = "autoresearch/experiment.py", bigym_eval: bool = False, bigym_baselines_tasks: str = "",
         measure_bigym_baselines: bool = False, bigym_check_model: str = "", out: str = ""):
    """``--profile bigym`` trains on the BiGym image and adds the BiGym benchmark (keep / discard by
    ``pareto.decide_bigym``); ``--experiment`` picks the experiment file (relative to the repo root);
    ``--bigym-eval`` also runs the BiGym benchmark under the default profile (reported, not decided on).

    One-off BiGym jobs (no training; Modal allows one local entrypoint per file, so they are flags here):
    ``--measure-bigym-baselines [--bigym-baselines-tasks A,B]`` measures ``bigym_baselines.json``;
    ``--bigym-check-model <run>/best [--out x.json]`` runs the BiGym benchmark alone on a saved checkpoint."""
    import pareto

    if prepare_pool:
        build_pool()
        return
    if measure_bigym_baselines:
        bigym_baselines(bigym_baselines_tasks)
        return
    if bigym_check_model:
        bigym_check(bigym_check_model, out)
        return
    if not tag:
        raise SystemExit("--tag is required (the run tag, e.g. sep23)")
    if profile not in PROFILES:
        raise SystemExit("--profile is one of %s" % ", ".join(PROFILES))
    with_bigym = profile == "bigym" or bigym_eval

    exp_rel = os.path.relpath(os.path.abspath(os.path.join(REPO, experiment)), REPO)
    exp_path = os.path.join(REPO, exp_rel)
    if exp_rel.startswith("..") or not os.path.exists(exp_path):
        raise SystemExit("no experiment file %s in the repository" % experiment)
    if _git("status", "--porcelain", "--", exp_path) or not _git("ls-files", "--", exp_path):
        raise SystemExit("commit %s first: results are keyed by commit" % exp_rel)
    commit = _git("rev-parse", "--short=7", "HEAD")
    desc = desc or _git("log", "-1", "--format=%s")
    runs = os.path.join(REPO, "autoresearch", "runs", tag)
    tsv = os.path.join(runs, "results.tsv")
    have = pareto.tsv_profile(tsv)
    if have is not None and have != profile:
        raise SystemExit("tag %r is a %s-profile tag; run it with --profile %s or start a new tag" % (tag, have, have))
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
    # a failed deploy is not the experiment's crash
    train_eval, latency, games, train_eval_bigym, bigym_cls = deployed_classes()
    logs = follow_logs(deployment_name())
    timings = {}
    try:
        import games_eval

        te = train_eval_bigym if profile == "bigym" else train_eval
        res = te().run.remote(source, tag, commit)
        timings["train_eval_s"] = round(time.time() - t0, 1)
        t1 = time.time()
        # the latency job, one games job per family and (bigym) one BiGym job per task chunk, all at once
        lat = latency().run.spawn(tag, commit)
        fams = {f: games().run.spawn(tag, commit, f) for f in games_eval.FAMILIES}
        bcalls = {}
        if with_bigym:
            import bigym_eval as be

            ck = os.path.join(ROOT, tag, commit)
            bcalls = {(t, c): bigym_cls().run.spawn(ck, t, c) for t in be.TASKS for c in range(be.CHUNKS[t])}
        res["summary"].update({k: v for k, v in lat.get().items() if k.startswith("latency")})
        timings["latency_s"] = round(time.time() - t1, 1)
        played = {}
        for f, call in fams.items():
            played.update(call.get())
        timings["games_s"] = round(time.time() - t1, 1)
        g = games_eval.summarize(played)
        if not g["complete"]:
            raise RuntimeError("games benchmark incomplete, missing %s" % g["missing"])
        res["summary"]["games"] = g["games"]
        res["games"] = {"per_game": g["per_game"], "results": played}
        if bcalls:
            b = collect_bigym(bcalls)
            timings["bigym_s"] = round(time.time() - t1, 1)
            res["summary"]["bigym"] = b["bigym"]
            res["bigym"] = b
    except Exception as e:
        print("crash: %r" % (e,))
        pareto.append_tsv(tsv, pareto.crash_row(commit, desc, profile, exp_rel), profile)
        print("status: crash")
        raise SystemExit(1)
    finally:
        time.sleep(3)  # let the last lines arrive
        logs.terminate()
    res["harness"] = version
    res.update(description=desc, total_s=round(time.time() - t0, 1), profile=profile, experiment=exp_rel,
               timings=timings)
    out = os.path.join(runs, commit + ".json")
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    s = res["summary"]
    print("---")
    keys = ("bigym",) if "bigym" in s else ()
    for k in keys + ("quality", "macro_acc", "ece_hard", "games", "params_m", "latency_x", "latency_ms",
                     "latency_ref_ms"):
        print("%-17s %.4f" % (k + ":", s[k]))
    print("%-17s %.1f" % ("train_seconds:", res["train_s"]))
    print("%-17s %.1f" % ("total_seconds:", res["total_s"]))
    print("%-17s %s" % ("profile:", profile))
    print("%-17s %s" % ("experiment:", exp_rel))
    if "bigym" in res:
        print_bigym(res["bigym"])
    rows = pareto.read_tsv(tsv)
    row = pareto.row_from_result(res, commit, desc)
    if profile == "bigym":
        status, why = pareto.decide_bigym(rows, row)
        row["status"] = status
        pareto.append_tsv(tsv, row, profile)
        print("status:           %s (%s)" % (status, why))
        print(pareto.show_bigym(pareto.read_tsv(tsv)))
        prunable = pareto.prunable_bigym(pareto.read_tsv(tsv))
    else:
        status, beaten = pareto.decide(pareto.frontier(rows), row)
        row["status"] = status
        pareto.append_tsv(tsv, row)
        print("status:           %s" % status)
        if beaten:
            print("now dominates:    %s" % ", ".join(p["commit"] for p in beaten))
        print(pareto.show(pareto.read_tsv(tsv)))
        prunable = pareto.prunable(pareto.read_tsv(tsv))
    if prune:
        # only commits the TSV has finished with (and, default profile, off the frontier): a concurrent run's
        # checkpoint is not in the TSV until that run decides, so it is never touched
        removed = prune_checkpoints.remote(tag, prunable)
        if removed:
            print("pruned checkpoints: %s" % ", ".join(removed))


def bigym_baselines(tasks: str = ""):
    """Measure the BiGym benchmark's random and oracle references on its own seeds (one CPU container per task and
    policy) and merge them into ``autoresearch/bigym_baselines.json``."""
    import bigym_eval as be

    todo = [t for t in (tasks.split(",") if tasks else be.TASKS) if t]
    t0 = time.time()
    calls = {(t, p): bigym_reference.spawn(t, p) for t in todo for p in (("random", "oracle") if t in be.REACH
                                                                         else ("random",))}
    got = {k: c.get() for k, c in calls.items()}
    entries = {t: be.baseline_entry(t, got[(t, "random")], got.get((t, "oracle"))) for t in todo}
    for t, e in entries.items():
        print("%-18s random %.4f (success %.2f)  expert %.4f  [%s]  %.0f s" % (
            t, e["random"], e["random_success"], e["expert"], e["expert_source"][:22],
            max(v["seconds"] for (tt, _), v in got.items() if tt == t)))
    meta = {"date": time.strftime("%Y-%m-%d"), "commit": _git("rev-parse", "HEAD"), "mujoco": MUJOCO, "bigym": BIGYM,
            "gl": sorted({v["gl"] for v in got.values()}), "eval": be.fingerprint(),
            "wall_seconds": round(time.time() - t0, 1)}
    be.write_baselines(entries, meta=meta)
    print("wrote", be.BASELINES_PATH)


def bigym_check(model: str, out: str = ""):
    """The BiGym benchmark alone on a checkpoint of the laya-checkpoints volume (``<run>/best`` under /ckpt/smolvlm,
    or ``<family>/<run>/best`` under /ckpt), e.g. to compare a fine-tune with the zero-shot model. Writes
    ``autoresearch/runs/bigym-checks/<model>-<date>.json`` (create-only)."""
    for base in ("/ckpt/smolvlm", "/ckpt"):  # resolved like ckpt_path, but from here (the client has no /ckpt)
        rel = os.path.join(base, model)[len("/ckpt/"):]
        ls = subprocess.run([sys.executable, "-m", "modal", "volume", "ls", "laya-checkpoints", rel],
                            capture_output=True, text=True)
        if ls.returncode == 0 and "vlm_agent_config.json" in ls.stdout:
            ck = os.path.join(base, model)
            break
    else:
        raise SystemExit("no checkpoint %r under /ckpt/smolvlm or /ckpt" % model)
    path = out or os.path.join(REPO, "autoresearch", "runs", "bigym-checks",
                               "%s-%s.json" % (model.replace("/", "_"), time.strftime("%Y%m%d-%H%M%S")))
    if os.path.exists(path):
        raise SystemExit("%s exists (create-only)" % path)
    t0 = time.time()
    b = run_bigym(ck)
    b.update(checkpoint=ck, model=model, total_seconds=round(time.time() - t0, 1), code=_git("rev-parse", "HEAD"),
             date=time.strftime("%Y-%m-%d"))
    print_bigym(b)
    print("wall %.0f s (slowest container %.0f s)" % (b["total_seconds"], b["wall_seconds"]))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(b, f, indent=1)
    print("wrote", path)
