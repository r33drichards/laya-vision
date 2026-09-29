"""The fixed BiGym benchmark behind the ``bigym`` objective (the ``bigym`` profile of the harness).

The saved checkpoint plays six BiGym tasks (``laya.bigymgames``: the Unitree H1 driven by 37 named motion
primitives, one ``choice`` question per decision over the head-camera frames) greedy, one forward per decision:
the most likely option of ``bigym_question(task, frames)`` is played, exactly the answer ``predict`` gives with one
option order (``laya.bigymgames.model_policy``), but batched over every live episode of a container.

* ``frames`` (1 or 4) is the checkpoint's ``bigym_frames`` config key (``FRAMES_DEFAULT`` when it has none), so an
  experiment picks it by writing the key; the token budgets are the checkpoint's (``head_max_len``, ``max_len``),
  and a question that would be cut raises, as ``predict(strict=True)`` does.
* Episodes: ``EPISODES`` per task, episode ``i`` of task ``t`` on seed ``SEED_BASE + 1000 * TASKS.index(t) + i``,
  in 780,000-785,999: off every range anything trains on (BiGym's demos carry 32-bit seeds in the millions to
  billions, experiments' rollouts use seeds below ``TRAIN_SEED_MAX``) and off the games benchmark's (700,000 to
  ~771,000) and the probe's (300,000+). An episode ends on success, on BiGym's termination, or after ``CAPS[task]``
  decisions (``laya.bigymgames.TASKS``' caps).
* Per episode a dense progress in [0, 1], the best over the episode (the start state included): reach tasks
  ``1 - d / d0`` clipped to [0, 1], ``d`` the distance from the allowed wrist nearest the target (``d0`` at reset);
  cupboard tasks the fraction of the way from the part's joint state at reset to the goal state (the top drawer,
  or the mean over the wall cabinet's two doors, each clipped to [0, 1]). A success counts 1.
* Per task ``normalized = (model - random) / (expert - random)`` clipped to [``CLIP_LO``, ``CLIP_HI``] against
  ``bigym_baselines.json`` (random: ``random_policy`` with one ``random.Random(seed)`` per episode, on the same
  episodes; expert: ``oracle_policy`` on the reach tasks, and 1.0 on the cupboard tasks, where BiGym's replayed
  human demos succeed 90-100% and there is no primitive oracle). ``bigym`` = the mean over the six tasks.

The harness runs one L4 container per ``(task, chunk)`` (``CHUNKS``), all in parallel with the games and latency
jobs. Success rates and the most played primitives per task are reported alongside (not scored).

Baselines: ``modal run autoresearch/harness.py --measure-bigym-baselines`` plays random and the oracle (no camera: the
physics does not depend on rendering) and writes ``bigym_baselines.json`` next to this file.

This file is part of the fixed harness: experiments do not edit it.
"""
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASELINES_PATH = os.path.join(HERE, "bigym_baselines.json")
TASKS = ("ReachTarget", "ReachTargetSingle", "DrawerTopOpen", "DrawerTopClose", "WallCupboardOpen",
         "WallCupboardClose")
REACH = ("ReachTarget", "ReachTargetSingle")
EPISODES = 32
SEED_BASE = 780_000
TRAIN_SEED_MAX = 100_000   # experiments' own rollouts use seeds below this (or BiGym's demo seeds), never these
CAPS = {"ReachTarget": 60, "ReachTargetSingle": 60, "DrawerTopOpen": 150, "DrawerTopClose": 150,
        "WallCupboardOpen": 150, "WallCupboardClose": 150}   # = laya.bigymgames.TASKS max_decisions
CHUNKS = {t: 2 if t in REACH else 3 for t in TASKS}   # containers per task, each playing its share of EPISODES in lockstep
FRAMES_DEFAULT = 4
CLIP_LO, CLIP_HI = -0.5, 1.5
FORWARD_BATCH = 16
CUPBOARD_EXPERT = 1.0


def task_seeds(task: str) -> List[int]:
    return [SEED_BASE + 1000 * TASKS.index(task) + i for i in range(EPISODES)]


def chunk_seeds(task: str, chunk: int) -> List[int]:
    n = CHUNKS[task]
    if not 0 <= chunk < n:
        raise ValueError("%s has chunks 0..%d, got %d" % (task, n - 1, chunk))
    seeds = task_seeds(task)
    per = -(-len(seeds) // n)
    return seeds[chunk * per:(chunk + 1) * per]


def is_eval_seed(seed: int) -> bool:
    return SEED_BASE <= int(seed) < SEED_BASE + 1000 * len(TASKS)


def pick_gl() -> str:
    """EGL when a context can be made (GPU containers), else OSMesa; must run before MuJoCo is imported. The same
    probe as ``modal_bigym._pick_gl``."""
    probe = "import mujoco; c = mujoco.GLContext(64, 64); c.make_current(); c.free()"
    for gl in ("egl", "osmesa"):
        env = dict(os.environ, MUJOCO_GL=gl, PYOPENGL_PLATFORM=gl)
        if subprocess.run([sys.executable, "-c", probe], env=env, capture_output=True, timeout=120).returncode == 0:
            os.environ["MUJOCO_GL"] = os.environ["PYOPENGL_PLATFORM"] = gl
            return gl
    raise RuntimeError("no headless GL backend for MuJoCo")


# ---------------------------------------------------------------------------------------------------------------
# Dense progress
# ---------------------------------------------------------------------------------------------------------------


def part_state(game) -> np.ndarray:
    """The task's articulated part: ``[top drawer]`` or the wall cabinet's ``[door, door]``, 0 closed .. 1 open."""
    from laya.bigymdemos import _part

    return np.array(_part(game.env, game.task), np.float64)


def start_state(game):
    """What progress is measured from: the reach distance ``d0`` at reset, or the part's joint state."""
    if game.task in REACH:
        return float(game.ground_truth()["distance"])
    return part_state(game)


def progress(game, start) -> float:
    """Dense task progress in [0, 1] of the current state (1 once the episode has succeeded)."""
    if game.success:
        return 1.0
    if game.task in REACH:
        d = float(game.ground_truth()["distance"])
        return float(np.clip(1.0 - d / max(1e-6, start), 0.0, 1.0))
    goal = float(game.spec["goal"])
    s = part_state(game)
    fr = [1.0 if abs(goal - s0) < 1e-6 else float(np.clip((v - s0) / (goal - s0), 0.0, 1.0)) for v, s0 in zip(s, start)]
    return float(np.mean(fr))


# ---------------------------------------------------------------------------------------------------------------
# Policies: policy(games, episode indices) -> one primitive name per game
# ---------------------------------------------------------------------------------------------------------------


def random_policy(seeds: Sequence[int]):
    """Uniform over the primitives with one ``random.Random(seed)`` per episode: independent of the batching."""
    rngs = {i: random.Random(s) for i, s in enumerate(seeds)}

    def policy(games, idx):
        return [rngs[i].choice(g.actions) for g, i in zip(games, idx)]

    return policy


def oracle_policy(games, idx):
    from laya import bigymgames as bg

    return [bg.oracle_action(g) for g in games]


class FramePrep:
    """Per-frame image preprocessing, cached while the frame is in some episode's history. With the processor
    backend and no splitting, the processor treats each image of a state on its own and the prefix ids depend on
    the image count only, so a k-frame state's prefix is the k frames' cached pixels concatenated under the ids of
    any k-image prefix: each frame goes through the processor once instead of k times. ``check`` compares that
    with ``vlm_prefix`` on the real frames (exact equality) the first time it is used."""

    def __init__(self, agent, threads: int = 8):
        from laya.vlm import vlm_prefix

        self.agent, self.vlm_prefix = agent, vlm_prefix
        prep = agent.prep
        self.fast = not prep.on_gpu and not prep.split_edge
        self.cache: Dict[int, tuple] = {}
        self.ids: Dict[int, list] = {}
        self.pool = ThreadPoolExecutor(max_workers=threads)
        self.checked = False

    def _one(self, frame):
        p = self.vlm_prefix(self.agent.processor, [frame], self.agent.prep)
        return frame, p["pixel_values"], p["pixel_attention_mask"]

    def prefixes(self, states: List[List]) -> List[Dict]:
        """``vlm_prefix`` of every state (a list of frames each)."""
        import torch

        if not self.fast:
            return list(self.pool.map(lambda fr: self.vlm_prefix(self.agent.processor, fr, self.agent.prep), states))
        live = {id(f): f for fr in states for f in fr}
        self.cache = {k: v for k, v in self.cache.items() if k in live}  # holds the frame: its id stays unique
        new = [f for k, f in live.items() if k not in self.cache]
        for f, pv, pam in self.pool.map(self._one, new):
            self.cache[id(f)] = (f, pv, pam)
        out = []
        for fr in states:
            n = len(fr)
            if n not in self.ids:
                self.ids[n] = self.vlm_prefix(self.agent.processor, list(fr), self.agent.prep)["ids"]
            out.append({"ids": self.ids[n], "pixel_values": torch.cat([self.cache[id(f)][1] for f in fr]),
                        "pixel_attention_mask": torch.cat([self.cache[id(f)][2] for f in fr]), "raw_images": None,
                        "n_images": n})
        if not self.checked and states:
            ref = self.vlm_prefix(self.agent.processor, list(states[0]), self.agent.prep)
            ok = (ref["ids"] == out[0]["ids"] and torch.equal(ref["pixel_values"], out[0]["pixel_values"])
                  and torch.equal(ref["pixel_attention_mask"], out[0]["pixel_attention_mask"]))
            if not ok:
                raise RuntimeError("cached per-frame preprocessing differs from vlm_prefix on the same frames")
            self.checked = True
        return out

    def close(self):
        self.pool.shutdown()


def model_policy(agent, task: str, frames: int, forward: Optional[Callable] = None):
    """Greedy over ``bigym_question(task, frames)`` for a batch of games: the argmax of the calibrated option
    probabilities, the identity option order (as ``predict``). ``policy.forwards`` counts model rows."""
    from laya import bigymgames as bg
    from laya.common import QTYPES, render_options, truncation_answer, truncation_error
    from laya.vlm import VLMAgent, build_vlm_inputs

    if forward is None:
        from games_eval import _forward as forward
    q = VLMAgent._to_internal(bg.bigym_question(task, frames)["action"])
    k = len(render_options(q))
    names = [bg.FROM_WORDS[w] for w in q["crit"]]
    max_len, head_max_len = agent.cfg.get("max_len", 1024), agent.cfg.get("head_max_len", 256)
    prep = FramePrep(agent)

    def policy(games, idx):
        from PIL import Image

        states = [g.frames(frames) if frames > 1 else [Image.fromarray(g.frame())] for g in games]
        items = []
        for pre in prep.prefixes(states):
            it = build_vlm_inputs(agent.processor, {}, q, max_len, head_max_len, prefix=pre)
            if not items:
                cut = truncation_answer(it["truncation"], q)
                if cut:
                    raise truncation_error("action", cut, max_len, head_max_len)
            it["qtype"] = QTYPES["choice"]
            items.append(it)
        probs = np.concatenate([forward(agent, items[s:s + FORWARD_BATCH], k)
                                for s in range(0, len(items), FORWARD_BATCH)], 0)
        policy.forwards += len(items)
        return [names[int(r.argmax())] for r in probs]

    policy.forwards = 0
    policy.close = prep.close
    return policy


# ---------------------------------------------------------------------------------------------------------------
# Lockstep play
# ---------------------------------------------------------------------------------------------------------------


def play(task: str, seeds: Sequence[int], policy, cap: Optional[int] = None, cameras: bool = True) -> Dict:
    """Play one episode per seed, all together: each round ``policy(live games, their indices)`` picks one
    primitive per live episode (one batched call) and every live episode steps once. Returns per-episode best
    progress, success and decisions, and the primitive counts."""
    from laya import bigymgames as bg

    cap = cap or CAPS[task]
    t0 = time.time()
    envs = [bg.make_env(task, cameras=cameras) for _ in seeds]
    games = [bg.BiGymGame(task, s, env=e) for s, e in zip(seeds, envs)]
    t_envs = time.time() - t0
    starts = [start_state(g) for g in games]
    best = [progress(g, s) for g, s in zip(games, starts)]
    counts: Counter = Counter()
    rounds = 0
    t_policy = 0.0
    while True:
        live = [i for i, g in enumerate(games) if not g.done and g.decisions < cap]
        if not live:
            break
        tp = time.time()
        acts = policy([games[i] for i in live], live)
        t_policy += time.time() - tp
        for i, a in zip(live, acts):
            counts[a] += 1
            games[i].step(a)
            best[i] = max(best[i], progress(games[i], starts[i]))
        rounds += 1
    eps = []
    for s, g, st, b in zip(seeds, games, starts, best):
        eps.append({"seed": int(s), "progress": round(float(b), 5), "final": round(progress(g, st), 5),
                    "success": bool(g.success), "decisions": int(g.decisions), "terminated": bool(g.terminated)})
    for e in envs:
        e.close()
    return {"task": task, "episodes": eps, "actions": dict(counts), "rounds": rounds, "cap": cap,
            "seconds": round(time.time() - t0, 1), "env_seconds": round(t_envs, 1),
            "policy_seconds": round(t_policy, 1)}


def run_chunk(agent, task: str, chunk: int) -> Dict:
    """The model's episodes of one chunk of ``task`` (what one harness container plays)."""
    frames = int(agent.cfg.get("bigym_frames", FRAMES_DEFAULT))
    if frames not in (1, 4):
        raise ValueError("bigym_frames must be 1 or 4, got %r" % frames)
    policy = model_policy(agent, task, frames)
    try:
        out = play(task, chunk_seeds(task, chunk), policy)
    finally:
        policy.close()
    out.update(chunk=chunk, frames=frames, forwards=policy.forwards,
               budgets={k: agent.cfg.get(k) for k in ("head_max_len", "max_len")})
    return out


# ---------------------------------------------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------------------------------------------


def load_baselines(path: str = BASELINES_PATH) -> Dict[str, Dict]:
    with open(path) as f:
        return json.load(f)["tasks"]


def check_baseline(task: str, base: Optional[Dict]) -> Dict:
    """The stored baseline for ``task``, which must have been measured on the same seeds and cap."""
    if not base or base.get("random") is None or base.get("expert") is None:
        raise KeyError("no random/expert baseline for %s in bigym_baselines.json" % task)
    want = {"seeds": [task_seeds(task)[0], task_seeds(task)[-1]], "episodes": EPISODES, "cap": CAPS[task]}
    got = {k: base.get(k) for k in want}
    if got != want:
        raise ValueError("stale BiGym baseline for %s: stored %s, eval %s (re-measure)" % (task, got, want))
    return base


def normalize(model: float, rnd: float, expert: float) -> Optional[float]:
    if expert == rnd:
        return None
    return float(min(CLIP_HI, max(CLIP_LO, (model - rnd) / (expert - rnd))))


def merge_chunks(chunks: Sequence[Dict]) -> Dict:
    """One task's chunk results as one: episodes in seed order, actions summed."""
    chunks = sorted(chunks, key=lambda c: c.get("chunk", 0))
    eps = sorted((e for c in chunks for e in c["episodes"]), key=lambda e: e["seed"])
    acts: Counter = Counter()
    for c in chunks:
        acts.update(c["actions"])
    out = {"task": chunks[0]["task"], "episodes": eps, "actions": dict(acts),
           "seconds": max(c["seconds"] for c in chunks), "chunks": len(chunks)}
    for k in ("frames", "budgets", "cap"):
        if k in chunks[0]:
            out[k] = chunks[0][k]
    return out


def task_summary(res: Dict) -> Dict:
    eps = res["episodes"]
    total = max(1, sum(res["actions"].values()))
    top = sorted(res["actions"].items(), key=lambda kv: -kv[1])[:5]
    return {"progress": float(np.mean([e["progress"] for e in eps])),
            "success": float(np.mean([e["success"] for e in eps])), "episodes": len(eps),
            "top_actions": [[a, round(n / total, 3)] for a, n in top]}


def task_score(progress: float, success: float, base: Dict) -> Optional[float]:
    """Half normalized dense progress, half normalized success rate: partial progress counts, but a policy that
    pushes a part most of the way without ever finishing tops out near 0.5. ``None`` when neither is defined."""
    p = normalize(progress, base["random"], base["expert"])
    s = normalize(success, base.get("random_success", 0.0), base.get("expert_success", CUPBOARD_EXPERT))
    parts = [v for v in (p, s) if v is not None]
    return float(np.mean(parts)) if parts else None


def summarize(results: Dict[str, Dict], baselines: Optional[Dict[str, Dict]] = None) -> Dict:
    """``results``: ``{task: merged result}``. Returns ``{"bigym", "per_task": {task: score}, "per_task_progress":
    {task: normalized progress}, "success": {task: rate}, "progress": {...}, "top_actions": {...}, "missing",
    "complete"}``; ``bigym`` averages the tasks present (``complete`` False when one is missing or short of
    episodes). Each task's score is ``task_score``."""
    baselines = load_baselines() if baselines is None else baselines
    per, per_prog, succ, prog, top = {}, {}, {}, {}, {}
    for task in TASKS:
        if task not in results:
            continue
        base = check_baseline(task, baselines.get(task))
        s = task_summary(results[task])
        if s["episodes"] != EPISODES:
            continue
        prog[task], succ[task], top[task] = s["progress"], s["success"], s["top_actions"]
        per[task] = task_score(s["progress"], s["success"], base)
        per_prog[task] = normalize(s["progress"], base["random"], base["expert"])
    missing = [t for t in TASKS if t not in per]
    vals = [v for v in per.values() if v is not None]
    return {"bigym": float(np.mean(vals)) if vals else float("nan"), "per_task": per, "per_task_progress": per_prog,
            "progress": prog,
            "success": succ, "success_mean": float(np.mean(list(succ.values()))) if succ else float("nan"),
            "top_actions": top, "missing": missing, "complete": not missing}


# ---------------------------------------------------------------------------------------------------------------
# Baselines (measured once; the eval never plays these)
# ---------------------------------------------------------------------------------------------------------------


def play_reference(task: str, policy: str) -> Dict:
    """``random`` or ``oracle`` (reach tasks) on the task's eval episodes, without the camera."""
    seeds = task_seeds(task)
    fn = random_policy(seeds) if policy == "random" else oracle_policy
    return play(task, seeds, fn, cameras=False)


def baseline_entry(task: str, rnd: Dict, oracle: Optional[Dict]) -> Dict:
    r = task_summary(rnd)
    e = {"random": r["progress"], "random_success": r["success"],
         "random_progress": [e["progress"] for e in rnd["episodes"]],
         "episodes": EPISODES, "seeds": [task_seeds(task)[0], task_seeds(task)[-1]], "cap": CAPS[task]}
    if oracle is not None:
        o = task_summary(oracle)
        e.update(expert=o["progress"], expert_source="oracle_policy (privileged greedy reach)",
                 expert_success=o["success"], expert_progress=[x["progress"] for x in oracle["episodes"]])
    else:
        e.update(expert=CUPBOARD_EXPERT, expert_source="1.0: BiGym's human demos replayed in the sim succeed 90-100% "
                                                       "(modal_bigym.py bigym_demos); no primitive oracle")
    return e


def write_baselines(entries: Dict[str, Dict], path: str = BASELINES_PATH, meta: Optional[Dict] = None) -> None:
    import re

    doc = {"about": "random and expert dense-progress references for autoresearch/bigym_eval.py, measured on the "
                    "eval's own seeds and caps; normalized = (model - random) / (expert - random)", "tasks": {}}
    if os.path.exists(path):
        with open(path) as f:
            doc = json.load(f)
    doc["tasks"].update(entries)
    doc["tasks"] = {t: doc["tasks"][t] for t in TASKS if t in doc["tasks"]}
    if meta:
        doc["measured"] = meta
    text = json.dumps(doc, indent=1)
    text = re.sub(r"\[\s*([-0-9.,e\s]*?)\s*\]", lambda m: "[" + " ".join(m.group(1).split()) + "]", text)
    with open(path, "w") as f:
        f.write(text + "\n")


def fingerprint() -> str:
    """Hash of the eval's fixed settings, recorded with every result."""
    spec = {"tasks": TASKS, "episodes": EPISODES, "seed_base": SEED_BASE, "caps": CAPS, "clip": [CLIP_LO, CLIP_HI]}
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:10]
