"""The fixed games benchmark behind the ``games`` objective: how well a saved checkpoint plays games from pixels.

Every game step is one ``choice`` question (``laya.games``): the screen is the image, the options are the game's
actions, and the model's most likely option is played (greedy, one option order, as ``predict`` answers it). The
suite, the episode seeds and the step caps below are fixed, so the random and expert reference scores are fixed
too; they are measured once and stored in ``game_baselines.json`` next to this file, and the eval only plays the
model.

Per game, ``normalized = (model - random) / (expert - random)`` clipped to ``[CLIP_LO, CLIP_HI]``, so one game
cannot dominate; ``games`` = the mean of ``normalized`` over the games (each game one vote).

Families (the unit the harness runs in one container):

* ``grid``: Maze 4x4 and 6x6 (score: solve rate within 4x the shortest path), Snake 10x10 (food eaten, 200-step
  cap). Pure Python.
* ``control``: CartPole, Acrobot, MountainCar, LunarLander from Gymnasium (score: summed reward, capped
  episodes, previous frame ghosted in). Needs ``gymnasium[classic-control,box2d]``.
* ``atari``: Freeway and Breakout (``ALE/<Game>-v5`` defaults, auto-FIRE, 1,000 agent steps), via
  ``laya.atari_train.play``. Needs ``ale-py`` and ``gymnasium``.
* ``doom``: ViZDoom ``basic`` (summed reward). Needs ``vizdoom``.

Play is batched lockstep: all episodes of a game step together and each step is one batched model forward over
every live episode's screen (``batched_probs``). For grid and control, if ``agent.cfg["search"]`` is truthy the
actions come from ``laya.search.plan(agent, envs, question, settings)`` instead, where ``envs`` are the adapters
below (``clone``, ``step``, ``render``, ``actions``, ``done``); search plays the ``single`` frame mode only.

Frame modes (``laya.frames``). Every family plays in the checkpoint's own game frame mode,
``laya.frames.mode_for(agent.cfg, family)``: ``cfg["game_frames"]`` (``"single"``, ``"trail-N"``, ``"stack-N"``), or
``stack-2`` for Atari from an old ``atari_frames: 2`` checkpoint, else ``single``. Each episode keeps its frame
history (the screens at its decision points, oldest first; the adapters' ``history()``, ``laya.atari_train.play``'s
``hists``, ``play_doom``'s per-instance history, reset with each episode and, in Atari, after an auto-FIRE), and the
state is ``laya.frames.state`` of it: at an episode's start the first frame is repeated. ``single`` plays exactly as
before modes existed (the control games' two-frame ghost, one frame elsewhere). For ``stack-N`` the vision tower
runs once per distinct frame (``FrameFeatureCache``, kept for N steps), as the Atari player does; the language model
sees N images per decision. The random and expert baselines do not depend on the mode.

Seeds: episode ``i`` of a game uses seed ``SEED_BASE[family] + i`` (see ``SUITE``). They lie in 700,000-799,999,
off every range the repo trains or evaluates on: Atari expert data 1,000+ (val) / 2,000+ (train) and its baselines
900,000+ / 950,000+; ViZDoom data 0+ (train) / 1,000,000+ (val); the public ``games_eval`` suite 50,000 (Doom),
100,000 (Atari) and 200,000 (grid, control). No grid or control training data exists.

Baselines: ``python autoresearch/games_eval.py baselines --families grid,control[,doom]`` plays random and expert
and merges them into ``game_baselines.json``; the Atari expert (CleanRL PPO, ``laya.atari_data.expert``) needs
jax / flax / opencv and is measured by ``atari_expert_scores`` under the same seeds and cap.
"""
import copy
import hashlib
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from laya import frames as F

HERE = os.path.dirname(os.path.abspath(__file__))
BASELINES_PATH = os.path.join(HERE, "game_baselines.json")
CLIP_LO, CLIP_HI = -0.5, 1.5
FAMILIES = ("grid", "control", "atari", "doom")
SEED_BASE = {"grid": 700_000, "control": 730_000, "atari": 760_000, "doom": 770_000}
DOOM_TICS = 4        # tics per decision, as play_doom and the doom_basic data
DOOM_PARALLEL = 16   # ViZDoom instances stepped in lockstep


@dataclass(frozen=True)
class GameSpec:
    """One game of the suite. ``cap`` is the most agent steps an episode may take (``None``: the game's own
    limit, i.e. 4x the shortest path for Maze). ``params`` are game constructor arguments."""
    name: str
    family: str
    episodes: int
    seed: int
    cap: Optional[int]
    score: str
    params: Dict = field(default_factory=dict)

    @property
    def seeds(self) -> List[int]:
        return [self.seed, self.seed + self.episodes - 1]


SUITE: Dict[str, GameSpec] = {s.name: s for s in (
    GameSpec("Maze4", "grid", 128, SEED_BASE["grid"], None, "solve rate", {"game": "maze", "size": 4}),
    GameSpec("Maze6", "grid", 64, SEED_BASE["grid"] + 10_000, None, "solve rate", {"game": "maze", "size": 6}),
    GameSpec("Snake10", "grid", 64, SEED_BASE["grid"] + 20_000, 200, "food eaten", {"game": "snake", "size": 10}),
    GameSpec("CartPole", "control", 16, SEED_BASE["control"], 200, "reward", {"game": "CartPole"}),
    GameSpec("Acrobot", "control", 16, SEED_BASE["control"] + 5_000, 200, "reward", {"game": "Acrobot"}),
    GameSpec("MountainCar", "control", 16, SEED_BASE["control"] + 10_000, 200, "reward", {"game": "MountainCar"}),
    GameSpec("LunarLander", "control", 16, SEED_BASE["control"] + 15_000, 300, "reward", {"game": "LunarLander"}),
    GameSpec("Freeway", "atari", 4, SEED_BASE["atari"], 1000, "game score", {"game": "Freeway"}),
    GameSpec("Breakout", "atari", 4, SEED_BASE["atari"] + 1_000, 1000, "game score", {"game": "Breakout"}),
    GameSpec("DoomBasic", "doom", 64, SEED_BASE["doom"], None, "reward", {"scenario": "basic"}),
)}


def family_games(family: str, suite: Dict[str, GameSpec] = SUITE) -> List[GameSpec]:
    if family not in FAMILIES:
        raise ValueError("unknown family %r (%s)" % (family, ", ".join(FAMILIES)))
    return [s for s in suite.values() if s.family == family]


# ---------------------------------------------------------------------------------------------------------------
# Adapters: the protocol laya.search.plan works against
# ---------------------------------------------------------------------------------------------------------------


class _Frames:
    """Frame history for an adapter: ``history()`` is the episode's screens at its decision points so far, oldest
    first, ending with the current one (at most ``laya.frames.MAX_FRAMES``). The current screen joins the history
    the first time it is asked for at a step; lockstep play asks every live episode every step."""

    _hist: List = []
    _seen = -1

    def history(self) -> List:
        if self._seen != self.steps:
            self._hist = (list(self._hist) + [self.frame()])[-F.MAX_FRAMES:]
            self._seen = self.steps
        return self._hist

    def state(self, mode: str) -> Dict:
        return F.state(self.history(), mode, self.family)

    def _copy_history(self, other) -> None:
        other._hist, other._seen = list(self._hist), self._seen


class GridAdapter(_Frames):
    """A ``laya.gridgames`` Maze or Snake episode. ``score`` is the benchmark metric for this episode (1 when the
    maze is solved; food eaten in Snake) and ``step`` returns its change as the reward."""

    family = "grid"

    def __init__(self, env, actions: Sequence[str], seed: int = 0):
        self.env, self.actions, self.seed = env, tuple(actions), seed
        self.searchable = True

    @property
    def score(self) -> float:
        return float(self.env.solved) if hasattr(self.env, "solved") else float(self.env.eaten)

    @property
    def steps(self) -> int:
        return self.env.steps

    @property
    def done(self) -> bool:
        return self.env.done

    def step(self, action: str):
        before = self.score
        self.env.step(action)
        return self.score - before, self.done

    def render(self):
        return self.env.render()

    def frame(self) -> np.ndarray:
        return np.asarray(self.env.render())

    def expert(self) -> str:
        return self.env.expert()

    def clone(self) -> "GridAdapter":
        c = GridAdapter(copy.deepcopy(self.env), self.actions, self.seed)
        self._copy_history(c)
        return c


class ControlAdapter(_Frames):
    """A ``laya.controlgames.ControlGame`` episode capped at ``cap`` steps; the reward is the environment's.
    Cloning deep-copies the Gymnasium env minus its pygame surface and clock (recreated on the next render).
    A game whose env cannot be deep-copied (LunarLander: its Box2D world) is not ``searchable``: it is played
    greedy, and ``clone`` raises TypeError."""

    _clonable: Dict[str, bool] = {}  # game -> whether its env survives deepcopy (probed once)
    family = "control"

    def __init__(self, game, cap: Optional[int], actions: Sequence[str], seed: int = 0):
        self.game, self.cap, self.actions, self.seed = game, cap, tuple(actions), seed

    @property
    def searchable(self) -> bool:
        name = self.game.game
        if name not in ControlAdapter._clonable:
            ControlAdapter._clonable[name] = _faithful_copy(name)
        return ControlAdapter._clonable[name]

    def _copy(self):
        return _copy_game(self.game)

    @property
    def score(self) -> float:
        return self.game.score

    @property
    def steps(self) -> int:
        return self.game.steps

    @property
    def done(self) -> bool:
        return self.game.done or (self.cap is not None and self.game.steps >= self.cap)

    def step(self, action: str):
        return self.game.step(action), self.done

    def render(self):
        return self.game.render()

    def frame(self) -> np.ndarray:
        return self.game.frame()

    def expert(self) -> str:
        return self.game.expert()

    def clone(self) -> "ControlAdapter":
        if not self.searchable:
            raise TypeError("%s cannot be cloned (its env does not survive deepcopy)" % self.game.game)
        c = ControlAdapter(self._copy(), self.cap, self.actions, self.seed)
        self._copy_history(c)
        return c

    def close(self) -> None:
        self.game.close()


def _copy_game(game):
    un = game.env.unwrapped
    memo = {id(getattr(un, k)): None for k in ("screen", "clock") if getattr(un, k, None) is not None}
    return copy.deepcopy(game, memo)


def _faithful_copy(name: str) -> bool:
    """Whether a deep copy of a ``name`` episode steps exactly like the original. LunarLander's copy does not
    raise but comes back without its Box2D bodies, so this steps both rather than trusting deepcopy."""
    from laya.controlgames import GAMES, ControlGame

    g = ControlGame(name, 0)
    try:
        a = GAMES[name]["actions"][-1]
        g.step(a)
        c = _copy_game(g)
        ok = all(g.step(a) == c.step(a) and np.array_equal(g.obs, c.obs) for _ in range(3))
        c.close()
        return bool(ok)
    except Exception:
        return False
    finally:
        g.close()


def question_for(spec: GameSpec, frames: str = "single") -> Dict:
    """The fixed question of a grid or control game; the control questions describe the frame mode's screen."""
    from laya.games import control_question, maze_question, snake_question

    if spec.family == "grid":
        return maze_question() if spec.params["game"] == "maze" else snake_question()
    if spec.family == "control":
        return control_question(spec.params["game"], frames)
    raise ValueError("no fixed question for %s" % spec.name)


def make_env(spec: GameSpec, i: int):
    """Episode ``i`` of a grid or control game as an adapter."""
    actions = list(question_for(spec)["action"]["criteria"])
    if spec.family == "grid":
        from laya.gridgames import ACTIONS, make_game

        assert tuple(actions) == ACTIONS
        return GridAdapter(make_game(spec.params["game"], spec.params["size"], spec.seed + i, spec.cap or 0), actions,
                           spec.seed + i)
    from laya.controlgames import GAMES, ControlGame

    assert tuple(actions) == GAMES[spec.params["game"]]["actions"]
    return ControlAdapter(ControlGame(spec.params["game"], spec.seed + i), spec.cap, actions, spec.seed + i)


# ---------------------------------------------------------------------------------------------------------------
# Batched model forward
# ---------------------------------------------------------------------------------------------------------------

_POOL: Optional[ThreadPoolExecutor] = None


def _pool() -> ThreadPoolExecutor:
    global _POOL
    if _POOL is None:
        _POOL = ThreadPoolExecutor(max_workers=max(1, min(16, os.cpu_count() or 1)))
    return _POOL


ProbsFn = Callable[..., np.ndarray]
FORWARD_BATCH = 16  # sequences per model forward: small enough to overlap with the next chunk's preprocessing


def _item(agent, q: Dict, frames: Sequence) -> Dict:
    """One sequence as ``laya.atari_train.action_probs`` builds it (the processor runs here on that backend):
    ``frames`` is the state's images, oldest first; one image is ``{"image": frame}``."""
    from laya.common import QTYPES
    from laya.vlm import build_vlm_inputs

    state = {"image": frames[0]} if len(frames) == 1 else {"images": list(frames)}
    it = build_vlm_inputs(agent.processor, state, q, agent.cfg.get("max_len", 1024), agent.cfg.get("head_max_len", 256))
    it["qtype"] = QTYPES["choice"]
    return it


def _text_item(agent, q: Dict, n_images: int) -> Dict:
    """The sequence for ``n_images`` images whose features come from elsewhere (the frame cache): the ids depend
    only on the image count (``laya.preprocess.prefix_ids``, what the processor writes without splitting), so no
    pixel is touched."""
    from laya.common import QTYPES
    from laya.preprocess import prefix_ids
    from laya.vlm import MASK_PREFIX_TEXT, PREFIX_TEXT, build_vlm_inputs, processor_readout

    if agent.prep.split_edge:
        raise ValueError("the frame cache needs one view per image (image_split_edge 0)")
    text = MASK_PREFIX_TEXT if processor_readout(agent.processor) == "mask" else PREFIX_TEXT
    prefix = {"ids": prefix_ids(agent.processor, text, n_images, agent.prep.image_seq_len), "pixel_values": None,
              "pixel_attention_mask": None, "raw_images": None, "n_images": n_images}
    it = build_vlm_inputs(agent.processor, {}, q, agent.cfg.get("max_len", 1024), agent.cfg.get("head_max_len", 256),
                          prefix=prefix)
    it["qtype"] = QTYPES["choice"]
    return it


def _encode(agent, frames: Sequence[np.ndarray]):
    """Vision tower + connector over raw frames -> ``[n, image_seq_len, d]`` (``laya.atari_train.encode_frames``),
    with the processor's per-frame CPU work spread over the thread pool on that backend."""
    import torch

    from laya.vlm import vlm_prefix

    if agent.prep.on_gpu:
        with torch.no_grad():
            return agent.model.encode_raw_images(list(frames))
    per = list(_pool().map(lambda f: vlm_prefix(agent.processor, [f], agent.prep), frames))
    dev, dtype = agent.device, agent.model.encoder.dtype
    pv = torch.stack([p["pixel_values"][0] for p in per]).to(dev, dtype)
    pam = torch.stack([p["pixel_attention_mask"][0] for p in per]).to(dev)
    with torch.no_grad():
        return agent.model.encode_images(pv, pam)


def _forward(agent, items: List[Dict], k: int, feats=None) -> np.ndarray:
    """``action_probs``'s forward and calibration (no feature cache) over prepared items. Checked on an L4 in
    bf16: ``batched_probs`` and ``action_probs`` on the same 24 frames agree exactly (max difference 0.0)."""
    import torch

    from laya.common import QTYPES, temp_bucket
    from laya.vlm import collate_vlm

    b = collate_vlm(items, agent.processor.tokenizer.pad_token_id, with_pixels=feats is None)
    dev, dtype = agent.device, agent.model.encoder.dtype
    if feats is not None:
        pix = dict(image_hidden_states=feats)
    elif b["pixel_values"] is not None:
        pix = dict(pixel_values=b["pixel_values"].to(dev, dtype), pixel_attention_mask=b["pixel_attention_mask"].to(dev))
    else:
        pix = dict(raw_pixels=b["raw_pixels"].to(dev), image_mask=b["image_mask"].to(dev))
    with torch.no_grad():
        logits, _ = agent.model(
            b["input_ids"].to(dev), b["attention_mask"].to(dev), b["marker_pos"].to(dev), b["marker_mask"].to(dev),
            b["qtype"].to(dev), option_span=b["option_span"].to(dev), **pix,
        )
    t = agent.temperature_by_options.get(temp_bucket(QTYPES["choice"], k), agent.temperature[QTYPES["choice"]])
    return torch.softmax(logits[:, :k].float() / max(1e-3, float(t)), -1).cpu().numpy()


def batched_probs(agent, frames: Sequence, question: Dict, prev_frames: Optional[Sequence] = None,
                  cache=None) -> np.ndarray:
    """The calibrated option probabilities (question order) for many frames, as
    ``laya.atari_train.action_probs(agent, frames, question, prev_frames)`` computes them -- the same sequences,
    forward and temperature, so the same answer as ``predict`` with one option order -- but fast for big batches.

    Each entry of ``frames`` is one state: a frame (one image) or a list of frames (``{"images": [...]}``, oldest
    first); ``prev_frames`` makes the states ``[prev, frame]`` (old callers). With a ``cache``
    (``laya.preprocess.FrameFeatureCache``; the states must then all have the same number of images) the vision
    tower runs only on frames the cache has not seen, and the language model gets their features: stack-N play
    encodes each frame once.

    With the Hugging Face processor backend (the released checkpoint) preprocessing costs 30-45 ms of CPU per
    frame (measured on a Modal L4 host; the forward is ~10 ms per frame there) and would dominate, so every frame's sequence is built on a thread pool and the model runs a forward of
    ``FORWARD_BATCH`` sequences as soon as they are ready, overlapping the GPU with the rest of the preprocessing.
    The device-side backend has no CPU work to spread. ``question`` is one question definition
    (``{"type": "choice", ...}``)."""
    from laya.common import render_options
    from laya.vlm import VLMAgent

    q = VLMAgent._to_internal(question)
    k = len(render_options(q))
    n = len(frames)
    if n == 0:
        return np.zeros((0, k), np.float32)
    states = [[np.asarray(g) for g in f] if isinstance(f, (list, tuple)) else [np.asarray(f)] for f in frames]
    if prev_frames is not None:
        states = [[np.asarray(p)] + st for p, st in zip(prev_frames, states)]
    if cache is not None:
        import torch

        m = len(states[0])
        if any(len(st) != m for st in states):
            raise ValueError("cached states must all have the same number of images")
        item = _text_item(agent, q, m)
        flat = [f for st in states for f in st]
        feats = torch.stack(cache.features(lambda fs: _encode(agent, fs), flat))
        return np.concatenate([_forward(agent, [item] * (min(n, s + FORWARD_BATCH) - s), k,
                                        feats[s * m:min(n, s + FORWARD_BATCH) * m])
                               for s in range(0, n, FORWARD_BATCH)], 0)
    if agent.prep.on_gpu:
        get = lambda j: _item(agent, q, states[j])  # noqa: E731
    else:
        futures = [_pool().submit(_item, agent, q, st) for st in states]
        get = lambda j: futures[j].result()  # noqa: E731
    return np.concatenate([_forward(agent, [get(j) for j in range(s, min(n, s + FORWARD_BATCH))], k)
                           for s in range(0, n, FORWARD_BATCH)], 0)


def frame_cache(mode: str, family: str):
    """The encoder-feature cache play uses in ``mode``: one for ``stack-N`` (N > 1), holding each frame's features
    for the N steps it stays in the window; None otherwise (one image per state, nothing repeats)."""
    from laya.preprocess import FrameFeatureCache

    kind, n = F.resolve(mode, family)
    return FrameFeatureCache(keep=n) if kind == "stack" and n > 1 else None


def state_input(history: Sequence, mode: str, family: str):
    """What ``batched_probs`` gets for one episode: its frame history's state in ``mode`` as a frame (one image)
    or a list of frames (a stack)."""
    imgs = [np.asarray(x) for x in F.state_images(F.state(history, mode, family))]
    return imgs[0] if len(imgs) == 1 else imgs


def move_fn(agent, question: Dict, mode: str, family: str, probs_fn: ProbsFn = None):
    """One decision of one episode exactly as play makes it, for timing (the harness's per-move latency):
    ``move(history) -> probs`` builds the state from the frame history (blending a trail), then runs the forward,
    with the mode's frame cache kept across calls, so consecutive histories of one episode encode only the new
    frame (the steady state of stack-N play)."""
    probs_fn = probs_fn or batched_probs
    cache = frame_cache(mode, family)
    q = question["action"]

    def move(history):
        x = state_input(history, mode, family)
        return probs_fn(agent, [x], q, cache=cache) if cache is not None else probs_fn(agent, [x], q)

    return move




def _key(x) -> tuple:
    frames = x if isinstance(x, list) else [x]
    return tuple((f.shape, hashlib.blake2b(f.tobytes(), digest_size=16).digest()) for f in frames)


def greedy_policy(agent, question: Dict, probs_fn: ProbsFn = batched_probs, mode: str = "single"):
    """``policy(envs) -> actions``: one batched forward over every env's state, argmax per env.

    In the ``single`` frame mode the state is the rendered screen (``render()``: the control games ghost the
    previous frame); in any other mode it is ``laya.frames.state`` of the env's frame history. The question is
    fixed, so the answer is a function of the state alone: each distinct state goes to the model once per game and
    repeats reuse its action. A maze agent that walks into a wall sees the same screen again and would bump it
    until the cap; this makes those steps free (and the answer to a state consistent across batch compositions).
    ``policy.frames`` counts the states actually sent to the model."""
    q = question["action"]
    memo: Dict = {}
    cache = None

    def policy(envs):
        nonlocal cache
        if mode == "single":
            frames = [np.asarray(e.render()) for e in envs]
        else:
            frames = [state_input(e.history(), mode, e.family) for e in envs]
            if cache is None and envs:
                cache = frame_cache(mode, envs[0].family)
        keys = [_key(f) for f in frames]
        new, seen = [], set()
        for j, k in enumerate(keys):
            if k not in memo and k not in seen:
                seen.add(k)
                new.append((k, j))
        if new:
            batch = [frames[j] for _, j in new]
            p = probs_fn(agent, batch, q, cache=cache) if cache is not None else probs_fn(agent, batch, q)
            for (k, _), row in zip(new, p):
                memo[k] = int(row.argmax())
            policy.frames += len(new)
        return [e.actions[memo[k]] for e, k in zip(envs, keys)]

    policy.frames = 0
    return policy


def search_policy(agent, question: Dict, settings):
    from laya import search

    return lambda envs: list(search.plan(agent, envs, question, settings))


def expert_env_policy(envs):
    return [e.expert() for e in envs]


def random_env_policy():
    """Uniform random with one ``random.Random(episode seed)`` per episode, so the draws do not depend on batching."""
    rngs: Dict[int, random.Random] = {}

    def policy(envs):
        return [rngs.setdefault(e.seed, random.Random(e.seed)).choice(e.actions) for e in envs]

    return policy


# ---------------------------------------------------------------------------------------------------------------
# Lockstep play
# ---------------------------------------------------------------------------------------------------------------


def play_lockstep(spec: GameSpec, policy, envs: Optional[List] = None) -> Dict:
    """Play every episode of a grid or control game together: each round, ``policy(live_envs)`` returns one action
    per live env (one batched call), and every live env steps once. Returns per-episode scores and steps."""
    envs = envs if envs is not None else [make_env(spec, i) for i in range(spec.episodes)]
    rounds, decisions = 0, 0
    while True:
        live = [e for e in envs if not e.done]
        if not live:
            break
        acts = policy(live)
        if len(acts) != len(live):
            raise ValueError("policy returned %d actions for %d envs" % (len(acts), len(live)))
        for e, a in zip(live, acts):
            if a not in e.actions:
                raise ValueError("policy returned %r, not one of %s" % (a, e.actions))
            e.step(a)
        rounds += 1
        decisions += len(live)
    scores = [e.score for e in envs]
    steps = [e.steps for e in envs]
    for e in envs:
        if hasattr(e, "close"):
            e.close()
    return {"scores": scores, "steps": steps, "mean": float(np.mean(scores)), "rounds": rounds,
            "decisions": decisions}


def atari_model_policy(agent, game: str, actions: Sequence[str], probs_fn: ProbsFn = batched_probs,
                       mode: Optional[str] = None):
    """``laya.atari_train.model_policy`` (greedy, frame mode from the checkpoint) with the batched forward. In a
    mode other than ``single`` the policy asks ``play`` for each episode's observation history (``hists``)."""
    from laya.games import atari_question

    q = atari_question(game, actions)["action"]
    mode = F.mode_for(getattr(agent, "cfg", None), "atari") if mode is None else mode
    cache = frame_cache(mode, "atari")

    def policy(obs, prevs, ids=None, hists=None):
        if mode == "single":
            p = probs_fn(agent, obs, q)
        else:
            x = [state_input(h, mode, "atari") for h in hists]
            p = probs_fn(agent, x, q, cache=cache) if cache is not None else probs_fn(agent, x, q)
        return [int(r.argmax()) for r in p]

    policy.wants_history = mode != "single"
    return policy


def play_atari(spec: GameSpec, policy) -> Dict:
    from laya.atari_train import play

    res = play(spec.params["game"], policy, spec.episodes, spec.cap, spec.seed)
    return {"scores": [float(s) for s in res["scores"]], "steps": res["steps"], "mean": res["mean_score"],
            "decisions": int(sum(res["steps"])), "actions": res["actions"]}


def _doom_game(scenario: str, labels: bool):
    import vizdoom as vzd

    g = vzd.DoomGame()
    g.load_config(os.path.join(vzd.scenarios_path, scenario + ".cfg"))
    g.set_window_visible(False)
    g.set_screen_format(vzd.ScreenFormat.RGB24)
    g.set_screen_resolution(vzd.ScreenResolution.RES_320X240)
    g.set_labels_buffer_enabled(labels)
    g.set_sound_enabled(False)
    g.init()
    return g


def play_doom(spec: GameSpec, policy: str, agent=None, probs_fn: ProbsFn = batched_probs,
              mode: Optional[str] = None) -> Dict:
    """ViZDoom ``basic`` as ``modal_app.play_doom`` plays it (320x240 RGB, ``DOOM_TICS`` tics per decision,
    ``doom_question``), but ``DOOM_PARALLEL`` instances step in lockstep so the model sees a batch per step.
    Episode ``i`` is ``set_seed(seed + i)`` on whichever instance plays it, so results do not depend on the
    batching. ``policy``: ``model``, ``expert`` (labels-buffer script, ATTACK when no monster is visible) or
    ``random``. The model plays in ``mode`` (default: the checkpoint's), from each episode's screens at its
    decision points (``DOOM_TICS`` apart), as ``prepare_doom_basic`` records them."""
    from laya.games import doom_basic_expert, doom_buttons, doom_question

    scenario = spec.params["scenario"]
    games = [_doom_game(scenario, labels=policy == "expert") for _ in range(min(DOOM_PARALLEL, spec.episodes))]
    buttons = doom_buttons(games[0])
    one_hot = {b: [i == j for j in range(len(buttons))] for i, b in enumerate(buttons)}
    q = doom_question(scenario, buttons)["action"]
    scores, steps = [0.0] * spec.episodes, [0] * spec.episodes
    todo = list(range(spec.episodes))
    slot: Dict[int, int] = {}  # game instance -> episode it is playing
    rngs = {i: random.Random(spec.seed + i) for i in range(spec.episodes)}
    if policy == "model":
        mode = F.mode_for(getattr(agent, "cfg", None), "doom") if mode is None else mode
        cache = frame_cache(mode, "doom")
    hist: Dict[int, List] = {}  # game instance -> its episode's screens so far

    def start(k):
        hist[k] = []
        if todo:
            ep = todo.pop(0)
            games[k].set_seed(spec.seed + ep)
            games[k].new_episode()
            slot[k] = ep
        else:
            slot.pop(k, None)

    for k in range(len(games)):
        start(k)
    while slot:
        live = sorted(slot)
        states = [games[k].get_state() for k in live]
        if policy == "model":
            if mode == "single":
                p = probs_fn(agent, [s.screen_buffer for s in states], q)
            else:
                for k, s in zip(live, states):
                    hist[k] = (hist[k] + [s.screen_buffer])[-F.MAX_FRAMES:]
                x = [state_input(hist[k], mode, "doom") for k in live]
                p = probs_fn(agent, x, q, cache=cache) if cache is not None else probs_fn(agent, x, q)
            acts = [buttons[int(r.argmax())] for r in p]
        elif policy == "expert":
            acts = [doom_basic_expert(s.labels) or "ATTACK" for s in states]
        else:
            acts = [rngs[slot[k]].choice(buttons) for k in live]
        for k, a in zip(live, acts):
            ep = slot[k]
            games[k].make_action(one_hot[a], DOOM_TICS)
            steps[ep] += 1
            if games[k].is_episode_finished() or (spec.cap and steps[ep] >= spec.cap):
                scores[ep] = float(games[k].get_total_reward())
                start(k)
    for g in games:
        g.close()
    return {"scores": scores, "steps": steps, "mean": float(np.mean(scores)), "decisions": int(sum(steps))}


# ---------------------------------------------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------------------------------------------


def load_baselines(path: str = BASELINES_PATH) -> Dict[str, Dict]:
    with open(path) as f:
        return json.load(f)["games"]


def check_baseline(spec: GameSpec, base: Optional[Dict]) -> Dict:
    """The stored baseline for ``spec``, which must have been measured on the same episodes, seeds and cap."""
    if not base or base.get("random") is None or base.get("expert") is None:
        raise KeyError("no random/expert baseline for %s in game_baselines.json" % spec.name)
    want = {"episodes": spec.episodes, "seeds": spec.seeds, "cap": spec.cap}
    got = {k: base.get(k) for k in want}
    if got != want:
        raise ValueError("stale baseline for %s: stored %s, suite %s (re-run the baselines)" % (spec.name, got, want))
    return base


def normalize(model: float, rnd: float, expert: float) -> Optional[float]:
    """(model - random) / (expert - random), clipped to [CLIP_LO, CLIP_HI]; None when the baselines tie."""
    if expert == rnd:
        return None
    return float(min(CLIP_HI, max(CLIP_LO, (model - rnd) / (expert - rnd))))


def play_model(spec: GameSpec, agent, probs_fn: ProbsFn = batched_probs) -> Dict:
    """One game with the model, in the checkpoint's frame mode; adds ``search`` (whether the search hook chose
    the actions) and ``frames`` (the mode)."""
    mode = F.mode_for(getattr(agent, "cfg", None), spec.family)
    if spec.family in ("grid", "control"):
        q = question_for(spec, mode)
        settings = (getattr(agent, "cfg", None) or {}).get("search")
        use_search = bool(settings) and make_env(spec, 0).searchable
        if use_search and mode != "single":
            raise ValueError("search plays the single frame mode only, not %r" % mode)
        policy = search_policy(agent, q, settings) if use_search else greedy_policy(agent, q, probs_fn, mode)
        out = play_lockstep(spec, policy)
        out["search"] = use_search
        out["model_frames"] = getattr(policy, "frames", None)
    elif spec.family == "atari":
        from laya.atari_train import game_actions

        out = play_atari(spec, atari_model_policy(agent, spec.params["game"], game_actions(spec.params["game"]),
                                                  probs_fn, mode))
    else:
        out = play_doom(spec, "model", agent, probs_fn, mode)
    out["frames"] = mode
    return out


def run_family(agent, family: str, suite: Dict[str, GameSpec] = SUITE, baselines: Optional[Dict[str, Dict]] = None,
               probs_fn: ProbsFn = batched_probs) -> Dict[str, Dict]:
    """Play every ``family`` game of ``suite`` with ``agent`` and score it against the stored baselines:
    ``{game: {"model", "normalized", "episodes", "seconds", ...}}``."""
    global _POOL
    baselines = load_baselines() if baselines is None else baselines
    out = {}
    try:
        _run_games(agent, family, suite, baselines, probs_fn, out)
    finally:
        if _POOL is not None:
            _POOL.shutdown()
            _POOL = None
    return out


def _run_games(agent, family, suite, baselines, probs_fn, out) -> None:
    specs = family_games(family, suite)
    bases = {s.name: check_baseline(s, baselines.get(s.name)) for s in specs}  # fail before playing anything
    for spec in specs:
        base = bases[spec.name]
        t0 = time.time()
        res = play_model(spec, agent, probs_fn)
        out[spec.name] = {"model": res["mean"], "normalized": normalize(res["mean"], base["random"], base["expert"]),
                          "episodes": spec.episodes, "seconds": round(time.time() - t0, 1),
                          "random": base["random"], "expert": base["expert"], "scores": res["scores"],
                          "decisions": res["decisions"], "model_frames": res.get("model_frames", res["decisions"]),
                          "search": res.get("search", False), "frames": res.get("frames", "single")}


def summarize(results: Dict[str, Dict], suite: Dict[str, GameSpec] = SUITE) -> Dict:
    """``{"games": mean normalized over the games present, "per_game": {game: normalized}, "missing": [...]}``.
    ``complete`` is False when a suite game is missing (a family failed); ``games`` then averages fewer votes."""
    per = {g: r["normalized"] for g, r in results.items() if r.get("normalized") is not None}
    missing = [g for g in suite if g not in per]
    return {"games": float(np.mean(list(per.values()))) if per else float("nan"), "per_game": per,
            "missing": missing, "complete": not missing}


# ---------------------------------------------------------------------------------------------------------------
# Baselines (run once; the eval never plays these)
# ---------------------------------------------------------------------------------------------------------------


def play_reference(spec: GameSpec, policy: str) -> Dict:
    """``random`` or ``expert`` on a grid, control or Doom game, on the suite's episodes."""
    if spec.family == "doom":
        return play_doom(spec, policy)
    if spec.family == "atari":
        if policy == "expert":
            return atari_expert_scores(spec)
        from laya.atari_train import game_actions, random_policy

        return play_atari(spec, random_policy(len(game_actions(spec.params["game"])), spec.seed))
    return play_lockstep(spec, expert_env_policy if policy == "expert" else random_env_policy())


def atari_expert_scores(spec: GameSpec) -> Dict:
    """The CleanRL PPO expert (``laya.atari_data.expert``, greedy) on the suite's seeds and cap: the same v5 env
    (sticky actions 0.25, 4 frames per decision, auto-FIRE, the cap counting decisions only) as ``play``.
    Needs jax, flax and opencv (``modal_atari_expert.py``'s image)."""
    from laya.atari_data import expert as X

    player, pol = X.Player(spec.params["game"]), X.Policy(X.agent_repo(spec.params["game"]))
    rng = np.random.default_rng(0)
    scores, steps = [], []
    for i in range(spec.episodes):
        scores.append(float(X.play(player, pol, "greedy", spec.seed + i, rng, max_steps=spec.cap)))
        steps.append(player.steps)  # decisions plus auto-FIRE presses
    return {"scores": scores, "steps": steps, "mean": float(np.mean(scores)), "decisions": int(sum(steps))}


def baseline_entry(spec: GameSpec, rnd: Dict, exp: Dict, note: str = "") -> Dict:
    e = {"random": rnd["mean"], "expert": exp["mean"], "episodes": spec.episodes, "seeds": spec.seeds,
         "cap": spec.cap, "family": spec.family, "score": spec.score,
         "random_scores": rnd["scores"], "expert_scores": exp["scores"]}
    if note:
        e["note"] = note
    return e


def measure_baselines(family: str, suite: Dict[str, GameSpec] = SUITE) -> Dict[str, Dict]:
    out = {}
    for spec in family_games(family, suite):
        t0 = time.time()
        out[spec.name] = baseline_entry(spec, play_reference(spec, "random"), play_reference(spec, "expert"))
        print("%-12s random %8.2f expert %8.2f (%.0fs)" % (spec.name, out[spec.name]["random"],
                                                          out[spec.name]["expert"], time.time() - t0))
    return out


def write_baselines(entries: Dict[str, Dict], path: str = BASELINES_PATH) -> None:
    """Merge ``entries`` into the baselines file."""
    doc = {"about": "random and expert reference scores for autoresearch/games_eval.py SUITE, measured on the "
                    "same episode seeds and caps as the model plays; normalized = (model - random) / "
                    "(expert - random)", "games": {}}
    if os.path.exists(path):
        with open(path) as f:
            doc = json.load(f)
    doc["games"].update(entries)
    doc["games"] = {k: doc["games"][k] for k in sorted(doc["games"], key=lambda g: list(SUITE).index(g)
                                                        if g in SUITE else len(SUITE))}
    text = json.dumps(doc, indent=1)
    text = re.sub(r"\[\s*([-0-9.,\s]*?)\s*\]", lambda m: "[" + " ".join(m.group(1).split()) + "]", text)  # one-line lists
    with open(path, "w") as f:
        f.write(text + "\n")


def main(argv: Sequence[str]) -> None:
    if len(argv) >= 1 and argv[0] == "baselines":
        fams = argv[2].split(",") if len(argv) >= 3 and argv[1] == "--families" else ["grid", "control"]
        for fam in fams:
            write_baselines(measure_baselines(fam))
        print("wrote", BASELINES_PATH)
    elif len(argv) >= 1 and argv[0] == "suite":
        for s in SUITE.values():
            print(json.dumps(dict(asdict(s), seeds=s.seeds)))
    else:
        raise SystemExit("usage: games_eval.py baselines [--families grid,control,doom] | suite")


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(HERE))
    main(sys.argv[1:])
