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
below (``clone``, ``step``, ``render``, ``actions``, ``done``).

Seeds: episode ``i`` of a game uses seed ``SEED_BASE[family] + i`` (see ``SUITE``). They lie in 700,000-799,999,
off every range the repo trains or evaluates on: Atari expert data 1,000+ (val) / 2,000+ (train) and its baselines
900,000+ / 950,000+; ViZDoom data 0+ (train) / 1,000,000+ (val); the public ``games_eval`` suite 50,000 (Doom),
100,000 (Atari) and 200,000 (grid, control). No grid or control training data exists.

Baselines: ``python autoresearch/games_eval.py baselines --families grid,control[,doom]`` plays random and expert
and merges them into ``game_baselines.json``; the Atari expert (CleanRL PPO, ``laya.atari_data.expert``) needs
jax / flax / opencv and is measured by ``atari_expert_scores`` under the same seeds and cap.
"""
import copy
import json
import math
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

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
    GameSpec("Maze6", "grid", 128, SEED_BASE["grid"] + 10_000, None, "solve rate", {"game": "maze", "size": 6}),
    GameSpec("Snake10", "grid", 64, SEED_BASE["grid"] + 20_000, 200, "food eaten", {"game": "snake", "size": 10}),
    GameSpec("CartPole", "control", 32, SEED_BASE["control"], 200, "reward", {"game": "CartPole"}),
    GameSpec("Acrobot", "control", 32, SEED_BASE["control"] + 5_000, 200, "reward", {"game": "Acrobot"}),
    GameSpec("MountainCar", "control", 32, SEED_BASE["control"] + 10_000, 200, "reward", {"game": "MountainCar"}),
    GameSpec("LunarLander", "control", 32, SEED_BASE["control"] + 15_000, 300, "reward", {"game": "LunarLander"}),
    GameSpec("Freeway", "atari", 4, SEED_BASE["atari"], 1000, "game score", {"game": "Freeway"}),
    GameSpec("Breakout", "atari", 4, SEED_BASE["atari"] + 1_000, 1000, "game score", {"game": "Breakout"}),
    GameSpec("DoomBasic", "doom", 32, SEED_BASE["doom"], None, "reward", {"scenario": "basic"}),
)}


def family_games(family: str, suite: Dict[str, GameSpec] = SUITE) -> List[GameSpec]:
    if family not in FAMILIES:
        raise ValueError("unknown family %r (%s)" % (family, ", ".join(FAMILIES)))
    return [s for s in suite.values() if s.family == family]


# ---------------------------------------------------------------------------------------------------------------
# Adapters: the protocol laya.search.plan works against
# ---------------------------------------------------------------------------------------------------------------


class GridAdapter:
    """A ``laya.gridgames`` Maze or Snake episode. ``score`` is the benchmark metric for this episode (1 when the
    maze is solved; food eaten in Snake) and ``step`` returns its change as the reward."""

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

    def expert(self) -> str:
        return self.env.expert()

    def clone(self) -> "GridAdapter":
        return GridAdapter(copy.deepcopy(self.env), self.actions, self.seed)


class ControlAdapter:
    """A ``laya.controlgames.ControlGame`` episode capped at ``cap`` steps; the reward is the environment's.
    Cloning deep-copies the Gymnasium env minus its pygame surface and clock (recreated on the next render).
    A game whose env cannot be deep-copied (LunarLander: its Box2D world) is not ``searchable``: it is played
    greedy, and ``clone`` raises TypeError."""

    _clonable: Dict[str, bool] = {}  # game -> whether its env survives deepcopy (probed once)

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

    def expert(self) -> str:
        return self.game.expert()

    def clone(self) -> "ControlAdapter":
        if not self.searchable:
            raise TypeError("%s cannot be cloned (its env does not survive deepcopy)" % self.game.game)
        return ControlAdapter(self._copy(), self.cap, self.actions, self.seed)

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


def question_for(spec: GameSpec) -> Dict:
    from laya.games import control_question, maze_question, snake_question

    if spec.family == "grid":
        return maze_question() if spec.params["game"] == "maze" else snake_question()
    if spec.family == "control":
        return control_question(spec.params["game"])
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


def batched_probs(agent, frames: Sequence, question: Dict, prev_frames: Optional[Sequence] = None) -> np.ndarray:
    """``laya.atari_train.action_probs`` over many frames: the calibrated option probabilities (question order),
    exactly what ``predict`` computes with one option order.

    With the Hugging Face processor backend (the released checkpoint) preprocessing is ~20-35 ms of CPU per frame
    and dominates, so the frames are split into chunks that run ``action_probs`` on a thread pool: the processor
    work runs in parallel and the GPU forwards interleave. The device-side backend has no CPU work, so it runs
    chunks of 64 one after another. ``question`` is one question definition (``{"type": "choice", ...}``).
    """
    from laya.atari_train import action_probs

    frames = [np.asarray(f) for f in frames]
    prev = None if prev_frames is None else [np.asarray(f) for f in prev_frames]
    n = len(frames)
    if n == 0:
        return np.zeros((0, len(question["criteria"])), np.float32)
    if agent.prep.on_gpu:
        size, runner = 64, map
    else:
        workers = _pool()._max_workers
        size, runner = max(1, min(32, math.ceil(n / workers))), _pool().map
    starts = list(range(0, n, size))
    outs = runner(lambda s: action_probs(agent, frames[s:s + size], question,
                                         None if prev is None else prev[s:s + size]), starts)
    return np.concatenate(list(outs), 0)


ProbsFn = Callable[..., np.ndarray]


def greedy_policy(agent, question: Dict, probs_fn: ProbsFn = batched_probs):
    """``policy(envs) -> actions``: one batched forward over every env's rendered screen, argmax per env."""
    q = question["action"]

    def policy(envs):
        p = probs_fn(agent, [np.asarray(e.render()) for e in envs], q)
        return [e.actions[int(row.argmax())] for e, row in zip(envs, p)]

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


def atari_model_policy(agent, game: str, actions: Sequence[str], probs_fn: ProbsFn = batched_probs):
    """``laya.atari_train.model_policy`` (greedy, frame count from the checkpoint) with the batched forward."""
    from laya.games import atari_question

    q = atari_question(game, actions)["action"]
    two = int(agent.cfg.get("atari_frames", 1)) == 2

    def policy(obs, prevs, ids=None):
        return [int(r.argmax()) for r in probs_fn(agent, obs, q, prevs if two else None)]

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


def play_doom(spec: GameSpec, policy: str, agent=None, probs_fn: ProbsFn = batched_probs) -> Dict:
    """ViZDoom ``basic`` as ``modal_app.play_doom`` plays it (320x240 RGB, ``DOOM_TICS`` tics per decision,
    ``doom_question``), but ``DOOM_PARALLEL`` instances step in lockstep so the model sees a batch per step.
    Episode ``i`` is ``set_seed(seed + i)`` on whichever instance plays it, so results do not depend on the
    batching. ``policy``: ``model``, ``expert`` (labels-buffer script, ATTACK when no monster is visible) or
    ``random``."""
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

    def start(k):
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
            p = probs_fn(agent, [s.screen_buffer for s in states], q)
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
    """One game with the model; adds ``search`` (whether the search hook chose the actions)."""
    if spec.family in ("grid", "control"):
        q = question_for(spec)
        settings = (getattr(agent, "cfg", None) or {}).get("search")
        use_search = bool(settings) and make_env(spec, 0).searchable
        policy = search_policy(agent, q, settings) if use_search else greedy_policy(agent, q, probs_fn)
        out = play_lockstep(spec, policy)
        out["search"] = use_search
        return out
    if spec.family == "atari":
        from laya.atari_train import game_actions

        return play_atari(spec, atari_model_policy(agent, spec.params["game"], game_actions(spec.params["game"]),
                                                   probs_fn))
    return play_doom(spec, "model", agent, probs_fn)


def run_family(agent, family: str, suite: Dict[str, GameSpec] = SUITE, baselines: Optional[Dict[str, Dict]] = None,
               probs_fn: ProbsFn = batched_probs) -> Dict[str, Dict]:
    """Play every ``family`` game of ``suite`` with ``agent`` and score it against the stored baselines:
    ``{game: {"model", "normalized", "episodes", "seconds", ...}}``."""
    baselines = load_baselines() if baselines is None else baselines
    out = {}
    for spec in family_games(family, suite):
        base = check_baseline(spec, baselines.get(spec.name))
        t0 = time.time()
        res = play_model(spec, agent, probs_fn)
        out[spec.name] = {"model": res["mean"], "normalized": normalize(res["mean"], base["random"], base["expert"]),
                          "episodes": spec.episodes, "seconds": round(time.time() - t0, 1),
                          "random": base["random"], "expert": base["expert"], "scores": res["scores"],
                          "decisions": res["decisions"], "search": res.get("search", False)}
    return out


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
