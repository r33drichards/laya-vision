"""``autoresearch/games_eval.py``: lockstep bookkeeping, caps, normalization, adapters and the search hook, on CPU
with stub policies (no model)."""
import dataclasses
import importlib.util
import os
import sys
import types

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location(
    "games_eval", os.path.join(os.path.dirname(__file__), "..", "autoresearch", "games_eval.py"))
ge = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ge)


def small(name, **kw):
    return {name: dataclasses.replace(ge.SUITE[name], **kw)}


class StubAgent:
    def __init__(self, cfg=None):
        self.cfg = cfg or {}


def fixed_probs(index, calls=None):
    """A ``probs_fn`` that always prefers option ``index`` and records each call's batch."""
    def fn(agent, frames, question, prev=None):
        if calls is not None:
            calls.append([np.asarray(f) for f in frames])
        p = np.full((len(frames), len(question["criteria"])), 0.1)
        p[:, index] = 1.0
        return p / p.sum(1, keepdims=True)
    return fn


def base_for(suite):
    return {s.name: ge.baseline_entry(s, ge.play_reference(s, "random"), ge.play_reference(s, "expert"))
            for s in suite.values()}


# -- suite and scoring ----------------------------------------------------------------------------------------


def test_suite_covers_the_objective():
    names = set(ge.SUITE)
    assert {"Maze4", "Maze6", "Snake10", "CartPole", "Acrobot", "MountainCar", "LunarLander", "Freeway",
            "Breakout", "DoomBasic"} == names
    for s in ge.SUITE.values():
        assert s.family in ge.FAMILIES and s.episodes >= 2
        assert 700_000 <= s.seeds[0] and s.seeds[1] < 800_000
    ranges = sorted(tuple(s.seeds) for s in ge.SUITE.values())
    assert all(a[1] < b[0] for a, b in zip(ranges, ranges[1:]))  # no two games share a seed


def test_stored_baselines_match_the_suite():
    stored = ge.load_baselines()
    for s in ge.SUITE.values():
        if s.name in stored:
            ge.check_baseline(s, stored[s.name])
            assert stored[s.name]["expert"] > stored[s.name]["random"]


def test_normalize_clips_and_summarize_averages():
    assert ge.normalize(5, 0, 10) == 0.5
    assert ge.normalize(100, 0, 10) == ge.CLIP_HI
    assert ge.normalize(-100, 0, 10) == ge.CLIP_LO
    assert ge.normalize(-150, -200, -100) == 0.5  # negative-reward games
    assert ge.normalize(3, 1, 1) is None
    res = {"Maze4": {"normalized": 1.5}, "CartPole": {"normalized": -0.5}, "Acrobot": {"normalized": 0.3}}
    out = ge.summarize(res)
    assert out["games"] == pytest.approx(1.3 / 3) and out["per_game"]["Acrobot"] == 0.3
    assert not out["complete"] and "Freeway" in out["missing"]


def test_stale_baseline_is_refused():
    suite = small("Maze4", episodes=4)
    base = base_for(suite)
    ge.check_baseline(suite["Maze4"], base["Maze4"])
    with pytest.raises(ValueError, match="stale"):
        ge.check_baseline(dataclasses.replace(suite["Maze4"], episodes=5), base["Maze4"])
    with pytest.raises(KeyError):
        ge.run_family(StubAgent(), "grid", suite, {}, fixed_probs(0))


# -- lockstep play ------------------------------------------------------------------------------------------


def test_lockstep_expert_matches_sequential_play():
    from laya import gridgames

    spec = dataclasses.replace(ge.SUITE["Snake10"], episodes=6)
    lock = ge.play_lockstep(spec, ge.expert_env_policy)
    seq = [gridgames.Snake(10, spec.seed + i, spec.cap) for i in range(spec.episodes)]
    for g in seq:
        while not g.done:
            g.step(g.expert())
    assert lock["scores"] == [float(g.eaten) for g in seq] and lock["steps"] == [g.steps for g in seq]
    assert max(lock["steps"]) <= spec.cap and lock["decisions"] == sum(lock["steps"])


def test_model_play_is_batched_capped_and_deterministic():
    suite = {**small("Maze4", episodes=8), **small("Snake10", episodes=5, cap=15)}
    base = base_for(suite)
    calls = []
    out = ge.run_family(StubAgent(), "grid", suite, base, fixed_probs(3, calls))  # always RIGHT
    again = ge.run_family(StubAgent(), "grid", suite, base, fixed_probs(3))
    assert {g: r["scores"] for g, r in out.items()} == {g: r["scores"] for g, r in again.items()}
    maze_calls = [c for c in calls if c[0].shape == calls[0][0].shape]
    assert len(maze_calls[0]) == 8  # the first round asks about every episode in one call
    assert sum(len(c) for c in calls) == out["Maze4"]["decisions"] + out["Snake10"]["decisions"]
    assert all(len(a) >= len(b) for a, b in zip(maze_calls, maze_calls[1:]))  # finished episodes drop out
    # always RIGHT never solves a maze (4x-shortest-path cap) and the snake hits the right wall on step 5
    assert out["Maze4"]["model"] == 0.0 and out["Maze4"]["normalized"] == 0.0
    assert out["Snake10"]["decisions"] == 5 * 5
    assert set(out["Maze4"]) >= {"model", "normalized", "episodes", "seconds"}


def test_policy_sees_exactly_the_rendered_screen():
    spec = dataclasses.replace(ge.SUITE["Maze6"], episodes=3)
    envs = [ge.make_env(spec, i) for i in range(3)]
    seen = []
    ge.greedy_policy(StubAgent(), ge.question_for(spec), fixed_probs(0, seen))(envs)
    assert all(np.array_equal(a, np.asarray(e.render())) for a, e in zip(seen[0], envs))


def test_random_baseline_does_not_depend_on_batching():
    spec = dataclasses.replace(ge.SUITE["Maze4"], episodes=6)
    together = ge.play_lockstep(spec, ge.random_env_policy())
    alone = [ge.play_lockstep(dataclasses.replace(spec, seed=spec.seed + i, episodes=1), ge.random_env_policy())
             for i in range(6)]
    assert together["steps"] == [a["steps"][0] for a in alone]


# -- adapters -----------------------------------------------------------------------------------------------


def _check_clone(env):
    before = np.asarray(env.render()).copy()
    c = env.clone()
    assert np.array_equal(np.asarray(c.render()), before) and c.actions == env.actions
    for _ in range(3):
        c.step(c.actions[0])
    assert np.array_equal(np.asarray(env.render()), before) and env.steps == 0 and c.steps == 3
    r, done = env.step(env.actions[-1])
    assert isinstance(r, float) and isinstance(done, bool) and done == env.done


@pytest.mark.parametrize("game", ["Maze4", "Snake10"])
def test_grid_clone_is_independent(game):
    _check_clone(ge.make_env(ge.SUITE[game], 0))


@pytest.mark.parametrize("game", ["CartPole", "Acrobot", "MountainCar"])
def test_control_clone_is_independent(game):
    pytest.importorskip("gymnasium")
    pytest.importorskip("pygame")
    env = ge.make_env(ge.SUITE[game], 0)
    assert env.searchable
    _check_clone(env)
    # a clone replays identically to the original
    a, b = ge.make_env(ge.SUITE[game], 1), None
    a.step(a.actions[0])
    b = a.clone()
    for k in range(5):
        assert a.step(a.actions[k % 2]) == b.step(b.actions[k % 2])
    assert np.array_equal(np.asarray(a.render()), np.asarray(b.render()))


def test_lunar_lander_is_played_greedy():
    pytest.importorskip("Box2D")
    env = ge.make_env(ge.SUITE["LunarLander"], 0)
    assert not env.searchable
    with pytest.raises(TypeError):
        env.clone()


def test_control_caps():
    pytest.importorskip("gymnasium")
    suite = small("MountainCar", episodes=3, cap=9)
    out = ge.run_family(StubAgent(), "control", suite, base_for(suite), fixed_probs(1))
    assert out["MountainCar"]["decisions"] == 27 and out["MountainCar"]["model"] == -9.0


# -- search hook --------------------------------------------------------------------------------------------


def test_search_hook_gets_adapters(monkeypatch):
    seen = []

    def plan(agent, envs, question, settings):
        seen.append((len(envs), settings, question["action"]["type"]))
        for e in envs:  # a planner may clone and roll out freely
            c = e.clone()
            c.step(c.actions[0])
        return [e.expert() for e in envs]

    import laya

    fake = types.ModuleType("laya.search")
    fake.plan = plan
    monkeypatch.setitem(sys.modules, "laya.search", fake)
    monkeypatch.setattr(laya, "search", fake, raising=False)
    suite = small("Maze4", episodes=4)
    out = ge.run_family(StubAgent({"search": {"depth": 2}}), "grid", suite, base_for(suite), fixed_probs(0))
    assert out["Maze4"]["search"] and out["Maze4"]["model"] == 1.0 and out["Maze4"]["normalized"] == 1.0
    assert seen[0] == (4, {"depth": 2}, "choice")
    if importlib.util.find_spec("Box2D") and importlib.util.find_spec("gymnasium"):
        suite = small("LunarLander", episodes=2, cap=5)
        out = ge.run_family(StubAgent({"search": {"depth": 2}}), "control", suite, base_for(suite), fixed_probs(0))
        assert not out["LunarLander"]["search"]
