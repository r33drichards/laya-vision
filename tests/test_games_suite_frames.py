"""The ``modal_app.py`` games suite (``games_eval`` / ``full_eval --parts games``) plays every family in the
checkpoint's game frame mode (``laya.frames.mode_for``), records it as ``frames``, and sends exactly the old inputs
in ``single``. Stub agents on the CPU; the Modal functions' bodies are called directly."""
import sys
import types

import numpy as np
import pytest

pytest.importorskip("modal")
import modal_app  # noqa: E402

from laya import frames as F  # noqa: E402
from laya.games import control_question, maze_question, snake_question  # noqa: E402

MODES = ("stack-2", "trail-4")


class PredictAgent:
    """Records every ``predict`` state and question; answers with a fixed cycle over the options."""

    def __init__(self, cfg=None):
        self.cfg = cfg or {}
        self.calls, self.choices = [], []

    def predict(self, state, question):
        opts = list(question["action"]["criteria"])
        a = opts[(len(self.calls) * 7 // 3) % len(opts)]
        self.calls.append((state, question))
        self.choices.append(a)
        return {"answers": {"action": {"choice": a}}}


def arrays(state):
    return [np.asarray(x) for x in F.state_images(state)]


def same_state(got, want):
    assert set(got) == set(want)
    a, b = arrays(got), arrays(want)
    assert len(a) == len(b) and all(np.array_equal(x, y) for x, y in zip(a, b))


# -- Maze and Snake -------------------------------------------------------------------------------------------------


def replay_grid(game, size, seed, choices, episodes, max_steps):
    """Each decision's frame history, from twin episodes stepped with the recorded choices."""
    from laya.gridgames import make_game

    hists, it = [], iter(choices)
    for i in range(episodes):
        env, frames = make_game(game, size, seed + i, max_steps), []
        while not env.done:
            frames.append(env.render())
            hists.append(list(frames))
            env.step(next(it))
    return hists


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("game", ["maze", "snake"])
def test_grid_plays_the_checkpoints_mode(game, mode):
    agent = PredictAgent({"game_frames": mode})
    out = modal_app._play_grid(game, "model", "stub", 2, 4 if game == "maze" else 6, 11, 6, agent=agent)
    assert out["frames"] == mode and out["policy"] == "model:stub"
    q = maze_question() if game == "maze" else snake_question()
    hists = replay_grid(game, out["size"], 11, agent.choices, 2, 6)
    assert len(hists) == len(agent.calls) == sum(e["steps"] for e in out["results"])
    kind, n = F.resolve(mode, "grid")
    for (state, question), hist in zip(agent.calls, hists):
        assert question == q
        same_state(state, F.state(hist, mode, "grid"))
        assert ("images" in state) == (kind == "stack") and len(arrays(state)) == (n if kind == "stack" else 1)
    first = arrays(agent.calls[0][0])
    assert all(np.array_equal(f, first[-1]) for f in first)  # an episode starts with its first frame repeated


@pytest.mark.parametrize("game", ["maze", "snake"])
def test_grid_single_is_unchanged(game):
    agent = PredictAgent({})
    out = modal_app._play_grid(game, "model", "stub", 2, 0, 5, 6, agent=agent)
    assert out["frames"] == "single"
    hists = replay_grid(game, out["size"], 5, agent.choices, 2, 6)
    for (state, _), hist in zip(agent.calls, hists):
        assert list(state) == ["image"] and np.array_equal(np.asarray(state["image"]), np.asarray(hist[-1]))


def test_grid_baselines_record_no_mode():
    out = modal_app._play_grid("maze", "random", "", 2, 4, 3, 6)
    assert "frames" not in out


# -- classic control ------------------------------------------------------------------------------------------------

gym = pytest.importorskip("gymnasium")


def replay_control(game, seed, choices, episodes):
    from laya.controlgames import ControlGame

    frames_at, renders, it = [], [], iter(choices)
    for i in range(episodes):
        env, frames = ControlGame(game, seed + i), []
        while not env.done:
            frames.append(env.frame())
            frames_at.append(list(frames))
            renders.append(np.asarray(env.render()))
            env.step(next(it))
        env.close()
    return frames_at, renders


@pytest.mark.parametrize("mode", MODES)
def test_control_plays_the_checkpoints_mode(mode):
    agent = PredictAgent({"game_frames": mode})
    out = modal_app._play_control("CartPole", "stub", 2, 17, agent=agent)
    assert out["frames"] == mode
    hists, _ = replay_control("CartPole", 17, agent.choices, 2)
    assert len(hists) == len(agent.calls)
    kind, n = F.resolve(mode, "control")
    for (state, question), hist in zip(agent.calls, hists):
        assert question == control_question("CartPole", mode)
        same_state(state, F.state(hist, mode, "control"))
        assert len(arrays(state)) == (n if kind == "stack" else 1)
    if mode == "trail-4":  # not the fixed two-frame ghost
        later = [(s, h) for (s, _), h in zip(agent.calls, hists) if len(h) >= 4]
        assert later and not np.array_equal(arrays(later[0][0])[0], F.blend(later[0][1][-2:]))


def test_control_single_is_unchanged_and_baselines_do_not_depend_on_the_mode():
    agent = PredictAgent({})
    out = modal_app._play_control("CartPole", "stub", 2, 17, agent=agent)
    assert out["frames"] == "single"
    _, renders = replay_control("CartPole", 17, agent.choices, 2)
    for (state, question), want in zip(agent.calls, renders):
        assert question == control_question("CartPole")
        assert list(state) == ["image"] and np.array_equal(np.asarray(state["image"]), want)
    other = modal_app._play_control("CartPole", "stub", 2, 17, agent=PredictAgent({"game_frames": "stack-2"}))
    for k in ("random_score", "expert_score", "random_solved", "expert_solved"):
        assert other[k] == out[k]


# -- Atari ----------------------------------------------------------------------------------------------------------


class AtariAgent:
    def __init__(self, cfg=None):
        self.cfg = cfg or {}
        self.prep = types.SimpleNamespace(on_gpu=False)


@pytest.fixture
def atari_calls(monkeypatch):
    pytest.importorskip("ale_py")
    import laya.atari_train as AT

    calls = []

    def fake(agent, frames, question, prev_frames=None, return_act=False, cache=None, stacks=None):
        calls.append({"frames": [np.asarray(f) for f in frames], "prev": prev_frames, "stacks": stacks})
        p = np.full((len(frames), len(question["criteria"])), 0.1)
        p[:, 3 % p.shape[1]] = 1.0
        return p / p.sum(1, keepdims=True)

    monkeypatch.setattr(AT, "action_probs", fake)
    return calls


@pytest.mark.parametrize("cfg, mode", [({"game_frames": "stack-2"}, "stack-2"), ({"game_frames": "trail-4"}, "trail-4"),
                                       ({}, "single"), ({"atari_frames": 2}, "stack-2")])
def test_atari_plays_the_checkpoints_mode(atari_calls, cfg, mode):
    out = modal_app._play_atari_game("Breakout", "stub", 1, 6, 100_000, 1, agent=AtariAgent(cfg))
    assert out["frames"] == mode
    assert len(atari_calls) == 6
    kind, n = F.resolve(mode, "atari")
    for c in atari_calls:
        if kind == "stack" and n > 1:
            assert c["stacks"] is not None and all(len(s) == n for s in c["stacks"])
            assert all(np.array_equal(s[-1], f) for s, f in zip(c["stacks"], c["frames"]))
        else:
            assert c["stacks"] is None and c["frames"][0].shape == (210, 160, 3)
    if kind == "stack":  # the episode's first decision: the start frame repeated
        s0 = atari_calls[0]["stacks"][0] if n > 1 else [atari_calls[0]["frames"][0]]
        assert all(np.array_equal(f, s0[-1]) for f in s0)


def test_atari_single_is_the_current_frame_and_random_does_not_depend_on_the_mode(atari_calls):
    from laya.atari_train import play

    single = modal_app._play_atari_game("Breakout", "stub", 1, 5, 100_000, 2, agent=AtariAgent({}))
    obs = []
    play("Breakout", lambda o, p, ids=None: obs.extend(np.asarray(x) for x in o) or [3] * len(o), 1, 5, 100_000)
    got = [f for c in atari_calls for f in c["frames"]]
    assert len(got) == len(obs) and all(np.array_equal(a, b) for a, b in zip(got, obs))
    stacked = modal_app._play_atari_game("Breakout", "stub", 1, 5, 100_000, 2, agent=AtariAgent({"game_frames": "stack-2"}))
    assert stacked["random_score"] == single["random_score"]


# -- ViZDoom --------------------------------------------------------------------------------------------------------


class FakeDoom:
    """Just enough of ``vizdoom.DoomGame``: the screen is a counter of the episode's tics."""

    class _B:
        def __init__(self, n):
            self.n = n

        def __str__(self):
            return "Button." + self.n

    def __init__(self):
        self.t, self.seed = 0, 0

    def get_available_buttons(self):
        return [self._B(b) for b in ("MOVE_LEFT", "MOVE_RIGHT", "ATTACK")]

    def set_seed(self, s):
        self.seed = s

    def new_episode(self):
        self.t = 0

    def get_state(self):
        return types.SimpleNamespace(screen_buffer=np.full((6, 8, 3), (self.seed % 50) * 5 + self.t, np.uint8),
                                     labels=[])

    def make_action(self, a, tics):
        self.t += 1

    def is_episode_finished(self):
        return self.t >= 3 + self.seed % 3

    def get_total_reward(self):
        return float(self.t)

    def get_game_variable(self, v):
        return 0

    def close(self):
        pass


@pytest.fixture
def fake_doom(monkeypatch):
    vzd = types.ModuleType("vizdoom")
    vzd.GameVariable = types.SimpleNamespace(KILLCOUNT="KILLCOUNT")
    monkeypatch.setitem(sys.modules, "vizdoom", vzd)
    monkeypatch.setattr(modal_app, "_doom_game", lambda scenario="basic", labels=False: FakeDoom())


@pytest.mark.parametrize("mode", ("single",) + MODES)
def test_doom_plays_the_checkpoints_mode(fake_doom, mode):
    agent = PredictAgent({"game_frames": mode} if mode != "single" else {})
    out = modal_app._play_doom("model", "stub", 4, 4, 50_000, agent=agent)
    assert out["frames"] == mode
    lengths = [3 + (50_000 + ep) % 3 for ep in range(4)]
    assert len(agent.calls) == sum(lengths)
    kind, n = F.resolve(mode, "doom")
    j = 0
    for ep, length in enumerate(lengths):
        base = ((50_000 + ep) % 50) * 5
        for t in range(length):
            state, _ = agent.calls[j]
            j += 1
            want = [np.full((6, 8, 3), base + k, np.uint8) for k in range(t + 1)]  # this episode's screens only
            same_state(state, F.state(want, mode, "doom"))
            if mode == "single":
                assert list(state) == ["image"] and np.array_equal(np.asarray(state["image"]), want[-1])
            elif kind == "stack":
                assert len(state["images"]) == n


def test_doom_baselines_record_no_mode(fake_doom):
    out = modal_app._play_doom("random", "", 3, 4, 50_000)
    assert "frames" not in out and out["episodes"] == 3


# -- the report -----------------------------------------------------------------------------------------------------


def test_eval_report_shows_the_mode():
    import importlib.util
    import os

    spec = importlib.util.spec_from_file_location("eval_report", os.path.join(os.path.dirname(__file__), "..", "scripts",
                                                                              "eval_report.py"))
    er = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(er)
    old = {"atari": [{"game": "Breakout", "frames": 1}], "doom": {"random": {}}}
    assert er.frame_modes(old) == ""
    assert er.frame_modes({"atari": [{"frames": 2}]}) == "stack-2"
    new = {"atari": [{"frames": "stack-2"}], "doom": {"model": {"frames": "stack-2"}, "random": {}},
           "maze": [{"policy": "expert"}, {"frames": "stack-2"}], "control": [{"frames": "stack-2"}]}
    assert er.frame_modes(new) == "stack-2"
    assert er.frame_modes({"atari": [{"frames": "stack-2"}], "control": [{"frames": "single"}]}) == \
        "atari stack-2, control single"
