"""Game frame modes (``laya.frames``): the helpers, ``"single"`` staying bit-identical to the one-image states,
histories in the toolkit, the benchmark (every family, every mode, stub policies) and the pool, the stack's token
budget and the encoder-feature cache path of ``games_eval.batched_probs``."""
import dataclasses
import hashlib
import importlib.util
import io
import os
import sys
import types

import numpy as np
import pytest
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "autoresearch"))
import toolkit  # noqa: E402

from laya import frames as F  # noqa: E402

_spec = importlib.util.spec_from_file_location("games_eval", os.path.join(HERE, "..", "autoresearch", "games_eval.py"))
ge = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ge)

MODES = ("single", "trail-3", "stack-3")


def decode(b) -> np.ndarray:
    with Image.open(io.BytesIO(b)) as im:
        return np.asarray(im.convert("RGB"))


def state_arrays(ex):
    return [decode(b) for b in F.state_images(ex["state"])]


# -- helpers --------------------------------------------------------------------------------------------------------


def test_parse_and_resolve():
    assert F.parse("single") == ("single", 0)
    assert F.parse("trail-4") == ("trail", 4) and F.parse("stack-5") == ("stack", 5)
    for bad in ("stack-0", "stack-6", "trail", "stack-x", "double", "Stack-2"):
        with pytest.raises(ValueError):
            F.parse(bad)
    assert F.resolve("single", "control") == ("trail", 2)
    for fam in ("grid", "atari", "doom"):
        assert F.resolve("single", fam) == ("stack", 1)
    assert F.frames_needed("stack-4", "grid") == 4 and F.images_per_state("trail-4", "atari") == 1
    assert F.images_per_state("stack-4", "control") == 4 and F.images_per_state("single", "control") == 1


def test_mode_for_reads_the_checkpoint_and_the_old_key():
    assert F.mode_for({}, "atari") == "single" and F.mode_for(None, "grid") == "single"
    assert F.mode_for({"atari_frames": 2}, "atari") == "stack-2"
    assert F.mode_for({"atari_frames": 2}, "doom") == "single"  # the old key only ever meant Atari
    assert F.mode_for({"game_frames": "trail-4", "atari_frames": 2}, "atari") == "trail-4"
    with pytest.raises(ValueError):
        F.mode_for({"game_frames": "stack-9"}, "grid")


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
def test_trail_weights(n):
    w = F.trail_weights(n)
    assert len(w) == n and abs(sum(w) - 1.0) < 1e-12
    assert all(a > b for a, b in zip(w, w[1:]))  # the current frame strongest, fading with age
    assert all(isinstance(x, float) for x in w)
    if n > 1:  # geometric: each older frame weighs r = GHOST / (1 - GHOST) of the next newer one
        r = F.GHOST / (1 - F.GHOST)
        assert all(abs(b / a - r) < 1e-12 for a, b in zip(w, w[1:]))
    if n == 2:
        assert w == [1 - F.GHOST, F.GHOST]  # exactly today's control ghost


def test_history_window_pads_with_the_first_frame():
    a, b, c = (np.full((2, 2, 3), v, np.uint8) for v in (10, 20, 30))
    assert F.window([a], 3) == [a, a, a]
    assert F.window([a, b], 3) == [a, a, b]
    assert F.window([a, b, c], 2) == [b, c]
    with pytest.raises(ValueError):
        F.window([], 2)
    h = F.History(keep=3)
    for f in (a, b, c, a):
        h.push(f)
    assert len(h) == 3 and h.frames[0] is b
    st = F.state([a], "stack-3", "atari")
    assert [x is a for x in st["images"]] == [True, True, True]
    assert F.state([a, b], "single", "atari") == {"image": b}
    assert F.state([a], "trail-3", "grid")["image"] is a  # a padded trail of one frame is that frame
    mix = F.state([a, b, c], "trail-3", "grid")["image"]
    w = F.trail_weights(3)
    assert mix.dtype == np.uint8 and int(mix[0, 0, 0]) == round(w[0] * 30 + w[1] * 20 + w[2] * 10)


def old_ghost(prev, cur):
    """``ControlGame.render`` before frame modes, verbatim."""
    mix = (1 - 0.35) * cur.astype(np.float32) + 0.35 * prev.astype(np.float32)
    return mix.round().astype(np.uint8)


def test_blend_is_the_old_ghost_bit_for_bit():
    rng = np.random.default_rng(0)
    for _ in range(20):
        prev, cur = (rng.integers(0, 256, (17, 23, 3), dtype=np.uint8) for _ in range(2))
        assert np.array_equal(F.blend([prev, cur]), old_ghost(prev, cur))
        assert np.array_equal(F.state([prev, cur], "single", "control")["image"], old_ghost(prev, cur))
    assert np.array_equal(F.blend([cur, cur.copy()]), cur)  # equal frames (a padded start) blend to themselves


# -- control render and questions ---------------------------------------------------------------------------------


def test_control_render_and_question_unchanged():
    pytest.importorskip("gymnasium")
    from laya.controlgames import ControlGame
    from laya.games import control_question

    env = ControlGame("CartPole", 3)
    hist = []
    prev = None
    while env.steps < 12:
        cur = env.frame()
        hist.append(cur)
        want = cur if prev is None else old_ghost(prev, cur)
        assert np.array_equal(np.asarray(env.render()), want)
        assert np.array_equal(F.state(hist, "single", "control")["image"], want)
        prev = cur
        env.step(env.expert())
    env.close()
    q = control_question("CartPole")["action"]["instructions"]
    assert q.endswith("A faint copy shows where things were one step earlier. Which action should you take now?")
    assert control_question("CartPole", "trail-2") == control_question("CartPole")
    assert "last 4 screens, oldest first" in control_question("CartPole", "stack-4")["action"]["instructions"]
    assert "faint" not in control_question("CartPole", "stack-1")["action"]["instructions"]


# -- toolkit ----------------------------------------------------------------------------------------------------------


def grid_digest(exs):
    h = hashlib.sha256()
    for ex in exs:
        h.update(decode(ex["state"]["image"]).tobytes())
        h.update(repr((ex["id"], ex["label"], [round(t, 6) for t in ex["target"]], ex.get("value"),
                       ex.get("next_target"))).encode())
    return h.hexdigest()[:20]


def test_toolkit_single_is_unchanged():
    """Digests of the pixels, ids and targets from the code before frame modes (origin/main d30b2f1)."""
    assert grid_digest(toolkit.maze_examples(120, workers=1)) == "7b20689f622b98481c14"
    assert grid_digest(toolkit.snake_examples(120, workers=1)) == "d5ca29618d2eec36dac9"


@pytest.mark.parametrize("fn", [toolkit.maze_examples, toolkit.snake_examples])
def test_grid_modes_share_one_trajectory(fn):
    single = fn(60, seed=3, workers=1)
    assert fn(60, seed=3, workers=1, frames="stack-1") == single == fn(60, seed=3, workers=1, frames="trail-1")
    stack = fn(60, seed=3, workers=1, frames="stack-4")
    trail = fn(60, seed=3, workers=1, frames="trail-4")
    for s, k, t in zip(single, stack, trail):
        assert s["id"] == k["id"] == t["id"] and s["target"] == k["target"] == t["target"]
        imgs = state_arrays(k)
        assert len(imgs) == 4 and np.array_equal(imgs[-1], decode(s["state"]["image"]))
        # the trail is the blend of the stack's frames (the grid blend equals the pixel blend)
        assert np.array_equal(decode(t["state"]["image"]), F.blend(imgs))


def test_grid_stack_is_the_walk_under_one_symmetry():
    exs = toolkit.maze_examples(80, seed=4, workers=1, frames="stack-3")
    seen_sym = set()
    for ex in exs:
        imgs = state_arrays(ex)
        t = int(ex["id"].split("-t")[1].split("-")[0])
        seen_sym.add(ex["id"].rsplit("-g", 1)[1])
        walls = [np.all(im == (30, 30, 30), -1) for im in imgs]
        assert all(np.array_equal(w, walls[-1]) for w in walls)  # one symmetry for every frame
        if t == 0:
            assert all(np.array_equal(im, imgs[-1]) for im in imgs)  # episode start: repeated
        for a, b in zip(imgs, imgs[1:]):  # consecutive frames differ by one move: two cells of >= 9 x 9 change
            if not np.array_equal(a, b):
                assert 0 < np.any(a != b, -1).mean() <= 2 / 81 + 1e-9
    assert len(seen_sym) > 4


gym = pytest.importorskip("gymnasium")


def replay_frames(game, seed, upto):
    """The frames of a zero-noise (expert) control episode up to step ``upto``."""
    from laya.controlgames import ControlGame

    env = ControlGame(game, seed)
    out = []
    while env.steps <= upto:
        out.append(env.frame())
        env.step(env.expert())
    env.close()
    return out


@pytest.mark.parametrize("mode", ["single", "trail-4", "stack-3"])
def test_control_histories_are_the_trajectorys_own_frames_flipped_alike(mode):
    exs = toolkit.control_examples("CartPole", 12, seed=7, eps=0.0, keep=0.2, flip=True, smooth=0.0, workers=1,
                                   frames=mode)
    assert exs[0]["q"]["ins"] == toolkit._internal(__import__("laya.games").games.control_question(
        "CartPole", mode))["ins"]
    kind, n = F.resolve(mode, "control")
    flips = 0
    for ex in exs:
        seed, t = int(ex["id"].split("-")[1]), int(ex["id"].split("-t")[1].split("-")[0])
        flip = ex["id"].endswith("-f")
        flips += flip
        past = F.window(replay_frames("CartPole", seed, t)[max(0, t - n + 1):t + 1], n)
        if flip:
            past = [np.ascontiguousarray(f[:, ::-1]) for f in past]
        got = state_arrays(ex)
        want = [F.blend(past)] if kind == "trail" else past
        assert len(got) == len(want) and all(np.array_equal(a, b) for a, b in zip(got, want))
    assert 0 < flips < len(exs)


def test_control_single_keeps_the_same_frames():
    """For up to two frames the keep decisions are drawn as before, so the same steps are kept."""
    a = toolkit.control_examples("Acrobot", 30, seed=2, workers=1)
    b = toolkit.control_examples("Acrobot", 30, seed=2, workers=1, frames="stack-2")
    assert [e["id"] for e in a] == [e["id"] for e in b]
    for x, y in zip(a, b):
        prev, cur = state_arrays(y)
        img = decode(x["state"]["image"])
        assert np.array_equal(img, F.blend([prev, cur]))


# -- benchmark --------------------------------------------------------------------------------------------------------


class StubAgent:
    def __init__(self, cfg=None):
        self.cfg = cfg or {}


def recording_probs(calls, index=0):
    def fn(agent, frames, question, prev=None, cache=None):
        calls.append((list(frames), cache))
        p = np.full((len(frames), len(question["criteria"])), 0.1)
        p[:, index % len(question["criteria"])] = 1.0
        return p / p.sum(1, keepdims=True)
    return fn


def small(name, **kw):
    return {name: dataclasses.replace(ge.SUITE[name], **kw)}


def base_for(suite):
    return {s.name: ge.baseline_entry(s, ge.play_reference(s, "random"), ge.play_reference(s, "expert"))
            for s in suite.values()}


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("game", ["Snake10", "Maze4", "CartPole", "LunarLander"])
def test_grid_and_control_play_every_mode(game, mode):
    fam = ge.SUITE[game].family
    suite = small(game, episodes=3, cap=6)
    calls = []
    out = ge.run_family(StubAgent({"game_frames": mode}), fam, suite, base_for(suite), recording_probs(calls, 3))
    assert out[game]["frames"] == mode
    kind, n = F.resolve(mode, fam)
    first, cache = calls[0]
    assert len(first) == 3
    if kind == "stack" and n > 1:
        assert cache is not None and cache.keep == n
        for x in first:  # the first decision: the start frame repeated
            assert isinstance(x, list) and len(x) == n and all(np.array_equal(f, x[0]) for f in x)
        env = ge.make_env(suite[game], 0)
        assert np.array_equal(first[0][-1], env.frame())
    else:
        assert cache is None and all(isinstance(x, np.ndarray) for x in first)
    if mode == "single":  # exactly the rendered screen, as always
        env = ge.make_env(suite[game], 1)
        assert np.array_equal(first[1], np.asarray(env.render()))


def test_single_benchmark_inputs_are_unchanged():
    """Frames the model is asked about in single mode equal ``render()`` every round (the pre-mode input)."""
    spec = dataclasses.replace(ge.SUITE["CartPole"], episodes=2, cap=8)
    envs = [ge.make_env(spec, i) for i in range(2)]
    twins = [ge.make_env(spec, i) for i in range(2)]
    calls = []
    policy = ge.greedy_policy(StubAgent(), ge.question_for(spec), recording_probs(calls, 1))
    while not all(e.done for e in envs):
        live = [i for i, e in enumerate(envs) if not e.done]
        want = [np.asarray(twins[i].render()) for i in live]
        acts = policy([envs[i] for i in live])
        got = calls[-1][0]
        assert all(np.array_equal(a, b) for a, b in zip(got, want))
        for i, a in zip(live, acts):
            envs[i].step(a)
            twins[i].step(a)


def test_stack_history_follows_the_episode():
    spec = dataclasses.replace(ge.SUITE["CartPole"], episodes=1, cap=6)
    env, twin = ge.make_env(spec, 0), ge.make_env(spec, 0)
    frames = []
    calls = []
    policy = ge.greedy_policy(StubAgent(), ge.question_for(spec, "stack-4"), recording_probs(calls, 1), "stack-4")
    while not env.done:
        frames.append(twin.frame())
        a = policy([env])[0]
        assert all(np.array_equal(x, y) for x, y in zip(calls[-1][0][0], F.window(frames, 4)))
        env.step(a)
        twin.step(a)
    c = env.clone() if env.searchable else None
    if c is not None:
        assert len(c.history()) == len(env.history())


def test_search_refuses_other_modes():
    suite = small("Maze4", episodes=2)
    with pytest.raises(ValueError, match="single"):
        ge.run_family(StubAgent({"search": {"depth": 1}, "game_frames": "stack-2"}), "grid", suite, base_for(suite),
                      recording_probs([]))


@pytest.mark.parametrize("mode", ["single", "trail-3", "stack-4"])
def test_atari_plays_every_mode(mode):
    pytest.importorskip("ale_py")
    spec = dataclasses.replace(ge.SUITE["Breakout"], episodes=2, cap=7)
    calls = []
    from laya.atari_train import game_actions

    policy = ge.atari_model_policy(StubAgent({"game_frames": mode}), "Breakout", game_actions("Breakout"),
                                   recording_probs(calls, 1))
    res = ge.play_atari(spec, policy)
    assert res["decisions"] == 14 and len(calls) == 7
    kind, n = F.resolve(mode, "atari")
    for step, (batch, cache) in enumerate(calls):
        for x in batch:
            if kind == "stack" and n > 1:
                assert cache is not None and cache.keep == n and len(x) == n
                if step == 0:  # the start frame repeated
                    assert all(np.array_equal(f, x[-1]) for f in x)
            else:
                assert isinstance(x, np.ndarray) and x.shape == (210, 160, 3)


def test_old_two_frame_atari_checkpoint_plays_stack_2():
    pytest.importorskip("ale_py")
    spec = dataclasses.replace(ge.SUITE["Freeway"], episodes=1, cap=3)
    calls = []
    from laya.atari_train import game_actions

    ge.play_atari(spec, ge.atari_model_policy(StubAgent({"atari_frames": 2}), "Freeway", game_actions("Freeway"),
                                              recording_probs(calls, 1)))
    assert all(len(x) == 2 for batch, _ in calls for x in batch)


def test_atari_train_play_resets_history_after_auto_fire():
    pytest.importorskip("ale_py")
    from laya.atari_train import play

    seen = []

    def policy(obs, prevs, ids=None, hists=None):
        for o, p, h in zip(obs, prevs, hists):
            assert h[-1] is o and (len(h) == 1 or h[-2] is p) and (len(h) > 1 or p is o)
            seen.append(len(h))
        return [3] * len(obs)  # Breakout LEFT: the ball is lost, lives go

    policy.wants_history = True
    play("Breakout", policy, episodes=1, max_steps=400, seed=0)
    assert max(seen) == F.MAX_FRAMES and seen.count(1) > 1  # restarted after a lost life, not only at the start


class FakeDoom:
    """Just enough of ``vizdoom.DoomGame`` for ``play_doom``: the screen is a counter of the episode's tics."""

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

    def close(self):
        pass


@pytest.mark.parametrize("mode", ["single", "trail-2", "stack-3"])
def test_doom_plays_every_mode(mode, monkeypatch):
    monkeypatch.setattr(ge, "_doom_game", lambda scenario, labels: FakeDoom())
    spec = dataclasses.replace(ge.SUITE["DoomBasic"], episodes=5)
    calls = []
    res = ge.play_doom(spec, "model", StubAgent({"game_frames": mode}), recording_probs(calls, 2))
    assert res["steps"] == [3 + (spec.seed + i) % 3 for i in range(5)]
    kind, n = F.resolve(mode, "doom")
    for batch, cache in calls:
        for x in batch:
            if kind == "stack" and n > 1:
                t = int(x[-1][0, 0, 0]) % 5
                ticks = [int(f[0, 0, 0]) - int(x[-1][0, 0, 0]) for f in x]
                assert ticks == [max(-t, j) for j in range(-n + 1, 1)]  # oldest first, padded at the start
            else:
                assert isinstance(x, np.ndarray)


# -- recorded data histories and the pool ---------------------------------------------------------------------------


def test_frame_history_from_recorded_steps():
    from laya.vlm_train import history_coverage, with_frame_history

    recs = [{"id": "train-000001-%03d" % t, "image": "images/%d.jpg" % t} for t in (0, 1, 2, 3, 5, 6)]
    recs.append({"id": "x", "image": "a.png", "episode": 4, "step": 2, "history": ["h0.png", "h1.png"]})
    out = with_frame_history(recs, 4)
    assert out[0]["history"] == [] and out[3]["history"] == ["images/0.jpg", "images/1.jpg", "images/2.jpg"]
    assert out[5]["history"] == ["images/5.jpg"]  # step 4 missing: the chain stops there
    assert out[6]["history"] == ["h0.png", "h1.png"]  # recorded histories are kept
    cov = history_coverage(out, 4)
    assert cov == {"records": 7, "full": 5, "partial": 2}


def test_pool_game_states():
    pytest.importorskip("modal")
    import harness

    def png(v):
        buf = io.BytesIO()
        Image.fromarray(np.full((4, 5, 3), v, np.uint8)).save(buf, format="PNG")
        return buf.getvalue()

    exs = [{"state": {"image": png(40)}, "history": [png(10), png(20), png(30)], "id": "a", "target": [1.0]},
           {"state": {"image": png(40)}, "history": [], "id": "b", "target": [1.0]}]
    single = harness.game_states(exs, "single", "atari")
    assert single == [{"state": {"image": png(40)}, "id": "a", "target": [1.0]},
                      {"state": {"image": png(40)}, "id": "b", "target": [1.0]}]
    st = harness.game_states(exs, "stack-3", "doom")
    assert [decode(b)[0, 0, 0] for b in st[0]["state"]["images"]] == [20, 30, 40]
    assert [decode(b)[0, 0, 0] for b in st[1]["state"]["images"]] == [40, 40, 40]
    tr = harness.game_states(exs, "trail-2", "atari")
    assert decode(tr[0]["state"]["image"])[0, 0, 0] == round(0.65 * 40 + 0.35 * 30)
    assert "history" not in tr[0] and exs[0]["history"]  # the pool's examples are not touched
    with pytest.raises(ValueError, match="history"):
        harness.game_states([{"state": {"image": png(1)}}], "stack-2", "atari")


# -- token budget, make_item / collate, and the cached forward (SmolVLM-256M) ----------------------------------------


@pytest.fixture(scope="module")
def agent():
    import torch

    from laya.vlm import VLMAgent

    torch.manual_seed(0)
    return VLMAgent(backbone="HuggingFaceTB/SmolVLM-256M-Instruct", device="cpu", preprocess="processor")


def test_stacks_fit_max_len_and_collate(agent):
    import random

    from laya.common import render_options
    from laya.games import atari_question, control_question
    from laya.vlm import collate_vlm
    from laya.vlm_train import make_item

    proc = agent.processor
    image_token = proc.tokenizer.convert_tokens_to_ids(proc.image_token)
    assert agent.cfg["max_len"] == 1024 and agent.prep.image_seq_len == 64
    exs = (toolkit.control_examples("LunarLander", 2, seed=1, workers=1, frames="stack-4")
           + toolkit.maze_examples(2, seed=1, workers=1, frames="stack-5"))
    atari18 = atari_question("Breakout", [a for a in __import__("laya.games").games.ATARI_ACTIONS])["action"]
    frame = np.zeros((210, 160, 3), np.uint8)
    for n in (4, 5):
        exs.append({"state": {"images": [frame] * n}, "q": toolkit._internal({"action": atari18}),
                    "target": [1.0 / 18] * 18, "label": 0})
    items = [make_item(proc, ex, random.Random(0)) for ex in exs]
    lengths = {}
    for ex, it in zip(exs, items):
        n = len(F.state_images(ex["state"]))
        assert it["n_images"] == n
        assert it["ids"].count(image_token) == 64 * n
        assert len(it["ids"]) <= 1024
        lengths[(n, len(render_options(ex["q"])))] = len(it["ids"])
    b = collate_vlm(items, proc.tokenizer.pad_token_id)
    assert b["pixel_values"].shape[:2] == (len(items), 5)
    # the longest: 5 images and all 18 Atari actions, still well inside 1024
    assert max(lengths.values()) < 700, lengths
    print("sequence lengths (images, options) -> tokens:", lengths)


@pytest.mark.parametrize("backend", ["processor", "gpu"])
def test_cached_stack_forward_matches_the_plain_one(agent, backend):
    from laya.preprocess import ImagePrep

    prep = ImagePrep.from_config(dict(agent.cfg, preprocess=backend), default_backend=backend)
    old = agent.prep
    prep.apply(agent.processor)
    agent.prep = agent.model.prep = prep
    try:
        from laya.games import control_question

        rng = np.random.default_rng(0)
        frames = [rng.integers(0, 256, (64, 80, 3), dtype=np.uint8) for _ in range(6)]
        q = control_question("CartPole", "stack-3")["action"]
        cache = ge.frame_cache("stack-3", "control")
        for t in range(len(frames)):
            x = ge.state_input(frames[:t + 1], "stack-3", "control")
            plain = ge.batched_probs(agent, [x], q)
            cached = ge.batched_probs(agent, [x], q, cache=cache)
            assert np.abs(plain - cached).max() < 2e-3
        assert cache.stats["misses"] == len(frames)  # every frame encoded once
        move = ge.move_fn(agent, control_question("CartPole", "stack-3"), "stack-3", "control")
        assert np.abs(move(frames[:3]) - ge.batched_probs(agent, [ge.state_input(frames[:3], "stack-3", "control")],
                                                          q)).max() < 2e-3
    finally:
        old.apply(agent.processor)
        agent.prep = agent.model.prep = old
