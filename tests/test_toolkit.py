"""``autoresearch/toolkit.py``: game training data (determinism, eval-seed disjointness, soft targets on optimal moves
only, symmetry correctness, values, the laya.games questions, ControlGame rendering, make_item, game_mix)."""
import io
import os
import random
import sys

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "autoresearch"))
import toolkit  # noqa: E402

from laya.games import CONTROL_ACTIONS, control_question, maze_question, snake_question  # noqa: E402
from laya.gridgames import ACTIONS, OPPOSITE, Maze, Snake, bfs_path  # noqa: E402
from laya.vlm import VLMAgent  # noqa: E402


def pixels(ex):
    return np.asarray(Image.open(io.BytesIO(ex["state"]["image"])).convert("RGB"))


def check_common(exs, n_opts):
    for ex in exs:
        t = ex["target"]
        assert len(t) == n_opts and min(t) >= 0 and abs(sum(t) - 1) < 1e-9
        assert t[ex["label"]] == max(t)
        assert 0.0 <= ex["value"] <= 1.0
        assert isinstance(ex["state"]["image"], bytes)


@pytest.fixture(scope="module")
def mazes():
    return toolkit.maze_examples(300, seed=11, workers=1)


@pytest.fixture(scope="module")
def snakes():
    return toolkit.snake_examples(300, seed=11, workers=1)


# -- determinism and seeds ------------------------------------------------------------------------------------------


def _key(exs):
    return [(e["id"], e["target"], e["label"], e["value"], e["state"]["image"]) for e in exs]


def test_deterministic_by_seed_and_independent_of_workers(mazes, snakes):
    assert _key(toolkit.maze_examples(300, seed=11, workers=2)) == _key(mazes)
    assert _key(toolkit.snake_examples(300, seed=11, workers=2)) == _key(snakes)
    assert _key(toolkit.maze_examples(300, seed=12, workers=1)) != _key(mazes)


def test_seeds_stay_below_the_eval_ranges(mazes):
    assert all(int(e["id"].split("-")[1]) < toolkit.EVAL_SEED_FLOOR for e in mazes)
    assert toolkit.EVAL_SEED_FLOOR <= 100_000 <= 200_000  # Atari eval seeds, GRID_SEED (Maze/Snake/control)
    with pytest.raises(ValueError):
        toolkit.maze_examples(50, seed=99_995)
    with pytest.raises(ValueError):
        toolkit.snake_examples(10, seed=200_000)


def test_questions_are_exactly_laya_games(mazes, snakes):
    assert all(e["q"] == VLMAgent._to_internal(maze_question()["action"]) for e in mazes)
    assert all(e["q"] == VLMAgent._to_internal(snake_question()["action"]) for e in snakes)
    assert list(mazes[0]["q"]["crit"]) == list(toolkit.GRID_ACTIONS) == list(ACTIONS)
    assert {e["dataset"] for e in mazes} == {"game_maze"} and {e["dataset"] for e in snakes} == {"game_snake"}


# -- symmetries -----------------------------------------------------------------------------------------------------


def test_symmetry_group_is_consistent():
    n = 7
    grid = np.arange(n * n).reshape(n, n)
    for sym in range(8):
        g = toolkit.sym_grid(grid, sym)
        assert sorted({toolkit.sym_action(a, sym) for a in ACTIONS}) == sorted(ACTIONS)
        for r in range(n):
            for c in range(n):
                assert g[toolkit.sym_pos((r, c), n, sym)] == grid[r, c]
                for a in ACTIONS:  # moving then mapping == mapping then moving the mapped way
                    dr, dc = toolkit._VEC[a]
                    nxt = toolkit.sym_pos((r + dr, c + dc), n, sym)
                    here = toolkit.sym_pos((r, c), n, sym)
                    dr2, dc2 = toolkit._VEC[toolkit.sym_action(a, sym)]
                    assert nxt == (here[0] + dr2, here[1] + dc2)
    assert len({toolkit.sym_grid(grid, s).tobytes() for s in range(8)}) == 8


def _maze_from_id(ex):
    _, seed, size, r, c, t, sym = ex["id"].split("-")
    m = Maze(int(size[1:]), int(seed))
    return m, (int(r[1:]), int(c[1:])), int(t[1:]), int(sym[1:])


def test_maze_targets_values_and_flips(mazes):
    check_common(mazes, 4)
    syms = set()
    for ex in mazes:
        m, pos, t, sym = _maze_from_id(ex)
        syms.add(sym)
        n = m.wall.shape[0]
        # the same state after the symmetry, as a real Maze: its BFS expert must agree with the mapped target
        f = Maze.__new__(Maze)
        f.__dict__.update(m.__dict__)
        f.wall = toolkit.sym_grid(m.wall, sym)
        f.pos, f.goal = toolkit.sym_pos(pos, n, sym), toolkit.sym_pos(m.goal, n, sym)
        assert np.array_equal(pixels(ex), np.asarray(f.render()))
        path = f.expert_path()
        opt = [a for a in ACTIONS if ex["target"][ACTIONS.index(a)] > 0]
        assert opt == [path[0]]  # perfect maze: one shortest path, so the soft target is that move
        for a in ACTIONS:  # every move with mass shortens the true distance; no other does
            dr, dc = toolkit._VEC[a]
            nxt = (f.pos[0] + dr, f.pos[1] + dc)
            if not f.wall[nxt]:
                d = len(bfs_path(lambda r, c: not f.wall[r, c], nxt, f.goal, f.wall.shape))
                assert (d == len(path) - 1) == (a in opt)
        assert ACTIONS[ex["label"]] == toolkit.sym_action(Maze.expert(_at(m, pos)), sym)
        d = len(path)
        want = 0.97 ** d if d <= m.max_steps - t else 0.0
        assert ex["value"] == pytest.approx(want)
    assert len(syms) == 8


def _at(m, pos):
    m.pos = pos
    return m


def test_maze_one_hot_and_no_flip():
    exs = toolkit.maze_examples(40, seed=3, soft=False, flip=False, workers=1)
    for ex in exs:
        m, pos, _, sym = _maze_from_id(ex)
        assert sym == 0
        assert ex["target"] == [float(a == Maze.expert(_at(m, pos))) for a in ACTIONS]


def _random_snakes(k=60, seed=0):
    rng, out = random.Random(seed), []
    for i in range(k):
        s = Snake(rng.choice((6, 8, 10)), 50_000 + i)
        for _ in range(rng.randrange(0, 120)):
            if s.dead:
                break
            safe = toolkit.snake_moves(s)[1]
            s.step(rng.choice(safe) if safe and rng.random() < 0.4 else s.expert())
        if s.dead is None:
            out.append(s)
    return out


def test_snake_moves_are_optimal_and_safe():
    for s in _random_snakes():
        opt, safe = toolkit.snake_moves(s)
        assert set(opt) <= set(safe) and OPPOSITE[s.heading] not in safe
        if s.food is None:
            continue
        body = set(list(s.body)[:-1])
        lens = {}
        for a in safe:
            _, nxt = s._next(a)
            p = [] if nxt == s.food else bfs_path(lambda r, c: (r, c) not in body, nxt, s.food, (s.size, s.size))
            if p is not None:
                lens[a] = len(p)
        if lens:
            assert set(opt) == {a for a, v in lens.items() if v == min(lens.values())}
            assert s.expert() in opt
        else:
            assert opt == []


def _sym_snake(s, sym):
    f = toolkit._snake_clone(s)
    f.body = type(s.body)(toolkit.sym_pos(p, s.size, sym) for p in s.body)
    f.food = None if s.food is None else toolkit.sym_pos(s.food, s.size, sym)
    f.heading = toolkit.sym_action(s.heading, sym)
    return f


def test_snake_flips_map_moves_and_pixels():
    for s in _random_snakes(25, seed=1):
        opt, safe = toolkit.snake_moves(s)
        for sym in range(8):
            f = _sym_snake(s, sym)
            fo, fs = toolkit.snake_moves(f)
            assert sorted(fo) == sorted(toolkit.sym_action(a, sym) for a in opt)
            assert sorted(fs) == sorted(toolkit.sym_action(a, sym) for a in safe)
            got = np.asarray(Image.open(io.BytesIO(toolkit._grid_png(toolkit._snake_grid(s), sym))).convert("RGB"))
            assert np.array_equal(got, np.asarray(f.render()))


def test_snake_examples(snakes):
    check_common(snakes, 4)
    assert set(e["value"] for e in snakes) <= {0.0, 1.0}
    assert any(sum(p > 0 for p in e["target"]) > 1 for e in snakes)  # real ties get soft targets
    assert len({e["id"].split("-")[1] for e in snakes}) >= 300 // 12  # many episodes, few states each


def test_snake_value_is_expert_survival():
    for s in _random_snakes(20, seed=2):
        c = toolkit._snake_clone(s)
        for _ in range(30):
            if c.dead:
                break
            c.step(c.expert())
        assert toolkit._snake_value(s, 30) == float(c.dead in (None, "won"))
        assert s.dead is None  # the clone never touches the original


# -- control --------------------------------------------------------------------------------------------------------

gym = pytest.importorskip("gymnasium")


@pytest.mark.parametrize("game", ["CartPole", "Acrobot"])
def test_control_frames_match_controlgame(game):
    from laya.controlgames import ControlGame

    exs = toolkit.control_examples(game, 25, seed=5, eps=0.0, keep=0.3, flip=False, smooth=0.0, workers=1)
    check_common(exs, len(CONTROL_ACTIONS[game]))
    assert exs[0]["q"] == VLMAgent._to_internal(control_question(game)["action"])
    assert exs[0]["dataset"] == "game_" + game.lower()
    first = [e for e in exs if e["id"].startswith("%s-5-" % game.lower())]
    env = ControlGame(game, 5)
    want = {int(e["id"].split("-t")[1]): e for e in first}
    while not env.done and want:
        img = env.render()  # the eval loop renders every step
        if env.steps in want:
            e = want.pop(env.steps)
            assert np.array_equal(pixels(e), np.asarray(img))
            assert env.actions[e["label"]] == env.expert() and e["target"][e["label"]] == 1.0
        env.step(env.expert())
    assert not want
    env.close()


def test_control_values_and_flips():
    exs = toolkit.control_examples("CartPole", 60, seed=9, workers=1)
    check_common(exs, 2)
    assert any(e["id"].endswith("-f") for e in exs) and any(not e["id"].endswith("-f") for e in exs)
    assert all(e["target"][e["label"]] == pytest.approx(0.95) for e in exs)  # smooth=0.1 over 2 actions
    mc = toolkit.control_examples("MountainCar", 30, seed=9, workers=1)
    assert not any(e["id"].endswith("-f") for e in mc)  # no mirror symmetry: the flag is on the right
    with pytest.raises(ValueError):
        toolkit.control_examples("Pong", 5)


def test_cartpole_and_acrobot_mirror_symmetry():
    """Mirroring the state mirrors the frame and swaps the expert's action, as the flip augmentation assumes."""
    from laya.controlgames import ControlGame

    for game, neg in (("CartPole", lambda s: -s), ("Acrobot", lambda s: -s)):
        env, mir = ControlGame(game, 3), ControlGame(game, 3)
        for _ in range(15):
            env.step(env.expert())
        mir.env.unwrapped.state = neg(np.array(env.env.unwrapped.state))
        mir.obs = np.array(mir.env.unwrapped._get_ob() if game == "Acrobot" else mir.env.unwrapped.state,
                           dtype=np.float32)
        a, b = np.asarray(env.frame())[:, ::-1].astype(int), np.asarray(mir.frame()).astype(int)
        assert (np.abs(a - b).max(-1) > 60).mean() < 0.01, game  # up to anti-aliasing / a pixel of rounding
        swap = {**toolkit.CONTROL_FLIPS[game], **{v: k for k, v in toolkit.CONTROL_FLIPS[game].items()}}
        assert swap[env.expert()] == mir.expert(), game
        env.close()
        mir.close()


# -- mixing and make_item -------------------------------------------------------------------------------------------


@pytest.mark.parametrize("alpha", [0.0, 0.5])
def test_game_mix_shares(mazes, snakes, alpha):
    from laya.vlm_train import mix_probabilities

    vqa = [{"dataset": "a"}] * 10 + [{"dataset": "b"}] * 40
    exs, w = toolkit.game_mix(vqa, mazes + snakes, 0.3, base_weights={"a": 3.0}, alpha=alpha)
    groups = {}
    for e in exs:
        groups.setdefault(e["dataset"], []).append(e)
    p = mix_probabilities(groups, w, alpha)
    assert p["game_maze"] + p["game_snake"] == pytest.approx(0.3)
    assert p["game_maze"] == pytest.approx(p["game_snake"])
    base = mix_probabilities({"a": groups["a"], "b": groups["b"]}, {"a": 3.0}, alpha)
    assert p["a"] / p["b"] == pytest.approx(base["a"] / base["b"])
    only, w1 = toolkit.game_mix(vqa, mazes, 1.0)
    assert {e["dataset"] for e in only} == {"game_maze"} and set(w1) == {"game_maze"}


@pytest.fixture(scope="module")
def processor():
    return VLMAgent(backbone="HuggingFaceTB/SmolVLM-256M-Instruct", device="cpu").processor


def test_examples_go_through_make_item(processor, mazes, snakes):
    from laya.vlm_train import make_item

    exs = mazes[:3] + snakes[:3] + toolkit.control_examples("CartPole", 2, seed=1, workers=1)
    for ex in exs:
        it = make_item(processor, ex, random.Random(0))
        assert sorted(it["target"]) == sorted(ex["target"]) and it["n_images"] == 1
        assert len(it["markers"]) == len(ex["target"])


# -- next-move targets ------------------------------------------------------------------------------------------------


def _unsym(target, sym):
    """The inverse of ``toolkit.sym_target``: the target before symmetry ``sym`` was applied."""
    return [target[ACTIONS.index(toolkit.sym_action(a, sym))] for a in ACTIONS]


def _by_step(exs, parse):
    """``{episode: {t: (example, sym)}}`` from ids; ``parse(id) -> (episode, t, sym)``."""
    out = {}
    for ex in exs:
        ep, t, sym = parse(ex["id"])
        out.setdefault(ep, {})[t] = (ex, sym)
    return out


def _maze_id(i):
    _, seed, _, _, _, t, sym = i.split("-")
    return int(seed), int(t[1:]), int(sym[1:])


def _snake_id(i):
    _, seed, t, sym = i.split("-")
    return int(seed), int(t[1:]), int(sym[1:])


def _check_next(eps, n_opts, unsym, sym_fn):
    """Every example t whose step t + 1 is in the data has next_target == step t + 1's target (mapped from its
    symmetry to this example's); the last step of every complete episode (all but the last one) has none."""
    pairs = 0
    last_ep = max(eps)
    for ep, steps in eps.items():
        assert sorted(steps) == list(range(len(steps)))  # every recorded step kept
        for t, (ex, sym) in steps.items():
            if t + 1 in steps:
                nxt, nsym = steps[t + 1]
                want = sym_fn(unsym(nxt["target"], nsym), sym)
                assert ex["next_target"] == want, ex["id"]
                assert len(ex["next_target"]) == n_opts and abs(sum(ex["next_target"]) - 1) < 1e-9
                pairs += 1
            elif ep != last_ep:  # the episode's last recorded step (the last episode may be cut by n)
                assert "next_target" not in ex, ex["id"]
    assert pairs > 20
    return pairs


@pytest.mark.parametrize("flip", [False, True])
def test_maze_next_target_is_the_next_steps_target(flip):
    exs = toolkit.maze_examples(150, seed=21, per_episode=10_000, flip=flip, smooth=0.1, workers=1)
    eps = _by_step(exs, _maze_id)
    assert len(eps) > 2
    if not flip:  # unflipped: literally the next recorded step's target
        for steps in eps.values():
            for t, (ex, _) in steps.items():
                if t + 1 in steps:
                    assert ex["next_target"] == steps[t + 1][0]["target"]
    else:
        assert len({s for steps in eps.values() for _, s in steps.values()}) == 8
    _check_next(eps, 4, _unsym, toolkit.sym_target)


@pytest.mark.parametrize("flip", [False, True])
def test_snake_next_target_is_the_next_steps_target(flip):
    exs = toolkit.snake_examples(200, seed=21, per_episode=10_000, episode_cap=40, flip=flip, workers=1)
    eps = _by_step(exs, _snake_id)
    if not flip:
        for steps in eps.values():
            for t, (ex, _) in steps.items():
                if t + 1 in steps:
                    assert ex["next_target"] == steps[t + 1][0]["target"]
    _check_next(eps, 4, _unsym, toolkit.sym_target)


def test_next_target_can_be_left_out():
    assert not any("next_target" in e for e in toolkit.maze_examples(30, seed=1, next_target=False, workers=1))
    assert not any("next_target" in e for e in toolkit.snake_examples(30, seed=1, next_target=False, workers=1))


@pytest.mark.parametrize("game,n,flip", [("CartPole", 80, True), ("Acrobot", 260, False), ("Acrobot", 260, True)])
def test_control_next_target_is_the_next_steps_expert(game, n, flip):
    actions = list(CONTROL_ACTIONS[game])
    swap = {**toolkit.CONTROL_FLIPS[game], **{v: k for k, v in toolkit.CONTROL_FLIPS[game].items()}}

    def mirror(vec, flipped):
        return [vec[actions.index(swap.get(a, a))] for a in actions] if flipped else list(vec)

    def parse(i):
        _, seed, t = i.split("-")[:3]
        return int(seed), int(t[1:]), i.endswith("-f")

    exs = toolkit.control_examples(game, n, seed=13, keep=1.0, flip=flip, workers=1)
    eps = _by_step(exs, parse)
    if flip:
        assert any(f for steps in eps.values() for _, f in steps.values())
    for steps in eps.values():  # the expert's action at t + 1, mirrored with this example
        for t, (ex, f) in steps.items():
            if t + 1 in steps:
                nxt, nf = steps[t + 1]
                assert ex["next_target"] == mirror(nxt["target"], nf != f) == mirror(mirror(nxt["target"], nf), f)
    _check_next(eps, len(actions), mirror, lambda v, f: mirror(v, f))


def test_next_target_goes_through_make_item(processor, mazes):
    from laya.vlm_train import make_item

    ex = next(e for e in mazes if "next_target" in e)
    for seed in range(4):
        it = make_item(processor, ex, random.Random(seed))
        assert it["next_target"] == pytest.approx([ex["next_target"][i] for i in it["order"]])
