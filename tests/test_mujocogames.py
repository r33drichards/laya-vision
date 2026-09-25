"""MuJoCo (``laya.mujocogames``): pushes and per-joint levels, experts beat the baselines, rendering, the questions
fit the model's budget, no model."""
import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("gymnasium")

from laya.games import CONTROL_ACTIONS, TORQUE_JOINTS, mujoco_questions  # noqa: E402
from laya.mujocogames import (GAMES, LEVELS, MujocoGame, expert_policy, is_torque_game, play_episodes,  # noqa: E402
                              questions, random_policy, still_policy)

SEED = 200_000
TORQUE = sorted(g for g in GAMES if is_torque_game(g))


@pytest.mark.parametrize("game", sorted(GAMES))
def test_actions_map_into_the_action_space_and_a_frame_renders(game):
    env = MujocoGame(game, seed=0)
    space = env.env.action_space
    if env.joints is None:
        assert tuple(CONTROL_ACTIONS[game]) == env.actions
        for a in env.actions:
            assert space.contains(env._env_action(a))
        assert (env._env_action("NONE") == 0).all()
    else:
        assert len(env.joints) == space.shape[0] == len(TORQUE_JOINTS[game])
        for lv in LEVELS:
            assert space.contains(env._env_action({j: lv for j in env.joints}))
        assert env._env_action({j: "STRONG_POS" for j in env.joints}).tolist() == pytest.approx(space.high.tolist())
        assert (env._env_action(still_policy(env)) == 0).all()
    frame = np.asarray(env.render())
    assert frame.shape == (256, 256, 3) and frame.dtype == np.uint8
    env.close()


@pytest.mark.parametrize("game", ("InvertedPendulum", "InvertedDoublePendulum"))
def test_pendulum_experts_hold_the_pole_to_the_limit(game):
    exp = play_episodes(game, expert_policy, episodes=2, seed=SEED)
    assert exp["solved_rate"] == 1.0 and exp["mean_steps"] == 1000
    assert play_episodes(game, random_policy(0), episodes=2, seed=SEED)["mean_steps"] < 50


@pytest.mark.parametrize("game", TORQUE)
def test_hub_experts_beat_random_and_doing_nothing(game):
    pytest.importorskip("stable_baselines3")
    pytest.importorskip("sb3_contrib")
    kw = {"episodes": 2, "seed": SEED, "max_steps": 100}
    exp = play_episodes(game, expert_policy, **kw)
    assert set(exp["actions"]) <= set(LEVELS)
    assert exp["mean_score"] > play_episodes(game, random_policy(0), **kw)["mean_score"]
    assert exp["mean_score"] > play_episodes(game, still_policy, **kw)["mean_score"]


def test_expert_levels_are_the_nearest_to_its_continuous_action():
    pytest.importorskip("stable_baselines3")
    env = MujocoGame("Walker2d", seed=3)
    for _ in range(5):
        a, act = env.expert_action(), env.expert()
        values = {"STRONG_NEG": -1.0, "NEG": -0.5, "NONE": 0.0, "POS": 0.5, "STRONG_POS": 1.0}
        for k, j in enumerate(env.joints):
            assert abs(values[act[j]] - a[k]) <= 0.25 + 1e-9
        env.step(act)
    env.close()


def test_right_moves_the_cart_right():
    for game in ("InvertedPendulum", "InvertedDoublePendulum"):
        env = MujocoGame(game, seed=0)
        x0 = env.env.unwrapped.data.qpos[0]
        for _ in range(3):
            env.step(GAMES[game]["actions"][-1])
        assert env.env.unwrapped.data.qpos[0] > x0  # the last action pushes toward +x, screen right
        env.close()


def test_episodes_are_seeded_and_count_levels_per_joint():
    a = play_episodes("Walker2d", random_policy(1), episodes=2, seed=5, max_steps=30)
    b = play_episodes("Walker2d", random_policy(1), episodes=2, seed=5, max_steps=30)
    assert a["results"] == b["results"] and a["solved_rate"] is None
    assert set(a["actions"]) <= set(LEVELS) and sum(a["actions"].values()) == 6 * sum(e["steps"] for e in a["results"])


def test_render_ghosts_the_previous_frame():
    env = MujocoGame("InvertedPendulum", seed=0)
    first = np.asarray(env.render())
    for _ in range(3):
        env.render()
        env.step("RIGHT")
    raw = env.frame()
    assert (raw != first).any()  # a fixed camera: the cart moved across the frame
    assert (np.asarray(env.render()) != raw).any()  # the previous frame shows through
    env.close()


def test_questions_fit_the_budget_without_merging_options():
    """Every question keeps all its options distinct after the model's truncation (``head_max_len`` 256): the
    one-question-per-game layout collapsed Humanoid's 35 options to 7."""
    from laya.vlm import OPTION_BULLET, VLMAgent, render_options

    tok = pytest.importorskip("transformers").AutoTokenizer.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct")
    for game in GAMES:
        qs = questions(game)
        if is_torque_game(game):
            assert list(qs) == [j.replace(" ", "_") for j in TORQUE_JOINTS[game]] and qs == mujoco_questions(game)
        for q in qs.values():
            internal = VLMAgent._to_internal(q)
            assert internal["t"] == "choice"
            opts = [tok(OPTION_BULLET + o, add_special_tokens=False)["input_ids"][:48] for o in render_options(internal)]
            head = tok(q["instructions"], add_special_tokens=False)["input_ids"]
            assert len(head) + sum(len(o) + 1 for o in opts) + 16 <= 256, game
            assert len(set(map(tuple, opts))) == len(opts)


def test_soft_levels_split_between_the_two_nearest_levels():
    from laya.mujocogames import soft_levels

    assert soft_levels(0.3) == [0.0, 0.0, 0.4, 0.6, 0.0]
    assert soft_levels(-1.0) == [1.0, 0.0, 0.0, 0.0, 0.0] and soft_levels(1.7) == [0.0, 0.0, 0.0, 0.0, 1.0]
    assert soft_levels(0.5) == [0.0, 0.0, 0.0, 1.0, 0.0]
    for a in np.linspace(-1, 1, 41):
        t = soft_levels(a)
        assert sum(t) == pytest.approx(1.0) and sum(v > 0 for v in t) <= 2
        assert np.dot(t, [-1, -0.5, 0, 0.5, 1]) == pytest.approx(a, abs=1e-3)  # the mean is the action


def test_jitter_moves_levels_at_most_one_notch():
    import random

    from laya.mujocogames import jitter

    env = MujocoGame("Hopper", seed=0)
    rng = random.Random(0)
    base = {j: "NONE" for j in env.joints}
    moved = [jitter(env, base, rng, 0.5) for _ in range(200)]
    assert {v for m in moved for v in m.values()} == {"NEG", "NONE", "POS"}
    assert all(m == base for m in (jitter(env, base, rng, 0.0) for _ in range(20)))
    assert jitter(env, {j: "STRONG_POS" for j in env.joints}, random.Random(1), 1.0)["thigh"] in ("POS", "STRONG_POS")
    env.close()


@pytest.mark.parametrize("game", ("InvertedPendulum", "Hopper"))
def test_expert_frames_match_play_and_label_every_question(game):
    pytest.importorskip("stable_baselines3")
    from laya.mujocogames import expert_frames

    frames = list(expert_frames(game, 6, seed=7, noise=0.0, stride=2))
    assert len(frames) == 6
    qs = questions(game)
    for fr in frames:
        assert [r["key"] for r in fr["records"]] == list(qs)
        for r in fr["records"]:
            assert r["question"] == qs[r["key"]] and len(r["target"]) == len(r["question"]["criteria"])
            assert sum(r["target"]) == pytest.approx(1.0, abs=1e-3)  # training renormalizes
            assert r["label"] == int(np.argmax(r["target"]))
    # with no noise the behaviour is the expert, so replaying its actions renders the same kept frames
    first = [f for f in frames if f["episode"] == 7]
    env, kept = MujocoGame(game, seed=7), {}
    while not env.done and env.steps <= first[-1]["step"]:
        env.frame()
        if env.steps % 2 == 0:
            kept[env.steps] = np.asarray(env.render())
        env.step(env.expert())
    for fr in first:
        assert np.array_equal(np.asarray(fr["image"]), kept[fr["step"]])
    env.close()


# ---- several camera views (opt-in ``views``) ----

def test_default_view_is_the_single_view_as_before():
    """No ``views``: one camera, a PIL image, ``{"image"}`` state and the old question text; naming the game's first
    view is the same thing."""
    from laya.mujocogames import VIEWS

    for game in ("Walker2d", "Reacher", "InvertedPendulum"):
        env, named = MujocoGame(game, seed=0), MujocoGame(game, seed=0, views=next(iter(VIEWS[game])))
        assert env.views == named.views and not env.multiview
        assert set(env.state()) == {"image"} and np.array_equal(np.asarray(env.render()), np.asarray(named.render()))
        assert env.frame().shape == (256, 256, 3) and np.array_equal(env.frame(), env.env.render())
        assert env.questions() == questions(game) == questions(game, None)
        env.close()
        named.close()
    for game in GAMES:
        assert all("cameras" not in q["instructions"] for q in questions(game).values())


@pytest.mark.parametrize("game", ("Walker2d", "Humanoid", "Ant", "Reacher", "Pusher", "Swimmer"))
def test_every_view_renders_distinct_ghosted_images(game):
    from laya.mujocogames import VIEWS

    env = MujocoGame(game, seed=0, views="all")
    n = len(VIEWS[game])
    assert env.views == tuple(VIEWS[game]) and n >= 2 and env.multiview
    first = env.frame()
    assert first.shape == (n, 256, 256, 3) and first.dtype == np.uint8
    assert np.array_equal(first[0], env.env.render())  # the first view is the environment's own camera
    assert len({v.tobytes() for v in first}) == n  # every camera sees something different
    imgs = env.render()
    assert isinstance(imgs, list) and [np.asarray(i).shape for i in imgs] == [(256, 256, 3)] * n
    assert set(env.state()) == {"images"} and len(env.state()["images"]) == n
    for _ in range(3):
        env.frame()
        env.step(random_policy(0)(env))
    cur, ghosted = env.frame(), np.stack([np.asarray(i) for i in env.render()])
    assert all((g != c).any() for g, c in zip(ghosted, cur))  # the previous frame shows through in every view
    env.close()


def test_views_are_deterministic_and_do_not_depend_on_render_history():
    """Each view depends on the state only: rendering every step or only at the end gives the same pixels, and the
    extra cameras leave the environment's own camera untouched."""
    a, b, c = (MujocoGame("Walker2d", seed=3, views="all"), MujocoGame("Walker2d", seed=3, views="all"),
               MujocoGame("Walker2d", seed=3))
    pick = random_policy(1)
    for _ in range(12):
        act = pick(a)
        a.frame()
        c.frame()
        for e in (a, b, c):
            e.step(act)
    assert np.array_equal(a.frame(), b.frame())
    assert np.array_equal(a.frame()[0], c.frame())
    for e in (a, b, c):
        e.close()


def test_view_names_are_checked_and_ordered():
    from laya.mujocogames import resolve_views

    assert resolve_views("Walker2d") == ("side",)
    assert resolve_views("Walker2d", "top,side") == ("top", "side") == resolve_views("Walker2d", ["top", "side"])
    for bad in ("nope", "side,side", []):
        with pytest.raises(ValueError):
            resolve_views("Walker2d", bad)
    env, one = MujocoGame("Walker2d", seed=0, views=("top", "side")), MujocoGame("Walker2d", seed=0)
    assert np.array_equal(env.frame()[1], one.frame())
    env.close()
    one.close()


def test_multiview_questions_name_the_cameras_and_keep_their_shared_prefix():
    qs, base = questions("Humanoid", "all"), questions("Humanoid")
    assert list(qs) == list(base)
    for k, q in qs.items():
        assert "from 4 cameras, one image each, in this order: side, front, top, three-quarter." in q["instructions"]
        assert q["instructions"].endswith(base[k]["instructions"].split(" A faint copy")[1])
        assert q["criteria"] == base[k]["criteria"]
    assert len({q["instructions"].rsplit("torque the", 1)[0] for q in qs.values()}) == 1  # the joint still comes last


def test_multiview_expert_frames_point_records_at_every_view():
    pytest.importorskip("stable_baselines3")
    from laya.mujocogames import expert_frames
    from laya.vlm_train import jsonl_example

    frames = list(expert_frames("Walker2d", 3, seed=7, noise=0.0, stride=2, views="side,three_quarter"))
    assert len(frames) == 3
    for fr in frames:
        assert "image" not in fr and len(fr["images"]) == 2
        assert all(np.asarray(i).shape == (256, 256, 3) for i in fr["images"])
        assert all("2 cameras" in r["question"]["instructions"] for r in fr["records"])
        r = fr["records"][0]
        rec = {"id": "x", "images": ["a.png", "b.png"], "question": r["question"], "label": r["label"],
               "target": r["target"]}
        assert jsonl_example(rec, "/root")["state"] == {"images": ["/root/a.png", "/root/b.png"]}
    env, kept = MujocoGame("Walker2d", seed=7, views="side,three_quarter"), {}
    while env.steps <= frames[-1]["step"]:
        env.frame()
        if env.steps % 2 == 0:
            kept[env.steps] = [np.asarray(i) for i in env.render()]
        env.step(env.expert())
    for fr in frames:
        assert all(np.array_equal(np.asarray(i), k) for i, k in zip(fr["images"], kept[fr["step"]]))
    env.close()


def test_record_tiles_the_views(tmp_path):
    pytest.importorskip("imageio")
    from laya.mujocogames import draw, record

    env, one = MujocoGame("Walker2d", seed=0, views="all"), MujocoGame("Walker2d", seed=0)
    assert draw(env.render(), env, still_policy(env), None, "still").shape == (512, 812, 3)  # 2x2 grid of 256 px
    assert draw(one.render(), one, still_policy(one), None, "still").shape == (512, 812, 3)  # one view at 2x
    out = tmp_path / "v.mp4"
    record(env, lambda e: (still_policy(e), None), "still", str(out), max_steps=3)
    assert out.stat().st_size > 0 and env.steps == 3
    env.close()
    one.close()
