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
