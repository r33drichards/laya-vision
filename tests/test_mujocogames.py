"""MuJoCo (``laya.mujocogames``): named pushes, experts beat the baselines, the lookahead leaves the episode alone,
rendering, no model."""
import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("gymnasium")

from laya.games import CONTROL_ACTIONS, control_question  # noqa: E402
from laya.mujocogames import (GAMES, MujocoGame, expert_policy, has_expert, play_episodes, random_policy,  # noqa: E402
                              still_policy)

SEED = 200_000


@pytest.mark.parametrize("game", sorted(GAMES))
def test_pushes_match_the_actions_and_the_action_space(game):
    spec = GAMES[game]
    env = MujocoGame(game, seed=0)
    assert tuple(CONTROL_ACTIONS[game]) == spec["actions"] and len(spec["pushes"]) == len(spec["actions"])
    assert np.asarray(spec["pushes"]).shape[1] == env.env.action_space.shape[0]
    for i, push in enumerate(spec["pushes"]):
        a = env._env_action(i)
        assert env.env.action_space.contains(a)
        assert a.tolist() == pytest.approx((np.asarray(push) * env.env.action_space.high).tolist())
    assert (env._env_action(spec["actions"].index("NONE")) == 0).all()
    frame = np.asarray(env.render())
    assert frame.shape == (256, 256, 3) and frame.dtype == np.uint8
    env.close()


@pytest.mark.parametrize("game", sorted(g for g in GAMES if has_expert(g)))
def test_expert_beats_random_and_doing_nothing(game):
    kw = {"episodes": 2, "seed": SEED, "max_steps": 100}
    exp = play_episodes(game, expert_policy, **kw)
    assert exp["mean_score"] > play_episodes(game, random_policy(0), **kw)["mean_score"]
    assert exp["mean_score"] > play_episodes(game, still_policy, **kw)["mean_score"]


@pytest.mark.parametrize("game", ("InvertedPendulum", "InvertedDoublePendulum"))
def test_pendulum_experts_hold_the_pole_to_the_limit(game):
    exp = play_episodes(game, expert_policy, episodes=2, seed=SEED)
    assert exp["solved_rate"] == 1.0 and exp["mean_steps"] == 1000
    assert play_episodes(game, random_policy(0), episodes=2, seed=SEED)["mean_steps"] < 50


def test_lookahead_leaves_the_episode_unchanged():
    planned, replayed = MujocoGame("Hopper", seed=3), MujocoGame("Hopper", seed=3)
    for _ in range(20):
        a = planned.expert()  # simulates every push, then restores the state
        planned.step(a)
        replayed.step(a)
        assert np.array_equal(planned.obs, replayed.obs) and planned.score == replayed.score
    planned.close(), replayed.close()


def test_right_moves_the_cart_right():
    for game in ("InvertedPendulum", "InvertedDoublePendulum"):
        env = MujocoGame(game, seed=0)
        x0 = env.env.unwrapped.data.qpos[0]
        for _ in range(3):
            env.step(GAMES[game]["actions"][-1])
        assert env.env.unwrapped.data.qpos[0] > x0  # the last action pushes toward +x, screen right
        env.close()


def test_episodes_are_seeded_and_unthresholded_games_report_no_solved_rate():
    a = play_episodes("Walker2d", random_policy(1), episodes=2, seed=5, max_steps=30)
    b = play_episodes("Walker2d", random_policy(1), episodes=2, seed=5, max_steps=30)
    assert a["results"] == b["results"] and a["solved_rate"] is None


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


def test_questions_offer_each_games_pushes():
    from laya.vlm import VLMAgent

    for game, spec in GAMES.items():
        internal = VLMAgent._to_internal(control_question(game)["action"])
        assert internal["t"] == "choice" and tuple(internal["crit"]) == spec["actions"]
