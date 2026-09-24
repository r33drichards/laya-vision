"""MuJoCo pendulums (``laya.mujocogames``): experts beat random, named pushes, fixed side view, no model."""
import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("gymnasium")

from laya.games import CONTROL_ACTIONS, control_question  # noqa: E402
from laya.mujocogames import GAMES, MujocoGame, expert_policy, play_episodes, random_policy  # noqa: E402


@pytest.mark.parametrize("game", sorted(GAMES))
def test_expert_solves_and_random_does_not(game):
    exp = play_episodes(game, expert_policy, episodes=3, seed=200_000)
    rnd = play_episodes(game, random_policy(0), episodes=3, seed=200_000)
    assert exp["solved_rate"] == 1.0 and exp["mean_steps"] == 1000  # the pole stays up to the time limit
    assert rnd["solved_rate"] == 0.0 and rnd["mean_steps"] < 50
    assert set(exp["actions"]) <= set(GAMES[game]["actions"])


def test_pushes_scale_the_actuator_range_and_right_moves_right():
    for game, spec in GAMES.items():
        env = MujocoGame(game, seed=0)
        high = float(env.env.action_space.high[0])
        for i, push in enumerate(spec["pushes"]):
            assert env._env_action(i).tolist() == [pytest.approx(push * high)]
        x0 = env.env.unwrapped.data.qpos[0]
        for _ in range(3):
            env.step(spec["actions"][-1])
        assert env.env.unwrapped.data.qpos[0] > x0  # the last action pushes toward +x, screen right
        env.close()


def test_episodes_are_seeded():
    a = play_episodes("InvertedDoublePendulum", random_policy(1), episodes=2, seed=5)
    b = play_episodes("InvertedDoublePendulum", random_policy(1), episodes=2, seed=5)
    assert a["results"] == b["results"]


def test_render_is_a_fixed_side_view_with_the_previous_frame_ghosted():
    env = MujocoGame("InvertedPendulum", seed=0)
    first = np.asarray(env.render())
    assert first.shape == (256, 256, 3) and first.dtype == np.uint8
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
        assert tuple(CONTROL_ACTIONS[game]) == spec["actions"] and len(spec["pushes"]) == len(spec["actions"])
        internal = VLMAgent._to_internal(control_question(game)["action"])
        assert internal["t"] == "choice" and tuple(internal["crit"]) == spec["actions"]
