"""Classic control (``laya.controlgames``): experts beat random, seeding, ghosted rendering, no model."""
import numpy as np
import pytest

from laya.controlgames import GAMES, ControlGame, expert_policy, normalized, play_episodes, random_policy
from laya.games import CONTROL_ACTIONS, control_question

pytest.importorskip("gymnasium")
pytest.importorskip("pygame")

# the expert's mean over 3 eval-seed episodes must reach this; random play stays well below
EXPERT_FLOOR = {"CartPole": 475.0, "Acrobot": -150.0, "MountainCar": -160.0, "LunarLander": 200.0}


@pytest.mark.parametrize("game", sorted(GAMES))
def test_expert_beats_random(game):
    if game == "LunarLander":
        pytest.importorskip("Box2D")
    exp = play_episodes(game, expert_policy, episodes=3, seed=200_000)
    rnd = play_episodes(game, random_policy(0), episodes=3, seed=200_000)
    assert exp["mean_score"] >= EXPERT_FLOOR[game] > rnd["mean_score"]
    assert set(exp["actions"]) <= set(GAMES[game]["actions"]) and sum(exp["actions"].values()) == sum(
        e["steps"] for e in exp["results"])
    assert normalized(exp["mean_score"], rnd["mean_score"], exp["mean_score"]) == 1.0


def test_episodes_are_seeded_and_capped():
    a = play_episodes("CartPole", random_policy(1), episodes=2, seed=5)
    b = play_episodes("CartPole", random_policy(1), episodes=2, seed=5)
    assert a["results"] == b["results"]
    capped = play_episodes("MountainCar", expert_policy, episodes=1, seed=0, max_steps=7)
    assert capped["results"][0]["steps"] == 7 and not capped["results"][0]["terminated"]


def test_render_ghosts_the_previous_frame():
    env = ControlGame("CartPole", seed=0)
    first = np.asarray(env.render())
    assert first.ndim == 3 and first.shape[2] == 3 and first.dtype == np.uint8
    for _ in range(3):
        env.render()  # a model policy renders every step, which keeps the previous frame
        env.step("RIGHT")
    raw = env.frame()
    ghosted = np.asarray(env.render())
    assert ghosted.shape == raw.shape and (ghosted != raw).any()  # the previous frame shows through
    env.close()


def test_questions_offer_each_games_actions():
    from laya.vlm import VLMAgent

    for game, spec in GAMES.items():
        assert tuple(CONTROL_ACTIONS[game]) == spec["actions"]
        internal = VLMAgent._to_internal(control_question(game)["action"])
        assert internal["t"] == "choice" and tuple(internal["crit"]) == spec["actions"]
