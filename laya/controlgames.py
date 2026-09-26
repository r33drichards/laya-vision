"""Classic control: CartPole, Acrobot, MountainCar and LunarLander from Gymnasium, played from pixels.

The same closed-loop test as ``laya.gridgames``: each step the environment's own ``rgb_array`` rendering is the
image and the model answers one ``choice`` over the game's discrete actions (``laya.games.control_question``).
A scripted expert and a random policy play the same seeded episodes, so a score reads as
``normalized`` = (model - random) / (expert - random), as for Atari.

A single frame hides velocity, which all four games need (which way is the pole falling, is the car rolling back),
so ``render`` ghosts the previous frame under the current one: the faint copy is where things were one step
earlier. That is the ``"single"`` game frame mode for these games (``laya.frames``: ``trail-2``); the other
modes build their state from ``frame()`` over the episode's history instead. Episodes end when the environment
terminates or at its own time limit (CartPole 500 steps, Acrobot 500, MountainCar 200, LunarLander 1000); the
score is the environment's summed reward.

Experts (10 episodes on the eval seeds): CartPole a linear controller on the pole angle and cart state (500, the
cap); Acrobot torque along the lower joint's velocity (about -85); MountainCar push the way the car is moving
(about -120); LunarLander Gymnasium's own ``heuristic`` controller (about 270; 200 counts as solved).

Needs ``gymnasium[classic-control,box2d]`` (pygame for rendering, Box2D for LunarLander); imported lazily so the
rest of ``laya`` does not depend on it.
"""
import os
import random
from typing import Dict, Optional

import numpy as np

from . import frames as F
from .frames import GHOST, blend  # GHOST: weight of the previous frame in ``render`` (0.35)

# game -> Gymnasium id, action names in the environment's action order, and the score counted as solved
GAMES = {
    "CartPole": {"env_id": "CartPole-v1", "actions": ("LEFT", "RIGHT"), "solved": 475.0},
    "Acrobot": {"env_id": "Acrobot-v1", "actions": ("CLOCKWISE", "NONE", "COUNTERCLOCKWISE"), "solved": -100.0},
    "MountainCar": {"env_id": "MountainCar-v0", "actions": ("LEFT", "NONE", "RIGHT"), "solved": -110.0},
    "LunarLander": {"env_id": "LunarLander-v3", "actions": ("NOOP", "LEFT_ENGINE", "MAIN_ENGINE", "RIGHT_ENGINE"),
                    "solved": 200.0},
}


def _expert_action(game: str, env, obs) -> int:
    if game == "CartPole":
        x, x_dot, theta, theta_dot = obs
        return int(theta + 0.5 * theta_dot + 0.01 * x + 0.1 * x_dot > 0)
    if game == "Acrobot":
        return 2 if obs[5] > 0 else 0 if obs[5] < 0 else 1  # pump energy: torque along the lower joint's swing
    if game == "MountainCar":
        return 2 if obs[1] >= 0 else 0  # push the way the car is already rolling
    if game == "LunarLander":
        from gymnasium.envs.box2d.lunar_lander import heuristic

        return int(heuristic(env.unwrapped, obs))
    raise ValueError(game)


class ControlGame:
    """One seeded episode of a ``GAMES`` environment, stepped by action name."""

    def __init__(self, game: str, seed: int = 0):
        if game not in GAMES:
            raise ValueError("unknown control game %r (%s)" % (game, ", ".join(GAMES)))
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")  # headless pygame, no audio device
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
        import gymnasium as gym

        self.game, self.actions = game, GAMES[game]["actions"]
        self.env = gym.make(GAMES[game]["env_id"], render_mode="rgb_array")
        self.obs, _ = self.env.reset(seed=seed)
        self.env.action_space.seed(seed)
        self.score, self.steps, self.terminated, self.truncated = 0.0, 0, False, False
        self._frame = self._prev = None

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated

    def expert(self) -> str:
        return self.actions[_expert_action(self.game, self.env, self.obs)]

    def step(self, action: str) -> float:
        self._prev, self._frame = self._frame, None
        self.obs, reward, self.terminated, self.truncated, _ = self.env.step(self.actions.index(action))
        self.score += float(reward)
        self.steps += 1
        return float(reward)

    def frame(self) -> np.ndarray:
        """The environment's current RGB frame (rendered once per step)."""
        if self._frame is None:
            self._frame = self.env.render()
        return self._frame

    def render(self):
        """The current frame with the previous one ghosted underneath, as a PIL image."""
        from PIL import Image

        cur = self.frame()
        return Image.fromarray(cur if self._prev is None else blend([self._prev, cur]))

    def close(self) -> None:
        self.env.close()


def play_episodes(game: str, policy, episodes: int, seed: int = 0, max_steps: int = 0) -> Dict:
    """Play ``episodes`` seeded episodes (seed ``seed + i``); ``policy(env) -> action name``. ``max_steps`` caps
    an episode below the environment's own limit (0 = no extra cap)."""
    from collections import Counter

    counts, eps = Counter(), []
    for i in range(episodes):
        env = ControlGame(game, seed + i)
        while not env.done and not (max_steps and env.steps >= max_steps):
            a = policy(env)
            counts[a] += 1
            env.step(a)
        eps.append({"score": round(env.score, 3), "steps": env.steps, "terminated": env.terminated})
        env.close()
    scores = [e["score"] for e in eps]
    return {"game": game, "episodes": episodes, "seed": seed, "actions": dict(counts), "results": eps,
            "mean_score": float(np.mean(scores)), "std_score": float(np.std(scores)),
            "solved_rate": float(np.mean([s >= GAMES[game]["solved"] for s in scores])),
            "mean_steps": float(np.mean([e["steps"] for e in eps]))}


def normalized(model: float, rnd: float, expert: float) -> Optional[float]:
    """(model - random) / (expert - random); ``None`` when the baselines tie."""
    return None if expert == rnd else (model - rnd) / (expert - rnd)


def expert_policy(env) -> str:
    return env.expert()


def random_policy(seed: int = 0):
    rng = random.Random(seed)
    return lambda env: rng.choice(env.actions)


def model_policy(agent, game: str, mode: str = "single"):
    """The model's most likely action, via ``predict``, in the game frame mode ``mode``
    (``laya.frames.mode_for(agent.cfg, "control")``): ``single`` sends the rendered (ghosted) screen, exactly as
    before modes existed; any other mode sends ``laya.frames.state`` of the episode's ``frame()`` history
    (``trail-N`` one blended image, ``stack-N`` N images) with the question describing that screen."""
    from laya.games import control_question

    if mode == "single":
        q = control_question(game)
        return lambda env: agent.predict({"image": env.render()}, q)["answers"]["action"]["choice"]
    q = control_question(game, mode)
    return F.episode_policy(lambda env, st: agent.predict(st, q)["answers"]["action"]["choice"],
                            lambda env: env.frame(), mode, "control")


__all__ = ["GAMES", "ControlGame", "play_episodes", "normalized", "expert_policy", "random_policy", "model_policy"]
