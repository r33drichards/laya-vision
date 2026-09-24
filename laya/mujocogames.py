"""MuJoCo: InvertedPendulum and InvertedDoublePendulum from Gymnasium, played from pixels.

The same closed-loop test as ``laya.controlgames``, which this module reuses: each step the environment's
``rgb_array`` rendering, with the previous frame ghosted underneath, is the image and the model answers one
``choice`` (``laya.games.control_question``). MuJoCo actions are continuous, so each game names a few pushes, a
fixed share of the actuator's range: three for the single pole, five for the double pole, which needs a gentle push
as well as a hard one. The camera is a fixed side view of the rail, so the cart moves across the frame and a push
to the right moves it right on screen. Episodes end when the pole falls or at 1000 steps; the score is the
environment's summed reward, ``normalized`` = (model - random) / (expert - random) as for classic control.

Experts use the same named pushes as the model. InvertedPendulum: a linear controller on cart and pole state with a
dead band (1000, the cap). InvertedDoublePendulum: LQR on the upright linearization of one environment step,
rounded to the nearest push (about 9350, the cap). Random play falls within about 5 steps (returns about
4 and 36).

Needs ``gymnasium[mujoco]``; imported lazily. Headless Linux renders through EGL (``MUJOCO_GL=egl``, set when no
display is present; ``libegl1``) or OSMesa (``MUJOCO_GL=osmesa``, ``libosmesa6``).
"""
import os
import sys
from typing import Dict

import numpy as np

from laya.controlgames import ControlGame, expert_policy, model_policy, normalized, random_policy
from laya.controlgames import play_episodes as _play_episodes

SIZE = 256  # rendered frame, pixels a side


def _side_view(lookat_z: float, distance: float) -> Dict:
    return {"trackbodyid": -1, "lookat": np.array([0.0, 0.0, lookat_z]), "distance": distance, "elevation": 0.0,
            "azimuth": 90.0}


# game -> Gymnasium id, action names and each one's share of the actuator range, the camera, and the score counted
# as solved (Gymnasium's reward threshold)
GAMES = {
    "InvertedPendulum": {"env_id": "InvertedPendulum-v5", "actions": ("LEFT", "NONE", "RIGHT"),
                         "pushes": (-1.0, 0.0, 1.0), "camera": _side_view(0.25, 2.6), "solved": 950.0},
    "InvertedDoublePendulum": {"env_id": "InvertedDoublePendulum-v5",
                               "actions": ("HARD_LEFT", "LEFT", "NONE", "RIGHT", "HARD_RIGHT"),
                               "pushes": (-1.0, -0.5, 0.0, 0.5, 1.0), "camera": _side_view(0.6, 3.2),
                               "solved": 9100.0},
}

# InvertedDoublePendulum-v5 LQR gain on (x, angle1, angle2, x', angle1', angle2'): the environment step (5 RK4 steps
# of 0.01 s) finite-differenced about upright rest, Q = diag(1, 10, 10, .1, .1, .1), R = 1, Riccati iterated to
# convergence. The control is -K @ state, in actuator units.
DOUBLE_K = np.array([0.2128, 1.8131, 5.5687, 0.3149, 0.9024, 0.8482])


def _nearest(pushes, u: float) -> int:
    return int(np.argmin(np.abs(np.asarray(pushes) - np.clip(u, -1.0, 1.0))))


def _expert_action(game: str, obs) -> int:
    if game == "InvertedPendulum":
        x, theta, x_dot, theta_dot = obs
        u = 0.3 * x + 5.0 * theta + 0.5 * x_dot + theta_dot
        return 2 if u > 0.05 else 0 if u < -0.05 else 1
    if game == "InvertedDoublePendulum":
        x, s1, s2, c1, c2, x_dot, w1, w2 = obs[:8]
        state = np.array([x, np.arctan2(s1, c1), np.arctan2(s2, c2), x_dot, w1, w2])
        return _nearest(GAMES[game]["pushes"], -float(DOUBLE_K @ state))
    raise ValueError(game)


class MujocoGame(ControlGame):
    """One seeded episode of a ``GAMES`` environment, stepped by action name."""

    SPECS = GAMES

    def _make(self, spec: Dict):
        if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
            os.environ.setdefault("MUJOCO_GL", "egl")  # read when mujoco is first imported
        import gymnasium as gym

        env = gym.make(spec["env_id"], render_mode="rgb_array", width=SIZE, height=SIZE,
                       default_camera_config=spec["camera"])
        self._high = float(env.action_space.high[0])
        return env

    def _env_action(self, index: int):
        return np.array([self.SPECS[self.game]["pushes"][index] * self._high], dtype=np.float32)

    def expert(self) -> str:
        return self.actions[_expert_action(self.game, self.obs)]


def play_episodes(game: str, policy, episodes: int, seed: int = 0, max_steps: int = 0) -> Dict:
    """``laya.controlgames.play_episodes`` over ``MujocoGame``."""
    return _play_episodes(game, policy, episodes, seed, max_steps, make=MujocoGame)


__all__ = ["GAMES", "MujocoGame", "play_episodes", "normalized", "expert_policy", "random_policy", "model_policy"]
