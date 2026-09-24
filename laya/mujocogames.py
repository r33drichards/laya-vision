"""MuJoCo: Gymnasium's eleven MuJoCo environments (``-v5``), played from pixels.

The same closed-loop test as ``laya.controlgames``, which this module reuses: each step the environment's
``rgb_array`` rendering, with the previous frame ghosted underneath, is the image and the model answers one
``choice`` (``laya.games.control_question``). MuJoCo actions are continuous, so each game names a few pushes, each a
fixed action vector in units of the actuator range:

- InvertedPendulum and InvertedDoublePendulum push the cart: three pushes for the single pole, five for the double
  pole, which needs a gentle push as well as a hard one. A fixed side view of the rail: a push right moves the cart
  right on screen.
- The other nine (Reacher, Pusher, Swimmer, Hopper, Walker2d, HalfCheetah, Ant, Humanoid, HumanoidStandup) torque
  one joint at a time: ``NONE``, then ``<JOINT>_POS`` / ``<JOINT>_NEG`` for each actuated joint
  (``laya.games.TORQUE_JOINTS``), 5 options for Reacher up to 35 for Humanoid. Every other joint gets no torque that
  step, so a gait that needs several joints at once has to be built across steps. Reacher and Pusher get a closer
  camera; the rest keep Gymnasium's own, which follows the body.

Episodes end when the environment terminates (a fallen pole or robot) or at its own time limit (50 steps for Reacher,
100 for Pusher, 1000 for the rest); the score is the environment's summed reward, ``normalized`` = (model - random)
/ (expert - random) as for classic control, ``None`` where a game has no expert.

Experts use the same named pushes as the model. InvertedPendulum: a linear controller on cart and pole state with a
dead band. InvertedDoublePendulum: LQR on the upright linearization of one environment step, rounded to the nearest
push. Both hold the pole to the 1000-step cap. Seven torque games use a lookahead planner (``lookahead``): it tries
each push, held for a few steps, in the simulator itself and takes the best return, so it sees the true state that
the model has to read off the screen. On Pusher and Humanoid no such planner beat doing nothing, so they have no
expert; compare with ``random_policy`` and ``still_policy`` (always ``NONE``).

Needs ``gymnasium[mujoco]``; imported lazily. Headless Linux renders through EGL (``MUJOCO_GL=egl``, set when no
display is present; ``libegl1``) or OSMesa (``MUJOCO_GL=osmesa``, ``libosmesa6``).
"""
import os
import sys
from typing import Dict, List, Optional

import numpy as np

from laya.controlgames import ControlGame, expert_policy, model_policy, normalized, random_policy
from laya.controlgames import play_episodes as _play_episodes
from laya.games import CONTROL_ACTIONS, TORQUE_JOINTS

SIZE = 256  # rendered frame, pixels a side


def _view(lookat, distance: float, elevation: float) -> Dict:
    return {"trackbodyid": -1, "lookat": np.array(lookat, dtype=float), "distance": distance,
            "elevation": elevation, "azimuth": 90.0}


def _torque_pushes(n: int, size: float) -> List[List[float]]:
    """``NONE``, then a push of ``size`` (share of the range) each way on each of ``n`` joints, as in
    ``laya.games.torque_actions``."""
    out = [[0.0] * n]
    for j in range(n):
        for sign in (1.0, -1.0):
            out.append([sign * size if k == j else 0.0 for k in range(n)])
    return out


def _torque_game(game: str, size: float = 1.0, lookahead: Optional[int] = None, camera: Optional[Dict] = None,
                 solved: Optional[float] = None) -> Dict:
    return {"env_id": game + "-v5", "actions": tuple(CONTROL_ACTIONS[game]),
            "pushes": _torque_pushes(len(TORQUE_JOINTS[game]), size), "camera": camera,
            "expert": ("lookahead", lookahead) if lookahead else None, "solved": solved}


# game -> Gymnasium id, action names and each one's action vector (share of the actuator range), the camera (None =
# Gymnasium's), the expert (None = none), and the score counted as solved (Gymnasium's reward threshold, if any).
# Torque sizes and lookahead horizons were tuned on seeds 200000-200002 against random and always-NONE play.
GAMES = {
    "InvertedPendulum": {"env_id": "InvertedPendulum-v5", "actions": ("LEFT", "NONE", "RIGHT"),
                         "pushes": [[-1.0], [0.0], [1.0]], "camera": _view((0, 0, 0.25), 2.6, 0.0),
                         "expert": ("pendulum", None), "solved": 950.0},
    "InvertedDoublePendulum": {"env_id": "InvertedDoublePendulum-v5",
                               "actions": ("HARD_LEFT", "LEFT", "NONE", "RIGHT", "HARD_RIGHT"),
                               "pushes": [[-1.0], [-0.5], [0.0], [0.5], [1.0]],
                               "camera": _view((0, 0, 0.6), 3.2, 0.0), "expert": ("double", None), "solved": 9100.0},
    # Reacher charges the squared torque every step, so its pushes are small
    "Reacher": _torque_game("Reacher", 0.3, 20, _view((0, 0, 0), 0.55, -90.0), -3.75),
    "Pusher": _torque_game("Pusher", 0.5, None, _view((0.3, -0.2, -0.3), 2.0, -75.0), 0.0),
    "Swimmer": _torque_game("Swimmer", 1.0, 5, None, 360.0),
    "Hopper": _torque_game("Hopper", 1.0, 5, None, 3800.0),
    "Walker2d": _torque_game("Walker2d", 1.0, 5),
    "HalfCheetah": _torque_game("HalfCheetah", 1.0, 5, None, 4800.0),
    "Ant": _torque_game("Ant", 1.0, 5, None, 6000.0),
    "Humanoid": _torque_game("Humanoid", 1.0),
    "HumanoidStandup": _torque_game("HumanoidStandup", 1.0, 5),
}

# InvertedDoublePendulum-v5 LQR gain on (x, angle1, angle2, x', angle1', angle2'): the environment step (5 RK4 steps
# of 0.01 s) finite-differenced about upright rest, Q = diag(1, 10, 10, .1, .1, .1), R = 1, Riccati iterated to
# convergence. The control is -K @ state, in actuator units.
DOUBLE_K = np.array([0.2128, 1.8131, 5.5687, 0.3149, 0.9024, 0.8482])
FALL_PENALTY = 1000.0  # what ``lookahead`` subtracts from a push that ends the episode


def _nearest(pushes, u: float) -> int:
    return int(np.argmin(np.abs(np.asarray(pushes)[:, 0] - np.clip(u, -1.0, 1.0))))


class MujocoGame(ControlGame):
    """One seeded episode of a ``GAMES`` environment, stepped by action name."""

    SPECS = GAMES

    def _make(self, spec: Dict):
        if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
            os.environ.setdefault("MUJOCO_GL", "egl")  # read when mujoco is first imported
        import gymnasium as gym

        kw = {"default_camera_config": spec["camera"]} if spec["camera"] else {}
        env = gym.make(spec["env_id"], render_mode="rgb_array", width=SIZE, height=SIZE, **kw)
        self._pushes = np.asarray(spec["pushes"], dtype=np.float64) * env.action_space.high
        return env

    def _env_action(self, index: int):
        return self._pushes[index].astype(np.float32)

    def lookahead(self, horizon: int) -> int:
        """The push whose return, held for ``horizon`` steps from the current state, is highest (a push that ends
        the episode loses ``FALL_PENALTY``). Runs on the simulator and restores it, so the episode is unchanged."""
        u = self.env.unwrapped
        d = u.data
        saved = [a.copy() for a in (d.qpos, d.qvel, d.act, d.qacc_warmstart, d.ctrl)]
        t0 = d.time
        best, best_i = -np.inf, 0
        for i, push in enumerate(self._pushes):
            ret = 0.0
            for _ in range(horizon):
                _, r, term, _, _ = u.step(push)
                ret += r
                if term:
                    ret -= FALL_PENALTY
                    break
            if ret > best:
                best, best_i = ret, i
            u.set_state(saved[0], saved[1])
            d.act[:], d.qacc_warmstart[:], d.ctrl[:] = saved[2], saved[3], saved[4]
            d.time = t0
        return best_i

    def expert(self) -> str:
        kind, horizon = self.SPECS[self.game]["expert"] or (None, None)
        obs = self.obs
        if kind == "pendulum":
            x, theta, x_dot, theta_dot = obs
            u = 0.3 * x + 5.0 * theta + 0.5 * x_dot + theta_dot
            i = 2 if u > 0.05 else 0 if u < -0.05 else 1
        elif kind == "double":
            x, s1, s2, c1, c2, x_dot, w1, w2 = obs[:8]
            state = np.array([x, np.arctan2(s1, c1), np.arctan2(s2, c2), x_dot, w1, w2])
            i = _nearest(self.SPECS[self.game]["pushes"], -float(DOUBLE_K @ state))
        elif kind == "lookahead":
            i = self.lookahead(horizon)
        else:
            raise ValueError("%s has no expert" % self.game)
        return self.actions[i]


def still_policy(env) -> str:
    """Always ``NONE``: the baseline of doing nothing, which a robot that only has to not fall can beat random with."""
    return "NONE"


def has_expert(game: str) -> bool:
    return GAMES[game]["expert"] is not None


def play_episodes(game: str, policy, episodes: int, seed: int = 0, max_steps: int = 0) -> Dict:
    """``laya.controlgames.play_episodes`` over ``MujocoGame``."""
    return _play_episodes(game, policy, episodes, seed, max_steps, make=MujocoGame)


__all__ = ["GAMES", "MujocoGame", "play_episodes", "normalized", "expert_policy", "random_policy", "still_policy",
           "has_expert", "model_policy"]
