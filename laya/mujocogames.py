"""MuJoCo: Gymnasium's eleven MuJoCo environments (``-v5``), played from pixels.

The same closed-loop test as ``laya.controlgames``, which this module reuses: each step the environment's
``rgb_array`` rendering, with the previous frame ghosted underneath, is the image. MuJoCo actions are continuous, so
the model picks named levels:

- InvertedPendulum and InvertedDoublePendulum push the cart, one ``choice`` per step (``laya.games.control_question``):
  three pushes for the single pole, five for the double pole. A fixed side view of the rail: a push right moves the
  cart right on screen.
- The other nine (Reacher, Pusher, Swimmer, Hopper, Walker2d, HalfCheetah, Ant, Humanoid, HumanoidStandup) set every
  joint at once: one ``choice`` per joint (``laya.games.mujoco_questions``) over five torque levels (``LEVELS``: full
  or half torque either way, or none), all answered in one ``predict``, which encodes the frame once. An action is a
  dict ``{joint key: level}``. Reacher and Pusher get a closer camera; the rest keep Gymnasium's own, which follows
  the body.

Why five levels per joint: mapping a strong expert's actions onto candidate discrete spaces, one joint pushed per
step kept at most 6% of its return on the walking robots, and 8-128 fixed multi-joint presets at most 53%, while five
levels per joint kept 78-100% (Walker2d .98, HumanoidStandup 1.00, Humanoid and Hopper .86).

Several cameras (opt-in): ``MujocoGame(..., views=...)`` renders named fixed views from ``VIEWS`` (the robots:
side, front, top, three-quarter, following the body; Reacher and Pusher: top and side) and the model gets them all in
one ``predict`` (``{"images": [...]}``, 64 image tokens each), with each question naming the cameras. Without
``views`` everything is as described above.

Episodes end when the environment terminates (a fallen pole or robot) or at its own time limit (50 steps for Reacher,
100 for Pusher, 1000 for the rest); the score is the environment's summed reward, ``normalized`` = (model - random)
/ (expert - random) as for classic control.

Experts use the same levels as the model. InvertedPendulum: a linear controller on cart and pole state with a dead
band. InvertedDoublePendulum: LQR on the upright linearization of one environment step, rounded to the nearest push.
Both hold the pole to the 1000-step cap. The torque games use the Farama Foundation's pretrained Stable-Baselines3
experts on the Hugging Face Hub (``HUB_EXPERTS``, pinned by commit; no licence is declared on them), their continuous
action rounded per joint to the nearest level (``expert_action`` gives the unrounded action, for soft labels). The
Ant expert matches its card only on mujoco 3.2.x; on 3.14 it falls in about a third of episodes. Needs
``stable-baselines3`` and ``sb3-contrib`` (TQC) for the experts.

Needs ``gymnasium[mujoco]``; imported lazily. Headless Linux renders through EGL (``MUJOCO_GL=egl``, set when no
display is present; ``libegl1``) or OSMesa (``MUJOCO_GL=osmesa``, ``libosmesa6``).
"""
import os
import random
import sys
from typing import Dict, List, Optional

import numpy as np

from laya.controlgames import GHOST, ControlGame, normalized
from laya.controlgames import play_episodes as _play_episodes
from laya.games import CONTROL_ACTIONS, TORQUE_JOINTS, TORQUE_LEVELS, joint_key

SIZE = 256  # rendered frame, pixels a side
LEVELS = tuple(TORQUE_LEVELS)  # STRONG_NEG, NEG, NONE, POS, STRONG_POS
LEVEL_VALUES = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])  # each level's share of the actuator range


def _view(lookat, distance: float, elevation: float) -> Dict:
    return {"trackbodyid": -1, "lookat": np.array(lookat, dtype=float), "distance": distance,
            "elevation": elevation, "azimuth": 90.0}


# game -> (Hub repo, file, commit, algorithm): the Farama Foundation's SB3 experts, the checkpoint in each repo whose
# return matches its model card (Hopper and Walker2d hold a second, weaker zip)
HUB_EXPERTS = {
    "Reacher": ("farama-minari/Reacher-v5-SAC-expert", "reacher-v5-sac-expert.zip",
                "ad8fadfbe3ab40ce9c1bae5b33641041056b3090", "SAC"),
    "Pusher": ("farama-minari/Pusher-v5-SAC-expert", "pusher-v5-sac-expert.zip",
               "3e103602f3fba3500ee6f21d762c0ff3e94892ff", "SAC"),
    "Swimmer": ("farama-minari/Swimmer-v5-PPO-expert", "swimmer-v5-PPO-expert.zip",
                "11ae7c53a3e3156e5399eea8855c64329a107d56", "PPO"),
    "Hopper": ("farama-minari/Hopper-v5-SAC-expert", "hopper-v5-SAC-expert.zip",
               "bff2655df9e6bfcafa9edb1aa88cf1541f9ee3af", "SAC"),
    "Walker2d": ("farama-minari/Walker2d-v5-SAC-expert", "walker2d-v5-SAC-expert.zip",
                 "2b0ab400beed313a86d84bd82a6e120aa89a61e3", "SAC"),
    "HalfCheetah": ("farama-minari/HalfCheetah-v5-TQC-expert", "halfcheetah-v5-TQC-expert.zip",
                    "995505ac89e1a96bec541dfe9ec8bd86b7e46fda", "TQC"),
    "Ant": ("farama-minari/Ant-v5-SAC-expert", "ant-v5-sac-expert.zip",
            "e2eb33e2b09c51a82716e1f106d1ff5ea19a88b5", "SAC"),
    "Humanoid": ("farama-minari/Humanoid-v5-TQC-expert", "humanoid-v5-TQC-expert.zip",
                 "e5a86ffdb70e6f4750f39c0464ac026a8437001a", "TQC"),
    "HumanoidStandup": ("farama-minari/HumanoidStandup-v5-SAC-expert", "humanoidstandup-v5-SAC-expert.zip",
                        "a8a03edfc28ef8713f73b3d25445eedd4bbb776d", "SAC"),
}


def _torque_game(game: str, camera: Optional[Dict] = None, solved: Optional[float] = None) -> Dict:
    joints = tuple(joint_key(j) for j in TORQUE_JOINTS[game])
    return {"env_id": game + "-v5", "actions": LEVELS, "joints": joints, "camera": camera, "expert": ("hub", None),
            "solved": solved}


# game -> Gymnasium id, the action names (per joint for torque games), the joints (torque games), the camera (None =
# Gymnasium's), the expert, and the score counted as solved (Gymnasium's reward threshold, if any)
GAMES = {
    "InvertedPendulum": {"env_id": "InvertedPendulum-v5", "actions": tuple(CONTROL_ACTIONS["InvertedPendulum"]),
                         "pushes": [[-1.0], [0.0], [1.0]], "camera": _view((0, 0, 0.25), 2.6, 0.0),
                         "expert": ("pendulum", None), "solved": 950.0},
    "InvertedDoublePendulum": {"env_id": "InvertedDoublePendulum-v5",
                               "actions": tuple(CONTROL_ACTIONS["InvertedDoublePendulum"]),
                               "pushes": [[-1.0], [-0.5], [0.0], [0.5], [1.0]],
                               "camera": _view((0, 0, 0.6), 3.2, 0.0), "expert": ("double", None), "solved": 9100.0},
    "Reacher": _torque_game("Reacher", _view((0, 0, 0), 0.55, -90.0), -3.75),
    "Pusher": _torque_game("Pusher", _view((0.3, -0.2, -0.3), 2.0, -75.0), 0.0),
    "Swimmer": _torque_game("Swimmer", None, 360.0),
    "Hopper": _torque_game("Hopper", None, 3800.0),
    "Walker2d": _torque_game("Walker2d"),
    "HalfCheetah": _torque_game("HalfCheetah", None, 4800.0),
    "Ant": _torque_game("Ant", None, 6000.0),
    "Humanoid": _torque_game("Humanoid"),
    "HumanoidStandup": _torque_game("HumanoidStandup"),
}


def _track(distance: float, azimuth: float, elevation: float, body: int = 1) -> Dict:
    """A camera that follows ``body``'s subtree centre of mass (body 1 is every robot's root: torso or its like)."""
    return {"trackbodyid": body, "distance": distance, "azimuth": azimuth, "elevation": elevation}


def _fixed(lookat, distance: float, azimuth: float, elevation: float) -> Dict:
    """A camera fixed in the world, looking at ``lookat``."""
    return {"lookat": np.array(lookat, dtype=float), "distance": distance, "azimuth": azimuth, "elevation": elevation}


def _body_views(distance: float, first: str = "side") -> Dict:
    """The robots' views: the environment's own camera (``first``, a side view that follows the robot) and three
    that follow its root body from ``distance``: from ahead (the robot walks toward +x, so toward it), straight
    down (+x to the right, as in the side view), and a three-quarter view from front-left and above."""
    return {first: None, "front": _track(distance, 180.0, -10.0), "top": _track(distance, 90.0, -89.0),
            "three_quarter": _track(distance, 135.0, -25.0)}


# game -> {view name: camera}, the first being the game's single view (None = the environment's own camera, exactly
# as rendered without ``views``). Azimuth 90 looks along +y, so +x is screen right; 180 looks along -x.
VIEWS = {
    "InvertedPendulum": {"side": None},
    "InvertedDoublePendulum": {"side": None},
    "Reacher": {"top": None, "side": _fixed((0, 0, 0.02), 0.5, 90.0, -30.0)},
    "Pusher": {"top": None, "side": _fixed((0.2, -0.3, -0.2), 2.0, 180.0, -15.0)},
    "Swimmer": {"oblique": None, "top": _track(3.0, 90.0, -89.0)},
    "Hopper": _body_views(3.0),
    "Walker2d": _body_views(4.0),
    "HalfCheetah": _body_views(4.0),
    "Ant": _body_views(4.0),
    "Humanoid": _body_views(4.0),
    "HumanoidStandup": _body_views(4.0),
}


def resolve_views(game: str, views=None) -> tuple:
    """``views`` as a tuple of ``VIEWS[game]`` names: None -> the game's single view, ``"all"`` -> every view, a
    name or comma-separated names -> those views."""
    names = VIEWS[game]
    if views is None:
        return (next(iter(names)),)
    if isinstance(views, str):
        views = tuple(names) if views == "all" else tuple(v for v in views.split(",") if v)
    views = tuple(views)
    bad = [v for v in views if v not in names]
    if bad or not views or len(set(views)) != len(views):
        raise ValueError("%s views must be distinct names from %s, got %r" % (game, ", ".join(names), views))
    return views


# InvertedDoublePendulum-v5 LQR gain on (x, angle1, angle2, x', angle1', angle2'): the environment step (5 RK4 steps
# of 0.01 s) finite-differenced about upright rest, Q = diag(1, 10, 10, .1, .1, .1), R = 1, Riccati iterated to
# convergence. The control is -K @ state, in actuator units.
DOUBLE_K = np.array([0.2128, 1.8131, 5.5687, 0.3149, 0.9024, 0.8482])
_HUB_MODELS: Dict[str, object] = {}


def _nearest(pushes, u: float) -> int:
    return int(np.argmin(np.abs(np.asarray(pushes)[:, 0] - np.clip(u, -1.0, 1.0))))


def hub_expert(game: str):
    """The ``HUB_EXPERTS`` policy for ``game``, downloaded at its pinned commit and loaded on CPU (cached)."""
    if game not in _HUB_MODELS:
        from huggingface_hub import hf_hub_download

        repo, fname, sha, algo = HUB_EXPERTS[game]
        if algo == "TQC":
            from sb3_contrib import TQC as cls
        else:
            import stable_baselines3

            cls = getattr(stable_baselines3, algo)
        path = hf_hub_download(repo, fname, revision=sha)
        # schedules pickled under another Python fail to load and are not needed for inference
        _HUB_MODELS[game] = cls.load(path, device="cpu", custom_objects={
            "learning_rate": 0.0, "lr_schedule": lambda _: 0.0, "clip_range": lambda _: 0.0})
    return _HUB_MODELS[game]


def is_torque_game(game: str) -> bool:
    return "joints" in GAMES[game]


class MujocoGame(ControlGame):
    """One seeded episode of a ``GAMES`` environment. Pendulums step by action name; torque games step by a dict
    ``{joint key: level}`` (``joints`` lists the keys).

    ``views`` (opt-in, see ``resolve_views``) picks cameras from ``VIEWS[game]``. With one view (the default: the
    game's own camera, as ever) ``render()`` is one ghosted PIL image and ``state()`` ``{"image": ...}``; with
    several, ``render()`` is a list of them in ``views`` order, ``state()`` ``{"images": [...]}`` and
    ``questions()`` names the cameras. Extra views render through the environment's own offscreen viewer (same GL
    context, scene options and lighting) with a camera of their own swapped in for the shot, so each view depends
    only on the simulation state."""

    SPECS = GAMES

    def __init__(self, game: str, seed: int = 0, views=None):
        if game not in self.SPECS:
            raise ValueError("unknown control game %r (%s)" % (game, ", ".join(self.SPECS)))
        self.views = resolve_views(game, views)
        self._cams: Dict = {}
        super().__init__(game, seed)

    @property
    def multiview(self) -> bool:
        return len(self.views) > 1

    def _shot(self, name: str) -> np.ndarray:
        spec = VIEWS[self.game][name]
        if spec is None:
            return self.env.render()
        import mujoco

        cam = self._cams.get(name)
        if cam is None:
            cam = self._cams[name] = mujoco.MjvCamera()
            if "trackbodyid" in spec:
                cam.type, cam.trackbodyid = mujoco.mjtCamera.mjCAMERA_TRACKING, spec["trackbodyid"]
            else:
                cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                cam.lookat[:] = spec["lookat"]
            cam.distance, cam.azimuth, cam.elevation = spec["distance"], spec["azimuth"], spec["elevation"]
        viewer = self.env.unwrapped.mujoco_renderer._get_viewer("rgb_array")
        saved, viewer.cam = viewer.cam, cam
        try:
            return viewer.render(render_mode="rgb_array")  # no camera_id: the swapped-in camera is used as set
        finally:
            viewer.cam = saved

    def frame(self) -> np.ndarray:
        """The current RGB frame (rendered once per step): ``[H, W, 3]`` for one view, ``[views, H, W, 3]`` for
        several."""
        if self._frame is None:
            shots = [self._shot(v) for v in self.views]
            self._frame = shots[0] if len(shots) == 1 else np.stack(shots)
        return self._frame

    def render(self):
        """The current frame with the previous one ghosted underneath: a PIL image, or a list, one per view."""
        if not self.multiview:
            return super().render()
        from PIL import Image

        cur = self.frame()
        if self._prev is not None:
            mix = (1 - GHOST) * cur.astype(np.float32) + GHOST * self._prev.astype(np.float32)
            cur = mix.round().astype(np.uint8)
        return [Image.fromarray(v) for v in cur]

    def state(self) -> Dict:
        """What the model is shown: ``{"image": ...}``, or ``{"images": [...]}`` with several views."""
        return {"images": self.render()} if self.multiview else {"image": self.render()}

    def questions(self) -> Dict[str, Dict]:
        """``questions(game, views)`` for this episode's views."""
        return questions(self.game, self.views)

    def _make(self, spec: Dict):
        if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
            os.environ.setdefault("MUJOCO_GL", "egl")  # read when mujoco is first imported
        try:  # triton's bundled LLVM must load before OSMesa's: the other order segfaults (e.g. loading an expert)
            import torch._dynamo  # noqa: F401
        except ImportError:
            pass
        import gymnasium as gym

        kw = {"default_camera_config": spec["camera"]} if spec["camera"] else {}
        env = gym.make(spec["env_id"], render_mode="rgb_array", width=SIZE, height=SIZE, **kw)
        self.joints = spec.get("joints")
        self._high = env.action_space.high.astype(np.float64)
        if self.joints is None:
            self._pushes = np.asarray(spec["pushes"], dtype=np.float64) * self._high
        return env

    def _env_action(self, action):
        if self.joints is None:
            return self._pushes[self.actions.index(action)].astype(np.float32)
        levels = np.array([LEVEL_VALUES[LEVELS.index(action[j])] for j in self.joints])
        return (levels * self._high).astype(np.float32)

    def expert_action(self) -> np.ndarray:
        """The torque game's Hub expert action for the current observation, in units of the actuator range."""
        model = hub_expert(self.game)
        obs = np.asarray(self.obs, dtype=np.float32)[: model.observation_space.shape[0]]  # HumanoidStandup's sees 45
        a, _ = model.predict(obs, deterministic=True)
        return np.clip(np.asarray(a, dtype=np.float64) / self._high, -1.0, 1.0)

    def expert(self):
        kind = self.SPECS[self.game]["expert"][0]
        obs = self.obs
        if kind == "hub":
            a = self.expert_action()
            return {j: LEVELS[int(np.argmin(np.abs(LEVEL_VALUES - a[k])))] for k, j in enumerate(self.joints)}
        if kind == "pendulum":
            x, theta, x_dot, theta_dot = obs
            u = 0.3 * x + 5.0 * theta + 0.5 * x_dot + theta_dot
            i = 2 if u > 0.05 else 0 if u < -0.05 else 1
        elif kind == "double":
            x, s1, s2, c1, c2, x_dot, w1, w2 = obs[:8]
            state = np.array([x, np.arctan2(s1, c1), np.arctan2(s2, c2), x_dot, w1, w2])
            i = _nearest(self.SPECS[self.game]["pushes"], -float(DOUBLE_K @ state))
        else:
            raise ValueError("%s has no expert" % self.game)
        return self.actions[i]


def expert_policy(env: MujocoGame):
    return env.expert()


def random_policy(seed: int = 0):
    """A uniformly random action: one of the pushes, or a random level on every joint."""
    rng = random.Random(seed)

    def pick(env):
        if env.joints is None:
            return rng.choice(env.actions)
        return {j: rng.choice(LEVELS) for j in env.joints}
    return pick


def still_policy(env):
    """``NONE`` everywhere: the baseline of doing nothing, which a robot that only has to not fall can beat random
    with."""
    return "NONE" if env.joints is None else {j: "NONE" for j in env.joints}


_FAINT = " A faint copy"


def questions(game: str, views=None) -> Dict[str, Dict]:
    """What the model is asked each step: ``{"action": ...}`` for a pendulum, one question per joint otherwise.
    With several ``views`` each question also names the cameras, in the order the images come; one view (the
    default) leaves the text as it always was."""
    from laya.games import control_question, mujoco_questions

    qs = mujoco_questions(game) if is_torque_game(game) else control_question(game)
    views = resolve_views(game, views)
    if len(views) > 1:
        note = " You see the %s from %d cameras, one image each, in this order: %s." % (
            "robot" if is_torque_game(game) else "scene", len(views), ", ".join(v.replace("_", "-") for v in views))
        for q in qs.values():
            assert _FAINT in q["instructions"]
            q["instructions"] = q["instructions"].replace(_FAINT, note + _FAINT, 1)
    return qs


def model_chooser(agent, game: str):
    """``choose(env) -> (action, probabilities)`` from the model's answers on the ghosted screen (every view of a
    multi-view episode, ``MujocoGame.state``). For a torque game the action is ``{joint: level}`` and the
    probabilities ``{joint: {level: p}}``."""
    cache: Dict[tuple, Dict] = {}

    def choose(env):
        qs = cache.get(env.views)
        if qs is None:
            qs = cache[env.views] = questions(game, env.views)
        ans = agent.predict(env.state(), qs)["answers"]
        if env.joints is None:
            return ans["action"]["choice"], ans["action"].get("probabilities")
        return {j: ans[j]["choice"] for j in env.joints}, {j: ans[j].get("probabilities") for j in env.joints}
    return choose


def model_policy(agent, game: str):
    """The model's most likely action on the rendered (ghosted) screen."""
    choose = model_chooser(agent, game)
    return lambda env: choose(env)[0]


def has_expert(game: str) -> bool:
    return GAMES[game]["expert"] is not None


def play_episodes(game: str, policy, episodes: int, seed: int = 0, max_steps: int = 0) -> Dict:
    """``laya.controlgames.play_episodes`` over ``MujocoGame``."""
    return _play_episodes(game, policy, episodes, seed, max_steps, make=MujocoGame)


def soft_levels(a: float) -> List[float]:
    """A distribution over ``LEVELS`` for a continuous action ``a`` (share of the range): its weight split between the
    two nearest levels, in proportion to closeness, so 0.3 gives NONE 0.4 and POS 0.6."""
    a = float(np.clip(a, -1.0, 1.0))
    k = min(int(np.searchsorted(LEVEL_VALUES, a, side="right")) - 1, len(LEVELS) - 2)
    w = (a - LEVEL_VALUES[k]) / (LEVEL_VALUES[k + 1] - LEVEL_VALUES[k])
    out = [0.0] * len(LEVELS)
    out[k], out[k + 1] = round(float(1.0 - w), 4), round(float(w), 4)
    return out


def jitter(env: MujocoGame, action, rng: random.Random, p: float):
    """``action`` with each joint's level (or the push) moved one notch up or down with probability ``p``."""
    def nudge(names, name):
        i = names.index(name)
        if rng.random() < p:
            i = min(max(i + rng.choice((-1, 1)), 0), len(names) - 1)
        return names[i]
    if env.joints is None:
        return nudge(env.actions, action)
    return {j: nudge(LEVELS, v) for j, v in action.items()}


def expert_frames(game: str, n: int, seed: int = 0, noise: float = 0.1, stride: int = 4, smooth: float = 0.1,
                  views=None):
    """Training frames labelled by the expert: yields ``{"episode", "step", "image", "records"}`` until ``n`` frames
    (``"images"``, one per view, in place of ``"image"`` when several ``views`` are set; the questions then name the
    cameras).

    Episodes run from seeds ``seed, seed + 1, ...``. The behaviour policy is the expert with each joint's level
    moved one notch with probability ``noise`` (``jitter``; DART-style), so the frames include states the expert has
    to recover from; the labels are always the noise-free expert's action in the state actually reached. Uniformly
    random actions are too violent for this: held for a few steps even 5% of the time, they topple the walkers and
    pendulums within 100-250 steps, while 10% jitter leaves episodes of hundreds of steps.
    Every ``stride``-th step is kept, rendered exactly as in play (the step before is rendered too, so the ghosted
    previous frame matches). ``records`` holds one ``{"key", "question", "label", "target"}`` per question asked
    about the frame: one per joint for a torque game, whose ``target`` is ``soft_levels`` of the expert's continuous
    action and ``label`` its argmax; the pendulum's one question has the expert's push, smoothed by ``smooth``.
    """
    rng = random.Random(seed)
    qs = questions(game, views)
    made, ep = 0, 0
    while made < n:
        env = MujocoGame(game, seed + ep, views)
        while not env.done and made < n:
            keep = env.steps % stride == 0
            if (env.steps + 1) % stride == 0:
                env.frame()  # the next kept frame ghosts this one
            if keep:
                if env.joints is None:
                    act = env.expert()
                    k = len(env.actions)
                    target = [(1 - smooth) * (a == act) + smooth / k for a in env.actions]
                    records = [{"key": "action", "question": qs["action"], "label": env.actions.index(act),
                                "target": target}]
                else:
                    cont = env.expert_action()
                    records = []
                    for i, j in enumerate(env.joints):
                        target = [float(t) for t in soft_levels(cont[i])]
                        records.append({"key": j, "question": qs[j], "label": int(np.argmax(target)),
                                        "target": target})
                    act = {j: LEVELS[r["label"]] for j, r in zip(env.joints, records)}
                yield {"episode": seed + ep, "step": env.steps, "images" if env.multiview else "image": env.render(),
                       "records": records}
                made += 1
            else:
                act = env.expert()
            env.step(jitter(env, act, rng, noise))
        env.close()
        ep += 1


PANEL_W = 300
_CHOSEN, _OTHER = (255, 200, 80), (90, 90, 110)


def tile(frames, names=None):
    """Several views in one image: a grid ``ceil(sqrt(n))`` wide, each view captioned with its name."""
    from PIL import Image, ImageDraw

    cols = int(np.ceil(np.sqrt(len(frames))))
    rows = int(np.ceil(len(frames) / cols))
    fw, fh = frames[0].width, frames[0].height
    out = Image.new("RGB", (cols * fw, rows * fh), (24, 24, 28))
    d = ImageDraw.Draw(out)
    for i, f in enumerate(frames):
        x, y = (i % cols) * fw, (i // cols) * fh
        out.paste(f, (x, y))
        if names:
            d.rectangle((x, y, x + 8 + 6 * len(names[i]), y + 14), fill=(0, 0, 0))
            d.text((x + 4, y + 2), names[i], fill=(255, 255, 255))
    return out


def draw(frame, env: MujocoGame, action, probs, label: str) -> np.ndarray:
    """A video frame: the screen at 2x (several views tiled at 1x, ``tile``), and a panel with the step, the return
    and the action: each push (the chosen one highlighted, with the model's probabilities when given), or for a
    torque game one row per joint with its five levels, shaded by probability, the chosen one outlined."""
    from PIL import Image, ImageDraw

    if isinstance(frame, (list, tuple)):
        screen = tile(frame, list(env.views))
    else:
        screen = frame.resize((frame.width * 2, frame.height * 2), Image.NEAREST)
    w, h = screen.width, max(screen.height, 2 * SIZE)
    img = Image.new("RGB", (w + PANEL_W, h), (24, 24, 28))
    img.paste(screen, (0, 0))
    d = ImageDraw.Draw(img)
    x = w + 12
    d.text((x, 12), "%s  (%s)" % (env.game, label), fill=(230, 230, 230))
    d.text((x, 30), "step %d   return %.1f" % (env.steps, env.score), fill=(170, 170, 170))
    if env.joints is None:
        row = min(26, (h - 60) // len(env.actions))
        for i, a in enumerate(env.actions):
            y = 56 + row * i
            chosen = a == action
            d.text((x, y), a, fill=_CHOSEN if chosen else (200, 200, 200))
            p = probs.get(a, 0.0) if probs else (1.0 if chosen else 0.0)
            d.rectangle((x + 160, y + 1, x + 160 + max(1, int(120 * p)), y + row - 3),
                        fill=_CHOSEN if chosen else _OTHER)
        return np.asarray(img)
    d.text((x + 150, 44), "-  -    0    +  +", fill=(150, 150, 150))
    row = min(26, (h - 70) // len(env.joints))  # 17 joints for Humanoid
    for i, j in enumerate(env.joints):
        y = 62 + row * i
        d.text((x, y), j, fill=(200, 200, 200))
        jp = (probs or {}).get(j) or {}
        for k, lv in enumerate(LEVELS):
            p = jp.get(lv, 1.0 if action and action.get(j) == lv else 0.0)
            box = (x + 150 + 24 * k, y + 1, x + 170 + 24 * k, y + row - 3)
            shade = tuple(int(o + (c - o) * p) for o, c in zip((40, 40, 48), _CHOSEN))
            d.rectangle(box, fill=shade, outline=(255, 255, 255) if action and action.get(j) == lv else None)
    return np.asarray(img)


def record(env: MujocoGame, choose, label: str, out_path: str, max_steps: int = 0) -> None:
    """Play ``env`` to its end (or ``max_steps``) with ``choose(env) -> (action, probabilities or None)``, writing
    each screen the policy saw to ``out_path`` (.webm: VP9, else H.264) in real time. Needs ``imageio[ffmpeg]``."""
    import imageio.v2 as imageio

    codec = {"codec": "libvpx-vp9", "ffmpeg_params": ["-b:v", "0", "-crf", "32"]} if out_path.endswith(".webm") else {}
    fps = round(1 / env.env.unwrapped.dt)
    with imageio.get_writer(out_path, fps=fps, macro_block_size=1, **codec) as out:
        while not env.done and not (max_steps and env.steps >= max_steps):
            frame = env.render()  # the ghosted screen, as the model sees it
            action, probs = choose(env)
            out.append_data(draw(frame, env, action, probs, label))
            env.step(action)
        out.append_data(draw(env.render(), env, "", None, label + (" - ended" if env.terminated else "")))


def baseline(agent, game: str, video: str, episodes: int = 3, baseline_episodes: int = 10, seed: int = 200_000,
             max_steps: int = 300) -> Dict:
    """A model on ``game``: its first episode recorded to ``video``, its mean return over ``episodes`` seeded
    episodes (seeds ``seed + i``), and random, still and (where there is one) expert play over ``baseline_episodes``
    episodes from the same seed. Every episode stops at ``max_steps`` (0 = the environment's limit)."""
    import time

    t0 = time.time()
    choose = model_chooser(agent, game)
    env = MujocoGame(game, seed)
    record(env, choose, "model", video, max_steps)
    model = [{"score": round(env.score, 3), "steps": env.steps, "terminated": env.terminated}]
    env.close()
    if episodes > 1:
        model += play_episodes(game, lambda e: choose(e)[0], episodes - 1, seed + 1, max_steps)["results"]
    rnd = play_episodes(game, random_policy(seed), baseline_episodes, seed, max_steps)
    still = play_episodes(game, still_policy, baseline_episodes, seed, max_steps)
    exp = play_episodes(game, expert_policy, baseline_episodes, seed, max_steps) if has_expert(game) else None
    m = float(np.mean([e["score"] for e in model]))
    return {"game": game, "model_score": round(m, 3), "model_scores": [e["score"] for e in model],
            "model_steps": [e["steps"] for e in model], "random_score": round(rnd["mean_score"], 3),
            "still_score": round(still["mean_score"], 3),
            "expert_score": None if exp is None else round(exp["mean_score"], 3),
            "normalized": None if exp is None else normalized(m, rnd["mean_score"], exp["mean_score"]),
            "episodes": episodes, "baseline_episodes": baseline_episodes, "seed": seed, "max_steps": max_steps,
            "seconds": round(time.time() - t0, 1)}


def baseline_table(rows) -> str:
    lines = ["%-22s %10s %10s %10s %10s %7s" % ("game", "model", "random", "still", "expert", "norm")]
    for r in rows:
        ex = "-" if r["expert_score"] is None else "%.1f" % r["expert_score"]
        nm = "-" if r["normalized"] is None else "%.2f" % r["normalized"]
        lines.append("%-22s %10.1f %10.1f %10.1f %10s %7s" % (r["game"], r["model_score"], r["random_score"],
                                                            r["still_score"], ex, nm))
    return "\n".join(lines)


__all__ = ["GAMES", "LEVELS", "HUB_EXPERTS", "MujocoGame", "play_episodes", "normalized", "expert_policy",
           "random_policy", "still_policy", "has_expert", "is_torque_game", "questions", "model_policy", "model_chooser",
           "record", "baseline", "baseline_table", "soft_levels", "jitter", "expert_frames", "VIEWS", "resolve_views",
           "tile"]
