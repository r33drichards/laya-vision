"""BiGym: bimanual mobile manipulation in MuJoCo, probed and played zero-shot from the head camera.

BiGym (https://github.com/chernyadev/bigym) drives a Unitree H1 with a 15-dim continuous joint action (floating
base X, Y, yaw; five joints per arm; two grippers) at 50 Hz. Laya answers discrete questions, so this module
bridges the two in two ways:

- **Control.** Each decision the model picks one of ``PRIMITIVES``, a named motion: move a wrist 3 cm along a
  robot-frame axis, tilt or roll a gripper, open or close it, step, sidestep, turn, crouch or stand, or stay.
  ``primitive_to_action`` turns a wrist move into joint deltas with damped least-squares IK on the wrist site; the
  action is applied once and the robot is left ``HOLD`` env steps to settle. A random policy and a privileged
  oracle play the same seeded episodes over the same primitives. The oracle greedily moves the relevant wrist
  toward the target on the reach tasks, and on the cupboard tasks it is replaced by the success rate of BiGym's
  human demonstrations replayed in the sim, which is reported as a reference only.
- **Probe.** Frames whose ground truth the simulator knows (task success, how far the drawer or door is open or
  how far the wrist is from the target, which wrist is closer) are put to the model as ``noul``, ``score`` and
  ``choice`` questions (``probe_questions``) and scored for accuracy and calibration.

Needs ``bigym`` (``pip install git+https://github.com/chernyadev/bigym``) and a headless GL (``MUJOCO_GL=egl`` or
``osmesa``); imported lazily so the rest of ``laya`` does not depend on it.
"""
import random
from typing import Dict, List, Optional

import numpy as np

HOLD = 5  # env steps (at 50 Hz) each primitive is held for: 0.1 s
CONTROL_FREQUENCY = 50
RESOLUTION = (256, 256)
WRIST_STEP = 0.03  # metres a wrist primitive moves the wrist
BASE_STEP = 0.05  # metres a base primitive moves the pelvis
CROUCH_STEP = 0.03  # metres a crouch / stand primitive lowers or raises the pelvis (the demos crouch ~0.11 m)
TURN_STEP = 0.2  # radians a turn primitive rotates the pelvis
WRIST_ROLL = 0.25  # radians a wrist-roll primitive turns the wrist joint (about 14 degrees; its range is +-90)
TILT_STEP = 0.26  # radians a tilt primitive turns the gripper's pointing direction (15 degrees)
IK_DAMPING = 0.05
TILT_HOLD = 0.3  # how hard a tilt holds the hand in place (a position row weight; the direction rows weigh 1)
MAX_FRAMES = 4  # decision frames kept for multi-frame questions
REACH_BINS = (0.3, 0.2, 0.1)  # wrist-target distance (m) thresholds for progress levels 1, 2, 3
OPEN_BINS = (0.3, 0.6, 0.9)  # task-direction open fraction thresholds for progress levels 1, 2, 3
SEED = 300_000  # eval episodes use seeds SEED + i
# BiGym's H1 head camera (``h1/head``: position in the head body, wxyz quaternion). The wall cabinet sits above
# the default view, so the wall tasks tilt it up by ``camera_tilt`` degrees about the camera's own x axis.
HEAD_POS = (0.12, 0.0, 0.69)
HEAD_QUAT = (0.68361293, 0.18171101, -0.18185577, -0.68306877)

_REACH = "reach"
_CUPBOARD = "cupboard"
TASKS = {
    "ReachTarget": {"module": "reach_target", "kind": _REACH, "hands": ("left", "right"), "max_decisions": 60,
                    "description": "touch the red ball with either hand"},
    "ReachTargetSingle": {"module": "reach_target", "kind": _REACH, "hands": ("left",), "max_decisions": 60,
                          "description": "touch the red ball with the left hand"},
    "DrawerTopOpen": {"module": "cupboards", "kind": _CUPBOARD, "part": "drawer", "goal": 1, "max_decisions": 150,
                      "description": "open the top drawer of the kitchen cabinet"},
    "DrawerTopClose": {"module": "cupboards", "kind": _CUPBOARD, "part": "drawer", "goal": 0, "max_decisions": 150,
                       "description": "close the top drawer of the kitchen cabinet"},
    "WallCupboardOpen": {"module": "cupboards", "kind": _CUPBOARD, "part": "wall", "goal": 1, "max_decisions": 150,
                         "camera_tilt": 30, "description": "open both doors of the wall cabinet"},
    "WallCupboardClose": {"module": "cupboards", "kind": _CUPBOARD, "part": "wall", "goal": 0, "max_decisions": 150,
                          "camera_tilt": 30, "description": "close both doors of the wall cabinet"},
}

# name -> description shown to the model; robot-frame axes: forward is where the robot faces, left is its left
_AXES = {"FORWARD": (1, 0, 0), "BACK": (-1, 0, 0), "LEFT": (0, 1, 0), "RIGHT": (0, -1, 0), "UP": (0, 0, 1),
         "DOWN": (0, 0, -1)}
_AXIS_WORDS = {"FORWARD": "forward, away from the robot", "BACK": "back, toward the robot", "LEFT": "to the left",
               "RIGHT": "to the right", "UP": "up", "DOWN": "down"}
PRIMITIVES: Dict[str, str] = {}
for _hand in ("LEFT", "RIGHT"):
    for _axis in _AXES:
        PRIMITIVES["%s_HAND_%s" % (_hand, _axis)] = "move the %s hand %s" % (_hand.lower(), _AXIS_WORDS[_axis])
for _hand in ("LEFT", "RIGHT"):
    PRIMITIVES["%s_GRIPPER_CLOSE" % _hand] = "close the %s gripper" % _hand.lower()
    PRIMITIVES["%s_GRIPPER_OPEN" % _hand] = "open the %s gripper" % _hand.lower()
# the wrist joint rolls the gripper about the forearm; it does not move the wrist point, so the hand moves never
# turn it. Positive rotation is clockwise as the head camera sees it (right-hand rule about the forward axis).
for _hand in ("LEFT", "RIGHT"):
    PRIMITIVES["%s_WRIST_CW" % _hand] = "roll the %s wrist clockwise" % _hand.lower()
    PRIMITIVES["%s_WRIST_CCW" % _hand] = "roll the %s wrist counterclockwise" % _hand.lower()
# tilts aim the gripper (its pointing direction) 15 degrees up, down, left or right. H1's arms have no wrist pitch
# or yaw joint (three shoulder joints, the elbow and the wrist roll), so with the hand held still the direction has
# one degree of freedom left: a tilt turns the direction first and lets the hand shift a few cm. Hand moves leave
# the direction free (holding it cost the reach oracle 40 points). The wrist roll turns the gripper about it.
for _hand in ("LEFT", "RIGHT"):
    for _dir in ("UP", "DOWN", "LEFT", "RIGHT"):
        PRIMITIVES["%s_TILT_%s" % (_hand, _dir)] = "tilt the %s gripper to point %s" % (_hand.lower(), _dir.lower())
PRIMITIVES.update({"BASE_FORWARD": "step the whole robot forward", "BASE_BACK": "step the whole robot back",
                   "BASE_TURN_LEFT": "turn the whole robot to the left",
                   "BASE_TURN_RIGHT": "turn the whole robot to the right",
                   "BASE_LEFT": "sidestep the whole robot to the left",
                   "BASE_RIGHT": "sidestep the whole robot to the right",
                   "BASE_DOWN": "crouch: lower the whole robot", "BASE_UP": "stand up: raise the whole robot",
                   "STAY": "do nothing and wait"})

# The option text the model reads: 37 options share the head budget (``head_max_len``, 256 tokens) with the
# instructions, so "NAME: description" (about 20 tokens each) would be cut and crowd the task out of the prompt.
# Short phrases fit whole; ``model_policy`` maps the chosen phrase back to the primitive's name.
OPTION_WORDS: Dict[str, str] = {}
for _hand in ("LEFT", "RIGHT"):
    for _axis in _AXES:
        OPTION_WORDS["%s_HAND_%s" % (_hand, _axis)] = "%s hand %s" % (_hand.lower(), _axis.lower())
for _hand in ("LEFT", "RIGHT"):
    OPTION_WORDS["%s_GRIPPER_CLOSE" % _hand] = "close %s gripper" % _hand.lower()
    OPTION_WORDS["%s_GRIPPER_OPEN" % _hand] = "open %s gripper" % _hand.lower()
for _hand in ("LEFT", "RIGHT"):
    OPTION_WORDS["%s_WRIST_CW" % _hand] = "roll %s wrist clockwise" % _hand.lower()
    OPTION_WORDS["%s_WRIST_CCW" % _hand] = "roll %s wrist counterclockwise" % _hand.lower()
for _hand in ("LEFT", "RIGHT"):
    for _dir in ("UP", "DOWN", "LEFT", "RIGHT"):
        OPTION_WORDS["%s_TILT_%s" % (_hand, _dir)] = "tilt %s gripper %s" % (_hand.lower(), _dir.lower())
OPTION_WORDS.update({"BASE_FORWARD": "step forward", "BASE_BACK": "step back", "BASE_TURN_LEFT": "turn left",
                     "BASE_TURN_RIGHT": "turn right", "BASE_LEFT": "sidestep left", "BASE_RIGHT": "sidestep right",
                     "BASE_DOWN": "crouch", "BASE_UP": "stand up",
                     "STAY": "wait"})
FROM_WORDS = {v: k for k, v in OPTION_WORDS.items()}

PROGRESS_LEVELS = ["not started: far from done", "partly done", "mostly done: nearly there", "done"]


# ---------------------------------------------------------------------------------------------------------
# Questions and labels (pure Python: no bigym needed)
# ---------------------------------------------------------------------------------------------------------

def bigym_question(task: str, frames: int = 1) -> Dict:
    """The control question for a ``TASKS`` entry: one ``choice`` over the ``OPTION_WORDS`` phrases (one per
    primitive, in ``PRIMITIVES`` order). With ``frames`` > 1 the state is that many head-camera frames, one per
    decision (0.1 s apart), oldest first. Sized to fit ``head_max_len`` whole (``tests/test_bigymgames.py``)."""
    if frames > 1:
        view = "Robot head camera, last %d views 0.1 s apart, oldest first; grippers at the bottom." % frames
    else:
        view = "Robot head camera view; grippers at the bottom."
    return {"action": {
        "type": "choice",
        "instructions": "%s Task: %s. Next move?" % (view, TASKS[task]["description"]),
        "criteria": {OPTION_WORDS[p]: "" for p in PRIMITIVES},
    }}


def probe_questions(task: str) -> Dict:
    """The perception probe for a task: questions whose answers the simulator knows (``labels``)."""
    desc = TASKS[task]["description"]
    qs = {
        "done": {"type": "noul", "instructions": "The image is a robot's head camera view. The robot's task is to "
                                                 "%s. Is the task already complete in this image?" % desc},
        "progress": {"type": "score", "instructions": "The image is a robot's head camera view. The robot's task "
                                                      "is to %s. How far along is the task?" % desc,
                     "criteria": list(PROGRESS_LEVELS)},
    }
    if TASKS[task]["kind"] == _REACH:
        qs["side"] = {"type": "choice", "instructions": "The image is a robot's head camera view; its left gripper "
                                                        "is on the left of the image. Which of the robot's hands is "
                                                        "closer to the red ball?",
                      "criteria": {"left": "the left hand", "right": "the right hand"}}
    return qs


def progress_level(task: str, value: float) -> int:
    """Ground-truth progress level 0..3 from ``value``: the reach distance (m) or the open fraction in [0, 1]."""
    spec = TASKS[task]
    if spec["kind"] == _REACH:
        return int(sum(value < b for b in REACH_BINS))
    frac = value if spec["goal"] == 1 else 1.0 - value
    return int(sum(frac >= b for b in OPEN_BINS))


def labels(task: str, truth: Dict) -> Dict:
    """Question id -> the index of the correct answer (``noul``: 1 = true) from ``BiGymGame.ground_truth()``."""
    out = {"done": int(bool(truth["success"])),
           "progress": progress_level(task, truth["distance"] if TASKS[task]["kind"] == _REACH else truth["open"])}
    if "closer" in truth:
        out["side"] = ["left", "right"].index(truth["closer"])
    return out


def normalized(model: float, rnd: float, expert: float) -> Optional[float]:
    """(model - random) / (expert - random); ``None`` when the baselines tie."""
    return None if expert == rnd else (model - rnd) / (expert - rnd)


# ---------------------------------------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------------------------------------

def make_env(task: str, cameras: bool = True):
    """A BiGym env for ``task``: delta joint positions with a floating base at 50 Hz, the head camera (tilted up
    for the wall-cabinet tasks) and the privileged target position."""
    import importlib

    from bigym.action_modes import JointPositionActionMode, PelvisDof
    from bigym.utils.observation_config import CameraConfig, ObservationConfig

    cls = getattr(importlib.import_module("bigym.envs." + TASKS[task]["module"]), task)
    view = {}
    if TASKS[task].get("camera_tilt"):
        import mujoco

        a, q = np.deg2rad(TASKS[task]["camera_tilt"]), np.zeros(4)
        mujoco.mju_mulQuat(q, np.array(HEAD_QUAT), np.array([np.cos(a / 2), np.sin(a / 2), 0.0, 0.0]))
        view = {"pos": HEAD_POS, "quat": tuple(float(v) for v in q)}
    cams = [CameraConfig("head", rgb=True, depth=False, resolution=RESOLUTION, **view)] if cameras else []
    # the floating base moves in X, Y, height and yaw (the DOFs BiGym's demos were recorded with)
    dofs = [PelvisDof.X, PelvisDof.Y, PelvisDof.Z, PelvisDof.RZ]
    return cls(action_mode=JointPositionActionMode(absolute=False, floating_base=True, floating_dofs=dofs),
               observation_config=ObservationConfig(cameras=cams, proprioception=True, privileged_information=True),
               control_frequency=CONTROL_FREQUENCY)


class BiGymGame:
    """One seeded BiGym episode, stepped by primitive name."""

    def __init__(self, task: str, seed: int = 0, env=None):
        if task not in TASKS:
            raise ValueError("unknown BiGym task %r (%s)" % (task, ", ".join(TASKS)))
        import mujoco
        from bigym.const import HandSide

        self.task, self.spec, self.actions = task, TASKS[task], tuple(PRIMITIVES)
        self.env = env or make_env(task)
        self.obs, _ = self.env.reset(seed=seed)
        self._mj = mujoco
        robot, physics = self.env.robot, self.env.mojo.physics
        self._model, self._data = self.env.mojo.model, self.env.mojo.data
        self._sides = {"left": HandSide.LEFT, "right": HandSide.RIGHT}
        self._site = {h: physics.bind(robot._wrist_sites[s].mjcf).element_id for h, s in self._sides.items()}
        self._pelvis = physics.bind(robot.pelvis.mjcf).element_id
        # limb actuators are left then right, five each; their joints' dof addresses give the Jacobian columns
        acts = [physics.bind(a).element_id for a in robot.limb_actuators]
        dofs = [int(self._model.jnt_dofadr[self._model.actuator_trnid[a][0]]) for a in acts]
        half = len(acts) // 2
        self._arm = {"left": (slice(0, half), dofs[:half]), "right": (slice(half, 2 * half), dofs[half:])}
        self._acts = acts
        self._n_base = robot.floating_base.dof_amount
        self._grip = np.zeros(len(robot.grippers), np.float32)  # gripper commands are absolute: 0 open, 1 closed
        self.steps, self.decisions = 0, 0
        self.success, self.terminated, self.truncated = False, False, False
        self._frame = self._prev = None
        # one head frame per decision, newest last (at most MAX_FRAMES); empty for a camera-less env (make_env
        # cameras=False), which is enough for the oracle, the random policy and the demo follower
        self._history = [self.frame()] if "rgb_head" in self.obs else []

    @property
    def done(self) -> bool:
        return self.success or self.terminated or self.truncated

    # -- geometry --------------------------------------------------------------------------------------------
    def hand_pos(self, hand: str) -> np.ndarray:
        return np.array(self._data.site_xpos[self._site[hand]])

    def _yaw_rot(self) -> np.ndarray:
        """Robot-frame (forward, left, up) axes in world coordinates, from the pelvis heading."""
        fwd = np.array(self._data.xmat[self._pelvis]).reshape(3, 3)[:, 0].copy()
        fwd[2] = 0
        fwd /= max(1e-9, np.linalg.norm(fwd))
        left = np.array([-fwd[1], fwd[0], 0.0])
        return np.stack([fwd, left, np.array([0.0, 0.0, 1.0])], axis=1)

    def world_dir(self, axis: str) -> np.ndarray:
        return self._yaw_rot() @ np.array(_AXES[axis], dtype=np.float64)

    def target(self) -> Optional[np.ndarray]:
        t = self.obs.get("target_position")
        return None if t is None else np.array(t, np.float64)

    def _jac(self, hand: str, rot: bool = False):
        jacp, jacr = np.zeros((3, self._model.nv)), np.zeros((3, self._model.nv))
        self._mj.mj_jacSite(self._model, self._data, jacp, jacr, self._site[hand])
        cols = self._arm[hand][1]
        return (jacp[:, cols], jacr[:, cols]) if rot else jacp[:, cols]

    def pointing(self, hand: str) -> np.ndarray:
        """The direction ``hand``'s gripper points (the wrist site's x axis, world frame, unit length)."""
        return np.array(self._data.site_xmat[self._site[hand]]).reshape(3, 3)[:, 0]

    def ik_delta(self, hand: str, dx: np.ndarray, omega: Optional[np.ndarray] = None) -> np.ndarray:
        """Joint deltas (the arm's five joints) for ``hand``, clipped so the joint targets stay inside their control
        ranges. With ``omega`` None: move the wrist by ``dx`` (world, metres), damped least squares on position
        only. With a rotation vector ``omega`` (world, radians): turn the gripper's pointing direction by it, over
        the four joints before the wrist roll (the roll does not change the direction), holding the wrist at
        ``dx`` with weight ``TILT_HOLD``."""
        if omega is None:
            J = self._jac(hand)
            dq = J.T @ np.linalg.solve(J @ J.T + IK_DAMPING ** 2 * np.eye(3), dx)
        else:
            Jp, Jr = self._jac(hand, rot=True)
            x = self.pointing(hand)
            P = np.eye(3) - np.outer(x, x)  # rotation about the pointing axis itself is the wrist roll's job
            A = np.vstack([TILT_HOLD * Jp[:, :4], P @ Jr[:, :4]])
            b = np.concatenate([TILT_HOLD * dx, P @ omega])
            dq = np.zeros(Jp.shape[1])
            dq[:4] = np.linalg.solve(A.T @ A + IK_DAMPING ** 2 * np.eye(4), A.T @ b)
        acts = np.array(self._acts)[self._arm[hand][0]]
        ranges, ctrl = self._model.actuator_ctrlrange[acts], np.array(self._data.ctrl[acts])
        return np.clip(ctrl + dq, ranges[:, 0], ranges[:, 1]) - ctrl

    def predicted_move(self, name: str, hand: str) -> np.ndarray:
        """First-order estimate of how primitive ``name`` moves ``hand``'s wrist (world, metres)."""
        parts = name.split("_") + ["", ""]
        if parts[1] == "HAND":
            if parts[0].lower() != hand:
                return np.zeros(3)
            return self._jac(hand) @ self.ik_delta(hand, WRIST_STEP * self.world_dir(parts[2]))
        if parts[1] == "TILT":
            if parts[0].lower() != hand:
                return np.zeros(3)
            return self._jac(hand) @ self.ik_delta(hand, np.zeros(3), self.tilt_vector(hand, parts[2]))
        if name in ("BASE_FORWARD", "BASE_BACK"):
            return self.world_dir("FORWARD") * (BASE_STEP if name == "BASE_FORWARD" else -BASE_STEP)
        if name in ("BASE_LEFT", "BASE_RIGHT"):
            return self.world_dir("LEFT") * (BASE_STEP if name == "BASE_LEFT" else -BASE_STEP)
        if name in ("BASE_UP", "BASE_DOWN"):
            return np.array([0.0, 0.0, CROUCH_STEP if name == "BASE_UP" else -CROUCH_STEP])
        return np.zeros(3)

    def tilt_vector(self, hand: str, direction: str) -> np.ndarray:
        """Rotation vector that tilts ``hand``'s gripper ``TILT_STEP`` toward ``direction`` (UP, DOWN, LEFT, RIGHT):
        up/down about the horizontal axis across the pointing direction, left/right about the vertical."""
        x, up = self.pointing(hand), np.array([0.0, 0.0, 1.0])
        if direction in ("LEFT", "RIGHT"):
            return (TILT_STEP if direction == "LEFT" else -TILT_STEP) * up
        across = np.cross(x, up)
        n = np.linalg.norm(across)
        across = across / n if n > 1e-6 else self.world_dir("LEFT")  # pointing straight up or down
        return (TILT_STEP if direction == "UP" else -TILT_STEP) * across

    # -- actions ---------------------------------------------------------------------------------------------
    def primitive_to_action(self, name: str) -> np.ndarray:
        """The first env action of primitive ``name``, clipped to the action space (BiGym rejects actions outside
        it). Base and arm entries are deltas; the gripper entries are the held absolute commands."""
        space = self.env.action_space
        act = np.zeros(space.shape, np.float32)
        parts = name.split("_") + ["", ""]
        if parts[1] == "HAND":
            hand = parts[0].lower()
            idx = np.arange(self._n_base, self._n_base + len(self._acts))[self._arm[hand][0]]
            act[idx] = self.ik_delta(hand, WRIST_STEP * self.world_dir(parts[2]))
        elif parts[1] == "TILT":
            hand = parts[0].lower()
            idx = np.arange(self._n_base, self._n_base + len(self._acts))[self._arm[hand][0]]
            act[idx] = self.ik_delta(hand, np.zeros(3), self.tilt_vector(hand, parts[2]))
        elif parts[1] == "WRIST":
            hand = parts[0].lower()
            a = np.array(self._acts)[self._arm[hand][0]][-1]  # the arm's last actuator is its wrist
            lo, hi = self._model.actuator_ctrlrange[a]
            ctrl = float(self._data.ctrl[a])
            want = WRIST_ROLL if parts[2] == "CW" else -WRIST_ROLL
            act[self._n_base + list(self._acts).index(a)] = float(np.clip(ctrl + want, lo, hi) - ctrl)
        elif parts[1] == "GRIPPER":
            self._grip[0 if parts[0] == "LEFT" else 1] = 1.0 if parts[2] == "CLOSE" else 0.0
        elif name in ("BASE_FORWARD", "BASE_BACK"):
            d = self.world_dir("FORWARD") * (BASE_STEP if name == "BASE_FORWARD" else -BASE_STEP)
            act[0:2] = d[:2]
        elif name in ("BASE_LEFT", "BASE_RIGHT"):
            d = self.world_dir("LEFT") * (BASE_STEP if name == "BASE_LEFT" else -BASE_STEP)
            act[0:2] = d[:2]
        elif name in ("BASE_TURN_LEFT", "BASE_TURN_RIGHT"):
            act[self._n_base - 1] = TURN_STEP if name == "BASE_TURN_LEFT" else -TURN_STEP
        elif name in ("BASE_UP", "BASE_DOWN"):
            act[2] = CROUCH_STEP if name == "BASE_UP" else -CROUCH_STEP  # base action order: X, Y, Z, yaw
        elif name != "STAY":
            raise ValueError("unknown primitive %r" % name)
        act[-len(self._grip):] = self._grip
        return np.clip(act, space.low, space.high).astype(np.float32)

    def step(self, name: str) -> bool:
        """Play primitive ``name``: its action once, then ``HOLD - 1`` settle steps. Only the last step renders
        the observation (BiGym's ``fast`` steps skip it); success is checked after every step. Returns
        ``success``."""
        self._prev, self._frame = self._frame, None
        rest = np.zeros(self.env.action_space.shape, np.float32)
        rest[-len(self._grip):] = self._grip
        e = self.env
        for i in range(HOLD):
            e.step(self.primitive_to_action(name) if i == 0 else rest, fast=True)
            self.steps += 1
            self.success = self.success or bool(e.success)
            self.terminated, self.truncated = bool(e.terminate), bool(e.truncate)
            if self.done:
                break
        self.obs = e.get_observation()
        if "rgb_head" in self.obs:
            self._history = (self._history + [self.frame()])[-MAX_FRAMES:]
        self.decisions += 1
        return self.success

    # -- lookahead -------------------------------------------------------------------------------------------
    def snapshot(self):
        """Everything ``step`` changes: the MuJoCo state plus the Python-side floating-base and gripper commands and
        this game's counters. ``restore`` puts it back, so a planner can try a primitive and undo it."""
        m, d, mj = self._model, self._data, self._mj
        spec = mj.mjtState.mjSTATE_INTEGRATION
        state = np.empty(mj.mj_stateSize(m, spec))
        mj.mj_getState(m, d, state, spec)
        fb = self.env.robot.floating_base
        return (state, fb._accumulated_actions.copy(), fb._last_action.copy(), self._grip.copy(),
                (self.steps, self.decisions, self.success, self.terminated, self.truncated, self.obs,
                 self._frame, self._prev, list(self._history)))

    def restore(self, snap) -> None:
        m, d, mj = self._model, self._data, self._mj
        state, acc, last, grip, counters = snap
        mj.mj_setState(m, d, state, mj.mjtState.mjSTATE_INTEGRATION)
        mj.mj_forward(m, d)
        fb = self.env.robot.floating_base
        fb._accumulated_actions, fb._last_action, self._grip = acc.copy(), last.copy(), grip.copy()
        (self.steps, self.decisions, self.success, self.terminated, self.truncated, self.obs, self._frame,
         self._prev, history) = counters
        self._history = list(history)

    # -- observation -----------------------------------------------------------------------------------------
    def frame(self) -> np.ndarray:
        """The head camera's current RGB frame, HxWx3 uint8."""
        if self._frame is None:
            self._frame = np.ascontiguousarray(np.asarray(self.obs["rgb_head"]).transpose(1, 2, 0))
        return self._frame

    def frames(self, k: int) -> List[np.ndarray]:
        """The last ``k`` decision frames, oldest first; at the start of an episode the first frame is repeated
        so there are always ``k``."""
        if not 1 <= k <= MAX_FRAMES:
            raise ValueError("frames must be 1..%d, got %d" % (MAX_FRAMES, k))
        h = self._history[-k:]
        return [h[0]] * (k - len(h)) + h

    def render(self, ghost: float = 0.0):
        """The current head frame as a PIL image, with the previous decision's frame blended in at ``ghost``."""
        from PIL import Image

        cur = self.frame()
        if not ghost or self._prev is None:
            return Image.fromarray(cur)
        mix = (1 - ghost) * cur.astype(np.float32) + ghost * self._prev.astype(np.float32)
        return Image.fromarray(mix.round().astype(np.uint8))

    def open_fraction(self) -> float:
        """How far the task's drawer or doors are open, in [0, 1] (the mean over the wall cabinet's two doors)."""
        e = self.env
        if self.spec["part"] == "drawer":
            return float(e.cabinet_drawers.get_state()[-1])
        return float(np.mean(e.cabinet_wall.get_state()))

    def set_open_fraction(self, frac: float) -> None:
        """Pose the task's drawer or doors at ``frac`` open (probe frames), and refresh the observation."""
        e = self.env
        if self.spec["part"] == "drawer":
            e.cabinet_drawers.set_state(np.array([0, 0, frac]))
        else:
            e.cabinet_wall.set_state(np.array([frac, frac]))
        self._mj.mj_forward(self._model, self._data)
        self.obs = e.get_observation()
        self.success = bool(e._success())
        self._frame = None

    def ground_truth(self) -> Dict:
        """What the simulator knows about the current state: ``success``, and ``distance`` (m, closest allowed
        wrist to the target) plus ``closer`` (the nearer wrist) for reach tasks, or ``open`` for cupboard tasks."""
        truth = {"success": bool(self.env._success())}
        if self.spec["kind"] == _REACH:
            t = self.target()
            d = {h: float(np.linalg.norm(self.hand_pos(h) - t)) for h in ("left", "right")}
            truth.update(distance=min(d[h] for h in self.spec["hands"]), closer=min(d, key=d.get),
                         distances=d)
        else:
            truth["open"] = self.open_fraction()
        return truth

    def close(self) -> None:
        self.env.close()


# ---------------------------------------------------------------------------------------------------------
# Policies and episodes
# ---------------------------------------------------------------------------------------------------------

def oracle_action(game: BiGymGame) -> str:
    """Privileged greedy reach: for the allowed wrist nearest the target, the primitive (a wrist move or a base
    step) whose IK-predicted wrist motion most reduces the distance. Only defined for reach tasks."""
    if game.spec["kind"] != _REACH:
        raise ValueError("no primitive oracle for %s" % game.task)
    t = game.target()
    hand = min(game.spec["hands"], key=lambda h: np.linalg.norm(game.hand_pos(h) - t))
    p = game.hand_pos(hand)
    cands = ["%s_HAND_%s" % (hand.upper(), a) for a in _AXES] + ["BASE_FORWARD", "BASE_BACK"]
    return min(cands, key=lambda c: float(np.linalg.norm(p + game.predicted_move(c, hand) - t)))


def oracle_policy(game: BiGymGame) -> str:
    return oracle_action(game)


def random_policy(seed: int = 0):
    rng = random.Random(seed)
    return lambda game: rng.choice(game.actions)


def model_policy(agent, task: str, frames: int = 1, ghost: float = 0.0):
    """The model's most likely primitive via ``predict``: from the head frame, or with ``frames`` > 1 from the
    last ``frames`` decision frames as ``{"images": [oldest, ..., now]}``, so it can see motion."""
    q = bigym_question(task, frames)

    def policy(game):
        state = {"images": game.frames(frames)} if frames > 1 else {"image": game.render(ghost)}
        return FROM_WORDS[agent.predict(state, q, strict=True)["answers"]["action"]["choice"]]
    return policy


def play_episodes(task: str, policy, episodes: int, seed: int = SEED, max_decisions: int = 0) -> Dict:
    """Play ``episodes`` seeded episodes (seed ``seed + i``); ``policy(game) -> primitive name``. An episode ends
    on success, on BiGym's own termination, or after ``max_decisions`` (default: the task's)."""
    from collections import Counter

    cap = max_decisions or TASKS[task]["max_decisions"]
    counts, eps, env = Counter(), [], make_env(task)
    for i in range(episodes):
        game = BiGymGame(task, seed + i, env=env)
        while not game.done and game.decisions < cap:
            a = policy(game)
            counts[a] += 1
            game.step(a)
        truth = game.ground_truth()
        eps.append({"seed": seed + i, "success": game.success, "decisions": game.decisions,
                    "terminated": bool(game.terminated), "final": {k: v for k, v in truth.items() if k != "success"}})
    env.close()
    wins = [e["decisions"] for e in eps if e["success"]]
    return {"task": task, "episodes": episodes, "seed": seed, "max_decisions": cap, "actions": dict(counts),
            "results": eps, "success_rate": float(np.mean([e["success"] for e in eps])),
            "mean_decisions_to_success": float(np.mean(wins)) if wins else None}


def demo_reference(task: str, amount: int = -1, seed: int = 0) -> Dict:
    """Success rate of BiGym's human demonstrations for ``task`` replayed in the sim (``DemoPlayer.validate_in_env``),
    the expert reference for the cupboard tasks. Demos download on first use to ``~/.bigym`` (about 120 MB);
    the reach tasks have none. Replays in BiGym's own settings: absolute joint positions at its minimum control
    frequency with the four floating-base DOFs the demos were recorded with, as its
    ``download_and_validate_demos.py`` does. ``seed`` fixes which demos ``amount`` picks (DemoStore shuffles with
    the global NumPy generator)."""
    import importlib

    from bigym.action_modes import JointPositionActionMode, PelvisDof
    from bigym.bigym_env import CONTROL_FREQUENCY_MIN
    from demonstrations.demo_player import DemoPlayer
    from demonstrations.demo_store import DemoStore
    from demonstrations.utils import Metadata

    cls = getattr(importlib.import_module("bigym.envs." + TASKS[task]["module"]), task)
    dofs = [PelvisDof.X, PelvisDof.Y, PelvisDof.Z, PelvisDof.RZ]
    env = cls(action_mode=JointPositionActionMode(absolute=True, floating_base=True, floating_dofs=dofs),
              control_frequency=CONTROL_FREQUENCY_MIN)
    np.random.seed(seed)
    demos = DemoStore().get_demos(Metadata.from_env(env, is_lightweight=True), amount=amount,
                                  frequency=CONTROL_FREQUENCY_MIN)
    ok = [bool(DemoPlayer.validate_in_env(d, env, CONTROL_FREQUENCY_MIN)) for d in demos]
    lengths = [len(d.timesteps) for d in demos]
    env.close()
    return {"task": task, "demos": len(ok), "success_rate": float(np.mean(ok)) if ok else None,
            "mean_length_steps": float(np.mean(lengths)) if lengths else None,
            "control_frequency": CONTROL_FREQUENCY_MIN}


# ---------------------------------------------------------------------------------------------------------
# Probe frames
# ---------------------------------------------------------------------------------------------------------

def probe_frames(task: str, n: int, seed: int = SEED, random_moves: int = 8) -> List[Dict]:
    """``n`` head-camera frames with ground truth, balanced over the task's progress levels.

    Reach tasks: frames along oracle and random rollouts (half each), so the wrist-target distance spans the
    levels. Cupboard tasks: the drawer or doors posed at an open fraction drawn uniformly across the task's
    progress levels, after a few random primitives so the arms are not always in the reset pose.
    Each row: ``{"image": HxWx3 uint8, "truth": ground_truth(), "labels": labels(), "seed": ...}``."""
    rng = random.Random(seed)
    env, rows = make_env(task), []
    spec = TASKS[task]
    i = 0
    while len(rows) < n:
        game = BiGymGame(task, seed + i, env=env)
        if spec["kind"] == _REACH:
            policy = oracle_policy if i % 2 == 0 else random_policy(seed + i)
            per_ep = 6
            keep = sorted(rng.sample(range(spec["max_decisions"]), per_ep))
            for d in range(spec["max_decisions"]):
                if d in keep or game.done:
                    truth = game.ground_truth()
                    rows.append({"image": game.frame().copy(), "truth": truth, "labels": labels(task, truth),
                                 "seed": seed + i, "decision": d})
                    if game.done or len(rows) >= n:
                        break
                game.step(policy(game))
        else:
            for _ in range(rng.randint(0, random_moves)):
                a = rng.choice([p for p in PRIMITIVES if "HAND" in p or "GRIPPER" in p])
                game.step(a)
            lo, hi = [(0.0, 0.3), (0.3, 0.6), (0.6, 0.9), (0.9, 1.0)][len(rows) % 4]
            frac = rng.uniform(lo, hi)
            game.set_open_fraction(frac if spec["goal"] == 1 else 1.0 - frac)
            truth = game.ground_truth()
            rows.append({"image": game.frame().copy(), "truth": truth, "labels": labels(task, truth),
                         "seed": seed + i})
        i += 1
    env.close()
    return rows[:n]


def probe_metrics(rows: List[Dict]) -> Dict:
    """Per question: accuracy, ECE, NLL and the prior-only baseline (label frequencies of these rows) for
    ``rows`` of ``{"qid", "label", "probs"}``."""
    from .common import ece_score

    out = {}
    for qid in sorted({r["qid"] for r in rows}):
        rs = [r for r in rows if r["qid"] == qid]
        P = np.array([r["probs"] for r in rs], np.float64)
        y = np.array([r["label"] for r in rs])
        k = P.shape[1]
        prior = np.bincount(y, minlength=k) / len(y)
        pred = P.argmax(1)
        eps = 1e-6
        out[qid] = {"n": len(rs), "acc": float((pred == y).mean()),
                    "ece": float(ece_score(P.max(1), (pred == y).astype(np.float64))),
                    "nll": float(-np.log(np.clip(P[np.arange(len(y)), y], eps, 1)).mean()),
                    "prior_acc": float(prior.max()), "prior_nll": float(-np.log(np.clip(prior[y], eps, 1)).mean()),
                    "label_counts": np.bincount(y, minlength=k).tolist(),
                    "pred_counts": np.bincount(pred, minlength=k).tolist()}
        if k == 2:  # does P(label 1) rank the frames, whatever the threshold (0.5 = chance)
            out[qid]["auroc"] = _auroc(P[:, 1], y)
        else:  # ordinal: mean absolute level error, and rank correlation, of the expected level
            level = (P * np.arange(k)).sum(1)
            out[qid]["mae"] = float(np.abs(level - y).mean())
            out[qid]["prior_mae"] = float(np.abs((prior * np.arange(k)).sum() - y).mean())
            out[qid]["spearman"] = _spearman(level, y)
    return out


def _ranks(x: np.ndarray) -> np.ndarray:
    """Average ranks (ties share their mean rank)."""
    x = np.asarray(x, np.float64)
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    r[order] = np.arange(len(x))
    for v in np.unique(x):
        r[x == v] = r[x == v].mean()
    return r


def _auroc(score: np.ndarray, y: np.ndarray) -> Optional[float]:
    pos, neg = score[y == 1], score[y == 0]
    if not len(pos) or not len(neg):
        return None
    return float((pos[:, None] > neg[None]).mean() + 0.5 * (pos[:, None] == neg[None]).mean())


def _spearman(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    ra, rb = _ranks(a), _ranks(b)
    if ra.std() == 0 or rb.std() == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def answer_probs(answer: Dict, question: Dict) -> List[float]:
    """A ``predict`` answer's probabilities in label order (``noul``: [false, true])."""
    if answer["type"] == "noul":
        return [1.0 - answer["noul"], answer["noul"]]
    if answer["type"] == "score":
        return [answer["probabilities"][str(i)] for i in range(len(question["criteria"]))]
    return [answer["probabilities"][k] for k in question["criteria"]]


__all__ = ["TASKS", "PRIMITIVES", "HOLD", "bigym_question", "probe_questions", "progress_level", "labels",
           "normalized", "make_env", "BiGymGame", "oracle_policy", "random_policy", "model_policy",
           "play_episodes", "demo_reference", "probe_frames", "probe_metrics", "answer_probs"]
