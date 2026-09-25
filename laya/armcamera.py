"""Arm camera: active perception with a camera on a robot arm's wrist, played from pixels.

A coloured cube sits on a stand; one of its faces carries a target marker (a white disc with a black bullseye). A
six-joint arm stands beside the stand with a camera fixed to its end, pointing out between the gripper fingers. The
model sees through that wrist camera (optionally with a fixed room camera as a second image) and moves the camera to
get a better view: the task ("find the marker") is to bring the marker to the centre of the wrist view, facing the
camera and large. Each seed draws the cube's yaw and which face carries the marker.

Actions (``ACTIONS``, one ``choice`` per step): named camera moves, ORBIT_LEFT / ORBIT_RIGHT (15 degrees around the
object), MOVE_UP / MOVE_DOWN (15 degrees of elevation), CLOSER / FARTHER (distance x0.8 / x1.25) and NONE. A
damped-least-squares IK controller (``ik``) turns each move into joint angles that put the camera at the new place
on a sphere around the object, looking at the object's centre with the horizon level. Why these and not joint
levels: the model then decides only *where to look from*, which it can read off the image (the marker is on the
left, so circle left); per-joint levels (``JOINT_LEVELS``, ``control="joints"``) are kept for comparison, but a joint
move swings the view in ways that depend on the whole arm pose, and the object leaves the frame after a few moves
(random joint levels: 0/20 successes, against 2/20 for random camera moves).

The camera lives on a lattice around the object (``AZIMUTHS`` x ``ELEVATIONS`` x ``DISTANCES``, 17 x 6 x 4): azimuth
-120..120 degrees from the arm's side (the arm cannot reach behind the stand), elevation 0..75 degrees, distance
0.4..0.2 m. A move past the lattice's edge leaves the camera where it is. Every lattice pose is reachable (checked by
the tests); the IK starts from the current joints, and from that pose's precomputed solution if it gets stuck at a
joint limit, so the moves always do what they say.

Reward: ``quality`` in [0, 1] is the marker's visibility times how squarely it faces the camera, how centred it is and
how large it looks (``marker_view``, computed from the true poses and checked against segmentation rendering). Each
step pays the change in quality, minus 0.01, plus 1 on success; success (``SUCCESS``: visible, within 20% of the
half-width of the centre, facing within about 40 degrees, at least 32 pixels in radius, which needs a distance of
0.32 m or less) ends the episode; otherwise it ends after ``MAX_STEPS`` (40). Seeds place the marker on a face some
lattice pose can see well, and start the camera at a pose at least ``MIN_START_MOVES`` (3) moves from success.

Experts plan on the lattice with the true poses: value iteration over the 408 camera poses gives the number of moves
to success from each, and the expert takes a move that shortens it (ties to the one whose next view is better).
``random_policy`` and ``still_policy`` are the baselines.

Kinematic: the arm's joints are set, not simulated (``mj_forward``), so nothing falls or collides. Needs ``mujoco``;
headless Linux renders through EGL (``MUJOCO_GL=egl``, set when no display is present).
"""
import os
import random
import sys
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np

SIZE = 256  # rendered frame, pixels a side
GHOST = 0.35  # weight of the previous wrist frame in ``render``
FOVY = 60.0  # wrist camera, degrees
FOCAL = SIZE / 2 / np.tan(np.radians(FOVY / 2))  # pixels
MAX_STEPS = 40
MIN_START_MOVES = 3  # the start pose is at least this many moves from success

OBJ_CENTER = np.array([0.0, 0.0, 0.57])  # the cube's centre, on the stand
HALF = 0.07  # the cube's half side
MARKER_R = 0.045  # the marker disc's radius
# face name -> (outward normal in the cube's frame, colour); the marker goes on one of the first five
FACES = {
    "front": ((1, 0, 0), (0.85, 0.15, 0.15)),   # red
    "back": ((-1, 0, 0), (0.15, 0.65, 0.2)),    # green
    "left": ((0, 1, 0), (0.15, 0.3, 0.85)),     # blue
    "right": ((0, -1, 0), (0.95, 0.8, 0.1)),    # yellow
    "top": ((0, 0, 1), (0.6, 0.25, 0.75)),      # purple
    "bottom": ((0, 0, -1), (0.4, 0.4, 0.4)),
}
MARKER_FACES = ("front", "back", "left", "right", "top")
SUCCESS = {"off": 0.2, "cos": 0.75, "size_px": 32.0}

AZIMUTHS = np.radians(np.arange(-120, 121, 15))  # 17, 0 = the arm's side of the stand (+x)
ELEVATIONS = np.radians(np.arange(0, 76, 15))  # 6
DISTANCES = 0.4 * 0.8 ** np.arange(4)  # 0.4 .. 0.205 m, camera to the object's centre

ACTIONS = {
    "ORBIT_LEFT": "circle the camera to the left around the object",
    "ORBIT_RIGHT": "circle the camera to the right around the object",
    "MOVE_UP": "raise the camera to look down on the object from higher",
    "MOVE_DOWN": "lower the camera to look at the object more from the side",
    "CLOSER": "move the camera closer to the object",
    "FARTHER": "move the camera away from the object",
    "NONE": "keep the camera where it is",
}
# lattice step (azimuth, elevation, distance index) of each move; ORBIT_RIGHT moves the camera toward its own right
MOVES = {"ORBIT_LEFT": (-1, 0, 0), "ORBIT_RIGHT": (1, 0, 0), "MOVE_UP": (0, 1, 0), "MOVE_DOWN": (0, -1, 0),
         "CLOSER": (0, 0, 1), "FARTHER": (0, 0, -1), "NONE": (0, 0, 0)}

JOINTS = ("base_yaw", "shoulder", "elbow", "wrist_pitch", "wrist_yaw", "wrist_roll")
JOINT_LEVELS = {"NEG": "turn the joint the negative way", "NONE": "leave the joint", "POS": "turn the joint the "
                "positive way"}
JOINT_STEP = np.radians(8.0)
HOME = np.array([0.0, -0.5, -1.4, -0.9, 0.0, 0.0])

ARM_BASE = (0.65, 0.0, 0.0)
ROOM_CAMERA = {"pos": (1.1, -1.3, 1.35), "lookat": (0.3, 0.0, 0.62)}


def _xyaxes(pos, lookat, up=(0, 0, 1)) -> str:
    f = np.asarray(lookat, float) - np.asarray(pos, float)
    f /= np.linalg.norm(f)
    x = np.cross(f, up)
    x /= np.linalg.norm(x)
    y = np.cross(x, f)
    return " ".join("%.5f" % v for v in (*x, *y))


def _cube_geoms() -> str:
    out = ['<geom name="cube" type="box" size="{h} {h} {h}" rgba="0.12 0.12 0.12 1"/>'.format(h=HALF - 0.001)]
    for name, (n, rgb) in FACES.items():  # a thin coloured plate on each face
        n = np.asarray(n, float)
        size = [HALF - 0.004 if a == 0 else 0.002 for a in n]
        out.append('<geom name="face_%s" type="box" pos="%s" size="%s" rgba="%.2f %.2f %.2f 1"/>' % (
            name, " ".join("%.4f" % v for v in n * (HALF - 0.002)), " ".join("%.4f" % v for v in size), *rgb))
    return "\n        ".join(out)


MJCF = """
<mujoco model="arm_camera">
  <compiler angle="radian"/>
  <option gravity="0 0 0"/>
  <visual>
    <global offwidth="{size}" offheight="{size}"/>
    <quality shadowsize="2048"/>
    <headlight ambient="0.35 0.35 0.35" diffuse="0.5 0.5 0.5" specular="0.1 0.1 0.1"/>
  </visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.72 0.72 0.7" rgb2="0.6 0.6 0.58" width="256" height="256"/>
    <material name="floor" texture="grid" texrepeat="6 6"/>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.55 0.65 0.8" rgb2="0.15 0.15 0.2"
             width="256" height="256"/>
    <material name="arm" rgba="0.8 0.8 0.82 1" specular="0.3"/>
    <material name="joint" rgba="0.25 0.25 0.3 1"/>
  </asset>
  <default>
    <geom contype="0" conaffinity="0"/>
    <joint type="hinge" damping="1"/>
  </default>
  <worldbody>
    <light pos="0.5 -0.5 2.5" dir="-0.2 0.2 -1" diffuse="0.6 0.6 0.6" castshadow="true"/>
    <geom name="floor" type="plane" size="3 3 0.1" material="floor"/>
    <camera name="room" pos="{room_pos}" xyaxes="{room_xy}" fovy="45"/>
    <body name="stand">
      <geom type="cylinder" pos="0 0 0.01" size="0.14 0.01" rgba="0.3 0.3 0.32 1"/>
      <geom type="cylinder" pos="0 0 0.245" size="0.04 0.235" rgba="0.45 0.45 0.48 1"/>
      <geom type="cylinder" pos="0 0 0.49" size="0.09 0.01" rgba="0.3 0.3 0.32 1"/>
    </body>
    <body name="object" pos="{obj}">
        {cube}
      <body name="marker">
        <geom name="marker_disc" type="cylinder" pos="0 0 0.0003" size="{mr} 0.0003" rgba="1 1 1 1"/>
        <geom name="marker_ring" type="cylinder" pos="0 0 0.0006" size="{mr2} 0.0003" rgba="0.05 0.05 0.05 1"/>
        <geom name="marker_inner" type="cylinder" pos="0 0 0.0009" size="{mr3} 0.0003" rgba="1 1 1 1"/>
        <geom name="marker_dot" type="cylinder" pos="0 0 0.0012" size="{mr4} 0.0003" rgba="0.05 0.05 0.05 1"/>
      </body>
    </body>
    <body name="base" pos="{base}">
      <geom type="cylinder" pos="0 0 0.03" size="0.12 0.03" material="joint"/>
      <joint name="base_yaw" axis="0 0 1" range="-3.1 3.1"/>
      <geom type="cylinder" pos="0 0 0.18" size="0.06 0.15" material="arm"/>
      <body name="upper" pos="0 0 0.35">
        <joint name="shoulder" axis="0 1 0" range="-2.0 1.3"/>
        <geom type="cylinder" fromto="0 -0.06 0 0 0.06 0" size="0.06" material="joint"/>
        <geom type="capsule" fromto="0 0 0 0 0 0.6" size="0.04" material="arm"/>
        <body name="fore" pos="0 0 0.6">
          <joint name="elbow" axis="0 1 0" range="-2.95 0.3"/>
          <geom type="cylinder" fromto="0 -0.05 0 0 0.05 0" size="0.05" material="joint"/>
          <geom type="capsule" fromto="0 0 0 0 0 0.53" size="0.033" material="arm"/>
          <body name="wrist" pos="0 0 0.53">
            <joint name="wrist_pitch" axis="0 1 0" range="-3.1 3.1"/>
            <geom type="sphere" size="0.04" material="joint"/>
            <body name="wrist2" pos="0 0 0.05">
              <joint name="wrist_yaw" axis="1 0 0" range="-2.2 2.2"/>
              <geom type="capsule" fromto="0 0 -0.03 0 0 0.03" size="0.028" material="arm"/>
              <body name="hand" pos="0 0 0.05">
                <joint name="wrist_roll" axis="0 0 1" range="-6.2 6.2"/>
                <geom type="box" pos="0 0 0.02" size="0.05 0.025 0.02" material="joint"/>
                <geom type="box" pos="0.04 0 0.07" size="0.008 0.015 0.035" material="arm"/>
                <geom type="box" pos="-0.04 0 0.07" size="0.008 0.015 0.035" material="arm"/>
                <geom type="cylinder" pos="0 0 0.045" size="0.015 0.006" rgba="0.05 0.05 0.05 1"/>
                <camera name="wrist" pos="0 0 0.052" xyaxes="1 0 0 0 -1 0" fovy="{fovy}"/>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
""".format(size=SIZE, room_pos=" ".join(map(str, ROOM_CAMERA["pos"])),
           room_xy=_xyaxes(ROOM_CAMERA["pos"], ROOM_CAMERA["lookat"]), obj=" ".join(map(str, OBJ_CENTER)),
           cube=_cube_geoms(), mr=MARKER_R, mr2=MARKER_R * 0.72, mr3=MARKER_R * 0.48, mr4=MARKER_R * 0.22,
           base=" ".join(map(str, ARM_BASE)), fovy=FOVY)


def _gl() -> None:
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        os.environ.setdefault("MUJOCO_GL", "egl")  # read when mujoco is first imported
    try:  # as in laya.mujocogames: triton's LLVM must load before the GL library's
        import torch._dynamo  # noqa: F401
    except ImportError:
        pass


def _mujoco():
    """``mujoco``, imported after ``_gl`` has picked the GL backend."""
    _gl()
    import mujoco

    return mujoco


def camera_target(state: Tuple[int, int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """The camera position and orientation (columns: right, up, backward, MuJoCo's camera frame) for a lattice
    state (azimuth, elevation, distance indices): on the sphere around the object, looking at its centre, level."""
    az, el, r = AZIMUTHS[state[0]], ELEVATIONS[state[1]], DISTANCES[state[2]]
    d = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    pos = OBJ_CENTER + r * d
    fwd = -d
    right = np.cross(fwd, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    return pos, np.stack([right, up, -fwd], axis=1)


def _yaw_matrix(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def marker_view(cam_pos, cam_mat, marker_pos, marker_normal) -> Dict:
    """How the marker looks from a camera (``cam_mat`` columns right, up, backward): ``visible`` (in front, inside
    the frame, its face toward the camera; the cube is convex and nothing else stands in between), ``u, v`` its
    centre in pixels, ``off`` the centre's distance from the image centre over the half-width, ``cos`` how squarely
    it faces the camera, ``size_px`` its apparent radius (geometric mean of the ellipse's axes), ``quality`` in
    [0, 1] and ``success``."""
    v = np.asarray(marker_pos) - np.asarray(cam_pos)
    x, y, depth = v @ cam_mat[:, 0], v @ cam_mat[:, 1], -(v @ cam_mat[:, 2])
    dist = float(np.linalg.norm(v))
    cos = float(np.asarray(marker_normal) @ (-v) / dist)
    out = {"visible": False, "u": None, "v": None, "off": 1.0, "cos": cos, "size_px": 0.0, "quality": 0.0,
           "success": False}
    if depth <= 1e-3 or cos <= 0.0:
        return out
    u, w = SIZE / 2 + FOCAL * x / depth, SIZE / 2 - FOCAL * y / depth
    if not (0 <= u < SIZE and 0 <= w < SIZE):
        return out
    off = float(np.hypot(u - SIZE / 2, w - SIZE / 2) / (SIZE / 2))
    size = float(FOCAL * MARKER_R / dist * np.sqrt(cos))
    q = cos * max(0.0, 1.0 - off) * min(1.0, size / SUCCESS["size_px"])
    ok = off <= SUCCESS["off"] and cos >= SUCCESS["cos"] and size >= SUCCESS["size_px"]
    out.update(visible=True, u=float(u), v=float(w), off=off, size_px=size, quality=float(q), success=bool(ok))
    return out


_MODEL = None
_LATTICE_Q: Dict[Tuple[int, int, int], np.ndarray] = {}


def _model():
    global _MODEL
    if _MODEL is None:
        _MODEL = _mujoco().MjModel.from_xml_string(MJCF)
    return _MODEL


def ik(model, data, pos, mat, q0, iters: int = 300, tol: float = 1e-4) -> Tuple[np.ndarray, float]:
    """Damped least squares on the wrist camera's pose: joint angles (within their ranges) putting the camera at
    ``pos`` with orientation ``mat``, starting from ``q0``. Returns ``(q, residual)``, the residual being the larger
    of the position error (m) and the orientation error (rad)."""
    mujoco = _mujoco()

    cam = model.camera("wrist").id
    body = model.cam_bodyid[cam]
    lo, hi = model.jnt_range[:, 0], model.jnt_range[:, 1]
    q = np.clip(np.array(q0, dtype=float), lo, hi)
    jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    err = np.inf
    for _ in range(iters):
        data.qpos[:] = q
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        mujoco.mj_camlight(model, data)
        cp, cm = data.cam_xpos[cam], data.cam_xmat[cam].reshape(3, 3)
        ep = pos - cp
        er = 0.5 * sum(np.cross(cm[:, i], mat[:, i]) for i in range(3))
        err = max(float(np.linalg.norm(ep)), float(np.linalg.norm(er)))
        if err < tol:
            break
        mujoco.mj_jac(model, data, jp, jr, cp, body)
        J, e = np.vstack([jp, 0.3 * jr]), np.concatenate([ep, 0.3 * er])
        dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), e)
        n = np.linalg.norm(dq)
        if n > 0.2:
            dq *= 0.2 / n
        q = np.clip(q + dq, lo, hi)
    return q, err


def lattice_solutions() -> Dict[Tuple[int, int, int], np.ndarray]:
    """IK solutions for every lattice pose, found by sweeping out from the middle of the lattice so neighbours start
    from each other's solution (one arm branch throughout). Cached. Used to start each episode; moves then run the IK
    from the current joints."""
    if _LATTICE_Q:
        return _LATTICE_Q
    mujoco = _mujoco()

    m = _model()
    d = mujoco.MjData(m)
    start = (len(AZIMUTHS) // 2, 2, 2)
    q, _ = ik(m, d, *camera_target(start), HOME, iters=2000)
    _LATTICE_Q[start] = q
    frontier = [start]
    while frontier:
        nxt = []
        for s in frontier:
            for dlt in MOVES.values():
                t = _clamp(tuple(a + b for a, b in zip(s, dlt)))
                if t not in _LATTICE_Q:
                    _LATTICE_Q[t], _ = ik(m, d, *camera_target(t), _LATTICE_Q[s], iters=2000)
                    nxt.append(t)
        frontier = nxt
    return _LATTICE_Q


def _clamp(s) -> Tuple[int, int, int]:
    return (min(max(s[0], 0), len(AZIMUTHS) - 1), min(max(s[1], 0), len(ELEVATIONS) - 1),
            min(max(s[2], 0), len(DISTANCES) - 1))


def lattice_states() -> List[Tuple[int, int, int]]:
    return [(a, e, r) for a in range(len(AZIMUTHS)) for e in range(len(ELEVATIONS)) for r in range(len(DISTANCES))]


class ArmCamera:
    """One seeded episode. ``control="camera"`` (default) steps by an ``ACTIONS`` name; ``control="joints"`` by a
    dict ``{joint: JOINT_LEVELS name}`` (8 degrees a step), the camera then no longer kept on the object."""

    game = "ArmCamera"

    def __init__(self, seed: int = 0, control: str = "camera", max_steps: int = MAX_STEPS):
        mujoco = _mujoco()
        self._mj = mujoco
        self.seed, self.control, self.max_steps = seed, control, max_steps
        self.actions = tuple(ACTIONS)
        self.joints = JOINTS if control == "joints" else None
        _model()
        self.model = mujoco.MjModel.from_xml_string(MJCF)  # its own: the object's pose is set per seed
        self.data = mujoco.MjData(self.model)
        rng = random.Random(seed)
        self.yaw = rng.uniform(-np.pi, np.pi)
        self._rot = _yaw_matrix(self.yaw)
        faces = list(MARKER_FACES)
        rng.shuffle(faces)
        for face in faces:  # the first face (in a seeded order) some lattice pose sees well
            self.face = face
            self._goals = {s for s in lattice_states() if self.view_from(*camera_target(s))["success"]}
            if self._goals:
                break
        self._dist = self._distances()
        starts = [s for s in lattice_states() if self._dist[s] >= MIN_START_MOVES]
        self.state = starts[rng.randrange(len(starts))]
        self.q = lattice_solutions()[self.state].copy()
        self._apply()
        self.score, self.steps, self.terminated, self.truncated = 0.0, 0, False, False
        self.blocked = 0  # moves the IK could not carry out
        self._frame = self._prev = self._renderer = None
        self.view = self.marker()
        self.start_quality = self.view["quality"]

    # geometry -------------------------------------------------------------------------------------------------
    def marker_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        n = self._rot @ np.asarray(FACES[self.face][0], float)
        return OBJ_CENTER + n * (HALF + 0.001), n

    def view_from(self, cam_pos, cam_mat) -> Dict:
        return marker_view(cam_pos, cam_mat, *self.marker_pose())

    def camera_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        cam = self.model.camera("wrist").id
        return self.data.cam_xpos[cam].copy(), self.data.cam_xmat[cam].reshape(3, 3).copy()

    def marker(self) -> Dict:
        """``marker_view`` from the wrist camera where it is now."""
        return self.view_from(*self.camera_pose())

    def _apply(self) -> None:
        m, d = self.model, self.data
        w = np.cos(self.yaw / 2), 0.0, 0.0, np.sin(self.yaw / 2)
        m.body("object").quat[:] = w
        n = np.asarray(FACES[self.face][0], float)  # marker body: its z along the face normal, on the face
        m.body("marker").pos[:] = n * HALF
        z = n
        x = np.array([0, 0, 1.0]) if abs(z[2]) < 0.5 else np.array([1.0, 0, 0])
        x = x - (x @ z) * z
        mat = np.stack([x, np.cross(z, x), z], axis=1)
        quat = np.zeros(4)
        self._mj.mju_mat2Quat(quat, mat.flatten())
        m.body("marker").quat[:] = quat
        d.qpos[:] = self.q
        self._mj.mj_forward(m, d)

    def _distances(self) -> Dict:
        """Moves to success from every lattice pose (value iteration on the lattice)."""
        states = lattice_states()
        dist = {s: (0 if s in self._goals else np.inf) for s in states}
        changed = True
        while changed:
            changed = False
            for s in states:
                if s in self._goals:
                    continue
                best = 1 + min(dist[_clamp(tuple(a + b for a, b in zip(s, dl)))] for dl in MOVES.values())
                if best < dist[s]:
                    dist[s], changed = best, True
        return dist

    # play -----------------------------------------------------------------------------------------------------
    @property
    def done(self) -> bool:
        return self.terminated or self.truncated

    @property
    def success(self) -> bool:
        return bool(self.view["success"])

    def step(self, action) -> float:
        self._prev, self._frame = self._frame, None
        if self.control == "joints":
            delta = np.array([{"NEG": -1, "NONE": 0, "POS": 1}[action[j]] for j in JOINTS]) * JOINT_STEP
            lo, hi = self.model.jnt_range[:, 0], self.model.jnt_range[:, 1]
            self.q = np.clip(self.q + delta, lo, hi)
        else:
            target = _clamp(tuple(a + b for a, b in zip(self.state, MOVES[action])))
            if target != self.state:
                pose = camera_target(target)
                q, err = ik(self.model, self.data, *pose, self.q)  # from the current joints: a smooth move
                if err >= 1e-3:  # stuck at a joint limit: start again from the lattice's solution
                    q, err = ik(self.model, self.data, *pose, lattice_solutions()[target])
                if err < 1e-3:
                    self.state, self.q = target, q
                else:
                    self.blocked += 1
        self._apply()
        before, self.view = self.view["quality"], self.marker()
        reward = self.view["quality"] - before - 0.01 + (1.0 if self.view["success"] else 0.0)
        self.score += reward
        self.steps += 1
        self.terminated = self.view["success"]
        self.truncated = not self.terminated and self.steps >= self.max_steps
        return reward

    def next_state(self, action) -> Tuple[int, int, int]:
        return _clamp(tuple(a + b for a, b in zip(self.state, MOVES[action])))

    def expert_actions(self) -> List[str]:
        """Every camera move on a shortest path to success (the true poses; the lattice's ideal views)."""
        best = min(self._dist[self.next_state(a)] for a in self.actions)
        return [a for a in self.actions if self._dist[self.next_state(a)] == best]

    def expert(self) -> str:
        """A shortest-path move, ties to the one whose next view has the higher quality."""
        return max(self.expert_actions(), key=lambda a: self.view_from(*camera_target(self.next_state(a)))["quality"])

    @property
    def moves_to_success(self) -> float:
        return self._dist[self.state]

    # pixels ---------------------------------------------------------------------------------------------------
    def _render_camera(self, camera: str, segmentation: bool = False) -> np.ndarray:
        if self._renderer is None:
            self._renderer = self._mj.Renderer(self.model, SIZE, SIZE)
        r = self._renderer
        if segmentation:
            r.enable_segmentation_rendering()
        r.update_scene(self.data, camera=camera)
        out = r.render().copy()
        if segmentation:
            r.disable_segmentation_rendering()
        return out

    def frame(self) -> np.ndarray:
        """The wrist camera's current RGB frame (rendered once per step)."""
        if self._frame is None:
            self._frame = self._render_camera("wrist")
        return self._frame

    def room(self):
        """The fixed room camera's view, as a PIL image."""
        from PIL import Image

        return Image.fromarray(self._render_camera("room"))

    def render(self):
        """The wrist frame with the previous one ghosted underneath, as a PIL image."""
        from PIL import Image

        cur = self.frame()
        if self._prev is None:
            return Image.fromarray(cur)
        mix = (1 - GHOST) * cur.astype(np.float32) + GHOST * self._prev.astype(np.float32)
        return Image.fromarray(mix.round().astype(np.uint8))

    def marker_pixels(self) -> int:
        """How many wrist-camera pixels show the marker (segmentation rendering)."""
        seg = self._render_camera("wrist", segmentation=True)
        ids = {self.model.geom(n).id for n in ("marker_disc", "marker_ring", "marker_inner", "marker_dot")}
        geom = seg[..., 1] == int(self._mj.mjtObj.mjOBJ_GEOM)
        return int((np.isin(seg[..., 0], list(ids)) & geom).sum())

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


# questions ----------------------------------------------------------------------------------------------------
GOAL = ("A cube with differently coloured faces sits on a stand; one face carries a target, a white disc with a "
        "black ring and a black dot. Move the camera until the target is in the centre of the camera's view, "
        "facing the camera squarely and large.")


def question(room: bool = False) -> Dict:
    """The question asked each step, about the wrist view (``room=False``) or the wrist and room views."""
    if room:
        seen = ("You control a robot arm with a camera on its gripper. The first image is what that camera sees; a "
                "faint copy shows its view one step earlier. The second image is a fixed camera watching the arm "
                "and the stand from across the room.")
    else:
        seen = ("You control a robot arm with a camera on its gripper; the image is what that camera sees, and a "
                "faint copy shows its view one step earlier.")
    return {"action": {"type": "choice",
                       "instructions": "%s %s Which way should you move the camera now?" % (seen, GOAL),
                       "criteria": dict(ACTIONS)}}


def state(env: ArmCamera, room: bool = False) -> Dict:
    """What ``predict`` gets: the ghosted wrist view, and the room view second when ``room``."""
    return {"images": [env.render(), env.room()]} if room else {"image": env.render()}


def model_chooser(agent, room: bool = False):
    q = question(room)

    def choose(env):
        ans = agent.predict(state(env, room), q)["answers"]["action"]
        return ans["choice"], ans.get("probabilities")
    return choose


# policies and episodes ----------------------------------------------------------------------------------------
def expert_policy(env: ArmCamera) -> str:
    return env.expert()


def random_policy(seed: int = 0):
    rng = random.Random(seed)

    def pick(env):
        if env.joints is not None:
            return {j: rng.choice(tuple(JOINT_LEVELS)) for j in env.joints}
        return rng.choice(env.actions)
    return pick


def still_policy(env) -> object:
    return {j: "NONE" for j in env.joints} if env.joints is not None else "NONE"


def play_episodes(policy, episodes: int, seed: int = 200_000, control: str = "camera",
                  max_steps: int = MAX_STEPS) -> Dict:
    """Seeded episodes (seed ``seed + i``) with ``policy(env) -> action``: success rate, return, steps."""
    counts, eps = Counter(), []
    for i in range(episodes):
        env = ArmCamera(seed + i, control=control, max_steps=max_steps)
        while not env.done:
            a = policy(env)
            counts.update(a.values() if isinstance(a, dict) else [a])
            env.step(a)
        eps.append({"score": round(env.score, 4), "steps": env.steps, "success": env.success,
                    "final_quality": round(env.view["quality"], 4), "blocked": env.blocked})
        env.close()
    return {"episodes": episodes, "seed": seed, "actions": dict(counts), "results": eps,
            "success_rate": float(np.mean([e["success"] for e in eps])),
            "mean_score": float(np.mean([e["score"] for e in eps])),
            "mean_steps": float(np.mean([e["steps"] for e in eps])),
            "mean_steps_success": float(np.mean([e["steps"] for e in eps if e["success"]] or [np.nan]))}


def expert_frames(n: int, seed: int = 0, noise: float = 0.2, room: bool = False, smooth: float = 0.05):
    """Training frames labelled by the expert: yields ``{"episode", "step", "state", "records"}`` until ``n``.

    The behaviour policy is the expert with a random move instead with probability ``noise``, so frames include
    detours; every step is kept (episodes are short) and rendered as in play. The target spreads ``1 - smooth``
    evenly over every shortest-path move (``expert_actions``), ``smooth`` over all."""
    rng = random.Random(seed)
    q = question(room)["action"]
    made, ep = 0, 0
    while made < n:
        env = ArmCamera(seed + ep)
        while not env.done and made < n:
            best = env.expert_actions()
            k = len(env.actions)
            target = [(1 - smooth) * (a in best) / len(best) + smooth / k for a in env.actions]
            yield {"episode": seed + ep, "step": env.steps, "state": state(env, room),
                   "records": [{"key": "action", "question": q, "label": env.actions.index(env.expert()),
                                "target": target}]}
            made += 1
            env.step(rng.choice(env.actions) if rng.random() < noise else env.expert())
        env.close()
        ep += 1


# video --------------------------------------------------------------------------------------------------------
PANEL_W = 260
_CHOSEN, _OTHER = (255, 200, 80), (90, 90, 110)


def draw(wrist, room, env: ArmCamera, action, probs, label: str) -> np.ndarray:
    """A video frame: the wrist view (as the model sees it) and the room view side by side, and a panel with the
    step, return, marker quality and the moves (the chosen one highlighted, with the model's probabilities)."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (2 * SIZE + PANEL_W, SIZE + 20), (24, 24, 28))
    img.paste(wrist, (0, 20))
    img.paste(room, (SIZE, 20))
    d = ImageDraw.Draw(img)
    d.text((6, 4), "wrist camera", fill=(200, 200, 200))
    d.text((SIZE + 6, 4), "room camera", fill=(200, 200, 200))
    x = 2 * SIZE + 12
    d.text((x, 8), "ArmCamera (%s)" % label, fill=(230, 230, 230))
    d.text((x, 26), "step %d  return %.2f" % (env.steps, env.score), fill=(170, 170, 170))
    v = env.view
    d.text((x, 42), "quality %.2f%s" % (v["quality"], "  SUCCESS" if v["success"] else ""),
           fill=(120, 230, 120) if v["success"] else (170, 170, 170))
    for i, a in enumerate(env.actions):
        y = 66 + 24 * i
        chosen = a == action
        d.text((x, y), a, fill=_CHOSEN if chosen else (200, 200, 200))
        p = probs.get(a, 0.0) if probs else (1.0 if chosen else 0.0)
        d.rectangle((x + 100, y + 1, x + 100 + max(1, int(130 * p)), y + 18), fill=_CHOSEN if chosen else _OTHER)
    return np.asarray(img)


def record(env: ArmCamera, choose, label: str, out_path: str, fps: int = 3) -> Dict:
    """Play ``env`` to its end with ``choose(env) -> (action, probabilities or None)``, writing each wrist view the
    policy saw, the room view and the move to ``out_path`` (.webm: VP9). Needs ``imageio[ffmpeg]``."""
    import imageio.v2 as imageio

    codec = {"codec": "libvpx-vp9", "ffmpeg_params": ["-b:v", "0", "-crf", "32"]} if out_path.endswith(".webm") else {}
    actions = []
    with imageio.get_writer(out_path, fps=fps, macro_block_size=1, **codec) as out:
        while not env.done:
            wrist = env.render()
            action, probs = choose(env)
            out.append_data(draw(wrist, env.room(), env, action, probs, label))
            actions.append(action)
            env.step(action)
        end = draw(env.render(), env.room(), env, "", None, label + (" - found" if env.success else " - time"))
        for _ in range(fps):
            out.append_data(end)
    return {"actions": actions, "success": env.success, "steps": env.steps, "score": round(env.score, 3)}


def contact_sheet(out_path: str, seed: int = 200_000, poses=((8, 0, 0), (8, 1, 2), (8, 4, 1), (2, 1, 1),
                                                               (14, 1, 1), (0, 2, 3), (16, 3, 2), (8, 0, 3))) -> None:
    """The room view and the wrist view from several lattice poses, labelled, in one PNG."""
    from PIL import Image, ImageDraw

    env = ArmCamera(seed)
    tiles = [("room camera, marker on %s" % env.face, env.room())]
    for s in poses:
        env.q = lattice_solutions()[s].copy()
        env.state = s
        env._apply()
        env._frame = None
        v = env.marker()
        tiles.append(("az %d el %d r %.2f q %.2f" % (round(np.degrees(AZIMUTHS[s[0]])),
                                                     round(np.degrees(ELEVATIONS[s[1]])),
                                                     DISTANCES[s[2]], v["quality"]), Image.fromarray(env.frame())))
        tiles.append(("room view of that pose", env.room()))
    env.close()
    cols = 6
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * SIZE, rows * (SIZE + 18)), (20, 20, 24))
    d = ImageDraw.Draw(sheet)
    for i, (text, im) in enumerate(tiles):
        x, y = (i % cols) * SIZE, (i // cols) * (SIZE + 18)
        sheet.paste(im, (x, y + 18))
        d.text((x + 4, y + 3), text, fill=(220, 220, 220))
    sheet.save(out_path)


__all__ = ["ACTIONS", "JOINTS", "JOINT_LEVELS", "ArmCamera", "marker_view", "camera_target", "ik",
           "lattice_solutions", "question", "state", "model_chooser", "expert_policy", "random_policy",
           "still_policy", "play_episodes", "expert_frames", "record", "draw", "contact_sheet"]
