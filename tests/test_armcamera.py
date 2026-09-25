"""Arm camera (``laya.armcamera``): the scene builds, every lattice pose is reachable, moves do what they say,
seeding, the marker metric agrees with segmentation rendering, the expert beats random and still, rendering, no
model."""
import os
import sys

import numpy as np
import pytest

if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")  # mujoco picks its GL backend when first imported
pytest.importorskip("mujoco")

from laya import armcamera as ac  # noqa: E402

SEED = 200_000


def test_every_lattice_pose_is_reachable_by_the_ik():
    import mujoco

    sol = ac.lattice_solutions()
    assert len(sol) == len(ac.AZIMUTHS) * len(ac.ELEVATIONS) * len(ac.DISTANCES)
    m = ac._model()
    d = mujoco.MjData(m)
    cam = m.camera("wrist").id
    for s, q in sol.items():
        d.qpos[:] = q
        mujoco.mj_forward(m, d)
        pos, mat = ac.camera_target(s)
        assert np.linalg.norm(d.cam_xpos[cam] - pos) < 1e-3
        assert np.abs(d.cam_xmat[cam].reshape(3, 3) - mat).max() < 1e-2  # looking at the object, horizon level


def _pose(env):
    pos, mat = env.camera_pose()
    return pos, mat[:, 0], np.linalg.norm(pos - ac.OBJ_CENTER)


def test_moves_go_where_they_are_named():
    env = ac.ArmCamera(SEED)
    env.max_steps = 1000
    env.state = (8, 2, 1)  # middle of the lattice, room to move every way
    env.q = ac.lattice_solutions()[env.state].copy()
    env._apply()
    for action, check in [
        ("ORBIT_RIGHT", lambda p0, r0, d0, p1, d1: (p1 - p0) @ r0 > 0.02 and abs(d1 - d0) < 1e-3),
        ("ORBIT_LEFT", lambda p0, r0, d0, p1, d1: (p1 - p0) @ r0 < -0.02 and abs(d1 - d0) < 1e-3),
        ("MOVE_UP", lambda p0, r0, d0, p1, d1: p1[2] > p0[2] + 0.02 and abs(d1 - d0) < 1e-3),
        ("MOVE_DOWN", lambda p0, r0, d0, p1, d1: p1[2] < p0[2] - 0.02),
        ("CLOSER", lambda p0, r0, d0, p1, d1: d1 < d0 - 0.02),
        ("FARTHER", lambda p0, r0, d0, p1, d1: d1 > d0 + 0.02),
        ("NONE", lambda p0, r0, d0, p1, d1: np.allclose(p0, p1)),
    ]:
        env.terminated = False
        p0, r0, d0 = _pose(env)
        env.step(action)
        p1, _, d1 = _pose(env)
        assert check(p0, r0, d0, p1, d1), action
        _, mat = env.camera_pose()  # still looking at the object's centre
        fwd = -mat[:, 2]
        assert np.allclose(fwd, (ac.OBJ_CENTER - p1) / np.linalg.norm(ac.OBJ_CENTER - p1), atol=1e-2)
    assert env.blocked == 0
    env.state = (0, 0, 0)  # at the lattice's corner a move outward leaves the camera where it is
    assert env.next_state("ORBIT_LEFT") == env.state and env.next_state("MOVE_DOWN") == env.state
    env.close()


def test_joint_levels_move_the_joints():
    env = ac.ArmCamera(SEED, control="joints")
    q0 = env.q.copy()
    env.step({j: ("POS" if j == "elbow" else "NONE") for j in ac.JOINTS})
    assert env.q[2] == pytest.approx(q0[2] + ac.JOINT_STEP) and np.allclose(np.delete(env.q, 2), np.delete(q0, 2))
    env.close()


def test_episodes_are_seeded():
    a = ac.play_episodes(ac.random_policy(1), 3, seed=5)
    b = ac.play_episodes(ac.random_policy(1), 3, seed=5)
    assert a["results"] == b["results"]
    envs = [ac.ArmCamera(s) for s in range(12)]
    assert len({(round(e.yaw, 6), e.face) for e in envs}) == 12  # seeds differ in yaw and marker face
    assert len({e.face for e in envs}) >= 3
    for e in envs:
        assert e.moves_to_success >= ac.MIN_START_MOVES and not e.success and e._goals
        e.close()


def test_expert_beats_random_and_still():
    kw = {"episodes": 8, "seed": SEED}
    exp = ac.play_episodes(ac.expert_policy, **kw)
    rnd = ac.play_episodes(ac.random_policy(0), **kw)
    still = ac.play_episodes(ac.still_policy, **kw)
    assert exp["success_rate"] == 1.0 and exp["mean_steps"] < 15
    assert exp["mean_score"] > rnd["mean_score"] and exp["mean_score"] > still["mean_score"]
    assert rnd["success_rate"] <= 0.5 and still["success_rate"] == 0.0
    # the expert walks a shortest path: its steps are the planner's distance from the start
    assert [e["steps"] for e in exp["results"]] == [ac.ArmCamera(SEED + i).moves_to_success for i in range(8)]
    assert sum(e["blocked"] for e in exp["results"]) == 0


def test_marker_metric_matches_segmentation():
    env = ac.ArmCamera(SEED)
    seen = unseen = 0
    for s in ac.lattice_states()[::7]:
        env.state, env.q = s, ac.lattice_solutions()[s].copy()
        env._apply()
        v, px = env.marker(), env.marker_pixels()
        if not v["visible"]:
            assert px < 100, (s, px)  # face turned away or centre out of frame: at most a sliver at an edge
            unseen += 1
        elif v["cos"] > 0.3 and v["off"] < 0.8:
            # the disc covers about pi * size^2 pixels (size already folds in the foreshortening)
            expect = np.pi * v["size_px"] ** 2
            assert 0.5 * expect < px < 1.6 * expect, (s, px, expect)
            seen += 1
    assert seen >= 5 and unseen >= 5
    env.close()


def test_marker_view_basics():
    pos, mat = ac.camera_target((8, 0, 3))  # straight at the +x side, close
    face_on = ac.marker_view(pos, mat, ac.OBJ_CENTER + [ac.HALF, 0, 0], np.array([1.0, 0, 0]))
    assert face_on["visible"] and face_on["success"] and face_on["off"] < 1e-6 and face_on["cos"] > 0.999
    away = ac.marker_view(pos, mat, ac.OBJ_CENTER - [ac.HALF, 0, 0], np.array([-1.0, 0, 0]))
    assert not away["visible"] and away["quality"] == 0.0
    far = ac.marker_view(*ac.camera_target((8, 0, 0)), ac.OBJ_CENTER + [ac.HALF, 0, 0], np.array([1.0, 0, 0]))
    assert far["visible"] and not far["success"] and far["size_px"] < face_on["size_px"]


def test_render_shapes_and_ghosting():
    env = ac.ArmCamera(SEED)
    first = np.asarray(env.render())
    assert first.shape == (256, 256, 3) and first.dtype == np.uint8
    assert np.asarray(env.room()).shape == (256, 256, 3)
    env.step("ORBIT_LEFT")
    raw = env.frame()
    ghosted = np.asarray(env.render())
    assert ghosted.shape == raw.shape and (ghosted != raw).any()
    st = ac.state(env, room=True)
    assert len(st["images"]) == 2
    env.close()


def test_questions_offer_the_moves():
    from laya.vlm import VLMAgent

    for room in (False, True):
        internal = VLMAgent._to_internal(ac.question(room)["action"])
        assert internal["t"] == "choice" and tuple(internal["crit"]) == tuple(ac.ACTIONS)


def test_expert_frames_label_shortest_path_moves():
    frames = list(ac.expert_frames(6, seed=SEED))
    assert len(frames) == 6
    for f in frames:
        (rec,) = f["records"]
        assert abs(sum(rec["target"]) - 1) < 1e-6 and rec["target"][rec["label"]] == max(rec["target"])
        assert np.asarray(f["state"]["image"]).shape == (256, 256, 3)
