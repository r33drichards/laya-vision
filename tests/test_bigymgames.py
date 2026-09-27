"""BiGym (``laya.bigymgames``): question schema and labels without the sim; with it, IK primitives move the wrist,
the reach oracle beats random, probe frames carry consistent ground truth, and the cupboard poses are read back."""
import importlib.util

import numpy as np
import pytest

from laya import bigymgames as bg
from laya.common import QTYPES

HAVE_SIM = all(importlib.util.find_spec(m) for m in ("mujoco", "bigym"))
sim = pytest.mark.skipif(not HAVE_SIM, reason="needs mujoco and bigym (and MUJOCO_GL=egl or osmesa)")


@pytest.mark.parametrize("task", sorted(bg.TASKS))
def test_questions_are_well_formed(task):
    q = bg.bigym_question(task)["action"]
    assert q["type"] == "choice" and list(q["criteria"]) == [bg.OPTION_WORDS[p] for p in bg.PRIMITIVES]
    assert bg.TASKS[task]["description"] in q["instructions"]
    probe = bg.probe_questions(task)
    assert {v["type"] for v in probe.values()} <= set(QTYPES)
    assert ("side" in probe) == (bg.TASKS[task]["kind"] == "reach")
    assert len(probe["progress"]["criteria"]) == 4


def test_progress_levels():
    assert [bg.progress_level("ReachTarget", d) for d in (0.5, 0.25, 0.15, 0.05)] == [0, 1, 2, 3]
    assert [bg.progress_level("DrawerTopOpen", f) for f in (0.0, 0.4, 0.7, 0.95)] == [0, 1, 2, 3]
    assert [bg.progress_level("DrawerTopClose", f) for f in (1.0, 0.6, 0.3, 0.05)] == [0, 1, 2, 3]
    lab = bg.labels("ReachTarget", {"success": True, "distance": 0.05, "closer": "right"})
    assert lab == {"done": 1, "progress": 3, "side": 1}


def test_probe_metrics_and_answer_probs():
    rows = [{"qid": "done", "label": 1, "probs": [0.2, 0.8]}, {"qid": "done", "label": 0, "probs": [0.9, 0.1]},
            {"qid": "done", "label": 0, "probs": [0.4, 0.6]}]
    m = bg.probe_metrics(rows)["done"]
    assert m["n"] == 3 and m["acc"] == pytest.approx(2 / 3) and m["prior_acc"] == pytest.approx(2 / 3)
    assert m["label_counts"] == [2, 1] and m["pred_counts"] == [1, 2] and m["auroc"] == 1.0
    lev = [{"qid": "progress", "label": i, "probs": list(np.eye(4)[i] * 0.7 + 0.075)} for i in range(4)]
    m = bg.probe_metrics(lev)["progress"]
    assert m["acc"] == 1.0 and m["spearman"] == pytest.approx(1.0) and m["mae"] < m["prior_mae"]
    assert bg.answer_probs({"type": "noul", "noul": 0.7}, {"type": "noul"}) == pytest.approx([0.3, 0.7])
    q = {"type": "choice", "criteria": {"left": "", "right": ""}}
    assert bg.answer_probs({"type": "choice", "probabilities": {"right": 0.9, "left": 0.1}}, q) == [0.1, 0.9]
    assert bg.normalized(0.5, 0.0, 1.0) == 0.5 and bg.normalized(0.3, 0.2, 0.2) is None


@sim
def test_primitives_stay_in_bounds_and_move_the_wrist():
    game = bg.BiGymGame("ReachTarget", seed=0)
    space = game.env.action_space
    for name in bg.PRIMITIVES:
        a = game.primitive_to_action(name)
        assert a.shape == space.shape and np.all(a >= space.low) and np.all(a <= space.high)
    for hand in ("left", "right"):
        for axis in ("FORWARD", "UP", "LEFT"):
            before = game.hand_pos(hand).copy()
            game.step("%s_HAND_%s" % (hand.upper(), axis))
            moved = game.hand_pos(hand) - before
            assert moved @ game.world_dir(axis) > 0.01, (hand, axis, moved)
    assert game.frame().shape == bg.RESOLUTION + (3,) and game.decisions == 6 and game.steps == 6 * bg.HOLD
    game.close()


@sim
def test_oracle_reaches_and_random_does_not():
    orc = bg.play_episodes("ReachTarget", bg.oracle_policy, episodes=3)
    rnd = bg.play_episodes("ReachTarget", bg.random_policy(0), episodes=3)
    assert orc["success_rate"] == 1.0 > rnd["success_rate"]
    assert all(e["final"]["distance"] <= 0.1 for e in orc["results"])
    again = bg.play_episodes("ReachTarget", bg.random_policy(0), episodes=3)
    assert [e["final"] for e in again["results"]] == [e["final"] for e in rnd["results"]]


@sim
def test_probe_frames_ground_truth():
    rows = bg.probe_frames("ReachTarget", 8)
    assert len(rows) == 8 and any(r["labels"]["done"] for r in rows)
    for r in rows:
        assert r["labels"] == bg.labels("ReachTarget", r["truth"]) and r["image"].dtype == np.uint8
        assert r["labels"]["done"] == int(r["truth"]["distance"] <= 0.1)
    rows = bg.probe_frames("DrawerTopClose", 8)
    assert [r["labels"]["progress"] for r in rows] == [0, 1, 2, 3] * 2
    assert all(r["labels"]["done"] == int(r["truth"]["open"] <= 0.1) for r in rows)


@sim
def test_open_fraction_round_trip():
    game = bg.BiGymGame("WallCupboardOpen", seed=0)
    assert game.ground_truth() == {"success": False, "open": pytest.approx(0.0, abs=0.02)}
    game.set_open_fraction(1.0)
    assert game.ground_truth()["success"] and game.open_fraction() == pytest.approx(1.0, abs=0.02)
    game.close()


def test_multi_frame_question():
    one, four = bg.bigym_question("DrawerTopClose")["action"], bg.bigym_question("DrawerTopClose", 4)["action"]
    assert "last 4 views 0.1 s apart, oldest first" in four["instructions"] and "views" not in one["instructions"]
    assert four["criteria"] == one["criteria"]
    assert [bg.FROM_WORDS[w] for w in one["criteria"]] == list(bg.PRIMITIVES)  # phrases map back, in order


@pytest.mark.parametrize("frames", [1, 4])
def test_questions_fit_the_budgets_whole(frames):
    """37 options share head_max_len with the instructions: nothing (least of all the task) may be cut."""
    transformers = pytest.importorskip("transformers")
    from laya.vlm import VLMAgent, build_vlm_inputs

    try:
        proc = transformers.AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct")
    except Exception as e:  # offline without a cached processor
        pytest.skip("SmolVLM processor unavailable: %r" % e)
    img = np.zeros((256, 256, 3), np.uint8)
    for task in bg.TASKS:
        qs = dict(bg.probe_questions(task), action=bg.bigym_question(task, frames)["action"])
        for qid, q in qs.items():
            it = build_vlm_inputs(proc, {"images": [img] * frames}, VLMAgent._to_internal(q), 1024, 256)
            tr = it["truncation"]
            assert not tr["options"] and not tr["instructions_tokens_dropped"], (task, qid, tr)


@sim
def test_frame_history_pads_then_rolls():
    game = bg.BiGymGame("ReachTarget", seed=0)
    first = game.frames(4)
    assert len(first) == 4 and all(f is first[0] for f in first)  # episode start: the first frame repeated
    for _ in range(5):
        game.step("LEFT_HAND_UP")
    fr = game.frames(4)
    assert len(fr) == 4 and fr[-1] is game.frame() and not np.array_equal(fr[0], fr[-1])
    assert game.frames(2) == fr[-2:]
    with pytest.raises(ValueError):
        game.frames(bg.MAX_FRAMES + 1)
    game.close()


@sim
def test_wrist_roll_turns_only_the_wrist():
    game = bg.BiGymGame("ReachTarget", seed=0)
    m, d = game._model, game._data
    qadr = {h: m.jnt_qposadr[m.dof_jntid[game._arm[h][1][-1]]] for h in ("left", "right")}
    for _ in range(3):
        game.step("STAY")  # let the reset pose settle
    before = {h: (float(d.qpos[qadr[h]]), game.hand_pos(h).copy()) for h in ("left", "right")}
    for _ in range(4):
        game.step("LEFT_WRIST_CW")
    turned = float(d.qpos[qadr["left"]]) - before["left"][0]
    assert turned == pytest.approx(4 * bg.WRIST_ROLL, abs=0.05)
    assert abs(float(d.qpos[qadr["right"]]) - before["right"][0]) < 0.02
    assert np.linalg.norm(game.hand_pos("left") - before["left"][1]) < 0.005  # the roll does not move the wrist point
    for _ in range(12):
        game.step("LEFT_WRIST_CW")
    assert float(d.qpos[qadr["left"]]) <= m.jnt_range[m.dof_jntid[game._arm["left"][1][-1]]][1] + 1e-3
    game.close()


@sim
def test_tilt_aims_the_gripper():
    game = bg.BiGymGame("ReachTarget", seed=0, env=bg.make_env("ReachTarget", cameras=False))
    for _ in range(3):
        game.step("STAY")
    for name, axis, sign in (("LEFT_TILT_UP", 2, 1), ("LEFT_TILT_DOWN", 2, -1), ("RIGHT_TILT_LEFT", 1, 1),
                             ("RIGHT_TILT_RIGHT", 1, -1)):
        hand = name.split("_")[0].lower()
        x0 = game.pointing(hand).copy()
        game.step(name)
        x1 = game.pointing(hand)
        turned = np.degrees(np.arccos(np.clip(x0 @ x1, -1, 1)))
        assert turned == pytest.approx(np.degrees(bg.TILT_STEP), abs=4), (name, turned)
        assert sign * (x1[axis] - x0[axis]) > 0.1, (name, x0, x1)  # it turned the named way
    game.close()


@sim
def test_restore_is_exact():
    """Trying moves and undoing them (the demo follower's lookahead) leaves the episode bit-identical to one played
    without them, so recorded labels replay to the same outcome."""
    import random

    rng = random.Random(0)
    seq = [rng.choice(list(bg.PRIMITIVES)) for _ in range(12)]
    env = bg.make_env("DrawerTopOpen", cameras=False)

    def play(probe):
        game, out = bg.BiGymGame("DrawerTopOpen", seed=3, env=env), []
        for p in seq:
            if probe:
                snap = game.snapshot()
                for q in ("LEFT_HAND_FORWARD", "BASE_BACK", "RIGHT_GRIPPER_CLOSE"):
                    game.step(q)
                    game.restore(snap)
            game.step(p)
            out.append(np.concatenate([game._data.qpos, game._data.qvel, game._data.ctrl]).copy())
        return np.array(out)

    assert np.array_equal(play(False), play(True))
    env.close()


@sim
def test_targets_do_not_wind_up_against_contact():
    """A hand pushed into the cabinet keeps its arm and base targets within ARM_LEAD / BASE_LEAD of the joints (the
    env integrates delta targets, which ran 22 cm ahead of the pelvis before)."""
    game = bg.BiGymGame("WallCupboardOpen", seed=0, env=bg.make_env("WallCupboardOpen", cameras=False))
    m, d = game._model, game._data
    for _ in range(40):
        game.step("BASE_FORWARD")
        game.step("LEFT_HAND_FORWARD")
    arm, base = np.array(game._acts), np.array(game._base_acts)
    step = max(bg.BASE_STEP, bg.WRIST_STEP)
    assert np.all(np.abs(d.ctrl[base[:2]] - game._joint_qpos(base[:2])) < bg.BASE_LEAD + step)
    assert np.all(np.abs(d.ctrl[arm] - game._joint_qpos(arm)) < bg.ARM_LEAD + 0.5)
    game.close()
