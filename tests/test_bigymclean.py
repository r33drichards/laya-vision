"""CPU tests for the BiGym label cleaning and probe rebalancing (``laya.bigymclean``); no BiGym or Modal needed."""
import pytest

from laya import bigymclean as bc
from laya import bigymdata as bd
from laya.bigymgames import FROM_WORDS, PRIMITIVES
from laya.vlm_train import episode_step, jsonl_example, with_next_targets

P = list(PRIMITIVES)


def kept(seq):
    return [p for p, k in zip(seq, bc.clean_sequence(seq)["keep"]) if k]


def test_inverse_covers_every_reversible_primitive_and_only_those():
    pairs = {p: bc.inverse(p) for p in P}
    assert pairs["STAY"] is None
    for p, q in pairs.items():
        if q is not None:
            assert q in PRIMITIVES and bc.inverse(q) == p and q != p
    assert pairs["LEFT_HAND_FORWARD"] == "LEFT_HAND_BACK"
    assert pairs["RIGHT_HAND_UP"] == "RIGHT_HAND_DOWN"
    assert pairs["LEFT_TILT_LEFT"] == "LEFT_TILT_RIGHT"
    assert pairs["RIGHT_WRIST_CW"] == "RIGHT_WRIST_CCW"
    assert pairs["LEFT_GRIPPER_CLOSE"] == "LEFT_GRIPPER_OPEN"
    assert pairs["BASE_TURN_LEFT"] == "BASE_TURN_RIGHT"
    assert pairs["BASE_UP"] == "BASE_DOWN" and pairs["BASE_LEFT"] == "BASE_RIGHT"
    assert sum(q is not None for q in pairs.values()) == len(P) - 1  # every primitive but STAY


def test_pair_rule_drops_both_and_only_same_hand_reversals():
    assert kept(["BASE_FORWARD", "LEFT_HAND_UP", "LEFT_HAND_DOWN", "BASE_FORWARD"]) == ["BASE_FORWARD"] * 2
    # other hand, or a different axis, is not an undo
    seq = ["LEFT_HAND_UP", "RIGHT_HAND_DOWN", "LEFT_HAND_LEFT", "LEFT_HAND_UP", "LEFT_TILT_UP", "LEFT_TILT_DOWN"]
    assert kept(seq) == seq[:4]
    # undo separated by another move is kept (only t -> t+1 counts)
    seq = ["LEFT_HAND_UP", "BASE_FORWARD", "LEFT_HAND_DOWN"]
    assert kept(seq) == seq
    assert kept(["RIGHT_GRIPPER_CLOSE", "RIGHT_GRIPPER_OPEN"]) == []
    assert kept(["LEFT_WRIST_CW", "LEFT_WRIST_CCW", "BASE_UP"]) == ["BASE_UP"]


def test_chains_net_out():
    a, b = "BASE_LEFT", "BASE_RIGHT"
    assert kept([a, b, a, b]) == []  # even chain: nothing left
    assert bc.undo_drops(["X_UP", a, b, a, "X_UP"]) == [False, True, True, False, False]  # odd: last kept
    assert kept([a, b, a, b, a, "BASE_FORWARD"]) == [a, "BASE_FORWARD"]
    assert bc.undo_drops([a, a, b]) == [False, True, True]  # A A A': the first A survives
    assert kept([a, b, b, a]) == []  # two pairs back to back
    assert bc.undo_drops([a, b, b, a]) == [True, True, True, True]
    # no cascading: removing the inner pair does not make the outer one adjacent
    seq = ["LEFT_HAND_UP", a, b, "LEFT_HAND_DOWN"]
    assert kept(seq) == ["LEFT_HAND_UP", "LEFT_HAND_DOWN"]
    assert bc.clean_sequence([a, b, a])["undo"] == 2


def test_stay_runs_keep_the_first():
    seq = ["STAY", "STAY", "STAY", "BASE_UP", "STAY", "BASE_UP", "STAY", "STAY"]
    c = bc.clean_sequence(seq)
    assert [p for p, k in zip(seq, c["keep"]) if k] == ["STAY", "BASE_UP", "STAY", "BASE_UP", "STAY"]
    assert c["stay"] == 3 and c["undo"] == 0
    # STAY breaks an undo pair's adjacency
    assert kept(["BASE_UP", "STAY", "BASE_DOWN"]) == ["BASE_UP", "STAY", "BASE_DOWN"]


def test_clean_control_records_report_and_loading():
    s1 = ["BASE_FORWARD", "LEFT_HAND_UP", "LEFT_HAND_DOWN", "STAY", "STAY", "BASE_FORWARD"]
    s2 = ["BASE_LEFT", "BASE_RIGHT", "BASE_LEFT", "BASE_RIGHT", "RIGHT_HAND_UP"]
    for frames in (1, 4):
        recs = (bd.control_records("DrawerTopClose", 3, s1, frames)
                + bd.control_records("WallCupboardOpen", 9, s2, frames))
        out, rep = bc.clean_control(recs, P)
        assert [r["id"] for r in out] == ["DrawerTopClose-3-0", "DrawerTopClose-3-3", "DrawerTopClose-3-5",
                                          "WallCupboardOpen-9-4"]
        assert rep["before"] == 11 and rep["after"] == 4 and rep["undo"] == 6 and rep["stay"] == 1
        t = rep["per_task"]["DrawerTopClose"]
        assert (t["before"], t["after"], t["undo"], t["stay"]) == (6, 3, 2, 1)
        assert t["labels_after"] == {"BASE_FORWARD": 2, "STAY": 1}
        assert [f["episode"] for f in rep["flagged"]] == ["WallCupboardOpen-9", "DrawerTopClose-3"]
        assert rep["flagged"][0]["frac"] == 0.8
        # frames stay as recorded (the f4 window of decision 5 still shows the dropped decisions' frames)
        if frames == 4:
            assert out[2]["images"] == [bd.image_path("DrawerTopClose", 3, d) for d in (2, 3, 4, 5)]
        for r in out:  # still valid examples, labels decode to the recorded primitive, ids keep their decision
            ex = jsonl_example(r, "/root")
            assert FROM_WORDS[list(r["question"]["criteria"])[ex["label"]]] == r["primitive"]
            assert episode_step(r)[1] == int(r["id"].rsplit("-", 1)[1])
        # next_target only links records that were consecutive decisions
        nt = with_next_targets(out)
        assert [("next_target" in r) for r in nt] == [False, False, False, False]
    with pytest.raises(ValueError):
        bc.clean_control(bd.control_records("DrawerTopClose", 3, s1, 1)[1:], P)


def _probe(task, seed, decisions_done):
    probes = [{"decision": d, "labels": {"done": int(done), "progress": 3 if done else 1}}
              for d, done in decisions_done]
    return bd.probe_records(task, seed, probes)


def test_oversample_reaches_target_per_task_with_unique_ids():
    recs = (_probe("DrawerTopClose", 1, [(d, d == 99) for d in range(100)])  # 1 positive, 99 negative
            + _probe("DrawerTopOpen", 2, [(d, d >= 57) for d in range(60)]))  # 3 positive, 57 negative
    out, rep = bc.oversample(recs, target=0.15)
    ids = [r["id"] for r in out]
    assert len(ids) == len(set(ids))
    assert out[:len(recs)] == recs
    assert rep["DrawerTopClose"]["positive_after"] == round(0.15 * 99 / 0.85) == 17
    assert rep["DrawerTopOpen"]["positive_after"] == round(0.15 * 57 / 0.85) == 10
    for task, r in rep.items():
        done = [x for x in out if x["id"].startswith(task + "-") and "-done-" in x["id"]]
        assert sum(x["label"] for x in done) == r["positive_after"] and len(done) == r["records_after"]
        assert abs(r["frac_after"] - 0.15) < 0.02
    # copies spread evenly: DrawerTopOpen's 3 positives get 7 copies as 3, 2, 2
    dups = [x["id"] for x in out if "-dup" in x["id"] and x["id"].startswith("DrawerTopOpen")]
    assert sorted(dups) == sorted(["DrawerTopOpen-2-done-%d-dup%d" % (d, c) for d, n in ((57, 3), (58, 2), (59, 2))
                                   for c in range(1, n + 1)])
    # copies are identical records apart from the id, and do not parse as game frames
    orig = {x["id"]: x for x in recs}
    for x in out[len(recs):]:
        base = x["id"].rsplit("-dup", 1)[0]
        assert {k: v for k, v in x.items() if k != "id"} == {k: v for k, v in orig[base].items() if k != "id"}
        assert x["label"] == 1 and bc.probe_question(base)[1] == "done" and episode_step(x) is None
    # progress records untouched
    assert [x for x in out if "-progress-" in x["id"]] == [x for x in recs if "-progress-" in x["id"]]
    with_next_targets(out)  # no duplicate (episode, step) keys


def test_oversample_never_drops_and_handles_no_positives():
    recs = _probe("DrawerTopClose", 1, [(0, True), (1, True), (2, False)])  # already above target
    out, rep = bc.oversample(recs, target=0.15)
    assert out == recs and rep["DrawerTopClose"]["positive_after"] == 2
    recs = _probe("DrawerTopClose", 1, [(0, False), (1, False)])
    out, rep = bc.oversample(recs)
    assert out == recs and rep["DrawerTopClose"]["positive_after"] == 0
