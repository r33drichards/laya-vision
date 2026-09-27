"""CPU tests for the BiGym behaviour-cloning record builders (``laya.bigymdata``); no BiGym or Modal needed."""
from laya import bigymdata as bd
from laya.bigymgames import FROM_WORDS, PRIMITIVES, bigym_question, labels, probe_questions
from laya.vlm_train import episode_step, jsonl_example


def test_label_index_is_the_option_position():
    for task in ("DrawerTopClose", "WallCupboardOpen"):
        for frames in (1, 4):
            opts = list(bigym_question(task, frames)["action"]["criteria"])
            for p in PRIMITIVES:
                assert FROM_WORDS[opts[bd.label_index(p)]] == p


def test_frame_window_matches_bigymgame_frames_padding():
    # BiGymGame.frames(k): the last k of the frames seen so far (decisions 0..d), the first repeated to pad
    for k in (1, 2, 4):
        for d in range(8):
            hist = list(range(d + 1))[-k:]
            assert bd.frame_window(d, k) == [hist[0]] * (k - len(hist)) + hist
    assert bd.frame_window(0, 4) == [0, 0, 0, 0]
    assert bd.frame_window(2, 4) == [0, 0, 1, 2]
    assert bd.frame_window(9, 4) == [6, 7, 8, 9]


def test_control_records_ids_images_and_labels():
    prims = ["STAY", "LEFT_HAND_UP", "BASE_DOWN", "RIGHT_GRIPPER_CLOSE", "LEFT_TILT_LEFT"]
    f1 = bd.control_records("DrawerTopClose", 17, prims, 1)
    f4 = bd.control_records("DrawerTopClose", 17, prims, 4)
    assert [r["id"] for r in f1] == ["DrawerTopClose-17-%d" % d for d in range(len(prims))]
    assert [episode_step(r) for r in f1] == [("DrawerTopClose-17", d) for d in range(len(prims))]
    assert f1[3]["image"] == "images/DrawerTopClose-17-0003.jpg" and "images" not in f1[3]
    assert f4[1]["images"] == ["images/DrawerTopClose-17-%04d.jpg" % i for i in (0, 0, 0, 1)]
    assert "image" not in f4[1]
    for recs, frames in ((f1, 1), (f4, 4)):
        assert recs[0]["question"] == bigym_question("DrawerTopClose", frames)["action"]
        for r, p in zip(recs, prims):
            opts = list(r["question"]["criteria"])
            assert FROM_WORDS[opts[r["label"]]] == p == r["primitive"]
            ex = jsonl_example(r, "/root")
            assert ex["label"] == r["label"] and ex["target"][r["label"]] == 1.0


def test_probe_records_one_per_frame_and_question():
    task = "WallCupboardOpen"
    truths = [{"success": False, "open": 0.1}, {"success": True, "open": 0.95}]
    probes = [{"decision": 0, "labels": labels(task, truths[0])}, {"decision": 7, "labels": labels(task, truths[1])}]
    recs = bd.probe_records(task, 5, probes)
    qs = probe_questions(task)
    assert len(recs) == 2 * len(qs)
    assert len({r["id"] for r in recs}) == len(recs)
    assert {episode_step(r) for r in recs} == {("%s-5-%s" % (task, q), d) for q in qs for d in (0, 7)}
    by = {r["id"]: r for r in recs}
    assert by["%s-5-done-7" % task]["label"] == 1 and by["%s-5-done-0" % task]["label"] == 0
    assert by["%s-5-progress-7" % task]["label"] == 3 and by["%s-5-progress-0" % task]["label"] == 0
    assert by["%s-5-done-7" % task]["image"] == "images/%s-5-0007.jpg" % task
    for r in recs:
        assert jsonl_example(r, "/root") is not None


def test_probe_decisions_include_final_frame():
    assert bd.probe_decisions(7, 3) == [0, 3, 6, 7]
    assert bd.probe_decisions(6, 3) == [0, 3, 6]
    assert bd.probe_decisions(0, 3) == [0]


def test_val_split_by_demo_is_deterministic():
    seeds = list(range(100, 200))
    val = bd.val_seeds("DrawerTopOpen", seeds)
    assert len(val) == 10 and set(val) <= set(seeds)
    assert bd.val_seeds("DrawerTopOpen", reversed(seeds)) == val  # order does not matter
    assert bd.val_seeds("DrawerTopClose", seeds) != val  # per task
    assert len(bd.val_seeds("DrawerTopOpen", seeds[:50])) == 5
    assert bd.val_seeds("X", [3]) == [] and len(bd.val_seeds("X", [3, 4])) == 1
