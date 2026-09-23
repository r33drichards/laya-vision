"""The Kaggle yoga converter (``laya.yoga``): no downloads."""
import random

import pytest

from laya.yoga import CONFUSABLE, POSES, pose_label, pose_options, unique_files, yoga_record


def test_pose_label():
    assert len(POSES) == 47
    assert pose_label("Parsva Virabhadrasana") == "Reverse Warrior (Parsva Virabhadrasana)"
    assert pose_label("Urdhva Mukha Svsnssana") == "Upward-Facing Dog (Urdhva Mukha Svanasana)"  # upstream typo fixed


def test_options_hold_the_pose_and_no_confusable():
    rng = random.Random(0)
    for folder in POSES:
        opts = pose_options(folder, 12, rng)
        assert len(opts) == len(set(opts)) == 12 and folder in opts
        assert not any(folder in s and o in s for s in CONFUSABLE for o in opts if o != folder)
    assert len(pose_options("Vrksasana", 0, rng)) == 47
    assert len(pose_options("Alanasana", 0, rng)) == 46


def test_yoga_record():
    rec = yoga_record("Parsva Virabhadrasana", "abc", random.Random(1), n_options=5)
    q = rec["question"]
    assert rec["id"] == "yoga-abc" and q["type"] == "choice" and len(q["criteria"]) == 5
    assert q["criteria"][rec["label"]] == "Reverse Warrior (Parsva Virabhadrasana)"
    assert yoga_record("Not A Pose", "x", random.Random(0)) is None


def test_yoga_record_loads_as_example():
    jsonl_example = pytest.importorskip("laya.vlm_train").jsonl_example
    rec = dict(yoga_record("Bakasana", "k", random.Random(0)), image="images/k.jpg")
    ex = jsonl_example(rec, "/data/vqa/yoga_poses", "yoga_poses")
    assert ex is not None and ex["label"] == rec["label"] and ex["state"]["image"].endswith("images/k.jpg")


def test_unique_files_drops_duplicates_and_cross_labelled():
    files = [("Bakasana", "a.png", b"1"), ("Bakasana", "b.png", b"1"),  # same pose twice: kept once
             ("Bitilasana", "c.png", b"2"), ("Marjaryasana", "d.png", b"2"),  # two poses: dropped
             ("Vrksasana", "e.png", b"3")]
    out = unique_files(files)
    assert sorted(v["name"] for v in out.values()) == ["a.png", "e.png"]
