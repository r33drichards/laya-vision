"""``laya.vlm_train.with_next_targets``: a recorded game frame's ``next_target`` is the target of step s + 1 of the
same episode (the Atari expert sets' ``episode`` / ``step`` fields, ViZDoom basic's ``<split>-<episode>-<step>``
ids), over the same options in the same order, and absent where there is no such successor. No model needed."""
import json

import pytest

from laya.vlm_train import episode_step, jsonl_example, load_jsonl_examples, with_next_targets

FREEWAY_Q = {"type": "choice", "instructions": "Play Freeway. Which action should the player take now?",
             "criteria": {"NOOP": "do nothing", "UP": "move up", "DOWN": "move down"}}
DOOM_Q = {"type": "choice", "instructions": "Play Doom. Which action should the player take now?",
          "criteria": {"MOVE_LEFT": "strafe left", "MOVE_RIGHT": "strafe right", "ATTACK": "shoot"}}


def atari(ep, step, target, question=FREEWAY_Q):
    rid = "expert-Freeway-e%06d-s%06d" % (ep, step)
    return {"id": rid, "image": "images/%s.png" % rid, "game": "Freeway", "actions": list(question["criteria"]),
            "label": max(range(len(target)), key=target.__getitem__), "target": target, "question": question,
            "source": "expert", "episode": ep, "step": step, "taken": 0}


def doom(rid, label):
    return {"id": rid, "image": "images/%s.jpg" % rid, "state_text": None, "question": DOOM_Q, "label": label}


def nt(recs):
    return {r["id"]: r.get("next_target") for r in with_next_targets(recs)}


def test_atari_successor_target_within_episode_and_none_at_episode_end():
    recs = [atari(2, 0, [0.1, 0.8, 0.1]), atari(2, 1, [0.0, 0.5, 0.5]), atari(2, 2, [0.2, 0.2, 0.6]),
            atari(3, 0, [1.0, 0.0, 0.0])]
    got = nt(recs)
    assert got["expert-Freeway-e000002-s000000"] == pytest.approx([0.0, 0.5, 0.5])
    assert got["expert-Freeway-e000002-s000001"] == pytest.approx([0.2, 0.2, 0.6])
    assert got["expert-Freeway-e000002-s000002"] is None  # episode 2 ends; episode 3's step 0 is not its successor
    assert got["expert-Freeway-e000003-s000000"] is None


def test_file_order_does_not_matter_and_records_are_copied():
    recs = [atari(5, 7, [0.0, 1.0, 0.0]), atari(4, 1, [0.0, 0.0, 1.0]), atari(5, 6, [1.0, 0.0, 0.0])]
    out = with_next_targets(recs)
    assert [r["id"] for r in out] == [r["id"] for r in recs]
    assert out[2]["next_target"] == [0.0, 1.0, 0.0]
    assert "next_target" not in out[0] and "next_target" not in out[1]
    assert all("next_target" not in r for r in recs)  # the inputs are not modified


def test_missing_successor_step_is_skipped_not_bridged():
    # the Atari recorder sometimes keeps every other step: step 4 -> 6 is a gap, so step 4 has no next_target
    got = nt([atari(0, 4, [1.0, 0.0, 0.0]), atari(0, 6, [0.0, 1.0, 0.0]), atari(0, 7, [0.0, 0.0, 1.0])])
    assert got["expert-Freeway-e000000-s000004"] is None
    assert got["expert-Freeway-e000000-s000006"] == [0.0, 0.0, 1.0]


def test_successor_target_is_normalised_like_jsonl_example_and_invalid_soft_falls_back_to_label():
    a, b = atari(1, 0, [1.0, 0.0, 0.0]), atari(1, 1, [2.0, 1.0, 1.0])
    assert nt([a, b])[a["id"]] == pytest.approx([0.5, 0.25, 0.25])
    c = dict(atari(1, 1, [0.0, 0.0, 1.0]), target=[0.5, 0.5])  # wrong length: jsonl_example uses the label
    c["label"] = 1
    assert nt([a, c])[a["id"]] == [0.0, 1.0, 0.0]
    assert nt([a, c])[a["id"]] == jsonl_example(c, "")["target"]


def test_options_must_match_in_the_same_order():
    swapped = dict(FREEWAY_Q, criteria={"UP": "move up", "NOOP": "do nothing", "DOWN": "move down"})
    got = nt([atari(0, 0, [1.0, 0.0, 0.0]), atari(0, 1, [0.0, 1.0, 0.0], question=swapped)])
    assert got["expert-Freeway-e000000-s000000"] is None  # same action set, different order: no target
    fewer = dict(FREEWAY_Q, criteria={"NOOP": "do nothing", "UP": "move up"})
    assert nt([atari(0, 0, [1.0, 0.0, 0.0]), atari(0, 1, [0.0, 1.0], question=fewer)])[
        "expert-Freeway-e000000-s000000"] is None
    # the next_target lines up with the frame's own target, option for option
    out = with_next_targets([atari(0, 0, [1.0, 0.0, 0.0]), atari(0, 1, [0.0, 0.25, 0.75])])
    ex = jsonl_example(out[0], "")
    assert len(ex["next_target"]) == len(ex["target"]) == 3 and ex["next_target"] == [0.0, 0.25, 0.75]


def test_doom_ids_encode_episode_and_step():
    assert episode_step(doom("train-000000-001", 0)) == ("train-000000", 1)
    assert episode_step(doom("train-002452-026", 0)) == ("train-002452", 26)
    assert episode_step(doom("val-000010-1000", 0)) == ("val-000010", 1000)  # %03d widens past 999
    assert episode_step({"id": "no-step-here"}) is None
    assert episode_step({"id": None}) is None
    # explicit fields win over the id (Atari ids also end in digits)
    assert episode_step(atari(3, 12, [1.0, 0.0, 0.0])) == (3, 12)


def test_doom_next_target_is_the_successors_one_hot_label():
    recs = [doom("train-000000-000", 0), doom("train-000000-001", 2), doom("train-000001-002", 1),
            doom("train-000001-003", 1), doom("train-000001-005", 0)]
    got = nt(recs)
    assert got["train-000000-000"] == [0.0, 0.0, 1.0]
    assert got["train-000000-001"] is None               # last recorded frame of episode 0
    assert got["train-000001-002"] == [0.0, 1.0, 0.0]
    assert got["train-000001-003"] is None               # step 4 not recorded (no monster in view)
    assert got["train-000001-005"] is None


def test_unplaceable_and_duplicate_records():
    assert nt([{"id": "x", "question": DOOM_Q, "label": 0}]) == {"x": None}
    with pytest.raises(ValueError, match="duplicate"):
        with_next_targets([doom("train-000000-001", 0), doom("train-000000-001", 1)])


def test_load_jsonl_examples_next_targets_flag(tmp_path):
    recs = [doom("train-000000-000", 0), doom("train-000000-001", 2), doom("train-000000-002", 1)]
    (tmp_path / "doom").mkdir()
    (tmp_path / "doom" / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    plain = load_jsonl_examples(str(tmp_path), "doom", "train")
    assert all("next_target" not in ex for ex in plain)
    with_nt = load_jsonl_examples(str(tmp_path), "doom", "train", next_targets=True)
    assert [ex["next_target"] if "next_target" in ex else None for ex in with_nt] == [[0.0, 0.0, 1.0],
                                                                                        [0.0, 1.0, 0.0], None]
    # otherwise identical to the plain examples
    assert [{k: v for k, v in ex.items() if k != "next_target"} for ex in with_nt] == plain
    # computed over the whole file before ``limit``: the last kept frame still gets its successor's target
    limited = load_jsonl_examples(str(tmp_path), "doom", "train", limit=2, next_targets=True)
    assert len(limited) == 2 and limited[1]["next_target"] == [0.0, 1.0, 0.0]
