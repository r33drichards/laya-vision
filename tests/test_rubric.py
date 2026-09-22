"""The rubric-dataset converters (``laya.rubric``): no downloads, rows are shaped like the dataset viewer shows them."""
import random

import pytest

from laya.common import render_options
from laya.rubric import (CRITERIA, INSTRUCTIONS, ava_record, balance_levels, clip_text, collapse_counts,
                         crisismmd_record, level_counts, richhf_level, richhf_records, rubric_question,
                         vlfeedback_records)
from laya.vlm_train import jsonl_example


def test_rubrics_are_short_and_ordered():
    for aspect, crit in CRITERIA.items():
        assert 3 <= len(crit) <= 5 and len(set(crit)) == len(crit)
        assert all(len(c.split()) <= 14 for c in crit), aspect  # options are cut to 48 tokens each
        assert all(len(INSTRUCTIONS[k]) >= 2 for k in INSTRUCTIONS)
    q = rubric_question("damage")
    assert q == {"type": "score", "instructions": INSTRUCTIONS["damage"][0], "criteria": CRITERIA["damage"]}
    assert rubric_question("helpfulness", random.Random(0))["instructions"] in INSTRUCTIONS["helpfulness"]
    assert rubric_question("aesthetics", None, "aesthetics_generated")["instructions"] == INSTRUCTIONS["aesthetics_generated"][0]


def test_clip_text():
    assert clip_text("  a  b\n\n\n\nc ", 100) == "a b\n\nc"
    long = " ".join(["word"] * 100)
    cut = clip_text(long, 50)
    assert len(cut) <= 54 and cut.endswith(" ...") and not cut[:-4].endswith(" ")


def test_vlfeedback_struct_of_lists():
    row = {"id": "LRV-1", "prompt": "Are there stop signs?",
           "completions": {"model": ["a", "b"], "response": ["No.", "Yes, two."],
                           "annotations": [{"Helpfulness": {"Rating": "2", "Rationale": "..."},
                                            "Ethical Considerations": {"Rating": "5", "Rationale": "..."},
                                            "Visual Faithfulness": {"Rating": "4", "Rationale": "..."}},
                                           {"Helpfulness": {"Rating": "N/A"}, "Visual Faithfulness": {"Rating": "1"}}]}}
    recs = vlfeedback_records(row, "vlf-0", max_texts=10)
    assert [r["id"] for r in recs] == ["vlf-0-0-helpfulness", "vlf-0-0-faithfulness", "vlf-0-1-faithfulness"]
    assert [r["label"] for r in recs] == [1, 3, 0]  # ethics dropped, N/A skipped
    assert recs[0]["state_text"] == "Question: Are there stop signs?\n\nResponse: No."
    assert recs[0]["question"]["criteria"] == CRITERIA["helpfulness"] and recs[0]["source"] == "VLFeedback/a"
    two = vlfeedback_records(row, "vlf-0", rng=random.Random(1), max_texts=2)
    assert len(two) == 2 and all(r["question"]["instructions"] in INSTRUCTIONS[r["id"].rsplit("-", 1)[1]] for r in two)
    # list-of-structs export, and a long response is clipped
    row2 = {"prompt": "q", "completions": [{"model": "m", "response": "x " * 2000, "annotations": {"Helpfulness": {"Rating": "5"}}}]}
    recs = vlfeedback_records(row2, "r", max_chars=100)
    assert len(recs) == 1 and recs[0]["label"] == 4 and len(recs[0]["state_text"]) < 130
    assert vlfeedback_records({"prompt": "", "completions": []}, "e") == []


def test_ava_soft_target():
    assert collapse_counts([1, 2, 7, 34, 53, 50, 28, 19, 7, 7]) == [3, 41, 103, 47, 14]
    rec = ava_record({"rating_counts": [1, 2, 7, 34, 53, 50, 28, 19, 7, 7]}, "ava-1")
    assert rec["label"] == 2 and rec["target"] == pytest.approx([3 / 208, 41 / 208, 103 / 208, 47 / 208, 14 / 208], abs=1e-4)
    assert rec["question"]["criteria"] == CRITERIA["aesthetics"] and rec["state_text"] is None
    assert ava_record({"rating_counts": [0] * 10}, "x") is None and ava_record({"rating_counts": [1] * 8}, "x") is None
    tie = ava_record({"rating_counts": [0, 0, 5, 5, 0, 0, 5, 5, 0, 0]}, "t")
    assert tie["label"] == 1  # ties go to the lower level
    with pytest.raises(ValueError):
        collapse_counts([1, 2, 3], 2)


def test_richhf_levels_and_state():
    assert [richhf_level(s) for s in (0.0, 1 / 12, 0.5, 7 / 12, 0.75, 1.0)] == [0, 0, 2, 2, 3, 4]
    assert richhf_level(None) is None and richhf_level(1.5) is None
    row = {"caption": "a Ferrari made of wood", "aesthetics_score": 2 / 3, "artifact_score": 7 / 12,
           "misalignment_score": 7 / 12, "overall_score": 0.5}
    recs = richhf_records(row, "rh-0", max_texts=10)
    assert [r["id"].rsplit("-", 1)[1] for r in recs] == ["aesthetics", "plausibility", "alignment", "overall"]
    assert [r["label"] for r in recs] == [3, 2, 2, 2]
    assert recs[0]["state_text"] is None and recs[2]["state_text"] == "Prompt: a Ferrari made of wood"
    assert recs[0]["question"]["instructions"] == INSTRUCTIONS["aesthetics_generated"][0]
    assert len(richhf_records(row, "rh-0", rng=random.Random(0), max_texts=2)) == 2
    no_caption = richhf_records(dict(row, caption=""), "rh-1", max_texts=10)
    assert [r["id"].rsplit("-", 1)[1] for r in no_caption] == ["aesthetics", "plausibility"]


def test_crisismmd():
    rec = crisismmd_record({"label": 2, "event_name": "hurricane_harvey", "tweet_text": "RT ..."}, "c-0")
    assert rec["label"] == 2 and rec["state_text"] is None and rec["question"]["criteria"] == CRITERIA["damage"]
    assert crisismmd_record({"label": "mild_damage"}, "c-1")["label"] == 1
    assert crisismmd_record({"label": "unknown"}, "c-2") is None and crisismmd_record({"label": 3}, "c-3") is None


def test_balance_levels():
    recs = [{"label": 4}] * 100 + [{"label": 3}] * 40 + [{"label": 2}] * 10 + [{"label": 0}] * 4
    out = balance_levels(recs, max_ratio=2.0, floor=5)
    # median level count is 10 (sorted counts 4, 10, 40, 100 -> index 2 is 40; cap = 80)
    assert level_counts(out) == {4: 80, 3: 40, 2: 10, 0: 4}
    assert level_counts(balance_levels(recs, max_ratio=0)) == level_counts(recs)
    out = balance_levels(recs, max_ratio=1.0, rng=random.Random(0), floor=5)
    assert level_counts(out) == {4: 40, 3: 40, 2: 10, 0: 4}
    assert level_counts(balance_levels(recs, max_ratio=0.1, floor=50)) == {4: 50, 3: 40, 2: 10, 0: 4}


def test_records_load_as_training_examples(tmp_path):
    """The records go through ``jsonl_example`` like any prepared dataset: k options, soft target kept."""
    rec = ava_record({"rating_counts": [1, 2, 7, 34, 53, 50, 28, 19, 7, 7]}, "ava-1")
    rec["image"] = "images/ava-1.jpg"
    ex = jsonl_example(rec, str(tmp_path), "score_ava")
    assert ex["q"]["t"] == "score" and len(render_options(ex["q"])) == 5
    assert ex["target"] == pytest.approx(rec["target"], abs=1e-3) and ex["label"] == 2
    assert render_options(ex["q"])[0].startswith("level 0: very poor")
    assert ex["state"]["image"].endswith("images/ava-1.jpg")
    vrec = vlfeedback_records({"prompt": "q", "completions": [{"response": "r", "annotations": {"Helpfulness": {"Rating": "3"}}}]}, "v")[0]
    vrec["image"] = "images/v.jpg"
    ex = jsonl_example(vrec, str(tmp_path), "score_vlfeedback")
    assert ex["label"] == 2 and ex["target"] == [0, 0, 1, 0, 0] and ex["state"]["context"].startswith("Question: q")
