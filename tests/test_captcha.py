"""Tests for the Open CaptchaWorld -> typed-decision conversion. No model download, no benchmark checkout."""
import json
import os
import sys

import pytest
from PIL import Image

from laya.captcha import EXCLUDED, KWAY_SPECS, Decision, Puzzle, _cells, load_puzzles

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples"))
from captcha_eval import auc, chance  # noqa: E402

import numpy as np  # noqa: E402


# ---------------------------------------------------------------------------------------------------------
# Grid indexing: must match static/js/script.js, which lays cells out row-major over grid_size = [rows, cols]
# ---------------------------------------------------------------------------------------------------------


def test_cells_are_row_major(tmp_path):
    p = tmp_path / "grid.png"
    Image.new("RGB", (300, 200)).save(p)
    boxes = _cells(str(p), rows=2, cols=3)
    assert len(boxes) == 6
    assert boxes[0] == (0, 0, 100, 100)      # row 0, col 0
    assert boxes[2] == (200, 0, 300, 100)    # row 0, col 2 -- across before down
    assert boxes[3] == (0, 100, 100, 200)    # row 1, col 0
    assert boxes[5] == (200, 100, 300, 200)


def test_cells_tile_the_image_exactly(tmp_path):
    p = tmp_path / "g.png"
    Image.new("RGB", (877, 439)).save(p)  # deliberately not divisible
    boxes = _cells(str(p), rows=5, cols=5)
    assert boxes[0][:2] == (0, 0)
    assert boxes[-1][2:] == (877, 439)
    for i in range(5):  # no gaps or overlaps along a row
        row = boxes[i * 5:(i + 1) * 5]
        for a, b in zip(row, row[1:]):
            assert a[2] == b[0]


# ---------------------------------------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------------------------------------


def _noul(p):
    return {"type": "noul", "noul": p, "confidence": max(p, 1 - p)}


def _puzzle(mode, truth, n, probs):
    decs = [Decision(qid="d%d" % i, images=[], question={}, truth=False, crops=[]) for i in range(n)]
    pz = Puzzle("T", "p", "prompt", mode, decs, truth, n)
    return pz, {"d%d" % i: _noul(v) for i, v in enumerate(probs)}


def test_argmax_passes_on_highest_probability():
    pz, ans = _puzzle("argmax", 2, 4, [0.1, 0.3, 0.9, 0.2])
    assert pz.grade(ans)[0] is True
    pz, ans = _puzzle("argmax", 2, 4, [0.1, 0.95, 0.9, 0.2])
    assert pz.grade(ans)[0] is False


def test_argmax_accepts_any_of_several_valid_answers():
    """Bingo allows more than one winning swap."""
    pz, ans = _puzzle("argmax", {1, 3}, 4, [0.1, 0.2, 0.3, 0.9])
    assert pz.grade(ans)[0] is True


def test_subset_requires_exact_set_match():
    pz, ans = _puzzle("subset", {0, 2}, 4, [0.9, 0.1, 0.8, 0.2])
    assert pz.grade(ans)[0] is True
    pz, ans = _puzzle("subset", {0, 2}, 4, [0.9, 0.6, 0.8, 0.2])  # one extra cell selected
    assert pz.grade(ans)[0] is False
    pz, ans = _puzzle("subset", {0, 2}, 4, [0.9, 0.1, 0.4, 0.2])  # one cell missed
    assert pz.grade(ans)[0] is False


def test_choice_mode_compares_the_chosen_key():
    d = Decision(qid="sum", images=[], question={}, truth="42", crops=[])
    pz = Puzzle("Dice_Count", "p", "", "choice", [d], "42", 10)
    ok, recs = pz.grade({"sum": {"type": "choice", "choice": "42", "confidence": 0.7}})
    assert ok is True and recs[0]["correct"] is True
    assert pz.grade({"sum": {"type": "choice", "choice": "7", "confidence": 0.7}})[0] is False


def test_records_expose_probability_and_label_for_noul():
    pz, ans = _puzzle("argmax", 0, 2, [0.8, 0.2])
    pz.decisions[0].truth = True
    _, recs = pz.grade(ans)
    assert recs[0]["prob"] == 0.8 and recs[0]["truth"] is True
    assert recs[0]["correct"] is True and recs[1]["correct"] is True  # 0.2 < 0.5 and label is False


# ---------------------------------------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------------------------------------


def test_auc_is_half_for_a_constant_responder():
    assert auc(np.array([0.7, 0.7, 0.7, 0.7]), np.array([1.0, 0.0, 1.0, 0.0])) == 0.5


def test_auc_ranks_correctly():
    assert auc(np.array([0.9, 0.8, 0.2, 0.1]), np.array([1.0, 1.0, 0.0, 0.0])) == 1.0
    assert auc(np.array([0.1, 0.2, 0.8, 0.9]), np.array([1.0, 1.0, 0.0, 0.0])) == 0.0


def test_chance_matches_the_option_count():
    pz, _ = _puzzle("argmax", 0, 8, [0.5] * 8)
    assert chance(pz) == pytest.approx(1 / 8)
    pz, _ = _puzzle("argmax", {0, 1}, 8, [0.5] * 8)
    assert chance(pz) == pytest.approx(2 / 8)
    pz, _ = _puzzle("subset", {0, 1}, 4, [0.5] * 4)  # C(4,2) = 6 equally likely subsets of that size
    assert chance(pz) == pytest.approx(1 / 6)


# ---------------------------------------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------------------------------------


def test_excluded_types_are_documented_not_silently_dropped():
    from laya.captcha import LOADERS

    assert set(EXCLUDED) & set(LOADERS) == set()
    assert len(EXCLUDED) + len(LOADERS) == 20, "every Open CaptchaWorld type is either loaded or excluded"


def test_load_rejects_unknown_types(tmp_path):
    with pytest.raises(ValueError, match="unsupported captcha types"):
        load_puzzles(str(tmp_path), ["Not_A_Type"])


def test_load_reports_a_missing_directory(tmp_path):
    with pytest.raises(FileNotFoundError, match="Select_Animal"):
        load_puzzles(str(tmp_path), ["Select_Animal"])


def test_kway_builds_one_decision_per_option(tmp_path):
    d = tmp_path / "Object_Match"
    d.mkdir()
    for n in ("reference1.png", "a.png", "b.png", "c.png"):
        Image.new("RGB", (32, 32)).save(d / n)
    (d / "ground_truth.json").write_text(json.dumps({
        "example1.png": {"reference_image": "reference1.png", "option_images": ["a.png", "b.png", "c.png"],
                         "correct_option_index": 1, "prompt": "Match the count."},
    }))
    (pz,) = load_puzzles(str(tmp_path), ["Object_Match"])
    assert pz.mode == "argmax" and pz.n_options == 3 and pz.truth == 1
    assert [dd.truth for dd in pz.decisions] == [False, True, False]
    for dd in pz.decisions:  # every decision shows the reference first, then one candidate
        assert len(dd.images) == 2 and dd.images[0].endswith("reference1.png")
    assert len(pz.decisions[0].state()["images"]) == 2


def test_grid_builds_one_decision_per_cell(tmp_path):
    d = tmp_path / "Select_Animal"
    d.mkdir()
    Image.new("RGB", (300, 200)).save(d / "image1.png")
    (d / "ground_truth.json").write_text(json.dumps({
        "image1.png": {"target_object": "fox", "grid_size": [2, 3], "correct_patches": [4],
                       "prompt": "Pick a fox"},
    }))
    (pz,) = load_puzzles(str(tmp_path), ["Select_Animal"])
    assert pz.mode == "argmax" and len(pz.decisions) == 6 and pz.truth == 4
    assert [dd.truth for dd in pz.decisions] == [False, False, False, False, True, False]
    assert "fox" in pz.decisions[0].question["instructions"]
    img, = pz.decisions[0].state()["images"]
    assert img.size == (100, 100)  # one cell, not the whole grid


def test_dice_ladder_is_deterministic_and_contains_the_answer(tmp_path):
    d = tmp_path / "Dice_Count"
    d.mkdir()
    Image.new("RGB", (64, 64)).save(d / "dice1.png")
    (d / "ground_truth.json").write_text(json.dumps({
        "dice1.png": {"sum": 37, "prompt": "Sum up the numbers on all the dice"},
    }))
    (a,) = load_puzzles(str(tmp_path), ["Dice_Count"])
    (b,) = load_puzzles(str(tmp_path), ["Dice_Count"])
    opts_a = list(a.decisions[0].question["criteria"])
    assert opts_a == list(b.decisions[0].question["criteria"]), "ladder must be stable across runs"
    assert "37" in opts_a and a.truth == "37" and a.n_options == len(opts_a)
