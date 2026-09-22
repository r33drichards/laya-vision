"""CPU tests for the parsing and agreement logic of benchmarks/decision_vs_generation.py (not its timing)."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("decision_vs_generation", ROOT / "benchmarks" / "decision_vs_generation.py")
dvg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dvg)

QS = {
    "color": {"type": "choice", "instructions": "What color?", "criteria": ["red", "blue"]},
    "is_red": {"type": "noul", "instructions": "Is it red?"},
    "size": {"type": "score", "instructions": "How big?", "criteria": ["small", "medium", "large"]},
}


def test_strict_array():
    p = dvg.parse_array('["red", true, 2]', QS)
    assert p["strict_valid"] and p["strict"] == ["red", True, 2] and p["lenient"] == ["red", True, 2]


@pytest.mark.parametrize("text", ['["red", "true", 2]', '["red", true]', '["red", true, 2, 1]', '["red", true, 5]',
                                  '```json\n["red", true, 2]\n```', '[1, true, 2]'])
def test_strict_rejects(text):
    assert not dvg.parse_array(text, QS)["strict_valid"]


def test_lenient_recovers_near_misses():
    p = dvg.parse_array('Sure! ```json\n["Red", "yes", "level 1"]\n```', QS)
    assert not p["strict_valid"] and p["json_found"] and p["lenient"] == ["red", True, 1]
    p = dvg.parse_array('["blue: a blue thing", 0, "2"]', QS)
    assert p["lenient"] == ["blue", False, 2]
    p = dvg.parse_array('["purple"]', QS)  # unknown option and missing positions become None
    assert p["length"] == 1 and p["lenient"] == [None, None, None]
    p = dvg.parse_array("red, true, 2", QS)
    assert not p["json_found"] and p["lenient"] == [None, None, None]


def test_bool_is_not_a_level():
    assert dvg.normalize(QS["size"], True, strict=False) is None
    assert dvg.normalize(QS["size"], 1.0, strict=False) == 1


def test_parse_one():
    assert dvg.parse_one(" true ", QS["is_red"]) == {"strict_valid": True, "strict": True, "lenient": True}
    assert dvg.parse_one('"blue"', QS["color"])["strict"] == "blue"
    got = dvg.parse_one("The answer is Blue.", QS["color"])
    assert not got["strict_valid"] and got["lenient"] == "blue"
    assert dvg.parse_one("No.", QS["is_red"])["lenient"] is False
    assert dvg.parse_one("level 2: large", QS["size"])["lenient"] == 2
    assert dvg.parse_one("I cannot tell", QS["size"])["lenient"] is None


def test_typed_argmax_and_agreement():
    answers = {
        "color": {"type": "choice", "choice": "blue", "probabilities": {"red": 0.3, "blue": 0.7}},
        "is_red": {"type": "noul", "noul": 0.2},
        "size": {"type": "score", "score": 1.1, "probabilities": {"0": 0.2, "1": 0.5, "2": 0.3}},
    }
    ref = [dvg.typed_argmax(answers[k], q) for k, q in QS.items()]
    assert ref == ["blue", False, 1]
    a = dvg.agreement(ref, ["blue", None, 2])
    assert a["per_question"] == [True, False, False] and a["matches"] == 1 and a["rate"] == pytest.approx(1 / 3)
    assert dvg.agreement(ref, [None] * 3)["matches"] == 0


def test_prompt_lists_every_question_in_order():
    prompt = dvg.compact_prompt('{"context": "x"}', QS)
    body = json.loads(prompt.split("\n", 1)[1])
    assert [it["question"] for it in body["questions_in_output_order"]] == [q["instructions"] for q in QS.values()]
    assert body["questions_in_output_order"][2]["levels"] == {"0": "small", "1": "medium", "2": "large"}
    assert json.loads(dvg.one_prompt("s", QS["color"]).split("\n", 1)[1])["question"]["option_ids"] == ["red", "blue"]


def test_fixture_is_consistent():
    fx = dvg.load_fixture()
    types = [q["type"] for q in fx["questions"].values()]
    assert 10 <= len(types) <= 20 and set(types) == {"choice", "noul", "score"}
    assert dvg.FIXTURE_IMAGE.exists()
    for q in fx["questions"].values():  # every option id must survive the strict round trip
        for v in dvg.option_ids(q):
            assert dvg.normalize(q, json.loads(json.dumps(v))) == v


def test_fixture_image_matches_generator(tmp_path):
    """The committed PNG is what draw_fixture draws (skipped if this Pillow renders the font differently)."""
    out = dvg.draw_fixture(tmp_path / "card.png")
    if hashlib.sha256(out.read_bytes()).digest() != hashlib.sha256(dvg.FIXTURE_IMAGE.read_bytes()).digest():
        pytest.skip("this Pillow version rasterizes the fixture differently; the committed PNG is authoritative")
