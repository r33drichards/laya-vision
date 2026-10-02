"""Tests for the MMAD mapping in ``laya/mmad.py``. Pure logic: no GPU, no Modal, no benchmark download.

Two things here are worth guarding. The letter mapping is the whole benchmark result -- MMAD grades an
answer letter, and a silent off-by-one between option text and its letter would score as a plausible-looking
accuracy rather than as a crash. And the calibration extras are this repo's own arithmetic rather than
MMAD's, so they get checked against a brute-force AUROC and a hand-computed ECE.
"""
import numpy as np
import pytest

from laya.mmad import build_questions, calibration_extras, question_text, to_letter


def conversation(options, answer, qtype="Anomaly Detection", question="Is there any defect in the object?"):
    return [{"Question": question, "Answer": answer, "Options": options, "type": qtype, "annotation": True}]


def predicted(q, m, pick_text):
    """What ``VLMAgent.predict`` returns when it puts nearly all its mass on ``pick_text``."""
    if q["type"] == "choice":
        keys = list({c: None for c in q["criteria"]})  # exactly what VLMAgent._to_internal does to a list
        return {"type": "choice", "choice": pick_text, "confidence": 0.9,
                "probabilities": {k: (0.97 if k == pick_text else 0.01) for k in keys}}
    return {"type": "noul", "noul": 0.97 if pick_text == "Yes." else 0.03, "confidence": 0.9}


# -- the letter mapping --------------------------------------------------------------------------------------


@pytest.mark.parametrize("options,answer", [
    ({"A": "Yes.", "B": "No."}, "A"),
    ({"A": "No.", "B": "Yes."}, "B"),   # MMAD balances the yes/no letter order; both must work
    ({"A": "No.", "B": "Yes."}, "A"),
    ({"A": "Foggy appearance.", "B": "Color fading.", "C": "Broken large.", "D": "Scratched surface."}, "C"),
    ({"A": "Unknown.", "B": "Maybe.", "C": "No.", "D": "Yes."}, "D"),  # the 4-option detection questions
])
def test_gold_option_recovers_its_letter(options, answer):
    questions, meta = build_questions(conversation(options, answer), noul_detection=False)
    m = meta["q0"]
    letter, probs = to_letter(predicted(questions["q0"], m, options[answer]), m)
    assert letter == answer
    assert set(probs) == set(options)
    assert max(probs, key=probs.get) == answer


def test_noul_head_respects_the_letter_order():
    """The noul head returns P(true); "Yes." is not always option A."""
    for options, yes_letter, no_letter in [({"A": "Yes.", "B": "No."}, "A", "B"),
                                           ({"A": "No.", "B": "Yes."}, "B", "A")]:
        questions, meta = build_questions(conversation(options, yes_letter), noul_detection=True)
        assert questions["q0"]["type"] == "noul"
        m = meta["q0"]
        assert to_letter({"type": "noul", "noul": 0.9, "confidence": 0.9}, m)[0] == yes_letter
        assert to_letter({"type": "noul", "noul": 0.1, "confidence": 0.9}, m)[0] == no_letter


def test_noul_only_claims_two_option_yes_no_questions():
    four = {"A": "Unknown.", "B": "Maybe.", "C": "No.", "D": "Yes."}
    questions, _ = build_questions(conversation(four, "D"), noul_detection=True)
    assert questions["q0"]["type"] == "choice", "a 4-option detection question has no yes/no head to use"


def test_options_keep_the_benchmarks_own_order():
    options = {"A": "Foggy appearance.", "B": "Color fading.", "C": "Broken large.", "D": "Scratched surface."}
    questions, meta = build_questions(conversation(options, "C", qtype="Defect Classification"), False)
    assert questions["q0"]["criteria"] == list(options.values())
    assert meta["q0"]["letters"] == ["A", "B", "C", "D"]


def test_recorded_prompt_text_names_every_option():
    options = {"A": "Yes.", "B": "No."}
    _, meta = build_questions(conversation(options, "A"), False)
    text = question_text(meta["q0"])["text"]
    assert text.startswith("Question: Is there any defect in the object?")
    for letter, body in options.items():
        assert "%s. %s" % (letter, body) in text


# -- the calibration extras ----------------------------------------------------------------------------------


def record(image, correct, got, probs, yes_letter=None, qtype="Anomaly Detection"):
    letters = sorted(probs)
    opts = {l: ("Yes." if l == yes_letter else "No.") for l in letters}
    text = "Question: q? \n" + "".join("%s. %s\n" % (l, opts[l]) for l in letters)
    return {"image": image, "question": {"type": "text", "text": text}, "question_type": qtype,
            "correct_answer": correct, "gpt_answer": got, "probabilities": probs}


def test_auroc_is_one_for_perfect_separation_whichever_letter_holds_yes():
    for yes, no in (("A", "B"), ("B", "A")):
        recs = [record("VisA/c/bad/%d.png" % i, yes, yes, {yes: p, no: round(1 - p, 4)}, yes)
                for i, p in enumerate([0.9, 0.8, 0.7])]
        recs += [record("VisA/c/good/%d.png" % i, no, no, {yes: p, no: round(1 - p, 4)}, yes)
                 for i, p in enumerate([0.3, 0.2, 0.1])]
        out = calibration_extras(recs)
        assert out["detection_auroc"] == 1.0
        assert out["detection_n"] == 6


def test_auroc_of_all_tied_scores_is_a_half():
    """A model that answers uniformly must score 0.5, which naive ranking gets wrong."""
    recs = [record("VisA/c/bad/%d.png" % i, "A", "A", {"A": 0.5, "B": 0.5}, "A") for i in range(4)]
    recs += [record("VisA/c/good/%d.png" % i, "B", "A", {"A": 0.5, "B": 0.5}, "A") for i in range(4)]
    assert calibration_extras(recs)["detection_auroc"] == 0.5


def test_auroc_matches_brute_force_with_ties():
    rng = np.random.default_rng(0)
    recs = []
    for i in range(300):
        y = int(rng.random() < 0.5)
        p = round(float(np.clip(rng.normal(0.6 if y else 0.4, 0.2), 0.01, 0.99)), 2)  # coarse -> many ties
        recs.append(record("VisA/c/%s/%d.png" % ("bad" if y else "good", i),
                           "A" if y else "B", "A" if p >= 0.5 else "B", {"A": p, "B": round(1 - p, 2)}, "A"))
    scored = [(r["probabilities"]["A"], 0.0 if "good" in r["image"] else 1.0) for r in recs]
    pos = [p for p, y in scored if y == 1]
    neg = [p for p, y in scored if y == 0]
    brute = sum((a > b) + 0.5 * (a == b) for a in pos for b in neg) / (len(pos) * len(neg))
    assert calibration_extras(recs)["detection_auroc"] == pytest.approx(brute, abs=1e-4)


def test_ece_and_accuracy():
    """Confidence 0.9 everywhere and half of them right: ECE is |0.5 - 0.9|."""
    recs = [record("d/x/good/%d.png" % i, "A", "A" if i % 2 == 0 else "B", {"A": 0.9, "B": 0.1}, "A")
            for i in range(10)]
    out = calibration_extras(recs)
    assert out["overall_accuracy"] == 0.5
    assert out["n_questions"] == 10
    assert out["ece"] == pytest.approx(0.4)


def test_threshold_sweep_finds_a_shifted_optimum():
    """Scores separable at 0.7 but not at 0.5: the oracle must find it, and CV must not beat it."""
    from laya.mmad import threshold_sweep

    recs = [record("d/x/bad/%d.png" % i, "A", "A", {"A": 0.8, "B": 0.2}, "A") for i in range(50)]
    recs += [record("d/x/good/%d.png" % i, "B", "A", {"A": 0.6, "B": 0.4}, "A") for i in range(50)]
    out = threshold_sweep(recs, folds=5)
    assert out["n"] == 100
    assert out["at_half"] == 0.5, out        # at 0.5 every image reads as defective
    assert out["oracle"] == 1.0, out         # a threshold in (0.6, 0.8] separates them perfectly
    assert 0.6 < out["oracle_threshold"] <= 0.8, out
    assert out["cv"] == 1.0, out             # the split is clean, so held-out folds agree
    assert out["cv"] <= out["oracle"], out


def test_threshold_sweep_cv_is_not_fooled_by_noise():
    """On pure noise the oracle still looks good in-sample; the cross-validated estimate must not."""
    import random as _random

    from laya.mmad import threshold_sweep

    rng = _random.Random(0)
    recs = []
    for i in range(300):
        y = i % 2
        p = round(rng.random(), 3)  # no signal whatsoever
        recs.append(record("d/x/%s/%d.png" % ("bad" if y else "good", i),
                           "A" if y else "B", "A" if p >= 0.5 else "B", {"A": p, "B": round(1 - p, 3)}, "A"))
    out = threshold_sweep(recs, folds=5)
    # the point is the gap, not its size: picking a threshold on the same data always beats chance a
    # little, even with nothing to find, and the cross-validated number must not inherit that
    assert out["oracle"] > 0.5, out
    assert out["cv"] < out["oracle"], out
    assert out["cv"] < 0.60, out                 # the honest estimate stays near chance


def test_image_url_survives_the_percent_encoded_revision():
    """The HF revision is ``refs%2Fpr%2F1`` and ``%2F`` is a printf conversion, so this cannot be a template."""
    from laya.mmad import image_url

    url = image_url("DS-MVTec/bottle/image/broken_large/000.png")
    assert url.endswith("/refs%2Fpr%2F1/DS-MVTec/bottle/image/broken_large/000.png")
    assert "%s" not in url
