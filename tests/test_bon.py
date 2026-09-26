"""laya.bon: the exact best-of-N expectation against brute-force enumeration of every N-subset (CPU, no model)."""
import itertools
import math
import random

import pytest

from laya.bon import (best_of_n, bon_curve, candidate_state, choice_candidates, verifier_questions, verifier_scores,
                      vlfeedback_candidates)


def brute(candidates, n):
    """Average over every n-subset: the verifier's pick (uniform over the tied top scores), a random pick, the best."""
    sel = rnd = orc = 0.0
    subsets = list(itertools.combinations(candidates, n))
    for sub in subsets:
        top = max(s for s, _ in sub)
        tied = [r for s, r in sub if s == top]
        sel += sum(tied) / len(tied)
        rnd += sum(r for _, r in sub) / n
        orc += max(r for _, r in sub)
    k = len(subsets)
    return {"selected": sel / k, "random": rnd / k, "oracle": orc / k}


def _close(a, b):
    assert set(a) == set(b)
    for key in a:
        assert math.isclose(a[key], b[key], rel_tol=1e-12, abs_tol=1e-12), (key, a, b)


@pytest.mark.parametrize("seed", range(200))
def test_matches_brute_force_binary_with_ties(seed):
    rng = random.Random(seed)
    m = rng.randint(1, 8)
    # few distinct scores so ties are common
    cands = [(rng.choice([0.0, 0.5, 1.0, 2.0]), rng.randint(0, 1)) for _ in range(m)]
    for n in range(1, m + 1):
        _close(best_of_n(cands, n), brute(cands, n))


@pytest.mark.parametrize("seed", range(200))
def test_matches_brute_force_graded_rewards(seed):
    rng = random.Random(1000 + seed)
    m = rng.randint(1, 7)
    cands = [(rng.choice([-1.0, 0.25, 3.0, rng.random()]), rng.choice([1.0, 2.5, 3.0, 5.0, rng.random()]))
             for _ in range(m)]
    for n in range(1, m + 1):
        _close(best_of_n(cands, n), brute(cands, n))


def test_known_values():
    # perfect verifier: always finds a correct one when the subset has one -> selected == oracle == pass@N
    cands = [(0.9, 1), (0.1, 0), (0.2, 0), (0.3, 0)]
    r = best_of_n(cands, 2)
    assert r["random"] == 0.25 and math.isclose(r["oracle"], 1 - math.comb(3, 2) / math.comb(4, 2))
    assert math.isclose(r["selected"], r["oracle"])
    # a constant score is a random pick at every N
    flat = [(0.0, 1), (0.0, 0), (0.0, 0), (0.0, 1), (0.0, 0)]
    for n in range(1, 6):
        assert math.isclose(best_of_n(flat, n)["selected"], 0.4)
    # N = M: the top-scored candidate(s), deterministically
    assert best_of_n([(1.0, 0), (2.0, 1), (2.0, 0)], 3)["selected"] == 0.5
    # N = 1: every method is a random pick
    r1 = best_of_n(cands, 1)
    assert r1["selected"] == r1["random"] == r1["oracle"] == 0.25


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        best_of_n([(0.0, 1)], 2)
    with pytest.raises(ValueError):
        best_of_n([(0.0, 1)], 0)
    with pytest.raises(ValueError):
        best_of_n([(float("nan"), 1), (0.0, 0)], 1)
    with pytest.raises(ValueError):
        best_of_n([(0.0, float("inf"))], 1)


def test_bon_curve_averages_groups_and_keeps_short_ones():
    g1 = [(0.9, 1), (0.1, 0), (0.2, 0), (0.3, 0)]
    g2 = [(0.5, 0), (0.6, 1)]  # shorter than N=3: scored at N=2
    curve = bon_curve([g1, g2], [1, 3])
    b1, b2 = best_of_n(g1, 3), best_of_n(g2, 2)
    for k in ("selected", "random", "oracle"):
        assert math.isclose(curve[3][k], (b1[k] + b2[k]) / 2)
    assert curve[3]["short_groups"] == 1 and curve[1]["short_groups"] == 0 and curve[3]["n_groups"] == 2
    assert math.isclose(curve[3]["gap_closed"], 1.0)  # a perfect verifier closes the whole gap
    assert curve[1]["gap_closed"] is None  # at N=1 the oracle gains nothing over random
    with pytest.raises(ValueError):
        bon_curve([], [1])


def test_choice_candidates():
    q = {"t": "choice", "ins": "What colour is the bus?", "crit": {"red": None, "blue": None, "green": "a dark green"}}
    text, responses, rewards = choice_candidates(q, 2)
    assert text == "What colour is the bus?"
    assert responses == ["red", "blue", "green: a dark green"] and rewards == [0, 0, 1]
    with pytest.raises(ValueError):
        choice_candidates(q, 3)
    with pytest.raises(ValueError):
        choice_candidates({"t": "score", "ins": "x", "crit": ["a", "b"]}, 0)


def test_vlfeedback_candidates_match_the_training_records():
    from laya.rubric import vlfeedback_records

    row = {"id": "x", "prompt": "Is  there a dog?", "completions": [
        {"model": "a", "response": "Yes, a dog.", "annotations": {"Helpfulness": {"Rating": "4"}, "Visual Faithfulness": {"Rating": "5"}}},
        {"model": "b", "response": "", "annotations": {"Helpfulness": {"Rating": "1"}}},
        {"model": "c", "response": "No.", "annotations": {"Helpfulness": {"Rating": "N/A"}}},
        {"model": "d", "response": "A cat " * 400, "annotations": {"Helpfulness": {"Rating": "2"}}},
    ]}
    prompt, responses, ratings = vlfeedback_candidates(row)
    assert prompt == "Is there a dog?" and ratings == [4, 2] and responses[0] == "Yes, a dog."
    assert len(responses[1]) <= 1200 + 4 and responses[1].endswith(" ...")
    # the same text the score head was trained on for this row
    recs = [r for r in vlfeedback_records(row, "x", max_texts=99) if r["id"].endswith("helpfulness")]
    assert [r["state_text"] for r in recs] == [candidate_state(None, prompt, s)["context"] for s in responses]
    # the parquet's struct-of-lists layout reads the same
    sol = {"prompt": row["prompt"], "completions": {k: [c.get(k) for c in row["completions"]] for k in ("model", "response", "annotations")}}
    assert vlfeedback_candidates(sol) == (prompt, responses, ratings)


def test_verifier_scores_asks_both_questions_per_candidate():
    calls = []

    class FakeAgent:
        def predict(self, state, questions, **kw):
            calls.append((state, questions, kw))
            good = state["context"].endswith("blue")
            return {"answers": {"correct": {"noul": 0.8 if good else 0.1}, "helpful": {"score": 3.5 if good else 1.0}}}

    out = verifier_scores(FakeAgent(), "IMG", "What colour?", ["red", "blue"], n_permutations=2)
    assert out == [{"correct": 0.1, "helpful": 1.0}, {"correct": 0.8, "helpful": 3.5}]
    assert calls[0][0] == candidate_state("IMG", "What colour?", "red") == {"image": "IMG", "context": "Question: What colour?\n\nResponse: red"}
    assert set(calls[0][1]) == {"correct", "helpful"} and calls[0][2] == {"n_permutations": 2}
    qs = verifier_questions()
    assert qs["correct"]["type"] == "noul" and qs["helpful"]["type"] == "score" and len(qs["helpful"]["criteria"]) == 5
