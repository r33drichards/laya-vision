"""``laya.robustness_form``: the question-form builders on hand-made rows (no model), the summary on synthetic
predictions with known answers, then one end-to-end run through ``laya.robustness.score_rows`` with the untrained
SmolVLM-256M agent on a tiny on-disk dataset (CPU is fine)."""
import json

import pytest
import torch
from PIL import Image

from laya import robustness as R
from laya import robustness_form as F
from laya.common import QTYPES, render_options
from laya.vlm_train import load_jsonl_examples


def square_png(path, color, size=64):
    img = Image.new("RGB", (size, size), (255, 255, 255))
    img.paste(Image.new("RGB", (size // 2, size // 2), color), (size // 4, size // 4))
    img.save(path)


COLORS = {"red": (220, 30, 30), "blue": (30, 60, 220), "green": (30, 170, 60), "yellow": (230, 210, 40)}


@pytest.fixture()
def dataset(tmp_path):
    """``<tmp>/sq/val.jsonl`` + images: 4 images, each with a 4-way colour question and a yes/no question, plus
    one text-only choice row (as in ``tests/test_robustness.py``)."""
    base = tmp_path / "sq"
    (base / "images").mkdir(parents=True)
    recs = []
    for i, c in enumerate(COLORS):
        square_png(base / "images" / ("%d.png" % i), COLORS[c])
        crit = list(COLORS)
        recs.append({"id": "c%d" % i, "image": "images/%d.png" % i,
                     "question": {"type": "choice", "instructions": "What color is the square?", "criteria": crit},
                     "label": crit.index(c)})
        recs.append({"id": "n%d" % i, "image": "images/%d.png" % i,
                     "question": {"type": "noul", "instructions": "The square is red.", "criteria": None},
                     "label": int(c == "red")})
    recs.append({"id": "t0", "state_text": "The sky is blue.",
                 "question": {"type": "choice", "instructions": "what color is the sky", "criteria": ["Blue", "red"]},
                 "label": 0})
    with open(base / "val.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    return str(tmp_path)


def rows_of(root):
    return R.source_rows(load_jsonl_examples(root, "sq", "val"), dataset="sq")


def src_row(t, ins, crit, label, target=None, gid="d/000000", **kw):
    k = len(render_options({"t": t, "crit": crit}))
    target = target or [float(i == label) for i in range(k)]
    return dict({"state": "", "q": {"t": t, "ins": ins, "crit": crit}, "target": target, "label": label,
                 "dataset": "d", "group_id": gid, "cluster": gid, "id": gid + "|orig", "family": "orig",
                 "variant": "orig"}, **kw)


# ---------------------------------------------------------------------------------------------------------
# Builders (no model)
# ---------------------------------------------------------------------------------------------------------


def test_noul_to_choice():
    src = src_row("noul", "The square is red.", None, 1, cluster="img:a", meta={"note": "x"})
    (v,) = F.form_choice_variants([src])
    assert v["q"] == {"t": "choice", "ins": "The square is red.", "crit": {"no": None, "yes": None}}
    assert render_options(v["q"]) == ["no", "yes"]  # label order = noul label order (false, true)
    assert v["label"] == 1 and v["target"] == [0.0, 1.0] and v["variant"] == "as_choice"
    assert v["group_id"] == src["group_id"] and v["cluster"] == "img:a" and v["id"] == "d/000000|form_choice/as_choice"
    assert v["meta"] == {"note": "x", "form": "noul_to_choice"} and src["meta"] == {"note": "x"}


def test_choice_to_noul():
    src = src_row("choice", "  What color is it? ", {"red": None, "blue": "a dark blue"}, 1, target=[0.3, 0.7],
                  order=[1, 0], shown_label=0)
    vs = F.form_choice_variants([src])
    assert [v["variant"] for v in vs] == ["opt0", "opt1"]
    assert [v["q"]["ins"] for v in vs] == ["Is the answer to this question 'red'? What color is it?",
                                           "Is the answer to this question 'blue: a dark blue'? What color is it?"]
    assert all(v["q"]["t"] == "noul" and v["q"]["crit"] is None for v in vs)
    assert [v["label"] for v in vs] == [0, 1]
    assert vs[0]["target"] == pytest.approx([0.7, 0.3]) and vs[1]["target"] == pytest.approx([0.3, 0.7])
    assert [v["meta"] for v in vs] == [{"form": "choice_to_noul", "option": i, "k": 2, "choice_label": 1}
                                       for i in range(2)]
    # the source's display order refers to its options, not to the new yes/no ones
    assert all("order" not in v and "shown_label" not in v for v in vs)


def test_negation_frames():
    assert F.negate_frame("The cat is on the mat.") == ("statement", "Is it false that the cat is on the mat?")
    assert F.negate_frame("US troops are visible") == ("statement", "Is it false that US troops are visible?")
    assert F.negate_frame("Is it red?") == ("question",
                                            "Decide whether the answer to this question is no: Is it red?")
    assert F.negate_frame("is the dog left of the cat")[0] == "question"  # question word, no "?"
    assert F.negate_frame("  ") is None
    (v,) = F.negation_variants([src_row("noul", "The cat is on the mat.", None, 1, target=[0.2, 0.8])])
    assert v["label"] == 0 and v["target"] == [0.8, 0.2] and v["q"]["t"] == "noul" and v["variant"] == "statement"
    assert v["meta"] == {"form": "negation"}


def test_rows_skipped_where_rules_do_not_apply():
    rows = [src_row("score", "How good?", ["bad", "ok", "good"], 2, gid="d/0"),
            src_row("noul", "The cat is here.", {"false": "the cat", "true": "the dog"}, 0, gid="d/1"),
            src_row("noul", "   ", None, 0, gid="d/2"),
            src_row("choice", "", {"a": None, "b": None}, 0, gid="d/3")]
    assert F.build(rows) == []
    assert F.negation_variants([src_row("choice", "Which?", {"a": None, "b": None}, 0)]) == []
    with pytest.raises(ValueError):
        F.build(rows, families=("nope",))


def test_build_on_dataset(dataset):
    rows = rows_of(dataset)
    v1, v2 = F.build(rows, seed=0), F.build(rows, seed=0)
    assert json.dumps(v1, sort_keys=True) == json.dumps(v2, sort_keys=True)
    assert len({r["id"] for r in v1}) == len(v1)
    ids = {r["id"] for r in rows}
    assert not ids & {r["id"] for r in v1}
    by = {r["group_id"]: r for r in rows}
    # 4 noul -> 4 choice rows, 4 four-way + 1 two-way choice rows -> 18 noul rows, 4 negated rows
    fams = [(r["family"], r["meta"]["form"]) for r in v1]
    assert fams.count(("form_choice", "noul_to_choice")) == 4 and fams.count(("form_choice", "choice_to_noul")) == 18
    assert fams.count(("negation", "negation")) == 4
    for r in v1:
        s = by[r["group_id"]]
        assert r["cluster"] == s["cluster"] and r["state"] == s["state"]
        assert len(r["target"]) == len(render_options(r["q"]))  # what collect_logits / make_item read
        if r["meta"]["form"] == "choice_to_noul":
            assert r["label"] == int(r["meta"]["option"] == s["label"]) and r["meta"]["k"] == len(s["target"])
        elif r["family"] == "negation":
            assert r["label"] == 1 - s["label"]
        else:
            assert r["label"] == s["label"]
    # composable with the main builder: no id collisions
    allv = R.build_variants(rows, families=("option_order", "text"), seed=0) + v1
    assert len({r["id"] for r in allv}) == len(allv)


# ---------------------------------------------------------------------------------------------------------
# Summary on synthetic predictions
# ---------------------------------------------------------------------------------------------------------


def pred(g, fam, var, label, probs, cl=None, meta=None):
    p = dict(id="%s|%s/%s" % (g, fam, var), group_id=g, cluster=cl or g, dataset="d", family=fam, variant=var,
             label=label, probs=probs, k=len(probs), pred=max(range(len(probs)), key=lambda i: probs[i]))
    if meta is not None:
        p["meta"] = meta
    return p


def test_summary_known_answers():
    n2c = {"form": "noul_to_choice"}
    c2n = lambda i, k, gold: {"form": "choice_to_noul", "option": i, "k": k, "choice_label": gold}  # noqa: E731
    neg = {"form": "negation"}
    preds = [
        # noul groups n1, n2 (yes/no), their choice forms and negations
        pred("n1", "orig", "orig", 1, [0.2, 0.8], cl="A"), pred("n2", "orig", "orig", 0, [0.6, 0.4], cl="A"),
        pred("n1", "form_choice", "as_choice", 1, [0.4, 0.6], cl="A", meta=n2c),   # gap 0.2, agree
        pred("n2", "form_choice", "as_choice", 0, [0.3, 0.7], cl="A", meta=n2c),   # gap 0.3, disagree
        pred("n1", "negation", "statement", 0, [0.85, 0.15], cl="A", meta=neg),    # sum 0.95, consistent
        pred("n2", "negation", "question", 1, [0.35, 0.65], cl="A", meta=neg),     # sum 1.05, consistent
        # choice group c1 (3 options, gold 2, choice argmax 2) and c2 (2 options, gold 0, argmax 0)
        pred("c1", "orig", "orig", 2, [0.1, 0.3, 0.6], cl="B"),
        pred("c1", "form_choice", "opt0", 0, [0.9, 0.1], cl="B", meta=c2n(0, 3, 2)),
        pred("c1", "form_choice", "opt1", 0, [0.2, 0.8], cl="B", meta=c2n(1, 3, 2)),
        pred("c1", "form_choice", "opt2", 1, [0.4, 0.6], cl="B", meta=c2n(2, 3, 2)),   # ranking picks opt1: wrong
        pred("c2", "orig", "orig", 0, [0.7, 0.3], cl="C"),
        pred("c2", "form_choice", "opt0", 1, [0.3, 0.7], cl="C", meta=c2n(0, 2, 0)),
        pred("c2", "form_choice", "opt1", 0, [0.5, 0.5], cl="C", meta=c2n(1, 2, 0)),   # 0.5 vs 0.7: picks opt0
        # an incomplete ranking (opt1 missing) is counted, not scored
        pred("c3", "orig", "orig", 0, [0.7, 0.3], cl="C"),
        pred("c3", "form_choice", "opt0", 1, [0.3, 0.7], cl="C", meta=c2n(0, 2, 0)),
    ]
    s = F.summarize_form(preds, n_boot=200)
    d = s["datasets"]["d"]
    a = d["form_choice"]["noul_to_choice"]
    assert a["n_groups"] == 2 and a["abs_gap"]["mean"] == pytest.approx(0.25) and a["abs_gap"]["max"] == pytest.approx(0.3)
    assert a["argmax_agree"] == pytest.approx(0.5) and a["acc"] == pytest.approx(0.5) and a["base_acc"] == 1.0
    assert a["mean_signed_gap"] == pytest.approx((-0.2 + 0.3) / 2)
    b = d["form_choice"]["choice_to_noul"]
    assert b["n_groups"] == 2 and b["n_incomplete"] == 1
    assert b["argmax_agree"] == pytest.approx(0.5) and b["acc"] == pytest.approx(0.5) and b["base_acc"] == 1.0
    assert b["p_yes_sum"] == {"mean": pytest.approx((1.5 + 1.2) / 2), "min": pytest.approx(1.2),
                              "max": pytest.approx(1.5)}
    # c1 row accuracy: opt0 right, opt1 wrong, opt2 right -> 2/3; c2: opt0 right, opt1 pred 0 (tie -> 0) right -> 1
    assert b["row_acc"] == pytest.approx((2 / 3 + 1) / 2)
    n = d["negation"]
    assert n["complement_sum"]["mean"] == pytest.approx(1.0) and n["complement_sum"]["min"] == pytest.approx(0.95)
    assert n["complement_sum"]["max"] == pytest.approx(1.05) and n["complement_sum"]["share_outside"] == 0
    assert n["acc"] == pytest.approx(1.0) and n["flip_rate"] == 0 and n["base_acc"] == pytest.approx(1.0)
    assert set(n["variants"]) == {"statement", "question"}
    assert s["macro"]["negation"]["complement_sum_mean"] == pytest.approx(1.0)
    json.dumps(s)

    # a negation whose answer does not flip is inconsistent, and a sum of 1.6 is outside the band
    bad = [p for p in preds if p["family"] != "negation"] + [
        pred("n1", "negation", "statement", 0, [0.2, 0.8], cl="A", meta=neg)]
    n = F.summarize_form(bad, n_boot=0)["datasets"]["d"]["negation"]
    assert n["flip_rate"] == 1.0 and n["acc"] == 0.0 and n["complement_sum"]["share_outside"] == 1.0
    assert n["complement_sum"]["mean"] == pytest.approx(1.6)


# ---------------------------------------------------------------------------------------------------------
# End to end with the untrained agent
# ---------------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def agent():
    from laya.vlm import VLMAgent

    torch.manual_seed(0)
    return VLMAgent(backbone="HuggingFaceTB/SmolVLM-256M-Instruct", device="cpu", preprocess="processor")


def test_form_rows_score_and_summarise(agent, dataset):
    rows = rows_of(dataset)
    variants = [dict(r) for r in rows] + F.build(rows, seed=0)
    preds = R.score_rows(agent.model, agent.processor, variants, agent.temperature, batch_size=8)
    assert [p["id"] for p in preds] == [v["id"] for v in variants]
    for p, v in zip(preds, variants):
        # the changed type and option count reach collect_logits
        assert p["qtype"] == QTYPES[v["q"]["t"]] and p["k"] == len(v["target"]) == len(p["probs"])
        assert abs(sum(p["probs"]) - 1) < 1e-3 and p["label"] == v["label"]
        if v["family"] != "orig":
            assert p["meta"] == v["meta"]
    s = F.summarize_form(preds, n_boot=50)["datasets"]["sq"]
    assert s["form_choice"]["noul_to_choice"]["n_groups"] == 4
    assert s["form_choice"]["choice_to_noul"]["n_groups"] == 5 and s["form_choice"]["choice_to_noul"]["n_incomplete"] == 0
    assert s["negation"]["n_groups"] == 4
    cs = s["negation"]["complement_sum"]
    assert 0 <= cs["min"] <= cs["mean"] <= cs["max"] <= 2 and 0 <= cs["share_outside"] <= 1
    json.dumps(s)
