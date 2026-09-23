"""``laya.robustness_options``: the option-set / abstention builders on hand-made rows (no model), the summary on
synthetic predictions with known answers, then one end-to-end run through ``robustness.score_rows`` with the
untrained SmolVLM-256M agent on a tiny on-disk dataset (CPU is fine)."""
import json
import math

import pytest
import torch
from PIL import Image

from laya import robustness as R
from laya import robustness_options as O
from laya.common import render_options
from laya.vlm_train import load_jsonl_examples


def square_png(path, color, size=64):
    img = Image.new("RGB", (size, size), (255, 255, 255))
    img.paste(Image.new("RGB", (size // 2, size // 2), color), (size // 4, size // 4))
    img.save(path)


COLORS = {"red": (220, 30, 30), "blue": (30, 60, 220), "green": (30, 170, 60), "yellow": (230, 210, 40)}
SHAPES = ["square", "circle", "triangle"]


@pytest.fixture()
def dataset(tmp_path):
    """``<tmp>/sq/val.jsonl`` + images (as in ``tests/test_robustness.py``): 4 images, each with a 4-way colour
    question and a yes/no question, plus one text-only row; and here a 3-way shape question per image, so the
    colour rows have options from other rows to borrow."""
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
        recs.append({"id": "s%d" % i, "image": "images/%d.png" % i,
                     "question": {"type": "choice", "instructions": "What shape is shown?", "criteria": SHAPES},
                     "label": 0})
    recs.append({"id": "t0", "state_text": "The sky is blue.",
                 "question": {"type": "choice", "instructions": "what color is the sky", "criteria": ["Blue", "red"]},
                 "label": 0})
    with open(base / "val.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    return str(tmp_path)


def rows_of(root):
    return R.source_rows(load_jsonl_examples(root, "sq", "val"), dataset="sq")


def row(g, opts, label, cluster=None, dataset="d", t="choice"):
    """A hand-made source row (no image)."""
    q = {"t": t, "ins": "q?", "crit": {o: None for o in opts} if t == "choice" else None}
    k = len(render_options(q))
    return {"state": "", "q": q, "target": [float(j == label) for j in range(k)], "label": label, "dataset": dataset,
            "group_id": g, "cluster": cluster or g, "id": g + "|orig", "family": "orig", "variant": "orig"}


def by_var(vs, v):
    return {r["group_id"]: r for r in vs if r["variant"] == v}


# ---------------------------------------------------------------------------------------------------------
# Builders (no model)
# ---------------------------------------------------------------------------------------------------------


def test_build_deterministic_ids_and_groups(dataset):
    rows = rows_of(dataset)
    v1, v2 = O.build(rows, seed=3), O.build(rows, seed=3)
    assert json.dumps(v1, sort_keys=True, default=str) == json.dumps(v2, sort_keys=True, default=str)
    assert len({r["id"] for r in v1}) == len(v1) and not any(r["family"] == "orig" for r in v1)
    src = {r["group_id"]: r for r in rows}
    for r in v1:
        s = src[r["group_id"]]
        assert r["cluster"] == s["cluster"] and s["q"]["t"] == "choice"  # noul rows skipped
        assert r["id"] == "%s|%s/%s" % (r["group_id"], r["family"], r["variant"])
        assert len(r["target"]) == len(render_options(r["q"])) and abs(sum(r["target"]) - 1) < 1e-9
        json.dumps(r["meta"])
    assert {r["variant"] for r in v1} == {v for vs in O.VARIANTS.values() for v in vs}
    # no clash with the main harness ids
    main = R.build_variants(rows, seed=3)
    assert not {r["id"] for r in main} & {r["id"] for r in v1}
    with pytest.raises(ValueError):
        O.build(rows, families=("nope",))


def test_add_distractor_never_own_row_nor_duplicate():
    rows = [row("a", ["red", "blue", "green"], 1, cluster="img1"),
            row("b", ["Red", "BLUE ", "green"], 0, cluster="img2"),       # all duplicates of a's
            row("c", ["red", "cat"], 0, cluster="img1"),                  # a's sibling: same cluster, never a donor
            row("e", ["Dog", "blue"], 1, cluster="img3"),
            row("n", [], 1, cluster="img4", t="noul")]
    for seed in range(20):
        vs = O.add_distractor_variants(rows, seed)
        d = by_var(vs, "add_distractor")
        assert "n" not in d
        for g, v in d.items():
            src = next(r for r in rows if r["group_id"] == g)
            old, new = list(src["q"]["crit"]), list(v["q"]["crit"])
            assert new[:-1] == old and v["label"] == src["label"] and v["target"] == src["target"] + [0.0]
            assert new[-1].strip().lower() not in {o.strip().lower() for o in old}
            donor = next(r for r in rows if r["group_id"] == v["meta"]["distractor_from"])
            assert donor["group_id"] != g and donor["cluster"] != src["cluster"] and new[-1] in donor["q"]["crit"]
            assert v["meta"]["orig_index"] == list(range(len(old))) + [None] and v["meta"]["added_index"] == len(old)
        # a's only non-duplicate from another cluster is e's "Dog" ("cat" is its sibling's)
        assert list(d["a"]["q"]["crit"])[-1] == "Dog"
    # "e" can borrow from a, b or c; the pick depends on the seed only
    picks = {list(by_var(O.add_distractor_variants(rows, s), "add_distractor")["e"]["q"]["crit"])[-1] for s in range(20)}
    assert picks <= {"red", "green", "Red", "cat"} and len(picks) > 1
    # nothing to borrow -> no variant
    assert O.add_distractor_variants([row("a", ["x", "y"], 0), row("b", ["X", "y"], 1)]) == []


def test_drop_wrong_remaps_label():
    rows = [row("g%d" % i, ["a", "b", "c", "d"], i) for i in range(4)] + [row("two", ["a", "b"], 0)]
    dropped = set()
    for seed in range(10):
        vs = O.drop_wrong_variants(rows, seed)
        assert "two" not in by_var(vs, "drop_wrong")  # k < 3 skipped
        for v in vs:
            src = next(r for r in rows if r["group_id"] == v["group_id"])
            i, keep = v["meta"]["dropped_index"], v["meta"]["orig_index"]
            assert i != src["label"] and keep == [j for j in range(4) if j != i]
            assert list(v["q"]["crit"])[v["label"]] == list(src["q"]["crit"])[src["label"]]
            assert v["target"][v["label"]] == 1.0 and len(v["target"]) == 3
            dropped.add(i)
    assert dropped == {0, 1, 2, 3}  # seeded, not always the same position
    # soft targets are renormalised
    s = dict(row("s", ["a", "b", "c"], 0), target=[0.6, 0.3, 0.1])
    v = O.drop_wrong_variants([s], 0)[0]
    assert sum(v["target"]) == pytest.approx(1) and v["target"][v["label"]] == max(v["target"])


def test_drop_gold_marks_no_gold():
    rows = [row("g", ["a", "b", "c", "d"], 2), row("two", ["a", "b"], 1), row("n", [], 0, t="noul")]
    vs = O.drop_gold_variants(rows)
    assert [v["group_id"] for v in vs] == ["g"]
    v = vs[0]
    assert v["family"] == "abstain" and v["meta"]["no_gold"] is True and v["label"] == -1
    assert list(v["q"]["crit"]) == ["a", "b", "d"] and v["target"] == [pytest.approx(1 / 3)] * 3
    assert v["meta"]["orig_index"] == [0, 1, 3] and v["meta"]["dropped_index"] == 2


def test_add_none_and_shuffled_donor(dataset):
    rows = rows_of(dataset)
    vs = O.add_none_variants(rows, seed=0)
    plain, sh = by_var(vs, "add_none"), by_var(vs, "add_none_shuffled")
    choice = [r for r in rows if r["q"]["t"] == "choice"]
    assert set(plain) == {r["group_id"] for r in choice}
    assert set(sh) == {r["group_id"] for r in choice if r["cluster"] != r["group_id"]}  # the text-only row: none
    control = {r["group_id"]: r for r in R.shuffle_variants(rows, seed=0)}
    for g, v in plain.items():
        src = next(r for r in rows if r["group_id"] == g)
        assert list(v["q"]["crit"])[-1] == O.NONE_OPTION and v["meta"]["none_index"] == len(src["target"])
        assert v["label"] == src["label"] and v["state"] == src["state"] and not v["meta"]["mismatched_image"]
    for g, v in sh.items():
        # the same donor as the image_shuffle control, and never the row's own image
        assert v["state"] == control[g]["state"] and v["donor_image"] == control[g]["donor_image"]
        assert v["state"]["image"] != next(r for r in rows if r["group_id"] == g)["state"]["image"]
        assert v["meta"]["mismatched_image"] and v["q"] == plain[g]["q"]
    # an existing "None of the above" option -> no variant
    assert O.add_none_variants([row("x", ["a", "None of the above "], 0)]) == []


# ---------------------------------------------------------------------------------------------------------
# Summary on synthetic predictions
# ---------------------------------------------------------------------------------------------------------


def _pred(g, fam, var, label, logits, meta=None):
    z = [float(v) for v in logits]
    m = max(z)
    e = [math.exp(v - m) for v in z]
    probs = [v / sum(e) for v in e]
    p = dict(id="%s|%s/%s" % (g, fam, var), group_id=g, cluster=g, dataset="d", family=fam, variant=var, label=label,
             logits=z, probs=probs, pred=max(range(len(z)), key=lambda j: z[j]), k=len(z))
    if meta is not None:
        p["meta"] = meta
    return p


def test_summary_known_answers():
    L = math.log
    preds = [
        _pred("g1", "orig", "orig", 0, [2, 1, 0]),
        _pred("g2", "orig", "orig", 1, [0, 1, 3]),  # wrong
        # g1: the distractor shifts z0 - z1 from 1 to 1.5, z0 - z2 from 2 to 2, z1 - z2 from 1 to 0.5 -> mean |.| 1/3
        _pred("g1", "option_set", "add_distractor", 0, [2, 0.5, 0, 5], {"orig_index": [0, 1, 2, None]}),
        # g2: no change among the originals; the distractor is not the argmax -> no flip
        _pred("g2", "option_set", "add_distractor", 1, [0, 1, 3, -1], {"orig_index": [0, 1, 2, None]}),
        # g1 drops option 2: label 0 kept, now wrong (flip); g2 drops option 0: label 1 -> 0, now right
        _pred("g1", "option_set", "drop_wrong", 0, [0, 1], {"orig_index": [0, 1]}),
        _pred("g2", "option_set", "drop_wrong", 0, [4, 3], {"orig_index": [1, 2]}),
        _pred("g1", "abstain", "drop_gold", -1, [L(0.9), L(0.1)], {"no_gold": True, "orig_index": [1, 2]}),
        _pred("g2", "abstain", "drop_gold", -1, [L(0.6), L(0.4)], {"no_gold": True, "orig_index": [0, 2]}),
        _pred("g1", "abstain", "add_none", 0, [3, 1, 0, 2], {"none_index": 3, "orig_index": [0, 1, 2, None]}),
        _pred("g2", "abstain", "add_none", 1, [0, 1, 0, 5], {"none_index": 3, "orig_index": [0, 1, 2, None]}),
        _pred("g1", "abstain", "add_none_shuffled", 0, [0, 0, 0, 3], {"none_index": 3, "orig_index": [0, 1, 2, None]}),
        _pred("g2", "abstain", "add_none_shuffled", 1, [0, 1, 0, 5], {"none_index": 3, "orig_index": [0, 1, 2, None]}),
    ]
    s = O.summarize_options(preds)["d"]
    ad = s["option_set"]["add_distractor"]
    assert ad["mean_abs_logodds_shift"] == pytest.approx((1 / 3 + 0) / 2)
    assert ad["mean_abs_logodds_shift_cal"] == pytest.approx((1 / 3 + 0) / 2, abs=1e-6)
    assert ad["flip_rate"] == pytest.approx(0.5) and ad["pick_distractor_rate"] == pytest.approx(0.5)
    assert ad["acc"] == 0 and ad["base_acc"] == 0.5 and ad["delta_acc"] == -0.5
    dw = s["option_set"]["drop_wrong"]
    # g1: z0 - z1 was 1, now -1 -> 2; g2: z1 - z2 was -2, now 1 -> 3
    assert dw["mean_abs_logodds_shift"] == pytest.approx(2.5) and dw["mean_logodds_shift"] == pytest.approx(0.5)
    # g1 flips 0 -> 1; g2's orig argmax 2 is kept and now picked as 1 (orig index 1) -> flip
    assert dw["flip_rate"] == 1.0 and dw["flip_rate_kept"] == 1.0 and dw["delta_acc"] == 0
    ab = s["abstain"]
    assert ab["drop_gold"]["mean_p_max"] == pytest.approx(0.75)
    assert ab["drop_gold"]["share_p_max_gt_0.5"] == 1.0 and ab["drop_gold"]["share_p_max_gt_0.8"] == 0.5
    assert ab["add_none"]["none_rate"] == 0.5 and ab["add_none_shuffled"]["none_rate"] == 1.0
    assert ab["none_rate_rise"] == {"n": 2, "value": 0.5}
    assert ab["add_none"]["acc"] == 0.5 and ab["add_none"]["delta_acc"] == 0
    json.dumps(s)


# ---------------------------------------------------------------------------------------------------------
# End to end with the untrained agent
# ---------------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def agent():
    from laya.vlm import VLMAgent

    torch.manual_seed(0)
    return VLMAgent(backbone="HuggingFaceTB/SmolVLM-256M-Instruct", device="cpu", preprocess="processor")


def test_score_and_summarize_end_to_end(agent, dataset):
    rows = rows_of(dataset)
    variants = [dict(r) for r in rows] + O.build(rows, seed=0)
    preds = R.score_rows(agent.model, agent.processor, variants, agent.temperature, batch_size=8)
    assert [p["id"] for p in preds] == [v["id"] for v in variants]
    for p, v in zip(preds, variants):
        assert p["k"] == len(v["target"]) == len(p["probs"]) and abs(sum(p["probs"]) - 1) < 1e-3
        assert p.get("meta") == v.get("meta")
    s = O.summarize_options(preds)["sq"]
    assert set(s["option_set"]) == {"add_distractor", "drop_wrong"}
    assert set(s["abstain"]) == {"drop_gold", "add_none", "add_none_shuffled", "none_rate_rise"}
    assert s["option_set"]["add_distractor"]["n"] == 9  # every choice row, the text-only one included
    assert s["option_set"]["drop_wrong"]["n"] == s["abstain"]["drop_gold"]["n"] == 8
    assert s["option_set"]["add_distractor"]["mean_abs_logodds_shift"] >= 0
    assert 0 <= s["abstain"]["drop_gold"]["share_p_max_gt_0.8"] <= s["abstain"]["drop_gold"]["share_p_max_gt_0.5"] <= 1
    json.dumps(s)
