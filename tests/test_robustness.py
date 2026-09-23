"""``laya.robustness``: the perturbation builders on hand-made rows (no model), then the scoring plumbing and the
image-shuffle control end to end with the untrained SmolVLM-256M agent on a tiny on-disk dataset (CPU is fine)."""
import json
import math

import numpy as np
import pytest
import torch
from PIL import Image

from laya import robustness as R
from laya.common import render_options
from laya.vlm_train import collect_logits, load_jsonl_examples


def square_png(path, color, size=64):
    img = Image.new("RGB", (size, size), (255, 255, 255))
    img.paste(Image.new("RGB", (size // 2, size // 2), color), (size // 4, size // 4))
    img.save(path)


COLORS = {"red": (220, 30, 30), "blue": (30, 60, 220), "green": (30, 170, 60), "yellow": (230, 210, 40)}


@pytest.fixture()
def dataset(tmp_path):
    """``<tmp>/sq/val.jsonl`` + images: 4 images, each with a 4-way colour question and a yes/no question (so
    images are shared by two rows, like the Cauldron sets), plus one text-only row."""
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


def rows_of(root, n=0, seed=0):
    return R.source_rows(load_jsonl_examples(root, "sq", "val"), n=n, seed=seed, dataset="sq")


def strip_images(rows):
    """Rows as JSON (PIL-free), for determinism checks."""
    return json.dumps(rows, sort_keys=True, default=str)


# ---------------------------------------------------------------------------------------------------------
# Builders (no model)
# ---------------------------------------------------------------------------------------------------------


def test_source_rows_sampling_and_groups(dataset):
    rows = rows_of(dataset)
    assert len(rows) == 9 and all(r["family"] == "orig" and r["id"] == r["group_id"] + "|orig" for r in rows)
    assert len({r["group_id"] for r in rows}) == 9
    # the two questions on one image share a cluster; the text-only row is its own
    assert rows[0]["cluster"] == rows[1]["cluster"] != rows[2]["cluster"]
    assert rows[-1]["cluster"] == rows[-1]["group_id"]
    a, b = rows_of(dataset, n=5, seed=1), rows_of(dataset, n=5, seed=1)
    assert [r["group_id"] for r in a] == [r["group_id"] for r in b] and len(a) == 5
    assert [r["group_id"] for r in a] == sorted(r["group_id"] for r in a)  # file order kept


def test_build_is_deterministic_and_ids_unique(dataset):
    rows = rows_of(dataset)
    v1, v2 = R.build_variants(rows, seed=3), R.build_variants(rows, seed=3)
    assert strip_images(v1) == strip_images(v2)
    assert len({r["id"] for r in v1}) == len(v1)
    src = {r["group_id"] for r in rows}
    assert all(r["group_id"] in src for r in v1)
    assert {r["family"] for r in v1} == {"orig"} | set(R.FAMILIES)
    # a different seed changes the seeded parts (random order, random crop, shuffle) only
    v3 = R.build_variants(rows, seed=4)
    assert [r["id"] for r in v1 if r["family"] == "text"] == [r["id"] for r in v3 if r["family"] == "text"]
    crop = lambda vs: [r["image_op"]["seed"] for r in vs if r["variant"] == "crop_random90"]  # noqa: E731
    assert crop(v1) != crop(v3)


def test_option_orders_and_labels(dataset):
    rows = rows_of(dataset)
    ov = R.order_variants(rows, seed=0)
    for r in ov:
        src = next(s for s in rows if s["group_id"] == r["group_id"])
        k = len(src["target"])
        assert sorted(r["order"]) == list(range(k)) and r["order"] != list(range(k))
        # label and target stay in label order; the gold option is displayed at shown_label
        assert r["label"] == src["label"] and r["target"] == src["target"] and r["q"] == src["q"]
        assert r["order"][r["shown_label"]] == r["label"]
    # 4 options: 3 shifts, the reversal, and a random order unless it repeats one of them; 2 options: one swap
    per = {}
    for r in ov:
        per.setdefault(r["group_id"], []).append(r["variant"])
    for s in rows:
        names = per[s["group_id"]]
        if len(s["target"]) == 2:
            assert names == ["shift1"]
        elif len(s["target"]) == 4:
            assert names[:4] == ["shift1", "shift2", "shift3", "reversed"] and len(names) in (4, 5)
        assert len({tuple(o) for o in [r["order"] for r in ov if r["group_id"] == s["group_id"]]}) == len(names)


def test_text_rules():
    q = {"t": "choice", "ins": "What color is the square?", "crit": {"red": None, "blue": None, "US": None}}
    assert R.TEXT_RULES["prefix"](q)["ins"] == "Question: What color is the square?"
    assert R.TEXT_RULES["prefix"](dict(q, ins="question: x")) is None
    assert R.TEXT_RULES["suffix"](q)["ins"] == "What color is the square? Choose the correct option."
    assert R.TEXT_RULES["double_spaces"](q)["ins"] == "What  color  is  the  square?"
    assert R.TEXT_RULES["first_case"](q)["ins"] == "what color is the square?"
    assert R.TEXT_RULES["first_case"](dict(q, ins="US flag?")) is None
    assert R.TEXT_RULES["end_punct"](q)["ins"] == "What color is the square"
    assert R.TEXT_RULES["end_punct"](dict(q, ins="what color is it"))["ins"] == "what color is it?"
    assert R.TEXT_RULES["end_punct"](dict(q, ins="the dog is left of the cat"))["ins"] == "the dog is left of the cat."
    oc = R.TEXT_RULES["option_case"](q)
    assert list(oc["crit"]) == ["Red", "Blue", "US"]  # acronyms keep their case
    assert list(R.TEXT_RULES["option_period"](dict(q, crit={"red": None, "3.5": None}))["crit"]) == ["red.", "3.5"]
    assert R.TEXT_RULES["option_case"](dict(q, crit={"a": None, "b": None})) is None  # single letters: no variant
    assert R.TEXT_RULES["option_case"](dict(q, crit={"red": None, "Red": None})) is None  # differ only in case
    n = {"t": "noul", "ins": "The cat is on the mat.", "crit": None}
    assert R.TEXT_RULES["noul_frame"](n)["ins"] == "Is it true that the cat is on the mat?"
    assert R.TEXT_RULES["noul_frame"](dict(n, ins="Is it red?"))["ins"] == \
        "Decide whether the answer to this question is yes: Is it red?"
    assert R.TEXT_RULES["noul_frame"](q) is None and R.TEXT_RULES["noul_options"](q) is None
    no = R.TEXT_RULES["noul_options"](n)
    assert render_options(no) == ["false: no", "true: yes"]


def test_text_variants_preserve_labels(dataset):
    rows = rows_of(dataset)
    tv = R.text_variants(rows)
    assert tv and all(r["family"] == "text" for r in tv)
    for r in tv:
        src = next(s for s in rows if s["group_id"] == r["group_id"])
        assert r["label"] == src["label"] and r["target"] == src["target"] and r["state"] == src["state"]
        assert len(render_options(r["q"])) == len(render_options(src["q"]))
        # a renamed option keeps its position, so the gold option text maps to the same label
        if r["variant"] in ("option_case", "option_period"):
            old, new = list(src["q"]["crit"]), list(r["q"]["crit"])
            assert new[r["label"]].rstrip(".").lower() == old[r["label"]].lower()
    names = {(r["group_id"].split("/")[1], r["variant"]) for r in tv}
    assert ("000001", "noul_frame") in names and ("000000", "noul_frame") not in names


def test_image_ops_and_realize(dataset):
    rows = rows_of(dataset)
    iv = R.image_variants(rows, seed=0)
    assert len(iv) == 8 * len(R.IMAGE_OPS)  # the text-only row gets none
    with Image.open(rows[0]["state"]["image"]) as im:
        orig = np.asarray(im.convert("RGB"), dtype=float)
    for r in iv[: len(R.IMAGE_OPS)]:
        a = R.realize(r)["state"]["image"]
        b = R.realize(r)["state"]["image"]
        assert np.array_equal(np.asarray(a), np.asarray(b))  # deterministic
        spec = r["image_op"]
        if spec["op"] == "crop":
            w = round(64 * math.sqrt(spec["area"]))
            assert a.size == (w, w)
        else:
            assert a.size == (64, 64)
            assert np.abs(np.asarray(a, dtype=float) - orig).mean() > 0  # something changed
    assert R.realize(rows[0]) is rows[0]
    # sibling questions on one image get the same random crop
    c = [r for r in iv if r["variant"] == "crop_random90"]
    assert c[0]["image_op"]["seed"] == c[1]["image_op"]["seed"] != c[2]["image_op"]["seed"]


def test_shuffle_is_a_derangement_over_images(dataset):
    rows = rows_of(dataset)
    sv = R.shuffle_variants(rows, seed=0)
    assert len(sv) == 8
    for r in sv:
        src = next(s for s in rows if s["group_id"] == r["group_id"])
        assert r["state"]["image"] != src["state"]["image"] and r["label"] == src["label"] and r["q"] == src["q"]
    # siblings get the same donor, and every image is used once as a donor
    donors = {r["cluster"]: r["state"]["image"] for r in sv}
    assert all(r["state"]["image"] == donors[r["cluster"]] for r in sv)
    assert len(set(donors.values())) == 4
    to = R.text_only_variants(rows)
    assert len(to) == 8 and all(r["state"] == "" for r in to)


def test_summary_on_hand_made_predictions():
    """Two groups on one cluster plus one alone: group averaging, flips, per-variant rows, option-order spread."""
    def p(g, fam, var, label, pred, cl, k=2, **kw):
        probs = [0.2] * k
        probs[pred] = 1 - 0.2 * (k - 1)
        return dict(id="%s|%s/%s" % (g, fam, var), group_id=g, cluster=cl, dataset="d", family=fam, variant=var,
                    label=label, pred=pred, k=k, probs=probs, **kw)
    preds = [p("g1", "orig", "orig", 1, 1, "A"), p("g2", "orig", "orig", 0, 1, "A"), p("g3", "orig", "orig", 0, 0, "B"),
             p("g1", "option_order", "shift1", 1, 0, "A", shown_label=0),
             p("g2", "option_order", "shift1", 0, 1, "A", shown_label=1),
             p("g3", "option_order", "shift1", 0, 0, "B", shown_label=1),
             p("g1", "text", "prefix", 1, 1, "A"), p("g1", "text", "suffix", 1, 0, "A"),
             p("g3", "text", "prefix", 0, 1, "B")]
    s = R.summarize(preds, n_boot=200)["datasets"]["d"]
    assert s["orig"]["acc"] == pytest.approx(2 / 3) and s["orig"]["flip_rate"] == 0
    oo = s["option_order"]
    assert oo["acc"] == pytest.approx(1 / 3) and oo["flip_rate"] == pytest.approx(1 / 3)
    assert oo["acc_by_order"] == {"orig": pytest.approx(2 / 3), "shift1": pytest.approx(1 / 3)}
    assert oo["acc_spread"] == pytest.approx(1 / 3) and oo["n_clusters"] == 2
    tx = s["text"]
    # g1: (1 + 0) / 2, g3: 0 -> group mean 0.25; flips g1 0.5, g3 1 -> 0.75; the same groups' base acc is 1
    assert tx["acc"] == pytest.approx(0.25) and tx["flip_rate"] == pytest.approx(0.75)
    assert tx["base_acc"] == pytest.approx(1.0) and tx["delta_acc"] == pytest.approx(-0.75)
    assert tx["variants"]["prefix"] == {"n": 2, "acc": 0.5, "delta_acc": -0.5, "flip_rate": 0.5}
    lo, hi = tx["acc_ci"]
    assert 0 <= lo <= tx["acc"] <= hi <= 1
    assert "|" in R.format_table(R.summarize(preds, n_boot=0))


# ---------------------------------------------------------------------------------------------------------
# Scoring plumbing with the untrained agent
# ---------------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def agent():
    from laya.vlm import VLMAgent

    torch.manual_seed(0)
    # the processor path, like the published checkpoints: the device-side one stacks raw frames and so cannot
    # batch images of different sizes (the crops, or any real photo set)
    return VLMAgent(backbone="HuggingFaceTB/SmolVLM-256M-Instruct", device="cpu", preprocess="processor")


def test_order_rows_match_physically_permuted_options(agent, dataset):
    """A row scored under display ``order`` gives the same label-order logits as the example with its choice
    options physically reordered (and its label remapped), mapped back."""
    src = rows_of(dataset)[0]
    order = [2, 0, 3, 1]
    perm = dict(src, q=dict(src["q"], crit={list(src["q"]["crit"])[i]: None for i in order}),
                target=[src["target"][i] for i in order], label=order.index(src["label"]))
    perm.pop("order", None)
    r = collect_logits(agent.model, agent.processor, [dict(src, order=order), perm], batch_size=2)
    z_order, z_perm = r[0]["logits"], r[1]["logits"]
    assert torch.allclose(z_order[torch.tensor(order)], z_perm, atol=1e-4)
    assert r[1]["label"] == perm["label"] and list(perm["q"]["crit"])[perm["label"]] == "red"


def test_image_shuffle_control_plumbing(agent, dataset):
    rows = rows_of(dataset)
    variants = R.build_variants(rows, families=("image_shuffle", "text_only"), seed=0)
    variants += R.image_variants(rows, seed=0, ops=("jpeg40",))  # one op keeps the loader's realize() path covered
    preds = R.score_rows(agent.model, agent.processor, variants, agent.temperature, batch_size=8)
    assert [p["id"] for p in preds] == [v["id"] for v in variants]
    assert all(len(p["probs"]) == p["k"] and abs(sum(p["probs"]) - 1) < 1e-3 for p in preds)
    by = {p["id"]: p for p in preds}
    # the shuffled row sees another image: its logits move off the source row's (the untrained model still
    # reads pixels); the text-only row has no image at all
    moved = [not np.allclose(by[g + "|image_shuffle/shuffled"]["logits"], by[g + "|orig"]["logits"], atol=1e-3)
             for g in {r["group_id"] for r in rows if r["cluster"] != r["group_id"]}]
    assert all(moved)
    s = R.summarize(preds, n_boot=50)["datasets"]["sq"]
    assert set(s) == {"orig", "image", "image_shuffle", "text_only"}
    assert s["image_shuffle"]["n_groups"] == s["text_only"]["n_groups"] == 8 and s["orig"]["n_groups"] == 9
    assert 0 <= s["image_shuffle"]["agree_with_text_only"] <= 1 and "majority_label_acc" in s["text_only"]
    json.dumps(s)  # JSON-able


def test_extra_families_end_to_end(agent, dataset):
    """Every opt-in family through ``build_variants`` -> plain ``score_rows`` (its default transform draws the
    ``typo`` ops) -> ``summarize``: the core table stays the core families, each extension gets its own block."""
    rows = rows_of(dataset)
    variants = R.build_variants(rows, families=R.EXTRA_FAMILIES, seed=0)
    assert {v["family"] for v in variants} == {"orig"} | set(R.EXTRA_FAMILIES)
    assert R.build_variants(rows, families=R.EXTRA_FAMILIES, seed=0) == variants  # deterministic
    preds = R.score_rows(agent.model, agent.processor, variants, agent.temperature, batch_size=8)
    assert [p["id"] for p in preds] == [v["id"] for v in variants]
    s = R.summarize(preds, n_boot=20, ece_floor_sims=20)
    assert set(s["datasets"]["sq"]) == {"orig"}
    assert {"options", "form", "injection", "ece_floor"} <= set(s)
    json.dumps(s)
    with pytest.raises(ValueError):
        R.build_variants(rows, families=("nope",))
