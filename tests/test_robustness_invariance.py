"""``laya.robustness.invariance``: the hostile-neighbour builder and the comparison on hand-made logits (no model),
then repeat / batch / padding / prefix-cache invariance end to end with the untrained SmolVLM-256M agent on a tiny
on-disk dataset (CPU is fine). The model is deterministic in principle, so every condition must match its
reference to float rounding and never flip an answer."""
import json

import numpy as np
import pytest
import torch
from PIL import Image

from laya import robustness as R
from laya.robustness import invariance as I
from laya.vlm_train import load_jsonl_examples

#: tolerances for the fp32 CPU conditions. Measured on the 9-row set below with the untrained agent: every
#: condition within 3.1e-7 in probability and 1.2e-6 in logit, repeats bitwise equal; ~100x headroom here.
TOL_DPROB = 1e-5
TOL_DLOGIT = 1e-4


def square_png(path, color, size=64):
    img = Image.new("RGB", (size, size), (255, 255, 255))
    img.paste(Image.new("RGB", (size // 2, size // 2), color), (size // 4, size // 4))
    img.save(path)


COLORS = {"red": (220, 30, 30), "blue": (30, 60, 220), "green": (30, 170, 60), "yellow": (230, 210, 40)}


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    """``<tmp>/sq/val.jsonl`` + images: 4 images, each with a 4-way colour question and a yes/no question (so
    images are shared by two rows), plus one text-only row."""
    root = tmp_path_factory.mktemp("inv")
    base = root / "sq"
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
    return str(root)


def rows_of(root, n=0, seed=0):
    return R.source_rows(load_jsonl_examples(root, "sq", "val"), n=n, seed=seed, dataset="sq")


# ---------------------------------------------------------------------------------------------------------
# No model
# ---------------------------------------------------------------------------------------------------------


def test_hostile_rows(dataset):
    rows = rows_of(dataset)
    h = I.hostile_rows(rows)
    assert [r["id"] for r in h] == ["hostile|long_state", "hostile|two_images", "hostile|text_only",
                                    "hostile|many_options"]
    assert len(h[0]["state"]["context"].split()) == I.LONG_STATE_WORDS and "image" in h[0]["state"]
    assert len(h[1]["state"]["images"]) == 2 and "image" not in h[2]["state"]
    assert len(h[3]["q"]["crit"]) == 8 and all(len(r["target"]) >= 2 for r in h)
    # text-only rows: no image neighbour at all
    h0 = I.hostile_rows([r for r in rows if "image" not in r["state"]])
    assert [r["id"] for r in h0] == ["hostile|long_state", "hostile|text_only", "hostile|many_options"]


def test_compare_on_hand_made_logits():
    rows = [{"id": "a", "q": {"t": "choice"}}, {"id": "b", "q": {"t": "noul"}}]
    ref = [np.array([1.0, 0.0, -1.0]), np.array([0.0, 0.1])]
    got = [np.array([1.0, 0.0, -1.0]), np.array([0.2, 0.1])]
    c = I.compare(rows, ref, got, [1.0, 1.0, 1.0])
    assert c["rows"][0] == {"id": "a", "max_abs_dprob": 0.0, "max_abs_dlogit": 0.0, "flip": False,
                            "pred_ref": 0, "pred": 0, "margin_ref": c["rows"][0]["margin_ref"],
                            "margin": c["rows"][0]["margin_ref"]}
    e = np.exp([1.0, 0.0, -1.0])
    assert c["rows"][0]["margin_ref"] == pytest.approx((e[0] - e[1]) / e.sum())
    b = c["rows"][1]
    assert b["flip"] and b["pred_ref"] == 1 and b["pred"] == 0 and b["max_abs_dlogit"] == pytest.approx(0.2)
    assert b["max_abs_dprob"] == pytest.approx(2 / (1 + np.exp(-0.1)) - 1)  # |sigmoid(0.1) - sigmoid(-0.1)|
    assert b["margin_ref"] == pytest.approx(b["max_abs_dprob"]) and b["margin"] == pytest.approx(b["max_abs_dprob"])
    s = c["summary"]
    assert s["n"] == 2 and s["n_flips"] == 1 and s["n_exact"] == 1 and s["max_abs_dlogit"] == pytest.approx(0.2)
    json.dumps(c)


# ---------------------------------------------------------------------------------------------------------
# Untrained agent
# ---------------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def agent():
    from laya.vlm import VLMAgent

    torch.manual_seed(0)
    # the processor path, like the published checkpoints (and it batches images of any size)
    return VLMAgent(backbone="HuggingFaceTB/SmolVLM-256M-Instruct", device="cpu", preprocess="processor")


@pytest.fixture(scope="module")
def result(agent, dataset):
    # two images with both their questions (so predict has multi-question calls) plus the text-only row: every
    # neighbour shape without scoring all nine rows under every condition
    rows = [r for r in rows_of(dataset) if r["cluster"].endswith(("0.png", "1.png")) or r["cluster"] == r["group_id"]]
    assert len(rows) == 5
    return I.run_invariance(agent, rows, batch_size=8)


def _assert_invariant(result, name):
    c = result["conditions"][name]
    s = c["summary"]
    assert s["n"] == result["meta"]["n_rows"]
    assert s["n_flips"] == 0, [r for r in c["rows"] if r["flip"]]
    assert s["max_abs_dprob"] < TOL_DPROB and s["max_abs_dlogit"] < TOL_DLOGIT, s


def test_repeat_is_bitwise(result):
    for name in ("repeat", "predict_repeat"):
        s = result["summary"][name]
        assert s["n_exact"] == s["n"] and s["n_flips"] == 0


@pytest.mark.parametrize("name", ["batched", "batched_reversed", "hostile"])
def test_batch_and_padding_invariance(result, name):
    _assert_invariant(result, name)


@pytest.mark.parametrize("name", ["predict_multi", "predict_multi_hostile", "predict_prefix_cache",
                                  "predict_multi_prefix_cache"])
def test_predict_invariance(result, name):
    _assert_invariant(result, name)


def test_result_shape(result):
    assert set(result["conditions"]) == set(I.BATCH_CONDITIONS) | set(I.PREDICT_CONDITIONS) | {"predict_vs_batch"}
    assert all(c["reference"] in ("alone", "predict_alone") for c in result["conditions"].values())
    assert result["meta"]["hostile"][0] == "hostile|long_state"
    assert "|" in I.format_table(result)
    json.dumps(result)  # JSON-able
