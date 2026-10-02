"""``laya.robustness.render``: the state / option rendering variants on hand-made rows and the summary on synthetic
predictions (no model)."""
import numpy as np

from laya import robustness as R
from laya.robustness import render as RR


def _rows():
    exs = [
        {"state": {"image": "a.png", "context": "Question: q?\n\nResponse: r"},
         "q": {"t": "score", "ins": "Rate.", "crit": ["bad", "good"]}, "target": [0.0, 1.0], "label": 1},
        {"state": {"image": "b.png", "context": "Plain hint"},
         "q": {"t": "choice", "ins": "Which?", "crit": {"Cab": None, "train": None}}, "target": [1.0, 0.0], "label": 0},
        {"state": {"image": "c.png"}, "q": {"t": "choice", "ins": "Which?", "crit": {"UP": "move up", "DOWN": "move down"}},
         "target": [0.0, 1.0], "label": 1},
        {"state": "", "q": {"t": "choice", "ins": "Which?", "crit": {"a": None, "A": None}}, "target": [1.0, 0.0], "label": 0},
    ]
    return R.source_rows(exs, dataset="d")


def test_state_variants():
    vs = [v for v in R.build_variants(_rows(), ["state_render"]) if v["family"] == "state_render"]
    by = {(v["group_id"], v["variant"]): v for v in vs}
    g0, g1 = "d/000000", "d/000001"
    assert set(v for g, v in by if g == g0) == {"prose", "text", "key_note", "struct_json"}  # struct_prose == text
    assert set(v for g, v in by if g == g1) == {"prose", "text", "key_note"}  # no Key: blocks
    assert by[(g0, "prose")]["state_format"] == "prose" and by[(g0, "prose")]["state"]["context"]
    assert by[(g0, "key_note")]["state"] == {"image": "a.png", "note": "Question: q?\n\nResponse: r"}
    assert by[(g0, "struct_json")]["state"] == {"image": "a.png", "Question": "q?", "Response": "r"}
    assert "state_format" not in by[(g0, "struct_json")]
    src = {r["group_id"]: r for r in _rows()}
    assert all((v["label"], v["target"], v["q"]) == (src[v["group_id"]]["label"], src[v["group_id"]]["target"],
                                                     src[v["group_id"]]["q"]) for v in vs)


def test_context_fields():
    assert RR.context_fields("Prompt: a cat") == {"Prompt": "a cat"}
    assert RR.context_fields("Lecture: x\ny\n\nQuestion: z") == {"Lecture": "x\ny", "Question": "z"}
    assert RR.context_fields("no key here") is None
    assert RR.context_fields("A: x\n\nA: y") is None


def test_option_variants():
    vs = [v for v in R.build_variants(_rows(), ["option_render"]) if v["family"] == "option_render"]
    got = {(v["group_id"], v["variant"]): list(v["q"]["crit"]) for v in vs}
    assert got == {("d/000001", "lower"): ["cab", "train"], ("d/000001", "title"): ["Cab", "Train"]}
    # described options (game actions) and case-only duplicates are skipped


def test_summary():
    rows = _rows()
    preds = []
    for r in rows:
        preds.append(dict(id=r["id"], group_id=r["group_id"], cluster=r["cluster"], dataset="d", family="orig",
                          variant="orig", label=r["label"], k=2, probs=[0.8, 0.2], pred=0))
        preds.append(dict(id=r["id"] + "x", group_id=r["group_id"], cluster=r["cluster"], dataset="d",
                          family="state_render", variant="prose", label=r["label"], k=2, probs=[0.3, 0.7], pred=1))
    s = RR.summarize_render(preds, n_boot=50)
    st = s["datasets"]["d"]["state_render"]["prose"]
    assert st["flip_rate"] == 1.0 and np.isclose(st["tv_mean"], 0.5)
    assert np.isclose(st["delta_acc"], 0.0)  # labels are half 0, half 1
    assert s["macro"]["state_render/prose"]["n_datasets"] == 1
    assert "render" in R.summarize(preds, n_boot=10) and "| d | state_render | prose |" in RR.format_table(s)
