"""Tests for the detector -> decision layer (``laya.regions``). A fake agent stands in for ``VLMAgent`` except in the
last test, which runs a fresh SmolVLM-256M agent end to end."""
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from laya.regions import (NONE_KEY, REGIONS_FORMAT_VERSION, Region, RegionPipeline, as_regions, draw_marks,
                          from_omniparser, map_regions, select, select_question, stream)


class FakeAgent:
    """Answers a ``choice`` by the option whose text contains ``want`` (else the first), records every call."""

    def __init__(self, want=None):
        self.want = want
        self.calls = []

    def predict(self, state, questions, **kw):
        self.calls.append((state, questions, kw))
        answers = {}
        for qid, q in questions.items():
            if q["type"] == "choice":
                keys = list(q["criteria"])
                hit = [k for k in keys if self.want and self.want in (q["criteria"][k] or k)]
                pick = hit[0] if hit else keys[0]
                p = {k: (0.9 if k == pick else 0.1 / (len(keys) - 1)) for k in keys}
                answers[qid] = {"type": "choice", "choice": pick, "probabilities": p, "confidence": 0.8}
            else:
                answers[qid] = {"type": "noul", "noul": float(state["image"].size[0]) / 1000, "confidence": 0.5}
        return {"model": "fake", "answers": answers, "provenance": {"fake": True}}


def screen(w=200, h=100):
    return Image.new("RGB", (w, h), (255, 255, 255))


def boxes(n):
    return [Region((10.0 * i, 10.0, 10.0 * i + 8, 30.0), "item %d" % i, 0.5) for i in range(n)]


# -- adapters -----------------------------------------------------------------------------------------------------


def test_adapters_convert_each_family():
    sv = SimpleNamespace(xyxy=np.array([[1, 2, 3, 4]]), confidence=np.array([0.7]), class_id=np.array([2]),
                         mask=None, data={"class_name": np.array(["car"])})
    r = as_regions(sv)[0]
    assert r.box == (1.0, 2.0, 3.0, 4.0) and r.label == "car" and r.score == pytest.approx(0.7)

    ul = SimpleNamespace(boxes=SimpleNamespace(xyxy=np.array([[0, 0, 5, 5]]), conf=np.array([0.9]), cls=np.array([0])),
                         names={0: "person"}, masks=None)
    assert as_regions([ul])[0].label == "person"

    sam = [{"bbox": [10, 20, 30, 40], "segmentation": np.zeros((2, 2), bool), "predicted_iou": 0.8, "area": 4}]
    r = as_regions(sam)[0]
    assert r.box == (10.0, 20.0, 40.0, 60.0) and r.mask is not None and r.meta == {"area": 4}

    hf = [{"score": 0.5, "label": "cat", "box": {"xmin": 1, "ymin": 2, "xmax": 3, "ymax": 4}}]
    assert as_regions(hf)[0].label == "cat"

    omni = [{"type": "icon", "bbox": [0.1, 0.2, 0.3, 0.4], "interactivity": True, "content": "Settings "}]
    r = as_regions(omni, image_size=(200, 100))[0]
    assert r.box == pytest.approx((20.0, 20.0, 60.0, 40.0)) and r.label == "Settings"
    assert r.meta == {"type": "icon", "interactivity": True}
    with pytest.raises(ValueError):
        from_omniparser(omni)  # fractional boxes need the image size

    assert as_regions([[0, 0, 1, 1]])[0].box == (0.0, 0.0, 1.0, 1.0)
    assert as_regions([]) == [] and as_regions(None) == []


def test_crop_box_pads_grows_and_clamps():
    r = Region((0, 0, 10, 10))
    x0, y0, x1, y1 = r.crop_box((100, 100), pad=0.1, min_size=32)
    assert (x0, y0) == (0, 0) and x1 - x0 >= 21 and y1 - y0 >= 21  # grown about the centre, clamped at 0
    assert Region((50, 50, 90, 90)).crop_box((100, 100), pad=0.5) == (30, 30, 100, 100)


# -- select ------------------------------------------------------------------------------------------------------


def test_select_question_keys_and_texts():
    q = select_question([Region((0, 0, 1, 1), "OK  button"), Region((0, 0, 1, 1))], [3, 7], "Click?", "none of them")
    assert q["type"] == "choice"
    assert q["criteria"] == {"3": "box 3: OK button", "7": "box 7", NONE_KEY: "none of them"}


def test_select_one_round_draws_marks_and_returns_region():
    agent = FakeAgent(want="item 2")
    regs = boxes(4)
    out = select(agent, screen(), regs, "Which is item 2?")
    assert out["index"] == 2 and out["number"] == 3 and out["region"] is regs[2]
    assert out["click"] == regs[2].center
    assert set(out["probabilities"]) == {"1", "2", "3", "4"}
    assert out["regions_format"] == REGIONS_FORMAT_VERSION and len(out["rounds"]) == 1
    state = agent.calls[0][0]
    assert state["image"].size == (200, 100)
    assert np.asarray(state["image"]).min() < 255  # marks were drawn


def test_select_knockout_and_none():
    agent = FakeAgent(want="item 7")
    out = select(agent, screen(), boxes(10), "Find item 7", max_options=4, none_option="nothing matches")
    # 10 -> groups of 4,4,2 -> 3 winners -> final round of 3 (+ none)
    assert [len(r) for r in out["rounds"]] == [3, 1]
    assert out["index"] == 7
    assert NONE_KEY in out["probabilities"] and len(out["probabilities"]) == 4
    assert all(NONE_KEY not in q["region"]["criteria"] for _, q, _ in agent.calls[:-1])  # none only in the final

    agent = FakeAgent(want="nothing")
    out = select(agent, screen(), boxes(3), "?", none_option="nothing matches")
    assert out["index"] is None and out["click"] is None

    assert select(FakeAgent(), screen(), [], "?")["index"] is None


def test_select_passes_predict_kwargs_and_state_text():
    agent = FakeAgent()
    select(agent, screen(), boxes(2), "?", state_text={"goal": "log in"}, n_permutations=2)
    state, _, kw = agent.calls[0]
    assert state["goal"] == "log in" and kw == {"n_permutations": 2}


def test_draw_marks_does_not_touch_input():
    img = screen()
    marked = draw_marks(img, boxes(3), [1, 2, 3])
    assert np.asarray(img).min() == 255 and np.asarray(marked).min() < 255


# -- map ---------------------------------------------------------------------------------------------------------


def test_map_crops_each_region():
    agent = FakeAgent()
    regs = [Region((0, 0, 50, 50), "a"), Region((100, 0, 190, 90), "b")]
    qs = {"ok": {"type": "noul", "instructions": "Is it fine?"}}
    out = map_regions(agent, screen(), regs, qs, pad=0.0, min_size=1, with_label=True)
    assert [o["index"] for o in out] == [0, 1]
    assert [o["crop_box"] for o in out] == [(0, 0, 50, 50), (100, 0, 190, 90)]
    assert [c[0]["image"].size for c in agent.calls] == [(50, 50), (90, 90)]
    assert agent.calls[1][0]["detector_label"] == "b"
    assert out[1]["answers"]["ok"]["noul"] == pytest.approx(0.09)


# -- stream ------------------------------------------------------------------------------------------------------


def test_stream_in_order_with_prefetch_and_inline():
    frames = [screen() for _ in range(5)]
    det = lambda img: [[0, 0, 10, 10]]  # noqa: E731
    for prefetch in (0, 2):
        recs = list(stream(frames, det, lambda f, r: len(r), prefetch=prefetch))
        assert [r["index"] for r in recs] == list(range(5))
        assert all(r["result"] == 1 and r["dropped"] == 0 for r in recs)


def test_stream_drop_stale_skips_frames_when_decider_is_slow():
    frames = [screen() for _ in range(20)]
    recs = list(stream(frames, lambda img: [], lambda f, r: time.sleep(0.02), prefetch=1, drop_stale=True))
    idx = [r["index"] for r in recs]
    assert idx == sorted(idx) and idx[-1] == 19 and len(idx) < 20
    assert sum(r["dropped"] for r in recs) + len(recs) == 20


def test_stream_raises_detector_errors_and_stops_on_close():
    def bad(img):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        list(stream([screen()], bad, lambda f, r: None))

    seen = []

    def frames():
        for i in range(1000):
            seen.append(i)
            yield screen()

    gen = stream(frames(), lambda img: [], lambda f, r: None, prefetch=1)
    next(gen)
    gen.close()
    time.sleep(0.3)
    assert len(seen) < 10
    assert not any(t.name == "laya-regions-detect" and t.is_alive() for t in threading.enumerate())


def test_pipeline_select_map_stream_and_filter():
    agent = FakeAgent(want="item 1")
    pipe = RegionPipeline(agent, detector=lambda img: boxes(5), filter=lambda rs: rs[:3])
    assert len(pipe.detect(screen())) == 3
    assert pipe.select(screen(), "item 1?")["index"] == 1
    assert len(pipe.map(screen(), {"x": {"type": "noul", "instructions": "?"}})) == 3
    recs = list(pipe.stream([screen(), screen()], select="item 1?"))
    assert [r["result"]["index"] for r in recs] == [1, 1]
    with pytest.raises(ValueError):
        pipe.stream([], select="a", map={})


# -- a real agent --------------------------------------------------------------------------------------------------


def test_select_and_map_with_a_fresh_vlm_agent():
    """Shape only: a fresh head is untrained, so the choice itself is arbitrary."""
    torch = pytest.importorskip("torch")
    from laya.vlm import VLMAgent

    torch.manual_seed(0)
    agent = VLMAgent(backbone="HuggingFaceTB/SmolVLM-256M-Instruct", device="cpu")
    img = screen(256, 128)
    regs = [Region((10, 10, 60, 60), "red square"), Region((100, 10, 150, 60), "blue square")]
    out = select(agent, img, regs, "Which box is the red square?", none_option="neither")
    assert set(out["probabilities"]) == {"1", "2", NONE_KEY}
    assert sum(out["probabilities"].values()) == pytest.approx(1.0, abs=1e-3)
    assert out["provenance"]["prompt_format_version"]
    maps = map_regions(agent, img, regs, {"red": {"type": "noul", "instructions": "Is this square red?"}})
    assert [0.0 <= m["answers"]["red"]["noul"] <= 1.0 for m in maps] == [True, True]
