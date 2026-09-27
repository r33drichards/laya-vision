"""The JSPaint labeller's pieces on synthetic canvases: ring coverage, judgement labels and soft direction targets."""
import math

import numpy as np
import pytest

from laya.paintdata import (DIRECTION_SPREAD, SECTORS, direction_target, label_judgements, ring_state,
                            target_circle)

PIL = pytest.importorskip("PIL")
from PIL import Image, ImageDraw  # noqa: E402

S = 512
CIRCLE = target_circle(S, S)


def canvas(draw):
    im = Image.new("RGB", (S, S), "white")
    draw(ImageDraw.Draw(im))
    return np.asarray(im)


def ring_box(c=CIRCLE):
    cx, cy, r = c
    return [cx - r, cy - r, cx + r, cy + r]


def test_blank_canvas():
    px = canvas(lambda d: None)
    assert ring_state(px, CIRCLE)["ink"] == 0
    assert label_judgements(px, CIRCLE) == {"progress": "0", "on_track": "on track", "drawn": "nothing"}


def test_full_ring_is_complete_circle():
    px = canvas(lambda d: d.ellipse(ring_box(), outline="black", width=4))
    rs = ring_state(px, CIRCLE)
    assert rs["covered"].all() and rs["stray"] < 0.01
    assert label_judgements(px, CIRCLE) == {"progress": "4", "on_track": "on track", "drawn": "circle"}


@pytest.mark.parametrize("extent,level", [(60, "1"), (200, "2"), (300, "3")])
def test_partial_arc_progress_levels(extent, level):
    px = canvas(lambda d: d.arc(ring_box(), 0, extent, fill="black", width=4))
    lab = label_judgements(px, CIRCLE)
    assert lab["progress"] == level and lab["drawn"] == "arc" and lab["on_track"] == "on track"


def test_stray_ink_is_off_track_and_unlabelled_shape():
    def draw(d):
        d.arc(ring_box(), 0, 90, fill="black", width=4)
        d.line([20, 20, 490, 60], fill="black", width=4)  # far from the ring
    lab = label_judgements(canvas(draw), CIRCLE)
    assert lab["on_track"] == "off track" and lab["drawn"] is None


def test_ring_elsewhere_does_not_count_as_progress():
    small = (70, 70, 30)  # nowhere near the target ring (distances 233-293 px from the centre vs r=154)
    px = canvas(lambda d: d.ellipse(ring_box(small), outline="black", width=4))
    lab = label_judgements(px, CIRCLE)
    assert lab["progress"] == "0" and lab["on_track"] == "off track"


def test_direction_target_spreads_to_neighbours():
    from laya.games import paint_question

    options = list(paint_question()["action"]["criteria"])
    t = direction_target(options, "N")
    assert math.isclose(sum(t), 1.0)
    assert t[options.index("N")] == pytest.approx(1 - 2 * DIRECTION_SPREAD)
    assert t[options.index("NbE")] == pytest.approx(DIRECTION_SPREAD)
    assert t[options.index("NbW")] == pytest.approx(DIRECTION_SPREAD)  # wraps around past N
    pen = direction_target(options, "PEN_DOWN")
    assert pen[options.index("PEN_DOWN")] == 1.0 and sum(pen) == 1.0


def test_sector_count():
    assert SECTORS == 36
