"""The JSPaint labeller's pieces on synthetic canvases: ring coverage, judgement labels and soft direction targets."""
import math

import numpy as np
import pytest

from laya.paintdata import (DIRECTION_SPREAD, MARGIN, SECTORS, direction_target, fits, implied_circle,
                            label_judgements, radius, ring_state, start_point)

PIL = pytest.importorskip("PIL")
from PIL import Image, ImageDraw  # noqa: E402

S = 512
R = radius(S)


def canvas(draw):
    im = Image.new("RGB", (S, S), "white")
    draw(ImageDraw.Draw(im))
    return np.asarray(im)


def box(cx, cy, r=R):
    return [cx - r, cy - r, cx + r, cy + r]


def test_blank_canvas():
    px = canvas(lambda d: None)
    assert implied_circle(px) is None
    assert label_judgements(px) == {"progress": "0", "on_track": "on track", "drawn": "nothing"}


@pytest.mark.parametrize("cx,cy", [(256, 256), (150, 330), (380, 140)])
def test_full_ring_anywhere_is_a_complete_circle(cx, cy):
    px = canvas(lambda d: d.ellipse(box(cx, cy), outline="black", width=4))
    c = implied_circle(px)
    assert abs(c[0] - cx) <= 2 and abs(c[1] - cy) <= 3  # centre one radius below the topmost ink
    assert ring_state(px, c)["covered"].all()
    assert label_judgements(px) == {"progress": "4", "on_track": "on track", "drawn": "circle"}


@pytest.mark.parametrize("extent,level", [(60, "1"), (200, "2"), (300, "3")])
def test_arc_drawn_clockwise_from_the_top(extent, level):
    # PIL angles run clockwise from 3 o'clock, so the top is 270
    px = canvas(lambda d: d.arc(box(256, 256), 270, 270 + extent, fill="black", width=4))
    lab = label_judgements(px)
    assert lab == {"progress": level, "on_track": "on track", "drawn": "arc"}


def test_stray_ink_is_off_track_and_unlabelled_shape():
    def draw(d):
        d.arc(box(256, 256), 270, 360, fill="black", width=4)
        d.line([60, 420, 470, 470], fill="black", width=4)  # far from the ring
    lab = label_judgements(canvas(draw))
    assert lab["on_track"] == "off track" and lab["drawn"] is None


def test_wrong_size_ring_is_off_track():
    px = canvas(lambda d: d.ellipse(box(256, 256, 40), outline="black", width=4))
    assert label_judgements(px)["on_track"] == "off track"


def test_arc_whose_circle_cannot_fit_is_off_track():
    px = canvas(lambda d: d.arc(box(256, S - 40), 270, 300, fill="black", width=4))  # top near the bottom edge
    assert not fits(implied_circle(px), S)
    assert label_judgements(px)["on_track"] == "off track"


def test_start_point_keeps_the_circle_on_the_canvas():
    assert start_point((256, 100), S) == (256, 100)
    x, y = start_point((500, 500), S)
    assert fits((x, y + R, R), S) and y + 2 * R <= S - MARGIN


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
