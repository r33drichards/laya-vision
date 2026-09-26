"""The circle verifier passes circles (including the rough polygon the eight-move action set draws) and fails the
shapes a policy might draw instead."""
import math
import random

import numpy as np
import pytest

from laya.circle_verifier import fit_circle, score_circle

PIL = pytest.importorskip("PIL")
from PIL import Image, ImageDraw  # noqa: E402

W, H = 683, 384  # JSPaint's default canvas


def canvas(draw):
    im = Image.new("RGB", (W, H), "white")
    draw(ImageDraw.Draw(im))
    return np.asarray(im)


def polygon(n, r=110, cx=340, cy=190, rot=math.pi / 8):
    return [(cx + r * math.cos(rot + 2 * math.pi * i / n), cy + r * math.sin(rot + 2 * math.pi * i / n))
            for i in range(n + 1)]


def test_fit_circle_recovers_centre_and_radius():
    t = np.linspace(0, 2 * np.pi, 200)
    cx, cy, r = fit_circle(300 + 80 * np.cos(t), 150 + 80 * np.sin(t))
    assert (round(cx), round(cy), round(r)) == (300, 150, 80)


@pytest.mark.parametrize("name,draw", [
    ("ring", lambda d: d.ellipse([230, 80, 450, 300], outline="black", width=4)),
    ("octagon", lambda d: d.line(polygon(8), fill="black", width=4)),
    ("coloured ring", lambda d: d.ellipse([100, 50, 300, 250], outline=(200, 0, 0), width=6)),
])
def test_circles_pass(name, draw):
    s = score_circle(canvas(draw))
    assert s["passed"], (name, s)
    assert s["score"] >= 0.8 and s["coverage"] == 1.0


@pytest.mark.parametrize("name,draw", [
    ("blank", lambda d: None),
    ("line", lambda d: d.line([100, 100, 500, 300], fill="black", width=4)),
    ("dot", lambda d: d.ellipse([300, 150, 310, 160], fill="black")),
    ("tiny ring", lambda d: d.ellipse([300, 150, 330, 180], outline="black", width=4)),
    ("half arc", lambda d: d.arc([230, 80, 450, 300], 0, 180, fill="black", width=4)),
    ("three-quarter arc", lambda d: d.arc([230, 80, 450, 300], 0, 270, fill="black", width=4)),
    ("square", lambda d: d.rectangle([230, 80, 450, 300], outline="black", width=4)),
    ("flat ellipse", lambda d: d.ellipse([140, 120, 540, 270], outline="black", width=4)),
    ("filled disc", lambda d: d.ellipse([230, 80, 450, 300], fill="black")),
    ("ring plus stray line", lambda d: (d.ellipse([230, 80, 450, 300], outline="black", width=4),
                                        d.line([0, 20, 680, 20], fill="black", width=4))),
])
def test_non_circles_fail(name, draw):
    s = score_circle(canvas(draw))
    assert not s["passed"], (name, s)
    assert s["score"] < 0.6


def test_random_scribbles_fail():
    for seed in range(5):
        rng = random.Random(seed)
        pts = [(rng.uniform(0, W), rng.uniform(0, H)) for _ in range(15)]
        assert not score_circle(canvas(lambda d: d.line(pts, fill="black", width=4)))["passed"]


def test_arc_scores_below_closed_circle():
    closed = score_circle(canvas(lambda d: d.ellipse([230, 80, 450, 300], outline="black", width=4)))
    arc = score_circle(canvas(lambda d: d.arc([230, 80, 450, 300], 0, 270, fill="black", width=4)))
    assert arc["score"] < closed["score"] and arc["max_gap_deg"] > 30
