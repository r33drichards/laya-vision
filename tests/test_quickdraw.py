"""Quick, Draw! doodles as drawing tasks, offline: stroke placement, resampling, rasterising, and the doodle
labeller's judgement labels (the browser and the network are not needed)."""
import math

import numpy as np
import pytest

from laya.quickdraw import (BITMAP, DOODLE_FRAC, DoodleLabeller, _resample, doodle_strokes, drawn_options,
                            render_strokes, to_bitmap)

PIL = pytest.importorskip("PIL")
from PIL import Image, ImageDraw  # noqa: E402

S = 512
HOUSE = {"key_id": "test", "strokes": [[(0, 120), (0, 255), (200, 255), (200, 120), (0, 120)],
                                       [(0, 120), (100, 0), (200, 120)]]}


class FakeEnv:
    def __init__(self, cursor=(256.0, 256.0), step=6):
        self.canvas_size, self.cursor, self.step_px, self.pen = S, cursor, step, False
        from laya.paintenv import compass_moves

        self.moves = compass_moves(32)


def test_resample_spacing_and_ends():
    pts = _resample([(0, 0), (30, 0), (30, 12)], 6)
    assert pts[0] == (0.0, 0.0) and pts[-1] == (30.0, 12.0)
    gaps = [math.dist(a, b) for a, b in zip(pts, pts[1:])]
    assert max(gaps) <= 6 + 1e-6


def test_doodle_starts_at_cursor_and_fits():
    strokes = doodle_strokes(HOUSE["strokes"], S, (200.0, 150.0), 6)
    assert strokes[0][0] == pytest.approx((200.0, 150.0))
    pts = np.array([p for s in strokes for p in s])
    assert (pts.max(0) - pts.min(0)).max() == pytest.approx(DOODLE_FRAC * S, rel=0.02)
    near_edge = doodle_strokes(HOUSE["strokes"], S, (500.0, 500.0), 6)  # would run off: shifted to fit
    pts = np.array([p for s in near_edge for p in s])
    assert pts.min() >= 8 - 1e-6 and pts.max() <= S - 8 + 1e-6


def test_bitmap_is_cropped_and_normalised():
    bm = to_bitmap(render_strokes(HOUSE["strokes"]))
    assert bm.shape == (BITMAP, BITMAP) and bm.max() <= 1.0 and bm.sum() > 20
    assert to_bitmap(np.zeros((64, 64), bool)).sum() == 0


def draw_strokes(strokes, upto=None):
    im = Image.new("RGB", (S, S), "white")
    d = ImageDraw.Draw(im)
    for s in strokes[:upto]:
        d.line(s, fill="black", width=4)
    return np.asarray(im)


def test_labeller_starts_by_pressing_at_the_cursor_and_labels_judgements():
    env = FakeEnv()
    lab = DoodleLabeller("house", HOUSE, drawn_options())
    lab.reset(env)
    assert lab.action(env, draw_strokes(lab.strokes, 0)) == "PEN_DOWN"
    blank = lab.judgements(draw_strokes(lab.strokes, 0))
    assert blank == {"progress": "0", "on_track": "on track", "drawn": "nothing"}
    lab.i, lab.j = len(lab.strokes), 0  # as if every stroke were drawn
    done = lab.judgements(draw_strokes(lab.strokes))
    assert done == {"progress": "4", "on_track": "on track", "drawn": "house"}
    assert lab.action(env, draw_strokes(lab.strokes)) == "DONE"


def test_stray_ink_is_off_track():
    env = FakeEnv()
    lab = DoodleLabeller("house", HOUSE, drawn_options())
    lab.reset(env)
    lab.i = 1
    px = draw_strokes(lab.strokes, 1)
    im = Image.fromarray(px.copy())
    ImageDraw.Draw(im).line([(10, 500), (500, 480), (20, 470), (490, 460)], fill="black", width=4)
    assert lab.judgements(np.asarray(im))["on_track"] == "off track"


def test_drawn_options_cover_training_categories_and_nothing():
    opts = drawn_options(("house", "apple"))
    assert list(opts) == ["house", "apple", "nothing"] and opts["apple"].startswith("a doodle of an apple")


def test_grouped_choice_moves_when_moving_is_likelier_than_any_pen_action():
    from laya.paintenv import COMPASS, grouped_choice

    probs = {d: 0.02 for d in COMPASS}  # moving: 0.66 in total, spread thin
    probs.update(E=0.04, PEN_DOWN=0.0, PEN_UP=0.30, DONE=0.02)
    assert max(probs, key=probs.get) == "PEN_UP" and grouped_choice(probs) == "E"
    probs.update(PEN_UP=0.9)
    assert grouped_choice(probs) == "PEN_UP"
