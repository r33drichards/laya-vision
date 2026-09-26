"""Score how well a canvas shows one drawn circle: the verifier behind the JSPaint "draw a circle" task.

Pure numpy, so it scores any RGB array (the JSPaint canvas, a PIL image, a synthetic test image) without a browser.
The steps:

1. **Ink**: pixels that differ from the background colour by more than ``ink_threshold`` in any channel.
2. **Circle fit**: an algebraic least-squares (Kasa) fit of ``x^2 + y^2 + a x + b y + c = 0`` to the ink pixels, giving
   the centre ``(cx, cy)`` and radius ``r``.
3. **Components**, each in [0, 1]:

   * ``roundness``: how constant the radius is around the ring, ``1 - std(r_k) / (round_tol * r)``, where ``r_k`` is
     the mean distance from the centre of the ring ink in angular sector ``k``. Averaging within a sector cancels the
     stroke width, so this measures the shape itself: a clean ring scores ~1, a 16-sided walk ~0.9, an octagon ~0.5,
     a square or a 4:3 ellipse 0;
   * ``coverage``: the fraction of ``bins`` equal angular sectors around the centre that hold ink on the ring, so an
     arc or a C-shape loses the missing sectors;
   * ``clean``: the fraction of ink on the ring band (within ``band * r`` of it), so scribbles, fills and stray
     strokes count against the drawing;
   * ``closure``: 1 when no angular gap is wider than ``max_gap_deg``, falling linearly to 0 at ``max_gap_deg + 90``,
     so an unclosed arc scores low even when what was drawn is round;
   * ``size``: 1 when ``r`` is between ``min_radius`` and ``max_radius`` of the shorter canvas side and the centre is
     on the canvas, else 0 (a dot or a straight line fits a tiny or enormous "circle").

4. ``score = size * roundness * coverage * clean * closure``. ``passed`` needs ``score >= pass_score`` and no
   angular gap wider than ``max_gap_deg`` (the circle is closed).

The tolerances pass a circle walked with the environment's 16 fine moves and fail an octagon, lines, arcs, squares,
ellipses, filled discs, dots and scribbles; ``tests/test_circle_verifier.py`` pins both sides. ``rms`` (the RMS
distance of all ink from the fitted circle, over ``r``) is reported alongside for diagnosis.
"""
from typing import Dict, Sequence

import numpy as np


def ink_mask(pixels, bg: Sequence[int] = (255, 255, 255), ink_threshold: int = 60) -> np.ndarray:
    """A boolean ``(h, w)`` mask of pixels whose colour differs from ``bg`` by more than ``ink_threshold``."""
    arr = np.asarray(pixels)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    arr = arr[..., :3].astype(np.int16)
    return (np.abs(arr - np.asarray(bg, dtype=np.int16)).max(axis=2) > ink_threshold)


def fit_circle(xs: np.ndarray, ys: np.ndarray):
    """Least-squares (Kasa) circle through the points: ``(cx, cy, r)``."""
    xs, ys = xs.astype(np.float64), ys.astype(np.float64)
    a = np.stack([xs, ys, np.ones_like(xs)], axis=1)
    b = xs ** 2 + ys ** 2
    (p, q, c), *_ = np.linalg.lstsq(a, b, rcond=None)
    cx, cy = p / 2.0, q / 2.0
    return float(cx), float(cy), float(np.sqrt(max(c + cx ** 2 + cy ** 2, 0.0)))


def _max_gap_deg(occupied: np.ndarray) -> float:
    """Widest run of empty angular bins (wrapping around), in degrees."""
    n = len(occupied)
    if not occupied.any():
        return 360.0
    run = best = 0
    for v in np.concatenate([occupied, occupied]):
        run = 0 if v else run + 1
        best = max(best, run)
    return min(best, n) * 360.0 / n


def score_circle(pixels, bg: Sequence[int] = (255, 255, 255), ink_threshold: int = 60, min_ink: int = 30,
                 round_tol: float = 0.05, band: float = 0.2, bins: int = 36, min_radius: float = 0.08,
                 max_radius: float = 0.5, max_gap_deg: float = 30.0, pass_score: float = 0.7) -> Dict:
    """Score an ``(h, w, 3)`` canvas for one drawn circle. Returns ``score`` in [0, 1], ``passed``, every component,
    and the fitted ``cx, cy, r`` (see the module docstring)."""
    mask = ink_mask(pixels, bg, ink_threshold)
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    out = {"score": 0.0, "passed": False, "ink": int(len(xs)), "roundness": 0.0, "rms": None, "coverage": 0.0,
           "clean": 0.0, "closure": 0.0, "size": 0.0, "max_gap_deg": 360.0, "cx": None, "cy": None, "r": None}
    if len(xs) < min_ink:
        return out
    cx, cy, r = fit_circle(xs, ys)
    out.update(cx=round(cx, 2), cy=round(cy, 2), r=round(r, 2))
    side = min(h, w)
    size_ok = min_radius * side <= r <= max_radius * side and 0 <= cx < w and 0 <= cy < h
    if r <= 0:
        return out
    d = np.hypot(xs - cx, ys - cy)
    dev = np.abs(d - r)
    on_ring = dev <= max(band * r, 3.0)
    clean = float(on_ring.mean())
    ang = np.arctan2(ys[on_ring] - cy, xs[on_ring] - cx)
    idx = ((ang + np.pi) / (2 * np.pi) * bins).astype(int) % bins
    occupied = np.zeros(bins, dtype=bool)
    occupied[idx] = True
    counts = np.bincount(idx, minlength=bins)[occupied]
    radii = np.bincount(idx, weights=d[on_ring], minlength=bins)[occupied] / counts
    roundness = float(np.clip(1.0 - np.std(radii) / (round_tol * r), 0.0, 1.0))
    coverage = float(occupied.mean())
    gap = _max_gap_deg(occupied)
    closure = float(np.clip(1.0 - (gap - max_gap_deg) / 90.0, 0.0, 1.0))
    score = float(size_ok) * roundness * coverage * clean * closure
    rms = float(np.sqrt(np.mean(dev ** 2)) / r)
    out.update(score=round(score, 4), roundness=round(roundness, 4), rms=round(rms, 4), coverage=round(coverage, 4),
               clean=round(clean, 4), closure=round(closure, 4), size=float(size_ok), max_gap_deg=round(gap, 1),
               passed=bool(score >= pass_score and gap <= max_gap_deg))
    return out


__all__ = ["ink_mask", "fit_circle", "score_circle"]
