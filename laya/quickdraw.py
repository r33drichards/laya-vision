"""Human doodles from Google's Quick, Draw! as drawing tasks for the JSPaint environment.

`Quick, Draw! <https://github.com/googlecreativelab/quickdraw-dataset>`_ has about 50 million doodles in 345
categories, each stored as the pen strokes a person drew, in order (CC BY 4.0). This module turns them into tasks
for ``laya.paintenv.JSPaintEnv`` ("draw a house"):

* ``fetch_drawings`` reads the first recognised doodles of a category from the simplified ndjson files (coordinates
  0-255, strokes simplified with Ramer-Douglas-Peucker), with an HTTP range request so a category costs a few
  megabytes, not the whole file.
* ``doodle_strokes`` scales a doodle to ``DOODLE_FRAC`` of the canvas, starts it where the cursor is (shifted only
  as far as needed to fit on the canvas, which is visible), and resamples every stroke to the environment's step.
* ``DoodleLabeller`` follows that doodle from any state: walk to the start of the next stroke, press, trace it with
  the compass moves, release, and ``DONE`` after the last stroke. If the pen strays, it lifts and returns to where
  the stroke left off, so it can label the states a model reaches after its own mistakes (DAgger). Its judgement
  labels come from the requested doodle: progress is the share of the doodle's length drawn, on track means little
  ink lies away from the doodle's strokes, and ``drawn`` is the category once the doodle is complete.
* ``render_strokes`` / ``to_bitmap`` rasterise doodles and canvases the same way for the independent scorer
  (``modal_app.py::train_quickdraw_classifier``), which grades a drawing by the probability a separately trained
  classifier gives the requested category, so the model is never graded by its own judgement.

The episode's doodle is not shown to the model: it only gets the category name, as a person would.
"""
import json
import math
import urllib.parse
import urllib.request
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from laya.circle_verifier import ink_mask
from laya.paintenv import _toward

BASE_URL = "https://storage.googleapis.com/quickdraw_dataset/full/simplified/%s.ndjson"
TRAIN_CATEGORIES = (
    "circle", "square", "triangle", "line", "zigzag", "star", "hexagon", "octagon", "diamond", "moon", "sun",
    "cloud", "house", "tree", "fish", "cup", "eye", "envelope", "umbrella", "key", "apple", "mountain",
    "smiley face", "hourglass",
)
HELDOUT_CATEGORIES = ("ladder", "stop sign", "cat", "flower", "lightning", "hat")
DOODLE_FRAC = 0.45  # the doodle's longer side, as a share of the canvas side
MAX_STROKES = 8
MARGIN = 8


def fetch_drawings(category: str, n: int, max_bytes: int = 4_000_000, skip: int = 0,
                   max_strokes: int = MAX_STROKES) -> List[Dict]:
    """The first ``n`` recognised doodles of ``category`` with at most ``max_strokes`` strokes, after skipping
    ``skip`` of them (so disjoint sets can be drawn from the same file). Each is ``{"key_id", "strokes": [[(x, y),
    ...], ...]}`` in the dataset's 0-255 coordinates."""
    url = BASE_URL % urllib.parse.quote(category)
    req = urllib.request.Request(url, headers={"Range": "bytes=0-%d" % (max_bytes - 1)})
    with urllib.request.urlopen(req, timeout=120) as r:
        text = r.read().decode("utf-8", errors="ignore")
    out, seen = [], 0
    for line in text.split("\n")[:-1]:  # the last line may be cut by the range
        d = json.loads(line)
        if not d.get("recognized") or len(d["drawing"]) > max_strokes:
            continue
        seen += 1
        if seen <= skip:
            continue
        out.append({"key_id": d["key_id"], "strokes": [list(zip(xs, ys)) for xs, ys in d["drawing"]]})
        if len(out) >= n:
            break
    return out


def _resample(points: Sequence[Tuple[float, float]], spacing: float) -> List[Tuple[float, float]]:
    """Points every ``spacing`` px along a polyline (always keeping its ends)."""
    pts = [tuple(map(float, p)) for p in points]
    if len(pts) == 1:
        return pts
    out, carry = [pts[0]], 0.0
    for a, b in zip(pts, pts[1:]):
        seg = math.dist(a, b)
        d = spacing - carry
        while d <= seg:
            t = d / seg
            out.append((a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])))
            d += spacing
        carry = seg - (d - spacing)
    if math.dist(out[-1], pts[-1]) > 1e-6:
        out.append(pts[-1])
    return out


def doodle_strokes(strokes: Sequence[Sequence[Tuple[float, float]]], side: int, start: Tuple[float, float],
                   step: float) -> List[List[Tuple[float, float]]]:
    """Canvas-space strokes: the doodle scaled so its longer side is ``DOODLE_FRAC * side``, placed so its first
    point is at ``start`` (shifted as little as needed to keep it ``MARGIN`` px inside the canvas), and resampled
    every ``step`` px."""
    pts = np.array([p for s in strokes for p in s], dtype=np.float64)
    lo, hi = pts.min(0), pts.max(0)
    scale = DOODLE_FRAC * side / max(1.0, float((hi - lo).max()))
    first = np.array(strokes[0][0], dtype=np.float64)
    off = np.array(start, dtype=np.float64) - first * scale
    blo, bhi = lo * scale + off, hi * scale + off
    off += np.maximum(0.0, MARGIN - blo) - np.maximum(0.0, bhi - (side - MARGIN))
    return [_resample([tuple(np.array(p) * scale + off) for p in s], step) for s in strokes]


def stroke_length(stroke: Sequence[Tuple[float, float]]) -> float:
    return sum(math.dist(a, b) for a, b in zip(stroke, stroke[1:]))


class DoodleLabeller:
    """Follows one human doodle (``play_labelled_episode``'s labeller interface). ``drawing`` is a
    ``fetch_drawings`` item; it is placed on the canvas at ``reset`` from where the cursor starts."""

    def __init__(self, category: str, drawing: Dict, shapes: Optional[Dict[str, str]] = None,
                 band: float = 10.0, stray_off_track: float = 0.15):
        self.category, self.drawing, self.band, self.stray_off_track = category, drawing, band, stray_off_track
        self.shapes = shapes

    def reset(self, env) -> None:
        self.env = env
        self.strokes = doodle_strokes(self.drawing["strokes"], env.canvas_size, env.cursor, env.step_px)
        self.i, self.j = 0, 0  # next stroke, next point in it
        self.lengths = [stroke_length(s) for s in self.strokes]
        self.total = max(1e-6, sum(self.lengths))
        self._target_mask = None

    # -- progress along the doodle -----------------------------------------------------------------------------
    def _advance(self, env) -> None:
        """Move the pointer past points the pen (down) has reached; start the next stroke once a finished one is
        released."""
        if self.i >= len(self.strokes):
            return
        s, step = self.strokes[self.i], env.step_px
        if env.pen:
            window = range(self.j, min(len(s), self.j + 4))
            reached = [k for k in window if math.dist(env.cursor, s[k]) <= 0.8 * step]
            if reached:
                self.j = reached[-1] + 1
        elif self.j >= len(s):
            self.i, self.j = self.i + 1, 0

    def drawn_fraction(self) -> float:
        done = sum(self.lengths[: self.i])
        if self.i < len(self.strokes) and self.j > 0:
            s = self.strokes[self.i]
            done += stroke_length(s[: min(self.j, len(s))])
        return min(1.0, done / self.total)

    @property
    def complete(self) -> bool:
        return self.i >= len(self.strokes)

    # -- labeller interface ------------------------------------------------------------------------------------
    def action(self, env, pixels: np.ndarray) -> str:
        self._advance(env)
        step, pos = env.step_px, env.cursor
        if self.complete:
            return "PEN_UP" if env.pen else "DONE"
        s = self.strokes[self.i]
        if env.pen:
            if self.j >= len(s):
                return "PEN_UP"
            if math.dist(pos, s[self.j]) > 3 * step:  # strayed: lift, then come back to where the stroke left off
                return "PEN_UP"
            return _toward(env.moves, pos, s[self.j])
        target = s[min(max(self.j - 1, 0), len(s) - 1)]  # the stroke's start, or where it was left off
        if math.dist(pos, target) <= 0.75 * step:
            return "PEN_DOWN"
        return _toward(env.moves, pos, target)

    def _stray(self, pixels: np.ndarray) -> float:
        """Share of the ink farther than ``band`` px from every stroke of the doodle."""
        ink = ink_mask(pixels)
        if not ink.any():
            return 0.0
        if self._target_mask is None or self._target_mask.shape != ink.shape:
            from PIL import Image, ImageDraw

            im = Image.new("L", (ink.shape[1], ink.shape[0]), 0)
            d = ImageDraw.Draw(im)
            for s in self.strokes:
                if len(s) > 1:
                    d.line(s, fill=255, width=int(2 * self.band))
                d.ellipse([s[0][0] - self.band, s[0][1] - self.band, s[0][0] + self.band, s[0][1] + self.band],
                          fill=255)
            self._target_mask = np.asarray(im) > 0
        return float((ink & ~self._target_mask).sum() / ink.sum())

    def judgements(self, pixels: np.ndarray) -> Dict[str, Optional[str]]:
        frac = self.drawn_fraction()
        ink = bool(ink_mask(pixels).any())
        stray = ink and self._stray(pixels) > self.stray_off_track
        if self.complete:
            progress = 4
        elif frac == 0 or not ink:
            progress = 0
        else:
            progress = 1 if frac < 0.4 else 2 if frac < 0.75 else 3
        if not ink:
            drawn = "nothing"
        elif self.complete and not stray and self.shapes and self.category in self.shapes:
            drawn = self.category
        else:
            drawn = None
        return {"progress": str(progress), "on_track": "off track" if stray else "on track", "drawn": drawn}

    def questions(self, task: str) -> Dict:
        from laya.games import paint_judgements

        return paint_judgements(task, shapes=self.shapes)


def drawn_options(categories: Sequence[str] = TRAIN_CATEGORIES) -> Dict[str, str]:
    """``drawn`` question options for doodle tasks: the categories, plus ``nothing``."""
    opts = {c: "a doodle of %s %s" % ("an" if c[0] in "aeiou" else "a", c) for c in categories}
    opts["nothing"] = "nothing, the canvas is blank"
    return opts


# -- rasterising for the independent classifier --------------------------------------------------------------------
BITMAP = 28


def to_bitmap(mask: np.ndarray, size: int = BITMAP) -> np.ndarray:
    """A boolean ink mask cropped to its ink, padded to a square with a 10% border, and resized to ``size`` px:
    ``(size, size)`` float32 in [0, 1]. Blank masks give zeros."""
    from PIL import Image

    ys, xs = np.nonzero(mask)
    if not len(xs):
        return np.zeros((size, size), np.float32)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    h, w = y1 - y0, x1 - x0
    side = int(max(h, w) * 1.2) + 2
    sq = np.zeros((side, side), np.uint8)
    oy, ox = (side - h) // 2, (side - w) // 2
    sq[oy:oy + h, ox:ox + w] = mask[y0:y1, x0:x1] * 255
    im = Image.fromarray(sq).resize((size, size), Image.BILINEAR)
    return np.asarray(im, np.float32) / 255.0


def render_strokes(strokes: Sequence[Sequence[Tuple[float, float]]], canvas: int = 256,
                   width: int = 4) -> np.ndarray:
    """A 0-255 doodle drawn as the environment would draw it at ``DOODLE_FRAC`` of a 512 canvas: an ink mask."""
    from PIL import Image, ImageDraw

    im = Image.new("L", (canvas, canvas), 0)
    d = ImageDraw.Draw(im)
    for s in strokes:
        if len(s) > 1:
            d.line([tuple(p) for p in s], fill=255, width=width, joint="curve")
        else:
            x, y = s[0]
            d.ellipse([x - width / 2, y - width / 2, x + width / 2, y + width / 2], fill=255)
    return np.asarray(im) > 0


def canvas_bitmap(pixels: np.ndarray) -> np.ndarray:
    """The classifier's input for a canvas (RGB pixels)."""
    return to_bitmap(ink_mask(pixels))


def make_classifier(n_classes: int):
    """A small CNN over ``BITMAP`` x ``BITMAP`` doodle bitmaps (the independent scorer)."""
    import torch.nn as nn

    return nn.Sequential(
        nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Dropout(0.3), nn.Linear(64 * 7 * 7, 256), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(256, n_classes))


def classifier_dataset(categories: Sequence[str], n_per: int, skip: int = 500, seed: int = 0):
    """Bitmaps and labels for the classifier: ``n_per`` doodles per category after the first ``skip`` (which the
    drawing episodes use), each rasterised with a random stroke width (3-6 px on a 256 canvas)."""
    import random

    rng = random.Random(seed)
    xs, ys = [], []
    for k, c in enumerate(categories):
        for d in fetch_drawings(c, n_per, max_bytes=12_000_000, skip=skip):
            xs.append(to_bitmap(render_strokes(d["strokes"], width=rng.randint(3, 6))))
            ys.append(k)
    return np.stack(xs).astype(np.float32), np.array(ys, dtype=np.int64)


def train_classifier(categories: Sequence[str], n_per: int = 2000, epochs: int = 8, device: str = "cpu",
                     seed: int = 0, val_frac: float = 0.1) -> Tuple[Dict, Dict]:
    """Train ``make_classifier`` on ``classifier_dataset``. Returns (checkpoint dict, metrics with held-back
    top-1 accuracy overall and per category)."""
    import torch

    torch.manual_seed(seed)
    x, y = classifier_dataset(categories, n_per, seed=seed)
    perm = np.random.default_rng(seed).permutation(len(y))
    n_val = int(len(y) * val_frac)
    va, tr = perm[:n_val], perm[n_val:]
    net = make_classifier(len(categories)).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    xt, yt = torch.from_numpy(x[tr])[:, None], torch.from_numpy(y[tr])
    for ep in range(epochs):
        net.train()
        order = torch.randperm(len(yt))
        for b in range(0, len(order), 256):
            idx = order[b:b + 256]
            loss = torch.nn.functional.cross_entropy(net(xt[idx].to(device)), yt[idx].to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        pred = net(torch.from_numpy(x[va])[:, None].to(device)).argmax(-1).cpu().numpy()
    per = {c: float((pred[y[va] == k] == k).mean()) for k, c in enumerate(categories) if (y[va] == k).any()}
    metrics = {"n_train": int(len(tr)), "n_val": int(n_val), "val_accuracy": float((pred == y[va]).mean()),
               "per_category": per, "epochs": epochs, "n_per": n_per}
    ckpt = {"categories": list(categories), "state_dict": {k: v.cpu() for k, v in net.state_dict().items()}}
    return ckpt, metrics


class DoodleScorer:
    """Grades a canvas for a category with the independent classifier (``modal_app.py::train_quickdraw_classifier``
    writes the checkpoint: ``{"categories", "state_dict"}``). ``score`` is P(category); ``passed`` when it is the
    classifier's top category. Usable as ``JSPaintEnv(scorer=...)``."""

    def __init__(self, path: str, device: str = "cpu"):
        import torch

        ck = torch.load(path, map_location=device)
        self.categories = list(ck["categories"])
        self.net = make_classifier(len(self.categories))
        self.net.load_state_dict(ck["state_dict"])
        self.net.eval().to(device)
        self.device = device

    def probabilities(self, pixels: np.ndarray) -> Dict[str, float]:
        import torch

        x = torch.from_numpy(canvas_bitmap(pixels))[None, None].to(self.device)
        with torch.no_grad():
            p = torch.softmax(self.net(x), -1)[0].cpu().numpy()
        return {c: float(v) for c, v in zip(self.categories, p)}

    def __call__(self, pixels: np.ndarray, task: str) -> Dict:
        ink = int(ink_mask(pixels).sum())
        if ink == 0:
            return {"score": 0.0, "passed": False, "top": None, "ink": 0}
        p = self.probabilities(pixels)
        top = max(p, key=p.get)
        return {"score": round(p.get(task, 0.0), 4), "passed": top == task, "top": top,
                "top_p": round(p[top], 4), "ink": ink}


__all__ = ["BASE_URL", "TRAIN_CATEGORIES", "HELDOUT_CATEGORIES", "DOODLE_FRAC", "fetch_drawings", "doodle_strokes",
           "stroke_length", "DoodleLabeller", "drawn_options", "to_bitmap", "render_strokes", "canvas_bitmap",
           "BITMAP", "make_classifier", "classifier_dataset", "train_classifier", "DoodleScorer"]
