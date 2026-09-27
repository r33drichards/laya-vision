"""Labelled training data for the JSPaint drawing task: actions and the model's own judgements, from any state.

The scripted ``laya.paintenv.circle_expert`` only works when it is in charge from the first step. Training data
needs labels for the states a learner actually reaches, including ones after its own mistakes, so this module
labels a state from what is on the canvas:

* **Target circle, read off the canvas**: the circle is drawn clockwise from its top, so any arc on the canvas
  implies its circle: radius ``RADIUS_FRAC`` of the canvas side (about a third of the canvas across) and centre
  one radius below the topmost ink (``implied_circle``). With nothing drawn yet, the pen's position is the top.
  Nothing about the target is invisible: an earlier labeller used a fixed circle centred on the canvas, and a
  model trained on it could not see where to press the button and walked past the start.
* **Coverage**: which of ``SECTORS`` angular sectors around that centre hold ink within ``band`` of the ring, read
  from the canvas as displayed (``JSPaintEnv.visible_pixels``, which includes the stroke still being drawn).
  **Stray ink**: the share of ink farther than that from the ring.
* **Action** (``label_action``): nothing drawn and the pen up -> press if a circle starting here fits on the
  canvas, else move toward the nearest start that fits (away from the edges, which are visible); pen down and on
  the ring -> the forward (clockwise) move that stays closest to the radius (``paintenv._around``); pen down but
  drifted off the ring, or the ring complete -> ``PEN_UP``; pen up and the ring complete -> ``DONE``; pen up with
  a partial ring -> walk to the end of the arc (the start of the uncovered gap, clockwise) and press there.
  Stray ink cannot be erased with the mouse-only actions, so the labeller finishes the circle anyway.
* **Judgements** (``label_judgements``), for ``laya.games.paint_judgements``, against the implied circle:
  ``progress`` 0-4 from coverage (0 nothing, 1 under 40%, 2 under 75%, 3 short of complete, 4 complete);
  ``on_track`` off track once stray ink is more than ``STRAY_OFF_TRACK`` of all ink or the implied circle does
  not fit on the canvas; ``drawn`` nothing / arc / circle when the canvas is only ring ink (anything else is left
  unlabelled rather than guessed).

``play_labelled_episode`` runs one episode under a behaviour policy (the labeller with probability ``1 - eps``,
else a random action, optionally after ``random_prefix`` random pen-up moves that put the cursor somewhere else)
and returns training records in the ``laya.vlm_train.load_jsonl_examples`` format. The action label is always the
labeller's action for the state, not the action the behaviour policy took (DAgger-style), with the probability
spread to the two neighbouring compass points (the moves on either side are nearly as good).
"""
import math
import random
from typing import Dict, List, Optional, Tuple

import numpy as np

from laya.circle_verifier import ink_mask
from laya.paintenv import COMPASS, PEN_ACTIONS, _around, _toward

SECTORS = 36
RADIUS_FRAC = 0.18
TOP_BAND = 8  # px below the topmost ink used to find the top of the circle
MARGIN = 8  # px a circle keeps from the canvas edge
STRAY_OFF_TRACK = 0.1
DIRECTION_SPREAD = 0.1  # target probability on each neighbouring compass point of the labelled move


def radius(side: int) -> float:
    return RADIUS_FRAC * side


def target_circle(width: int, height: int, radius_frac: float = 0.3) -> Tuple[float, float, float]:
    """A fixed circle centred on the canvas (used by ``examples/jspaint_judge_check.py`` for the old expert)."""
    return width / 2.0, height / 2.0, radius_frac * min(width, height)


def implied_circle(pixels: np.ndarray, pen_pos=None, side: Optional[int] = None):
    """The circle the canvas implies. Its radius is fixed (``radius``) and it is drawn clockwise from its top, so
    its top is where the drawing starts: near the topmost ink the ring follows ``y = top + (x - x0)^2 / 2r``, and a
    least-squares fit of that parabola (its curvature is known) to the ink within ``TOP_BAND`` px of the topmost
    ink gives ``x0`` whichever way the arc runs off. The centre is one radius below. The estimate depends only on
    the start of the drawing, so it holds steady while the rest is drawn. With no ink, the circle whose top is
    ``pen_pos`` (or None when that is not given)."""
    side = side or min(pixels.shape[:2])
    r = radius(side)
    ys, xs = np.nonzero(ink_mask(pixels))
    if not len(xs):
        if pen_pos is None:
            return None
        return float(pen_pos[0]), float(pen_pos[1] + r), r
    top = ys.min()
    band = ys <= top + TOP_BAND
    x, y = xs[band].astype(np.float64), ys[band].astype(np.float64)
    x0 = float(x.mean())
    if x.max() - x.min() >= 4:
        a = 1.0 / (2 * r)
        slope, _ = np.polyfit(x, y - a * x * x, 1)  # y - a x^2 = -2 a x0 x + c
        x0 = float(np.clip(-slope / (2 * a), x.min(), x.max()))
    return x0, float(top + 2 + r), r  # +2: half the 4 px brush above the path


def fits(circle, side: int) -> bool:
    cx, cy, r = circle
    return MARGIN <= cx - r and cx + r <= side - MARGIN and MARGIN <= cy - r and cy + r <= side - MARGIN


def start_point(pos, side: int) -> Tuple[float, float]:
    """The nearest cursor position from which a circle drawn clockwise from its top fits on the canvas."""
    r = radius(side)
    return (min(max(pos[0], MARGIN + r + 1), side - MARGIN - r - 1),
            min(max(pos[1], MARGIN + 1), side - MARGIN - 2 * r - 1))


def ring_state(pixels: np.ndarray, circle, band: Optional[float] = None) -> Dict:
    """Coverage of the ring and stray ink, from the canvas pixels."""
    cx, cy, r = circle
    band = band if band is not None else max(4.0, 0.08 * r)
    ys, xs = np.nonzero(ink_mask(pixels))
    covered = np.zeros(SECTORS, dtype=bool)
    if len(xs) == 0:
        return {"covered": covered, "coverage": 0.0, "ink": 0, "stray": 0.0}
    d = np.hypot(xs - cx, ys - cy)
    on = np.abs(d - r) <= band
    ang = np.arctan2(ys[on] - cy, xs[on] - cx)
    covered[((ang + np.pi) / (2 * np.pi) * SECTORS).astype(int) % SECTORS] = True
    return {"covered": covered, "coverage": float(covered.mean()), "ink": int(len(xs)),
            "stray": float(1.0 - on.mean())}


def _gap_start(covered: np.ndarray, pos, circle) -> Tuple[float, float]:
    """The ring point where the nearest uncovered run begins, going clockwise (increasing screen angle)."""
    cx, cy, r = circle
    starts = [k for k in range(SECTORS) if not covered[k] and covered[k - 1]] or [
        k for k in range(SECTORS) if not covered[k]]
    pts = []
    for k in starts:
        a = (k + 0.25) / SECTORS * 2 * math.pi - math.pi  # a little into the uncovered sector
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return min(pts, key=lambda p: math.dist(p, pos))


def label_action(env, pixels: np.ndarray) -> str:
    """The labeller's action for the environment's current state (see the module docstring)."""
    side, step, pos = env.canvas_size, env.step_px, env.cursor
    circle = implied_circle(pixels, pos if env.pen else None, side)
    if circle is None:  # nothing drawn, pen up: press where a circle fits, else move to where one does
        target = start_point(pos, side)
        if math.dist(pos, target) <= 0.75 * step:
            return "PEN_DOWN"
        return _toward(env.moves, pos, target)
    cx, cy, r = circle
    ring = ring_state(pixels, circle)
    complete = bool(ring["covered"].all())
    off_ring = abs(math.dist(pos, (cx, cy)) - r)
    if env.pen:
        if complete or off_ring > 1.5 * step:
            return "PEN_UP"
        return _around(env.moves, pos, (cx, cy), r, step)
    if complete:
        return "DONE"
    target = _gap_start(ring["covered"], pos, circle)
    if math.dist(pos, target) <= 0.75 * step and off_ring <= 0.5 * step:
        return "PEN_DOWN"
    return _toward(env.moves, pos, target)


def label_judgements(pixels: np.ndarray) -> Dict[str, Optional[str]]:
    """Labels for ``paint_judgements`` against the implied circle: ``progress`` (level index as a string),
    ``on_track`` and ``drawn`` (None when the canvas is not a clean ring or part of one)."""
    side = min(pixels.shape[:2])
    circle = implied_circle(pixels, None, side)
    if circle is None:
        return {"progress": "0", "on_track": "on track", "drawn": "nothing"}
    ring = ring_state(pixels, circle)
    cov, complete = ring["coverage"], bool(ring["covered"].all())
    progress = 4 if complete else 0 if cov == 0 else 1 if cov < 0.4 else 2 if cov < 0.75 else 3
    stray = ring["stray"] > STRAY_OFF_TRACK or not fits(circle, side)
    drawn = None if stray else "circle" if complete else "arc"
    return {"progress": str(progress), "on_track": "off track" if stray else "on track", "drawn": drawn}


def direction_target(options: List[str], action: str) -> List[float]:
    """A soft target over the action question's options: the labelled move, with ``DIRECTION_SPREAD`` on each
    neighbouring compass point among ``options``; one-hot for pen actions."""
    t = [0.0] * len(options)
    idx = options.index(action)
    if action in PEN_ACTIONS:
        t[idx] = 1.0
        return t
    moves = [o for o in options if o not in PEN_ACTIONS]
    order = sorted(moves, key=COMPASS.index)
    i = order.index(action)
    for nb in (order[(i - 1) % len(order)], order[(i + 1) % len(order)]):
        t[options.index(nb)] += DIRECTION_SPREAD
    t[idx] = 1.0 - sum(t)
    return t


def _question_record(q: Dict) -> Dict:
    return {"type": q["type"], "instructions": q["instructions"], "criteria": q["criteria"]}


def play_labelled_episode(env, seed: int, rid: str, save_frame, eps: float = 0.0, random_prefix: int = 0,
                          judge_every: int = 3, task: str = "circle", messy: bool = False, driver=None,
                          beta: float = 0.5, labeller=None) -> Tuple[List[Dict], Dict]:
    """Play one episode and return (training records, episode summary).

    ``save_frame(name, image) -> path`` stores an observation and returns the path the records should reference.
    Each step yields an action record; every ``judge_every`` steps, and once for the final canvas, also one record
    per labelled judgement. With ``messy`` the ``random_prefix`` steps may press the button too, leaving stray ink
    to judge and recover from; otherwise they are pen-up moves that only relocate the cursor.

    ``driver(env) -> action`` (e.g. a trained ``paintenv.ModelPolicy``) takes the step with probability ``1 - beta``
    instead of the labeller (DAgger): the episode then visits the states the model itself reaches, including its
    mistakes, and every one of them is still labelled with the labeller's action.

    ``labeller`` supplies the labels: ``reset(env)`` after the environment resets, ``action(env, pixels)``,
    ``judgements(pixels)`` and ``questions(task)`` (the judgement questions). ``CircleLabeller`` by default;
    ``laya.quickdraw.DoodleLabeller`` follows a human Quick, Draw! doodle."""
    from laya.games import paint_question

    labeller = labeller or CircleLabeller()
    rng = random.Random(seed)
    q_act = paint_question(task, env.directions, env.step_px)["action"]
    options = list(q_act["criteria"])
    q_judge = labeller.questions(task)
    records: List[Dict] = []
    env.reset(seed)
    labeller.reset(env)
    frame_paths: List[str] = []

    def frame_path(i: int) -> str:
        while len(frame_paths) <= i:
            k = len(frame_paths)
            frame_paths.append(save_frame("%s-%03d" % (rid, k), env._recent_obs[-1]))
        return frame_paths[i]

    def judge_records(base, pixels, t):
        for name, lab in labeller.judgements(pixels).items():
            if lab is None:
                continue
            q = q_judge[name]
            idx = int(lab) if q["type"] == "score" else list(q["criteria"]).index(lab)
            records.append(dict(base, id="%s-%03d-%s" % (rid, t, name), question=_question_record(q), label=idx))

    def base_state(t):
        now = frame_path(t)
        prev = frame_paths[max(0, t - (len(env._recent_obs) - 1))]
        return {"images": [prev, now], "state_text": env.state_text()}

    while not env.done:
        t = env.steps
        pixels = env.visible_pixels()  # includes the stroke in progress (canvas_pixels would not)
        base = base_state(t)
        label = labeller.action(env, pixels)
        records.append(dict(base, id="%s-%03d-action" % (rid, t), question=_question_record(q_act),
                            label=options.index(label), target=direction_target(options, label)))
        if t % judge_every == 0:
            judge_records(base, pixels, t)
        if t < random_prefix:  # relocate the cursor; with messy, stray strokes too
            act = rng.choice([a for a in env.actions if a != "DONE"] if messy else list(env.moves))
        elif rng.random() < eps:
            act = rng.choice([a for a in env.actions if a != "DONE"])
        elif driver is not None and rng.random() >= beta:
            act = driver(env)
        else:
            act = label
        env.step(act)
        if not env.done:
            frame_path(env.steps)
    frame_path(env.steps)  # the final canvas, judged once more (it is where "complete" and "circle" are true)
    judge_records(base_state(env.steps), env.visible_pixels(), env.steps)
    summary = {"seed": seed, "eps": eps, "random_prefix": random_prefix, "messy": messy,
               "driver": driver is not None, "beta": beta if driver is not None else None, "steps": env.steps,
               "records": len(records), "verifier": env.result["score"],
               "final": labeller.judgements(env.visible_pixels())}
    return records, summary


class CircleLabeller:
    """The circle labeller (``label_action`` / ``label_judgements``) in the interface ``play_labelled_episode``
    takes."""

    def reset(self, env) -> None:
        pass

    def action(self, env, pixels: np.ndarray) -> str:
        return label_action(env, pixels)

    def judgements(self, pixels: np.ndarray) -> Dict[str, Optional[str]]:
        return label_judgements(pixels)

    def questions(self, task: str) -> Dict:
        from laya.games import paint_judgements

        return paint_judgements(task)


class LabellerPolicy:
    """``label_action`` as a policy (for checking that the labeller draws a circle on its own)."""

    def __call__(self, env) -> str:
        return label_action(env, env.visible_pixels())


__all__ = ["SECTORS", "RADIUS_FRAC", "radius", "implied_circle", "fits", "start_point", "target_circle",
           "ring_state", "label_action", "label_judgements", "direction_target",
           "play_labelled_episode", "CircleLabeller", "LabellerPolicy"]
