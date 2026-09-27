"""Labelled training data for the JSPaint drawing task: actions and the model's own judgements, from any state.

The scripted ``laya.paintenv.circle_expert`` only works when it is in charge from the first step. Training data
needs labels for the states a learner actually reaches, including ones after its own mistakes, so this module
labels a state from what is on the canvas:

* **Target circle**: centred on the canvas, radius ``radius_frac`` of the canvas side (the task asks for a circle
  about a third of the canvas across).
* **Coverage**: which of ``SECTORS`` angular sectors around the centre hold ink within ``band`` of the ring, read
  from the canvas as displayed (``JSPaintEnv.visible_pixels``, which includes the stroke still being drawn).
  **Stray ink**: the share of ink farther than that from the ring.
* **Action** (``label_action``): pen down and still on the ring -> the forward move that stays closest to the
  radius (``paintenv._around``); pen down but drifted off the ring, or the ring complete -> ``PEN_UP``; pen up
  and the ring complete -> ``DONE``; pen up otherwise -> walk to the start of the nearest uncovered gap (so the
  clockwise stroke fills it) and press once within half a step of the radius. Stray ink is ignored: it cannot be
  erased with the mouse-only actions, so the labeller finishes the circle anyway.
* **Judgements** (``label_judgements``), for ``laya.games.paint_judgements``: ``progress`` 0-4 from coverage (0 no
  ring ink, 1 under 40%, 2 under 75%, 3 short of complete, 4 complete); ``on_track`` off track once stray ink is
  more than ``STRAY_OFF_TRACK`` of all ink; ``drawn`` nothing / arc / circle when the canvas is only ring ink
  (anything else is left unlabelled rather than guessed).

``play_labelled_episode`` runs one episode under a behaviour policy (the labeller with probability ``1 - eps``,
else a random action, optionally after ``random_prefix`` random steps, which gives off-track and recovery states)
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
STRAY_OFF_TRACK = 0.1
DIRECTION_SPREAD = 0.1  # target probability on each neighbouring compass point of the labelled move


def target_circle(width: int, height: int, radius_frac: float = 0.3) -> Tuple[float, float, float]:
    return width / 2.0, height / 2.0, radius_frac * min(width, height)


def ring_state(pixels: np.ndarray, circle, band: Optional[float] = None) -> Dict:
    """Coverage of the target ring and stray ink, from the true canvas pixels."""
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


def _sector(angle: float) -> int:
    return int((angle + math.pi) / (2 * math.pi) * SECTORS) % SECTORS


def _gap_start(covered: np.ndarray, pos, circle) -> Tuple[float, float]:
    """The ring point where the nearest uncovered run begins, going clockwise (increasing screen angle)."""
    cx, cy, r = circle
    if not covered.any():  # nothing yet: start at the ring point nearest the cursor
        d = math.dist(pos, (cx, cy))
        ux, uy = ((pos[0] - cx) / d, (pos[1] - cy) / d) if d > 1e-6 else (1.0, 0.0)
        return cx + r * ux, cy + r * uy
    starts = [k for k in range(SECTORS) if not covered[k] and covered[k - 1]]
    pts = []
    for k in starts:
        a = (k + 0.25) / SECTORS * 2 * math.pi - math.pi  # a little into the uncovered sector
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return min(pts, key=lambda p: math.dist(p, pos))


def label_action(env, pixels: np.ndarray, circle) -> str:
    """The labeller's action for the environment's current state (see the module docstring)."""
    cx, cy, r = circle
    ring = ring_state(pixels, circle)
    step, pos = env.step_px, env.cursor
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


def label_judgements(pixels: np.ndarray, circle) -> Dict[str, Optional[str]]:
    """Labels for ``paint_judgements``: ``progress`` (level index as a string), ``on_track`` and ``drawn`` (None
    when the canvas is not a clean ring or part of one)."""
    ring = ring_state(pixels, circle)
    cov, complete = ring["coverage"], bool(ring["covered"].all())
    if ring["ink"] == 0 or cov == 0:
        progress = 0
    elif complete:
        progress = 4
    else:
        progress = 1 if cov < 0.4 else 2 if cov < 0.75 else 3
    stray = ring["ink"] > 0 and ring["stray"] > STRAY_OFF_TRACK
    if ring["ink"] == 0:
        drawn = "nothing"
    elif not stray and complete:
        drawn = "circle"
    elif not stray and cov > 0:
        drawn = "arc"
    else:
        drawn = None
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
                          judge_every: int = 3, task: str = "circle") -> Tuple[List[Dict], Dict]:
    """Play one episode and return (training records, episode summary).

    ``save_frame(name, image) -> path`` stores an observation and returns the path the records should reference.
    Each step yields an action record; every ``judge_every`` steps also one record per labelled judgement."""
    from laya.games import paint_judgements, paint_question

    rng = random.Random(seed)
    circle = target_circle(env.canvas_size, env.canvas_size)
    q_act = paint_question(task, env.directions, env.step_px)["action"]
    options = list(q_act["criteria"])
    q_judge = paint_judgements(task)
    records: List[Dict] = []
    env.reset(seed)
    frame_paths: List[str] = []

    def frame_path(i: int) -> str:
        while len(frame_paths) <= i:
            k = len(frame_paths)
            frame_paths.append(save_frame("%s-%03d" % (rid, k), env._recent_obs[-1]))
        return frame_paths[i]

    while not env.done:
        t = env.steps
        pixels = env.visible_pixels()  # includes the stroke in progress (canvas_pixels would not)
        now = frame_path(t)
        prev = frame_paths[max(0, t - (len(env._recent_obs) - 1))]
        base = {"images": [prev, now], "state_text": env.state_text()}
        label = label_action(env, pixels, circle)
        records.append(dict(base, id="%s-%03d-action" % (rid, t), question=_question_record(q_act),
                            label=options.index(label), target=direction_target(options, label)))
        if t % judge_every == 0:
            for name, lab in label_judgements(pixels, circle).items():
                if lab is None:
                    continue
                q = q_judge[name]
                crit = q["criteria"]
                idx = int(lab) if q["type"] == "score" else list(crit).index(lab)
                records.append(dict(base, id="%s-%03d-%s" % (rid, t, name), question=_question_record(q), label=idx))
        if t < random_prefix or rng.random() < eps:
            act = rng.choice([a for a in env.actions if a != "DONE"])
        else:
            act = label
        env.step(act)
        if not env.done:
            frame_path(env.steps)
    summary = {"seed": seed, "eps": eps, "random_prefix": random_prefix, "steps": env.steps,
               "records": len(records), "verifier": env.result["score"],
               "final": label_judgements(env.visible_pixels(), circle)}
    return records, summary


class LabellerPolicy:
    """``label_action`` as a policy (for checking that the labeller draws a circle on its own)."""

    def __call__(self, env) -> str:
        return label_action(env, env.visible_pixels(), target_circle(env.canvas_size, env.canvas_size))


__all__ = ["SECTORS", "target_circle", "ring_state", "label_action", "label_judgements", "direction_target",
           "play_labelled_episode", "LabellerPolicy"]
