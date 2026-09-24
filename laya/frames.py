"""Game frame modes: how a game's recent screens become the model's state, in one place for training and play.

A game step is one ``choice`` question about a state built from the frame history ``[f_0, ..., f_t]`` of the
episode (the screens at the decision points so far, oldest first). The mode says how:

* ``"single"``: what every game did before modes existed. Maze, Snake, Atari and ViZDoom give the current frame
  alone; the classic-control games (``laya.controlgames``) give the current frame with the previous one ghosted in
  at weight ``GHOST``, which is exactly ``"trail-2"``.
* ``"trail-N"``: one image, the last ``N`` frames blended with fading weights, the current frame strongest. With
  age ``k`` (0 = current) and ``r = GHOST / (1 - GHOST)`` (0.35 / 0.65), frame ``k`` gets

      w_k = r**k / sum_{j<N} r**j  =  GHOST**k (1 - GHOST)**(N-1-k) / sum_j GHOST**j (1 - GHOST)**(N-1-j)

  (the second form is how it is computed), a geometric fade that sums to 1. ``trail-2`` is ``(0.65, 0.35)``,
  bit for bit today's control ghost; ``trail-3`` is ``(0.54, 0.29, 0.16)``, ``trail-4`` ``(0.50, 0.27, 0.15,
  0.08)``. Pixels are blended in float32 and rounded to uint8. Still one image per step.
* ``"stack-N"``: ``{"images": [f_{t-N+1}, ..., f_t]}``, oldest first: ``N`` images per step, each ``image_seq_len``
  tokens (64 for SmolVLM at 512).

At an episode's start, where fewer than ``N`` frames exist, the first frame is repeated (``window``); the Atari
auto-FIRE after a lost life starts the history again (``laya.atari_train.play``). ``stack-1`` and ``trail-1`` are
the current frame alone. ``N`` is at most ``MAX_FRAMES`` (the autoresearch pool stores 4 previous frames).

The mode is part of the model: a checkpoint names it in its config as ``game_frames``; older two-frame Atari
checkpoints say ``atari_frames: 2``, read as ``stack-2`` for Atari (``mode_for``).
"""
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

GHOST = 0.35     # weight of the previous frame in "single" control play (``laya.controlgames``) and trail-2
MAX_FRAMES = 5   # frames one state may span
FAMILIES = ("grid", "control", "atari", "doom")
_MODE = re.compile(r"^(trail|stack)-([1-9][0-9]*)$")


def parse(mode: str) -> Tuple[str, int]:
    """``"single"`` -> ``("single", 0)``; ``"trail-N"`` / ``"stack-N"`` -> ``(kind, N)``. Raises on anything else."""
    if mode == "single":
        return "single", 0
    m = _MODE.match(str(mode))
    if not m or not 1 <= int(m.group(2)) <= MAX_FRAMES:
        raise ValueError("game frame mode %r: expected 'single', 'trail-N' or 'stack-N' with 1 <= N <= %d"
                         % (mode, MAX_FRAMES))
    return m.group(1), int(m.group(2))


def resolve(mode: str, family: str) -> Tuple[str, int]:
    """The concrete ``(kind, N)`` a mode means for a game family: ``single`` is ``("trail", 2)`` for control and
    ``("stack", 1)`` (the current frame) elsewhere."""
    if family not in FAMILIES:
        raise ValueError("unknown game family %r (%s)" % (family, ", ".join(FAMILIES)))
    kind, n = parse(mode)
    if kind == "single":
        return ("trail", 2) if family == "control" else ("stack", 1)
    return kind, n


def frames_needed(mode: str, family: str) -> int:
    """How many frames of history a state in this mode looks at."""
    return resolve(mode, family)[1]


def images_per_state(mode: str, family: str) -> int:
    kind, n = resolve(mode, family)
    return n if kind == "stack" else 1


def mode_for(cfg: Optional[Dict], family: str) -> str:
    """A checkpoint's frame mode for ``family``: ``cfg["game_frames"]``, else ``stack-2`` for Atari when the old
    ``atari_frames`` key says 2, else ``single``. Validated."""
    cfg = cfg or {}
    mode = cfg.get("game_frames")
    if not mode:
        mode = "stack-2" if family == "atari" and int(cfg.get("atari_frames", 1) or 1) == 2 else "single"
    parse(mode)
    return mode


def trail_weights(n: int, ghost: float = GHOST) -> List[float]:
    """Blend weights by age, current frame first: ``w_k`` of the module docstring. Plain Python floats, so a
    float32 frame times a weight stays float32 (numpy 2's promotion rules), as ``ControlGame.render`` always was."""
    if n < 1:
        raise ValueError("a trail needs at least one frame")
    raw = [ghost ** k * (1.0 - ghost) ** (n - 1 - k) for k in range(n)]
    total = sum(raw)
    return [float(w / total) for w in raw]


def window(history: Sequence, n: int) -> List:
    """The last ``n`` entries of ``history`` (oldest first), left-padded with its first entry when shorter."""
    if not len(history):
        raise ValueError("empty frame history")
    h = list(history)[-n:]
    return [h[0]] * (n - len(h)) + h


def as_array(frame) -> np.ndarray:
    """A frame as an ``(h, w, 3)`` uint8 array: arrays pass through, PIL images, paths and encoded bytes are decoded."""
    if isinstance(frame, np.ndarray):
        return frame
    if isinstance(frame, str):  # a path
        from PIL import Image

        with Image.open(frame) as im:
            return np.asarray(im.convert("RGB"))
    if isinstance(frame, (bytes, bytearray, memoryview)):
        import io

        from PIL import Image

        with Image.open(io.BytesIO(frame)) as im:
            return np.asarray(im.convert("RGB"))
    return np.asarray(frame.convert("RGB") if hasattr(frame, "convert") else frame)


def blend(frames: Sequence, ghost: float = GHOST) -> np.ndarray:
    """``trail-N`` of ``frames`` (oldest first, N = ``len(frames)``) as a uint8 array; one frame comes back as is."""
    if all(f is frames[-1] for f in frames):  # one frame, or an episode's first step padded with itself
        return as_array(frames[-1])
    arrs = [as_array(f) for f in frames]
    w = trail_weights(len(arrs), ghost)
    mix = w[0] * arrs[-1].astype(np.float32)
    for k in range(1, len(arrs)):
        mix = mix + w[k] * arrs[-1 - k].astype(np.float32)
    return mix.round().astype(np.uint8)


def state(history: Sequence, mode: str, family: str, encode: Optional[Callable[[Any], Any]] = None) -> Dict:
    """The model's state for a frame history (oldest first, ending with the current frame, at least one frame).

    ``{"image": x}`` for one image (single frame or trail) or ``{"images": [...]}`` for ``stack-N`` with N > 1.
    Frames are passed through untouched where no blending happens (arrays, PIL images, encoded bytes or paths all
    work there); ``encode``, if given, maps every output image (e.g. to PNG bytes)."""
    enc = encode or (lambda x: x)
    kind, n = resolve(mode, family)
    win = window(history, n)
    if n == 1:
        return {"image": enc(win[-1])}
    if kind == "trail":
        return {"image": enc(blend(win))}
    return {"images": [enc(f) for f in win]}


def state_images(st: Dict) -> List:
    """The images of a state from ``state``, in order."""
    return [st["image"]] if "image" in st else list(st["images"])


class History:
    """One episode's frames at its decision points, oldest first, bounded to the last ``keep``."""

    def __init__(self, keep: int = MAX_FRAMES):
        self.keep, self.frames = keep, []

    def reset(self, frame=None) -> None:
        self.frames = [] if frame is None else [frame]

    def push(self, frame) -> None:
        self.frames = (self.frames + [frame])[-self.keep:]

    def state(self, mode: str, family: str, encode=None) -> Dict:
        return state(self.frames, mode, family, encode)

    def __len__(self) -> int:
        return len(self.frames)


def episode_policy(decide: Callable[[Any, Dict], Any], frame: Callable[[Any], Any], mode: str, family: str):
    """``policy(env)`` for play loops that ask once per decision of one live episode at a time
    (``laya.gridgames.play_episodes``, ``laya.controlgames.play_episodes``): it keeps the episode's frame history
    (``frame(env)`` at each decision point; a new env object starts a new episode) and returns
    ``decide(env, state)`` with ``state`` this mode's state of that history."""
    hist, cur = History(frames_needed(mode, family)), {"env": None, "steps": None}

    def policy(env):
        if env is not cur["env"]:
            hist.reset()
            cur.update(env=env, steps=None)
        if env.steps != cur["steps"]:
            hist.push(frame(env))
            cur["steps"] = env.steps
        return decide(env, hist.state(mode, family))

    return policy


__all__ = ["GHOST", "MAX_FRAMES", "FAMILIES", "parse", "resolve", "frames_needed", "images_per_state", "mode_for",
           "trail_weights", "window", "as_array", "blend", "state", "state_images", "History",
           "episode_policy"]
