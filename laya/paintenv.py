"""JSPaint as an RL environment: the model sees screenshots of the real app and drives it with the mouse only.

`JSPaint <https://github.com/r33drichards/jspaint>`_ (an MS Paint clone) runs unmodified in headless Chromium through
Playwright; ``JSPaintServer`` serves a local checkout as static files. Each episode:

* **reset** clears the canvas to white (``api_for_cypress_tests.reset_for_next_test()``), selects the Brush tool
  (so strokes are 4 px wide and visible) and puts the cursor at a seeded point on the canvas;
* **observation** is a screenshot of the canvas with the cursor drawn on it (headless screenshots have no OS cursor):
  a red ring when the pen is up, a filled red dot when it is down. ``note()`` carries the pen state and step count
  as text;
* **actions** are mouse-only. ``env.actions`` is the discrete set the image model chooses from: a ``step_px`` move
  toward each of ``directions`` compass points (32 by default: ``N``, ``NbE``, ``NNE``, ... every 11.25 degrees, 6 px,
  fine enough to walk a round circle; 16 and 8 give coarser sets), plus ``PEN_DOWN``, ``PEN_UP`` and
  ``DONE``. Each maps onto the tool API ``move_mouse(dx, dy)`` / ``mouse_down()`` /
  ``mouse_up()``, which sends real pointer events to the page. ``TOOLS`` describes that API as JSON-schema tools, so
  a tool-calling agent can drive the same environment with ``call_tool``;
* **reward** is the task verifier on the true canvas pixels (``canvas_pixels()``, read from the page, so the cursor
  overlay never counts as ink): ``laya.circle_verifier.score_circle`` for ``task="circle"``. It is paid at the end of
  the episode, or as the per-step change in score with ``reward="shaped"``. An episode ends on ``DONE`` or after
  ``max_steps``.

``circle_expert`` (a scripted policy that walks a circle with the same moves), ``random_policy`` and
``model_policy`` (``VLMAgent.predict`` on the screenshot) are the reference points; ``play_episodes`` runs any of
them. ``examples/jspaint_circle.py`` is the command-line runner.

Needs ``pip install playwright pillow`` and a Chromium: set ``executable_path`` (or ``LAYA_CHROMIUM``) to use an
existing binary instead of ``playwright install``.
"""
import base64
import functools
import http.server
import io
import math
import os
import random
import threading
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np

COMPASS = ("N", "NbE", "NNE", "NEbN", "NE", "NEbE", "ENE", "EbN", "E", "EbS", "ESE", "SEbE", "SE", "SEbS", "SSE",
           "SbE", "S", "SbW", "SSW", "SWbS", "SW", "SWbW", "WSW", "WbS", "W", "WbN", "WNW", "NWbW", "NW", "NWbN", "NNW",
           "NbW")
PEN_ACTIONS = ("PEN_DOWN", "PEN_UP", "DONE")


def compass_bearing(name: str) -> float:
    """Degrees clockwise from straight up (``N``) of a ``COMPASS`` point."""
    return COMPASS.index(name) * 11.25


def compass_moves(directions: int = 32) -> Dict[str, Tuple[float, float]]:
    """``{name: (ux, uy)}`` unit vectors (screen y points down) for 8, 16 or 32 compass points, clockwise from ``N``
    (the 32-point names add the "by" points, e.g. ``NbE`` is one point east of north)."""
    if directions not in (8, 16, 32):
        raise ValueError("directions must be 8, 16 or 32")
    every = 32 // directions
    return {COMPASS[i]: (round(math.sin(math.radians(11.25 * i)), 6), round(-math.cos(math.radians(11.25 * i)), 6))
            for i in range(0, 32, every)}


MOVES = compass_moves(32)
ACTIONS = tuple(MOVES) + PEN_ACTIONS
TASKS = {"circle": "Draw a circle on the canvas."}

TOOLS = [
    {"name": "move_mouse", "description": "Move the mouse by (dx, dy) canvas pixels. If the button is held down, "
                                          "this draws a brush stroke along the way.",
     "input_schema": {"type": "object", "properties": {"dx": {"type": "integer"}, "dy": {"type": "integer"}},
                      "required": ["dx", "dy"]}},
    {"name": "mouse_down", "description": "Press the left mouse button (start drawing).",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "mouse_up", "description": "Release the left mouse button (stop drawing).",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "screenshot", "description": "Return a screenshot of the canvas with the cursor marked.",
     "input_schema": {"type": "object", "properties": {}}},
]

CURSOR = (230, 30, 30)


def default_chromium() -> Optional[str]:
    """``LAYA_CHROMIUM``, else the Claude Code sandbox's preinstalled Chromium, else ``None`` (Playwright's own)."""
    path = os.environ.get("LAYA_CHROMIUM") or "/opt/pw-browsers/chromium"
    return path if os.path.exists(path) else None


class JSPaintServer:
    """Serve a JSPaint checkout on ``127.0.0.1`` (a free port) from a background thread; ``.url`` is the app."""

    def __init__(self, root: str):
        if not os.path.exists(os.path.join(root, "index.html")):
            raise FileNotFoundError("no JSPaint checkout at %r (index.html missing)" % root)

        class Quiet(http.server.SimpleHTTPRequestHandler):
            def log_message(self, *args):
                pass

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Quiet, directory=root))
        self.url = "http://127.0.0.1:%d/" % self.httpd.server_port
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class JSPaintEnv:
    """One headless JSPaint page; ``reset`` starts an episode and ``step(action)`` plays one of ``env.actions``."""

    def __init__(self, url: str, task: str = "circle", step_px: int = 6, directions: int = 32, max_steps: int = 260,
                 reward: str = "terminal", headless: bool = True, executable_path: Optional[str] = None,
                 viewport: Tuple[int, int] = (800, 600), keep_frames: bool = False):
        if task not in TASKS:
            raise ValueError("unknown task %r (one of %s)" % (task, ", ".join(TASKS)))
        if reward not in ("terminal", "shaped"):
            raise ValueError("reward must be 'terminal' or 'shaped'")
        from playwright.sync_api import sync_playwright

        self.url, self.task, self.step_px, self.max_steps, self.reward_mode = url, task, step_px, max_steps, reward
        self.keep_frames, self.directions = keep_frames, directions
        self.moves = compass_moves(directions)
        self.actions = tuple(self.moves) + PEN_ACTIONS
        self._pw = sync_playwright().start()
        self.browser = self._pw.chromium.launch(headless=headless,
                                                executable_path=executable_path or default_chromium())
        self.page = self.browser.new_page(viewport={"width": viewport[0], "height": viewport[1]})
        # Only the app itself: no fonts, analytics or update checks from the network.
        self.page.route("**/*", lambda route: route.continue_() if route.request.url.startswith(url)
                        else route.abort())
        self.page.goto(url)
        self.page.wait_for_function("window.api_for_cypress_tests")
        self.episode, self.steps, self.pen, self.finished = -1, 0, False, False
        self.cursor = (0.0, 0.0)
        self.trajectory: List[Dict] = []
        self.frames: List = []
        self._last_score = 0.0
        self.result: Optional[Dict] = None

    # -- episode ----------------------------------------------------------------------------------------------
    def reset(self, seed: int = 0):
        """Blank canvas, Brush tool, cursor at a seeded point in the middle half of the canvas. Returns the
        observation."""
        self.page.mouse.up()
        self.page.evaluate("api_for_cypress_tests.reset_for_next_test()")
        self.page.click(".tool[title='Brush']")
        r = self.page.evaluate("(() => { const r = document.querySelector('.main-canvas').getBoundingClientRect();"
                               " return [r.x, r.y, r.width, r.height]; })()")
        self.origin, self.width, self.height = (r[0], r[1]), int(r[2]), int(r[3])
        rng = random.Random(seed)
        self.cursor = (rng.uniform(0.25, 0.75) * self.width, rng.uniform(0.25, 0.75) * self.height)
        self.page.mouse.move(*self._client(self.cursor))
        self.episode += 1
        self.seed, self.steps, self.pen, self.finished = seed, 0, False, False
        self.trajectory, self.frames, self._last_score, self.result = [], [], 0.0, None
        obs = self.render()
        if self.keep_frames:
            self.frames.append(obs)
        return obs

    @property
    def done(self) -> bool:
        return self.finished or self.steps >= self.max_steps

    def step(self, action: str):
        """Play one of ``env.actions``. Returns ``(observation, reward, done, info)``; ``info["verifier"]`` holds the
        full verifier result once the episode is over."""
        if action not in self.actions:
            raise ValueError("unknown action %r" % action)
        if self.done:
            raise RuntimeError("episode is over; call reset()")
        if action in self.moves:
            dx, dy = self.moves[action]
            self.move_mouse(dx * self.step_px, dy * self.step_px)
        elif action == "PEN_DOWN":
            self.mouse_down()
        elif action == "PEN_UP":
            self.mouse_up()
        else:
            self.finished = True
        self.steps += 1
        self.trajectory.append({"step": self.steps, "action": action, "x": round(self.cursor[0], 1),
                                "y": round(self.cursor[1], 1), "pen": self.pen})
        info: Dict = {}
        reward = 0.0
        if self.done:
            self.mouse_up()
            self.result = self.verify()
            info["verifier"] = self.result
            reward = self.result["score"] - (self._last_score if self.reward_mode == "shaped" else 0.0)
        elif self.reward_mode == "shaped":
            score = self.verify()["score"]
            reward, self._last_score = score - self._last_score, score
        obs = self.render()
        if self.keep_frames:
            self.frames.append(obs)
        return obs, reward, self.done, info

    def verify(self) -> Dict:
        from laya.circle_verifier import score_circle

        return score_circle(self.canvas_pixels())

    # -- mouse-only tool API ------------------------------------------------------------------------------------
    def _client(self, xy) -> Tuple[float, float]:
        return self.origin[0] + xy[0], self.origin[1] + xy[1]

    def move_mouse(self, dx: float, dy: float) -> None:
        """Move by ``(dx, dy)`` canvas pixels, clamped to the canvas; draws while the button is down."""
        x = min(max(self.cursor[0] + dx, 1.0), self.width - 2.0)
        y = min(max(self.cursor[1] + dy, 1.0), self.height - 2.0)
        self.cursor = (x, y)
        steps = max(2, int(math.hypot(dx, dy) // 4)) if self.pen else 1
        self.page.mouse.move(*self._client(self.cursor), steps=steps)

    def mouse_down(self) -> None:
        if not self.pen:
            self.page.mouse.down()
            self.pen = True

    def mouse_up(self) -> None:
        if self.pen:
            self.page.mouse.up()
            self.pen = False

    def call_tool(self, name: str, args: Optional[Dict] = None):
        """Run one of ``TOOLS`` by name (for tool-calling agents); ``screenshot`` returns the observation."""
        args = args or {}
        if name == "move_mouse":
            self.move_mouse(float(args["dx"]), float(args["dy"]))
        elif name == "mouse_down":
            self.mouse_down()
        elif name == "mouse_up":
            self.mouse_up()
        elif name == "screenshot":
            return self.render()
        else:
            raise ValueError("unknown tool %r" % name)
        return None

    # -- observations --------------------------------------------------------------------------------------------
    def screenshot(self, full_page: bool = False):
        """A PIL screenshot of the canvas (or the whole app with ``full_page``), without the cursor."""
        from PIL import Image

        clip = None if full_page else {"x": self.origin[0], "y": self.origin[1], "width": self.width,
                                       "height": self.height}
        return Image.open(io.BytesIO(self.page.screenshot(clip=clip))).convert("RGB")

    def render(self, full_page: bool = False):
        """The observation: a screenshot with the cursor drawn on it (ring = pen up, dot = pen down)."""
        from PIL import ImageDraw

        img = self.screenshot(full_page)
        x, y = self._client(self.cursor) if full_page else self.cursor
        d = ImageDraw.Draw(img)
        if self.pen:
            d.ellipse([x - 5, y - 5, x + 5, y + 5], fill=CURSOR)
        else:
            d.ellipse([x - 7, y - 7, x + 7, y + 7], outline=CURSOR, width=2)
            d.line([x - 11, y, x + 11, y], fill=CURSOR)
            d.line([x, y - 11, x, y + 11], fill=CURSOR)
        return img

    def note(self) -> str:
        return "Task: %s The pen is %s. Step %d of %d." % (TASKS[self.task], "down (drawing)" if self.pen else
                                                          "up (not drawing)", self.steps + 1, self.max_steps)

    def canvas_pixels(self) -> np.ndarray:
        """The true canvas as an ``(h, w, 3)`` uint8 array, read from the page (transparent pixels read as white)."""
        from PIL import Image

        data = self.page.evaluate("main_canvas.toDataURL('image/png')")
        img = Image.open(io.BytesIO(base64.b64decode(data.split(",", 1)[1]))).convert("RGBA")
        white = Image.new("RGBA", img.size, (255, 255, 255, 255))
        return np.asarray(Image.alpha_composite(white, img).convert("RGB"))

    def close(self) -> None:
        self.browser.close()
        self._pw.stop()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# -- policies ------------------------------------------------------------------------------------------------------
def _toward(moves, pos, target) -> str:
    """The move whose unit direction best matches ``target - pos``."""
    vx, vy = target[0] - pos[0], target[1] - pos[1]
    return max(moves, key=lambda a: moves[a][0] * vx + moves[a][1] * vy)


def _around(moves, pos, centre, r, step) -> str:
    """Of the moves that advance clockwise round ``centre`` by at least a third of a step, the one that lands
    closest to radius ``r`` (ties to the larger advance)."""
    ang = math.atan2(pos[1] - centre[1], pos[0] - centre[0])

    def outcome(a):
        x, y = pos[0] + moves[a][0] * step, pos[1] + moves[a][1] * step
        adv = (math.atan2(y - centre[1], x - centre[0]) - ang + math.pi) % (2 * math.pi) - math.pi
        return abs(math.hypot(x - centre[0], y - centre[1]) - r), -adv

    forward = [a for a in moves if -outcome(a)[1] >= step / r / 3] or list(moves)
    return min(forward, key=outcome)


def circle_expert(radius_frac: float = 0.3):
    """A scripted policy: walk (pen up) straight to the nearest point of a circle centred on the canvas, press once
    within half a step of its radius, trace one
    full turn plus a little overlap (each step takes the forward move that stays closest to the radius), release,
    ``DONE``. Deterministic given the environment state."""
    state: Dict = {}

    def policy(env: JSPaintEnv) -> str:
        if state.get("episode") != (id(env), env.episode):
            cx, cy = env.width / 2.0, env.height / 2.0
            r = radius_frac * min(env.width, env.height)
            state.clear()
            state.update(episode=(id(env), env.episode), c=(cx, cy), r=r, phase="approach", turned=0.0, prev=None)
        cx, cy = state["c"]
        r, pos, step = state["r"], env.cursor, env.step_px
        if state["phase"] == "approach":
            d = math.dist(pos, (cx, cy))
            if abs(d - r) > step * 0.5:
                ux, uy = ((pos[0] - cx) / d, (pos[1] - cy) / d) if d > 1e-6 else (1.0, 0.0)
                return _toward(env.moves, pos, (cx + r * ux, cy + r * uy))
            state["phase"] = "trace"
            state["prev"] = math.atan2(pos[1] - cy, pos[0] - cx)
            return "PEN_DOWN"
        if state["phase"] == "trace":
            ang = math.atan2(pos[1] - cy, pos[0] - cx)
            delta = (ang - state["prev"] + math.pi) % (2 * math.pi) - math.pi
            state["turned"] += delta
            state["prev"] = ang
            if state["turned"] < 2 * math.pi + 0.3:
                return _around(env.moves, pos, (cx, cy), r, step)
            state["phase"] = "lift"
            return "PEN_UP"
        return "DONE"

    return policy


def random_policy(seed: int = 0):
    rng = random.Random(seed)
    return lambda env: rng.choice(env.actions)


class ModelPolicy:
    """The model's most likely action from the screenshot plus the text note, via ``predict``. ``last`` keeps the
    latest answer (probabilities over ``env.actions``) for logging."""

    def __init__(self, agent, task: str = "circle"):
        self.agent, self.task, self.last, self.provenance, self._questions = agent, task, None, None, {}

    def __call__(self, env: JSPaintEnv) -> str:
        from laya.games import paint_question

        key = (env.directions, env.step_px)
        if key not in self._questions:
            self._questions[key] = paint_question(self.task, env.directions, env.step_px)
        out = self.agent.predict({"image": env.render(), "note": env.note()}, self._questions[key])
        self.last, self.provenance = out["answers"]["action"], out.get("provenance")
        return self.last["choice"]


def model_policy(agent, task: str = "circle") -> ModelPolicy:
    return ModelPolicy(agent, task)


def play_episodes(env: JSPaintEnv, policy, episodes: int, seed: int = 0, on_step=None) -> Dict:
    """Play ``episodes`` seeded episodes (seed ``seed + i``) with ``policy(env) -> action``. ``on_step(env, action)``
    is called after each step (for logging). Returns per-episode verifier results and the summary."""
    counts, eps = Counter(), []
    for i in range(episodes):
        env.reset(seed + i)
        while not env.done:
            a = policy(env)
            counts[a] += 1
            env.step(a)
            if on_step:
                on_step(env, a)
        eps.append({"seed": seed + i, "steps": env.steps, "ended": "done" if env.finished else "capped",
                    **env.result})
    scores = [e["score"] for e in eps]
    return {"task": env.task, "episodes": episodes, "seed": seed, "directions": env.directions,
            "step_px": env.step_px, "max_steps": env.max_steps,
            "mean_score": float(np.mean(scores)), "median_score": float(np.median(scores)),
            "pass_rate": float(np.mean([e["passed"] for e in eps])),
            "mean_steps": float(np.mean([e["steps"] for e in eps])), "actions": dict(counts), "results": eps}


__all__ = ["ACTIONS", "MOVES", "COMPASS", "PEN_ACTIONS", "compass_bearing", "compass_moves", "TASKS", "TOOLS",
           "JSPaintServer",
           "JSPaintEnv", "circle_expert", "random_policy",
           "ModelPolicy", "model_policy", "play_episodes", "default_chromium"]
