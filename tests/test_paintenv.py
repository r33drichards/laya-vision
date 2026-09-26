"""The JSPaint environment end to end in headless Chromium: mouse actions reach the real canvas, and the scripted
circle expert passes the verifier. Skipped without Playwright, a Chromium, or a JSPaint checkout (JSPAINT_DIR, default
a sibling of this repo)."""
import os

import numpy as np
import pytest

pytest.importorskip("playwright")
pytest.importorskip("PIL")

from laya.circle_verifier import ink_mask  # noqa: E402
from laya.games import paint_question  # noqa: E402
from laya.paintenv import ACTIONS, TOOLS, compass_moves, JSPaintEnv, JSPaintServer, circle_expert, play_episodes, \
    random_policy  # noqa: E402

JSPAINT = os.environ.get("JSPAINT_DIR", os.path.join(os.path.dirname(__file__), "..", "..", "jspaint"))
if not os.path.exists(os.path.join(JSPAINT, "index.html")):
    pytest.skip("no JSPaint checkout at %s" % JSPAINT, allow_module_level=True)


@pytest.fixture(scope="module")
def env():
    server = JSPaintServer(JSPAINT)
    try:
        e = JSPaintEnv(server.url)
    except Exception as err:  # no Chromium installed
        server.close()
        pytest.skip("cannot launch Chromium: %s" % err)
    yield e
    e.close()
    server.close()


def test_question_covers_actions_and_tools_are_mouse_only():
    assert set(paint_question()["action"]["criteria"]) == set(ACTIONS)
    assert len(ACTIONS) == 32 + 3
    assert set(paint_question(directions=8)["action"]["criteria"]) == set(compass_moves(8)) | {"PEN_DOWN", "PEN_UP",
                                                                                                "DONE"}
    assert set(compass_moves(16)) < set(compass_moves(32))
    for ux, uy in compass_moves(32).values():
        assert abs(ux * ux + uy * uy - 1) < 1e-5
    assert {t["name"] for t in TOOLS} == {"move_mouse", "mouse_down", "mouse_up", "screenshot"}


def test_reset_is_blank_and_observation_is_canvas(env):
    obs = env.reset(seed=3)
    assert obs.size == (env.width, env.height) == (512, 512)
    assert not ink_mask(env.canvas_pixels()).any()
    assert not env.done and not env.pen


def test_pen_down_moves_draw_and_pen_up_moves_do_not(env):
    env.reset(seed=0)
    x0, y0 = env.cursor
    env.step("PEN_DOWN")
    for _ in range(4):
        env.step("E")
    env.step("PEN_UP")
    for _ in range(3):
        env.step("S")
    ink = ink_mask(env.canvas_pixels())
    ys, xs = np.nonzero(ink)
    assert len(xs) > 50
    assert abs(np.median(ys) - y0) <= 3  # one horizontal stroke on the starting row
    assert xs.min() >= x0 - 5 and xs.max() <= x0 + 4 * env.step_px + 5
    assert env.cursor == (x0 + 4 * env.step_px, y0 + 3 * env.step_px)


def test_cursor_is_clamped_to_canvas(env):
    env.reset(seed=0)
    for _ in range(200):
        env.step("NW")
    assert env.cursor == (1.0, 1.0)


def test_done_ends_episode_with_verifier_result(env):
    env.reset(seed=0)
    _, reward, done, info = env.step("DONE")
    assert done and reward == 0.0 and info["verifier"]["score"] == 0.0
    with pytest.raises(RuntimeError):
        env.step("N")


def test_call_tool_draws(env):
    env.reset(seed=1)
    env.call_tool("mouse_down")
    env.call_tool("move_mouse", {"dx": 60, "dy": 30})
    env.call_tool("mouse_up")
    assert ink_mask(env.canvas_pixels()).sum() > 50
    assert env.call_tool("screenshot").size == (env.width, env.height)


def test_expert_passes_and_random_does_not(env):
    expert = play_episodes(env, circle_expert(), episodes=2, seed=0)
    assert expert["pass_rate"] == 1.0 and expert["mean_score"] >= 0.9
    assert all(e["roundness"] >= 0.9 for e in expert["results"])
    rnd = play_episodes(env, random_policy(0), episodes=3, seed=0)
    assert rnd["pass_rate"] == 0.0 and rnd["mean_score"] < expert["mean_score"]
