"""BiGym demos as primitive labels (``laya.bigymdemos``): the waypoint cost, lookahead that undoes itself, and a
follower that completes a real demo with primitives alone. Needs mujoco + bigym; the follow test also needs the
demos cached in ~/.bigym (``laya.bigymdemos.demo_waypoints`` downloads them, about 120 MB)."""
import importlib.util
import pathlib

import numpy as np
import pytest

from laya import bigymdemos as bd

HAVE_SIM = all(importlib.util.find_spec(m) for m in ("mujoco", "bigym"))
sim = pytest.mark.skipif(not HAVE_SIM, reason="needs mujoco and bigym (and MUJOCO_GL=egl or osmesa)")
HAVE_DEMOS = (pathlib.Path.home() / ".bigym" / "demonstrations").exists()


def test_cost_is_zero_at_the_waypoint_and_wraps_yaw():
    w = np.zeros(len(bd.FIELDS))
    assert bd.cost(w, w) == 0.0
    s = w.copy()
    s[bd.FIELDS.index("yaw")] = 2 * np.pi - 0.1  # -0.1 rad, not 6.18
    assert bd.cost(s, w) == pytest.approx(bd.W_YAW * 0.1)
    s = w.copy()
    s[bd.FIELDS.index("gl")] = 1.0
    assert bd.cost(s, w) == pytest.approx(bd.W_GRIP)


@sim
def test_lookahead_leaves_the_game_unchanged():
    from laya import bigymgames as bg

    game = bg.BiGymGame("ReachTarget", seed=0, env=bg.make_env("ReachTarget", cameras=False))
    game.step("LEFT_HAND_UP")
    before, steps = bd._state(game), game.steps
    target = before.copy()
    target[bd.FIELDS.index("lz")] += 0.1
    assert bd.lookahead(game, target) == "LEFT_HAND_UP"
    assert game.steps == steps and np.allclose(bd._state(game), before, atol=1e-3)
    game.close()


@sim
@pytest.mark.skipif(not HAVE_DEMOS, reason="BiGym demos not downloaded")
def test_follower_completes_a_drawer_demo():
    demos = bd.demo_waypoints("DrawerTopClose", amount=5, seed=0)
    assert all(d["success_step"] is not None for d in demos)
    runs = [bd.follow("DrawerTopClose", d) for d in demos]
    assert sum(r["success"] for r in runs) >= 3  # 4/5 when written
    assert all(lab["primitive"] in bd.bg.PRIMITIVES for r in runs for lab in r["labels"])
