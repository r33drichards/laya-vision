"""Labels and baselines of the DrivingBench offline eval (benchmarks/drivingbench_offline.py), on synthetic tracks."""
import importlib.util
import math
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "drivingbench_offline", Path(__file__).resolve().parents[1] / "benchmarks" / "drivingbench_offline.py")
db = importlib.util.module_from_spec(spec)
spec.loader.exec_module(db)


def arc(turn_deg_per_m, n=60, step=0.25):
    """Points driven north at `step` m per point, turning `turn_deg_per_m` degrees per metre (positive left)."""
    pts, x, y, h = [], 0.0, 0.0, 90.0  # h: math angle, 90 = north
    for i in range(n):
        pts.append({"t": i * 0.2, "x": x, "y": y, "speed_mps": 1.0, "progress": i / n})
        h += turn_deg_per_m * step
        x, y = x + step * math.cos(math.radians(h)), y + step * math.sin(math.radians(h))
    return pts


@pytest.mark.parametrize("rate,label", [(6.0, "left"), (-6.0, "right"), (0.0, "straight"), (1.0, "straight")])
def test_turn_sign_and_label(rate, label):
    deg, end = db.turn_deg(arc(rate), 0, 5.0)
    assert db.sign_direction(deg, 10.0) == label
    assert deg == pytest.approx(rate * 4.0, abs=2.0)  # first metre to last metre of 5 m: ~4 m of turning
    assert end is not None


def test_turn_needs_the_whole_stretch():
    assert db.turn_deg(arc(0.0, n=10), 0, 5.0) == (None, None)  # 2.25 m of track left


def test_command_in_force():
    calls = [
        {"t": 1.0, "tool": "observe"},
        {"t": 2.0, "tool": "set_motion", "arguments": {"direction": "left", "steering_percent": 40}},
        {"t": 5.0, "tool": "set_motion", "arguments": {"direction": "right", "steering_percent": 0}},
        {"t": 7.0, "tool": "set_motion", "arguments": {"direction": "right", "steering_percent": 30},
         "outcome": {"status": "rejected"}},
        {"t": 9.0, "tool": "stop_now", "arguments": {}},
    ]
    assert [db.command_at(calls, t) for t in (0.5, 3.0, 6.0, 8.0, 9.5)] == \
        ["stop", "left", "straight", "straight", "stop"]


def test_summary_counts_stop_as_wrong():
    rows = [
        {"label": "left", "laya_direction": "stop", "laya_top_direction": "left", "keep_steering": "left",
         "command": "left", "laya_p": {"left": 0.4, "straight": 0.1, "right": 0.1, "stop": 0.4}},
        {"label": "right", "laya_direction": "right", "laya_top_direction": "right", "keep_steering": "straight",
         "command": "right", "laya_p": {"left": 0.1, "straight": 0.1, "right": 0.7, "stop": 0.1}},
    ]
    s = db.summarize(rows)
    assert (s["laya_choice_acc"], s["laya_direction_acc"], s["laya_stop_rate"]) == (0.5, 1.0, 0.5)
    assert s["laya_mean_p_label"] == pytest.approx(0.55) and s["keep_steering_acc"] == 0.5
