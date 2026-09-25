"""The DrivingBench driver's decision rules (examples/drivingbench_drive.py), without a model or a car."""
import importlib.util
from pathlib import Path

import pytest

from laya import driving

spec = importlib.util.spec_from_file_location(
    "drivingbench_drive", Path(__file__).resolve().parents[1] / "examples" / "drivingbench_drive.py")
drive = importlib.util.module_from_spec(spec)
spec.loader.exec_module(drive)
STOP, STRAIGHT, SHARP_LEFT = list(driving.MANEUVERS)[0], "go straight", "turn sharply left"


def answer(choice, p):
    rest = (1 - p) / (len(driving.MANEUVERS) - 1)
    return {"choice": choice, "probabilities": {m: (p if m == choice else rest) for m in driving.MANEUVERS}}


def test_confident_maneuver_moves():
    kind, target, reason = driving.decide(answer(SHARP_LEFT, 0.8), min_prob=0.5)
    assert (kind, target) == ("motion", ("left", 60))
    assert reason == "laya: turn sharply left (p=0.80)"


@pytest.mark.parametrize("choice,p", [(STOP, 0.9), (STOP, 0.2), (STRAIGHT, 0.49)])
def test_stop_choice_or_low_probability_stops(choice, p):
    kind, target, reason = driving.decide(answer(choice, p), min_prob=0.5)
    assert (kind, target) == ("stop", None)
    assert len(reason) <= 200  # the harness's reason limit


def test_question_offers_every_maneuver_and_the_objective():
    q = driving.question("Stop at the red cone.")["action"]
    assert q["type"] == "choice" and q["criteria"] == list(driving.MANEUVERS)
    assert "Stop at the red cone." in q["instructions"]
    for target in driving.MANEUVERS.values():
        assert target is None or (target[0] in ("left", "right", "straight") and 0 <= target[1] <= 100)


@pytest.mark.parametrize("summary,why", [
    ({"state": "held"}, ""),
    ({"state": "executing", "reason": "waiting_for_res"}, ""),
    ({"error": "observation_unavailable"}, "observation_unavailable"),
    ({"state": "held", "camera_reason": "camera_stale"}, "camera_stale"),
    ({"state": "unavailable", "reason": "native_offline"}, "native_offline"),
])
def test_unready(summary, why):
    assert drive.unready(summary) == why


def test_direction_probs_sum_the_maneuvers_per_direction():
    p = driving.direction_probs(answer(SHARP_LEFT, 0.6))
    assert p["left"] == pytest.approx(0.6 + 0.08) and p["stop"] == pytest.approx(0.08)
    assert sum(p.values()) == pytest.approx(1.0)
