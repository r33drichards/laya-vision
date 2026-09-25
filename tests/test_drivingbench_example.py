"""The DrivingBench driver's decision rules (examples/drivingbench_drive.py), without a model or a car."""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "drivingbench_drive", Path(__file__).resolve().parents[1] / "examples" / "drivingbench_drive.py")
drive = importlib.util.module_from_spec(spec)
spec.loader.exec_module(drive)
STOP, STRAIGHT, SHARP_LEFT = list(drive.MANEUVERS)[0], "go straight", "turn sharply left"


def answer(choice, p):
    rest = (1 - p) / (len(drive.MANEUVERS) - 1)
    return {"choice": choice, "probabilities": {m: (p if m == choice else rest) for m in drive.MANEUVERS}}


def test_confident_maneuver_moves():
    kind, target, reason = drive.decide(answer(SHARP_LEFT, 0.8), min_prob=0.5)
    assert (kind, target) == ("motion", ("left", 60))
    assert reason == "laya: turn sharply left (p=0.80)"


@pytest.mark.parametrize("choice,p", [(STOP, 0.9), (STOP, 0.2), (STRAIGHT, 0.49)])
def test_stop_choice_or_low_probability_stops(choice, p):
    kind, target, reason = drive.decide(answer(choice, p), min_prob=0.5)
    assert (kind, target) == ("stop", None)
    assert len(reason) <= 200  # the harness's reason limit


def test_question_offers_every_maneuver_and_the_objective():
    q = drive.question("Stop at the red cone.")["action"]
    assert q["type"] == "choice" and q["criteria"] == list(drive.MANEUVERS)
    assert "Stop at the red cone." in q["instructions"]
    for target in drive.MANEUVERS.values():
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
