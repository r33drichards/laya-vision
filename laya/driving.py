"""The driving policy shared by the DrivingBench driver (examples/drivingbench_drive.py) and its offline eval
(benchmarks/drivingbench_offline.py): one `choice` question over a few maneuvers, read from the camera frame."""
from typing import Dict, Optional, Tuple

# maneuver -> (direction, steering_percent); None means stop_now
MANEUVERS: Dict[str, Optional[Tuple[str, int]]] = {
    "stop: a cone, person, vehicle or wall is close ahead, or the way is unclear": None,
    "go straight": ("straight", 0),
    "turn gently left": ("left", 25),
    "turn sharply left": ("left", 60),
    "turn gently right": ("right", 25),
    "turn sharply right": ("right", 60),
}
DIRECTIONS = ("left", "straight", "right")

# The DrivingBench v1 cone course, from the objective in the benchmark's prompt, shortened to fit head_max_len.
CONE_COURSE = ("Drive through the course marked by small multicolored mini-cones in a backwards-U parking lot, staying "
               "between the cones, and park in the area marked by blue mini-cones at the end.")


def question(objective: str) -> Dict[str, Dict]:
    return {
        "action": {
            "type": "choice",
            "instructions": "You are driving a car at walking pace in an empty parking lot, seen from its "
                            "windshield camera. Objective: %s What should the car do next?" % objective,
            "criteria": list(MANEUVERS),
        }
    }


def direction_question(objective: str) -> Dict[str, Dict]:
    """The same question with one option per direction and no stop: which way the lane goes, without the maneuver
    list's two-left, two-right, one-straight imbalance."""
    q = question(objective)["action"]
    return {"action": dict(q, instructions=q["instructions"].replace("What should the car do next?",
                                                                     "Which way should the car steer next?"),
                           criteria=["steer left", "go straight", "steer right"])}


def state(image, speed_mps=None, steering_percent=None, command_state=None) -> Dict:
    """The predict state: the camera frame(s) plus what observe reports about the car."""
    return {"images": image if isinstance(image, list) else [image], "note": {
        "speed_mps": speed_mps,
        "steering_percent_positive_left": steering_percent,
        "command_state": command_state,
    }}


def decide(answer: Dict, min_prob: float):
    """The model's answer -> ("stop", None, reason) or ("motion", (direction, steering_percent), reason)."""
    choice = answer["choice"]
    p = answer["probabilities"][choice]
    reason = "laya: %s (p=%.2f)" % (choice.split(":")[0], p)
    if MANEUVERS[choice] is None:
        return "stop", None, reason
    if p < min_prob:
        return "stop", None, "laya: unsure, best %s (p=%.2f < %.2f)" % (choice, p, min_prob)
    return "motion", MANEUVERS[choice], reason


def direction_probs(answer: Dict) -> Dict[str, float]:
    """P(stop) and P(left / straight / right), summing the maneuvers that steer each way."""
    out = {"stop": 0.0, **{d: 0.0 for d in DIRECTIONS}}
    for maneuver, p in answer["probabilities"].items():
        target = MANEUVERS[maneuver]
        out["stop" if target is None else target[0]] += p
    return out
