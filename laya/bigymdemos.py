"""BiGym human demonstrations as ``laya.bigymgames`` primitive labels.

A demo is a 20 Hz stream of absolute joint targets (4 floating-base DOFs, 10 arm joints, 2 grippers). Laya picks
one of 37 primitives per 0.1 s decision. ``follow`` bridges them closed-loop: the demo is replayed in BiGym's own
settings to get ``waypoints`` (where the pelvis, both wrists, their rolls, which way each gripper points and whether it is closed, every
``every`` demo steps), then a follower in the eval env (``make_env``: delta joints, 4 floating DOFs, 50 Hz) picks,
each decision, the primitive whose simulated effect (tried and undone, ``lookahead``) brings the robot closest to
the current waypoint, advancing
through the waypoints as it reaches them. The chosen primitives are the labels; whether the follower completes the
task says whether those labels are good enough to learn from.

The demos also crouch (the pelvis drops up to ~0.11 m); the eval env has the same four floating DOFs, so the
follower crouches with ``BASE_DOWN`` / ``BASE_UP``.

Needs bigym (and its ``demonstrations`` package); demos download to ``~/.bigym`` on first use (about 120 MB).
"""
from typing import Dict, List, Optional

import numpy as np

from . import bigymgames as bg

EVERY = 2  # demo steps (20 Hz) per waypoint: 0.1 s, one decision
# waypoint cost weights: metres per unit of each state error
W_BASE, W_YAW, W_WRIST, W_GRIP = 0.5, 0.2, 0.05, 0.2
W_POINT = 0.1  # per radian between the gripper's pointing direction and the demo's (30 degrees ~ 5 cm)
REACHED = 0.06  # a waypoint counts as reached below this cost
PATIENCE = 4  # decisions without a new best cost on a waypoint (by IMPROVE) before aiming at the next one
IMPROVE = 0.002
FIELDS = ("px", "py", "pz", "yaw", "lx", "ly", "lz", "rx", "ry", "rz", "wl", "wr", "gl", "gr",
          "plx", "ply", "plz", "prx", "pry", "prz")  # the last six: each gripper's pointing direction


def _yaw(xmat) -> float:
    fwd = np.asarray(xmat).reshape(3, 3)[:, 0]
    return float(np.arctan2(fwd[1], fwd[0]))


def demo_waypoints(task: str, amount: int = 20, seed: int = 0, every: int = EVERY) -> List[Dict]:
    """Replay ``amount`` demos of ``task`` (BiGym's settings: absolute joints, 4 floating DOFs, 20 Hz) and return,
    per demo, its seed, whether it succeeded and its waypoints ``[T, len(FIELDS)]`` up to the success step."""
    import importlib

    from bigym.action_modes import JointPositionActionMode, PelvisDof
    from bigym.bigym_env import CONTROL_FREQUENCY_MIN
    from bigym.const import HandSide
    from demonstrations.demo_player import DemoPlayer
    from demonstrations.demo_store import DemoStore
    from demonstrations.utils import Metadata

    cls = getattr(importlib.import_module("bigym.envs." + bg.TASKS[task]["module"]), task)
    dofs = [PelvisDof.X, PelvisDof.Y, PelvisDof.Z, PelvisDof.RZ]
    env = cls(action_mode=JointPositionActionMode(absolute=True, floating_base=True, floating_dofs=dofs),
              control_frequency=CONTROL_FREQUENCY_MIN)
    np.random.seed(seed)  # DemoStore shuffles with the global generator
    demos = DemoStore().get_demos(Metadata.from_env(env, is_lightweight=True), amount=amount,
                                  frequency=CONTROL_FREQUENCY_MIN)
    robot, data = env.robot, env.mojo.data
    pel = env.mojo.physics.bind(robot.pelvis.mjcf).element_id
    sites = [env.mojo.physics.bind(robot._wrist_sites[side].mjcf).element_id for side in (HandSide.LEFT, HandSide.RIGHT)]
    nb = robot.floating_base.dof_amount
    wl, wr = nb + 4, nb + 9  # the wrists are each arm's fifth actuator
    out = []
    for demo in demos:
        steps = DemoPlayer._get_timesteps_for_replay(demo, env, CONTROL_FREQUENCY_MIN)
        env.reset(seed=demo.seed)
        rows, success = [], None
        for i, st in enumerate(steps):
            env.step(st.executed_action, fast=True)
            q = robot.qpos_actuated
            a = st.executed_action
            rows.append([data.xpos[pel][0], data.xpos[pel][1], data.xpos[pel][2], _yaw(data.xmat[pel]),
                         *robot.get_hand_pos(HandSide.LEFT), *robot.get_hand_pos(HandSide.RIGHT),
                         q[wl], q[wr], float(a[-2] > 0.5), float(a[-1] > 0.5),
                         *[v for i in sites for v in np.asarray(data.site_xmat[i]).reshape(3, 3)[:, 0]]])
            if env.success:
                success = i
                break
        w = np.array(rows[every - 1::every] + ([rows[-1]] if len(rows) % every else []), np.float64)
        out.append({"seed": int(demo.seed), "uuid": str(demo.uuid), "success_step": success, "steps": len(rows),
                    "waypoints": w})
    env.close()
    return out


def _state(game: bg.BiGymGame) -> np.ndarray:
    d, m = game._data, game._model
    wrist = [float(d.qpos[m.jnt_qposadr[m.dof_jntid[game._arm[h][1][-1]]]]) for h in ("left", "right")]
    p = d.xpos[game._pelvis]
    return np.array([p[0], p[1], p[2], _yaw(d.xmat[game._pelvis]), *game.hand_pos("left"), *game.hand_pos("right"),
                     *wrist, *game._grip, *game.pointing("left"), *game.pointing("right")], np.float64)


def cost(s: np.ndarray, w: np.ndarray) -> float:
    """Distance from state ``s`` to waypoint ``w`` (both ``FIELDS`` order), in metres-equivalent."""
    dyaw = (s[3] - w[3] + np.pi) % (2 * np.pi) - np.pi
    return float(np.linalg.norm(s[4:7] - w[4:7]) + np.linalg.norm(s[7:10] - w[7:10])
                 + W_BASE * np.linalg.norm(s[0:3] - w[0:3]) + W_YAW * abs(dyaw)
                 + W_WRIST * np.abs(s[10:12] - w[10:12]).sum() + W_GRIP * np.abs(s[12:14] - w[12:14]).sum()
                 + W_POINT * (_angle(s[14:17], w[14:17]) + _angle(s[17:20], w[17:20])))


def _angle(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.arccos(np.clip(a @ b / max(1e-9, np.linalg.norm(a) * np.linalg.norm(b)), -1.0, 1.0)))


def lookahead(game: bg.BiGymGame, target: np.ndarray) -> str:
    """The primitive whose simulated outcome (one decision, then undone) is closest to ``target``; ``STAY`` wins
    ties, so the follower only moves when a move helps."""
    snap = game.snapshot()
    scores = {}
    for p in bg.PRIMITIVES:
        game.step(p)
        scores[p] = cost(_state(game), target) - (1e-4 if p == "STAY" else 0.0)
        game.restore(snap)
    return min(scores, key=scores.get)


def follow(task: str, demo: Dict, max_decisions: Optional[int] = None, env=None) -> Dict:
    """Follow one demo's waypoints with primitives in the eval env. Returns the labels (one primitive per
    decision, with the waypoint index it was aiming at), whether the task succeeded, and how far along the demo
    the follower got."""
    w = demo["waypoints"]
    game = bg.BiGymGame(task, demo["seed"], env=env or bg.make_env(task, cameras=False))
    cap = max_decisions or 20 * len(w) + 200  # one part at a time is many times slower than the demo
    k, labels, skipped = 0, [], 0
    best_c, since = np.inf, 0  # best cost reached on waypoint k, and decisions since it last improved

    def advance():
        nonlocal k, best_c, since
        k, best_c, since = k + 1, np.inf, 0

    while not game.done and game.decisions < cap:
        s = _state(game)
        while k < len(w) - 1 and cost(s, w[k]) < REACHED:
            advance()
        c = cost(s, w[k])
        if c < best_c - IMPROVE:
            best_c, since = c, 0
        elif since >= PATIENCE and k < len(w) - 1:  # oscillating or stalled on this waypoint: move on
            advance()
            skipped += 1
        best = lookahead(game, w[k])
        while best == "STAY" and k < len(w) - 1:  # no move gets closer to this waypoint: aim at the next one
            advance()
            skipped += 1
            best = lookahead(game, w[k])
        labels.append({"decision": game.decisions, "primitive": best, "waypoint": k})
        game.step(best)
        since += 1
    out = {"seed": demo["seed"], "uuid": demo["uuid"], "success": bool(game.success), "decisions": game.decisions,
           "waypoints": len(w), "reached": k, "skipped": skipped, "final_cost": cost(_state(game), w[-1]),
           "demo_decisions": len(w), "labels": labels}
    if env is None:
        game.close()
    return out


def follow_report(task: str, amount: int = 20, seed: int = 0) -> Dict:
    """``follow`` every one of ``amount`` demos: follower success rate against the demos' own, and the label mix."""
    from collections import Counter

    demos = demo_waypoints(task, amount, seed)
    env = bg.make_env(task, cameras=False)
    runs = [follow(task, d, env=env) for d in demos]
    env.close()
    counts = Counter(lab["primitive"] for r in runs for lab in r["labels"])
    return {"task": task, "demos": len(demos),
            "demo_success": float(np.mean([d["success_step"] is not None for d in demos])),
            "follower_success": float(np.mean([r["success"] for r in runs])),
            "mean_decisions": float(np.mean([r["decisions"] for r in runs])),
            "mean_demo_decisions": float(np.mean([r["demo_decisions"] for r in runs])),
            "labels": dict(counts.most_common()), "runs": [{k: v for k, v in r.items() if k != "labels"} for r in runs]}
