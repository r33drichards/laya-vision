"""BiGym human demonstrations as ``laya.bigymgames`` primitive labels.

A demo is a 20 Hz stream of absolute joint targets (4 floating-base DOFs, 10 arm joints, 2 grippers). Laya picks
one of 37 primitives per 0.1 s decision. ``follow`` bridges them closed-loop. The demo is replayed in BiGym's own
settings to get ``waypoints``: every ``every`` demo steps, where the pelvis and both wrists are, the wrist rolls,
which way each gripper points and whether it is closed. A follower in the eval env (``make_env``: delta joints,
4 floating DOFs, 50 Hz) then picks, each decision, the primitive whose simulated effect (tried and undone,
``lookahead``) brings the robot closest to the current waypoint, advancing through the waypoints as it reaches
them. The chosen primitives are the labels; whether the follower completes the task says whether those labels are
good enough to learn from.

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
END_PATIENCE = 100  # decisions on the last waypoint without a new best cost before giving up (no labels from idling)
# Grasps. A demo closes a gripper where its fingers straddle the handle; closing a few cm off closes on air or on the
# door face (the follower reaches the gripper-closed waypoint by cost, and W_GRIP alone outweighs REACHED). So a
# gripper may only close within GRASP_TOL (m) and GRASP_ANGLE (rad) of the demo's pose at its closing waypoint, and
# the follower does not aim past that waypoint until the gripper is closed. GRASP_PATIENCE decisions without
# getting there lift the gate, so a grasp the primitives cannot line up still happens.
GRASP_TOL = 0.02
GRASP_ANGLE = 0.26
GRASP_PATIENCE = 40
# Within FINE_RADIUS (m) of a grasp the base does not step or turn: a 5 cm base step shifts both hands at once and
# lands a finger on the handle bar (a 6 mm bar 2.7 cm off the door), which then blocks the hand short of the grasp
# (it may still crouch: the demos crouch onto the drawer handle). And a hand whose gripper is closed (the demo's
# too) neither tilts nor rolls: those turn the arm about the grasp and lever the handle out of the fingers (a 4 cm
# hand drop from one tilt, measured).
FINE_RADIUS = 0.05
_BASE_MOVES = ("BASE_FORWARD", "BASE_BACK", "BASE_LEFT", "BASE_RIGHT", "BASE_TURN_LEFT", "BASE_TURN_RIGHT")
# The follower also sees the part (door / drawer open fraction, 0..1) and, choosing a move, pays W_PART per unit it
# is off the demo's at the waypoint: a move that knocks a held door shut or lets it slip back costs, one that
# pulls it along the hinge's arc pays. Following the demo's wrist path alone, a straight 3 cm-step pull loses the
# handle at ~0.35 open (the arc turns away from the line); with it the doors reach the demo's ~0.8 before the
# release. (Only the choice: REACHED and PATIENCE still use ``cost``.)
W_PART = 1.0
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
    sites = [env.mojo.physics.bind(robot._wrist_sites[side].mjcf).element_id
             for side in (HandSide.LEFT, HandSide.RIGHT)]
    nb = robot.floating_base.dof_amount
    wl, wr = nb + 4, nb + 9  # the wrists are each arm's fifth actuator
    out = []
    for demo in demos:
        steps = DemoPlayer._get_timesteps_for_replay(demo, env, CONTROL_FREQUENCY_MIN)
        env.reset(seed=demo.seed)
        rows, parts, success = [], [], None
        for i, st in enumerate(steps):
            env.step(st.executed_action, fast=True)
            q = robot.qpos_actuated
            a = st.executed_action
            rows.append([data.xpos[pel][0], data.xpos[pel][1], data.xpos[pel][2], _yaw(data.xmat[pel]),
                         *robot.get_hand_pos(HandSide.LEFT), *robot.get_hand_pos(HandSide.RIGHT),
                         q[wl], q[wr], float(a[-2] > 0.5), float(a[-1] > 0.5),
                         *[v for i in sites for v in np.asarray(data.site_xmat[i]).reshape(3, 3)[:, 0]]])
            parts.append(_part(env, task))
            if env.success:
                success = i
                break
        pick = lambda r: np.array(r[every - 1::every] + ([r[-1]] if len(r) % every else []), np.float64)
        out.append({"seed": int(demo.seed), "uuid": str(demo.uuid), "success_step": success, "steps": len(rows),
                    "waypoints": pick(rows), "part": pick(parts)})
    env.close()
    return out


def _part(env, task: str) -> List[float]:
    """How far open the task's part is: the wall cabinet's two doors, or the top drawer (normalised, 0 closed)."""
    if "part" not in bg.TASKS[task]:
        return []
    if bg.TASKS[task]["part"] == "drawer":
        return [float(env.cabinet_drawers.get_state()[-1])]
    return [float(v) for v in env.cabinet_wall.get_state()]


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


def lookahead(game: bg.BiGymGame, target: np.ndarray, banned=(), part: Optional[np.ndarray] = None) -> str:
    """The primitive whose simulated outcome (one decision, then undone) is closest to ``target`` (plus ``W_PART``
    per unit the part ends off ``part``, when given); ``STAY`` wins ties, so the follower only moves when a move
    helps. ``banned`` primitives are not tried."""
    snap = game.snapshot()
    scores = {}
    for p in bg.PRIMITIVES:
        if p in banned:
            continue
        game.step(p)
        scores[p] = cost(_state(game), target) - (1e-4 if p == "STAY" else 0.0)
        if part is not None and W_PART:
            scores[p] += W_PART * float(np.abs(np.array(_part(game.env, game.task)) - part).sum())
        game.restore(snap)
    return min(scores, key=scores.get)


_HANDS = (("left", 12, 4, 14), ("right", 13, 7, 17))  # hand, its gripper field, wrist xyz, pointing xyz


def grasp_waypoints(w: np.ndarray) -> List[List[int]]:
    """Per hand (left, right): the waypoints where the demo closes that gripper."""
    return [[k for k in range(len(w)) if w[k][g] > 0.5 and (k == 0 or w[k - 1][g] < 0.5)] for _, g, _, _ in _HANDS]


def _grasp_gate(game: bg.BiGymGame, w: np.ndarray, k: int, grasps, waited: int, done=frozenset()):
    """(banned primitives, the last waypoint the follower may aim at, whether a grasp is gated). Gripper commands
    already held are banned (no-ops). An open gripper whose demo closes it at waypoint ``g <= k`` holds the
    follower at ``g`` and may close only once lined up (``GRASP_TOL``, ``GRASP_ANGLE``) or after
    ``GRASP_PATIENCE`` decisions there (``waited``). ``done`` holds the (hand index, waypoint) grasps already
    closed on, which do not hold the follower again once let go."""
    banned, limit, gated = set(), len(w) - 1, False
    for i, (hand, gi, pi, di) in enumerate(_HANDS):
        H = hand.upper()
        # a gripper command that is already held changes nothing but lasts GRIP_HOLD: a long STAY, not a label
        banned.add("%s_GRIPPER_%s" % (H, "CLOSE" if game._grip[i] > 0.5 else "OPEN"))
        if game._grip[i] > 0.5:
            if w[k][gi] > 0.5:  # holding: keep the grasp's orientation
                banned.update("%s_TILT_%s" % (H, d) for d in ("UP", "DOWN", "LEFT", "RIGHT"))
                banned.update(("%s_WRIST_CW" % H, "%s_WRIST_CCW" % H))
            continue
        due = [g for g in grasps[i] if g <= k]
        nxt = [g for g in grasps[i] if g > k]
        pending = (bool(due) and (i, due[-1]) not in done  # not if closed on already, nor if the demo let go
                   and all(w[j][gi] > 0.5 for j in range(due[-1], k + 1)))
        if not pending and not nxt:
            continue
        g = due[-1] if pending else nxt[0]
        limit = min(limit, g)
        dist = np.linalg.norm(game.hand_pos(hand) - w[g][pi:pi + 3])
        if dist < FINE_RADIUS:
            banned.update(_BASE_MOVES)
        if not pending:
            continue
        lined_up = dist < GRASP_TOL and _angle(game.pointing(hand), w[g][di:di + 3]) < GRASP_ANGLE
        if not lined_up and waited < GRASP_PATIENCE:
            banned.add("%s_GRIPPER_CLOSE" % H)
            gated = True
    return banned, limit, gated


def follow(task: str, demo: Dict, max_decisions: Optional[int] = None, env=None) -> Dict:
    """Follow one demo's waypoints with primitives in the eval env. Returns the labels (one primitive per
    decision, with the waypoint index it was aiming at), whether the task succeeded, and how far along the demo
    the follower got."""
    w, parts = demo["waypoints"], demo.get("part")
    game = bg.BiGymGame(task, demo["seed"], env=env or bg.make_env(task, cameras=False))
    cap = max_decisions or 20 * len(w) + 200  # one part at a time is many times slower than the demo
    k, labels, skipped = 0, [], 0
    best_c, since = np.inf, 0  # best cost reached on waypoint k, and decisions since it last improved
    grasps, waited, done = grasp_waypoints(w), 0, set()  # decisions held at a grasp waypoint; grasps made

    def advance(limit):
        nonlocal k, best_c, since
        if k < limit:
            k, best_c, since = k + 1, np.inf, 0
            return True
        return False

    while not game.done and game.decisions < cap:
        s = _state(game)
        banned, limit, gated = _grasp_gate(game, w, k, grasps, waited, done)
        while cost(s, w[k]) < REACHED and advance(limit):
            banned, limit, gated = _grasp_gate(game, w, k, grasps, waited, done)
        c = cost(s, w[k])
        if c < best_c - IMPROVE:
            best_c, since = c, 0
        elif k == len(w) - 1 and since >= END_PATIENCE:  # at the end and getting nowhere: stop
            break
        elif since >= PATIENCE and advance(limit):  # oscillating or stalled on this waypoint: move on
            skipped += 1
            banned, limit, gated = _grasp_gate(game, w, k, grasps, waited, done)
        waited = waited + 1 if k == limit and gated else 0
        best = lookahead(game, w[k], banned, None if parts is None else parts[k])
        while best == "STAY":  # no move gets closer to this waypoint: aim at the next one
            if advance(limit):
                skipped += 1
            elif gated and waited < GRASP_PATIENCE:  # held at a grasp and stuck short of it: lift the gate
                waited = GRASP_PATIENCE
            else:
                break
            banned, limit, gated = _grasp_gate(game, w, k, grasps, waited, done)
            best = lookahead(game, w[k], banned, None if parts is None else parts[k])
        for i, (hand, _, _, _) in enumerate(_HANDS):
            if best == "%s_GRIPPER_CLOSE" % hand.upper():
                done.update((i, g) for g in grasps[i] if g <= k)
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
