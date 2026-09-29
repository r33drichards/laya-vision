#!/usr/bin/env python3
"""The keep / discard rule for autoresearch: a Pareto frontier over four objectives.

| objective   | better | margin        | what                                                                  |
|-------------|--------|---------------|-----------------------------------------------------------------------|
| `quality`   | higher | 0.005 (abs)   | macro dataset accuracy minus ECE on single-answer questions            |
| `games`     | higher | 0.04 (abs)    | mean normalized game score, 0 = random play, 1 = the scripted expert   |
| `params_m`  | lower  | 1% (rel)      | parameters of the saved model, millions                                |
| `latency_x` | lower  | 5% (rel)      | median predict time / the base checkpoint's, timed in the same L4 run  |

Upstream autoresearch keeps an experiment when its single metric (val_bpb) improves. Here an experiment is kept when
it extends the frontier: no already-kept result is at least as good on every objective within the noise margins.
Kept results that the new one strictly beats on every objective drop off the frontier. Progress is tracked as the
frontier's hypervolume (the volume of normalized objective space it dominates against a fixed reference point),
which only grows when the frontier moves outward.

    python autoresearch/pareto.py add runs/<tag>/<commit>.json --tsv runs/<tag>/results.tsv --desc "..."
    python autoresearch/pareto.py show --tsv runs/<tag>/results.tsv

Profiles. The rule above is the ``default`` profile. The ``bigym`` profile (``harness.py --profile bigym``) has one
main objective and two guard-rails instead of a frontier (``decide_bigym``): a result is kept iff

* ``bigym`` > the best kept ``bigym`` + ``BIGYM_MARGIN`` (the BiGym control score, ``bigym_eval.py``), and
* ``quality`` >= the baseline's - ``QUALITY_GUARD``, and ``games`` >= the baseline's - ``GAMES_GUARD``, and
* ``params_m`` <= the baseline's * (1 + ``PARAMS_GUARD``) (size and latency are frozen at the baseline model),

where the baseline is the tag's first kept result (the first result is always kept). Its TSV has its own columns
(``BIGYM_COLUMNS``); ``show`` and ``add`` pick the profile from the TSV header.

Standard library only. This file is part of the fixed harness: experiments do not edit it.
"""
import argparse
import csv
import json
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

# (name, direction, margin, margin is relative). A result must beat every kept result by more than the margin in at
# least one objective to count as new. quality and latency_x margins were measured on the unchanged baseline (4 runs
# on the pooled harness: quality 0.6691-0.6739, latency_x 0.998 / 1.007 on L4 hosts timing 52 / 77 ms raw). games is
# deterministic for a given checkpoint (fixed seeds, greedy play) and the baseline repeated exactly (-0.0344 twice,
# every game identical), but that baseline plays degenerately (0 on the mazes, Acrobot, MountainCar), so retraining
# noise did not move it; 0.03 is one game moving 0.3 in the 10-game mean. Retraining a stronger recipe (15 text
# layers + game data) moved games by 0.013 (0.148 / 0.135: DoomBasic 0.78 / 0.66, every other game within 0.02).
# On a real player (sep23-v2, 20 layers, games 45%) a repeat moved games by 0.003 (0.2375 / 0.2409) while single
# games moved up to 0.12 (Acrobot 0.24 / 0.16, MountainCar 0.31 / 0.43), which the 10-game mean averages out.
# A third run of that recipe (sep24-next) gave 0.2007 (MountainCar 0.20, LunarLander 0.01): spread 0.040 over three
# runs, so the games margin is 0.04. Quality repeats within 0.004 for most recipes, but a recipe that trains the
# vision tower gave 0.6634 / 0.6759 across two runs: repeat such a recipe before trusting a quality keep from it.
# latency_x was 3% from the full model timed twice; the 15-layer architecture timed three times gave 0.768 / 0.749 /
# 0.733 (4.6% spread, host noise alone: 9a06403 changed only the LR and was kept on latency), so the margin is 5%.
OBJECTIVES = (
    ("quality", "max", 0.005, False),
    ("games", "max", 0.04, False),
    ("params_m", "min", 0.01, True),
    ("latency_x", "min", 0.05, True),
)
NAMES = tuple(o[0] for o in OBJECTIVES)

# Hypervolume reference point: quality from 0, games from -0.5 (the clip floor of a game's normalized score), params
# and latency up to 1.5x the baseline's.
REF_SCALE = 1.5
GAMES_FLOOR = -0.5

# -- the bigym profile ------------------------------------------------------------------------------------------
# PROVISIONAL margins, to be replaced after the baseline noise runs (repeat the tag's baseline twice and set each to
# about the spread seen): BIGYM_MARGIN is the gain in the mean normalized BiGym score that counts as real;
# QUALITY_GUARD and GAMES_GUARD are how far quality and games may fall below the tag's baseline (the default
# profile's measured noise is 0.005 quality and 0.04 games, so these guards allow a little real forgetting).
BIGYM_MARGIN = 0.03
QUALITY_GUARD = 0.01
GAMES_GUARD = 0.05
PARAMS_GUARD = 0.01   # relative: the model's size is frozen (an accidental architecture change is refused)
PROFILES = ("default", "bigym")
BIGYM_COLUMNS = ["commit", "bigym", "bigym_success", "quality", "macro_acc", "ece_hard", "games", "params_m",
                 "latency_x", "status", "profile", "experiment", "task_success", "description"]
BIGYM_FLOATS = ("bigym", "bigym_success")

COLUMNS = ["commit", "quality", "macro_acc", "ece_hard", "games", "params_m", "latency_x", "status", "description"]
FLOATS = ("quality", "macro_acc", "ece_hard", "games", "params_m", "latency_x")


def _better_eq(a: float, b: float, direction: str, margin: float = 0.0, relative: bool = False) -> bool:
    """``a`` is at least as good as ``b``, giving ``b`` the benefit of ``margin``."""
    slack = abs(b) * margin if relative else margin
    return a >= b - slack if direction == "max" else a <= b + slack


def dominates(a: Dict, b: Dict, eps: bool = True) -> bool:
    """``a`` is at least as good as ``b`` on every objective (with the noise margins in ``b``'s favour when
    ``eps``): ``b`` then adds nothing ``a`` does not already offer. Without ``eps``, ``a`` must also be strictly
    better somewhere (plain Pareto dominance)."""
    if eps:
        return all(_better_eq(a[n], b[n], d, m, rel) for n, d, m, rel in OBJECTIVES)
    return all(_better_eq(a[n], b[n], d) for n, d, _, _ in OBJECTIVES) and any(a[n] != b[n] for n in NAMES)


def frontier(rows: Sequence[Dict]) -> List[Dict]:
    """The kept rows no other kept row strictly dominates, in file order."""
    kept = [r for r in rows if r["status"] == "keep"]
    return [r for r in kept if not any(dominates(o, r, eps=False) for o in kept if o is not r)]


def decide(front: Sequence[Dict], cand: Dict) -> Tuple[str, List[Dict]]:
    """``("keep", beaten)`` when no frontier point dominates ``cand`` within the margins (``beaten`` lists the
    frontier points ``cand`` strictly dominates), else ``("discard", [])``. The first result is always kept."""
    if any(dominates(p, cand, eps=True) for p in front):
        return "discard", []
    return "keep", [p for p in front if dominates(cand, p, eps=False)]


FINISHED = ("keep", "discard", "crash")


def prunable(rows: Sequence[Dict]) -> List[str]:
    """Commits whose saved checkpoints may be deleted: recorded in the TSV with a finished status and not on the
    frontier. A commit that is not in the TSV at all (a run still going, e.g. a concurrent experiment whose fresh
    checkpoint is saved but not yet decided) is never among them, and neither is one that has any frontier row."""
    front = {r["commit"] for r in frontier(rows)}
    return sorted({r["commit"] for r in rows if r["status"] in FINISHED and r["commit"] not in front})


def _normalized(p: Dict, base: Dict) -> Tuple[float, ...]:
    """A point with every objective turned into one to minimize against ``_ref()``:
    (-quality, -(games - floor), params / base params, latency / base latency)."""
    return (-p["quality"], -(p["games"] - GAMES_FLOOR), p["params_m"] / base["params_m"],
            p["latency_x"] / base["latency_x"])


def _ref() -> Tuple[float, ...]:
    return (0.0, 0.0, REF_SCALE, REF_SCALE)


def _hv(points: List[Tuple[float, ...]], ref: Tuple[float, ...]) -> float:
    """Exact hypervolume of minimized points against ``ref``, by slicing along the first coordinate."""
    pts = [p for p in points if all(x < r for x, r in zip(p, ref))]
    if not pts:
        return 0.0
    if len(ref) == 1:
        return ref[0] - min(p[0] for p in pts)
    xs = sorted({p[0] for p in pts})
    vol = 0.0
    for i, x in enumerate(xs):
        nxt = xs[i + 1] if i + 1 < len(xs) else ref[0]
        vol += (nxt - x) * _hv([p[1:] for p in pts if p[0] <= x], ref[1:])
    return vol


def hypervolume(points: Sequence[Dict], base: Dict) -> float:
    """Volume of normalized objective space the points dominate (quality from 0, games from ``GAMES_FLOOR``,
    params and latency up to ``REF_SCALE`` times the baseline's)."""
    return _hv([_normalized(p, base) for p in points], _ref())


def read_tsv(path: str) -> List[Dict]:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    for r in rows:
        for k in FLOATS + BIGYM_FLOATS:
            if k in r:
                r[k] = float(r[k])
    return rows


def tsv_profile(path: str) -> Optional[str]:
    """The profile a TSV was written for (from its header), or None when it does not exist yet."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        head = f.readline().rstrip("\n").split("\t")
    return "bigym" if "bigym" in head else "default"


def columns(profile: str = "default") -> List[str]:
    if profile not in PROFILES:
        raise ValueError("unknown profile %r (%s)" % (profile, ", ".join(PROFILES)))
    return BIGYM_COLUMNS if profile == "bigym" else COLUMNS


def append_tsv(path: str, row: Dict, profile: Optional[str] = None) -> None:
    """Append ``row``; a new file gets ``profile``'s header (default: the row's ``profile``, else ``default``),
    an existing one keeps its own, and a row for another profile's TSV is refused."""
    have = tsv_profile(path)
    want = profile or row.get("profile") or have or "default"
    if have is not None and have != want:
        raise ValueError("%s is a %s-profile TSV; refusing a %s row (use a new tag)" % (path, have, want))
    cols = columns(want)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a") as f:
        if have is None:
            f.write("\t".join(cols) + "\n")
        f.write("\t".join(_cell(row.get(c, "")) for c in cols) + "\n")


def _cell(v) -> str:
    if isinstance(v, float):
        return "%.4f" % v
    return str(v).replace("\t", " ").replace("\n", " ")


def row_from_result(res: Dict, commit: str, description: str) -> Dict:
    s = res["summary"]
    row = {k: float(s[k]) for k in FLOATS}
    row.update(commit=commit, status="", description=description)
    if res.get("profile", "default") == "bigym":
        b = res["bigym"]
        row.update(bigym=float(s["bigym"]), bigym_success=float(b["success_mean"]), profile="bigym",
                   experiment=res.get("experiment", ""),
                   task_success=" ".join("%.2f" % b["success"].get(t, float("nan")) for t in b["tasks"]))
    return row


def crash_row(commit: str, description: str, profile: str = "default", experiment: str = "") -> Dict:
    row = {k: 0.0 for k in FLOATS + (BIGYM_FLOATS if profile == "bigym" else ())}
    row.update(commit=commit, status="crash", description=description)
    if profile == "bigym":
        row.update(profile="bigym", experiment=experiment, task_success="-")
    return row


# -- bigym profile: one objective, two guard-rails ---------------------------------------------------------------

def bigym_baseline(rows: Sequence[Dict]) -> Optional[Dict]:
    """The tag's baseline: its first kept result."""
    return next((r for r in rows if r["status"] == "keep"), None)


def best_bigym(rows: Sequence[Dict]) -> Optional[Dict]:
    kept = [r for r in rows if r["status"] == "keep"]
    return max(kept, key=lambda r: r["bigym"]) if kept else None


def decide_bigym(rows: Sequence[Dict], cand: Dict) -> Tuple[str, str]:
    """``(status, reason)`` for ``cand`` under the bigym profile (see the module docstring): keep iff it beats the
    best kept ``bigym`` by more than ``BIGYM_MARGIN`` and stays within the quality, games and size guard-rails of
    the baseline (the first kept row). The first result is always kept."""
    base, best = bigym_baseline(rows), best_bigym(rows)
    if base is None:
        return "keep", "baseline (first result)"
    fails = []
    if not cand["bigym"] > best["bigym"] + BIGYM_MARGIN:
        fails.append("bigym %.4f not above best %.4f + %.3f" % (cand["bigym"], best["bigym"], BIGYM_MARGIN))
    if cand["quality"] < base["quality"] - QUALITY_GUARD:
        fails.append("quality %.4f below baseline %.4f - %.3f" % (cand["quality"], base["quality"], QUALITY_GUARD))
    if cand["games"] < base["games"] - GAMES_GUARD:
        fails.append("games %.4f below baseline %.4f - %.3f" % (cand["games"], base["games"], GAMES_GUARD))
    if cand["params_m"] > base["params_m"] * (1 + PARAMS_GUARD):
        fails.append("params %.1fM above the frozen %.1fM" % (cand["params_m"], base["params_m"]))
    if fails:
        return "discard", "; ".join(fails)
    return "keep", "bigym %.4f > %.4f + %.3f within the guard-rails" % (cand["bigym"], best["bigym"], BIGYM_MARGIN)


def prunable_bigym(rows: Sequence[Dict]) -> List[str]:
    """Finished commits with no kept row: their checkpoints may go (every kept checkpoint stays)."""
    kept = {r["commit"] for r in rows if r["status"] == "keep"}
    return sorted({r["commit"] for r in rows if r["status"] in FINISHED and r["commit"] not in kept})


def show_bigym(rows: Sequence[Dict]) -> str:
    lines = ["%d experiments: %d keep, %d discard, %d crash (bigym profile)" % (
        len(rows), sum(r["status"] == "keep" for r in rows), sum(r["status"] == "discard" for r in rows),
        sum(r["status"] == "crash" for r in rows))]
    base, best = bigym_baseline(rows), best_bigym(rows)
    if base:
        lines.append("guard-rails: quality >= %.4f, games >= %.4f (baseline %s); next keep needs bigym > %.4f" % (
            base["quality"] - QUALITY_GUARD, base["games"] - GAMES_GUARD, base["commit"],
            best["bigym"] + BIGYM_MARGIN))
        lines.append("kept (task success: %s):" % "ReachTarget ReachTargetSingle DrawerTopOpen DrawerTopClose "
                                                  "WallCupboardOpen WallCupboardClose")
        for r in rows:
            if r["status"] == "keep":
                lines.append("  %s  bigym %+.4f  success %.3f [%s]  quality %.4f  games %+.3f  %s" % (
                    r["commit"], r["bigym"], r["bigym_success"], r.get("task_success", ""), r["quality"],
                    r["games"], r["description"]))
    return "\n".join(lines)


def show(rows: Sequence[Dict]) -> str:
    front = frontier(rows)
    base = next((r for r in rows if r["status"] == "keep"), None)
    lines = ["%d experiments: %d keep, %d discard, %d crash" % (
        len(rows), sum(r["status"] == "keep" for r in rows), sum(r["status"] == "discard" for r in rows),
        sum(r["status"] == "crash" for r in rows))]
    if base:
        lines.append("hypervolume %.4f (baseline alone %.4f)" % (hypervolume(front, base), hypervolume([base], base)))
        lines.append("frontier (%d):" % len(front))
        for r in sorted(front, key=lambda r: r["params_m"]):
            lines.append("  %s  quality %.4f  games %+.3f  %7.1fM params  %5.3fx latency  %s" % (
                r["commit"], r["quality"], r["games"], r["params_m"], r["latency_x"], r["description"]))
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add", help="decide keep/discard for a result file and append it to the TSV")
    a.add_argument("result", help="result JSON from harness.py, or 'crash'")
    a.add_argument("--tsv", required=True)
    a.add_argument("--commit", default="")
    a.add_argument("--desc", default="")
    a.add_argument("--profile", choices=PROFILES, default=None, help="default: the TSV's, else the result's")
    s = sub.add_parser("show", help="print the frontier and its hypervolume (bigym profile: the kept results)")
    s.add_argument("--tsv", required=True)
    args = ap.parse_args(argv)
    rows = read_tsv(args.tsv)
    if args.cmd == "show":
        print(show_bigym(rows) if tsv_profile(args.tsv) == "bigym" else show(rows))
        return 0
    profile = args.profile or tsv_profile(args.tsv)
    if args.result == "crash":
        append_tsv(args.tsv, crash_row(args.commit or "-", args.desc, profile or "default"), profile)
        print("status: crash")
        return 0
    with open(args.result) as f:
        res = json.load(f)
    row = row_from_result(res, args.commit or res.get("commit", "-"), args.desc or res.get("description", ""))
    if (profile or res.get("profile", "default")) == "bigym":
        status, why = decide_bigym(rows, row)
        row["status"] = status
        append_tsv(args.tsv, row, "bigym")
        print("status: %s (%s)" % (status, why))
        return 0
    front = frontier(rows)
    status, beaten = decide(front, row)
    row["status"] = status
    base = next((r for r in rows if r["status"] == "keep"), row)
    hv_before = hypervolume(front, base) if front else 0.0
    append_tsv(args.tsv, row)
    hv_after = hypervolume(frontier(read_tsv(args.tsv)), base)
    print("status: %s" % status)
    print("hypervolume: %.4f -> %.4f" % (hv_before, hv_after))
    if beaten:
        print("now dominates: %s" % ", ".join(p["commit"] for p in beaten))
    return 0


if __name__ == "__main__":
    sys.exit(main())
