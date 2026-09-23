#!/usr/bin/env python3
"""The keep / discard rule for autoresearch: a Pareto frontier over four objectives.

| objective   | better | margin        | what                                                                  |
|-------------|--------|---------------|-----------------------------------------------------------------------|
| `quality`   | higher | 0.005 (abs)   | macro dataset accuracy minus ECE on single-answer questions            |
| `games`     | higher | 0.03 (abs)    | mean normalized game score, 0 = random play, 1 = the scripted expert   |
| `params_m`  | lower  | 1% (rel)      | parameters of the saved model, millions                                |
| `latency_x` | lower  | 3% (rel)      | median predict time / the base checkpoint's, timed in the same L4 run  |

Upstream autoresearch keeps an experiment when its single metric (val_bpb) improves. Here an experiment is kept when
it extends the frontier: no already-kept result is at least as good on every objective within the noise margins.
Kept results that the new one strictly beats on every objective drop off the frontier. Progress is tracked as the
frontier's hypervolume (the volume of normalized objective space it dominates against a fixed reference point),
which only grows when the frontier moves outward.

    python autoresearch/pareto.py add runs/<tag>/<commit>.json --tsv runs/<tag>/results.tsv --desc "..."
    python autoresearch/pareto.py show --tsv runs/<tag>/results.tsv

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
# noise did not move it; 0.03 is one game moving 0.3 in the 10-game mean, kept until a stronger player is repeated.
OBJECTIVES = (
    ("quality", "max", 0.005, False),
    ("games", "max", 0.03, False),
    ("params_m", "min", 0.01, True),
    ("latency_x", "min", 0.03, True),
)
NAMES = tuple(o[0] for o in OBJECTIVES)

# Hypervolume reference point: quality from 0, games from -0.5 (the clip floor of a game's normalized score), params
# and latency up to 1.5x the baseline's.
REF_SCALE = 1.5
GAMES_FLOOR = -0.5

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
        for k in FLOATS:
            r[k] = float(r[k])
    return rows


def append_tsv(path: str, row: Dict) -> None:
    new = not os.path.exists(path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a") as f:
        if new:
            f.write("\t".join(COLUMNS) + "\n")
        f.write("\t".join(_cell(row[c]) for c in COLUMNS) + "\n")


def _cell(v) -> str:
    if isinstance(v, float):
        return "%.4f" % v
    return str(v).replace("\t", " ").replace("\n", " ")


def row_from_result(res: Dict, commit: str, description: str) -> Dict:
    s = res["summary"]
    row = {k: float(s[k]) for k in FLOATS}
    row.update(commit=commit, status="", description=description)
    return row


def crash_row(commit: str, description: str) -> Dict:
    row = {k: 0.0 for k in FLOATS}
    row.update(commit=commit, status="crash", description=description)
    return row


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
    s = sub.add_parser("show", help="print the frontier and its hypervolume")
    s.add_argument("--tsv", required=True)
    args = ap.parse_args(argv)
    rows = read_tsv(args.tsv)
    if args.cmd == "show":
        print(show(rows))
        return 0
    if args.result == "crash":
        append_tsv(args.tsv, crash_row(args.commit or "-", args.desc))
        print("status: crash")
        return 0
    with open(args.result) as f:
        res = json.load(f)
    row = row_from_result(res, args.commit or res.get("commit", "-"), args.desc or res.get("description", ""))
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
