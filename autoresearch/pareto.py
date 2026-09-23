#!/usr/bin/env python3
"""The keep / discard rule for autoresearch: a Pareto frontier over (quality up, params down, latency down).

Upstream autoresearch keeps an experiment when its single metric (val_bpb) improves. Here there are three
objectives, so an experiment is kept when it extends the frontier: no already-kept result is at least as good on
all three within the noise margins below. Kept results that the new one strictly beats on all three drop off the
frontier. Progress is tracked as the frontier's hypervolume (the volume of objective space it dominates, in
normalized units against a fixed reference point), which only grows when the frontier moves outward.

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

# Noise margins: a result must beat every kept result by more than these in at least one objective to count as
# new. Starting values, not yet measured: re-run the baseline experiment a few times and set them just above the
# spread you see (program.md asks for this at setup).
EPS_QUALITY = 0.005   # absolute, on the quality score
EPS_PARAMS = 0.01     # relative
EPS_LATENCY = 0.03    # relative

# Hypervolume reference point, in units of the first (baseline) result: quality 0, 1.5x its params, 1.5x its latency.
REF_SCALE = 1.5

COLUMNS = ["commit", "quality", "macro_acc", "ece_hard", "params_m", "latency_ms", "status", "description"]


def dominates(a: Dict, b: Dict, eps: bool = True) -> bool:
    """``a`` is at least as good as ``b`` on every objective (with the noise margins in ``b``'s favour when
    ``eps``): ``b`` then adds nothing ``a`` does not already offer."""
    if eps:
        return (a["quality"] >= b["quality"] - EPS_QUALITY and a["params_m"] <= b["params_m"] * (1 + EPS_PARAMS)
                and a["latency_ms"] <= b["latency_ms"] * (1 + EPS_LATENCY))
    return (a["quality"] >= b["quality"] and a["params_m"] <= b["params_m"] and a["latency_ms"] <= b["latency_ms"]
            and (a["quality"] > b["quality"] or a["params_m"] < b["params_m"] or a["latency_ms"] < b["latency_ms"]))


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


def _normalized(points: Sequence[Dict], base: Dict) -> List[Tuple[float, float, float]]:
    """(quality, params / base params, latency / base latency) per point."""
    return [(p["quality"], p["params_m"] / base["params_m"], p["latency_ms"] / base["latency_ms"]) for p in points]


def _hv2d(pts: Sequence[Tuple[float, float]], ref: Tuple[float, float]) -> float:
    """Area dominated by 2-D points (both minimized) inside the reference box."""
    area, best_y = 0.0, ref[1]
    for x, y in sorted(p for p in pts if p[0] < ref[0] and p[1] < ref[1]):
        if y < best_y:
            area += (ref[0] - x) * (best_y - y)
            best_y = y
    return area


def hypervolume(points: Sequence[Dict], base: Dict) -> float:
    """Volume of normalized objective space the points dominate: quality from 0 up to the point, params and
    latency from the point up to ``REF_SCALE`` times the baseline's. Exact, by slicing along quality."""
    pts = [p for p in _normalized(points, base) if p[0] > 0 and p[1] < REF_SCALE and p[2] < REF_SCALE]
    levels = sorted({p[0] for p in pts}, reverse=True)
    vol = 0.0
    for i, q in enumerate(levels):
        below = levels[i + 1] if i + 1 < len(levels) else 0.0
        vol += (q - below) * _hv2d([(p[1], p[2]) for p in pts if p[0] >= q], (REF_SCALE, REF_SCALE))
    return vol


def read_tsv(path: str) -> List[Dict]:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    for r in rows:
        for k in ("quality", "macro_acc", "ece_hard", "params_m", "latency_ms"):
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
    return {"commit": commit, "quality": s["quality"], "macro_acc": s["macro_acc"], "ece_hard": s["ece_hard"],
            "params_m": s["params_m"], "latency_ms": s["latency_ms"], "status": "", "description": description}


def crash_row(commit: str, description: str) -> Dict:
    return {"commit": commit, "quality": 0.0, "macro_acc": 0.0, "ece_hard": 0.0, "params_m": 0.0, "latency_ms": 0.0,
            "status": "crash", "description": description}


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
            lines.append("  %s  quality %.4f  acc %.4f  ece %.4f  %7.1fM params  %6.1f ms  %s" % (
                r["commit"], r["quality"], r["macro_acc"], r["ece_hard"], r["params_m"], r["latency_ms"], r["description"]))
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
