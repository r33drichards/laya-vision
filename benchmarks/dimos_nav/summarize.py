"""Score a `modal_dimos_nav.py::main` output directory the way the System One navigation page does.

    python benchmarks/dimos_nav/summarize.py results/dimos-nav/<name> --scenes <dimos>/dimos/evals/suites/scenes/habitat

Per case, from dimos's own `nav_metrics.json` (success radius 1 m) and the case's geodesic distance:
arrival, SPL = arrival * geodesic / max(path, geodesic), SoftSPL = (1 - final / start distance, floored at 0)
* geodesic / max(path, geodesic), time to target, and the checkpoint's call latency and how much of each
WorldState it had to cut (from the server's call log). A case without `nav_metrics.json` (crash, timeout
before grading) counts as a failure with SPL 0 and is listed under `errors`. `answers` counts the model's picks
over every call (and, for `noul` questions, how often P(true) reached the agent's 0.7 stop threshold).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics
import tarfile


def case_table(scenes_dir: str) -> dict:
    """case id -> case record, with the ids `habitat_nav.cases_for` gives."""
    out = {}
    for f in sorted(glob.glob(os.path.join(scenes_dir, "*.json"))):
        scene = json.load(open(f))
        seen: dict[str, int] = {}
        for c in scene["cases"]:
            slug = re.sub(r"[^A-Za-z0-9]+", "_", c["label"]).strip("_").lower()
            seen[slug] = seen.get(slug, 0) + 1
            cid = f"{scene['scene_id']}_{slug}" + (f"_{seen[slug]}" if seen[slug] > 1 else "")
            out[cid] = {**c, "scene_id": scene["scene_id"]}
    return out


def read_case(path: str) -> dict:
    got: dict = {"calls": []}
    with tarfile.open(path) as tar:
        for m in tar.getmembers():
            base = os.path.basename(m.name)
            if not m.isfile():
                continue
            if base == "nav_metrics.json" and "raw" not in m.name.split("/")[-2:-1]:
                got.setdefault("nav", json.load(tar.extractfile(m)))
            elif base == "case.json":
                got["meta"] = json.load(tar.extractfile(m))
            elif base == "systemone.jsonl":
                got["calls"] = [json.loads(line) for line in tar.extractfile(m).read().decode().splitlines() if line]
    return got


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--scenes", required=True)
    ap.add_argument("--json", default=None, help="write the per-case rows and totals here")
    args = ap.parse_args()

    cases = case_table(args.scenes)
    rows, errors = [], []
    for path in sorted(glob.glob(os.path.join(args.run_dir, "*.tar.gz"))):
        cid = os.path.basename(path)[: -len(".tar.gz")]
        c, got = cases[cid], read_case(path)
        nav = got.get("nav")
        geo = c["geodesic_m"]
        lat = [x["latency_s"] for x in got["calls"]]
        state_cut = [max((t.get("state_tokens_dropped", 0) for t in x["truncated"].values()), default=0)
                     for x in got["calls"]]
        row = {"case": cid, "scene": c["scene_id"], "label": c["label"], "difficulty": c.get("difficulty"),
               "geodesic_m": geo, "calls": len(lat),
               "latency_p50_s": statistics.median(lat) if lat else None,
               "state_tokens_dropped_p50": statistics.median(state_cut) if state_cut else None}
        if nav is None:
            errors.append(cid)
            row.update(arrived=False, spl=0.0, soft_spl=0.0, error=True)
        else:
            eff = geo / max(nav["path_length_m"], geo, 1e-6)
            start = nav.get("start_distance_m") or nav["straight_line_m"]
            soft = max(0.0, 1.0 - nav["final_distance_m"] / max(start, 1e-6))
            row.update(arrived=bool(nav["reached"]), spl=float(nav["reached"]) * eff, soft_spl=soft * eff,
                       final_distance_m=nav["final_distance_m"], path_length_m=nav["path_length_m"],
                       time_to_object_s=nav["time_to_object_s"], duration_s=nav["duration_s"], bumps=nav["bumps"],
                       finished_declared=nav["finished_declared"])
        rows.append(row)

    n = len(rows)
    picks: dict = {}
    for path in sorted(glob.glob(os.path.join(args.run_dir, "*.tar.gz"))):
        for call in read_case(path)["calls"]:
            for q, a in call["answers"].items():
                if a["type"] == "choice" and q != "target":
                    picks.setdefault(q, {}).setdefault(a["choice"], 0)
                    picks[q][a["choice"]] += 1
                elif a["type"] == "noul":
                    picks.setdefault(q, {}).setdefault("above_0.7", 0)
                    picks[q]["above_0.7"] += a["noul"] >= 0.7
    arrived = [r for r in rows if r["arrived"]]
    tot = {
        "cases": n, "errors": len(errors),
        "arrival": len(arrived) / n if n else None,
        "mean_spl": sum(r["spl"] for r in rows) / n if n else None,
        "mean_soft_spl": sum(r["soft_spl"] for r in rows) / n if n else None,
        "median_time_to_target_s": statistics.median(r["time_to_object_s"] for r in arrived) if arrived else None,
        "latency_p50_s": statistics.median(x for r in rows for x in [r["latency_p50_s"]] if x is not None)
        if any(r["latency_p50_s"] is not None for r in rows) else None,
    }
    by = {}
    for key in ("difficulty", "scene"):
        for v in sorted({str(r[key]) for r in rows}):
            sub = [r for r in rows if str(r[key]) == v]
            by.setdefault(key, {})[v] = {"n": len(sub), "arrival": sum(r["arrived"] for r in sub) / len(sub),
                                         "mean_spl": sum(r["spl"] for r in sub) / len(sub)}
    print(json.dumps({"totals": tot, "by": by, "errors": errors, "answers": picks}, indent=2))
    print("\n| case | difficulty | geodesic m | arrived | SPL | SoftSPL | final m | calls |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        print("| %s | %s | %.1f | %s | %.3f | %.3f | %s | %d |" % (
            r["case"], r["difficulty"], r["geodesic_m"], "yes" if r["arrived"] else "no", r["spl"], r["soft_spl"],
            "%.2f" % r["final_distance_m"] if "final_distance_m" in r else "-", r["calls"]))
    if args.json:
        if os.path.exists(args.json):
            raise SystemExit(f"{args.json} exists (create-only)")
        json.dump({"totals": tot, "by": by, "errors": errors, "answers": picks, "rows": rows}, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
