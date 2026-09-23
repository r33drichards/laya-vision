#!/usr/bin/env python3
"""Markdown report of ``full_eval`` results, for a pull-request comment or a GitHub Actions job summary.

    python scripts/eval_report.py eval-results/*.json [--run-url URL] [--status datasets=success,games=failure]

Takes one or more ``full_eval`` result files (the ``eval`` workflow writes one per part: datasets, games,
latency) for the same checkpoint and merges them into one report. ``--status`` names each part's job result, so
a part whose job failed or was skipped says so instead of silently missing. The first line is a hidden marker
(``<!-- laya-eval model=... -->``) that the workflow uses to update its earlier comment for the same checkpoint.
Standard library only, so the reporting job needs no torch.
"""
import argparse
import json
import sys
from typing import Dict, List, Optional

PARTS = ("datasets", "games", "latency")


def marker(model: str) -> str:
    return "<!-- laya-eval model=%s -->" % model


def merge(results: List[Dict]) -> Dict:
    """Part files -> one result: the first non-empty value of each part, errors concatenated."""
    if not results:
        raise ValueError("no result files")
    models = {r.get("model") for r in results}
    if len(models) != 1:
        raise ValueError("results are for different checkpoints: %s" % sorted(map(str, models)))
    out = {"model": results[0]["model"], "code": results[0].get("code") or {}, "errors": []}
    for r in results:
        for part in PARTS:
            if r.get(part) and not out.get(part):
                out[part] = r[part]
                if part == "datasets":
                    out["val_split"] = r.get("val_split", "val")
        out["errors"] += r.get("errors") or []
    return out


def group_of(name: str) -> str:
    """Which ``DATASET_GROUPS`` group a prepared dataset belongs to (by its name prefix)."""
    for prefix in ("cauldronfull", "cauldron", "score", "eval"):
        if name.startswith(prefix + "_"):
            return prefix
    return "vqa"


def _pct(v: float) -> str:
    return "%.1f%%" % (100 * v)


def _vs_votes(m: Dict) -> str:
    """The human-vote / ordinal columns: cross-entropy against the vote histogram with the prior's in brackets."""
    parts = []
    for key in ("xent", "soft_xent"):
        if key in m:
            prior = m.get("prior_" + key)
            parts.append("%.3f%s" % (m[key], " (%.3f)" % prior if prior is not None else ""))
    if "mae" in m:
        parts.append("%.2f levels off" % m["mae"])
    return ", ".join(parts)


def datasets_section(ds: Dict, split: str) -> List[str]:
    cal = ds["val_calibrated"]
    names = sorted(n for n in cal if n != "all")
    lines = ["#### Datasets (%s split, calibrated)" % split, ""]
    groups: Dict[str, List[str]] = {}
    for n in names:
        groups.setdefault(group_of(n), []).append(n)
    lines += ["| group | sets | questions | mean acc | mean ECE |", "|---|---:|---:|---:|---:|"]
    for g in sorted(groups):
        ms = [cal[n] for n in groups[g]]
        lines.append("| %s | %d | %d | %s | %.3f |" % (g, len(ms), sum(m["n"] for m in ms),
                                                     _pct(sum(m["acc"] for m in ms) / len(ms)),
                                                     sum(m["ece"] for m in ms) / len(ms)))
    if "all" in cal:
        a = cal["all"]
        lines.append("| **all** (pooled) | %d | %d | %s | %.3f |" % (len(names), a["n"], _pct(a["acc"]), a["ece"]))
    lines += ["", "<details><summary>Per dataset</summary>", "",
              "| dataset | n | acc | ECE | NLL | vs human votes: xent (prior) |", "|---|---:|---:|---:|---:|---|"]
    for n in names:
        m = cal[n]
        lines.append("| %s | %d | %s | %.3f | %.3f | %s |" % (n, m["n"], _pct(m["acc"]), m["ece"], m["nll"], _vs_votes(m)))
    temps = ds.get("temperature")
    lines += ["", "Temperatures (choice, score, noul): %s" % ", ".join("%.3f" % t for t in temps) if temps else "",
              "</details>", ""]
    return lines


def latency_section(lat: Dict) -> List[str]:
    return ["#### Latency", "", "`predict` on an L4 in bf16: median **%.1f ms**, p90 %.1f ms." % (lat["median_ms"], lat["p90_ms"]), ""]


def _fmt(v: Optional[float], fmt: str = "%.1f") -> str:
    return "–" if v is None else fmt % v


def games_section(games: Dict) -> List[str]:
    lines = ["#### Games", ""]
    if games.get("atari"):
        lines += ["**Atari** (greedy; normalized: 0 = random, 1 = expert)", "",
                  "| game | model | random | expert | normalized | top actions |", "|---|---:|---:|---:|---:|---|"]
        for r in sorted(games["atari"], key=lambda r: r["game"]):
            total = max(1, sum(r["actions"].values()))
            top = ", ".join("%s %d%%" % (a, 100 * n / total) for a, n in sorted(r["actions"].items(), key=lambda kv: -kv[1])[:3])
            lines.append("| %s | %.1f | %.1f | %s | %s | %s |" % (r["game"], r["model_score"], r["random_score"],
                                                               _fmt(r.get("expert_score")), _fmt(r.get("normalized"), "%.2f"), top))
        lines.append("")
    if games.get("doom"):
        lines += ["**ViZDoom basic**", "", "| policy | mean reward | kill rate | steps / episode |", "|---|---:|---:|---:|"]
        for r in games["doom"].values():
            lines.append("| %s | %.1f | %s | %.1f |" % (r["policy"], r["mean_reward"], _pct(r["kill_rate"]), r["mean_steps"]))
        lines.append("")
    if games.get("maze"):
        lines += ["**Maze** (efficiency: shortest path / steps taken, on solved mazes)", "",
                  "| policy | size | solved | efficiency | steps / episode |", "|---|---:|---:|---:|---:|"]
        for r in sorted(games["maze"], key=lambda r: (r["size"], r["policy"])):
            lines.append("| %s | %d | %s | %.2f | %.1f |" % (r["policy"], r["size"], _pct(r["solve_rate"]), r["efficiency"], r["mean_steps"]))
        lines.append("")
    if games.get("snake"):
        lines += ["**Snake**", "", "| policy | size | food / episode | best | steps / episode | how episodes ended |",
                  "|---|---:|---:|---:|---:|---|"]
        for r in sorted(games["snake"], key=lambda r: (r["size"], r["policy"])):
            ends = ", ".join("%s %d" % kv for kv in sorted(r["ends"].items()))
            lines.append("| %s | %d | %.1f | %d | %.1f | %s |" % (r["policy"], r["size"], r["mean_eaten"], r["max_eaten"], r["mean_steps"], ends))
        lines.append("")
    return lines


def render(result: Dict, status: Optional[Dict[str, str]] = None, run_url: str = "") -> str:
    code = result.get("code") or {}
    commit = (code.get("commit") or "")[:8] or "unknown commit"
    where = "`%s@%s`%s" % (code.get("branch") or "?", commit, " (uncommitted changes)" if code.get("dirty") else "")
    lines = [marker(result["model"]), "### Eval: `%s`" % result["model"], "",
             "Code %s%s" % (where, " · [workflow run](%s)" % run_url if run_url else ""), ""]
    status = status or {}
    for part in PARTS:
        state = status.get(part)
        if state and state not in ("success", "skipped") or (state == "success" and not result.get(part)):
            lines.append("> ⚠️ **%s**: %s%s" % (part, state if state != "success" else "no results",
                                                   " – [logs](%s)" % run_url if run_url else ""))
    if result.get("errors"):
        lines += ["", "<details><summary>%d error(s) inside the run</summary>" % len(result["errors"]), ""]
        lines += ["- %s: `%s`" % (e["what"], e["error"].replace("`", "'")[:300]) for e in result["errors"]]
        lines += ["", "</details>"]
    lines.append("")
    if result.get("datasets"):
        lines += datasets_section(result["datasets"], result.get("val_split", "val"))
    if result.get("latency"):
        lines += latency_section(result["latency"])
    if result.get("games"):
        lines += games_section(result["games"])
    return "\n".join(lines).rstrip() + "\n"


def parse_status(s: str) -> Dict[str, str]:
    """``"datasets=success,games=failure"`` -> ``{"datasets": "success", "games": "failure"}``."""
    out = {}
    for kv in s.split(","):
        if "=" in kv:
            k, v = kv.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="*", help="full_eval result JSON files for one checkpoint")
    ap.add_argument("--model", default="", help="checkpoint name, for the report when no result file exists")
    ap.add_argument("--run-url", default="")
    ap.add_argument("--status", default="", help="part=job result, comma-separated")
    args = ap.parse_args(argv)
    loaded = []
    for path in args.results:
        with open(path) as f:
            loaded.append(json.load(f))
    if loaded:
        result = merge(loaded)
    elif args.model:
        result = {"model": args.model}
    else:
        ap.error("give result files or --model")
    sys.stdout.write(render(result, parse_status(args.status), args.run_url))
    return 0


if __name__ == "__main__":
    sys.exit(main())
