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
import os
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
    out = {"model": results[0]["model"], "code": results[0].get("code") or {}, "errors": [],
           "started": min((r["started"] for r in results if r.get("started")), default=None)}
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
    gpu = (", %s" % ds["gpu"]) if ds.get("gpu") else ""  # results from before evaluate recorded it have none
    lines = ["#### Datasets (%s split, calibrated%s)" % (split, gpu), ""]
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
    if games.get("control"):
        lines += ["**Classic control** (greedy; normalized: 0 = random, 1 = scripted expert; solved: share of episodes at the solved score)", "",
                  "| game | model | random | expert | normalized | solved | top actions |", "|---|---:|---:|---:|---:|---:|---|"]
        for r in sorted(games["control"], key=lambda r: r["game"]):
            lines.append("| %s | %.1f | %.1f | %.1f | %s | %s | %s |" % (r["game"], r["model_score"], r["random_score"], r["expert_score"],
                                                                     _fmt(r.get("normalized"), "%.2f"), _pct(r["model_solved"]), _top_actions(r["actions"])))
        lines.append("")
    return lines


def _top_actions(actions: Dict[str, int], k: int = 3) -> str:
    total = max(1, sum(actions.values()))
    return ", ".join("%s %d%%" % (a, 100 * n / total) for a, n in sorted(actions.items(), key=lambda kv: (-kv[1], kv[0]))[:k])


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
    ap.add_argument("--html", default="", help="also write a self-contained HTML report to this path")
    ap.add_argument("--title", default="", help="title for --html / --doc (default: the checkpoint name)")
    ap.add_argument("--doc", default="", help="also write a standalone Markdown report with Mermaid charts to this path")
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
    if args.doc:
        rel = source_links(args.doc, args.results)
        with open(args.doc, "w") as f:
            f.write(render_doc(result, args.title, rel))
    if args.html:
        with open(args.html, "w") as f:
            f.write(render_html(result, parse_status(args.status), args.run_url, args.title))
    return 0



# -- HTML ---------------------------------------------------------------------------------------------------------
# A self-contained page (inline CSS and SVG, fonts from Google Fonts with system fallbacks, no scripts) that opens
# with plain-language findings computed from the results, then one section per part with a chart and the numbers.

GROUP_LABELS = {"vqa": "VQA (official val)", "cauldron": "Cauldron holdouts", "cauldronfull": "Cauldron holdouts (full)",
                "score": "Rubric scoring", "eval": "Held-out eval sets"}
GROUP_ABOUT = {
    "vqa": "The three original post-training sets on their official validation splits.",
    "cauldron": "Held-out rows of the 19 Cauldron subsets the model was post-trained on.",
    "cauldronfull": "Held-out rows of the uncapped Cauldron preparation.",
    "score": "Graded-level questions (the score head): response quality, aesthetics, generated-image ratings, damage.",
    "eval": "Sets the model never trained on: photo quality, prompt alignment, CIFAR-10H, facial expressions, "
            "VizWiz answerability and POPE hallucination probes.",
}

CSS = """
:root{--ground:#F5F7F3;--panel:#FFFFFF;--ink:#18211E;--muted:#5D6A64;--rule:#D9E0DB;--accent:#0E6B63;--accent-soft:#D5EAE6;
--good:#2F7A4C;--warn:#A86B12;--bad:#B0322A;--bar:#0E6B63;--bar2:#9AB5AD;--bar3:#C9D3CE}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;--ground:#101513;--panel:#161D1A;
--ink:#E3EAE6;--muted:#93A39C;--rule:#2A3531;--accent:#5BC6B7;--accent-soft:#1C3531;--good:#6CC08A;--warn:#E0A94A;
--bad:#EE7B6F;--bar:#5BC6B7;--bar2:#4C6A62;--bar3:#33423D}}
:root[data-theme="dark"]{color-scheme:dark;--ground:#101513;--panel:#161D1A;--ink:#E3EAE6;--muted:#93A39C;--rule:#2A3531;
--accent:#5BC6B7;--accent-soft:#1C3531;--good:#6CC08A;--warn:#E0A94A;--bad:#EE7B6F;--bar:#5BC6B7;--bar2:#4C6A62;--bar3:#33423D}
body{background:var(--ground);color:var(--ink);font:15px/1.55 "IBM Plex Sans",system-ui,-apple-system,"Segoe UI",sans-serif;
padding-inline:16px;padding-block:28px 64px}
.wrap{max-width:1040px;margin:0 auto;display:flex;flex-direction:column;gap:40px}
h1,h2,h3{font-family:"IBM Plex Sans Condensed","Arial Narrow",system-ui,sans-serif;text-wrap:balance;margin:0;line-height:1.15}
h1{font-size:34px;font-weight:600;letter-spacing:-.01em}
h2{font-size:24px;font-weight:600}
h3{font-size:17px;font-weight:600}
p{margin:0;max-width:68ch}
.mono,td.num,.stat b{font-family:"IBM Plex Mono",ui-monospace,Menlo,monospace;font-variant-numeric:tabular-nums}
.eyebrow{font:600 12px/1.2 "IBM Plex Mono",ui-monospace,monospace;letter-spacing:.08em;text-transform:uppercase;color:var(--accent)}
header{display:flex;flex-direction:column;gap:10px;border-bottom:1px solid var(--rule);padding-bottom:22px}
.meta{display:flex;flex-wrap:wrap;gap:6px 18px;color:var(--muted);font-size:13px}
.meta code{font-family:"IBM Plex Mono",ui-monospace,monospace;color:var(--ink)}
section{display:flex;flex-direction:column;gap:16px}
.lede{color:var(--muted)}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:1px;background:var(--rule);border:1px solid var(--rule)}
.stat{background:var(--panel);padding:14px 16px;display:flex;flex-direction:column;gap:4px}
.stat span{font-size:12px;color:var(--muted);letter-spacing:.02em}
.stat b{font-size:24px;font-weight:500}
.stat small{font-size:12px;color:var(--muted)}
.findings{list-style:none;margin:0;padding:0;display:flex;flex-direction:column;gap:10px}
.findings li{display:grid;grid-template-columns:74px 1fr;gap:12px;align-items:baseline}
.pill{font:600 11px/1 "IBM Plex Mono",ui-monospace,monospace;letter-spacing:.06em;text-transform:uppercase;
padding:5px 7px;border-radius:3px;text-align:center;border:1px solid currentColor}
.pill.good{color:var(--good)}.pill.warn{color:var(--warn)}.pill.bad{color:var(--bad)}.pill.info{color:var(--accent)}
.alert{border-left:3px solid var(--bad);padding:10px 14px;background:var(--panel)}
.chart{overflow-x:auto}
.chart svg{display:block;max-width:100%;height:auto}
.chart text{fill:var(--ink);font:12px "IBM Plex Sans",system-ui,sans-serif}
.chart text.dim{fill:var(--muted)}
.chart text.val{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:11px}
.table{overflow-x:auto;border:1px solid var(--rule);background:var(--panel)}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{padding:7px 10px;text-align:left;border-bottom:1px solid var(--rule);white-space:nowrap}
th{font-weight:600;color:var(--muted);font-size:12px}
td.num,th.num{text-align:right}
tr:last-child td{border-bottom:0}
.muted{color:var(--muted)}
.good{color:var(--good)}.warn{color:var(--warn)}.bad{color:var(--bad)}
details{border:1px solid var(--rule);background:var(--panel)}
details summary{cursor:pointer;padding:10px 14px;font-weight:600}
details summary:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
details .table{border:0;border-top:1px solid var(--rule)}
.stack{display:flex;flex-direction:column;gap:28px}
.card{display:flex;flex-direction:column;gap:10px}
dl.gloss{display:grid;grid-template-columns:max-content 1fr;gap:6px 16px;margin:0;font-size:13px}
dl.gloss dt{font-family:"IBM Plex Mono",ui-monospace,monospace;color:var(--accent)}
dl.gloss dd{margin:0;color:var(--muted)}
@media (max-width:560px){h1{font-size:27px}.findings li{grid-template-columns:1fr;gap:4px}.findings .pill{justify-self:start}}
"""


def _e(s) -> str:
    import html
    return html.escape(str(s), quote=True)


def _hbars(rows: List[tuple], vmax: float = 1.0, fmt=lambda v: "%.0f%%" % (100 * v), label_w: int = 190,
           width: int = 640, marks: Optional[Dict[str, float]] = None) -> str:
    """Horizontal bars: rows of (label, value, css color var, note). ``marks`` draws named reference ticks."""
    bar_h, gap, top = 16, 8, 22
    plot_w = width - label_w - 70
    h = top + len(rows) * (bar_h + gap) + 6
    x = lambda v: label_w + max(0.0, min(v, vmax)) / vmax * plot_w  # noqa: E731
    out = ['<svg viewBox="0 0 %d %d" width="%d" role="img">' % (width, h, width)]
    for t in (0.0, 0.25, 0.5, 0.75, 1.0):
        v = t * vmax
        out.append('<line x1="%.1f" x2="%.1f" y1="%d" y2="%d" stroke="var(--rule)" stroke-width="1"/>' % (x(v), x(v), top - 6, h - 4))
        out.append('<text class="dim val" x="%.1f" y="%d" text-anchor="middle">%s</text>' % (x(v), top - 10, _e(fmt(v))))
    for i, (label, v, color, note) in enumerate(rows):
        y = top + i * (bar_h + gap)
        out.append('<text x="%d" y="%d" text-anchor="end">%s</text>' % (label_w - 10, y + bar_h - 4, _e(label)))
        out.append('<rect x="%d" y="%d" width="%.1f" height="%d" fill="var(%s)"/>' % (label_w, y, max(1.0, x(v) - label_w), bar_h, color))
        out.append('<text class="val" x="%.1f" y="%d">%s%s</text>' % (x(v) + 6, y + bar_h - 4, _e(fmt(v)),
                                                                    (' <tspan class="dim">%s</tspan>' % _e(note)) if note else ""))
    out.append("</svg>")
    return "".join(out)


def _pair_bars(rows: List[tuple], width: int = 640, label_w: int = 190) -> str:
    """Model vs prior cross-entropy per dataset: two thin bars each, shorter is better."""
    bar_h, gap, top = 9, 12, 24
    vmax = max([max(m, p) for _, m, p in rows] + [0.1]) * 1.05
    plot_w = width - label_w - 70
    h = top + len(rows) * (2 * bar_h + 2 + gap) + 4
    x = lambda v: label_w + v / vmax * plot_w  # noqa: E731
    out = ['<svg viewBox="0 0 %d %d" width="%d" role="img">' % (width, h, width),
           '<rect x="%d" y="4" width="10" height="10" fill="var(--bar)"/><text x="%d" y="13">model</text>' % (label_w, label_w + 14),
           '<rect x="%d" y="4" width="10" height="10" fill="var(--bar3)"/><text x="%d" y="13">always the average vote</text>'
           % (label_w + 70, label_w + 84)]
    for i, (label, m, p) in enumerate(rows):
        y = top + i * (2 * bar_h + 2 + gap)
        out.append('<text x="%d" y="%d" text-anchor="end">%s</text>' % (label_w - 10, y + bar_h + 5, _e(label)))
        for k, (v, color) in enumerate(((m, "--bar"), (p, "--bar3"))):
            yy = y + k * (bar_h + 2)
            out.append('<rect x="%d" y="%d" width="%.1f" height="%d" fill="var(%s)"/>' % (label_w, yy, max(1.0, x(v) - label_w), bar_h, color))
            out.append('<text class="val" x="%.1f" y="%d">%.3f</text>' % (x(v) + 6, yy + bar_h - 1, v))
    out.append("</svg>")
    return "".join(out)


def _table(head: List[str], rows: List[List[str]], num_from: int = 1) -> str:
    th = "".join('<th class="%s">%s</th>' % ("num" if i >= num_from else "", _e(h)) for i, h in enumerate(head))
    body = "".join("<tr>%s</tr>" % "".join('<td class="%s">%s</td>' % ("num" if i >= num_from else "", c) for i, c in enumerate(r))
                   for r in rows)
    return '<div class="table"><table><thead><tr>%s</tr></thead><tbody>%s</tbody></table></div>' % (th, body)


def _short(name: str) -> str:
    for prefix in ("cauldronfull_", "cauldron_", "score_", "eval_"):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def findings(result: Dict) -> List[tuple]:
    """Plain-language takeaways as (severity, sentence): good / warn / bad / info."""
    out = []
    ds = (result.get("datasets") or {}).get("val_calibrated") or {}
    names = [n for n in ds if n != "all"]
    if names:
        a = ds.get("all")
        best = max(names, key=lambda n: ds[n]["acc"])
        worst = min(names, key=lambda n: ds[n]["acc"])
        if a:
            out.append(("info", "Across %d datasets and %s questions it answers %s correctly (pooled). Best: %s at %s; "
                                "weakest: %s at %s." % (len(names), "{:,}".format(a["n"]), _pct(a["acc"]), _short(best),
                                                         _pct(ds[best]["acc"]), _short(worst), _pct(ds[worst]["acc"]))))
            sev = "good" if a["ece"] < 0.03 else "warn" if a["ece"] < 0.08 else "bad"
            out.append((sev, "Calibration: across all questions, its stated confidence and its actual accuracy differ by "
                             "%.1f points on average (ECE %.3f pooled; under 0.03 means probabilities can be read at "
                             "face value)." % (100 * a["ece"], a["ece"])))
        # ECE scores confidence against the single most-voted label, which says little on sets trained against
        # human vote spreads (a model that copies a 45/40/15 split is "underconfident" by construction), so only
        # hard-label sets are flagged here; vote sets are judged against their prior below
        voted = {n for n in names if "soft_xent" in ds[n] or "xent" in ds[n]}
        off = sorted([n for n in names if ds[n]["ece"] > 0.1 and n not in voted], key=lambda n: -ds[n]["ece"])
        if off:
            out.append(("warn", "Poorly calibrated on %d hard-label set%s (ECE above 0.10): %s." % (
                len(off), "" if len(off) == 1 else "s", ", ".join("%s %.2f" % (_short(n), ds[n]["ece"]) for n in off[:6]))))
        vs = []
        for n in names:
            for key in ("soft_xent", "xent"):
                if key in ds[n] and "prior_" + key in ds[n]:
                    vs.append((n, ds[n][key], ds[n]["prior_" + key]))
                    break
        if vs:
            beat = [n for n, m, p in vs if m < p]
            lost = [n for n, m, p in vs if m >= p]
            sev = "good" if not lost else "warn" if len(beat) >= len(lost) else "bad"
            out.append((sev, "Against human vote spreads it beats the always-predict-the-average baseline on %d of %d "
                             "sets%s. (ECE is not meaningful on these sets: it compares confidence with the single "
                             "most-voted answer, while the model is trained to spread probability like the voters.)"
                        % (len(beat), len(vs), (" (not on %s)" % ", ".join(_short(n) for n in lost)) if lost else "")))
    g = result.get("games") or {}
    for r in g.get("atari") or []:
        if r.get("normalized") is not None:
            sev = "good" if r["normalized"] >= 0.5 else "warn" if r["model_score"] > r["random_score"] else "bad"
            out.append((sev, "%s: scores %.1f against %.1f for random play and %.1f for the expert (normalized %.2f)."
                        % (r["game"], r["model_score"], r["random_score"], r["expert_score"], r["normalized"])))
        else:
            sev = "good" if r["model_score"] > r["random_score"] else "bad"
            out.append((sev, "%s: scores %.1f against %.1f for random play (no expert baseline)." % (r["game"], r["model_score"], r["random_score"])))
    doom = g.get("doom") or {}
    model = next((r for p, r in doom.items() if p == "model"), None)
    if model and "expert" in doom:
        sev = "good" if model["mean_reward"] >= 0.8 * doom["expert"]["mean_reward"] else "warn" if model["mean_reward"] > doom.get("random", {}).get("mean_reward", -1e9) else "bad"
        same = doom.get("always_attack") and abs(doom["always_attack"]["mean_reward"] - model["mean_reward"]) < 1e-6
        out.append((sev, "ViZDoom basic: mean reward %.1f (expert %.1f, random %.1f), kills in %s of episodes.%s"
                    % (model["mean_reward"], doom["expert"]["mean_reward"], doom.get("random", {}).get("mean_reward", float("nan")),
                       _pct(model["kill_rate"]), " Identical to always pressing attack." if same else "")))
    for game in ("maze", "snake"):
        mods = [r for r in g.get(game) or [] if str(r["policy"]).startswith("model")]
        if not mods:
            continue
        if game == "maze":
            solved = ", ".join("%d&times;%d: %s" % (r["size"], r["size"], _pct(r["solve_rate"])) for r in sorted(mods, key=lambda r: r["size"]))
            best = max(r["solve_rate"] for r in mods)
            rnd = max([x["solve_rate"] for x in g[game] if x["policy"] == "random"] or [0.0])
            out.append(("good" if best >= 0.5 else "warn" if best > rnd else "bad",
                        "Maze: solves %s (the shortest-path expert solves all; random play at most %s)." % (solved, _pct(rnd))))
        else:
            r = mods[0]
            exp = next((x for x in g[game] if x["policy"] == "expert" and x["size"] == r["size"]), None)
            out.append(("good" if exp and r["mean_eaten"] >= 0.5 * exp["mean_eaten"] else "warn" if r["mean_eaten"] >= 1 else "bad",
                        "Snake: eats %.1f food per game on a %d&times;%d board%s; games end by %s."
                        % (r["mean_eaten"], r["size"], r["size"], (" (expert %.1f)" % exp["mean_eaten"]) if exp else "",
                           ", ".join("%s %d" % kv for kv in sorted(r["ends"].items())))))
    for r in sorted(g.get("control") or [], key=lambda r: r["game"]):
        norm = r.get("normalized")
        sev = "good" if norm is not None and norm >= 0.5 else "warn" if r["model_score"] > r["random_score"] else "bad"
        out.append((sev, "%s: scores %.1f against %.1f for random play and %.1f for the scripted expert (normalized %s); "
                         "solved in %s of episodes." % (r["game"], r["model_score"], r["random_score"], r["expert_score"],
                                                        _fmt(norm, "%.2f"), _pct(r["model_solved"]))))
    lat = result.get("latency")
    if lat:
        out.append(("info", "Latency: %.0f ms median per predict call on an L4 in bf16 (p90 %.0f ms)." % (lat["median_ms"], lat["p90_ms"])))
    return out


def _stat(label: str, value: str, note: str = "") -> str:
    return '<div class="stat"><span>%s</span><b>%s</b>%s</div>' % (_e(label), value, ("<small>%s</small>" % note) if note else "")


def render_html(result: Dict, status: Optional[Dict[str, str]] = None, run_url: str = "", title: str = "") -> str:
    code = result.get("code") or {}
    status = status or {}
    ds = result.get("datasets") or {}
    cal = ds.get("val_calibrated") or {}
    names = sorted(n for n in cal if n != "all")
    g = result.get("games") or {}
    lat = result.get("latency")
    parts = []
    name = title or result["model"]
    parts.append('<title>%s</title>' % _e(name))
    parts.append('<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
                 '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&'
                 'family=IBM+Plex+Sans+Condensed:wght@500;600&family=IBM+Plex+Sans:wght@400;600&display=swap">')
    parts.append("<style>%s</style>" % CSS)
    parts.append('<div class="wrap">')

    meta = ['checkpoint <code>%s</code>' % _e(result["model"])]
    if code.get("commit"):
        meta.append('code <code>%s@%s</code>%s' % (_e(code.get("branch") or "?"), _e(code["commit"][:8]),
                                                   ' <span class="warn">(uncommitted changes)</span>' if code.get("dirty") else ""))
    if result.get("started"):
        meta.append("run %s" % _e(result["started"].replace("T", " ")[:16] + " UTC"))
    if cal:
        meta.append("split <code>%s</code>" % _e(result.get("val_split", "val")))
    if run_url:
        meta.append('<a href="%s">workflow run</a>' % _e(run_url))
    parts.append('<header><div class="eyebrow">Laya Vision eval report</div><h1>%s</h1><div class="meta">%s</div></header>'
                 % (_e(name), "".join("<span>%s</span>" % m for m in meta)))

    alerts = []
    for part in PARTS:
        state = status.get(part)
        if state and state not in ("success", "skipped") or (state == "success" and not result.get(part)):
            alerts.append("<b>%s</b>: %s" % (part, _e(state if state != "success" else "no results")))
    if result.get("errors"):
        alerts += ["<b>%s</b> failed: <span class=\"mono\">%s</span>" % (_e(e["what"]), _e(e["error"][:200])) for e in result["errors"]]
    if alerts:
        parts.append('<div class="alert">%s</div>' % "<br>".join(alerts))

    stats = []
    if "all" in cal:
        stats.append(_stat("Pooled accuracy", _pct(cal["all"]["acc"]), "%s questions, %d sets" % ("{:,}".format(cal["all"]["n"]), len(names))))
        stats.append(_stat("Calibration error (ECE)", "%.3f" % cal["all"]["ece"], "0 = confidence matches accuracy"))
    if lat:
        stats.append(_stat("Latency, median", "%.0f ms" % lat["median_ms"], "per predict, L4 bf16"))
    mz = [r for r in g.get("maze") or [] if str(r["policy"]).startswith("model")]
    if mz:
        r = max(mz, key=lambda r: r["size"])
        stats.append(_stat("Maze solved, %d\u00d7%d" % (r["size"], r["size"]), _pct(r["solve_rate"]), "expert 100%"))
    sn = [r for r in g.get("snake") or [] if str(r["policy"]).startswith("model")]
    if sn:
        stats.append(_stat("Snake food per game", "%.1f" % sn[0]["mean_eaten"], "%d&times;%d board" % (sn[0]["size"], sn[0]["size"])))
    fs = findings(result)
    parts.append('<section><h2>What happened</h2>%s<ul class="findings">%s</ul></section>' % (
        ('<div class="stats">%s</div>' % "".join(stats)) if stats else "",
        "".join('<li><span class="pill %s">%s</span><span>%s</span></li>' % (sev, {"good": "good", "warn": "watch", "bad": "weak", "info": "note"}[sev], text)
                for sev, text in fs)))

    if names:
        groups: Dict[str, List[str]] = {}
        for n in names:
            groups.setdefault(group_of(n), []).append(n)
        sec = ['<section><div class="eyebrow">Datasets</div><h2>Accuracy and calibration by dataset</h2>'
               '<p class="lede">Calibrated answers (the per-type temperature applied), as <code>predict</code> returns them. '
               'Bars are accuracy; the note is the calibration error (ECE), where lower means its confidence can be taken at face value. '
               '* marks sets scored against human vote spreads, where accuracy and ECE use only the most-voted answer; '
               'read those in the human-disagreement chart instead.</p>']
        order = [k for k in ("eval", "score", "vqa", "cauldron", "cauldronfull") if k in groups]
        for grp in order:
            rows = sorted(groups[grp], key=lambda n: -cal[n]["acc"])
            bars = [(_short(n), cal[n]["acc"], "--bar", "ECE %.2f%s" % (cal[n]["ece"], "*" if ("xent" in cal[n] or "soft_xent" in cal[n]) else ""))
                    for n in rows]
            mean = sum(cal[n]["acc"] for n in rows) / len(rows)
            sec.append('<div class="card"><h3>%s <span class="muted mono">&middot; mean %s</span></h3><p class="lede">%s</p>'
                       '<div class="chart">%s</div></div>' % (_e(GROUP_LABELS.get(grp, grp)), _pct(mean), _e(GROUP_ABOUT.get(grp, "")), _hbars(bars)))
        vs = []
        for n in names:
            for key in ("soft_xent", "xent"):
                if key in cal[n] and "prior_" + key in cal[n]:
                    vs.append((_short(n), cal[n][key], cal[n]["prior_" + key]))
                    break
        if vs:
            sec.append('<div class="card"><h3>Against human disagreement</h3><p class="lede">For sets where every image has many '
                       'human votes, cross-entropy between the model\'s probabilities and the vote spread. Shorter is better; the pale bar '
                       'is a model that always predicts the dataset\'s average vote, so a teal bar shorter than its pale bar means the model '
                       'has learned something per image.</p><div class="chart">%s</div></div>' % _pair_bars(vs))
        rows = []
        for n in names:
            m = cal[n]
            vsh = _vs_votes(m) or '<span class="muted">&ndash;</span>'
            rows.append([_e(n), "{:,}".format(m["n"]), _pct(m["acc"]), "%.3f" % m["ece"], "%.3f" % m["nll"], vsh])
        sec.append('<details><summary>All %d datasets, every number</summary>%s</details>' % (
            len(names), _table(["dataset", "questions", "accuracy", "ECE", "NLL", "vs votes: xent (prior), levels off"], rows)))
        temps = ds.get("temperature")
        if temps:
            sec.append('<p class="muted" style="font-size:13px">Temperatures (choice, score, yes/no): <span class="mono">%s</span></p>'
                       % ", ".join("%.3f" % t for t in temps))
        sec.append("</section>")
        parts.append("".join(sec))

    if g and any(g.values()):
        sec = ['<section><div class="eyebrow">Games</div><h2>Playing games from pixels</h2><p class="lede">Each step the screen is the '
               'image and the options are the game\'s buttons. Every policy plays the same seeded episodes.</p><div class="stack">']
        if g.get("atari"):
            rows = []
            for r in sorted(g["atari"], key=lambda r: r["game"]):
                total = max(1, sum(r["actions"].values()))
                top = ", ".join("%s %d%%" % (a, 100 * n / total) for a, n in sorted(r["actions"].items(), key=lambda kv: -kv[1])[:2])
                norm = r.get("normalized")
                cls = "" if norm is None else "good" if norm >= 0.5 else "warn" if norm > 0 else "bad"
                rows.append([_e(r["game"]), "%.1f" % r["model_score"], "%.1f" % r["random_score"],
                             "&ndash;" if r.get("expert_score") is None else "%.1f" % r["expert_score"],
                             '<span class="%s">%s</span>' % (cls, "&ndash;" if norm is None else "%.2f" % norm), _e(top)])
            sec.append('<div class="card"><h3>Atari</h3><p class="lede">Greedy play, %d episodes, 4,500-step cap. Normalized: 0 = random, 1 = expert.</p>%s</div>'
                       % (len(g["atari"][0].get("model_scores") or [0]) or 3, _table(["game", "model", "random", "expert", "norm.", "most used"], rows)))
        if g.get("doom"):
            d = g["doom"]
            bars = [(("model" if p == "model" else p.replace("_", " ")), r["mean_reward"], "--bar" if p == "model" else "--bar2",
                     "kills %s" % _pct(r["kill_rate"])) for p, r in d.items()]
            lo = min(0.0, min(b[1] for b in bars))
            hi = max(b[1] for b in bars) or 1.0
            shifted = [(l, v - lo, c, n) for l, v, c, n in bars]
            sec.append('<div class="card"><h3>ViZDoom &middot; basic</h3><p class="lede">Mean episode reward (a kill is +100, each step costs; '
                       'bars start at %.0f).</p><div class="chart">%s</div></div>' % (lo, _hbars(shifted, vmax=hi - lo, label_w=190, width=640,
                                                                                            fmt=lambda v, lo=lo: "%.0f" % (v + lo))))
        if g.get("maze"):
            rows = sorted(g["maze"], key=lambda r: (r["size"], 0 if str(r["policy"]).startswith("model") else 1, r["policy"]))
            bars = [("%d&times;%d %s" % (r["size"], r["size"], "model" if str(r["policy"]).startswith("model") else r["policy"]),
                     r["solve_rate"], "--bar" if str(r["policy"]).startswith("model") else "--bar2",
                     ("eff. %.2f" % r["efficiency"]) if r["solve_rate"] else "") for r in rows]
            svg = _hbars([(l.replace("&times;", "×"), v, c, n) for l, v, c, n in bars], label_w=120, width=460)
            sec.append('<div class="card"><h3>Maze</h3><p class="lede">Share of mazes solved within 4&times; the shortest path; '
                       'efficiency is shortest path over steps taken.</p><div class="chart">%s</div></div>' % svg)
        if g.get("snake"):
            rows = sorted(g["snake"], key=lambda r: (r["size"], 0 if str(r["policy"]).startswith("model") else 1))
            vmax = max(r["mean_eaten"] for r in rows) or 1.0
            bars = [("%s" % ("model" if str(r["policy"]).startswith("model") else r["policy"]), r["mean_eaten"],
                     "--bar" if str(r["policy"]).startswith("model") else "--bar2",
                     ", ".join("%s %d" % kv for kv in sorted(r["ends"].items()))) for r in rows]
            sec.append('<div class="card"><h3>Snake &middot; %d&times;%d</h3><p class="lede">Food eaten per game, and how the games ended.</p>'
                       '<div class="chart">%s</div></div>' % (rows[0]["size"], rows[0]["size"],
                                                            _hbars(bars, vmax=vmax, fmt=lambda v: "%.1f" % v)))
        if g.get("control"):
            rows = []
            for r in sorted(g["control"], key=lambda r: r["game"]):
                norm = r.get("normalized")
                cls = "" if norm is None else "good" if norm >= 0.5 else "warn" if norm > 0 else "bad"
                rows.append([_e(r["game"]), "%.1f" % r["model_score"], "%.1f" % r["random_score"], "%.1f" % r["expert_score"],
                             '<span class="%s">%s</span>' % (cls, "&ndash;" if norm is None else "%.2f" % norm),
                             _pct(r["model_solved"]), _e(_top_actions(r["actions"], 2))])
            sec.append('<div class="card"><h3>Classic control</h3><p class="lede">Gymnasium CartPole, Acrobot, MountainCar and LunarLander, '
                       'greedy, %d episodes each; the screen ghosts the previous frame so motion is visible. Normalized: 0 = random, '
                       '1 = scripted expert.</p>%s</div>'
                       % (g["control"][0].get("episodes") or 0, _table(["game", "model", "random", "expert", "norm.", "solved", "most used"], rows)))
        sec.append("</div></section>")
        parts.append("".join(sec))

    if lat:
        extra = [(k, lat[k]) for k in ("n", "mean_views_per_image", "mean_input_tokens") if k in lat]
        parts.append('<section><div class="eyebrow">Latency</div><h2>Speed</h2><p>One <code>predict</code> call with one question on a real '
                     'validation image, preprocessing included: median <b class="mono">%.1f ms</b>, p90 <b class="mono">%.1f ms</b> on an NVIDIA L4 in bf16.%s</p></section>'
                     % (lat["median_ms"], lat["p90_ms"], (" " + ", ".join("%s %s" % (_e(k.replace("_", " ")), _e(round(v, 1) if isinstance(v, float) else v)) for k, v in extra) + ".") if extra else ""))

    parts.append('<section><div class="eyebrow">Reading the numbers</div><h2>Glossary</h2><dl class="gloss">'
                 '<dt>accuracy</dt><dd>Share of questions where the most likely option is the labelled answer.</dd>'
                 '<dt>ECE</dt><dd>Expected calibration error: average gap between stated confidence and actual accuracy, 15 bins. 0.02 means a 70% answer is right about 68&ndash;72% of the time.</dd>'
                 '<dt>NLL</dt><dd>Negative log-likelihood of the right answer; punishes confident mistakes.</dd>'
                 '<dt>xent / soft_xent</dt><dd>Cross-entropy against the human vote spread (score questions / choice and yes-no questions). Compare with the prior: always predicting the average vote.</dd>'
                 '<dt>levels off</dt><dd>For rubric scores: how far the expected level is from the humans\' expected level.</dd>'
                 '<dt>normalized</dt><dd>Atari or classic-control score rescaled so random play is 0 and the expert is 1.</dd></dl></section>')
    parts.append("</div>")
    page = "\n".join(parts)
    return page.encode("ascii", "xmlcharrefreplace").decode("ascii")


# -- Markdown document (for committing to the repo) ---------------------------------------------------------------
# A standalone report that GitHub renders with charts: the charts are Mermaid ``xychart-beta`` blocks, which GitHub
# draws natively, so the file is plain text, diffs cleanly, and needs no images. Deterministic: everything comes
# from the result files in a fixed order and nothing is stamped at render time, so the same inputs give the same
# bytes.

SEVERITY_LABELS = {"good": "good", "warn": "watch", "bad": "weak", "info": "note"}
# teal bars (the HTML report's accent) and a red reference line, readable on GitHub's light and dark grounds
MERMAID_INIT = '%%{init: {"themeVariables": {"xyChart": {"plotColorPalette": "#1F8A7E, #D2553F"}}}}%%'


def _mm_label(s: str) -> str:
    """A Mermaid xychart category: quoted, without characters that break the list."""
    return '"%s"' % str(s).replace('"', "'").replace(",", " ").replace("[", "(").replace("]", ")")


def mermaid_hbar(title: str, labels: List[str], values: List[float], axis: str, lo: float, hi: float,
                 line: Optional[List[float]] = None) -> List[str]:
    """A horizontal Mermaid bar chart (plus an optional line series drawn over the same categories)."""
    fmt = lambda v: ("%.3f" % v).rstrip("0").rstrip(".") if v != int(v) else "%d" % v  # noqa: E731
    out = ["```mermaid", MERMAID_INIT, "xychart-beta horizontal", '    title "%s"' % title.replace('"', "'"),
           "    x-axis [%s]" % ", ".join(_mm_label(l) for l in labels),
           '    y-axis "%s" %s --> %s' % (axis, fmt(lo), fmt(hi)),
           "    bar [%s]" % ", ".join(fmt(round(v, 3)) for v in values)]
    if line is not None:
        out.append("    line [%s]" % ", ".join(fmt(round(v, 3)) for v in line))
    return out + ["```", ""]


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_BLOB = "https://github.com/r33drichards/laya-vision/blob/main/"


def source_links(doc: str, results: List[str], root: str = REPO_ROOT) -> List[str]:
    """How the report links its result files: relative to the report, except for a report inside the MkDocs source
    (``site-docs/``), which cannot link outside its own folder (``mkdocs build --strict`` fails), so there each file
    is linked on GitHub."""
    doc = os.path.abspath(doc)
    if os.path.commonpath([doc, os.path.join(root, "site-docs")]) == os.path.join(root, "site-docs"):
        return [REPO_BLOB + os.path.relpath(os.path.abspath(p), root).replace(os.sep, "/") for p in results]
    base = os.path.dirname(doc)
    return [os.path.relpath(os.path.abspath(p), base) for p in results]


def render_doc(result: Dict, title: str = "", sources: Optional[List[str]] = None) -> str:
    import html as _html

    code = result.get("code") or {}
    ds = (result.get("datasets") or {})
    cal = ds.get("val_calibrated") or {}
    names = sorted(n for n in cal if n != "all")
    g = result.get("games") or {}
    lat = result.get("latency")
    name = title or result["model"]
    L = ["# %s" % name, ""]
    meta = ["Checkpoint `%s`" % result["model"]]
    if code.get("commit"):
        meta.append("code `%s@%s`%s" % (code.get("branch") or "?", code["commit"][:8], " (uncommitted changes)" if code.get("dirty") else ""))
    if result.get("started"):
        meta.append("run %s UTC" % result["started"].replace("T", " ")[:16])
    if cal:
        meta.append("`%s` split" % result.get("val_split", "val"))
    L += [" · ".join(meta), ""]
    if sources:
        L += ["Generated by `scripts/eval_report.py` from %s." % ", ".join("[`%s`](%s)" % (s.rsplit("/", 1)[-1], s) for s in sources), ""]
    if result.get("errors"):
        L += ["> **Errors during the run:** " + "; ".join("%s: `%s`" % (e["what"], e["error"][:120].replace("`", "'")) for e in result["errors"]), ""]

    L += ["## What happened", ""]
    rows = []
    if "all" in cal:
        rows += [("Pooled accuracy", _pct(cal["all"]["acc"]), "{:,} questions over {} datasets".format(cal["all"]["n"], len(names))),
                 ("Calibration error (ECE)", "%.3f" % cal["all"]["ece"], "0 = stated confidence matches accuracy")]
    if lat:
        rows.append(("Latency, median", "%.1f ms" % lat["median_ms"], "per `predict`, L4, bf16, preprocessing included"))
    if rows:
        L += ["| | | |", "|---|---:|---|"] + ["| %s | **%s** | %s |" % r for r in rows] + [""]
    for sev, text in findings(result):
        L.append("- **%s** · %s" % (SEVERITY_LABELS[sev], _html.unescape(text)))
    L.append("")

    if names:
        groups: Dict[str, List[str]] = {}
        for n in names:
            groups.setdefault(group_of(n), []).append(n)
        L += ["## Datasets", "",
              "Calibrated answers, as `predict` returns them. Accuracy counts the most likely option against the labelled "
              "answer; ECE is the gap between stated confidence and accuracy. Sets marked \\* are scored against human vote "
              "spreads, where accuracy and ECE only see the most-voted answer: read those in [Against human "
              "disagreement](#against-human-disagreement).", ""]
        for grp in [k for k in ("eval", "score", "vqa", "cauldron", "cauldronfull") if k in groups]:
            members = sorted(groups[grp], key=lambda n: (-cal[n]["acc"], n))
            mean = sum(cal[n]["acc"] for n in members) / len(members)
            L += ["### %s · mean %s" % (GROUP_LABELS.get(grp, grp), _pct(mean)), "", GROUP_ABOUT.get(grp, ""), ""]
            L += mermaid_hbar("%s: accuracy (%%)" % GROUP_LABELS.get(grp, grp), [_short(n) for n in members],
                              [100 * cal[n]["acc"] for n in members], "accuracy (%)", 0, 100)
            L += ["| dataset | questions | accuracy | ECE | NLL |", "|---|---:|---:|---:|---:|"]
            for n in members:
                m = cal[n]
                star = "\\*" if ("xent" in m or "soft_xent" in m) else ""
                L.append("| %s%s | %s | %s | %.3f | %.3f |" % (_short(n), star, "{:,}".format(m["n"]), _pct(m["acc"]), m["ece"], m["nll"]))
            L.append("")
        vs = []
        for n in names:
            for key in ("soft_xent", "xent"):
                if key in cal[n] and "prior_" + key in cal[n]:
                    vs.append((n, cal[n][key], cal[n]["prior_" + key], cal[n].get("mae")))
                    break
        if vs:
            vs.sort(key=lambda t: (t[1] - t[2], t[0]))
            L += ["### Against human disagreement", "",
                  "For sets where every image has many human votes: cross-entropy between the model's probabilities and the "
                  "vote spread, lower is better, compared with a model that always predicts the dataset's average vote. The "
                  "chart shows the baseline's cross-entropy divided by the model's: above the red line at 1, the model "
                  "tracks how people vote on each image better than the average does; below it, worse.", ""]
            ratios = [p / m for _, m, p, _ in vs]
            L += mermaid_hbar("Average-vote baseline / model cross-entropy (above 1 = beats the baseline)", [_short(n) for n, _, _, _ in vs],
                              ratios, "baseline / model", 0, max(2, round(max(ratios) * 1.1 + 0.05, 1)), [1.0] * len(vs))
            L += ["| dataset | model | average vote | difference | levels off |", "|---|---:|---:|---:|---:|"]
            for n, m, p, mae in vs:
                L.append("| %s | %.3f | %.3f | %s%.3f %s | %s |" % (_short(n), m, p, "+" if m - p > 0 else "", m - p,
                                                                   "better" if m < p else "worse",
                                                                   "%.2f" % mae if mae is not None else "–"))
            L.append("")
        temps = ds.get("temperature")
        if temps:
            L += ["Temperatures (choice, score, yes/no): %s." % ", ".join("%.3f" % t for t in temps), ""]

    if g and any(g.values()):
        L += ["## Games", "", "Each step the screen is the image and the options are the game's buttons; every policy plays "
                              "the same seeded episodes.", ""]
        if g.get("atari"):
            L += ["### Atari", "", "Greedy play, 4,500-step cap. Normalized: 0 = random, 1 = expert.", "",
                  "| game | model | random | expert | normalized | most used actions |", "|---|---:|---:|---:|---:|---|"]
            for r in sorted(g["atari"], key=lambda r: r["game"]):
                total = max(1, sum(r["actions"].values()))
                top = ", ".join("%s %d%%" % (a, 100 * n / total) for a, n in sorted(r["actions"].items(), key=lambda kv: (-kv[1], kv[0]))[:3])
                L.append("| %s | %.1f | %.1f | %s | %s | %s |" % (r["game"], r["model_score"], r["random_score"],
                                                                 _fmt(r.get("expert_score")), _fmt(r.get("normalized"), "%.2f"), top))
            L.append("")
        if g.get("doom"):
            L += ["### ViZDoom · basic", "", "A kill is +100 and every step costs, so waiting and missing go negative.", "",
                  "| policy | mean reward | kill rate | steps per episode |", "|---|---:|---:|---:|"]
            for p in sorted(g["doom"], key=lambda p: (p != "model", p)):
                r = g["doom"][p]
                L.append("| %s | %.1f | %s | %.1f |" % ("**model**" if p == "model" else p.replace("_", " "), r["mean_reward"], _pct(r["kill_rate"]), r["mean_steps"]))
            L.append("")
        if g.get("maze"):
            rows = sorted(g["maze"], key=lambda r: (r["size"], not str(r["policy"]).startswith("model"), r["policy"]))
            lbl = lambda r: "%d×%d %s" % (r["size"], r["size"], "model" if str(r["policy"]).startswith("model") else r["policy"])  # noqa: E731
            L += ["### Maze", "", "Share of mazes solved within 4× the shortest path; efficiency is shortest path over steps taken.", ""]
            L += mermaid_hbar("Maze: solved (%)", [lbl(r) for r in rows], [100 * r["solve_rate"] for r in rows], "solved (%)", 0, 100)
            L += ["| maze | policy | solved | efficiency | steps per episode |", "|---|---|---:|---:|---:|"]
            for r in rows:
                L.append("| %d×%d | %s | %s | %.2f | %.1f |" % (r["size"], r["size"], "**model**" if str(r["policy"]).startswith("model") else r["policy"],
                                                               _pct(r["solve_rate"]), r["efficiency"], r["mean_steps"]))
            L.append("")
        if g.get("snake"):
            rows = sorted(g["snake"], key=lambda r: (r["size"], not str(r["policy"]).startswith("model"), r["policy"]))
            L += ["### Snake", "", "Food eaten per game, and how the games ended.", ""]
            L += mermaid_hbar("Snake: food per game", ["%d×%d %s" % (r["size"], r["size"], "model" if str(r["policy"]).startswith("model") else r["policy"]) for r in rows],
                              [r["mean_eaten"] for r in rows], "food per game", 0, round(max(r["mean_eaten"] for r in rows) * 1.1 + 0.5))
            L += ["| board | policy | food per game | best game | steps per game | endings |", "|---|---|---:|---:|---:|---|"]
            for r in rows:
                L.append("| %d×%d | %s | %.1f | %d | %.1f | %s |" % (r["size"], r["size"], "**model**" if str(r["policy"]).startswith("model") else r["policy"],
                                                                  r["mean_eaten"], r["max_eaten"], r["mean_steps"],
                                                                  ", ".join("%s %d" % kv for kv in sorted(r["ends"].items()))))
            L.append("")
        if g.get("control"):
            rows = sorted(g["control"], key=lambda r: r["game"])
            L += ["### Classic control", "", "Gymnasium CartPole, Acrobot, MountainCar and LunarLander from pixels, greedy; the screen ghosts "
                  "the previous frame so motion is visible. Normalized: 0 = random, 1 = scripted expert; solved is the share of "
                  "episodes reaching the environment's solved score.", ""]
            L += mermaid_hbar("Classic control: normalized score", [r["game"] for r in rows],
                              [max(-1.0, min(1.5, r["normalized"])) if r.get("normalized") is not None else 0.0 for r in rows],
                              "normalized", -1, 1.5, [1.0] * len(rows))
            L += ["| game | model | random | expert | normalized | solved | most used actions |", "|---|---:|---:|---:|---:|---:|---|"]
            for r in rows:
                L.append("| %s | %.1f | %.1f | %.1f | %s | %s | %s |" % (r["game"], r["model_score"], r["random_score"], r["expert_score"],
                                                                        _fmt(r.get("normalized"), "%.2f"), _pct(r["model_solved"]),
                                                                        _top_actions(r["actions"])))
            L.append("")

    if lat:
        extra = ", ".join("%s %s" % (k.replace("_", " "), round(lat[k], 1) if isinstance(lat[k], float) else lat[k])
                          for k in ("n", "mean_views_per_image", "mean_input_tokens") if k in lat)
        L += ["## Latency", "", "One `predict` call with one question on a real validation image, preprocessing included, on an "
                                "NVIDIA L4 in bf16: median **%.1f ms**, p90 %.1f ms%s." % (lat["median_ms"], lat["p90_ms"], (" (%s)" % extra) if extra else ""), ""]

    L += ["## Reading the numbers", "",
          "- **accuracy**: share of questions where the most likely option is the labelled answer.",
          "- **ECE**: expected calibration error, the average gap between stated confidence and accuracy (15 bins). "
          "0.02 means a 70% answer is right about 68–72% of the time.",
          "- **NLL**: negative log-likelihood of the right answer; punishes confident mistakes.",
          "- **cross-entropy against human votes**: how far the model's probabilities are from the vote spread; compare "
          "with always predicting the dataset's average vote.",
          "- **levels off**: for rubric scores, how far the model's expected level is from the voters' expected level.",
          "- **normalized** (Atari, classic control): score rescaled so random play is 0 and the expert is 1.", ""]
    return "\n".join(L).rstrip() + "\n"


if __name__ == "__main__":
    sys.exit(main())
