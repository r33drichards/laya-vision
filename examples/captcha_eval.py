#!/usr/bin/env python3
"""Score the image model on Open CaptchaWorld, as typed decisions.

    python examples/captcha_eval.py --data /path/to/OpenCaptchaWorld/captcha_data

Reports pass@1 per puzzle (the benchmark's own metric) next to per-decision accuracy and ECE, plus a
random-guess baseline so a near-chance number is recognisable as one.

``--control shuffle`` re-runs with each puzzle's images replaced by another puzzle's of the same type.
If the score does not drop, the model is answering from the question text alone and the real score means
nothing -- the failure mode that killed this repo's SigLIP projector branch.
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import laya
from laya.captcha import EXCLUDED, LOADERS, load_puzzles
from laya.common import ece_score


def chance(p) -> float:
    """Probability a uniform random guesser passes this puzzle."""
    from math import comb

    if p.mode == "choice":
        return 1.0 / p.n_options
    if p.mode == "argmax":
        n_correct = len(p.truth) if isinstance(p.truth, (set, frozenset)) else 1
        return n_correct / p.n_options
    return 1.0 / comb(p.n_options, len(p.truth)) if len(p.truth) <= p.n_options else 0.0


def group_decisions(puzzle):
    """Bucket a puzzle's decisions by the images they use, so each image set is encoded once."""
    groups = defaultdict(list)
    for d in puzzle.decisions:
        key = tuple(zip(d.images, d.crops or [None] * len(d.images)))
        groups[key].append(d)
    return list(groups.values())


def run(agent, puzzles, control=None, verbose=False):
    rows, t0 = [], time.time()
    by_type = defaultdict(list)
    for p in puzzles:
        by_type[p.ctype].append(p)

    # for the ablation, map each puzzle to another puzzle's images within the same type
    swap = {}
    if control == "shuffle":
        for t, ps in by_type.items():
            for i, p in enumerate(ps):
                swap[id(p)] = ps[(i + 1) % len(ps)]

    n_calls = 0
    for i, p in enumerate(puzzles):
        answers = {}
        donor = swap.get(id(p))
        dgroups = group_decisions(donor) if donor is not None else None
        for gi, group in enumerate(group_decisions(p)):
            # keep this puzzle's questions; under the ablation, show the donor's images instead
            src = group[0] if dgroups is None else dgroups[gi % len(dgroups)][0]
            qs = {d.qid: d.question for d in group}
            answers.update(agent.predict(src.state(), qs)["answers"])
            n_calls += 1
        passed, recs = p.grade(answers)
        rows.append({"type": p.ctype, "pid": p.pid, "passed": bool(passed),
                     "chance": chance(p), "records": recs})
        if verbose and (i + 1) % 25 == 0:
            done = sum(r["passed"] for r in rows)
            print("  %4d/%d  pass@1 %.3f  (%.1f s)" % (i + 1, len(puzzles), done / len(rows), time.time() - t0),
                  flush=True)
    return rows, time.time() - t0, n_calls


def auc(prob, truth):
    """Rank AUC: probability the model scores a true option above a false one. 0.5 is blind."""
    pos, neg = prob[truth == 1], prob[truth == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order), float)
    ranks[order] = np.arange(1, len(order) + 1)
    # average ranks over ties so a constant output scores exactly 0.5
    vals = np.concatenate([pos, neg])
    for v in np.unique(vals):
        m = vals == v
        ranks[m] = ranks[m].mean()
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def _stats(rs):
    recs = [d for r in rs for d in r["records"]]
    conf = np.array([d["conf"] for d in recs])
    corr = np.array([float(d["correct"]) for d in recs])
    nl = [d for d in recs if d["prob"] is not None]
    prob = np.array([d["prob"] for d in nl])
    truth = np.array([float(d["truth"]) for d in nl])
    return {
        "n": len(rs),
        "pass_at_1": float(np.mean([r["passed"] for r in rs])),
        "chance": float(np.mean([r["chance"] for r in rs])),
        "decision_accuracy": float(corr.mean()),
        "ece": float(ece_score(conf, corr)),
        "say_yes": float((prob >= 0.5).mean()) if len(prob) else float("nan"),
        "base_rate": float(truth.mean()) if len(truth) else float("nan"),
        "auc": auc(prob, truth) if len(prob) else float("nan"),
    }


HDR = "%-20s %5s %7s %7s  %8s %6s  %6s %6s %6s"
ROW = "%-20s %5d %7.3f %7.3f  %8.3f %6.3f  %6.3f %6.3f %6.3f"


def report(rows, title):
    print("\n" + "=" * 92)
    print(title)
    print("=" * 92)
    print(HDR % ("type", "n", "pass@1", "chance", "dec.acc", "ECE", "AUC", "yes%", "true%"))
    print("-" * 92)
    by_type = defaultdict(list)
    for r in rows:
        by_type[r["type"]].append(r)

    per_type = {}
    for t in sorted(by_type):
        s = _stats(by_type[t])
        per_type[t] = s
        print(ROW % (t, s["n"], s["pass_at_1"], s["chance"], s["decision_accuracy"], s["ece"],
                     s["auc"], s["say_yes"], s["base_rate"]))

    overall = _stats(rows)
    print("-" * 92)
    print(ROW % ("ALL", overall["n"], overall["pass_at_1"], overall["chance"], overall["decision_accuracy"],
                 overall["ece"], overall["auc"], overall["say_yes"], overall["base_rate"]))
    print("\nAUC 0.5 = the model ranks correct and incorrect options identically (blind).")
    print("yes%% far above true%% means it accepts everything, which makes dec.acc uninformative.")
    overall["per_type"] = per_type
    return overall


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="path to OpenCaptchaWorld/captcha_data")
    ap.add_argument("--model", default="thaitea/laya-vision-smolvlm-256m")
    ap.add_argument("--types", nargs="*", default=None, help="subset of types (default: all supported)")
    ap.add_argument("--limit", type=int, default=0, help="cap puzzles per type, for a quick pass")
    ap.add_argument("--device", default=None)
    ap.add_argument("--control", choices=["shuffle"], default=None, help="image ablation")
    ap.add_argument("--out", default=None, help="write per-puzzle results as JSON")
    ap.add_argument("-q", "--quiet", action="store_true")
    args = ap.parse_args()

    puzzles = load_puzzles(args.data, args.types)
    if args.limit:
        kept, seen = [], defaultdict(int)
        for p in puzzles:
            if seen[p.ctype] < args.limit:
                kept.append(p)
                seen[p.ctype] += 1
        puzzles = kept

    n_dec = sum(len(p.decisions) for p in puzzles)
    print("%d puzzles, %d decisions, %d types" % (len(puzzles), n_dec, len({p.ctype for p in puzzles})))
    print("excluded types: %s" % ", ".join("%s (%s)" % kv for kv in sorted(EXCLUDED.items())))
    print("loading %s ..." % args.model, flush=True)
    agent = laya.load_vlm(args.model, device=args.device)

    rows, secs, n_calls = run(agent, puzzles, control=args.control, verbose=not args.quiet)
    summary = report(rows, "Open CaptchaWorld as typed decisions -- %s%s"
                     % (args.model, "  [CONTROL: shuffled images]" if args.control else ""))
    print("\n%d decisions in %d forward calls, %.1f s (%.0f ms/call)"
          % (n_dec, n_calls, secs, 1000 * secs / max(1, n_calls)))

    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"model": args.model, "control": args.control, "summary": summary,
                       "seconds": secs, "puzzles": rows}, fh, indent=2)
        print("wrote %s" % args.out)


if __name__ == "__main__":
    main()
