"""Summarise a ``run.py`` result, next to the benchmark's published per-item results for the same 80 items.

    python benchmarks/image_jevbench/score.py eval-results/image-jevbench-laya-vision-8b318c9
"""
import argparse
import collections
import gzip
import json
import re
import urllib.request

from build import MMC_REPO, MMC_REVISION

SRC = f"https://raw.githubusercontent.com/{MMC_REPO}/{MMC_REVISION}/data/raw/benchmarks/jevbench/multimodal-preview/source/"
SETS = ["CLEVR-HOPE", "Geometry3K", "ArxivQA", "FinQA"]


def ece(rows, bins=10):
    """Ten-bin ECE on the top option's probability, the benchmark's calibration measure."""
    b = collections.defaultdict(list)
    for r in rows:
        c = max(r["probabilities"].values())
        b[min(int(c * bins), bins - 1)].append((c, r["correct"]))
    return sum(len(v) / len(rows) * abs(sum(c for c, _ in v) - sum(k for _, k in v)) / len(v) for v in b.values())


def chance_corrected(rows):
    return sum((r["correct"] - 1 / r["n_options"]) / (1 - 1 / r["n_options"]) for r in rows) / len(rows)


def published():
    """Per-item correctness of the systems in the benchmark's multimodal preview (``RUN-RESULT.md``)."""
    get = lambda p: urllib.request.urlopen(SRC + p, timeout=60).read().decode()
    log = lambda p: {m[1]: m[2] == "ok" for m in re.finditer(r"^((?:mm|syn)-\d+) (ok|miss(?:/error)?)$", get(p), re.M)}
    js = lambda *ps: {x["item_id"]: x["correct"] is True for p in ps for x in json.loads(get(p))}
    return {"GPT-5.6 Luna": js("results-luna.json", "results-luna-extension.json"),
            "Gemini 3.1 Flash-Lite": js("results-gemini.json", "results-gemini-extension.json"),
            "AlexWortega/openjev 4B v2": log("results-openjev-real.log"),
            "Mapika/decider-2b-vision": log("results-decider-real.log"),
            "kshetrajna12/reflex 4B": log("results-reflex-real.log")}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("result", help="prefix passed to run.py --out")
    ap.add_argument("--offline", action="store_true", help="skip the published comparison")
    args = ap.parse_args()
    rows = [json.loads(line) for line in gzip.open(args.result + ".predictions.jsonl.gz", "rt")]
    meta = json.load(open(args.result + ".meta.json"))
    rebuilt = [r for r in rows if r["set"] == "rebuilt"]
    ex = [r for r in rows if r["set"] == "public_examples"]

    print(f"## {meta['model']}@{meta['revision'][:7]}\n")
    print("| Set | n | Accuracy | Chance | Chance-corrected | ECE (10 bins) |\n|---|---:|---:|---:|---:|---:|")
    for name, rs in [*((s, [r for r in rebuilt if r["dataset"] == s]) for s in SETS),
                     ("**Rebuilt, all**", rebuilt), ("Public examples", ex)]:
        acc = sum(r["correct"] for r in rs) / len(rs)
        chance = sum(1 / r["n_options"] for r in rs) / len(rs)
        print(f"| {name} | {len(rs)} | {acc:.1%} | {chance:.1%} | {chance_corrected(rs):+.1%} | {ece(rs):.3f} |")
    print("\nPublic examples: " + ", ".join(f"{r['id'].split(':')[1]} {'✓' if r['correct'] else '✗'}" for r in ex))
    if args.offline:
        return

    print("\n## Same 80 items, the benchmark's published per-item results\n")
    print("| System | " + " | ".join(SETS) + " | All 80 |\n|---|" + "---:|" * (len(SETS) + 1))
    byds = {r["id"]: r["dataset"] for r in rebuilt}
    table = {f"**{meta['model']}**": {r["id"]: r["correct"] for r in rebuilt}, **published()}
    for name, res in table.items():
        cells = [sum(res[i] for i in byds if byds[i] == s) / sum(byds[i] == s for i in byds) for s in SETS]
        allc = sum(res[i] for i in byds) / len(byds)
        print(f"| {name} | " + " | ".join(f"{c:.0%}" for c in cells) + f" | {allc:.1%} |")


if __name__ == "__main__":
    main()
