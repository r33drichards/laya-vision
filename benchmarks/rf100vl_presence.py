"""Score RF100-VL presence questions from row-level evidence, per dataset, per domain and over the benchmark.

    python benchmarks/rf100vl_presence.py results/raw/<name>.predictions.jsonl.gz [--out results/rf100vl-<name>.json]

RF100-VL is scored by COCO box mAP; Laya Vision draws no boxes, so the rows (``modal run modal_app.py::evidence
--datasets rf100vl --val-split test``, questions from ``laya.evalsets.rf100vl_records``) ask, for every test image
and every class of its dataset, whether at least one instance is there. Per source dataset this reports:

* ``presence_ap``: for each class with at least one positive image, the average precision of ranking the dataset's
  test images by P(present); the mean over those classes. It is the image-level analogue of detection AP (no
  localisation) and **not comparable to RF100-VL mAP**. ``chance_ap`` is what a constant score gets: the mean over
  the same classes of their prevalence.
* ``auroc`` over all of the dataset's questions, ``acc`` at P >= 0.5, ``balanced_acc`` (mean of the recall on present
  and absent), and ``prevalence`` (share of questions whose answer is yes).

Domains and the overall ``macro`` average datasets without weights, as the paper averages mAP. The row id
(``rf100vl-<dataset>-<image_id>-c<class>``) and the row's ``dataset`` (``rf100vl_<domain>``) carry everything needed.
Standard library only.
"""
import argparse
import gzip
import json
import os
import re
import sys
from collections import defaultdict

ID_RE = re.compile(r"^rf100vl-(?P<dataset>.+)-(?P<image>\d+_\d+)-c(?P<cls>\d+)$")
METRICS = ("presence_ap", "chance_ap", "auroc", "acc", "balanced_acc", "prevalence")


def average_precision(scores, labels):
    """Non-interpolated AP (the mean of precision at each positive's rank), ties broken by input order."""
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    hits = total = 0
    for rank, i in enumerate(order, 1):
        if labels[i]:
            hits += 1
            total += hits / rank
    return total / hits if hits else None


def auroc(scores, labels):
    """Probability a random positive scores above a random negative, ties counted half (Mann-Whitney)."""
    pairs = sorted(zip(scores, labels))
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if not n_pos or not n_neg:
        return None
    rank_sum, i = 0.0, 0
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        mid = (i + 1 + j) / 2  # average 1-based rank of the tie block
        rank_sum += mid * sum(lab for _, lab in pairs[i:j])
        i = j
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def score_dataset(rows):
    """rows: (image, cls, p_yes, label) for one source dataset -> its metrics."""
    by_cls = defaultdict(list)
    for img, cls, p, y in rows:
        by_cls[cls].append((p, y))
    aps, chance = [], []
    for cls, items in sorted(by_cls.items()):
        labels = [y for _, y in items]
        if any(labels):
            aps.append(average_precision([p for p, _ in items], labels))
            chance.append(sum(labels) / len(labels))
    p = [r[2] for r in rows]
    y = [r[3] for r in rows]
    pos = [pi >= 0.5 for pi, yi in zip(p, y) if yi]
    neg = [pi < 0.5 for pi, yi in zip(p, y) if not yi]
    recalls = [sum(v) / len(v) for v in (pos, neg) if v]
    return {"images": len({r[0] for r in rows}), "classes": len(by_cls), "classes_present": len(aps), "questions": len(rows),
            "presence_ap": sum(aps) / len(aps) if aps else None, "chance_ap": sum(chance) / len(chance) if chance else None,
            "auroc": auroc(p, y), "acc": sum((pi >= 0.5) == bool(yi) for pi, yi in zip(p, y)) / len(rows),
            "balanced_acc": sum(recalls) / len(recalls), "prevalence": sum(y) / len(y)}


def macro(results):
    out = {"datasets": len(results)}
    for m in METRICS:
        vals = [r[m] for r in results if r.get(m) is not None]
        out[m] = sum(vals) / len(vals) if vals else None
    return out


def summarize(rows_path):
    opener = gzip.open if rows_path.endswith(".gz") else open
    per_ds, domain_of = defaultdict(list), {}
    with opener(rows_path, "rt") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if not r["dataset"].startswith("rf100vl_"):
                continue
            m = ID_RE.match(r["id"] or "")
            if not m or r["qtype"] != "noul":
                raise ValueError("not an RF100-VL presence row: %r" % r["id"])
            ds = m["dataset"]
            domain_of[ds] = r["dataset"][len("rf100vl_"):]
            per_ds[ds].append((m["image"], int(m["cls"]), float(r["probs_calibrated"][1]), int(r["label"])))
    datasets = {ds: dict(score_dataset(rows), domain=domain_of[ds]) for ds, rows in sorted(per_ds.items())}
    domains = {dom: macro([d for d in datasets.values() if d["domain"] == dom])
               for dom in sorted(set(domain_of.values()))}
    return {"rows": os.path.basename(rows_path), "macro": macro(list(datasets.values())), "domains": domains,
            "datasets": datasets}


def _fmt(v):
    return "" if v is None else "%.3f" % v


def markdown(res):
    lines = ["| Domain | Datasets | Presence AP | Chance AP | AUROC | Balanced acc | Acc | Prevalence |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, m in list(res["domains"].items()) + [("**all (macro)**", res["macro"])]:
        lines.append("| %s | %d | %s | %s | %s | %s | %s | %s |" % (name, m["datasets"], *(_fmt(m[k]) for k in (
            "presence_ap", "chance_ap", "auroc", "balanced_acc", "acc", "prevalence"))))
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("rows", help="*.predictions.jsonl[.gz] from modal_app.py::evidence")
    ap.add_argument("--out", help="write the summary JSON here (refuses an existing file)")
    args = ap.parse_args(argv)
    res = summarize(args.rows)
    if not res["datasets"]:
        sys.exit("no rf100vl_* rows in %s" % args.rows)
    if args.out:
        if os.path.exists(args.out):
            sys.exit("%s exists; results are create-only" % args.out)
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)
            f.write("\n")
    print(markdown(res))


if __name__ == "__main__":
    main()
