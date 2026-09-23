"""The sampling noise floor of ECE for the robustness report (``laya.robustness``).

ECE on a finite sample is biased upward: even a *perfectly calibrated* model, whose row with max probability ``c``
is right with probability exactly ``c``, scores a positive ECE, because each bin's accuracy is an average of a
few Bernoulli draws. So an ECE of 0.05 on 300 rows says nothing on its own; it has to be read against the ECE the
same confidences would score if they were calibrated. ``ece_noise_floor`` simulates that: keep the model's
confidences, redraw correctness as ``Bernoulli(conf)`` ``n_sim`` times, and score each draw with the same
``laya.common.ece_score`` and 15 bins as ``robustness._ece``. The mean of those ECEs is the floor, its 95th
percentile the level a calibrated model stays under 95% of the time.

``summarize_floor`` does this per dataset and family (``"orig"`` included), on exactly the rows and the
confidence / correctness ``robustness._ece`` uses (max probability; argmax == label, pooled over all rows of the
family, not group-weighted), so ``ece`` matches ``robustness.summarize``. It reports ``ece_ratio`` (ECE over the
floor mean; ~1 is indistinguishable from calibrated) and ``ece_above_floor`` (ECE above the floor's p95). The
floor is a null for *this* sample size and confidence histogram, not an interval for the ECE (it is not the
bootstrap ``ece_ci``). Rows are not independent: a perturbed family holds several variants of each source row,
and Cauldron / VQAv2 images carry several questions, so the independent floor understates the noise there. The
``*_clustered`` floor gives rows of one image cluster a shared draw (fully correlated errors), the conservative
bound; the truth sits between the two. Deterministic given ``seed``; each cell gets its own stream seeded by
``(seed, dataset, family)``, so one cell's floor does not depend on which other cells exist. Offline on committed predictions::

    python -m laya.robustness_floor results/robustness/predictions.jsonl.gz
"""
import json
from typing import Dict, Sequence

import numpy as np

from .common import ece_score
from .robustness import FAMILIES, _ece, _seed_for

BINS = 15  # ``robustness._ece`` calls ``ece_score`` with its default bin count


def ece_noise_floor(conf, n_sim: int = 200, seed: int = 0, bins: int = BINS, groups=None) -> Dict:
    """ECE of a perfectly calibrated model with confidences ``conf`` (max probabilities): correctness drawn
    ``Bernoulli(conf)`` ``n_sim`` times, each draw scored by ``ece_score`` with ``bins`` bins. Returns ``mean``,
    ``p95`` and ``std`` of the simulated ECEs, ``n`` rows and ``n_sim``; NaNs for an empty ``conf``.

    ``groups`` (one hashable per row): rows of a group share one uniform draw (row correct iff ``u < conf``), the
    most correlated a calibrated model's errors on them can be, so the floor becomes an upper bound for rows that
    are variants of one source (or questions on one image) rather than independent samples."""
    conf = np.asarray(conf, dtype=float)
    if conf.size == 0 or n_sim < 1:
        return {"mean": float("nan"), "p95": float("nan"), "std": float("nan"), "n": int(conf.size), "n_sim": int(n_sim)}
    rng = np.random.default_rng(seed)
    if groups is None:
        u = rng.random((n_sim, conf.size))
    else:
        ids: Dict = {}
        gi = np.array([ids.setdefault(g, len(ids)) for g in groups])  # first-appearance order
        u = rng.random((n_sim, len(ids)))[:, gi]
    draws = u < conf  # one row per simulated calibrated model
    e = np.array([ece_score(conf, d.astype(float), bins=bins) for d in draws])
    return {"mean": float(e.mean()), "p95": float(np.percentile(e, 95)), "std": float(e.std()), "n": int(conf.size),
            "n_sim": int(n_sim)}


def floor_stats(rows: Sequence[Dict], n_sim: int = 200, seed: int = 0) -> Dict:
    """For one set of prediction rows: its ECE (``robustness._ece``), the noise floor of its confidences with rows
    independent, and the comparison (``ece_ratio`` = ece / floor mean, ``ece_above_floor`` = ece > floor p95);
    then the same floor with rows of one ``cluster`` (image) sharing their draw (``*_clustered``), the
    conservative one for a family with several variants per source row."""
    ece = _ece(rows)
    conf = [max(r["probs"]) for r in rows]
    fl = ece_noise_floor(conf, n_sim=n_sim, seed=seed)
    cl = ece_noise_floor(conf, n_sim=n_sim, seed=seed, groups=[r["cluster"] for r in rows])
    ratio = ece / fl["mean"] if fl["mean"] > 0 else float("nan")
    return {"n_rows": len(rows), "n_clusters": len({r["cluster"] for r in rows}), "ece": ece,
            "ece_floor_mean": fl["mean"], "ece_floor_p95": fl["p95"], "ece_floor_std": fl["std"],
            "ece_ratio": float(ratio), "ece_above_floor": bool(ece > fl["p95"]),
            "ece_floor_mean_clustered": cl["mean"], "ece_floor_p95_clustered": cl["p95"],
            "ece_above_floor_clustered": bool(ece > cl["p95"])}


def summarize_floor(preds: Sequence[Dict], n_sim: int = 200, seed: int = 0) -> Dict:
    """Per dataset and family (``"orig"`` first, then ``FAMILIES`` order, as in ``robustness.summarize``):
    ``floor_stats``. ``"macro"`` averages ``ece``, the floor mean and the ratio over datasets, and counts the
    datasets whose ECE is above the floor."""
    out: Dict[str, Dict] = {}
    for name in sorted({p["dataset"] for p in preds}):
        rows = [p for p in preds if p["dataset"] == name]
        rep = {}
        for fam in ("orig",) + FAMILIES:
            fr = [p for p in rows if p["family"] == fam]
            if fr:
                rep[fam] = floor_stats(fr, n_sim=n_sim, seed=_seed_for("ece_floor", seed, name, fam))
        out[name] = rep
    macro: Dict[str, Dict] = {}
    for fam in ("orig",) + FAMILIES:
        per = [out[d][fam] for d in out if fam in out[d]]
        if per:
            macro[fam] = {"n_datasets": len(per), "n_above_floor": sum(s["ece_above_floor"] for s in per),
                          "n_above_floor_clustered": sum(s["ece_above_floor_clustered"] for s in per)}
            for k in ("ece", "ece_floor_mean", "ece_ratio"):
                macro[fam][k] = float(np.nanmean([s[k] for s in per]))
    return {"datasets": out, "macro": macro, "n_sim": n_sim, "seed": seed}


def format_floor_table(summary: Dict) -> str:
    """A markdown table: one line per dataset and family, ECE against its calibrated noise floor (``*``: above
    the floor's p95), with rows independent and with each image cluster sharing its draw."""
    lines = ["| dataset | family | rows | clusters | ECE | floor mean | floor p95 | ECE / floor | above p95 "
             "| clustered p95 | above clustered p95 |",
             "|---|---|---:|---:|---:|---:|---:|---:|:---:|---:|:---:|"]
    for name, rep in summary["datasets"].items():
        for fam, s in rep.items():
            lines.append("| %s | %s | %d | %d | %.3f | %.3f | %.3f | %.1f | %s | %.3f | %s |" % (
                name, fam, s["n_rows"], s["n_clusters"], s["ece"], s["ece_floor_mean"], s["ece_floor_p95"],
                s["ece_ratio"], "*" if s["ece_above_floor"] else "", s["ece_floor_p95_clustered"],
                "*" if s["ece_above_floor_clustered"] else ""))
    return "\n".join(lines)


def main(argv=None):
    import argparse
    import gzip

    ap = argparse.ArgumentParser(description="ECE against its calibrated noise floor, from committed robustness "
                                             "predictions (jsonl or jsonl.gz).")
    ap.add_argument("predictions")
    ap.add_argument("--n-sim", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    opener = gzip.open if a.predictions.endswith(".gz") else open
    with opener(a.predictions, "rt") as f:
        preds = [json.loads(line) for line in f if line.strip()]
    s = summarize_floor(preds, a.n_sim, a.seed)
    print(format_floor_table(s))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(s, f, indent=1)


if __name__ == "__main__":
    main()
