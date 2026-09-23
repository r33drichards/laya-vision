"""Verify that published headline numbers are backed by committed row-level evidence.

    python benchmarks/verify_published.py            # from the repository root; exits 1 on any mismatch
    python benchmarks/verify_published.py --root DIR # another tree with the same layout (the tests use a fixture)

Checks, in order:

1. ``results/raw/SHA256SUMS`` (``sha256sum -c`` format): every listed file exists with that hash, and every file
   under ``results/raw/`` except ``SHA256SUMS`` and ``README.md`` is listed.
2. For each claim in ``results/claims.json``, the evidence rows (``*.predictions.jsonl[.gz]``, written by
   ``modal run modal_app.py::evidence``) are self-consistent: ``probs_calibrated`` is ``softmax(logits / T[qtype])``
   with the temperatures in the evidence's ``*.meta.json``, those temperatures are the checkpoint's (the metrics
   JSON's ``temperature``), and the scored weights are the claimed ones (``weights_sha256``, when the claim has it).
3. Per dataset, accuracy and calibrated ECE (max-probability confidence, 15 equal bins, ``laya.common.ece_score``)
   are recomputed from the rows and compared with
   * the metrics the evidence job itself computed (``meta.metrics.val_calibrated``, tolerance 1e-4: the rows are
     rounded to 6 decimals),
   * the committed metrics JSON (``docs/*-metrics.json`` at ``metrics_path``): ``n`` exactly, ``acc`` within
     ``tolerance.acc`` and ``ece`` within ``tolerance.ece`` (absolute). The evidence is a re-run on possibly another
     GPU type under bf16 autocast, so a few near-tied rows may flip; the defaults (0.005 accuracy, 0.01 ECE) are
     set in ``claims.json`` before looking at a re-run and not tuned to it,
   * the README table cell (a percentage with one decimal): ``|100 * acc - cell| <= 0.05 + 100 * tolerance.acc``.

Only the standard library and numpy are needed; nothing is imported from ``laya``.
"""
import argparse
import gzip
import hashlib
import json
import os
import sys

import numpy as np


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def check_sums(raw_dir):
    """Problems with ``raw_dir/SHA256SUMS`` (empty list: all good) and the number of files it covers."""
    problems, listed = [], set()
    path = os.path.join(raw_dir, "SHA256SUMS")
    if not os.path.exists(path):
        return ["%s is missing" % path], 0
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            digest, name = line.rstrip("\n").split(None, 1)
            name = name.lstrip("*").strip()
            listed.add(name)
            full = os.path.join(raw_dir, name)
            if not os.path.exists(full):
                problems.append("SHA256SUMS lists %s, which does not exist" % name)
            elif sha256(full) != digest:
                problems.append("sha256 mismatch for %s" % name)
    for root, _, files in os.walk(raw_dir):
        for fn in files:
            rel = os.path.relpath(os.path.join(root, fn), raw_dir)
            if rel not in ("SHA256SUMS", "README.md") and rel not in listed:
                problems.append("%s is not listed in SHA256SUMS" % rel)
    return problems, len(listed)


def read_rows(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        return [json.loads(line) for line in f if line.strip()]


QTYPES = {"choice": 0, "score": 1, "noul": 2}


def softmax(z):
    z = np.asarray(z, dtype=np.float64)
    e = np.exp(z - z.max())
    return e / e.sum()


def ece(conf, correct, bins=15):
    """Expected calibration error, exactly ``laya.common.ece_score``."""
    conf, correct = np.asarray(conf, dtype=np.float64), np.asarray(correct, dtype=np.float64)
    if len(conf) == 0:
        return float("nan")
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (conf > lo) & (conf <= hi)
        if sel.any():
            e += sel.mean() * abs(conf[sel].mean() - correct[sel].mean())
    return float(e)


def recompute(rows):
    """``{dataset: {"n", "acc", "ece"}}`` from the rows' calibrated probabilities."""
    groups = {}
    for r in rows:
        p = r["probs_calibrated"]
        top = max(range(len(p)), key=p.__getitem__)
        groups.setdefault(r["dataset"], []).append((max(p), float(top == r["label"])))
    out = {}
    for name, g in groups.items():
        a = np.array(g)
        out[name] = {"n": len(g), "acc": float(a[:, 1].mean()), "ece": ece(a[:, 0], a[:, 1])}
    return out


def readme_cells(path, row_contains, columns):
    """``{dataset: percent}`` from the Markdown table row containing ``row_contains``; ``columns`` maps datasets to
    header names."""
    with open(path) as f:
        lines = [ln.strip() for ln in f]
    header = None
    for ln in lines:
        if not ln.startswith("|"):
            header = None
            continue
        cells = [c.strip() for c in ln.strip("|").split("|")]
        if header is None:
            header = cells
            continue
        if row_contains in ln:
            out = {}
            for ds, col in columns.items():
                cell = cells[header.index(col)]
                out[ds] = float(cell.rstrip("%").strip())
            return out
    raise KeyError("no table row containing %r in %s" % (row_contains, path))


def dig(obj, path):
    for k in path:
        obj = obj[k]
    return obj


def verify_claim(root, claim, tol):
    problems, checks = [], 0
    rows = read_rows(os.path.join(root, claim["evidence"]))
    with open(os.path.join(root, claim["meta"])) as f:
        meta = json.load(f)
    with open(os.path.join(root, claim["metrics_json"])) as f:
        metrics = json.load(f)
    temps = meta["temperature"]
    if "temperature" in metrics:
        checks += 1
        if any(abs(a - b) > 1e-6 for a, b in zip(temps, metrics["temperature"])):
            problems.append("%s: evidence temperatures %s != checkpoint's %s"
                            % (claim["meta"], temps, metrics["temperature"]))
    if claim.get("weights_sha256"):
        checks += 1
        if meta.get("weights_sha256") != claim["weights_sha256"]:
            problems.append("%s: scored weights %s, the claim is about %s"
                            % (claim["meta"], meta.get("weights_sha256"), claim["weights_sha256"]))
    if len(rows) != meta["n_rows"]:
        problems.append("%s: %d rows, meta says %d" % (claim["evidence"], len(rows), meta["n_rows"]))
    bad = 0
    for r in rows:
        p = softmax(np.asarray(r["logits"]) / temps[QTYPES[r["qtype"]]])
        if np.abs(p - np.asarray(r["probs_calibrated"])).max() > 1e-4 or r["option_order"] != list(range(len(p))):
            bad += 1
    checks += 1
    if bad:
        problems.append("%s: %d rows whose probs_calibrated are not softmax(logits / T)" % (claim["evidence"], bad))
    got = recompute(rows)
    published = dig(metrics, claim["metrics_path"])
    job = meta.get("metrics", {}).get("val_calibrated", {})
    readme = readme_cells(os.path.join(root, claim["readme"]["file"]), claim["readme"]["row_contains"],
                          claim["readme"]["columns"]) if claim.get("readme") else {}
    report = {}
    for ds in claim["datasets"]:
        if ds not in got:
            problems.append("%s: no rows for dataset %s" % (claim["evidence"], ds))
            continue
        g, pub = got[ds], published[ds]
        report[ds] = {"n": g["n"], "acc": round(g["acc"], 6), "ece": round(g["ece"], 6),
                      "published_acc": pub["acc"], "published_ece": pub["ece"]}
        if ds in job:
            checks += 2
            for key in ("acc", "ece"):
                if abs(g[key] - job[ds][key]) > 1e-4:
                    problems.append("%s %s: rows give %s=%.6f, the evidence job reported %.6f"
                                    % (claim["evidence"], ds, key, g[key], job[ds][key]))
        checks += 3
        if g["n"] != pub["n"]:
            problems.append("%s: %d evidence rows, %s has n=%d" % (ds, g["n"], claim["metrics_json"], pub["n"]))
        for key in ("acc", "ece"):
            if not abs(g[key] - pub[key]) <= tol[key]:
                problems.append("%s %s: recomputed %.4f vs %.4f in %s (tolerance %.4f)"
                                % (ds, key, g[key], pub[key], claim["metrics_json"], tol[key]))
        if ds in readme:
            checks += 1
            report[ds]["readme_acc_pct"] = readme[ds]
            if not abs(100 * g["acc"] - readme[ds]) <= 0.05 + 100 * tol["acc"] + 1e-9:
                problems.append("%s acc: recomputed %.2f%% vs %.1f%% in %s (tolerance %.2f points)"
                                % (ds, 100 * g["acc"], readme[ds], claim["readme"]["file"], 0.05 + 100 * tol["acc"]))
    return problems, checks, report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    args = ap.parse_args(argv)
    raw_dir = os.path.join(args.root, "results", "raw")
    problems, n_files = check_sums(raw_dir)
    checks = n_files
    with open(os.path.join(args.root, "results", "claims.json")) as f:
        spec = json.load(f)
    reports = {}
    for claim in spec["claims"]:
        tol = dict(spec["tolerance"], **claim.get("tolerance", {}))
        p, c, report = verify_claim(args.root, claim, tol)
        problems += p
        checks += c
        reports[claim["evidence"]] = report
    print(json.dumps({"status": "mismatch" if problems else "ok", "checks": checks, "problems": problems,
                      "recomputed": reports}, indent=2))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
