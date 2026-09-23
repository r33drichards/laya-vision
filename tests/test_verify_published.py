"""CPU test for benchmarks/verify_published.py on a tiny synthetic tree (no model, no network)."""
import gzip
import hashlib
import json
import os
import random
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "benchmarks"))
import verify_published  # noqa: E402

from laya.common import ece_score  # noqa: E402

TEMPS = [2.0, 1.5, 1.2]


def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


def make_tree(root, readme_acc=None):
    """Two datasets of synthetic rows, their evidence files, SHA256SUMS, a metrics JSON, a README table, claims."""
    rng = random.Random(0)
    rows = []
    for ds, n, k, qtype in (("toy_a", 40, 4, "choice"), ("toy_b", 30, 2, "noul")):
        for i in range(n):
            z = np.array([rng.gauss(0, 2) for _ in range(k)])
            p = softmax(z / TEMPS[verify_published.QTYPES[qtype]])
            rows.append({"dataset": ds, "id": "%s-%d" % (ds, i), "index": i, "qtype": qtype, "label": rng.randrange(k),
                         "option_order": list(range(k)), "logits": [round(float(v), 6) for v in z],
                         "probs_calibrated": [round(float(v), 6) for v in p], "input_ids_sha256": "0" * 64})
    metrics = {}
    for ds in ("toy_a", "toy_b"):
        sel = [r for r in rows if r["dataset"] == ds]
        conf = np.array([max(r["probs_calibrated"]) for r in sel])
        correct = np.array([float(int(np.argmax(r["probs_calibrated"])) == r["label"]) for r in sel])
        metrics[ds] = {"n": len(sel), "acc": float(correct.mean()), "ece": ece_score(conf, correct)}
    raw = root / "results" / "raw"
    raw.mkdir(parents=True)
    with gzip.GzipFile(raw / "toy.predictions.jsonl.gz", "wb", mtime=0) as f:
        f.write("".join(json.dumps(r) + "\n" for r in rows).encode())
    (raw / "toy.meta.json").write_text(json.dumps({"temperature": TEMPS, "n_rows": len(rows),
                                                   "metrics": {"val_calibrated": metrics}}))
    (raw / "README.md").write_text("notes\n")
    write_sums(raw)
    (root / "docs").mkdir()
    (root / "docs" / "toy-metrics.json").write_text(json.dumps({"temperature": TEMPS, "final": {"val_calibrated": metrics}}))
    acc = readme_acc or {ds: round(100 * m["acc"], 1) for ds, m in metrics.items()}
    (root / "README.md").write_text(
        "# toy\n\n| Checkpoint | Toy A | Toy B |\n|---|---|---|\n| [me/toy](https://x) | %.1f%% | %.1f%% |\n"
        % (acc["toy_a"], acc["toy_b"]))
    claims = {"tolerance": {"acc": 0.005, "ece": 0.01}, "claims": [{
        "evidence": "results/raw/toy.predictions.jsonl.gz", "meta": "results/raw/toy.meta.json",
        "metrics_json": "docs/toy-metrics.json", "metrics_path": ["final", "val_calibrated"],
        "datasets": ["toy_a", "toy_b"],
        "readme": {"file": "README.md", "row_contains": "[me/toy]", "columns": {"toy_a": "Toy A", "toy_b": "Toy B"}}}]}
    (root / "results" / "claims.json").write_text(json.dumps(claims))
    return metrics


def write_sums(raw):
    names = sorted(p.name for p in raw.iterdir() if p.name not in ("SHA256SUMS", "README.md"))
    (raw / "SHA256SUMS").write_text("".join("%s  %s\n" % (hashlib.sha256((raw / n).read_bytes()).hexdigest(), n)
                                            for n in names))


def test_passes_on_consistent_evidence(tmp_path, capsys):
    make_tree(tmp_path)
    assert verify_published.main(["--root", str(tmp_path)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "ok" and out["checks"] > 10


def test_ece_matches_laya():
    rng = np.random.default_rng(0)
    conf, correct = rng.uniform(0.3, 1.0, 200), rng.integers(0, 2, 200).astype(float)
    assert verify_published.ece(conf, correct) == pytest.approx(ece_score(conf, correct), abs=1e-12)


def test_fails_on_a_tampered_row(tmp_path):
    make_tree(tmp_path)
    path = tmp_path / "results" / "raw" / "toy.predictions.jsonl.gz"
    rows = verify_published.read_rows(str(path))
    rows[0]["label"] = (rows[0]["label"] + 1) % len(rows[0]["logits"])
    with gzip.GzipFile(path, "wb", mtime=0) as f:
        f.write("".join(json.dumps(r) + "\n" for r in rows).encode())
    assert verify_published.main(["--root", str(tmp_path)]) == 1  # checksum no longer matches
    write_sums(tmp_path / "results" / "raw")
    assert verify_published.main(["--root", str(tmp_path)]) == 1  # the job's own metrics no longer match the rows


def test_fails_on_an_unlisted_file(tmp_path):
    make_tree(tmp_path)
    (tmp_path / "results" / "raw" / "extra.jsonl").write_text("{}\n")
    assert verify_published.main(["--root", str(tmp_path)]) == 1


def test_fails_when_the_readme_disagrees(tmp_path, capsys):
    metrics = make_tree(tmp_path)
    make_tree(tmp_path / "bad", readme_acc={"toy_a": 100 * metrics["toy_a"]["acc"] + 2.0,
                                            "toy_b": 100 * metrics["toy_b"]["acc"]})
    assert verify_published.main(["--root", str(tmp_path / "bad")]) == 1
    out = json.loads(capsys.readouterr().out)
    assert any("toy_a acc" in p and "README.md" in p for p in out["problems"])


def test_fails_when_probs_are_not_the_calibrated_logits(tmp_path):
    make_tree(tmp_path)
    meta = tmp_path / "results" / "raw" / "toy.meta.json"
    m = json.loads(meta.read_text())
    m["temperature"] = [1.0, 1.0, 1.0]
    meta.write_text(json.dumps(m))
    write_sums(tmp_path / "results" / "raw")
    assert verify_published.main(["--root", str(tmp_path)]) == 1
