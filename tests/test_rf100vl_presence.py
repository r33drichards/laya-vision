"""benchmarks/rf100vl_presence.py: the metrics on hand-checked cases and a summary over rows shaped like evidence."""
import gzip
import importlib.util
import json
import os

import pytest

_spec = importlib.util.spec_from_file_location(
    "rf100vl_presence", os.path.join(os.path.dirname(__file__), "..", "benchmarks", "rf100vl_presence.py"))
P = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P)


def test_average_precision():
    assert P.average_precision([0.9, 0.8, 0.7], [1, 0, 1]) == pytest.approx((1 + 2 / 3) / 2)
    assert P.average_precision([0.1, 0.9], [1, 0]) == pytest.approx(0.5)
    assert P.average_precision([0.5, 0.5], [0, 0]) is None


def test_auroc():
    assert P.auroc([0.9, 0.1], [1, 0]) == 1.0
    assert P.auroc([0.1, 0.9], [1, 0]) == 0.0
    assert P.auroc([0.5, 0.5, 0.5], [1, 0, 1]) == 0.5  # ties count half
    assert P.auroc([0.8, 0.6, 0.7, 0.2], [1, 1, 0, 0]) == pytest.approx(0.75)
    assert P.auroc([0.3], [1]) is None


def test_summarize(tmp_path):
    def row(ds, dom, img, cls, p, y):
        return {"dataset": "rf100vl_" + dom, "id": "rf100vl-%s-%s-c%d" % (ds, img, cls), "qtype": "noul",
                "label": y, "probs_calibrated": [1 - p, p]}
    rows = [row("x-ray-id", "lab_imaging", "78_1", 0, 0.9, 1), row("x-ray-id", "lab_imaging", "78_1", 1, 0.2, 0),
            row("x-ray-id", "lab_imaging", "78_2", 0, 0.4, 0), row("x-ray-id", "lab_imaging", "78_2", 1, 0.7, 1),
            row("2024-frc", "industrial", "87_0", 0, 0.6, 0), row("2024-frc", "industrial", "87_1", 0, 0.3, 1),
            {"dataset": "aokvqa", "id": "q", "qtype": "choice", "label": 0, "probs_calibrated": [1, 0]}]
    path = tmp_path / "r.predictions.jsonl.gz"
    with gzip.open(path, "wt") as f:
        f.write("\n".join(json.dumps(r) for r in rows) + "\n")
    res = P.summarize(str(path))
    xr, frc = res["datasets"]["x-ray-id"], res["datasets"]["2024-frc"]
    assert xr["domain"] == "lab_imaging" and xr["images"] == 2 and xr["questions"] == 4
    assert xr["presence_ap"] == 1.0 and xr["auroc"] == 1.0 and xr["acc"] == 1.0 and xr["chance_ap"] == 0.5
    assert frc["presence_ap"] == 0.5 and frc["auroc"] == 0.0 and frc["balanced_acc"] == 0.0
    assert res["macro"]["datasets"] == 2 and res["macro"]["presence_ap"] == 0.75
    assert set(res["domains"]) == {"lab_imaging", "industrial"}
    assert "**all (macro)**" in P.markdown(res)
