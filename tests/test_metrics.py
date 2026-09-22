"""``metrics_from``'s ordinal metrics for ``score`` records: no model, records are built by hand."""
import math

import pytest
import torch

from laya.common import QTYPES
from laya.vlm_train import format_metrics, metrics_from


def rec(logits, target, qtype, dataset, label=None):
    t = torch.tensor(target, dtype=torch.float32)
    return {"logits": torch.tensor(logits, dtype=torch.float32), "target": t, "qtype": QTYPES[qtype], "dataset": dataset,
            "label": int(t.argmax()) if label is None else label}


def test_choice_groups_have_no_ordinal_metrics():
    m = metrics_from([rec([2.0, 0.0, 0.0], [1, 0, 0], "choice", "c"), rec([0.0, 2.0], [1, 0], "noul", "n")])
    assert set(m) == {"all", "c", "n"} and "mae" not in m["all"] and "mae" not in m["c"]
    assert m["c"]["acc"] == 1.0 and m["n"]["acc"] == 0.0
    assert "mae" not in format_metrics(m)


def test_score_mae_and_soft_cross_entropy():
    # a confident, correct prediction: mae 0 (up to the softmax tail), xent = nll
    r = rec([10.0, 0.0, 0.0, 0.0, 0.0], [1, 0, 0, 0, 0], "score", "s")
    m = metrics_from([r])
    assert m["s"]["n_score"] == 1 and m["s"]["mae"] == pytest.approx(0.0, abs=1e-3)
    assert m["s"]["xent"] == pytest.approx(m["s"]["nll"], abs=1e-6)
    # uniform prediction over 5 levels against a one-hot label at level 4: E_p = 2, mae 2, xent = ln 5
    m = metrics_from([rec([0.0] * 5, [0, 0, 0, 0, 1], "score", "s")])
    assert m["s"]["mae"] == pytest.approx(2.0) and m["s"]["xent"] == pytest.approx(math.log(5))
    # a soft target: a model that reproduces the vote histogram exactly has mae 0 and xent = the target's entropy,
    # while argmax accuracy and nll still see a "label"
    hist = [0.1, 0.2, 0.4, 0.2, 0.1]
    m = metrics_from([rec([math.log(h) for h in hist], hist, "score", "ava")])
    assert m["ava"]["mae"] == pytest.approx(0.0, abs=1e-6)
    assert m["ava"]["xent"] == pytest.approx(-sum(h * math.log(h) for h in hist), abs=1e-6)
    assert m["ava"]["acc"] == 1.0 and m["ava"]["nll"] == pytest.approx(-math.log(0.4), abs=1e-6)
    # temperature scaling applies to the score type's temperature, and 'all' aggregates the score rows only
    both = [rec([0.0] * 5, [0, 0, 0, 0, 1], "score", "s"), rec([2.0, 0.0, 0.0], [1, 0, 0], "choice", "c")]
    m = metrics_from(both, temperatures=(1.0, 2.0, 1.0))
    assert m["all"]["n"] == 2 and m["all"]["n_score"] == 1 and m["all"]["mae"] == pytest.approx(2.0)
    assert "mae=" in format_metrics(m) and "xent=" in format_metrics(m)
