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


def test_soft_choice_and_noul_cross_entropy_with_prior():
    # human vote shares on a choice set: soft_xent is the cross-entropy against them, prior_soft_xent that of the
    # set's mean histogram; hard-label records in the same group do not count toward either
    votes = [[0.9, 0.1, 0.0], [0.2, 0.8, 0.0]]
    recs = [rec([math.log(0.9), math.log(0.1), -30.0], votes[0], "choice", "h"),
            rec([math.log(0.2), math.log(0.8), -30.0], votes[1], "choice", "h"),
            rec([2.0, 0.0, 0.0], [1, 0, 0], "choice", "h")]
    m = metrics_from(recs)
    ent = [-sum(v * math.log(v) for v in t if v) for t in votes]
    assert m["h"]["n"] == 3 and m["h"]["n_soft"] == 2
    assert m["h"]["soft_xent"] == pytest.approx(sum(ent) / 2, abs=1e-5)
    mean = [0.55, 0.45]
    prior = sum(-sum(v * math.log(p) for v, p in zip(t, mean)) for t in votes) / 2
    assert m["h"]["prior_soft_xent"] == pytest.approx(prior, abs=1e-5) and m["h"]["prior_soft_xent"] > m["h"]["soft_xent"]
    assert "mae" not in m["h"] and "xent" not in m["h"]
    assert "soft_xent=" in format_metrics(m) and "(prior " in format_metrics(m)
    # a yes/no set with only hard labels gets neither
    m = metrics_from([rec([0.0, 2.0], [0, 1], "noul", "pope")])
    assert "soft_xent" not in m["pope"] and "prior_soft_xent" not in m["pope"]


def test_score_prior_is_the_mean_histogram():
    t1, t2 = [0.5, 0.5, 0.0], [0.0, 0.5, 0.5]
    m = metrics_from([rec([0.0] * 3, t1, "score", "q"), rec([0.0] * 3, t2, "score", "q")])
    mean = [0.25, 0.5, 0.25]
    prior = sum(-sum(v * math.log(p) for v, p in zip(t, mean) if v) for t in (t1, t2)) / 2
    assert m["q"]["prior_xent"] == pytest.approx(prior, abs=1e-5) and m["q"]["xent"] == pytest.approx(math.log(3), abs=1e-5)


def test_ece_floor_is_opt_in_and_uses_the_reported_confidences():
    from laya.common import ece_score
    from laya.robustness_floor import ece_floor_fields

    g = torch.Generator().manual_seed(0)
    recs = [rec((torch.randn(3, generator=g) * 2).tolist(), [1, 0, 0], "choice", "c") for _ in range(60)]
    recs += [rec((torch.randn(5, generator=g) * 2).tolist(), [0, 1, 3, 1, 0], "score", "s") for _ in range(40)]
    assert "ece_floor" not in metrics_from(recs)["c"]  # off by default: training loops call this every eval
    m = metrics_from(recs, temperatures=(1.5, 0.7, 1.0), ece_floor_sims=50)
    for name, rs in (("c", recs[:60]), ("s", recs[60:]), ("all", recs)):
        conf = [float(torch.softmax(r["logits"] / (1.5 if r["qtype"] == QTYPES["choice"] else 0.7), -1).max()) for r in rs]
        right = [float(int(torch.softmax(r["logits"], -1).argmax()) == r["label"]) for r in rs]
        assert m[name]["ece"] == pytest.approx(ece_score(torch.tensor(conf).numpy(), torch.tensor(right).numpy()))
        assert {k: m[name][k] for k in ("ece_floor", "ece_floor_p95")} == pytest.approx(ece_floor_fields(conf, name, n_sim=50))
