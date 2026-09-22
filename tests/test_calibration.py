"""``laya.calibration`` on synthetic logits: no model, CPU only."""
import json
import math
import warnings

import numpy as np
import pytest

from laya.calibration import (Calibration, bootstrap_draws, calibrate_records, fit_rule, fit_temperature, group_folds,
                              log_probs, pad_logits, resolve_temperature)
from laya.common import QTYPES


def synthetic(n=400, k=4, true_t=2.5, scale=4.0, seed=0, qtype="choice", group_size=3):
    """Labels drawn from softmax(z / true_t) while the model reports softmax(z): overconfident by ``true_t``."""
    rng = np.random.default_rng(seed)
    recs = []
    for i in range(n):
        z = rng.normal(0, scale, k)
        p = np.exp(z / true_t - (z / true_t).max())
        recs.append({"logits": z, "label": int(rng.choice(k, p=p / p.sum())), "qtype": QTYPES[qtype],
                     "group": "%s-%d" % (qtype, i // group_size), "checkpoint_t": 1.0})
    return recs


def arrays(recs):
    return (pad_logits([r["logits"] for r in recs]), np.array([r["label"] for r in recs]),
            np.array([r["qtype"] for r in recs]))


def test_overconfident_logits_recover_t_above_one():
    Z, y, _ = arrays(synthetic(n=3000, true_t=2.5))
    assert fit_temperature(Z, y) == pytest.approx(2.5, rel=0.1)
    Z, y, _ = arrays(synthetic(n=3000, true_t=0.5, scale=1.0, seed=1))
    assert fit_temperature(Z, y) == pytest.approx(0.5, rel=0.15)


def test_argmax_unchanged_and_calibration_helps():
    recs = synthetic(n=600)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cal = calibrate_records(recs, folds=5, bootstrap=200)
    ev = cal.evidence["all"]
    assert ev["accuracy_unchanged"] and cal.temperatures["choice"] > 1.5 and cal.sources["choice"] == "per_type"
    assert ev["ece"]["calibrated_oof"]["value"] < ev["ece"]["raw"]["value"]
    assert ev["nll"]["calibrated_oof"]["value"] < ev["nll"]["raw"]["value"]
    lo, hi = ev["ece_improvement"]["raw-calibrated_oof"]["ci95"]
    assert lo > 0 and lo <= ev["ece_improvement"]["raw-calibrated_oof"]["value"] <= hi
    Z, _, _ = arrays(recs)
    for t in (0.1, 0.7, 3.0, 19.0):
        assert np.array_equal(log_probs(Z, t).argmax(1), Z.argmax(1))


def test_group_disjoint_folds():
    groups = ["img%d" % (i // 4) for i in range(203)] + [("__row__", i) for i in range(7)]
    fold = group_folds(groups, 5, seed=3)
    by_group = {}
    for g, f in zip(groups, fold):
        assert by_group.setdefault(g, f) == f  # every row of a group in one fold
    assert set(fold) == set(range(5))
    sizes = np.bincount(list({g: f for g, f in zip(groups, fold)}.values()))
    assert sizes.max() - sizes.min() <= 1  # groups dealt evenly
    assert np.array_equal(fold, group_folds(groups, 5, seed=3))


def test_bootstrap_resamples_whole_groups():
    groups = ["a"] * 3 + ["b"] * 1 + ["c"] * 5
    for d in bootstrap_draws(groups, 50, seed=1):
        counts = {g: 0 for g in "abc"}
        for i in d:
            counts[groups[i]] += 1
        assert counts["a"] % 3 == 0 and counts["c"] % 5 == 0


def test_per_type_pooled_and_fallback_rules():
    recs = synthetic(n=100, qtype="choice") + synthetic(n=10, k=2, qtype="noul", seed=2)
    Z, y, q = arrays(recs)
    fit = fit_rule(Z, y, q, per_type=True, min_rows=30)
    assert fit["sources"] == {"choice": "per_type", "noul": "pooled"} and fit["fitted_on"] == {"choice": 100, "noul": 110}
    fit = fit_rule(Z, y, q, per_type=False, min_rows=30)
    assert set(fit["sources"].values()) == {"pooled"} and fit["temperatures"]["choice"] == fit["temperatures"]["noul"]
    assert fit_rule(Z[:20], y[:20], q[:20], min_rows=30)["temperatures"] == {}
    small = synthetic(n=12)
    for r in small:
        r["checkpoint_t"] = 1.7
    with pytest.warns(UserWarning, match="nothing fitted"):
        cal = calibrate_records(small, folds=3, bootstrap=20)
    assert cal.temperatures == {}
    # nothing fitted: the out-of-fold variant is the checkpoint's temperature, so identical ECE
    assert cal.evidence["all"]["ece"]["calibrated_oof"]["value"] == cal.evidence["all"]["ece"]["checkpoint"]["value"]


def test_mixed_types_report_per_type_and_save_load(tmp_path):
    recs = synthetic(n=120) + synthetic(n=90, k=2, qtype="noul", true_t=1.5, seed=4) \
        + synthetic(n=60, k=5, qtype="score", seed=5)
    cal = calibrate_records(recs, bootstrap=50, checkpoint={"model": "m", "config_sha256": "c", "weights_sha256": "w"})
    assert set(cal.evidence) == {"all", "choice", "noul", "score"} and len(cal.fold_temperatures) == 5
    path = str(tmp_path / "cal.json")
    cal.save(path)
    json.load(open(path))
    back = Calibration.load(path)
    assert back == cal
    assert back.check({"model": "x", "config_sha256": "c", "weights_sha256": "w"})
    with pytest.warns(UserWarning, match="different checkpoint"):
        assert not back.check({"model": "m", "config_sha256": "c", "weights_sha256": "other"})
    with pytest.raises(ValueError, match="different checkpoint"):
        back.check({"model": "m", "config_sha256": "c", "weights_sha256": "other"}, strict=True)
    assert "choice=" in back.summary()


def test_resolve_temperature():
    assert resolve_temperature() == {}
    assert resolve_temperature(2) == {0: 2.0, 1: 2.0, 2: 2.0}
    assert resolve_temperature({"noul": 0.5}) == {QTYPES["noul"]: 0.5}
    for bad in (0, -1.0, math.inf, math.nan):
        with pytest.raises(ValueError):
            resolve_temperature(bad)
    with pytest.raises(ValueError):
        resolve_temperature({"choice": 0.0})
    with pytest.raises(ValueError):
        resolve_temperature({"yesno": 1.0})
    with pytest.raises(TypeError):
        resolve_temperature(True)
    cal = calibrate_records(synthetic(n=60), bootstrap=0)
    assert resolve_temperature(calibration=cal) == {0: cal.temperatures["choice"]}
    with pytest.raises(ValueError, match="not both"):
        resolve_temperature(1.0, cal)
