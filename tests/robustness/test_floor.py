"""``laya.robustness.floor``: the calibrated-ECE noise floor on synthetic prediction rows (no model)."""
import numpy as np

from laya import robustness as R
from laya.robustness import floor as F


def synth(n, conf_fn, acc_fn, seed, dataset="toy", family="orig"):
    """``n`` two-option prediction rows whose max probability is ``conf_fn(rng)`` and which are right with
    probability ``acc_fn(conf)``."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        c = float(conf_fn(rng))
        label = int(rng.integers(2))
        right = rng.random() < acc_fn(c)
        pred = label if right else 1 - label
        probs = [0.0, 0.0]
        probs[pred], probs[1 - pred] = c, 1 - c
        out.append({"id": "%s/%d|%s" % (dataset, i, family), "group_id": "%s/%d" % (dataset, i),
                    "cluster": "%s/%d" % (dataset, i), "dataset": dataset, "family": family, "variant": family,
                    "label": label, "pred": pred, "probs": probs, "k": 2, "qtype": 1})
    return out


def uniform_conf(rng):
    return rng.uniform(0.5, 1.0)


def test_deterministic_and_seeded():
    conf = np.random.default_rng(1).uniform(0.5, 1.0, 300)
    a, b = F.ece_noise_floor(conf, seed=3), F.ece_noise_floor(conf, seed=3)
    assert a == b and a["n_sim"] == 200 and a["n"] == 300
    assert F.ece_noise_floor(conf, seed=4)["mean"] != a["mean"]
    assert 0 < a["mean"] <= a["p95"] < 1
    preds = synth(200, uniform_conf, lambda c: c, 0)
    assert F.summarize_floor(preds, n_sim=50, seed=0) == F.summarize_floor(preds, n_sim=50, seed=0)
    # a cell's floor does not depend on which other cells exist
    other = synth(100, uniform_conf, lambda c: c, 1, dataset="zzz")
    assert (F.summarize_floor(preds + other, n_sim=50)["datasets"]["toy"]
            == F.summarize_floor(preds, n_sim=50)["datasets"]["toy"])


def test_empty():
    fl = F.ece_noise_floor([])
    assert np.isnan(fl["mean"]) and fl["n"] == 0


def test_calibrated_is_at_floor():
    """Rows that are right with probability exactly their confidence: ECE / floor near 1, and (over several
    seeded sets) rarely above p95."""
    ratios, above = [], []
    for s in range(10):
        st = F.floor_stats(synth(400, uniform_conf, lambda c: c, s), n_sim=200, seed=s)
        ratios.append(st["ece_ratio"])
        above.append(st["ece_above_floor"])
    assert 0.6 < np.mean(ratios) < 1.4
    assert sum(above) <= 3


def test_overconfident_is_flagged():
    """Confidences ~0.9 on rows right ~60% of the time are far above the floor."""
    st = F.floor_stats(synth(300, lambda r: r.uniform(0.85, 0.95), lambda c: 0.6, 0), n_sim=200)
    assert st["ece_above_floor"] and st["ece_ratio"] > 3
    assert st["ece"] > 0.2


def test_floor_shrinks_with_n():
    rng = np.random.default_rng(0)
    floors = [F.ece_noise_floor(rng.uniform(0.5, 1.0, n), seed=0)["mean"] for n in (50, 200, 800, 3200)]
    assert all(a > b for a, b in zip(floors, floors[1:]))
    assert floors[0] > 2 * floors[-1]


def test_clustered_floor_is_higher():
    """Eight variants per source row with near-identical confidences: sharing a draw per cluster leaves ~n/8
    independent samples, so the clustered floor is well above the independent one."""
    rng = np.random.default_rng(0)
    base = rng.uniform(0.5, 0.98, 100)
    conf = np.repeat(base, 8) + rng.uniform(-0.01, 0.01, 800)
    groups = np.repeat(np.arange(100), 8)
    ind = F.ece_noise_floor(conf, seed=0)
    cl = F.ece_noise_floor(conf, seed=0, groups=groups)
    assert cl["mean"] > 1.5 * ind["mean"]
    # one row per group: identical to the independent floor
    assert F.ece_noise_floor(base, seed=0, groups=range(100)) == F.ece_noise_floor(base, seed=0)


def test_matches_robustness_summary():
    """``ece`` per dataset x family is exactly the one ``robustness.summarize`` reports."""
    # the text rows share group ids with the orig rows (the ids differ by family), as real variants do
    preds = synth(120, uniform_conf, lambda c: c, 0) + synth(120, uniform_conf, lambda c: 0.7, 1, family="text")
    s = R.summarize(preds, n_boot=0)
    f = F.summarize_floor(preds, n_sim=20)
    for fam in ("orig", "text"):
        assert f["datasets"]["toy"][fam]["ece"] == s["datasets"]["toy"][fam]["ece"]
    assert f["macro"]["orig"]["n_datasets"] == 1
    table = F.format_floor_table(f)
    assert table.count("\n") == 3 and "| toy | text |" in table


def test_ece_floor_fields_per_dataset():
    """The keys ``metrics_from`` stores next to a set's ECE: the independent floor, seeded by the set's name only."""
    conf = np.random.default_rng(2).uniform(0.5, 1.0, 282)
    f = F.ece_floor_fields(conf, "cauldron_ai2d")
    assert set(f) == {"ece_floor", "ece_floor_p95"} and 0 < f["ece_floor"] <= f["ece_floor_p95"] < 1
    assert f == F.ece_floor_fields(conf, "cauldron_ai2d")  # deterministic
    full = F.ece_noise_floor(conf, seed=R._seed_for("ece_floor", 0, "cauldron_ai2d"))
    assert f == {"ece_floor": full["mean"], "ece_floor_p95": full["p95"]}
    assert F.ece_floor_fields(conf, "aokvqa")["ece_floor"] != f["ece_floor"]  # own stream per set
    assert F.ece_floor_fields([], "empty") == {}
