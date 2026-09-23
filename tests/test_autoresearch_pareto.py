"""``autoresearch/pareto.py``: the keep/discard rule and hypervolume over four objectives, on hand-made results."""
import importlib.util
import itertools
import json
import os
import random

import pytest

_spec = importlib.util.spec_from_file_location(
    "pareto", os.path.join(os.path.dirname(__file__), "..", "autoresearch", "pareto.py"))
P = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P)


def pt(q, params, lat, games=0.0, commit="c", status="keep"):
    return {"commit": commit, "quality": q, "macro_acc": q, "ece_hard": 0.0, "games": games, "params_m": params,
            "latency_x": lat, "status": status, "description": ""}


def test_first_result_is_kept_and_noise_is_discarded():
    base = pt(0.70, 256.0, 1.0, 0.1, "base")
    assert P.decide([], base) == ("keep", [])
    # within every margin: nothing new
    assert P.decide([base], pt(0.703, 255.0, 0.99, 0.12))[0] == "discard"
    # clearly better quality at the same cost: kept, and it strictly dominates the baseline
    status, beaten = P.decide([base], pt(0.72, 256.0, 1.0, 0.1))
    assert status == "keep" and [b["commit"] for b in beaten] == ["base"]


def test_trade_offs_extend_the_frontier():
    base = pt(0.70, 256.0, 1.0, 0.1, "base")
    small = pt(0.66, 160.0, 0.7, 0.1, "small")       # worse quality, but much smaller and faster
    gamer = pt(0.68, 256.0, 1.0, 0.6, "gamer")       # worse quality, much better at games
    assert P.decide([base], small) == ("keep", []) and P.decide([base], gamer) == ("keep", [])
    front = P.frontier([base, small, gamer])
    assert {r["commit"] for r in front} == {"base", "small", "gamer"}
    # worse on everything than one frontier point: discarded
    assert P.decide(front, pt(0.65, 170.0, 0.75, 0.05))[0] == "discard"
    # a small games gain within the margin is not enough on its own
    assert P.decide(front, pt(0.70, 256.0, 1.0, 0.12))[0] == "discard"


def test_hypervolume_grows_only_when_the_frontier_moves_out():
    base = pt(0.70, 256.0, 1.0, 0.1, "base")
    hv0 = P.hypervolume([base], base)
    assert hv0 == pytest.approx(0.70 * (0.1 - P.GAMES_FLOOR) * 0.5 * 0.5)
    small = pt(0.66, 160.0, 0.7, 0.1, "small")
    hv1 = P.hypervolume([base, small], base)
    assert hv1 > hv0
    # adding a dominated point changes nothing
    assert P.hypervolume([base, small, pt(0.6, 200.0, 0.9, 0.0)], base) == pytest.approx(hv1)
    # points beyond the reference box contribute nothing
    assert P.hypervolume([base, pt(0.99, 500.0, 1.0, 0.9)], base) == pytest.approx(hv0)


def test_hypervolume_matches_brute_force():
    """Exact slicing against inclusion-exclusion over boxes, on random 4-D point sets."""
    rng = random.Random(0)
    base = pt(0.7, 200.0, 1.0, 0.0)
    for _ in range(20):
        pts = [pt(rng.uniform(0.3, 0.9), rng.uniform(100, 290), rng.uniform(0.5, 1.4), rng.uniform(-0.4, 1.0))
               for _ in range(rng.randint(1, 5))]
        norm = [P._normalized(p, base) for p in pts]
        ref = P._ref()
        brute = 0.0
        for k in range(1, len(norm) + 1):
            for combo in itertools.combinations(norm, k):
                corner = [max(c[d] for c in combo) for d in range(4)]
                vol = 1.0
                for d in range(4):
                    vol *= max(0.0, ref[d] - corner[d])
                brute += (-1) ** (k + 1) * vol
        assert P.hypervolume(pts, base) == pytest.approx(brute, abs=1e-9)


def test_cli_appends_rows_and_decides(tmp_path, capsys):
    tsv = str(tmp_path / "results.tsv")

    def result(q, params, lat, games):
        f = tmp_path / ("r%d.json" % len(list(tmp_path.iterdir())))
        f.write_text(json.dumps({"summary": {"quality": q, "macro_acc": q + 0.05, "ece_hard": 0.05, "games": games,
                                             "params_m": params, "latency_x": lat}}))
        return str(f)
    assert P.main(["add", result(0.70, 256.0, 1.0, 0.1), "--tsv", tsv, "--commit", "aaa", "--desc", "baseline"]) == 0
    assert "status: keep" in capsys.readouterr().out
    P.main(["add", result(0.70, 256.0, 1.0, 0.1), "--tsv", tsv, "--commit", "bbb", "--desc", "same again"])
    assert "status: discard" in capsys.readouterr().out
    P.main(["add", "crash", "--tsv", tsv, "--commit", "ccc", "--desc", "oom"])
    P.main(["add", result(0.64, 150.0, 0.6, 0.1), "--tsv", tsv, "--commit", "ddd", "--desc", "drop 10 layers"])
    out = capsys.readouterr().out
    assert "status: keep" in out and "hypervolume:" in out
    rows = P.read_tsv(tsv)
    assert [r["status"] for r in rows] == ["keep", "discard", "crash", "keep"]
    assert open(tsv).readline().rstrip("\n").split("\t") == P.COLUMNS
    P.main(["show", "--tsv", tsv])
    shown = capsys.readouterr().out
    assert "frontier (2)" in shown and "aaa" in shown and "ddd" in shown and "games" in shown
