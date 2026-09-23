"""``autoresearch/pareto.py``: the keep/discard rule and hypervolume, on hand-made results."""
import importlib.util
import json
import os

import pytest

_spec = importlib.util.spec_from_file_location(
    "pareto", os.path.join(os.path.dirname(__file__), "..", "autoresearch", "pareto.py"))
P = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P)


def pt(q, params, lat, commit="c", status="keep"):
    return {"commit": commit, "quality": q, "macro_acc": q, "ece_hard": 0.0, "params_m": params, "latency_x": lat,
            "status": status, "description": ""}


def test_first_result_is_kept_and_noise_is_discarded():
    base = pt(0.70, 256.0, 80.0, "base")
    assert P.decide([], base) == ("keep", [])
    # within every margin: nothing new
    assert P.decide([base], pt(0.703, 255.0, 79.0))[0] == "discard"
    # clearly better quality at the same cost: kept, and it strictly dominates the baseline
    status, beaten = P.decide([base], pt(0.72, 256.0, 80.0))
    assert status == "keep" and [b["commit"] for b in beaten] == ["base"]


def test_trade_offs_extend_the_frontier():
    base = pt(0.70, 256.0, 80.0, "base")
    small = pt(0.66, 160.0, 55.0, "small")       # worse, but much smaller and faster
    assert P.decide([base], small) == ("keep", [])
    front = P.frontier([base, small])
    assert {r["commit"] for r in front} == {"base", "small"}
    # worse on everything than one frontier point: discarded
    assert P.decide(front, pt(0.65, 170.0, 60.0))[0] == "discard"
    # a result between them that neither beats: kept
    assert P.decide(front, pt(0.69, 200.0, 65.0))[0] == "keep"


def test_hypervolume_grows_only_when_the_frontier_moves_out():
    base = pt(0.70, 256.0, 80.0, "base")
    hv0 = P.hypervolume([base], base)
    assert hv0 == pytest.approx(0.70 * 0.5 * 0.5)
    small = pt(0.66, 160.0, 55.0, "small")
    hv1 = P.hypervolume([base, small], base)
    assert hv1 > hv0
    # adding a dominated point changes nothing
    assert P.hypervolume([base, small, pt(0.6, 200.0, 70.0)], base) == pytest.approx(hv1)
    # points beyond the reference box contribute nothing
    assert P.hypervolume([base, pt(0.99, 500.0, 80.0)], base) == pytest.approx(hv0)


def test_cli_appends_rows_and_decides(tmp_path, capsys):
    tsv = str(tmp_path / "results.tsv")
    def result(q, params, lat):
        f = tmp_path / ("r%d.json" % len(list(tmp_path.iterdir())))
        f.write_text(json.dumps({"summary": {"quality": q, "macro_acc": q + 0.05, "ece_hard": 0.05,
                                             "params_m": params, "latency_x": lat}}))
        return str(f)
    assert P.main(["add", result(0.70, 256.0, 80.0), "--tsv", tsv, "--commit", "aaa", "--desc", "baseline"]) == 0
    assert "status: keep" in capsys.readouterr().out
    P.main(["add", result(0.70, 256.0, 80.0), "--tsv", tsv, "--commit", "bbb", "--desc", "same again"])
    assert "status: discard" in capsys.readouterr().out
    P.main(["add", "crash", "--tsv", tsv, "--commit", "ccc", "--desc", "oom"])
    P.main(["add", result(0.64, 150.0, 50.0), "--tsv", tsv, "--commit", "ddd", "--desc", "drop 10 layers"])
    out = capsys.readouterr().out
    assert "status: keep" in out and "hypervolume:" in out
    rows = P.read_tsv(tsv)
    assert [r["status"] for r in rows] == ["keep", "discard", "crash", "keep"]
    assert open(tsv).readline().rstrip("\n").split("\t") == P.COLUMNS
    P.main(["show", "--tsv", tsv])
    shown = capsys.readouterr().out
    assert "frontier (2)" in shown and "aaa" in shown and "ddd" in shown
