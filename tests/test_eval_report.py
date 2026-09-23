"""``scripts/eval_report.py``: merging part result files and the Markdown it renders, on hand-built results."""
import importlib.util
import json
import os

import pytest

_spec = importlib.util.spec_from_file_location(
    "eval_report", os.path.join(os.path.dirname(__file__), "..", "scripts", "eval_report.py"))
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)

CODE = {"commit": "76d7b42abcdef", "branch": "claude/x", "dirty": False}


def _m(n, acc, ece=0.05, nll=0.7, **kw):
    return dict(n=n, acc=acc, ece=ece, nll=nll, **kw)


def _datasets():
    cal = {"aokvqa": _m(100, 0.6), "cauldron_ai2d": _m(200, 0.7), "cauldron_raven": _m(200, 0.3),
           "score_ava": _m(50, 0.4, n_score=50, mae=0.8, xent=1.3, prior_xent=1.4),
           "eval_cifar10h": _m(80, 0.9, n_soft=80, soft_xent=0.5, prior_soft_xent=2.2), "all": _m(630, 0.6)}
    return {"val_raw": cal, "val_calibrated": cal, "temperature": [1.1, 0.9, 1.0], "dataset_meta": {}}


def _games():
    return {"atari": [{"game": "Galaxian", "model_score": 500.0, "random_score": 300.0, "expert_score": None,
                       "normalized": None, "actions": {"LEFTFIRE": 3, "FIRE": 1}}],
            "doom": {"expert": {"policy": "expert", "mean_reward": 75.8, "kill_rate": 1.0, "mean_steps": 20.0}},
            "maze": [{"policy": "model:m", "size": 4, "solve_rate": 0.5, "efficiency": 0.8, "mean_steps": 30.0}],
            "snake": [{"policy": "expert", "size": 10, "mean_eaten": 26.3, "max_eaten": 40, "mean_steps": 232.0,
                       "ends": {"self": 9, "wall": 1}}]}


def test_group_of():
    assert [R.group_of(n) for n in ("aokvqa", "cauldron_ai2d", "cauldronfull_ai2d", "score_ava", "eval_pope_random")] == \
        ["vqa", "cauldron", "cauldronfull", "score", "eval"]



def test_datasets_heading_names_the_gpu_when_recorded():
    assert R.datasets_section(dict(_datasets(), gpu="NVIDIA L4"), "val")[0] == "#### Datasets (val split, calibrated, NVIDIA L4)"
    assert R.datasets_section(_datasets(), "val")[0] == "#### Datasets (val split, calibrated)"  # older results

def test_merge_takes_each_part_from_its_file():
    parts = [{"model": "m", "code": CODE, "datasets": _datasets(), "games": None, "latency": None, "val_split": "test"},
             {"model": "m", "code": CODE, "datasets": None, "games": _games(), "latency": None,
              "errors": [{"what": "atari game", "error": "boom"}]},
             {"model": "m", "code": CODE, "datasets": None, "games": None, "latency": {"median_ms": 41.0, "p90_ms": 45.0}}]
    r = R.merge(parts)
    assert r["datasets"] and r["games"] and r["latency"]["median_ms"] == 41.0 and r["val_split"] == "test"
    assert r["errors"] == [{"what": "atari game", "error": "boom"}]
    with pytest.raises(ValueError):
        R.merge([parts[0], dict(parts[1], model="other")])


def test_render_starts_with_the_marker_and_has_every_section():
    r = R.merge([{"model": "run/best", "code": CODE, "datasets": _datasets(), "games": _games(),
                  "latency": {"median_ms": 41.0, "p90_ms": 45.0}}])
    md = R.render(r, {"datasets": "success", "games": "success", "latency": "success"}, "https://run")
    assert md.startswith(R.marker("run/best") + "\n")
    assert "`claude/x@76d7b42a`" in md and "[workflow run](https://run)" in md and "⚠️" not in md
    assert "| cauldron | 2 | 400 | 50.0% | 0.050 |" in md  # group means over its sets
    assert "| eval_cifar10h | 80 | 90.0% | 0.050 | 0.700 | 0.500 (2.200) |" in md
    assert "0.80 levels off" in md and "median **41.0 ms**" in md
    assert "| Galaxian | 500.0 | 300.0 | – | – | LEFTFIRE 75%, FIRE 25% |" in md
    assert "| model:m | 4 | 50.0% | 0.80 | 30.0 |" in md and "self 9, wall 1" in md


def test_render_flags_failed_and_empty_parts():
    r = R.merge([{"model": "m", "code": CODE, "datasets": None, "games": None, "latency": {"median_ms": 1.0, "p90_ms": 2.0}}])
    md = R.render(r, {"datasets": "failure", "games": "success", "latency": "success"}, "https://run")
    assert "⚠️ **datasets**: failure" in md and "⚠️ **games**: no results" in md and "**latency**" not in md.split("####")[0]
    md = R.render({"model": "m"}, {"datasets": "failure", "games": "skipped"})
    assert "⚠️ **datasets**: failure" in md and "games" not in md


def test_cli_writes_the_report(tmp_path, capsys):
    f = tmp_path / "latency.json"
    f.write_text(json.dumps({"model": "m", "code": CODE, "latency": {"median_ms": 40.0, "p90_ms": 44.0}}))
    assert R.main([str(f), "--status", "latency=success,games=skipped"]) == 0
    out = capsys.readouterr().out
    assert out.startswith(R.marker("m")) and "median **40.0 ms**" in out
    assert R.main(["--model", "m", "--status", "datasets=failure"]) == 0
    assert "⚠️ **datasets**: failure" in capsys.readouterr().out


def test_html_report_is_self_contained_and_explains_the_run():
    r = R.merge([{"model": "run/best", "code": CODE, "started": "2026-09-23T01:40:00+00:00", "datasets": _datasets(),
                  "games": _games(), "latency": {"median_ms": 41.0, "p90_ms": 45.0}}])
    page = R.render_html(r, {"datasets": "success", "games": "success", "latency": "success"}, "https://run")
    assert page.startswith("<title>run/best</title>") and page.isascii()
    assert "<script" not in page and "<svg" in page and "What happened" in page
    assert "prefers-color-scheme:dark" in page and ':root[data-theme="dark"]' in page
    fs = R.findings(r)
    assert any("630 questions" in t for _, t in fs)  # pooled accuracy line
    assert any("Galaxian: scores 500.0 against 300.0 for random play" in t for _, t in fs)
    assert any(t.startswith("Against human vote spreads it beats") for _, t in fs)
    assert "Maze: solves 4&times;4: 50.0%" in page


def test_markdown_doc_is_deterministic_with_mermaid_charts():
    r = R.merge([{"model": "run/best", "code": CODE, "started": "2026-09-23T01:40:00+00:00", "datasets": _datasets(),
                  "games": _games(), "latency": {"median_ms": 41.0, "p90_ms": 45.0}}])
    doc = R.render_doc(r, "Run scorecard", ["../../eval-results/a.json"])
    assert doc == R.render_doc(r, "Run scorecard", ["../../eval-results/a.json"])
    assert doc.startswith("# Run scorecard\n") and "[`a.json`](../../eval-results/a.json)" in doc
    charts = doc.split("```mermaid\n")[1:]
    assert charts and all(c.startswith(R.MERMAID_INIT + "\nxychart-beta horizontal\n") for c in charts)
    assert '    x-axis ["iconqa' not in doc  # the fixture has no iconqa; labels come from the results only
    assert '"ai2d", "raven"' in doc.replace('"aokvqa"', "")  # cauldron group sorted by accuracy, descending
    assert "| cifar10h | 0.500 | 2.200 | -1.700 better |" in doc
    assert "- **weak** ·" not in doc or "- **good** ·" in doc
    assert "&times;" not in doc and "×" in doc
