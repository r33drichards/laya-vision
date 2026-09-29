"""The autoresearch BiGym track: ``pareto.decide_bigym`` (keep / discard for ``--profile bigym``), the fixed BiGym
benchmark's scoring and seeds (``autoresearch/bigym_eval.py``), and, with the simulator, its dense progress."""
import importlib.util
import json
import os
import re
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
AR = os.path.join(HERE, "..", "autoresearch")
sys.path.insert(0, AR)


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(AR, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


P = _load("pareto")
BE = _load("bigym_eval")

try:
    import bigym  # noqa: F401
    import mujoco  # noqa: F401

    HAVE_SIM = bool(os.environ.get("MUJOCO_GL"))
except Exception:
    HAVE_SIM = False
sim = pytest.mark.skipif(not HAVE_SIM, reason="needs mujoco and bigym (and MUJOCO_GL=egl or osmesa)")


def row(bigym, quality=0.68, games=0.30, params=201.2, commit="c", status="keep"):
    return {"commit": commit, "bigym": bigym, "bigym_success": 0.0, "quality": quality, "macro_acc": quality,
            "ece_hard": 0.0, "games": games, "params_m": params, "latency_x": 0.85, "status": status,
            "profile": "bigym", "experiment": "autoresearch/experiment_bigym.py", "task_success": "0 0 0 0 0 0",
            "description": ""}


# -- keep / discard ------------------------------------------------------------------------------------------------

def test_first_bigym_result_is_the_baseline():
    assert P.decide_bigym([], row(0.0))[0] == "keep"
    # a crashed first run is not a baseline
    assert P.decide_bigym([row(0.0, status="crash")], row(-0.1))[0] == "keep"


def test_bigym_must_beat_the_best_kept_by_the_margin():
    base = row(0.10, commit="base")
    assert P.decide_bigym([base], row(0.10 + P.BIGYM_MARGIN))[0] == "discard"   # not strictly above
    assert P.decide_bigym([base], row(0.10 + P.BIGYM_MARGIN + 1e-3))[0] == "keep"
    better = row(0.30, commit="better")
    rows = [base, better, row(0.9, status="discard")]  # a discarded result never sets the bar
    assert P.decide_bigym(rows, row(0.32))[0] == "discard"   # beats the baseline, not the best kept
    assert P.decide_bigym(rows, row(0.34))[0] == "keep"


def test_guard_rails_are_against_the_baseline():
    base = row(0.10, quality=0.68, games=0.30, commit="base")
    good = 0.10 + P.BIGYM_MARGIN + 0.05
    assert P.decide_bigym([base], row(good, quality=0.68 - P.QUALITY_GUARD, games=0.30 - P.GAMES_GUARD))[0] == "keep"
    status, why = P.decide_bigym([base], row(good, quality=0.68 - P.QUALITY_GUARD - 1e-3))
    assert status == "discard" and "quality" in why
    status, why = P.decide_bigym([base], row(good, games=0.30 - P.GAMES_GUARD - 1e-3))
    assert status == "discard" and "games" in why
    status, why = P.decide_bigym([base], row(good, params=201.2 * 1.05))
    assert status == "discard" and "params" in why
    # the guards stay at the baseline's level even after a later keep with better quality
    later = row(0.5, quality=0.75, games=0.5, commit="later")
    assert P.decide_bigym([base, later], row(0.6, quality=0.68, games=0.30))[0] == "keep"


def test_bigym_tsv_round_trip_and_cli(tmp_path, capsys):
    tsv = str(tmp_path / "results.tsv")
    res = {"summary": {"bigym": 0.05, "quality": 0.68, "macro_acc": 0.70, "ece_hard": 0.02, "games": 0.3,
                       "params_m": 201.2, "latency_x": 0.85},
           "bigym": {"success_mean": 0.03, "tasks": list(BE.TASKS), "success": {t: 0.03 for t in BE.TASKS}},
           "profile": "bigym", "experiment": "autoresearch/experiment_bigym.py", "commit": "aaaaaaa"}
    path = tmp_path / "a.json"
    path.write_text(json.dumps(res))
    assert P.main(["add", str(path), "--tsv", tsv]) == 0
    assert "status: keep" in capsys.readouterr().out
    res["summary"]["bigym"] = 0.06
    path.write_text(json.dumps(dict(res, commit="bbbbbbb")))
    P.main(["add", str(path), "--tsv", tsv])
    assert "status: discard" in capsys.readouterr().out
    P.main(["add", "crash", "--tsv", tsv, "--commit", "ccccccc"])
    rows = P.read_tsv(tsv)
    assert [r["status"] for r in rows] == ["keep", "discard", "crash"]
    assert P.tsv_profile(tsv) == "bigym" and rows[0]["bigym"] == pytest.approx(0.05)
    assert rows[0]["experiment"] == "autoresearch/experiment_bigym.py"
    assert P.prunable_bigym(rows) == ["bbbbbbb", "ccccccc"]
    P.main(["show", "--tsv", tsv])
    out = capsys.readouterr().out
    assert "bigym profile" in out and "aaaaaaa" in out and "success" in out and "quality" in out
    with pytest.raises(ValueError):  # a tag keeps one profile
        P.append_tsv(tsv, dict(row(0.1), profile="default"), "default")


def test_default_profile_tsv_is_unchanged(tmp_path):
    tsv = str(tmp_path / "results.tsv")
    P.append_tsv(tsv, {"commit": "a", "quality": 0.7, "macro_acc": 0.7, "ece_hard": 0.0, "games": 0.1,
                       "params_m": 256.0, "latency_x": 1.0, "status": "keep", "description": "x"})
    assert open(tsv).readline().rstrip("\n").split("\t") == P.COLUMNS
    assert P.tsv_profile(tsv) == "default"


# -- the benchmark's fixed settings and scoring ---------------------------------------------------------------------

def test_eval_seeds_are_off_every_training_and_games_range():
    games = _load("games_eval")
    all_seeds = [s for t in BE.TASKS for s in BE.task_seeds(t)]
    assert len(set(all_seeds)) == len(all_seeds) == BE.EPISODES * len(BE.TASKS)
    assert all(BE.is_eval_seed(s) and s >= BE.TRAIN_SEED_MAX for s in all_seeds)
    game_seeds = {s for spec in games.SUITE.values() for s in range(spec.seed, spec.seed + spec.episodes)}
    assert not game_seeds & set(all_seeds)
    assert max(game_seeds) < min(all_seeds)
    assert not BE.is_eval_seed(BE.TRAIN_SEED_MAX - 1) and not BE.is_eval_seed(300_000)
    for t in BE.TASKS:  # the chunks partition the task's episodes
        assert sum((BE.chunk_seeds(t, c) for c in range(BE.CHUNKS[t])), []) == BE.task_seeds(t)


def test_caps_match_the_task_table():
    from laya.bigymgames import TASKS

    assert set(BE.TASKS) <= set(TASKS)
    assert {t: TASKS[t]["max_decisions"] for t in BE.TASKS} == BE.CAPS


def test_bigym_pins_match_modal_bigym():
    src = open(os.path.join(HERE, "..", "modal_bigym.py")).read()
    har = open(os.path.join(AR, "harness.py")).read()
    for name in ("BIGYM", "MUJOCO"):
        pat = r'^%s = "([^"]+)"' % name
        assert re.search(pat, src, re.M).group(1) == re.search(pat, har, re.M).group(1)


def _fake(task, progress, success=None, chunk=0):
    success = success or [False] * len(progress)
    seeds = BE.task_seeds(task)
    return {"task": task, "chunk": chunk, "episodes": [{"seed": s, "progress": p, "success": ok, "decisions": 1}
                                                       for s, p, ok in zip(seeds, progress, success)],
            "actions": {"STAY": len(progress)}, "seconds": 1.0}


def _bases(rnd=0.1, exp=1.0):
    return {t: {"random": rnd, "expert": exp, "episodes": BE.EPISODES,
                "seeds": [BE.task_seeds(t)[0], BE.task_seeds(t)[-1]], "cap": BE.CAPS[t]} for t in BE.TASKS}


def test_summarize_normalizes_clips_and_averages():
    n = BE.EPISODES
    res = {t: _fake(t, [0.1] * n) for t in BE.TASKS}  # random level everywhere
    res["ReachTarget"] = _fake("ReachTarget", [1.0] * n, [True] * n)  # expert level
    s = BE.summarize(res, _bases())
    assert s["complete"] and s["per_task"]["ReachTarget"] == pytest.approx(1.0)
    assert s["per_task"]["DrawerTopOpen"] == pytest.approx(0.0)
    assert s["bigym"] == pytest.approx(1.0 / len(BE.TASKS))
    assert s["success"]["ReachTarget"] == 1.0 and s["top_actions"]["ReachTarget"][0][0] == "STAY"
    assert BE.normalize(-5.0, 0.1, 1.0) == BE.CLIP_LO and BE.normalize(0.1, 0.1, 0.1) is None
    # a missing task, or a task short of episodes, leaves the benchmark incomplete
    part = dict(res)
    part["WallCupboardOpen"] = _fake("WallCupboardOpen", [0.5] * (n // 2))
    assert not BE.summarize(part, _bases())["complete"]


def test_chunks_merge_in_seed_order():
    t, n = "DrawerTopOpen", BE.EPISODES
    a, b = _fake(t, [0.2] * n), _fake(t, [0.4] * n, chunk=1)
    a["episodes"], b["episodes"] = a["episodes"][n // 2:], b["episodes"][:n // 2]
    m = BE.merge_chunks([a, b])
    assert [e["seed"] for e in m["episodes"]] == BE.task_seeds(t) and m["actions"] == {"STAY": 2 * n}


def test_stale_baselines_are_refused():
    b = _bases()
    b["ReachTarget"]["cap"] = 10
    with pytest.raises(ValueError):
        BE.check_baseline("ReachTarget", b["ReachTarget"])
    with pytest.raises(KeyError):
        BE.check_baseline("ReachTarget", None)


def test_committed_baselines_match_the_eval():
    if not os.path.exists(BE.BASELINES_PATH):
        pytest.skip("bigym_baselines.json not measured yet")
    base = BE.load_baselines()
    for t in BE.TASKS:
        e = BE.check_baseline(t, base.get(t))
        assert 0.0 <= e["random"] < e["expert"] <= 1.0


# -- with the simulator ------------------------------------------------------------------------------------------

@sim
def test_progress_is_dense_and_one_on_success():
    from laya import bigymgames as bg

    g = bg.BiGymGame("ReachTarget", 1, env=bg.make_env("ReachTarget", cameras=False))
    st = BE.start_state(g)
    assert BE.progress(g, st) == pytest.approx(0.0, abs=1e-9)
    for _ in range(12):
        if g.done:
            break
        g.step(bg.oracle_action(g))
    assert BE.progress(g, st) > 0.3
    g.close()
    g = bg.BiGymGame("DrawerTopClose", 1, env=bg.make_env("DrawerTopClose", cameras=False))
    st = BE.start_state(g)
    assert st.shape == (1,) and BE.progress(g, st) == pytest.approx(0.0, abs=1e-6)
    g.set_open_fraction(0.5 * float(st[0]))
    assert BE.progress(g, st) == pytest.approx(0.5, abs=1e-6)
    g.set_open_fraction(0.0)
    assert g.success and BE.progress(g, st) == 1.0
    g.close()


@sim
def test_lockstep_play_matches_one_at_a_time():
    """Batching does not change an episode: random play on two seeds together equals each seed alone."""
    seeds = BE.task_seeds("ReachTargetSingle")[:2]
    both = BE.play("ReachTargetSingle", seeds, BE.random_policy(seeds), cap=8, cameras=False)
    one = BE.play("ReachTargetSingle", seeds[1:], BE.random_policy(seeds[1:]), cap=8, cameras=False)
    assert both["episodes"][1] == one["episodes"][0]
    assert sum(both["actions"].values()) == sum(e["decisions"] for e in both["episodes"])


@sim
def test_model_policy_prefixes_match_vlm_prefix():
    """The cached per-frame preprocessing gives exactly ``vlm_prefix``'s pixels and ids (checked inside), and the
    policy answers through a stub forward in primitive names."""
    transformers = pytest.importorskip("transformers")
    from laya import bigymgames as bg
    from laya.preprocess import ImagePrep

    try:
        proc = transformers.AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct")
    except Exception as e:  # offline without a cached processor
        pytest.skip("SmolVLM processor unavailable: %r" % e)
    prep = ImagePrep(backend="processor")
    prep.apply(proc)

    class Agent:
        processor, cfg = proc, {"head_max_len": 256, "max_len": 1024}

    Agent.prep = prep
    seen = []

    def forward(agent, items, k):
        seen.extend(items)
        out = np.zeros((len(items), k))
        out[:, list(bg.PRIMITIVES).index("LEFT_HAND_UP")] = 1.0
        return out

    pol = BE.model_policy(Agent(), "ReachTarget", 4, forward=forward)
    seeds = BE.task_seeds("ReachTarget")[:2]
    res = BE.play("ReachTarget", seeds, pol, cap=3)
    pol.close()
    assert set(res["actions"]) == {"LEFT_HAND_UP"} and pol.forwards == 6
    assert all(it["n_images"] == 4 and it["pixel_values"].shape[0] == 4 for it in seen)
