"""Modal jobs for the game-only Atari model: the plain SmolVLM backbone with a fresh head, trained only on Atari
frames (no photo-VQA data), then scored by playing ALE games.

    modal run modal_atari_train.py::datasets                       # which (source, game) pairs are ready
    modal run modal_atari_train.py::smoke                          # end to end on a tiny synthetic dataset
    modal run --detach modal_atari_train.py::train_atari --run-name atari-v1 [--sources expert,atari_head,jat]
    modal run --detach modal_atari_train.py::train_atari --run-name atari8-2f --sources expert2f --frames 2 \
        --games Breakout,Pong --init-from atari-expert-v1/best
    modal run --detach modal_atari_train.py::train_atari --run-name atari-ab-base --sources expert2f \
        --games Breakout,Pong --backbone HuggingFaceTB/SmolVLM-256M-Base   # A/B against the Instruct default
    modal run --detach modal_atari_train.py::ab_backbone                # both arms of that A/B, then play them
    modal run modal_atari_train.py::atari_eval --model atari-v1/best [--games Breakout,Pong] [--episodes 3] [--sample]
    modal run modal_atari_train.py::renormalize --results play.json [--out play_renorm.json]

Volumes (created out of band; never ``modal deploy`` this app):
    laya-datasets     -> /data       (read-only; /data/atari/<source>/<Game>/, see docs/atari-data-format.md)
    laya-hf-cache     -> /cache/hf   (HF_HOME)
    laya-checkpoints  -> /ckpt       (this app writes only under /ckpt/smolvlm/atari-*)
"""
import json
import os
import statistics
import time
from typing import Dict

import modal

app = modal.App("laya-atari")

hf_vol = modal.Volume.from_name("laya-hf-cache")
data_vol = modal.Volume.from_name("laya-datasets")
ckpt_vol = modal.Volume.from_name("laya-checkpoints")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.14.0",
        "torchvision==0.29.0",
        "transformers==5.17.0",
        "safetensors",
        "huggingface_hub",
        "numpy",
        "pillow",
        "num2words",
        "ale-py",
        "gymnasium",
    )
    .env({"HF_HOME": "/cache/hf", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_python_source("laya")
)

BACKBONE = "HuggingFaceTB/SmolVLM-256M-Instruct"
CKPT_ROOT = "/ckpt/smolvlm"
ATARI_ROOT = "/data/atari"
SYNTH_ROOT = "/tmp/atari_synth"


def _split(s: str):
    return [x for x in s.split(",") if x]


@app.function(image=image, timeout=10 * 60, volumes={"/data": data_vol.read_only()})
def list_ready(root: str = ATARI_ROOT):
    """Ready (source, game) pairs with their frame format and record counts from meta.json."""
    from laya.atari_train import ready_datasets

    data_vol.reload()
    return [{"name": d["name"], "frame_format": d["frame_format"], "train": (d["meta"].get("train") or {}).get("records"),
             "val": (d["meta"].get("val") or {}).get("records"), "expert_score": d["meta"].get("expert_score"),
             "random_score": d["meta"].get("random_score")} for d in ready_datasets(root)]


@app.local_entrypoint()
def datasets():
    rows = list_ready.remote()
    for r in rows:
        print("%-30s %-12s train %6s val %5s  expert %s random %s" % (r["name"], r["frame_format"], r["train"], r["val"],
                                                                     r["expert_score"], r["random_score"]))
    print("%d ready datasets" % len(rows))


@app.function(
    image=image,
    gpu="A100",
    cpu=24,
    memory=65536,
    timeout=150 * 60,
    retries=modal.Retries(max_retries=3, initial_delay=10.0),
    volumes={"/cache/hf": hf_vol, "/data": data_vol.read_only(), "/ckpt": ckpt_vol},
)
def train_atari(
    run_name: str = "atari-v1",
    sources: str = "expert,atari_head,jat",
    games: str = "",
    passes: float = 1.5,
    max_minutes: float = 85.0,
    batch_size: int = 32,
    lr_head: float = 1e-4,
    lr_backbone: float = 2e-5,
    lr_ref_batch: int = 16,
    warmup_frac: float = 0.03,
    n_evals: int = 6,
    val_per_ds: int = 300,
    final_val_per_ds: int = 1000,
    train_eval_per_game: int = 100,
    n_calib: int = 60,
    max_passes: float = 4.0,
    max_train_per_ds: int = 0,
    num_workers: int = 22,
    synthetic: bool = False,
    frames: int = 2,
    init_from: str = "",
    restart: bool = False,
    state_every_min: float = 10.0,
    crash_at_step: int = 0,
    select_by: str = "nll",
    play_check_games: str = "",
    play_check_episodes: int = 3,
    play_check_steps: int = 400,
    group_size: int = 4,
    sigma: float = 0.3,
    sigma_end: float = 0.0,
    w_sph: float = 0.75,
    w_ce: float = 1.0,
    w_ce_schedule: str = "const",
    train_act: bool = False,
    td_lambda: float = 0.0,
    top_episode_frac: float = 0.0,
    balance: str = "game",
    image_size: int = 0,
    preprocess: str = "",
    backbone: str = "",
):
    """Train SmolVLM (fresh head, vision tower frozen) on Atari frames only; save /ckpt/smolvlm/<run_name>/best.

    * Data: every ready ``/data/atari/<source>/<Game>`` for ``sources`` (and ``games`` if given). Games are sampled
      equally, the sources of a game are pooled. ``passes`` counts samples over the pooled train set;
      ``max_passes`` caps the passes over any one game (a capped game leaves the mix). ``max_minutes`` caps the
      training wall-clock (the LR schedule follows whichever of step or time progress is further).
    * ``n_calib`` frames per (source, game), from held-out train episodes, are kept out of training for fitting
      temperatures.
    * ``n_evals`` periodic evals on up to ``val_per_ds`` val frames per (source, game): per-game accuracy, ECE and
      NLL against the labels, plus ``train_eval_per_game`` seen train frames per game for the overfitting gap.
      ``best/`` is saved whenever the mean per-game val NLL improves (calibration matters more than accuracy).
    * The final model is the best one, with per-type and per-option-count temperatures fitted on the holdout,
      then scored on up to ``final_val_per_ds`` val frames per (source, game), raw and calibrated.
    * ``frames=2`` (the default; two frames beat one on every comparison so far) gives the model
      ``{"images": [prev_image, image]}`` from the ``expert2f`` layout instead of the single frame, repeating
      ``image`` for a record without a ``prev_image``; the value is saved in the checkpoint config, so
      ``play_atari`` matches it by default. ``frames=1`` is only for reproducing the old single-frame runs.
    * ``init_from`` (a run under /ckpt/smolvlm, e.g. ``atari-expert-v1/best``) continues from a trained checkpoint
      instead of a fresh head; question types absent from the calibration holdout keep its temperature.
    * Durability: every ``state_every_min`` minutes (clamped to ``vlm_train.MIN_STATE_MINUTES``, and backed off
      further when writes are slow) and after every eval the run writes ``<out>/state.pt``
      (weights, optimizer, step, RNG, per-game sample counts, elapsed training time, best-so-far and the log)
      atomically, and a restarted container resumes from it. ``max_minutes`` counts training time across
      attempts. ``restart`` ignores an existing state; ``crash_at_step`` raises once at that step to test the
      resume path (Modal retries the container).
    * Objective (defaults keep the old behaviour): ``group_size``, ``sigma`` annealed to ``sigma_end`` when > 0,
      ``w_sph``, and ``w_ce_schedule="anneal"`` to decay the cross-entropy weight to 0 (calibration);
      ``train_act`` trains the act/escalate head on its cost matrix.
    * Targets: ``td_lambda`` > 0 blends each target toward the action actually taken by the percentile of its
      return-to-go, and ``top_episode_frac`` keeps only that fraction of episodes per game by episode score.
    * ``select_by="play"`` keeps the checkpoint with the best short in-run play score instead of the best val
      NLL; both are logged either way, with the step each would pick.
    * ``balance="game_source"`` samples every (game, source) pair equally instead of pooling a game's sources, so
      mixing e.g. 20k ``expert2f`` with 5k ``dagger1`` frames per game gives a 50/50 mix by samples.
    * ``backbone`` (a Hub id, default ``BACKBONE``) is the pretrained SmolVLM a fresh run starts from, e.g.
      ``HuggingFaceTB/SmolVLM-256M-Base`` to A/B the pre-instruction-tuned checkpoint against the Instruct one.
      It must share the 256M architecture and processor settings (``laya.preprocess`` checks the latter). Ignored
      with ``init_from``, whose checkpoint records its own backbone.
    """
    import math

    import torch

    from safetensors.torch import load_file

    from laya.atari_train import (even_subsample, filter_top_episodes, fit_option_temperatures, load_atari,
                                  outcome_targets, per_game_metrics, play_check, scale_records, write_synthetic)
    from laya.vlm import VLMAgent
    from laya.vlm_train import collect_logits, fit_temperatures_from, train

    if not run_name.startswith("atari-"):
        raise SystemExit("run_name must start with 'atari-' (this app writes only /ckpt/smolvlm/atari-*)")
    t_start = time.time()
    print("GPU:", torch.cuda.get_device_name(0), "| torch", torch.__version__)
    out_dir = os.path.join(CKPT_ROOT, run_name)
    root = ATARI_ROOT
    if synthetic:
        root = SYNTH_ROOT
        print("synthetic data:", write_synthetic(root, n_train=96, n_val=24))
    else:
        data_vol.reload()
    data = load_atari(root, _split(sources), _split(games), n_calib=n_calib, val_limit=final_val_per_ds,
                      train_limit=max_train_per_ds or None, frames=frames)
    if not data["train"]:
        raise SystemExit("no ready Atari data for sources=%s games=%s" % (sources, games))
    if top_episode_frac:
        before = len(data["train"])
        data["train"] = filter_top_episodes(data["train"], top_episode_frac)
        print("top %.0f%% of episodes by score: %d -> %d train frames" % (100 * top_episode_frac, before, len(data["train"])))
    if td_lambda:
        n_blend = outcome_targets(data["train"], td_lambda)
        print("outcome-blended targets (strength %.2f) on %d of %d train frames" % (td_lambda, n_blend, len(data["train"])))
    names, train_ex = data["games"], data["train"]
    # ItemStream samples each group of its balance key equally; "dataset" carries the game for per-game metrics
    bkey = "game" if balance == "game" else "game_source"
    train_by_game = [dict(ex, dataset=ex["game"], game_source="%s|%s" % (ex["game"], ex["source"])) for ex in train_ex]
    val_small = []
    for ds in data["datasets"]:
        val_small += even_subsample([ex for ex in data["val"] if ex["dataset"] == ds["name"]], val_per_ds)
    train_eval = []
    for g in names:
        train_eval += [ex for ex in train_by_game if ex["game"] == g][:train_eval_per_game]
    log_mix = {}
    for ex in train_ex:
        log_mix[ex["source"]] = log_mix.get(ex["source"], 0) + 1

    scale = math.sqrt(batch_size / lr_ref_batch)
    lr_h, lr_b = lr_head * scale, lr_backbone * scale
    steps = int(math.ceil(passes * len(train_ex) / batch_size))
    eval_every = max(1, steps // max(1, n_evals))
    warmup = max(1, int(warmup_frac * steps))
    sizes: Dict[str, int] = {}
    for ex in train_by_game:
        sizes[ex[bkey]] = sizes.get(ex[bkey], 0) + 1
    per_game = passes * len(train_ex) / len(sizes)
    print("plan: %d steps x batch %d (%.2f passes of %d frames), warmup %d, eval every %d on %d val frames, "
          "lr head %.2e backbone %.2e, max %.0f min" % (steps, batch_size, passes, len(train_ex), warmup, eval_every,
                                                        len(val_small), lr_h, lr_b, max_minutes))
    print("balancing by %s; expected passes per group with equal sampling (before max_passes=%s): %s"
          % (bkey, max_passes or None, {g: round(per_game / sizes[g], 2) for g in sorted(sizes)}))

    # additive: ``image_size``/``preprocess`` override the input resolution and preprocessing path (see
    # laya.preprocess). Both are saved in the checkpoint config, so play-eval matches training automatically.
    prep_kw = {k: v for k, v in (("image_size", image_size), ("preprocess", preprocess)) if v}
    if init_from:
        agent = VLMAgent(os.path.join(CKPT_ROOT, init_from), device="cuda", **prep_kw)
        print("initialised from %s (temperatures %s)" % (init_from, [round(t, 3) for t in agent.temperature]))
    else:
        agent = VLMAgent(backbone=backbone or BACKBONE, device="cuda", **prep_kw)
        print("fresh head on backbone %s" % agent.cfg["backbone"])
    print("preprocessing: %r -> %d image tokens per frame" % (agent.prep, agent.prep.image_seq_len))
    init_temps = list(agent.temperature)
    agent.cfg["atari_frames"] = frames
    hf_vol.commit()
    model, proc = agent.model, agent.processor
    ev_kw = dict(batch_size=64, num_workers=num_workers)
    datasets_log = [{k: d[k] for k in ("name", "frame_format", "n_train", "n_calib", "n_val", "n_soft")} for d in data["datasets"]]
    log = {"run": run_name, "games": names, "datasets": datasets_log, "frames": frames, "source_frames": log_mix,
           "args": dict(sources=sources, games=games, frames=frames, init_from=init_from, passes=passes,
                        max_minutes=max_minutes, batch_size=batch_size, lr_head=lr_h, lr_backbone=lr_b,
                        warmup=warmup, steps=steps, eval_every=eval_every, max_passes=max_passes, n_calib=n_calib,
                        val_per_ds=val_per_ds, synthetic=synthetic, select_by=select_by, group_size=group_size,
                        sigma=sigma, sigma_end=sigma_end or None, w_sph=w_sph, w_ce=w_ce,
                        w_ce_schedule=w_ce_schedule, train_act=train_act, td_lambda=td_lambda,
                        top_episode_frac=top_episode_frac, balance=balance, backbone=agent.cfg["backbone"]),
           "evals": []}
    best = {"nll": math.inf, "play": -math.inf, "step": None, "state": None, "nll_step": None, "play_step": None}

    # ------------------------------------------------------------------ durability: resume from state.pt
    state_path = os.path.join(out_dir, "state.pt")
    restart_marker = os.path.join(out_dir, "restarted")
    call_id = modal.current_function_call_id() or str(t_start)
    marked = open(restart_marker).read().strip() if os.path.exists(restart_marker) else ""
    if restart and marked != call_id:
        # only this call's first attempt restarts: its retries must still resume from their own state
        for f_ in (state_path, os.path.join(out_dir, "crashed")):
            if os.path.exists(f_):
                os.remove(f_)
                print("--restart: removed %s" % os.path.basename(f_))
        os.makedirs(out_dir, exist_ok=True)
        with open(restart_marker, "w") as f_:
            f_.write(call_id)
        ckpt_vol.commit()
    resume = None
    if os.path.exists(state_path):
        blob = torch.load(state_path, map_location="cpu", weights_only=False)
        model.load_state_dict(blob["model"])
        model.to("cuda")
        resume, log, best = blob["train"], blob["log"], dict(best, **blob["best"])
        print("resuming %s from state.pt at step %d (%d evals so far, best nll %.4f)"
              % (run_name, resume["step"], len(log["evals"]), best["nll"]), flush=True)

    def save_state(step, tstate):
        ts = time.time()
        os.makedirs(out_dir, exist_ok=True)
        blob = {"train": tstate, "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "best": {k: best[k] for k in ("nll", "play", "step", "nll_step", "play_step")}, "log": log}
        tmp = state_path + ".tmp"
        torch.save(blob, tmp)
        os.replace(tmp, state_path)  # atomic: a torn write never replaces a good state
        ckpt_vol.commit()
        print("  wrote state.pt at step %d (%.1f s)" % (step, time.time() - ts), flush=True)

    play_games = _split(play_check_games) or names[:3]
    play_baselines = {g: expert_baseline(g) for g in play_games}

    def write_log():
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "metrics.json"), "w") as f:
            json.dump(log, f, indent=2)

    def eval_fn(step, keep=True):
        te = time.time()
        val_m = per_game_metrics(collect_logits(model, proc, val_small, **ev_kw))
        tr_m = per_game_metrics(collect_logits(model, proc, train_eval, **ev_kw))
        row = {"step": step, "passes": round(step * batch_size / len(train_ex), 3), "val": val_m, "train": tr_m}
        log["evals"].append(row)
        print("[eval step %d, %.2f passes] val mean per-game acc %.4f ece %.4f nll %.4f | train acc %.4f nll %.4f"
              % (step, row["passes"], val_m["mean"]["acc"], val_m["mean"]["ece"], val_m["mean"]["nll"],
                 tr_m["mean"]["acc"], tr_m["mean"]["nll"]), flush=True)
        print("  " + " | ".join("%s %.3f/%.2f" % (g, m["acc"], m["nll"]) for g, m in val_m["games"].items()), flush=True)
        pc = play_check(agent, play_games, play_baselines, play_check_episodes, play_check_steps, frames)
        row["play_check"] = pc
        print("  play check (%d ep x %d steps): mean normalised %.3f | %s"
              % (play_check_episodes, play_check_steps, pc["mean_normalized"],
                 ", ".join("%s %.0f (%s)" % (g, pc[g]["score"], "-" if pc[g]["normalized"] is None else "%.2f" % pc[g]["normalized"])
                           for g in play_games)), flush=True)
        if not keep:  # the reference eval before training must not become the best checkpoint
            write_log()
            print("  eval took %.1f min" % ((time.time() - te) / 60), flush=True)
            return
        improved = val_m["mean"]["nll"] < best["nll"]
        if improved:
            best.update(nll=val_m["mean"]["nll"], nll_step=step)
        if pc["mean_normalized"] > best["play"]:
            best.update(play=pc["mean_normalized"], play_step=step)
        chosen = (step == best["play_step"]) if select_by == "play" else improved
        if chosen:
            best.update(step=step, state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
            agent.save(os.path.join(out_dir, "best"))
            print("  new best by %s (step %d, val nll %.4f, play %.3f); saved %s/best"
                  % (select_by, step, val_m["mean"]["nll"], pc["mean_normalized"], out_dir), flush=True)
        log["best_step"], log["best_mean_val_nll"] = best["step"], best["nll"]
        log["best_by_nll_step"], log["best_by_play_step"], log["best_play"] = best["nll_step"], best["play_step"], best["play"]
        write_log()
        ckpt_vol.commit()
        print("  eval + save took %.1f min" % ((time.time() - te) / 60), flush=True)

    if resume is None:
        eval_fn(0, keep=False)  # untrained head: the reference point for NLL
    t_train = time.time() - (resume["elapsed_s"] if resume else 0.0)

    def maybe_eval(step):
        # Called every few steps as a cheap probe; evals at 1/n_evals, 2/n_evals, ... of training progress
        # (steps or wall-clock, whichever is further). Returns True only when it really evaluated, so that the
        # training loop writes state.pt after an eval and not on every probe.
        if crash_at_step and step >= crash_at_step and not os.path.exists(os.path.join(out_dir, "crashed")):
            open(os.path.join(out_dir, "crashed"), "w").close()
            ckpt_vol.commit()
            raise RuntimeError("crash_at_step %d: simulated preemption" % crash_at_step)
        progress = max(step / steps, (time.time() - t_train) / (max_minutes * 60))
        if progress >= len(log["evals"]) / n_evals:
            eval_fn(step)
            return True
        return False

    stats = {}
    losses = train(
        model, proc, train_by_game, steps=steps, batch_size=batch_size, freeze="full", lr_head=lr_h, lr_backbone=lr_b,
        device="cuda", log_every=100, max_minutes=max_minutes, num_workers=num_workers, warmup=warmup,
        eval_fn=maybe_eval, eval_every=min(25, eval_every), max_passes=max_passes or None, stats=stats,
        balance_key=bkey, group_size=group_size, sigma=sigma, sigma_end=sigma_end or None, w_sph=w_sph, w_ce=w_ce,
        w_ce_schedule=w_ce_schedule, train_act=train_act, resume=resume, save_state_fn=save_state,
        save_state_every_min=state_every_min,
    )
    if log["evals"][-1]["step"] != stats["steps"]:
        eval_fn(stats["steps"])
    chunk = max(1, len(losses) // 10)
    log["train_stats"] = dict(stats, passes={g: round(stats["samples_per_dataset"].get(g, 0) / sizes[g], 2) for g in sorted(sizes)})
    log["loss_curve"] = [{"steps": "%d-%d" % (i, min(i + chunk, len(losses)) - 1), "mean_loss": sum(losses[i:i + chunk]) / len(losses[i:i + chunk])}
                         for i in range(0, len(losses), chunk)]
    print("train stats:", json.dumps(log["train_stats"]))
    print("loss curve (10 chunks):", ", ".join("%.3f" % c["mean_loss"] for c in log["loss_curve"]))

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    elif best["step"] is not None and os.path.exists(os.path.join(out_dir, "best", "model.safetensors")):
        model.load_state_dict(load_file(os.path.join(out_dir, "best", "model.safetensors")))  # resumed run
        model.to("cuda")
    else:
        print("WARNING: no checkpoint was ever kept (no eval improved); using the final weights")
    model.eval()
    print("final model: best checkpoint by %s from step %s (val nll %.4f at step %s, play %.3f at step %s)"
          % (select_by, best["step"], best["nll"], best["nll_step"], best["play"], best["play_step"]))
    calib_records = collect_logits(model, proc, data["calib"], **ev_kw)
    temps = fit_temperatures_from(calib_records)
    n_by_type = [sum(r["qtype"] == t for r in calib_records) for t in range(3)]
    temps = [t if n >= 10 else init_temps[i] for i, (t, n) in enumerate(zip(temps, n_by_type))]
    by_options = fit_option_temperatures(calib_records)
    val_records = collect_logits(model, proc, data["val"], **ev_kw)
    log["temperature"], log["temperature_by_options"] = temps, by_options
    log["final"] = {"step": best["step"], "n_val": len(val_records), "val_raw": per_game_metrics(val_records),
                    "val_calibrated": per_game_metrics(scale_records(val_records, temps, by_options))}
    print("temperatures (choice, score, noul):", [round(t, 3) for t in temps], "| by option count:",
          {k: round(v, 3) for k, v in by_options.items()})
    for key in ("val_raw", "val_calibrated"):
        m = log["final"][key]
        print("[final %s] mean per-game acc %.4f ece %.4f nll %.4f | pooled acc %.4f ece %.4f nll %.4f" % (
            key, m["mean"]["acc"], m["mean"]["ece"], m["mean"]["nll"], m["all"]["acc"], m["all"]["ece"], m["all"]["nll"]))
    cal = log["final"]["val_calibrated"]
    print("%-28s %6s %7s %7s %7s" % ("per source/game (calibrated)", "n", "acc", "ece", "nll"))
    for n_, m in cal["datasets"].items():
        print("%-28s %6d %7.3f %7.3f %7.3f" % (n_, m["n"], m["acc"], m["ece"], m["nll"]))

    agent.temperature, agent.temperature_by_options = temps, by_options
    agent.save(os.path.join(out_dir, "best"))
    write_log()
    ckpt_vol.commit()
    print("saved %s/best with temperatures (%.1f min total)" % (out_dir, (time.time() - t_start) / 60))
    return {"run": run_name, "games": names, "best_step": best["step"], "best_mean_val_nll": best["nll"],
            "best_by_nll_step": best["nll_step"], "best_by_play_step": best["play_step"], "best_play": best["play"],
            "temperature": temps, "temperature_by_options": by_options,
            "final_mean": {k: log["final"][k]["mean"] for k in ("val_raw", "val_calibrated")}, "train_stats": log["train_stats"]}


@app.function(image=image, timeout=5 * 60, volumes={"/ckpt": ckpt_vol.read_only()})
def run_games(model: str):
    """Games a run was trained on, from its metrics.json (``model`` like ``atari-v1/best``)."""
    ckpt_vol.reload()
    with open(os.path.join(CKPT_ROOT, model.split("/")[0], "metrics.json")) as f:
        return json.load(f)["games"]


# Games whose expert baseline makes the normalised score meaningless (excluded) or unstable (flagged).
BASELINE_EXCLUDE = {"Solaris": "expert scores below random"}
BASELINE_FLAGS = {
    "Skiing": "expert is NOOP on every frame; copying it is trivial",
    "PrivateEye": "expert barely above random",
    "Pitfall": "expert stands still, barely above random",
    "MontezumaRevenge": "expert barely above random",
    "Tutankham": "greedy expert gets stuck",
    "DoubleDunk": "stalling until the step cap scores well",
    "Tennis": "stalling until the step cap scores well",
}


def expert_baseline(game: str, root: str = ATARI_ROOT) -> dict:
    """Expert and random scores from /data/atari/expert/<game>/meta.json, read fresh on every call.

    Prefers ``expert_score_cap4500`` / ``random_score_cap4500`` (measured under play-eval's 4,500-step cap and
    settings); falls back to the uncapped scores with ``capped`` False.
    """
    try:
        with open(os.path.join(root, "expert", game, "meta.json")) as f:
            meta = json.load(f)
    except (OSError, ValueError):
        return {"expert": None, "random": None, "capped": False}
    if meta.get("expert_score_cap4500") is not None and meta.get("random_score_cap4500") is not None:
        return {"expert": meta["expert_score_cap4500"], "random": meta["random_score_cap4500"], "capped": True}
    return {"expert": meta.get("expert_score"), "random": meta.get("random_score"), "capped": False}


def normalize(r: dict, base: dict) -> dict:
    """Add (model - random) / (expert - random) from ``base`` to a play result, with exclusion / flag notes."""
    r = dict(r, expert_score=base["expert"], baseline_random=base["random"], baseline_capped=base["capped"],
             normalized=None, flag=None)
    if base["expert"] is not None and base["random"] is not None and base["expert"] != base["random"]:
        r["normalized"] = (r["model_score"] - base["random"]) / (base["expert"] - base["random"])
    g = r["game"]
    if g in BASELINE_EXCLUDE:
        r["flag"] = "excluded: " + BASELINE_EXCLUDE[g]
    elif g in BASELINE_FLAGS:
        r["flag"] = BASELINE_FLAGS[g]
    elif r["normalized"] is not None and not -0.5 <= r["normalized"] <= 1.5:
        r["flag"] = "normalised score outside [-0.5, 1.5]"
    elif r["normalized"] is not None and not base["capped"]:
        r["flag"] = "uncapped baseline"
    return r


@app.function(image=image, timeout=5 * 60, volumes={"/data": data_vol.read_only()})
def expert_baselines(games: list):
    data_vol.reload()
    return {g: expert_baseline(g) for g in games}


@app.function(image=image, gpu="L4", cpu=4, timeout=60 * 60,
              retries=modal.Retries(max_retries=3, initial_delay=10.0),
              volumes={"/cache/hf": hf_vol, "/data": data_vol.read_only(), "/ckpt": ckpt_vol.read_only()})
def play_atari(game: str, model: str, episodes: int = 3, max_steps: int = 4500, seed: int = 100_000,
               random_episodes: int = 10, sample: bool = False, frames: int = 0, gate: bool = False,
               image_size: int = 0, preprocess: str = ""):
    """Play ``ALE/<game>-v5`` with a checkpoint (bf16) and with random actions, same settings.

    The model's action is the most likely one, or with ``sample`` drawn from its calibrated probabilities.
    ``frames`` is 1, 2 (the previous observation goes in too, by the ``expert2f`` rule), or 0 to take the
    checkpoint's own ``atari_frames``. With ``gate``, the act head decides each step: when it says escalate
    rather than act, the previous action is repeated.
    The model sees the raw RGB observation and the ``laya.games.atari_question`` question; FIRE is pressed on
    reset and after each lost life (not counted toward ``max_steps``). Episode i uses seed ``seed + i``.
    ``random_score`` is measured here under the same settings; the normalised score uses the expert meta.json
    baselines (``expert_baseline``), read at eval time.
    """
    from laya.atari_train import game_actions, model_policy, play, random_policy
    from laya.vlm import VLMAgent

    t0 = time.time()
    actions = game_actions(game)
    rnd = play(game, random_policy(len(actions), seed), random_episodes, max_steps, seed)
    path = os.path.join(CKPT_ROOT, model)
    # additive: by default the checkpoint's own recorded resolution and preprocessing path are used; overriding
    # them measures what moving an existing checkpoint to a different path would cost (see laya.preprocess)
    prep_kw = {k: v for k, v in (("image_size", image_size), ("preprocess", preprocess)) if v}
    agent = VLMAgent(path if os.path.exists(path) else model, device="cuda", dtype="bf16", **prep_kw)
    n_frames = frames or int(agent.cfg.get("atari_frames", 1))
    print("%s: %r, %d image tokens per frame" % (game, agent.prep, agent.prep.image_seq_len), flush=True)
    pstats: dict = {}
    res = play(game, model_policy(agent, game, actions, sample, seed, n_frames, gate, pstats), episodes, max_steps, seed)
    data_vol.reload()
    out = normalize({"game": game, "model": model, "sample": sample, "frames": n_frames,
                     "image_size": agent.prep.image_size, "preprocess": agent.prep.backend,
                     "model_score": res["mean_score"],
                     "model_scores": res["scores"], "model_steps": res["steps"], "model_capped": res["capped"],
                     "actions": res["actions"], "random_score": rnd["mean_score"], "random_steps": rnd["steps"],
                     "gate": gate, "gated_frac": round(pstats.get("gated", 0) / max(1, pstats.get("steps", 1)), 4),
                     "act_p_mean": round(pstats["act_p_sum"] / max(1, pstats["steps"]), 4) if gate else None,
                     "seconds": round(time.time() - t0, 1)}, expert_baseline(game))
    print(json.dumps(out))
    return out


def _summary(results):
    lines = ["%-18s %10s %10s %10s %10s %8s %6s  %s" % ("game", "model", "random", "base rnd", "expert", "norm", "steps",
                                                         "top actions / flag")]
    norms, clean, beat = [], [], 0
    for r in sorted(results, key=lambda r: r["game"]):
        top = ", ".join("%s %d%%" % (a, 100 * c / max(1, sum(r["actions"].values())))
                        for a, c in sorted(r["actions"].items(), key=lambda kv: -kv[1])[:3])
        f = lambda v: "-" if v is None else "%.1f" % v  # noqa: E731
        lines.append("%-18s %10.1f %10.1f %10s %10s %8s %6d  %s" % (
            r["game"], r["model_score"], r["random_score"], f(r.get("baseline_random")), f(r["expert_score"]),
            "-" if r["normalized"] is None else "%.2f" % r["normalized"], sum(r["model_steps"]) / len(r["model_steps"]),
            top + ("  [%s]" % r["flag"] if r.get("flag") else "")))
        beat += r["model_score"] > r["random_score"]
        if r["normalized"] is not None and r["game"] not in BASELINE_EXCLUDE:
            norms.append(r["normalized"])
            if not r.get("flag"):
                clean.append(r["normalized"])
    med = statistics.median(norms) if norms else None
    med_clean = statistics.median(clean) if clean else None
    fmt = lambda v: "-" if v is None else "%.3f" % v  # noqa: E731
    lines.append("median normalised score %s over %d games (Solaris excluded); %s over %d unflagged games; "
                 "beats random (own 10-episode random) in %d of %d games"
                 % (fmt(med), len(norms), fmt(med_clean), len(clean), beat, len(results)))
    return "\n".join(lines), {"median_normalized": med, "n_normalized": len(norms), "median_normalized_unflagged": med_clean,
                              "n_unflagged": len(clean), "beat_random": beat, "n_games": len(results)}


@app.local_entrypoint()
def atari_eval(model: str, games: str = "", episodes: int = 3, max_steps: int = 4500, sample: bool = False,
               frames: int = 0, gate: bool = False, image_size: int = 0, preprocess: str = "", out: str = "",
               restart: bool = False):
    """modal run modal_atari_train.py::atari_eval --model atari-v1/best  -- every trained game in parallel on L4s.

    With ``out``, each game's result is written as it lands and a re-run skips the games already in that file
    (matching model, episodes, cap, sampling, frames, gate and the preprocessing overrides), so an interrupted
    evaluation only redoes what is missing; ``restart`` ignores the existing file.
    """
    game_list = _split(games) or run_games.remote(model)
    keys = dict(model=model, sample=sample, frames=frames, gate=gate, episodes=episodes, max_steps=max_steps,
                image_size=image_size, preprocess=preprocess)
    results, done = [], set()
    if out and os.path.exists(out) and not restart:
        with open(out) as f:
            prev = json.load(f)
        if all(prev.get(k) == v for k, v in keys.items() if k in prev):
            results = [r for r in prev.get("results", []) if r["game"] in game_list]
            done = {r["game"] for r in results}
            print("resuming: %d of %d games already played (%s)" % (len(done), len(game_list), ", ".join(sorted(done))))
    todo = [g for g in game_list if g not in done]
    print("playing %d games x %d episodes with %s (%s, frames=%s%s): %s" % (len(todo), episodes, model,
          "sampled" if sample else "greedy", frames or "from checkpoint", ", gated" if gate else "", ", ".join(todo)))

    def write(final=False):
        if not out:
            return
        text, summary = _summary(results)
        with open(out, "w") as f:
            json.dump(dict(keys, results=results, summary=summary), f, indent=2)
        if final:
            print(text)
            print("wrote", out)

    for r in play_atari.starmap([(g, model, episodes, max_steps, 100_000, 10, sample, frames, gate, image_size,
                                  preprocess) for g in todo], return_exceptions=True):
        if isinstance(r, Exception):
            print("failed:", repr(r))
        else:
            results.append(r)
            write()
    if out:
        write(final=True)
    else:
        print(_summary(results)[0])


@app.local_entrypoint()
def renormalize(results: str, out: str = ""):
    """modal run modal_atari_train.py::renormalize --results play.json  -- recompute normalised scores of saved
    atari_eval results from the current expert meta.json baselines (no replay)."""
    with open(results) as f:
        data = json.load(f)
    base = expert_baselines.remote(sorted({r["game"] for r in data["results"]}))
    data["results"] = [normalize(r, base[r["game"]]) for r in data["results"]]
    text, data["summary"] = _summary(data["results"])
    print("%s (%s)" % (data["model"], "sampled" if data.get("sample") else "greedy"))
    print(text)
    if out:
        with open(out, "w") as f:
            json.dump(data, f, indent=2)
        print("wrote", out)


@app.local_entrypoint()
def ab_backbone(
    games: str = "Breakout,Pong",
    sources: str = "expert2f",
    backbones: str = BACKBONE + ",HuggingFaceTB/SmolVLM-256M-Base",
    prefix: str = "atari-ab",
    passes: float = 1.0,
    max_minutes: float = 12.0,
    n_evals: int = 3,
    episodes: int = 5,
    max_steps: int = 4500,
    train_only: bool = False,
    restart: bool = False,
):
    """A/B two pretrained backbones under identical cheap settings, then play both.

        modal run --detach modal_atari_train.py::ab_backbone            # Instruct vs 256M-Base, Breakout+Pong

    One ``train_atari`` run per backbone (fresh head, same games, sources, passes and budget; ``max_minutes`` is
    training time only, evals add a few minutes) runs in parallel on an A100 each, named ``<prefix>-<tag>`` where
    ``tag`` is the last path component of the backbone id, lower-cased. Then each ``best`` plays ``games`` for
    ``episodes`` greedy episodes on L4s, and the summary prints val accuracy / NLL and median normalised play
    score side by side. ``train_only`` stops after training; a rerun without it resumes each arm from its saved
    ``state.pt`` (no training left, only the final scoring is redone) and then plays.
    """
    bb = _split(backbones)
    game_list = _split(games)
    runs = ["%s-%s" % (prefix, b.rsplit("/", 1)[-1].lower()) for b in bb]
    print("training %s on %s (%s), %.2f passes, %.0f min each" % (", ".join(runs), games, sources, passes, max_minutes))
    calls = [train_atari.spawn(run_name=r, sources=sources, games=games, passes=passes, max_minutes=max_minutes,
                               n_evals=n_evals, backbone=b, restart=restart) for r, b in zip(runs, bb)]
    trained = {}
    for r, b, c in zip(runs, bb, calls):
        try:
            trained[r] = c.get()
        except Exception as e:  # keep the other arm's result
            print("%s (%s) failed: %r" % (r, b, e))
    if train_only or not trained:
        for r, res in trained.items():
            print(r, json.dumps(res["final_mean"], indent=1))
        return
    play = {}
    for r in trained:
        play[r] = list(play_atari.starmap([(g, r + "/best", episodes, max_steps) for g in game_list],
                                          return_exceptions=True))
    print("\n%-36s %8s %8s %8s %8s  %s" % ("run", "val acc", "val nll", "ece", "median", "per game"))
    for r, b in zip(runs, bb):
        if r not in trained:
            continue
        m = trained[r]["final_mean"]["val_calibrated"]
        ok = [x for x in play[r] if not isinstance(x, Exception)]
        for x in play[r]:
            if isinstance(x, Exception):
                print("%s play failed: %r" % (r, x))
        text, summ = _summary(ok)
        per_game = ", ".join("%s %s" % (x["game"], "-" if x["normalized"] is None else "%.2f" % x["normalized"])
                             for x in sorted(ok, key=lambda x: x["game"]))
        med = summ["median_normalized"]
        print("%-36s %8.4f %8.4f %8.4f %8s  %s" % (b, m["acc"], m["nll"], m["ece"], "-" if med is None else "%.3f" % med,
                                                  per_game))
    for r in trained:
        print("\n== %s ==\n%s" % (r, _summary([x for x in play[r] if not isinstance(x, Exception)])[0]))


@app.local_entrypoint()
def smoke(frames: int = 2, init_from: str = ""):
    """End to end on synthetic data: a few training steps on the A100 job, then play-eval of the saved checkpoint."""
    r = train_atari.remote(run_name="atari-smoke", passes=4.0, max_minutes=3.0, n_evals=2, val_per_ds=24, n_calib=8,
                           num_workers=8, synthetic=True, frames=frames, init_from=init_from)
    print(json.dumps(r, indent=1))
    results = list(play_atari.starmap([(g, "atari-smoke/best", 2, 200) for g in r["games"]]))
    print(_summary(results)[0])
