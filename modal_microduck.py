"""Fine-tune Laya Vision to drive quackd's simulated Microduck, without losing what it could already do.

    modal run --detach modal_microduck.py::finetune --run-name microduck-kick-ft2

The first attempt (``microduck-kick-ft1``, ``modal_app.py::finetune_long`` on the duck frames and three VQA sets)
learned the task in the 2D simulator and forgot most of its games, because no game data was in its mix. This job
replays the recipe the checkpoint came from (``autoresearch/experiment.py``: the autoresearch data pool's Cauldron and
rubric sets, its expert game frames, and the Maze, Snake and classic-control examples ``autoresearch/toolkit.py``
generates) alongside the duck frames, from both simulators:

* ``microduck_kick``: quackd's 2D simulator, ``scripts/collect_microduck_rollouts.py``
* ``microduck3d_kick``: quackd's MuJoCo simulator, upstream's Microduck on its trained walking policy, the same
  script with ``--sim mujoco``

Draws: the duck sets ``duck_frac`` of the time (split ``duck_3d_share`` to the 3D set), the game sets ``game_frac``,
and the recipe's sets the rest in the recipe's proportions. Temperatures are refitted on the pool's calibration tail
only, as the recipe's are. The last ``n_calib`` frames of each duck set are held out of training all the same and
scored afterwards. ``microduck-kick-ft2`` fitted them on the duck frames too: the duck question has four options,
the same temperature bucket as A-OKVQA's, and frames the model answers almost surely right pulled the choice
temperature from 3.86 to 2.27, which left A-OKVQA's calibrated ECE at 0.24. ``recalibrate`` refits a saved run on
the pool alone. The checkpoint goes to
``/ckpt/smolvlm/<run_name>/best`` with ``metrics.json`` beside it, so every ``modal_app.py`` job that takes a run name
(``evaluate``, ``games_eval``, ``publish``) takes this one.
"""
import json
import math
import os
import sys
import time

import modal

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO, "autoresearch"))

import harness as H  # noqa: E402  (the autoresearch harness: its image, volumes and data pool)

app = modal.App("laya-microduck")
# the harness's image already carries the laya package and toolkit.py; this job imports the harness itself too
image = H.image.add_local_file(os.path.join(REPO, "autoresearch", "harness.py"), "/root/harness.py")

DUCK_SETS = ("microduck_kick", "microduck3d_kick")
CONTROL_GAMES = ("CartPole", "Acrobot", "MountainCar", "LunarLander")


@app.function(image=image, gpu="A100", cpu=24, memory=65536, timeout=3 * 60 * 60, volumes=H.VOLUMES)
def finetune(
    run_name: str = "microduck-kick-ft2",
    init: str = "autoresearch/full/long-sep24-b64/best",
    steps: int = 4000,
    max_minutes: float = 45.0,
    batch_size: int = 32,
    lr_head: float = 1.41e-4,
    lr_backbone: float = 2.83e-5,
    warmup: int = 60,
    duck_frac: float = 0.35,
    duck_3d_share: float = 0.6,
    game_frac: float = 0.30,
    n_calib: int = 300,
    eval_every: int = 1000,
    w_next: float = 0.5,
):
    import random

    import torch

    import toolkit
    from laya.vlm import VLMAgent
    from laya.vlm_train import collect_logits, fit_temperatures_from, load_jsonl_examples, metrics_from, train

    t0 = time.time()
    torch.manual_seed(0)
    random.seed(0)
    out_dir = os.path.join("/ckpt/smolvlm", run_name)
    if os.path.exists(os.path.join(out_dir, "best", "vlm_agent_config.json")):
        raise SystemExit("%s/best exists; pick a new --run-name" % out_dir)
    print("GPU:", torch.cuda.get_device_name(0), flush=True)

    # the recipe's data: the pool's train and calibration parts, and its expert game frames
    pool = H.load_pool(("train", "calib", "games"))
    recipe = [ex for exs in pool["train"].values() for ex in exs]
    calib = [ex for name in H.CALIB_DATASETS for ex in pool["calib"][name]]
    duck_calib = []
    games = [ex for exs in pool["games"].values() for ex in exs]
    games += toolkit.maze_examples(20000) + toolkit.snake_examples(20000)
    for g in CONTROL_GAMES:
        games += toolkit.control_examples(g, 5000)
    print("recipe %d examples over %d sets, games %d, calib %d (%.0f s)"
          % (len(recipe), len(pool["train"]), len(games), len(calib), time.time() - t0), flush=True)

    # the duck frames: the last n_calib of each train split calibrate, the val splits score
    duck_train, duck_val = [], {}
    for name in DUCK_SETS:
        tr = load_jsonl_examples("/data/vqa", name, "train")
        duck_train += tr[:-n_calib]
        duck_calib += tr[-n_calib:]
        duck_val[name] = load_jsonl_examples("/data/vqa", name, "val")
        print("%s: %d train, %d calib, %d val" % (name, len(tr) - n_calib, n_calib, len(duck_val[name])), flush=True)

    # shares: game_mix gives the games game_frac and the rest to the non-game sets by weight, so the duck weights
    # are set to make the two duck sets duck_frac of all draws
    recipe_w = {"score_vlfeedback": 3.0}
    recipe_total = sum(recipe_w.get(n, 1.0) for n in pool["train"])
    duck_total = recipe_total * duck_frac / (1.0 - game_frac - duck_frac)
    base_w = dict(recipe_w, microduck3d_kick=duck_total * duck_3d_share,
                  microduck_kick=duck_total * (1.0 - duck_3d_share))
    data, mix = toolkit.game_mix(recipe + duck_train, games, game_frac, base_weights=base_w)
    # with mix_alpha 0 a dataset's weight is its share of draws, whatever its size
    shares = {k: v / sum(mix.values()) for k, v in mix.items()}
    print("draw shares: duck 2D %.3f, duck 3D %.3f, games %.3f, recipe sets %.3f" % (
        shares["microduck_kick"], shares["microduck3d_kick"],
        sum(v for k, v in shares.items() if k.startswith("game")),
        sum(v for k, v in shares.items() if k in pool["train"])), flush=True)

    agent = VLMAgent(H.ckpt_path(init), device="cuda")
    has_next = bool(agent.cfg.get("next_head"))
    print("initialised from %s (next head: %s)" % (init, has_next), flush=True)
    model, proc = agent.model, agent.processor
    log = {"run": run_name, "init": init, "args": dict(
        steps=steps, max_minutes=max_minutes, batch_size=batch_size, lr_head=lr_head, lr_backbone=lr_backbone,
        warmup=warmup, duck_frac=duck_frac, duck_3d_share=duck_3d_share, game_frac=game_frac, n_calib=n_calib,
        w_next=w_next if has_next else 0.0), "draw_shares": shares, "evals": []}

    def duck_eval(step):
        model.eval()
        row = {"step": step}
        for name, exs in duck_val.items():
            m = metrics_from(collect_logits(model, proc, exs, batch_size=64, num_workers=8))["all"]
            row[name] = {k: round(float(m[k]), 4) for k in ("acc", "ece", "nll")}
        model.train()
        log["evals"].append(row)
        print("[eval step %d] %s" % (step, " | ".join("%s acc %.4f" % (n, row[n]["acc"]) for n in duck_val)),
              flush=True)

    def maybe_eval(step):
        if step and step % eval_every == 0:
            duck_eval(step)
            return True
        return False

    stats = {}
    losses = train(model, proc, data, steps=steps, batch_size=batch_size, freeze="full", lr_head=lr_head,
                   lr_backbone=lr_backbone, warmup=warmup, mix_weights=mix, w_next=w_next if has_next else 0.0,
                   max_minutes=max_minutes, num_workers=12, log_every=100, device="cuda", eval_fn=maybe_eval,
                   eval_every=math.gcd(25, eval_every), stats=stats)
    model.eval()
    duck_eval(stats.get("steps", len(losses)))

    temps = fit_temperatures_from(collect_logits(model, proc, calib, batch_size=32, num_workers=8))
    agent.temperature, agent.temperature_by_options = list(temps), {}
    held = metrics_from(collect_logits(model, proc, duck_calib, batch_size=64, num_workers=8), temps)["all"]
    log["duck_calib_calibrated"] = {k: round(float(held[k]), 4) for k in ("acc", "ece", "nll")}
    agent.save(os.path.join(out_dir, "best"))
    chunk = max(1, len(losses) // 10)
    log.update(temperature=temps, train_stats=stats,
               loss_curve=[sum(losses[i:i + chunk]) / len(losses[i:i + chunk]) for i in range(0, len(losses), chunk)])
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(log, f, indent=2)
    H.ckpt_vol.commit()
    print("temperatures (choice, score, noul):", [round(t, 3) for t in temps])
    print("saved %s/best (%.1f min)" % (out_dir, (time.time() - t0) / 60), flush=True)
    return {k: log[k] for k in ("run", "evals", "temperature", "draw_shares")}


@app.function(image=image, gpu=["A10G", "L4", "A100"], cpu=16, memory=32768, timeout=60 * 60, volumes=H.VOLUMES)
def recalibrate(run: str = "microduck-kick-ft2/best", out_run: str = "microduck-kick-ft2-cal"):
    """Refit ``run``'s temperatures on the autoresearch pool's calibration tail alone and save the result as a new run
    (``/ckpt/smolvlm/<out_run>/best``, same weights): checkpoints are create-only. Only the probabilities move; every
    answer, and so every accuracy and every game, stays as it was."""
    import shutil

    from laya.vlm import VLMAgent
    from laya.vlm_train import collect_logits, fit_temperatures_from, load_jsonl_examples, metrics_from

    out_dir = os.path.join("/ckpt/smolvlm", out_run)
    if os.path.exists(out_dir):
        raise SystemExit("%s exists; pick a new --out-run" % out_dir)
    src = H.ckpt_path(run)
    agent = VLMAgent(src, device="cuda")
    pool = H.load_pool(("calib",))
    calib = [ex for name in H.CALIB_DATASETS for ex in pool["calib"][name]]
    before = list(agent.temperature)
    temps = fit_temperatures_from(collect_logits(agent.model, agent.processor, calib, batch_size=32, num_workers=8))
    agent.temperature, agent.temperature_by_options = list(temps), {}
    agent.save(os.path.join(out_dir, "best"))
    log = {"run": out_run, "from": run, "temperature_before": before, "temperature": temps, "duck_val_calibrated": {}}
    src_metrics = os.path.join(os.path.dirname(src), "metrics.json")
    if os.path.exists(src_metrics):
        shutil.copy(src_metrics, os.path.join(out_dir, "training_metrics.json"))
    for name in DUCK_SETS:
        m = metrics_from(collect_logits(agent.model, agent.processor, load_jsonl_examples("/data/vqa", name, "val"),
                                        batch_size=64, num_workers=8), temps)["all"]
        log["duck_val_calibrated"][name] = {k: round(float(m[k]), 4) for k in ("acc", "ece", "nll")}
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(log, f, indent=2)
    H.ckpt_vol.commit()
    print(json.dumps(log, indent=1))
    return log
