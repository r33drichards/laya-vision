"""GPU smoke run of ``laya.game_rl`` (RL from game rewards, GRPO) on the 2 h stack-2 checkpoint: RL only, ~10 minutes
on CartPole, LunarLander and Acrobot (training seeds 10,000-99,999), then the games benchmark's episodes for those
games (``games_eval``, seeds 730,000+) before and after. Not the harness.

    modal run autoresearch/diag/rl_smoke.py --name <new-name> [--minutes 10] [--lr 2e-6] [--returns episode]

The RL checkpoint goes to ``/ckpt/autoresearch/rl-smoke/<name>`` (create-only: an existing name is refused) and the
result JSON to ``autoresearch/runs/full/rl-smoke-<name>.json``.
"""
import json
import os

import modal

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
img = (modal.Image.debian_slim(python_version="3.12")
       .pip_install("torch==2.14.0", "torchvision==0.29.0", "transformers==5.17.0", "safetensors", "huggingface_hub",
                    "numpy", "pillow", "num2words", "gymnasium[classic-control,box2d]==1.3.0")
       .env({"HF_HOME": "/cache/hf", "TOKENIZERS_PARALLELISM": "false", "SDL_VIDEODRIVER": "dummy",
             "SDL_AUDIODRIVER": "dummy"})
       .add_local_dir(REPO + "/laya", "/root/laya", ignore=["**/__pycache__"])
       .add_local_file(REPO + "/autoresearch/games_eval.py", "/root/games_eval.py")
       .add_local_file(REPO + "/autoresearch/game_baselines.json", "/root/game_baselines.json"))
ckpt_vol = modal.Volume.from_name("laya-checkpoints")
vols = {"/cache/hf": modal.Volume.from_name("laya-hf-cache"), "/ckpt": ckpt_vol}
app = modal.App("laya-rl-smoke")
CKPT = "/ckpt/autoresearch/full/long-sep24-b64-stack2/best"
OUT = "/ckpt/autoresearch/rl-smoke"
GAMES = ("CartPole", "LunarLander", "Acrobot")


def bench(path: str) -> dict:
    """The benchmark's control episodes for GAMES, loaded as the harness's Games job loads a checkpoint (bf16)."""
    import torch

    import games_eval as ge
    from laya.vlm import VLMAgent

    agent = VLMAgent(path, device="cuda", dtype="bf16")
    suite = {g: ge.SUITE[g] for g in GAMES}
    res = ge.run_family(agent, "control", suite)
    del agent
    torch.cuda.empty_cache()
    return {g: {k: r[k] for k in ("model", "normalized", "scores", "seconds", "frames")} for g, r in res.items()}


@app.function(image=img, gpu="H100", cpu=16, memory=65536, timeout=90 * 60, volumes=vols)
def run(name: str, minutes: float, lr: float, returns: str, group: int, episodes: int, temperature: float,
        kl: float) -> dict:
    import time

    import torch

    import games_eval as ge
    from laya.game_rl import GameRL, RLConfig
    from laya.vlm import VLMAgent, set_trainable

    out_dir = os.path.join(OUT, name)
    if os.path.exists(out_dir):
        raise SystemExit("%s exists (create-only)" % out_dir)
    t0 = time.time()
    before = bench(CKPT)
    print("before:", json.dumps({g: round(r["normalized"], 3) for g, r in before.items()}), flush=True)

    agent = VLMAgent(CKPT, device="cuda")  # fp32 weights, bf16 autocast, as training runs
    n_train = set_trainable(agent.model, "full")  # everything but the vision tower, as the recipes train
    cfg = RLConfig(games=GAMES, group=group, episodes_per_phase=episodes, temperature=temperature, lr=lr, kl=kl,
                   returns=returns, seed=1)
    rl = GameRL(agent, cfg, baselines=ge.load_baselines())
    print("RL: %d trainable params, modes %s" % (n_train, rl.modes), flush=True)
    t_rl = time.time()
    rl.run(minutes)
    rl_s = time.time() - t_rl
    agent.model.eval()
    agent.cfg["rl"] = {"from": CKPT, "games": list(GAMES), "minutes": minutes, "lr": lr, "returns": returns,
                       "group": group, "episodes_per_phase": episodes, "temperature": temperature, "kl": kl}
    agent.save(out_dir)
    ckpt_vol.commit()
    hist = [{k: v for k, v in h.items() if k != "time"} for h in rl.history]
    summary = rl.summary()
    del agent, rl
    torch.cuda.empty_cache()
    after = bench(out_dir)
    print("after:", json.dumps({g: round(r["normalized"], 3) for g, r in after.items()}), flush=True)
    return {"name": name, "checkpoint": out_dir, "from": CKPT, "config": agent_cfg(cfg), "rl_seconds": rl_s,
            "summary": summary, "phases": hist, "before": before, "after": after,
            "total_seconds": time.time() - t0, "gpu": torch.cuda.get_device_name(0)}


def agent_cfg(cfg) -> dict:
    from dataclasses import asdict

    return {k: v for k, v in asdict(cfg).items()}


@app.local_entrypoint()
def main(name: str, minutes: float = 10.0, lr: float = 2e-6, returns: str = "episode", group: int = 8,
         episodes: int = 16, temperature: float = 1.0, kl: float = 0.0):
    path = os.path.join(REPO, "autoresearch", "runs", "full", "rl-smoke-%s.json" % name)
    if os.path.exists(path):
        raise SystemExit("%s exists (create-only)" % path)
    res = run.remote(name, minutes, lr, returns, group, episodes, temperature, kl)
    with open(path, "w") as f:
        json.dump(res, f, indent=1)
    print("wrote", path)
    for h in res["phases"]:
        print("phase %d: %s | rollout %.0f s (%.0f%%), %.2f ep/s" % (
            h["phase"], " ".join("%s %.1f" % (g, r["mean_score"]) for g, r in h["games"].items()),
            h["rollout_s"], 100 * h["rollout_frac"], h["episodes_per_s"]))
    for g in res["before"]:
        print("%-12s before %.3f (%.1f)  after %.3f (%.1f)" % (g, res["before"][g]["normalized"],
                                                             res["before"][g]["model"], res["after"][g]["normalized"],
                                                             res["after"][g]["model"]))
