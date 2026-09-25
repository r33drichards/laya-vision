"""CartPole diagnostic: is the 2 h checkpoints' CartPole failure (agreement with the expert 0.51, RIGHT 96% of the
time) a data/format problem, or can the model not see what the expert decides on?

1. **Format check.** The toolkit's training states and the benchmark's play states for the same seeded episode,
   step by step: same pixels, same expert label.
2. **Fit test.** From each 2 h checkpoint, train on toolkit CartPole examples alone (nothing else in the mix) for
   ``MINUTES`` at the recipe's LRs, with the vision tower frozen (the recipe) and trained. Before and after:
   accuracy on the training frames, accuracy on held-out expert-driven states binned by how far the expert is from
   its decision boundary (``|theta + 0.5 theta_dot + 0.01 x + 0.1 x_dot|``), and greedy play (benchmark cap, seeds
   the benchmark does not use). A model that cannot fit its own training frames is perception-limited, and no
   amount of sampling, search or RL fixes that; one that fits them but does not play is a data/coverage problem.

Seeds: training episodes 20_000+, held-out and play 60_000+ (both below the 100_000 eval floor).
``modal run autoresearch/diag/cartpole_fit.py`` writes ``autoresearch/runs/full/cartpole-fit.json``."""
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
       .add_local_file(REPO + "/autoresearch/game_baselines.json", "/root/game_baselines.json")
       .add_local_file(REPO + "/autoresearch/toolkit.py", "/root/toolkit.py"))
vols = {"/cache/hf": modal.Volume.from_name("laya-hf-cache"), "/ckpt": modal.Volume.from_name("laya-checkpoints"),
        "/data": modal.Volume.from_name("laya-datasets")}
app = modal.App("laya-cartpole-fit")
CKPTS = {"single": "/ckpt/autoresearch/full/long-sep24-b64/best",
         "stack-2": "/ckpt/autoresearch/full/long-sep24-b64-stack2/best"}
N_TRAIN, N_HELD_EPISODES, PLAY_EPISODES, CAP = 3000, 6, 8, 200
MINUTES, BATCH, LR_HEAD, LR_BACKBONE = 12, 32, 5e-5, 1e-5
TRAIN_SEED, HELD_SEED = 20_000, 60_000
BINS = (0.0, 0.02, 0.05, 0.1, 0.2, 1e9)


def margin(obs) -> float:
    x, x_dot, theta, theta_dot = [float(v) for v in obs]
    return theta + 0.5 * theta_dot + 0.01 * x + 0.1 * x_dot


@app.function(image=img, cpu=4, memory=8192, timeout=1200)
def format_check() -> dict:
    """Toolkit examples (eps=0, keep=1, no flip, so every step is kept and labelled) against a benchmark-style
    replay of the same seeded episode: pixels and labels per step, in both frame modes."""
    import io

    import numpy as np
    from PIL import Image

    import games_eval as ge
    import toolkit
    from laya import frames as F

    out = {}
    for mode in ("single", "stack-2"):
        exs = toolkit.control_examples("CartPole", 150, seed=TRAIN_SEED, eps=0.0, keep=1.0, flip=False, frames=mode)
        exs = [e for e in exs if e["id"].startswith("cartpole-%d-" % TRAIN_SEED)]
        spec = ge.SUITE["CartPole"]
        from laya.controlgames import ControlGame
        env = ge.ControlAdapter(ControlGame("CartPole", TRAIN_SEED), None, ge.question_for(spec)["action"]["criteria"])
        hist, by_t = F.History(F.frames_needed(mode, "control")), {}
        while not env.done and env.steps < 150:
            hist.push(env.frame())
            by_t[env.steps] = (hist.state(mode, "control"), env.expert())
            env.step(env.expert())
        pix_diff, lab_diff, n = 0, 0, 0
        for e in exs:
            t = int(e["id"].split("-t")[1])
            if t not in by_t:
                continue
            st, lab = by_t[t]
            a = [np.asarray(Image.open(io.BytesIO(b)).convert("RGB")) for b in F.state_images(e["state"])]
            b = [np.asarray(x) for x in F.state_images(st)]
            n += 1
            pix_diff += int(len(a) != len(b) or any(not np.array_equal(u, v) for u, v in zip(a, b)))
            lab_diff += int(("LEFT", "RIGHT")[e["label"]] != lab)
        q_train = e["q"]
        q_bench = ge.question_for(spec, mode)["action"]
        out[mode] = {"steps_compared": n, "pixel_mismatches": pix_diff, "label_mismatches": lab_diff,
                     "question_same": q_train["ins"] == q_bench["instructions"]
                     and list(q_train["crit"]) == list(q_bench["criteria"])}
    # label balance and margin spread of the training distribution (eps 0.3, the toolkit default)
    from laya.controlgames import ControlGame
    import random
    ms, rights = [], 0
    for i in range(10):
        g, rng = ControlGame("CartPole", TRAIN_SEED + i), random.Random(i)
        while not g.done:
            m = margin(g.obs)
            ms.append(abs(m)); rights += int(m > 0)
            g.step(rng.choice(g.actions) if rng.random() < 0.3 else g.expert())
    ms = np.asarray(ms)
    out["train_distribution"] = {"states": len(ms), "right_frac": round(rights / len(ms), 3),
                                 "abs_margin_pct": {p: round(float(np.percentile(ms, p)), 4) for p in (10, 25, 50, 75, 90)}}
    print(out, flush=True)
    return out


@app.function(image=img, gpu="H100", cpu=16, memory=131072, timeout=7200, volumes=vols)
def fit(arm: dict) -> dict:
    import time

    import numpy as np
    import torch

    import games_eval as ge
    import toolkit
    import laya.vlm_train as vt
    from laya import frames as F
    from laya.controlgames import ControlGame
    from laya.games import control_question
    from laya.vlm import VLMAgent

    torch.manual_seed(0)
    agent = VLMAgent(CKPTS[arm["ckpt"]], device="cuda")
    mode = F.mode_for(agent.cfg, "control")
    q = control_question("CartPole", mode)["action"]
    train_ex = toolkit.control_examples("CartPole", arm.get("n_train", N_TRAIN), seed=TRAIN_SEED, frames=mode, workers=16)
    probe_src = train_ex
    if arm.get("mix"):  # the other control games alongside, as in the recipe's mix (probe stays CartPole)
        for g in ("Acrobot", "MountainCar", "LunarLander"):
            train_ex = train_ex + toolkit.control_examples(g, N_TRAIN, seed=TRAIN_SEED, frames=mode, workers=8)
    probe = probe_src[:400]

    def choose(st):
        p = agent.predict(st, {"a": q})["answers"]["a"]["probabilities"]
        return max(("LEFT", "RIGHT"), key=lambda a: p[a])

    def measure():
        agent.model.eval()
        hit = sum(choose(e["state"]) == ("LEFT", "RIGHT")[e["label"]] for e in probe)
        bins = {i: [0, 0] for i in range(len(BINS) - 1)}
        preds = {"LEFT": 0, "RIGHT": 0}
        for i in range(N_HELD_EPISODES):  # expert-driven held-out states, every step up to the cap
            g, hist = ControlGame("CartPole", HELD_SEED + i), F.History(F.frames_needed(mode, "control"))
            while not g.done and g.steps < CAP:
                hist.push(g.frame())
                m, e = choose(hist.state(mode, "control")), g.expert()
                preds[m] += 1
                b = next(k for k in range(len(BINS) - 1) if abs(margin(g.obs)) < BINS[k + 1])
                bins[b][0] += int(m == e); bins[b][1] += 1
                g.step(e)
        scores = []
        for i in range(PLAY_EPISODES):  # greedy closed-loop play, benchmark cap
            g, hist = ControlGame("CartPole", HELD_SEED + 100 + i), F.History(F.frames_needed(mode, "control"))
            while not g.done and g.steps < CAP:
                hist.push(g.frame())
                g.step(choose(hist.state(mode, "control")))
            scores.append(g.score)
        tot = [sum(v[0] for v in bins.values()), sum(v[1] for v in bins.values())]
        return {"train_acc": round(hit / len(probe), 3), "heldout_acc": round(tot[0] / tot[1], 3),
                "heldout_by_abs_margin": {"%g-%g" % (BINS[k], BINS[k + 1]): (round(v[0] / v[1], 3) if v[1] else None, v[1])
                                          for k, v in bins.items()},
                "heldout_pred": preds, "play_scores": scores, "play_mean": float(np.mean(scores))}

    before = measure()
    print(arm, "before", before, flush=True)
    if arm["vision"]:
        base = vt.set_trainable
        vt.set_trainable = lambda model, mode="head", n_last=4: base(model, mode, n_last=n_last, train_vision=True)
    stats, t0 = {}, time.time()
    extra = {"lr_vision": arm["lr_vision"]} if arm.get("lr_vision") is not None else {}
    losses = vt.train(agent.model, agent.processor, train_ex, steps=10**9, batch_size=BATCH, freeze="full", **extra,
                      lr_head=LR_HEAD, lr_backbone=LR_BACKBONE, warmup=20, max_minutes=arm.get("minutes", MINUTES), num_workers=12,
                      log_every=100, device="cuda", stats=stats)
    after = measure()
    print(arm, "after", after, flush=True)
    return {"arm": arm, "mode": mode, "n_train": len(train_ex), "steps": len(losses),
            "train_minutes": round((time.time() - t0) / 60, 1),
            "loss_first100": float(np.mean(losses[:100])), "loss_last100": float(np.mean(losses[-100:])),
            "before": before, "after": after}


@app.local_entrypoint()
def main(lowlr: bool = False, big: bool = False):
    """``--lowlr``: the vision tower trained at its own low LR (``train(lr_vision=...)``), CartPole alone or with the
    other control games; writes ``cartpole-fit-lowlr.json``."""
    import json

    if big:  # ~17x the frames (50,000, ~3,300 episodes) and 30 minutes: data-limited or perception-limited?
        arms = [{"ckpt": "stack-2", "vision": False, "n_train": 50_000, "minutes": 30},
                {"ckpt": "stack-2", "vision": True, "lr_vision": 1e-6, "n_train": 50_000, "minutes": 30}]
        res = {"fit": list(fit.map(arms))}
        print(json.dumps(res, indent=1))
        json.dump(res, open(REPO + "/autoresearch/runs/full/cartpole-fit-50k.json", "x"), indent=1)
        return
    if lowlr:
        arms = [{"ckpt": "stack-2", "vision": True, "lr_vision": 1e-6},
                {"ckpt": "stack-2", "vision": True, "lr_vision": 2e-7},
                {"ckpt": "stack-2", "vision": True, "lr_vision": 1e-6, "mix": True},
                {"ckpt": "single", "vision": True, "lr_vision": 1e-6}]
        res = {"fit": list(fit.map(arms))}
        print(json.dumps(res, indent=1))
        json.dump(res, open(REPO + "/autoresearch/runs/full/cartpole-fit-lowlr.json", "x"), indent=1)
        return

    fc = format_check.spawn()
    arms = [{"ckpt": c, "vision": v} for c in ("single", "stack-2") for v in (False, True)]
    res = {"format_check": None, "fit": list(fit.map(arms))}
    res["format_check"] = fc.get()
    print(json.dumps(res, indent=1))
    json.dump(res, open(REPO + "/autoresearch/runs/full/cartpole-fit.json", "w"), indent=1)
