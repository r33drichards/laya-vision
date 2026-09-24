"""ACH diagnostic: per-move agreement with the expert on expert-driven vs model-driven states (control games), and
held-out expert-frame accuracy (Breakout), for the 2 h single and stack-2 checkpoints in their own frame modes."""
import modal

REPO = "/home/user/laya-vision"
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
app = modal.App("laya-ach2")
CKPTS = {"single": "/ckpt/autoresearch/full/long-sep24-b64/best",
         "stack-2": "/ckpt/autoresearch/full/long-sep24-b64-stack2/best"}
GAMES = ("CartPole", "LunarLander", "MountainCar", "Acrobot")
EPISODES, CAP = 6, 200


@app.function(image=img, gpu="L4", cpu=8, memory=32768, timeout=3600, volumes=vols)
def run(label: str) -> dict:
    import numpy as np
    import games_eval as ge
    from laya import frames as F
    from laya.vlm import VLMAgent
    from laya.vlm_train import load_jsonl_examples

    agent = VLMAgent(CKPTS[label], device="cuda", dtype="bf16")
    out = {"label": label, "mode": F.mode_for(agent.cfg, "control"), "control": {}}

    def act(hist, mode, fam, q, actions):
        p = agent.predict(hist.state(mode, fam), {"a": q})["answers"]["a"]["probabilities"]
        return max(actions, key=lambda a: p[a])

    for game in GAMES:
        spec = ge.SUITE[game]
        mode = F.mode_for(agent.cfg, "control")
        q = ge.question_for(spec, mode)["action"]
        res = {}
        for driver in ("expert", "model"):
            agree, total, by_t, lens = 0, 0, {}, []
            preds = {}
            for i in range(EPISODES):
                env = ge.make_env(spec, i)
                hist = F.History(F.frames_needed(mode, "control"))
                t = 0
                while not env.done and t < CAP:
                    hist.push(env.frame())
                    m, e = act(hist, mode, "control", q, env.actions), env.expert()
                    preds[m] = preds.get(m, 0) + 1
                    ok = int(m == e)
                    agree += ok; total += 1
                    b = min(t // 25, 7)
                    s = by_t.setdefault(b, [0, 0]); s[0] += ok; s[1] += 1
                    env.step(e if driver == "expert" else m)
                    t += 1
                lens.append(t)
            res[driver] = {"agree": round(agree / max(1, total), 3), "n": total, "ep_len": lens, "pred_hist": preds,
                           "by_step25": {k * 25: round(v[0] / v[1], 3) for k, v in sorted(by_t.items())}}
        out["control"][game] = res
        print(label, game, res, flush=True)

    # Breakout: held-out expert frames (val split of the history re-recording), accuracy vs the expert's argmax
    mode = F.mode_for(agent.cfg, "atari")
    exs = load_jsonl_examples("/data/atari/experthist", "Breakout", "val", history=4)[:600]
    hit, hist_pred = 0, {}
    from laya.games import atari_question
    for ex in exs:
        frames = list(ex.get("history") or []) + [ex["state"]["image"]]
        st = F.state(frames, mode, "atari")
        qd = ex["q"]
        q = {"type": "choice", "instructions": qd["ins"], "criteria": list(qd["crit"]) if isinstance(qd["crit"], (list, tuple)) else qd["crit"]}
        p = agent.predict(st, {"a": q})["answers"]["a"]["probabilities"]
        opts = list(p.keys())
        pi = int(np.argmax([p[o] for o in opts]))
        t = np.asarray(ex["target"], dtype=float)
        hit += int(t[pi] >= t.max() - 1e-9)
        hist_pred[opts[pi]] = hist_pred.get(opts[pi], 0) + 1
    out["breakout_expert_frames"] = {"acc": round(hit / len(exs), 3), "n": len(exs), "pred_hist": hist_pred}
    print(label, "Breakout", out["breakout_expert_frames"], flush=True)
    return out


@app.local_entrypoint()
def main():
    import json
    res = list(run.map(["single", "stack-2"]))
    print(json.dumps(res, indent=1))
    json.dump(res, open(REPO + "/autoresearch/runs/full/ach-breakout-control.json", "w"), indent=1)
