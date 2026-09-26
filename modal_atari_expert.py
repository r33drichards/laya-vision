"""Modal jobs for expert-labelled Atari frames (``laya.atari_data.expert``): pretrained CleanRL PPO agents play each
game, and every full-colour frame is saved with the agent's action probabilities.

    modal run modal_atari_expert.py::main --games Breakout,Pong --train-frames 0    # expert vs random scores, writes nothing
    modal run modal_atari_expert.py::main --games Pong --train-frames 0 --sticky 0   # the agents' own env: no sticky actions
    modal run modal_atari_expert.py::main --games Breakout --train-frames 2000 --val-frames 200   # small test
    modal run --detach modal_atari_expert.py::main                                # every game in expert.AGENTS
    modal run modal_atari_expert.py::rescore --cap 4500                    # add capped baselines to every meta.json
    modal run modal_atari_expert.py::table                                     # per-game table from the meta.json files
    modal run --detach modal_atari_expert.py::history --games Freeway,Breakout   # experthist: records with 4 past frames

Volumes (created out of band; never ``modal deploy`` this app):
    laya-datasets  -> /data      (this app writes only under /data/atari/expert/ and, create-only, under
                                  /data/atari/experthist/; see site-docs/reference/atari-data-format.md)
    laya-hf-cache  -> /cache/hf  (HF_HOME; agent weights)
"""
import json
import os
import time

import modal

app = modal.App("laya-atari-expert")

hf_vol = modal.Volume.from_name("laya-hf-cache")
data_vol = modal.Volume.from_name("laya-datasets")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.14.0", index_url="https://download.pytorch.org/whl/cpu")  # `import laya` needs torch
    .pip_install(
        "ale-py==0.12.1",
        "gymnasium==1.3.0",
        "jax[cpu]",
        "flax",
        "opencv-python-headless",
        "pillow",
        "numpy",
        "huggingface_hub",
    )
    .env({"HF_HOME": "/cache/hf"})
    .add_local_python_source("laya")
)

OUT_ROOT = "/data/atari/expert"


def _commit(vol, tries: int = 8):
    for i in range(tries):
        try:
            vol.commit()
            return
        except Exception as e:  # concurrent commits from many containers can conflict; back off and retry
            print("commit failed (%s), retry %d" % (e, i + 1))
            time.sleep(5 * (i + 1))
    vol.commit()


@app.function(image=image, cpu=2, memory=6144, timeout=6 * 60 * 60, max_containers=57,
              volumes={"/data": data_vol, "/cache/hf": hf_vol})
def generate_game(game: str, train_frames: int, val_frames: int, eval_episodes: int, sticky: float = 0.25) -> dict:
    from laya.atari_data import expert

    t0 = time.time()
    out = os.path.join(OUT_ROOT, game)
    if train_frames > 0 and os.path.exists(os.path.join(out, "_READY")):
        os.remove(os.path.join(out, "_READY"))  # readers must not use a game while it is being rewritten
        _commit(data_vol)
    meta = expert.generate(game, OUT_ROOT, train_frames, val_frames, eval_episodes, sticky=sticky,
                           log=lambda s: print(s, flush=True))
    try:
        hf_vol.commit()
    except Exception as e:
        print("hf cache commit failed:", e)
    if train_frames > 0:
        _commit(data_vol)
        open(os.path.join(out, "_READY"), "w").close()
        _commit(data_vol)
    meta["seconds"] = round(time.time() - t0)
    return meta


def _row(m: dict) -> str:
    return "%-17s %-58s %11.1f %11.1f %6s %5s" % (
        m["game"], m["agent"]["repo"].split("/")[1].split("-v5-")[1], m["expert_score"], m["random_score"],
        m.get("train", {}).get("records", "-"), m.get("val", {}).get("records", "-"))


HEADER = "%-17s %-58s %11s %11s %6s %5s" % ("game", "agent", "expert", "random", "train", "val")


@app.function(image=image)
def list_games() -> list:
    from laya.atari_data.expert import AGENTS

    return sorted(AGENTS)


@app.local_entrypoint()
def main(games: str = "", train_frames: int = 20_000, val_frames: int = 1_000, eval_episodes: int = 5,
         sticky: float = 0.25):
    if sticky != 0.25 and train_frames > 0:
        raise SystemExit("--sticky is a score-only pipeline check; use it with --train-frames 0")
    names = [g.strip() for g in games.split(",") if g.strip()] or list_games.remote()
    rows, failed = [], []
    for name, res in zip(names, generate_game.map(names, kwargs=dict(train_frames=train_frames,
                                                                    val_frames=val_frames,
                                                                    eval_episodes=eval_episodes, sticky=sticky),
                                                   return_exceptions=True, order_outputs=True)):
        if isinstance(res, BaseException):
            failed.append((name, repr(res)[:200]))
            print("FAILED", name, repr(res)[:200])
        else:
            rows.append(res)
            print(_row(res), "(%ds)" % res["seconds"], flush=True)
    print("\n" + HEADER)
    for m in rows:
        print(_row(m))
    for name, err in failed:
        print("FAILED %s: %s" % (name, err))


@app.function(image=image, volumes={"/data": data_vol}, timeout=10 * 60)
def read_metas() -> list:
    metas = []
    if os.path.isdir(OUT_ROOT):
        for game in sorted(os.listdir(OUT_ROOT)):
            p = os.path.join(OUT_ROOT, game, "meta.json")
            if os.path.exists(p):
                with open(p) as f:
                    m = json.load(f)
                m["ready"] = os.path.exists(os.path.join(OUT_ROOT, game, "_READY"))
                metas.append(m)
    return metas


@app.local_entrypoint()
def table():
    metas = read_metas.remote()
    print(HEADER + " ready")
    for m in metas:
        print(_row(m), m["ready"])


@app.function(image=image, cpu=2, memory=4096, timeout=2 * 60 * 60, max_containers=57, volumes={"/cache/hf": hf_vol})
def capped_baselines(game: str, cap: int, episodes: int) -> dict:
    from laya.atari_data import expert

    return expert.baseline_scores(game, cap, episodes)


@app.function(image=image, volumes={"/data": data_vol}, timeout=20 * 60)
def add_capped_baselines(results: list, cap: int, episodes: int) -> int:
    """Add ``expert_score_cap<cap>`` / ``random_score_cap<cap>`` to each game's meta.json; one commit at the end."""
    import numpy as np

    for r in results:
        path = os.path.join(OUT_ROOT, r["game"], "meta.json")
        with open(path) as f:
            meta = json.load(f)
        meta.update({"expert_score_cap%d" % cap: float(np.mean(r["expert"])), "expert_scores_cap%d" % cap: r["expert"],
                     "random_score_cap%d" % cap: float(np.mean(r["random"])), "random_scores_cap%d" % cap: r["random"]})
        meta.setdefault("eval_caps", {})[str(cap)] = (
            "same env, seeds and policies as expert_score / random_score (%d episodes each), but each episode ends "
            "after %d decisions (auto-FIRE steps not counted)" % (episodes, cap))
        with open(path + ".tmp", "w") as f:
            json.dump(meta, f, indent=1)
        os.replace(path + ".tmp", path)
    _commit(data_vol)
    return len(results)


@app.local_entrypoint()
def rescore(cap: int = 4500, episodes: int = 5, games: str = ""):
    names = [g.strip() for g in games.split(",") if g.strip()] or list_games.remote()
    results = list(capped_baselines.map(names, kwargs=dict(cap=cap, episodes=episodes)))
    print("%-17s %12s %12s" % ("game", "expert_cap", "random_cap"))
    for r in results:
        print("%-17s %12.1f %12.1f" % (r["game"], sum(r["expert"]) / len(r["expert"]), sum(r["random"]) / len(r["random"])))
    print("wrote", add_capped_baselines.remote(results, cap, episodes), "meta.json files")


HISTORY_ROOT = "/data/atari/experthist"
HISTORY_FRAMES = 4  # previous decision screens stored per record: enough for a 5-frame state


@app.function(image=image, cpu=2, memory=8192, timeout=6 * 60 * 60, max_containers=8,
              volumes={"/data": data_vol, "/cache/hf": hf_vol})
def history_game(game: str, train_frames: int = 20_000, val_frames: int = 1_000, eval_episodes: int = 5) -> dict:
    """``/data/atari/experthist/<game>/``: the ``expert`` recording replayed with the same seed and settings, each
    record also carrying ``history`` (up to ``HISTORY_FRAMES`` previous decision screens, see
    ``laya.atari_data.expert.play``). The baselines are copied from ``/data/atari/expert/<game>/meta.json`` and the
    random baseline episodes replayed only to advance the RNG, so the trajectories, records and frames should match
    ``expert`` exactly; ``replay_check`` reports how many do. Create-only: an existing directory is refused."""
    from laya.atari_data import expert

    t0 = time.time()
    data_vol.reload()
    out = os.path.join(HISTORY_ROOT, game)
    if os.path.exists(out):
        raise FileExistsError("%s exists; experthist is create-only" % out)
    with open(os.path.join(OUT_ROOT, game, "meta.json")) as f:
        one = json.load(f)
    baselines = {k: v for k, v in one.items()
                 if k.startswith(("expert_score", "random_score")) or k in ("eval", "eval_caps")}
    staging = os.path.join(HISTORY_ROOT, ".staging-%s-%d" % (game, int(t0)))  # renamed into place when complete
    meta = expert.generate(game, staging, train_frames, val_frames, eval_episodes, log=lambda s: print(s, flush=True),
                           source="experthist", history=HISTORY_FRAMES, baselines=baselines, create_only=True)
    work = os.path.join(staging, game)
    check = {}
    for split in ("train", "val"):  # the replay against the original recording, record by record
        with open(os.path.join(OUT_ROOT, game, split + ".jsonl")) as f:
            old = [json.loads(line) for line in f if line.strip()]
        with open(os.path.join(work, split + ".jsonl")) as f:
            new = [json.loads(line) for line in f if line.strip()]
        key = lambda r: (r["episode"], r["step"], r["label"], r["taken"], tuple(r["target"]))  # noqa: E731
        same = sum(key(a) == key(b) for a, b in zip(old, new))
        import numpy as np
        from PIL import Image

        pix = 0
        for a, b in list(zip(old, new))[:: max(1, len(new) // 200)]:  # a sample of the images, pixel for pixel
            with Image.open(os.path.join(OUT_ROOT, game, a["image"])) as fa, Image.open(os.path.join(work, b["image"])) as fb:
                pix += bool(np.array_equal(np.asarray(fa.convert("RGB")), np.asarray(fb.convert("RGB"))))
        check[split] = {"old": len(old), "new": len(new), "same_records": same,
                        "images_checked": len(new[:: max(1, len(new) // 200)]), "images_equal": pix}
    meta["replay_check"] = check
    with open(os.path.join(work, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    if os.path.exists(out):
        raise FileExistsError("%s appeared while recording; left the recording in %s" % (out, staging))
    os.rename(work, out)
    os.rmdir(staging)
    _commit(data_vol)
    open(os.path.join(out, "_READY"), "w").close()
    _commit(data_vol)
    meta["seconds"] = round(time.time() - t0)
    print(game, json.dumps(check), "%ds" % meta["seconds"], flush=True)
    return meta


@app.local_entrypoint()
def history(games: str = "Freeway,Breakout", train_frames: int = 20_000, val_frames: int = 1_000):
    names = [g.strip() for g in games.split(",") if g.strip()]
    for res in history_game.map(names, kwargs=dict(train_frames=train_frames, val_frames=val_frames),
                                return_exceptions=True, order_outputs=False):
        if isinstance(res, BaseException):
            print("FAILED", repr(res)[:300], flush=True)
        else:
            print("READY", _row(res), json.dumps(res["replay_check"]), "(%ds)" % res["seconds"], flush=True)
