"""Modal jobs that convert Atari-HEAD human play (Zenodo record 3451402) into the shared Atari training-data format.

    modal run --detach modal_atari_head.py::main                 # every game: download, convert, verify
    modal run modal_atari_head.py::main --games breakout,freeway # a subset (Atari-HEAD names)
    modal run modal_atari_head.py::verify_all                    # re-check what is on the volume, print a table

Writes only /data/atari/atari_head/<Game>/{train.jsonl, val.jsonl, images/, meta.json, _READY} on the
``laya-datasets`` volume (see site-docs/reference/atari-data-format.md). Conversion logic lives in ``laya/atari_data/atari_head.py``.
Never ``modal deploy`` this app.
"""
import json
import os

import modal

app = modal.App("laya-atari-head")

data_vol = modal.Volume.from_name("laya-datasets")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.14.0", index_url="https://download.pytorch.org/whl/cpu")  # only for `import laya`
    .pip_install("numpy", "pillow", "requests", "ale-py", "gymnasium")
    .add_local_python_source("laya")
)

ROOT = "/data/atari/atari_head"


def _minimal_actions(game: str):
    import ale_py
    import gymnasium as gym

    gym.register_envs(ale_py)
    env = gym.make("ALE/%s-v5" % game)
    actions = env.unwrapped.get_action_meanings()
    env.close()
    return actions


@app.function(image=image, cpu=4, memory=8192, timeout=3 * 60 * 60, volumes={"/data": data_vol}, retries=1)
def convert(dataset_game: str, max_train: int = 20000, max_val: int = 1000) -> dict:
    """Download one game's zip to local disk, write its train/val/images/meta.json to the volume, then _READY."""
    import shutil
    import time

    import requests

    from laya.atari_data.atari_head import FILE_URL, GAMES, convert_game, download

    game = GAMES[dataset_game]
    t0 = time.time()
    meta_csv = requests.get(FILE_URL % "meta_data.csv", timeout=60).text
    zip_path = "/tmp/%s.zip" % dataset_game
    size = download(FILE_URL % (dataset_game + ".zip"), zip_path)
    print("%s: downloaded %.0f MB in %.0fs" % (dataset_game, size / 1e6, time.time() - t0), flush=True)

    final_dir = os.path.join(ROOT, game)
    tmp_dir = final_dir + ".tmp"
    shutil.rmtree(tmp_dir, ignore_errors=True)
    os.makedirs(tmp_dir)
    meta = convert_game(zip_path, dataset_game, tmp_dir, meta_csv, _minimal_actions(game),
                        max_train=max_train, max_val=max_val)
    os.remove(zip_path)
    shutil.rmtree(final_dir, ignore_errors=True)
    os.rename(tmp_dir, final_dir)
    data_vol.commit()
    open(os.path.join(final_dir, "_READY"), "w").close()
    data_vol.commit()
    print("%s: done in %.0fs, train %d val %d dropped %s" % (game, time.time() - t0, meta["train"]["records"],
                                                            meta["val"]["records"], meta["dropped"]), flush=True)
    return meta


@app.function(image=image, cpu=2, memory=4096, timeout=30 * 60, volumes={"/data": data_vol.read_only()})
def verify(game: str, n_images: int = 8) -> dict:
    """Re-load one converted game from the volume and check it against the format spec."""
    import random
    from collections import Counter

    from PIL import Image

    from laya.games import atari_question

    d = os.path.join(ROOT, game)
    row = {"game": game, "ready": os.path.exists(os.path.join(d, "_READY")), "problems": []}
    if not row["ready"]:
        row["problems"].append("no _READY")
        return row
    meta = json.load(open(os.path.join(d, "meta.json")))
    actions = _minimal_actions(game)
    question = atari_question(game, actions)["action"]
    eps = {}
    for split in ("train", "val"):
        recs = [json.loads(l) for l in open(os.path.join(d, split + ".jsonl"))]
        labels = Counter(actions[r["label"]] for r in recs)
        eps[split] = {r["episode"] for r in recs}
        row[split] = len(recs)
        row[split + "_eps"] = len(eps[split])
        if split == "train":
            row["top_labels"] = ", ".join("%s %.0f%%" % (a, 100 * c / max(1, len(recs))) for a, c in labels.most_common(3))
            row["n_labels"] = len(labels)
        if len(recs) != meta[split]["records"] or dict(labels) != meta[split]["labels"]:
            row["problems"].append("%s counts differ from meta.json" % split)
        for r in recs:
            if r["actions"] != actions or r["question"] != question or r["game"] != game or r["source"] != "atari_head" \
                    or not 0 <= r["label"] < len(actions) or "target" in r:
                row["problems"].append("bad record %s" % r["id"])
                break
        for r in random.Random(0).sample(recs, min(n_images, len(recs))):
            im = Image.open(os.path.join(d, r["image"]))
            im.load()
            if (im.mode, im.size) != ("RGB", (160, 210)):
                row["problems"].append("image %s is %s %s" % (r["id"], im.mode, im.size))
    if eps["train"] & eps["val"]:
        row["problems"].append("train/val share trials %s" % sorted(eps["train"] & eps["val"]))
    row["frame_format"] = meta["frame_format"]
    row["dropped"] = meta["dropped"]
    row["license"] = meta["license"]
    row["gaze"] = meta["gaze"]["available"]
    return row


def _table(rows):
    print("%-17s %6s %4s %5s %4s %8s %7s %5s  %-13s %s" % ("game", "train", "eps", "val", "eps", "!minset", "invalid",
                                                            "noimg", "format", "top train labels"))
    for r in rows:
        if "train" not in r:
            print("%-17s %s" % (r["game"], "; ".join(r["problems"])))
            continue
        dr = r["dropped"]
        print("%-17s %6d %4d %5d %4d %8d %7d %5d  %-13s %s%s" % (
            r["game"], r["train"], r["train_eps"], r["val"], r["val_eps"], dr["not_in_minimal_set"],
            dr["invalid_action"], dr["missing_image"], r["frame_format"], r["top_labels"],
            ("  PROBLEMS: " + "; ".join(r["problems"])) if r["problems"] else ""))


@app.function(image=image, timeout=10 * 60, volumes={"/data": data_vol.read_only()})
def ready_games() -> list:
    return sorted(g for g in os.listdir(ROOT) if os.path.exists(os.path.join(ROOT, g, "_READY"))) \
        if os.path.isdir(ROOT) else []


@app.function(image=image)
def dataset_games() -> dict:
    from laya.atari_data.atari_head import GAMES

    return GAMES


@app.local_entrypoint()
def main(games: str = "all", max_train: int = 20000, max_val: int = 1000):
    GAMES = dataset_games.remote()  # importing laya locally would need torch
    todo = sorted(GAMES) if games == "all" else games.split(",")
    failed = {}
    results = list(convert.map(todo, kwargs={"max_train": max_train, "max_val": max_val}, return_exceptions=True))
    for g, res in zip(todo, results):
        if isinstance(res, Exception):
            failed[g] = repr(res)
            print("FAILED %s: %r" % (g, res))
    done = [GAMES[g] for g in todo if g not in failed]
    _table(list(verify.map(done)))
    for g, err in failed.items():
        print("skipped %s: %s" % (g, err))


@app.local_entrypoint()
def verify_all():
    _table(list(verify.map(ready_games.remote())))
