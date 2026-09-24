"""Modal jobs that convert the Atari subsets of ``jat-project/jat-dataset`` (``laya.atari_data.jat``) into
``/data/atari/jat/<Game>/`` in the format of ``site-docs/reference/atari-data-format.md``. CPU only.

    modal run modal_atari_jat.py::inspect --games pong,breakout     # stream one episode per game, print format facts
    modal run --detach modal_atari_jat.py::convert                  # every game, in parallel
    modal run modal_atari_jat.py::convert --games pong,breakout     # a subset
    modal run modal_atari_jat.py::verify [--game Pong]              # re-load outputs, per-game table

Volumes (created out of band; never ``modal deploy`` this app):
    laya-hf-cache     -> /cache/hf   (HF_HOME)
    laya-datasets     -> /data       (this app writes only under /data/atari/jat/)
"""
import json

import modal

app = modal.App("laya-atari-jat")

hf_vol = modal.Volume.from_name("laya-hf-cache")
data_vol = modal.Volume.from_name("laya-datasets")

image = (
    modal.Image.debian_slim(python_version="3.12")
    # torch/transformers only because ``import laya`` pulls them in; the conversion itself doesn't use them.
    .pip_install("torch==2.14.0", index_url="https://download.pytorch.org/whl/cpu")
    .pip_install("transformers==5.17.0", "safetensors", "ale-py==0.12.1", "gymnasium==1.3.0", "pyarrow",
                 "huggingface_hub[hf_xet]", "numpy", "pillow")
    .env({"HF_HOME": "/cache/hf", "HF_HUB_ENABLE_HF_TRANSFER": "0"})
    .add_local_python_source("laya")
)

OUT_ROOT = "/data/atari/jat"
VOLUMES = {"/cache/hf": hf_vol, "/data": data_vol}
SECRETS = [modal.Secret.from_name("huggingface-thaitea")]


@app.function(image=image)
def jat_names():
    from laya.atari_data.jat import JAT_TO_ALE

    return list(JAT_TO_ALE)


def _games(games: str):
    return jat_names.remote() if not games else [g.strip() for g in games.split(",") if g.strip()]


@app.function(image=image, cpu=2, memory=8192, timeout=20 * 60, volumes=VOLUMES, secrets=SECRETS)
def inspect_one(jat_name: str):
    from laya.atari_data.jat import inspect_game

    return inspect_game(jat_name)


@app.function(image=image, cpu=4, memory=16384, timeout=3 * 60 * 60, volumes=VOLUMES,
              secrets=SECRETS, max_containers=20, retries=1)
def convert_one(jat_name: str):
    import traceback

    from laya.atari_data.jat import convert_game

    try:
        meta = convert_game(jat_name, OUT_ROOT, commit=data_vol.commit)
    except Exception as e:  # report and continue with the other games
        traceback.print_exc()
        return {"jat": jat_name, "error": "%s: %s" % (type(e).__name__, e)}
    return {"jat": jat_name, "game": meta["game"], "train": meta["train"]["records"],
            "val": meta["val"]["records"], "n_actions": len(meta["actions"]),
            "action_max": max(x for x in (meta["train"]["source_action_max"], meta["val"]["source_action_max"])
                              if x is not None),
            "dropped": meta["dropped"]["not_in_minimal_set"], "top": list(meta["train"]["labels"].items())[:3]}


@app.function(image=image, cpu=2, memory=4096, timeout=30 * 60, volumes=VOLUMES)
def verify_all(game: str = "Pong", n_images: int = 8):
    import os
    from collections import Counter

    import numpy as np
    from PIL import Image

    from laya.games import atari_question

    data_vol.reload()
    rows = []
    for g in sorted(os.listdir(OUT_ROOT)):
        d = os.path.join(OUT_ROOT, g)
        if not os.path.exists(os.path.join(d, "_READY")):
            rows.append({"game": g, "ready": False})
            continue
        meta = json.load(open(os.path.join(d, "meta.json")))
        rows.append({"game": g, "ready": True, "train": meta["train"]["records"], "val": meta["val"]["records"],
                     "frame_format": meta["frame_format"], "n_actions": len(meta["actions"]),
                     "top": list(meta["train"]["labels"].items())[:3]})

    # Deep check of one game: re-read the JSONL files, check fields, split disjointness, labels and images.
    d = os.path.join(OUT_ROOT, game)
    meta = json.load(open(os.path.join(d, "meta.json")))
    q = atari_question(game, meta["actions"])["action"]
    recs = {s: [json.loads(l) for l in open(os.path.join(d, s + ".jsonl"))] for s in ("train", "val")}
    for s, rs in recs.items():
        assert all(r["actions"] == meta["actions"] and r["question"] == q and r["source"] == "jat"
                   and 0 <= r["label"] < len(r["actions"]) and "target" not in r for r in rs), s
    tr_eps = {r["episode"] for r in recs["train"]}
    va_eps = {r["episode"] for r in recs["val"]}
    labels = Counter(meta["actions"][r["label"]] for r in recs["train"])
    step = max(1, len(recs["train"]) // n_images)
    imgs = [Image.open(os.path.join(d, r["image"])) for r in recs["train"][::step][:n_images]]
    deep = {"game": game, "train": len(recs["train"]), "val": len(recs["val"]),
            "episodes_train": len(tr_eps), "episodes_val": len(va_eps), "overlap": len(tr_eps & va_eps),
            "labels": dict(labels.most_common()),
            "images_opened": len(imgs), "image_modes": sorted({"%s %dx%d" % (im.mode, *im.size) for im in imgs}),
            "images_non_blank": sum(int(np.asarray(im).std() > 0) for im in imgs),
            "n_image_files": len(os.listdir(os.path.join(d, "images")))}
    return rows, deep


@app.local_entrypoint()
def inspect(games: str = "pong,breakout,mspacman,spaceinvaders,montezumarevenge,seaquest", save: bool = False):
    import pickle

    for r in inspect_one.map(_games(games)):
        sample = {k: r.pop(k) for k in ("sample_stack", "sample_shape", "sample_actions")}
        if save or min(r["stack_shift_match"] or [0]) < 0.9:  # raw uint8 frames + actions, for local analysis
            with open("jat_inspect_%s.pkl" % r["game"], "wb") as fh:
                pickle.dump(dict(sample, minimal_set=r["minimal_set"]), fh)
        schema = r.pop("schema")
        print(json.dumps(r))
    print(schema)


@app.local_entrypoint()
def convert(games: str = ""):
    results = list(convert_one.map(_games(games), return_exceptions=True))
    for r in results:
        print(json.dumps(r if isinstance(r, dict) else {"error": repr(r)}))
    bad = [r for r in results if not isinstance(r, dict) or "error" in r]
    print("converted %d, failed %d" % (len(results) - len(bad), len(bad)))


@app.local_entrypoint()
def verify(game: str = "Pong"):
    rows, deep = verify_all.remote(game)
    print("%-18s %6s %5s %-11s %3s  %s" % ("game", "train", "val", "format", "n", "top train labels"))
    for r in rows:
        if not r["ready"]:
            print("%-18s NOT READY" % r["game"])
            continue
        top = ", ".join("%s %.0f%%" % (k, 100 * v / max(1, r["train"])) for k, v in r["top"])
        print("%-18s %6d %5d %-11s %3d  %s" % (r["game"], r["train"], r["val"], r["frame_format"], r["n_actions"], top))
    print("ready: %d / %d" % (sum(r["ready"] for r in rows), len(rows)))
    print(json.dumps(deep, indent=1))
