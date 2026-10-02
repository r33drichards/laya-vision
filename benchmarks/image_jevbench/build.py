"""Rebuild the reproducible part of Image JevBench (https://benchmarkheaven.com/image-jev-bench) from pinned sources.

The benchmark's 228-item public split is not released (the page publishes aggregates only). What its public repository
does publish, at ``MMC_REVISION``:

- ``items-extended-real.json``: the 128 items of its multimodal preview, with question, options, gold answer and the
  upstream dataset row of each, but no images. The 80 image-reasoning items (CLEVR-HOPE, Geometry3K, ArxivQA, FinQA,
  20 each) are rebuilt here from the upstream rows. The 48 computer/browser-use items (ScreenSpot, Mind2Web) are
  skipped: their five click markers were drawn at positions that are not published.
- 8 public examples with the benchmark's own images (``public/image-jev/examples``).

Each upstream row is fetched through the Hugging Face datasets-server at a pinned commit (checked in the asset URL),
and its question and answer are checked against the item before it is kept. FinQA's table is rendered the way the
benchmark's published FinQA example is (``cell | cell`` lines from the ``table`` field); the other images are the
source images. Writes ``<work>/items.json`` and ``<work>/img/``; needs ``node`` to read the examples module.

    python benchmarks/image_jevbench/build.py --work /tmp/image-jevbench
"""
import argparse
import hashlib
import io
import json
import re
import subprocess
import time
import urllib.parse
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

MMC_REPO = "fstandhartinger/model-market-comparison"
MMC_REVISION = "33b337e3bb592abdaf1b39143e8f7dcd8030bf67"
ITEMS_PATH = "data/raw/benchmarks/jevbench/multimodal-preview/source/items-extended-real.json"
EXAMPLES_PATH = "lib/image-jev-public-examples.mjs"

SOURCES = {  # dataset -> (Hub repo, config, pinned commit)
    "CLEVR-HOPE": ("user9000/CLEVR-HOPE", "HOP00", "2d0c19f23dc5de19f83f4e9928a793c95632327a"),
    "Geometry3K": ("hiyouga/geometry3k", "default", "fd21e533e1e50d0662a2bf7b223e60511bd5f8b7"),
    "ArxivQA": ("mm-eval/ArxivQA", "default", "e13bb29ff5b2c2ae5f05cc0b1ecc2b24a6833c81"),
    "FinQA": ("bevaya/FinQA", "default", "3d6a736bc67e06bc15fbf3618d88204a57c5b25e"),
}


def get(url: str, tries: int = 5) -> bytes:
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return r.read()
        except Exception:  # the datasets-server rate-limits now and then
            if i == tries - 1:
                raise
            time.sleep(2 ** (i + 1))


def mmc_file(path: str) -> bytes:
    return get(f"https://raw.githubusercontent.com/{MMC_REPO}/{MMC_REVISION}/{path}")


def source_row(dataset: str, split: str, idx: int) -> dict:
    repo, config, _ = SOURCES[dataset]
    q = urllib.parse.urlencode({"dataset": repo, "config": config, "split": split, "offset": idx, "length": 1})
    return json.loads(get(f"https://datasets-server.huggingface.co/rows?{q}"))["rows"][0]["row"]


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace("<image>", "")).strip().lower()


def same_answer(a: str, b: str) -> bool:
    """FinQA's options drop trailing zeros ("11%" for the source's "11.0%")."""
    a, b = norm(a), norm(b)
    if a == b:
        return True
    try:
        return abs(float(a.rstrip("%").replace(",", "")) - float(b.rstrip("%").replace(",", ""))) < 1e-6
    except ValueError:
        return False


def render_table(table, path: Path) -> None:
    """Plain ``cell | cell`` lines on white, 640 px wide: the layout of the benchmark's published FinQA example."""
    font = ImageFont.load_default(size=11)
    top, lh = 14, 22
    img = Image.new("RGB", (640, top + lh * len(table) + 16), "white")
    dr = ImageDraw.Draw(img)
    for i, r in enumerate(table):
        dr.text((12, top + i * lh), " | ".join(str(c) for c in r), fill="black", font=font)
    img.save(path)


def fetch_image(url: str, dataset: str, path: Path) -> None:
    assert SOURCES[dataset][2] in url, f"datasets-server served {url}, not revision {SOURCES[dataset][2]}"
    Image.open(io.BytesIO(get(url))).convert("RGB").save(path)


def examples(work: Path) -> list:
    src = work / "image-jev-public-examples.mjs"
    src.write_bytes(mmc_file(EXAMPLES_PATH))
    js = ("import(process.argv[1]).then(m => console.log(JSON.stringify(m.PUBLIC_IMAGE_JEV_EXAMPLES)))")
    return json.loads(subprocess.run(["node", "-e", js, src.resolve().as_uri()], check=True, capture_output=True,
                                     text=True).stdout)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--work", required=True, type=Path)
    args = ap.parse_args()
    (args.work / "img").mkdir(parents=True, exist_ok=True)

    out = []
    for it in json.loads(mmc_file(ITEMS_PATH)):
        ds = it["dataset"]
        if ds not in SOURCES:
            continue
        r = source_row(ds, it["split"], it["source_row"])
        path = args.work / "img" / f"{it['id']}.png"
        gold_text = it["rubric"]["criteria"][it["gold"]]
        if ds == "CLEVR-HOPE":
            question = r["query"]
            assert it["gold"] == r["answer"], it["id"]
            fetch_image(r["image"]["src"], ds, path)
        elif ds == "Geometry3K":
            question = r["problem"]
            assert same_answer(gold_text, r["answer"]), it["id"]
            fetch_image(r["images"][0]["src"], ds, path)
        elif ds == "ArxivQA":
            question = json.loads(r["messages"])[0]["question"]
            assert it["gold"] == r["answer"], it["id"]
            fetch_image(r["media"][0]["src"], ds, path)
        else:  # FinQA
            question = r["question"]
            assert same_answer(gold_text, r["answer"]), it["id"]
            render_table(r["table"], path)
        assert norm(question) == norm(it["rubric"]["instructions"]), it["id"]
        out.append({
            "id": it["id"], "set": "rebuilt", "dataset": ds, "image": str(path.relative_to(args.work)),
            "image_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "question": it["rubric"]["instructions"], "options": it["rubric"]["criteria"], "gold": it["gold"],
            "source": {"repo": SOURCES[ds][0], "config": SOURCES[ds][1], "revision": SOURCES[ds][2],
                       "split": it["split"], "row": it["source_row"]},
        })
        print(it["id"], ds, flush=True)

    for e in examples(args.work):
        raw = mmc_file("public" + e["image"])
        path = args.work / "img" / f"example-{e['key']}.png"
        Image.open(io.BytesIO(raw)).convert("RGB").save(path)
        out.append({
            "id": f"example:{e['key']}", "set": "public_examples", "dataset": e["category"],
            "source_item_id": e["sourceItemId"], "image": str(path.relative_to(args.work)),
            "image_sha256": hashlib.sha256(raw).hexdigest(), "question": e["question"],
            "options": {o["label"]: o["text"] for o in e["options"]}, "gold": e["correctLabel"],
            "source": {"repo": MMC_REPO, "revision": MMC_REVISION, "path": "public" + e["image"]},
        })
    (args.work / "items.json").write_text(json.dumps(out, indent=1))
    print(len(out), "items ->", args.work / "items.json")


if __name__ == "__main__":
    main()
