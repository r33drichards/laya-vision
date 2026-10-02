"""Score a laya-vision checkpoint on the items ``build.py`` wrote, one ``choice`` question per item.

The benchmark's items are already typed questions (``instructions`` + ``criteria`` keyed by option label), so they
go to ``predict`` unchanged. Writes ``<out>.predictions.jsonl.gz`` (one row per item) and ``<out>.meta.json``;
refuses to overwrite either.

    python benchmarks/image_jevbench/run.py --work /tmp/image-jevbench \
        --out eval-results/image-jevbench-laya-vision-8b318c9
"""
import argparse
import gzip
import json
import statistics
import time
from pathlib import Path

from PIL import Image

import laya

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("--work", required=True, type=Path)
ap.add_argument("--out", required=True, help="output prefix")
ap.add_argument("--model", default="thaitea/laya-vision")
ap.add_argument("--revision", default="8b318c99d7ad3ce19c24369263463882eada9d1e")
ap.add_argument("--device", default=None)
args = ap.parse_args()
pred_path, meta_path = Path(args.out + ".predictions.jsonl.gz"), Path(args.out + ".meta.json")
for p in (pred_path, meta_path):
    if p.exists():
        raise SystemExit(f"{p} exists; results are create-only, pass a new --out")

items = json.loads((args.work / "items.json").read_text())
agent = laya.load_vlm(args.model, revision=args.revision, device=args.device)
rows = []
for it in items:
    img = Image.open(args.work / it["image"]).convert("RGB")
    t0 = time.perf_counter()
    res = agent.predict({"image": img}, {"q": {"type": "choice", "instructions": it["question"],
                                               "criteria": it["options"]}})
    dt = time.perf_counter() - t0
    a = res["answers"]["q"]
    rows.append({"id": it["id"], "set": it["set"], "dataset": it["dataset"], "gold": it["gold"], "pred": a["choice"],
                 "correct": a["choice"] == it["gold"], "probabilities": a["probabilities"],
                 "n_options": len(it["options"]), "truncated": a.get("truncated"), "latency_s": round(dt, 4),
                 "image_sha256": it["image_sha256"]})
    print(it["id"], a["choice"], it["gold"], "ok" if rows[-1]["correct"] else "miss", f"{dt:.2f}s", flush=True)

with gzip.open(pred_path, "wt") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
meta_path.write_text(json.dumps({
    "benchmark": "Image JevBench, reproducible subset (benchmarks/image_jevbench)",
    "model": args.model, "revision": args.revision, "n": len(rows),
    "items_sha256": __import__("hashlib").sha256((args.work / "items.json").read_bytes()).hexdigest(),
    "median_latency_s": statistics.median(r["latency_s"] for r in rows),
    "provenance": res.get("provenance"),
}, indent=1, default=str) + "\n")
print("wrote", pred_path, meta_path)
