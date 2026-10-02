"""C1: the oracle-zoom ceiling. Does zooming in on the target help a checkpoint at all?

The checkpoint sees every image as one 512 px tile (64 image tokens), so a 2250x1500 photo loses most of a 45 px
object. V*Bench (``craigwu/vstar_bench``) asks multiple-choice questions about such small objects in high-resolution
images and ships the target objects' boxes, which gives a "cheating" upper bound for a glance-then-zoom design: if
even a crop around the true target does not beat the downscaled full image, the zoom branch cannot help whichever
way the zoom location is chosen. Subsets ``direct_attributes`` (colour / material of one object, 4 options) and
``relative_position`` (is A left or right of B, 2 options); questions and options are the benchmark's own
``test_questions.jsonl`` (option order and label as published), boxes the per-image json (``[x, y, w, h]``).
Conditions, paired over the same questions:

* ``full``: the whole image (the processor downsizes it to 512).
* ``oracle_crop``: a square crop around the union of the target boxes, padded by ``--pad`` of the union's size per
  side and at least ``--min-size`` px, clamped to the image (``laya.regions.Region.crop_box``, then squared so the
  tile resize does not change the aspect).
* ``oracle_crop_plus_full``: ``{"images": [full, crop]}`` (two tiles, 128 image tokens).
* ``random_crop``: a crop of the oracle crop's size at a random position that does not overlap the union box
  (the control for "any crop helps").

Prints accuracy (95% Wilson interval), chance, mean confidence, ECE and per-subset accuracy per condition, and the
paired comparison of each condition with ``full`` (flips each way, two-sided exact binomial / McNemar p).
``--out`` writes one JSON row per (item, condition). Exploratory: not a published number.

    python benchmarks/zoom_probe.py --data /path/to/vstar --out /tmp/zoom_probe.jsonl
"""
import argparse
import json
import math
import os
import random
import re
import time

import torch

torch.set_num_threads(1)

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from laya import load_vlm  # noqa: E402
from laya.common import ece_score  # noqa: E402
from laya.regions import Region  # noqa: E402

CHECKPOINT = "thaitea/laya-vision"
REVISION = "f2fe3c12cb6d04c59d8a190250bf3fb40fc828dc"
DATASET = "craigwu/vstar_bench"
DATASET_REVISION = "d9ae62c903da0c98336e85c5ee89cd863b04b4da"
SUBSETS = ("direct_attributes", "relative_position")
CONDITIONS = ("full", "oracle_crop", "oracle_crop_plus_full", "random_crop")


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def binom_two_sided(k, n):
    """Exact two-sided sign test (McNemar on the discordant pairs): P(|X - n/2| >= |k - n/2|), X ~ Bin(n, 1/2)."""
    if n == 0:
        return 1.0
    pmf = [math.comb(n, i) / 2.0 ** n for i in range(n + 1)]
    return min(1.0, sum(p for p in pmf if p <= pmf[k] * (1 + 1e-9)))


def load_items(root):
    items = []
    with open(os.path.join(root, "test_questions.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            if r["category"] not in SUBSETS:
                continue
            lines = r["text"].split("\n")
            opts = re.findall(r"^\(([A-Z])\) (.*)$", r["text"], re.M)
            criteria = [t.strip().rstrip(".") for _, t in opts]
            with open(os.path.join(root, r["image"].rsplit(".", 1)[0] + ".json")) as g:
                meta = json.load(g)
            boxes = [(x, y, x + w, y + h) for x, y, w, h in meta["bbox"]]
            items.append({"id": r["question_id"], "image": r["image"], "subset": r["category"],
                          "question": lines[0], "criteria": criteria,
                          "answer": criteria[[k for k, _ in opts].index(r["label"])],
                          "targets": meta["target_object"], "boxes": boxes})
    return items


def square(box, size):
    """Grow the shorter side of ``box`` to the longer one (centred, shifted to stay inside ``size``)."""
    W, H = size
    x0, y0, x1, y1 = box
    s = min(max(x1 - x0, y1 - y0), W, H)

    def fit(a0, a1, lim):
        c = (a0 + a1) / 2.0
        lo = int(round(c - s / 2.0))
        lo = max(0, min(lo, lim - s))
        return lo, lo + s

    x0, x1 = fit(x0, x1, W)
    y0, y1 = fit(y0, y1, H)
    return x0, y0, x1, y1


def union(boxes):
    return (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))


def overlaps(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def random_box(rng, size, w, h, avoid, tries=2000):
    """A ``w`` x ``h`` box inside ``size`` that does not overlap ``avoid``; None if there is none after ``tries``."""
    W, H = size
    for _ in range(tries):
        x0, y0 = rng.randint(0, W - w), rng.randint(0, H - h)
        b = (x0, y0, x0 + w, y0 + h)
        if not overlaps(b, avoid):
            return b
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="local snapshot of %s @ %s" % (DATASET, DATASET_REVISION))
    ap.add_argument("--n", type=int, default=0, help="first n items (0: all)")
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--pad", type=float, default=0.5)
    ap.add_argument("--min-size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out")
    ap.add_argument("--save-examples", help="directory for the full image and crops of the first item")
    args = ap.parse_args()

    conds = args.conditions.split(",")
    items = load_items(args.data)
    if args.n:
        items = items[:args.n]
    agent = load_vlm(CHECKPOINT, revision=REVISION, device=args.device)
    out = open(args.out, "w") if args.out else None
    print("checkpoint %s @ %s; %s @ %s; n=%d; pad=%.2f min_size=%d" % (
        CHECKPOINT, agent.source["revision"], DATASET, DATASET_REVISION, len(items), args.pad, args.min_size))
    res = {c: {} for c in conds}
    t0 = time.perf_counter()
    for i, it in enumerate(items):
        img = Image.open(os.path.join(args.data, it["image"])).convert("RGB")
        ub = union(it["boxes"])
        crop_box = square(Region(ub).crop_box(img.size, pad=args.pad, min_size=args.min_size), img.size)
        cw, ch = crop_box[2] - crop_box[0], crop_box[3] - crop_box[1]
        rng = random.Random("%s-%d" % (it["id"], args.seed))
        rand_box = random_box(rng, img.size, cw, ch, ub)
        crop = img.crop(crop_box)
        if i == 0 and args.save_examples:
            img.save(os.path.join(args.save_examples, "full.png"))
            crop.save(os.path.join(args.save_examples, "oracle_crop.png"))
            if rand_box:
                img.crop(rand_box).save(os.path.join(args.save_examples, "random_crop.png"))
        q = {"type": "choice", "instructions": it["question"], "criteria": it["criteria"]}
        for c in conds:
            if c == "full":
                state = {"image": img}
            elif c == "oracle_crop":
                state = {"image": crop}
            elif c == "oracle_crop_plus_full":
                state = {"images": [img, crop]}
            else:
                if rand_box is None:
                    continue
                state = {"image": img.crop(rand_box)}
            a = agent.predict(state, {"q": q})["answers"]["q"]
            row = {"id": it["id"], "subset": it["subset"], "condition": c, "image": it["image"],
                   "image_size": list(img.size), "crop_box": list(rand_box if c == "random_crop" else crop_box)
                   if c != "full" else None, "union_box": list(ub), "answer": it["answer"], "choice": a["choice"],
                   "correct": a["choice"] == it["answer"], "probabilities": a["probabilities"],
                   "conf": max(a["probabilities"].values()), "n_options": len(it["criteria"])}
            res[c][it["id"]] = row
            if out:
                out.write(json.dumps(row) + "\n")
                out.flush()
        if (i + 1) % 10 == 0:
            print("  %d/%d items, %.1f s/item" % (i + 1, len(items), (time.perf_counter() - t0) / (i + 1)), flush=True)
    if out:
        out.close()

    print("\n%-22s %4s %6s %17s %6s %6s %6s %8s %8s" % ("condition", "n", "acc", "95% CI", "chance", "conf", "ECE",
                                                     "attr", "relpos"))
    for c in conds:
        rows = list(res[c].values())
        if not rows:
            continue
        corr = np.array([r["correct"] for r in rows], dtype=float)
        conf = np.array([r["conf"] for r in rows])
        k, n = int(corr.sum()), len(rows)
        lo, hi = wilson(k, n)
        chance = float(np.mean([1.0 / r["n_options"] for r in rows]))
        sub = []
        for s in SUBSETS:
            sc = [r["correct"] for r in rows if r["subset"] == s]
            sub.append("%.3f/%d" % (np.mean(sc), len(sc)) if sc else "-")
        print("%-22s %4d %6.3f  [%5.3f, %5.3f] %6.3f %6.3f %6.3f %8s %8s" % (
            c, n, k / n, lo, hi, chance, conf.mean(), ece_score(conf, corr), *sub))
    if "full" in conds:
        print("\npaired vs full: gained = wrong under full, right under condition")
        for c in conds:
            if c == "full":
                continue
            ids = [i for i in res[c] if i in res["full"]]
            gain = sum(res[c][i]["correct"] and not res["full"][i]["correct"] for i in ids)
            loss = sum(res["full"][i]["correct"] and not res[c][i]["correct"] for i in ids)
            d = np.mean([res[c][i]["correct"] for i in ids]) - np.mean([res["full"][i]["correct"] for i in ids])
            print("  %-22s n=%d  delta=%+.3f  gained=%d lost=%d  p=%.4g" % (c, len(ids), d, gain, loss,
                                                                          binom_two_sided(gain, gain + loss)))


if __name__ == "__main__":
    main()
