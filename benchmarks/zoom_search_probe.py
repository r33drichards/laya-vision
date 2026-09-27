"""C2: can the checkpoint find where to zoom? Glance-then-zoom on V*Bench without the oracle box.

``benchmarks/zoom_probe.py`` (C1) showed that on V*Bench ``direct_attributes`` the checkpoint goes from 22% on the
downscaled full image to 75% on a crop around the true target: the answer is there once the model looks in the right
place. This probe replaces the oracle box with the model's own search. Every variant ends at the same zoom level, a
cell of a 4x4 grid over the image grown by ``--grow`` per side, and answers the benchmark question on that crop
alone, so the variants differ only in how the cell is chosen:

* ``oracle_tile``: the cell containing the target box's centre (the ceiling at this zoom level, no search).
* ``choice``: two ``choice`` questions, "Where in the image is the <target>?" over the four quadrants, on the full
  image and then on the chosen quadrant (2 calls, 64 image tokens each).
* ``hier``: ``noul`` "Is there a <target> in this image?" on each quadrant, then on each quarter of the best one
  (8 crops scored, ZoomEye-style).
* ``grid``: the same ``noul`` on all 16 cells, the best one is the zoom (16 crops scored).

Search crops are scored in one batched call per step (``laya.search.score_states``, checkpoint temperatures, one
option order), the answer with ``VLMAgent.predict``. Reports accuracy (95% Wilson), hit rate (the chosen cell holds
the target box's centre), accuracy given a hit / a miss, calls and image tokens per question, ECE, and the paired
comparison with the full-image answer from C1's rows if ``--c1-rows`` is given. Only ``direct_attributes`` (the
subset where C1 showed zoom helps; ``relative_position`` stayed at chance even with the oracle crop).
Exploratory: not a published number.

    python benchmarks/zoom_search_probe.py --data /path/to/vstar --variant grid --out /tmp/c2_grid.jsonl
"""
import argparse
import json
import os
import sys
import time

import torch

torch.set_num_threads(1)

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from zoom_probe import CHECKPOINT, DATASET, DATASET_REVISION, REVISION, binom_two_sided, load_items, wilson  # noqa: E402

from laya import load_vlm  # noqa: E402
from laya.common import ece_score  # noqa: E402
from laya.search import score_states  # noqa: E402

VARIANTS = ("oracle_tile", "choice", "hier", "grid", "hier3", "oracle_centred")
QUADS = ["top left", "top right", "bottom left", "bottom right"]
TOKENS_PER_VIEW = 64


def sub(box, g, i, j):
    """Cell (row i, column j) of a g x g grid over ``box``."""
    x0, y0, x1, y1 = box
    w, h = (x1 - x0) / g, (y1 - y0) / g
    return (x0 + j * w, y0 + i * h, x0 + (j + 1) * w, y0 + (i + 1) * h)


def quads(box):
    return [sub(box, 2, i, j) for i in range(2) for j in range(2)]  # top left, top right, bottom left, bottom right


def grow(box, frac, size):
    x0, y0, x1, y1 = box
    dx, dy = (x1 - x0) * frac, (y1 - y0) * frac
    W, H = size
    return (int(max(0, x0 - dx)), int(max(0, y0 - dy)), int(min(W, x1 + dx)), int(min(H, y1 + dy)))


def centred(pt, side, size):
    """A ``side`` x ``side`` box centred on ``pt``, shifted to stay inside ``size``."""
    W, H = size
    side = int(min(side, W, H))
    x0 = int(max(0, min(pt[0] - side / 2.0, W - side)))
    y0 = int(max(0, min(pt[1] - side / 2.0, H - side)))
    return (x0, y0, x0 + side, y0 + side)


def contains(box, pt):
    return box[0] <= pt[0] < box[2] and box[1] <= pt[1] < box[3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="local snapshot of %s @ %s" % (DATASET, DATASET_REVISION))
    ap.add_argument("--variant", choices=VARIANTS, required=True)
    ap.add_argument("--n", type=int, default=0, help="first n items (0: all)")
    ap.add_argument("--grow", type=float, default=0.25, help="grow the final cell by this fraction per side")
    ap.add_argument("--c1-rows", help="zoom_probe.py rows, for the paired comparison with the full image")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--oracle-grid", type=int, default=4, help="grid size for oracle_tile (8: one level below hier)")
    ap.add_argument("--out")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    items = [it for it in load_items(args.data) if it["subset"] == "direct_attributes"]
    if args.n:
        items = items[:args.n]
    agent = load_vlm(CHECKPOINT, revision=REVISION, device=args.device)
    out = open(args.out, "w") if args.out else None
    print("C2 %s: checkpoint %s @ %s; %s @ %s direct_attributes; n=%d; grow=%.2f" % (
        args.variant, CHECKPOINT, agent.source["revision"], DATASET, DATASET_REVISION, len(items), args.grow))

    def noul_scores(crops, target):
        q = {"type": "noul", "instructions": "Is there a %s in this image?" % target}
        return score_states(agent, crops, q, batch_size=8)["probs"][:, 1]

    def where(img, target):
        q = {"type": "choice", "instructions": "Where in the image is the %s?" % target,
             "criteria": {p: "the %s is in the %s of the image" % (target, p) for p in QUADS}}
        a = agent.predict({"image": img}, {"q": q})["answers"]["q"]
        return QUADS.index(a["choice"]), a["probabilities"]

    rows = []
    t0 = time.perf_counter()
    for n_done, it in enumerate(items, 1):
        img = Image.open(os.path.join(args.data, it["image"])).convert("RGB")
        W, H = img.size
        full = (0, 0, W, H)
        target = it["targets"][0]
        tb = it["boxes"][0]
        centre = ((tb[0] + tb[2]) / 2.0, (tb[1] + tb[3]) / 2.0)
        trace, calls = {}, 0
        if args.variant == "oracle_tile":
            g = args.oracle_grid
            cell = next(sub(full, g, i, j) for i in range(g) for j in range(g) if contains(sub(full, g, i, j), centre))
        elif args.variant == "choice":
            k1, p1 = where(img, target)
            q1 = quads(full)[k1]
            k2, p2 = where(img.crop(tuple(int(v) for v in q1)), target)
            cell = quads(q1)[k2]
            calls, trace = 2, {"step1": p1, "step2": p2}
        elif args.variant == "hier":
            level1 = quads(full)
            s1 = noul_scores([img.crop(grow(b, 0.1, img.size)) for b in level1], target)
            q1 = level1[int(s1.argmax())]
            level2 = quads(q1)
            s2 = noul_scores([img.crop(grow(b, 0.1, img.size)) for b in level2], target)
            cell = level2[int(s2.argmax())]
            calls, trace = 8, {"step1": [round(float(v), 4) for v in s1], "step2": [round(float(v), 4) for v in s2]}
        elif args.variant in ("hier3", "oracle_centred"):
            side = 1.5 * max(W, H) / 8  # an 8x8 cell's long side, grown 25% per side
            if args.variant == "oracle_centred":
                point = centre
            else:
                box, steps = full, []
                for _ in range(3):
                    level = quads(box)
                    sc = noul_scores([img.crop(grow(b, 0.1, img.size)) for b in level], target)
                    steps.append([round(float(v), 4) for v in sc])
                    box = level[int(sc.argmax())]
                # centre the crop on the score-weighted centroid of the last level's four cells
                w = np.maximum(sc - sc.min(), 1e-6)
                cs = [((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0) for b in level]
                point = (float(np.dot(w, [c[0] for c in cs]) / w.sum()), float(np.dot(w, [c[1] for c in cs]) / w.sum()))
                calls, trace = 12, {"steps": steps, "point": [round(point[0]), round(point[1])]}
            cell = centred(point, side, img.size)
        else:
            cells = [sub(full, 4, i, j) for i in range(4) for j in range(4)]
            s = noul_scores([img.crop(grow(b, 0.1, img.size)) for b in cells], target)
            cell = cells[int(s.argmax())]
            calls, trace = 16, {"scores": [round(float(v), 4) for v in s]}
        final = cell if args.variant in ("hier3", "oracle_centred") else grow(cell, args.grow, img.size)
        q = {"type": "choice", "instructions": it["question"], "criteria": it["criteria"]}
        a = agent.predict({"image": img.crop(final)}, {"q": q})["answers"]["q"]
        row = {"id": it["id"], "variant": args.variant, "target": target, "answer": it["answer"], "choice": a["choice"],
               "correct": a["choice"] == it["answer"], "conf": max(a["probabilities"].values()),
               "probabilities": a["probabilities"], "hit": contains(final, centre), "cell_hit": contains(cell, centre),
               "final_box": list(final), "search_calls": calls, "image_tokens": (calls + 1) * TOKENS_PER_VIEW,
               "trace": trace, "n_options": len(it["criteria"])}
        rows.append(row)
        if out:
            out.write(json.dumps(row) + "\n")
            out.flush()
        if n_done % 10 == 0:
            print("  %d/%d, %.1f s/item, acc so far %.3f" % (n_done, len(items), (time.perf_counter() - t0) / n_done,
                                                           np.mean([r["correct"] for r in rows])), flush=True)
    if out:
        out.close()

    corr = np.array([r["correct"] for r in rows], dtype=float)
    hit = np.array([r["cell_hit"] for r in rows])
    k, n = int(corr.sum()), len(rows)
    lo, hi = wilson(k, n)
    print("\nvariant %s  n=%d" % (args.variant, n))
    print("  accuracy        %.3f  [%.3f, %.3f]  (chance %.3f)" % (k / n, lo, hi, np.mean([1 / r["n_options"] for r in rows])))
    print("  cell hit rate   %.3f   (final crop holds the target centre: %.3f)" % (hit.mean(), np.mean([r["hit"] for r in rows])))
    if hit.any():
        print("  acc | hit       %.3f  (n=%d)" % (corr[hit].mean(), hit.sum()))
    if (~hit).any():
        print("  acc | miss      %.3f  (n=%d)" % (corr[~hit].mean(), (~hit).sum()))
    print("  mean conf %.3f  ECE %.3f" % (np.mean([r["conf"] for r in rows]),
                                          ece_score(np.array([r["conf"] for r in rows]), corr)))
    print("  image tokens / question %d  (search calls %d)" % (rows[0]["image_tokens"], rows[0]["search_calls"]))
    print("  wall %.2f s/item" % ((time.perf_counter() - t0) / n))
    if args.c1_rows:
        base = {}
        with open(args.c1_rows) as f:
            for line in f:
                r = json.loads(line)
                if r["condition"] == "full":
                    base[r["id"]] = r["correct"]
        pairs = [(base[r["id"]], r["correct"]) for r in rows if r["id"] in base]
        gain = sum(1 for b, c in pairs if c and not b)
        loss = sum(1 for b, c in pairs if b and not c)
        print("  vs full image (C1): full acc %.3f, gained %d, lost %d, McNemar p=%.2g" % (
            np.mean([b for b, _ in pairs]), gain, loss, binom_two_sided(gain, gain + loss)))


if __name__ == "__main__":
    main()
