"""B1: can a checkpoint read Set-of-Mark numbers? A synthetic probe for ``laya.regions.select``.

Every image is 512x512 (the checkpoint's input size, so nothing is resized) with coloured squares on a grey
background. Conditions:

* ``digit_large`` / ``digit_small``: one digit 1-9 drawn at 200 px / 18 px (the default mark tag size at 512 px).
  "What number is written in the image?" over "1".."9". Can the model read a digit at all, and at mark size?
* ``marks_N`` (N = 2, 4, 8): N squares of distinct colours, each outlined and numbered in black (the numbers are a
  random permutation, so they carry no position), "Which box contains the <colour> square?" over ``box k``. The
  option texts are only the numbers: the answer has to come from reading the mark next to the right square.
* ``marks_4_large``: as ``marks_4`` with 64 px tags.
* ``position_4``: 4 squares in the 4 quadrants, no marks, options "the square at the top left" etc. The same
  question answered by position words instead of marks, as a control for "does it find the colour at all".

Prints accuracy (with a 95% Wilson interval), chance, mean confidence, ECE and how often the first option was
chosen, per condition; ``--out`` writes one JSON row per question. Exploratory: not a published number.

    python benchmarks/marks_probe.py --n 200 --out /tmp/marks_probe.jsonl
"""
import argparse
import json
import math
import random
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from laya import load_vlm
from laya.common import ece_score
from laya.regions import Region, draw_marks, select_question

CHECKPOINT = "thaitea/laya-vision"
REVISION = "f2fe3c12cb6d04c59d8a190250bf3fb40fc828dc"
SIZE = 512
BG = (200, 200, 200)
COLORS = {"red": (220, 30, 30), "green": (30, 170, 50), "blue": (30, 70, 220), "yellow": (240, 220, 20),
          "purple": (130, 40, 170), "orange": (250, 140, 20), "pink": (250, 120, 190), "brown": (120, 70, 30)}
QUADRANTS = ["top left", "top right", "bottom left", "bottom right"]


def font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def digit_item(rng, size):
    img = Image.new("RGB", (SIZE, SIZE), BG)
    d = rng.randint(1, 9)
    draw = ImageDraw.Draw(img)
    f = font(size)
    x0, y0, x1, y1 = draw.textbbox((0, 0), str(d), font=f)
    x, y = rng.randint(10, SIZE - (x1 - x0) - 10), rng.randint(10, SIZE - (y1 - y0) - 10)
    draw.text((x - x0, y - y0), str(d), fill=(0, 0, 0), font=f)
    q = {"type": "choice", "instructions": "What number is written in the image?",
         "criteria": [str(i) for i in range(1, 10)]}
    return img, q, str(d)


def squares(rng, n, cells):
    """n squares of distinct colours in n distinct cells of a 4x4 grid (or the given cell list)."""
    names = rng.sample(list(COLORS), n)
    img = Image.new("RGB", (SIZE, SIZE), BG)
    draw = ImageDraw.Draw(img)
    regs = []
    for name, (cx, cy, cw) in zip(names, cells):
        s = int(cw * 0.45)
        x0 = cx + rng.randint(int(cw * 0.25), cw - s - 8)
        y0 = cy + rng.randint(int(cw * 0.25), cw - s - 8)
        draw.rectangle([x0, y0, x0 + s, y0 + s], fill=COLORS[name])
        regs.append(Region((x0 - 6, y0 - 6, x0 + s + 6, y0 + s + 6), name))
    return img, regs, names


def marks_item(rng, n, font_size=None):
    grid = [(c * 128, r * 128, 128) for r in range(4) for c in range(4)]
    img, regs, names = squares(rng, n, rng.sample(grid, n))
    nums = rng.sample(range(1, n + 1), n)
    order = sorted(range(n), key=lambda i: nums[i])  # options in number order
    img = draw_marks(img, regs, nums, font_size=font_size, color=(0, 0, 0))
    target = rng.randrange(n)
    q = select_question([regs[i] for i in order], [nums[i] for i in order],
                        "Which box contains the %s square?" % names[target], describe=False)
    return img, q, str(nums[target])


def position_item(rng):
    cells = [(0, 0, 256), (256, 0, 256), (0, 256, 256), (256, 256, 256)]
    img, regs, names = squares(rng, 4, cells)
    target = rng.randrange(4)
    q = {"type": "choice", "instructions": "Which square is %s?" % names[target],
         "criteria": {p: "the square at the %s" % p for p in QUADRANTS}}
    return img, q, QUADRANTS[target]


CONDITIONS = {
    "digit_large": lambda rng: digit_item(rng, 200),
    "digit_small": lambda rng: digit_item(rng, 18),
    "marks_2": lambda rng: marks_item(rng, 2),
    "marks_4": lambda rng: marks_item(rng, 4),
    "marks_8": lambda rng: marks_item(rng, 8),
    "marks_4_large": lambda rng: marks_item(rng, 4, font_size=64),
    "position_4": position_item,
}


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--n-permutations", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out")
    ap.add_argument("--save-examples", help="directory for the first image of each condition")
    args = ap.parse_args()

    agent = load_vlm(CHECKPOINT, revision=REVISION, device=args.device)
    out = open(args.out, "w") if args.out else None
    print("checkpoint %s @ %s, n=%d per condition, n_permutations=%d" % (CHECKPOINT, agent.source["revision"],
                                                                        args.n, args.n_permutations))
    print("%-14s %6s %17s %6s %6s %6s %7s %6s" % ("condition", "acc", "95% CI", "chance", "conf", "ECE", "first%", "s/it"))
    for cond in args.conditions.split(","):
        rng = random.Random("%s-%d" % (cond, args.seed))
        conf, correct, first = [], [], []
        t0 = time.perf_counter()
        for i in range(args.n):
            img, q, target = CONDITIONS[cond](rng)
            if i == 0 and args.save_examples:
                img.save("%s/%s.png" % (args.save_examples, cond))
            a = agent.predict({"image": img}, {"q": q}, n_permutations=args.n_permutations)["answers"]["q"]
            keys = list(a["probabilities"])
            conf.append(max(a["probabilities"].values()))
            correct.append(a["choice"] == target)
            first.append(a["choice"] == keys[0])
            if out:
                out.write(json.dumps({"condition": cond, "i": i, "target": target, "choice": a["choice"],
                                      "probabilities": a["probabilities"]}) + "\n")
        k, n = int(sum(correct)), len(correct)
        lo, hi = wilson(k, n)
        chance = 1.0 / len(keys)
        print("%-14s %6.3f  [%5.3f, %5.3f] %6.3f %6.3f %6.3f %6.1f%% %6.2f" % (
            cond, k / n, lo, hi, chance, float(np.mean(conf)),
            ece_score(np.array(conf), np.array(correct, dtype=float)), 100 * np.mean(first),
            (time.perf_counter() - t0) / n), flush=True)
    if out:
        out.close()


if __name__ == "__main__":
    main()
