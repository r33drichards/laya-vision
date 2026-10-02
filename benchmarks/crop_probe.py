"""A-bundle: does a checkpoint know what a ground-truth object crop is, is it calibrated, and does it beat SigLIP?

Crops of COCO val2017 ground-truth boxes (``detection-datasets/coco`` split ``val``, streamed and pinned), for 8
common classes. Boxes with area < 32x32 px are dropped; the mirror has no ``iscrowd`` field, so crowd boxes cannot
be filtered here. At most ``--per-image`` crops per class per image, ``--n`` crops per class, cropped with
``Region(box).crop(image, pad=0.1, min_size=32)`` (what ``laya.regions.map_regions`` does). Per crop:

* **verification** (``noul``): "Is this a <true class>?" (positive) and "Is this a <random other of the 8>?"
  (hard negative). AUROC over the 2n*8 answers, accuracy at 0.5, ECE on confidence ``max(p, 1-p)``, mean P(yes)
  on positives / negatives.
* **8-way** (``choice`` over the 8 class names): accuracy (95% Wilson interval), mean confidence, ECE, top
  confusions.

Baseline: ``google/siglip-base-patch16-224`` (pinned), prompt "a photo of a <class>.". Verification uses
``sigmoid(logit)`` (SigLIP is trained with a sigmoid loss, so this is its native probability); 8-way uses the
softmax over the 8 prompts' logits. The three Laya questions are one ``predict`` call per crop, SigLIP is one
image forward per crop against the 8 prompt embeddings (computed once); latency is wall time of that per crop with ``torch.set_num_threads(1)``.

``--out`` writes one JSON row per crop. Exploratory: not a published number.

    python benchmarks/crop_probe.py --n 40 --out /tmp/crop_probe.jsonl
"""
import argparse
import json
import math
import random
import time
from collections import Counter, defaultdict

import numpy as np
import torch

from laya import load_vlm
from laya.common import ece_score
from laya.regions import Region

CHECKPOINT = "thaitea/laya-vision"
REVISION = "f2fe3c12cb6d04c59d8a190250bf3fb40fc828dc"
DATASET = "detection-datasets/coco"
DATASET_REVISION = "cf0b22332314a937e9dc8a1957b21725430bb41d"
SIGLIP = "google/siglip-base-patch16-224"
SIGLIP_REVISION = "7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed"
CLASSES = ["person", "car", "dog", "cat", "chair", "bottle", "bird", "bicycle"]
MIN_AREA = 32 * 32


def article(c):
    return "an" if c[0] in "aeiou" else "a"


def noul_q(c):
    return {"type": "noul", "instructions": "Is this %s %s?" % (article(c), c)}


def feats(x):
    """``get_*_features`` returns a tensor on older transformers and an output with ``pooler_output`` on newer."""
    return x if torch.is_tensor(x) else x.pooler_output


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def auroc(scores, labels):
    """Mann-Whitney AUROC (ties count half)."""
    s, y = np.asarray(scores, float), np.asarray(labels, bool)
    pos, neg = s[y], s[~y]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(gt / (len(pos) * len(neg)))


def sample_crops(n, per_image, seed, max_images):
    """Stream COCO val and keep up to n ground-truth crops per class (in stream order)."""
    from datasets import load_dataset

    ds = load_dataset(DATASET, split="val", streaming=True, revision=DATASET_REVISION)
    names = ds.features["objects"]["category"].feature.names
    rng = random.Random(seed)
    counts = Counter()
    items = []
    for i, row in enumerate(ds):
        if i >= max_images or all(counts[c] >= n for c in CLASSES):
            break
        objs = row["objects"]
        by_cls = defaultdict(list)
        for bid, cat, box in zip(objs["bbox_id"], objs["category"], objs["bbox"]):
            c = names[cat]
            x0, y0, x1, y1 = box
            if c in CLASSES and (x1 - x0) * (y1 - y0) >= MIN_AREA:
                by_cls[c].append((bid, box))
        if not by_cls:
            continue
        image = row["image"].convert("RGB")
        for c, objs_c in by_cls.items():
            rng.shuffle(objs_c)
            for bid, box in objs_c[:per_image]:
                if counts[c] >= n:
                    break
                counts[c] += 1
                items.append({"image_id": row["image_id"], "bbox_id": bid, "cls": c,
                              "box": [round(v, 2) for v in box], "crop": Region(tuple(box), c).crop(image, 0.1, 32)})
    print("sampled %d crops from %d images: %s" % (len(items), i, dict(counts)), flush=True)
    return items


def report_noul(name, rows, key):
    p = np.array([r[key] for r in rows])
    y = np.array([r["label"] for r in rows], bool)
    correct = (p >= 0.5) == y
    conf = np.maximum(p, 1 - p)
    k, n = int(correct.sum()), len(correct)
    lo, hi = wilson(k, n)
    print("%-7s verify  acc %.3f [%.3f, %.3f]  AUROC %.3f  ECE %.3f  conf %.3f  P(yes|pos) %.3f  P(yes|neg) %.3f" % (
        name, k / n, lo, hi, auroc(p, y), ece_score(conf, correct.astype(float)), conf.mean(), p[y].mean(),
        p[~y].mean()))


def report_choice(name, rows, key):
    probs = [r[key] for r in rows]
    pred = [max(pp, key=pp.get) for pp in probs]
    truth = [r["cls"] for r in rows]
    conf = np.array([max(pp.values()) for pp in probs])
    correct = np.array([a == b for a, b in zip(pred, truth)], float)
    k, n = int(correct.sum()), len(correct)
    lo, hi = wilson(k, n)
    print("%-7s 8-way   acc %.3f [%.3f, %.3f]  ECE %.3f  conf %.3f  NLL %.3f" % (
        name, k / n, lo, hi, ece_score(conf, correct), conf.mean(),
        float(np.mean([-math.log(max(pp[t], 1e-12)) for pp, t in zip(probs, truth)]))))
    per = {c: np.mean([cc for cc, t in zip(correct, truth) if t == c]) for c in CLASSES}
    print("        per-class acc: " + "  ".join("%s %.2f" % (c, v) for c, v in per.items()))
    conf_pairs = Counter((t, p) for t, p in zip(truth, pred) if t != p).most_common(5)
    print("        top confusions (true->pred): " + ", ".join("%s->%s %d" % (t, p, m) for (t, p), m in conf_pairs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40, help="crops per class")
    ap.add_argument("--per-image", type=int, default=2, help="max crops of one class from one image")
    ap.add_argument("--max-images", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--out")
    ap.add_argument("--save-examples", help="directory for the first crop of each class")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    items = sample_crops(args.n, args.per_image, args.seed, args.max_images)
    if args.save_examples:
        seen = set()
        for it in items:
            if it["cls"] not in seen:
                seen.add(it["cls"])
                it["crop"].save("%s/%s_%d.png" % (args.save_examples, it["cls"], it["bbox_id"]))

    from transformers import AutoModel, AutoProcessor

    siglip = AutoModel.from_pretrained(SIGLIP, revision=SIGLIP_REVISION).eval()
    proc = AutoProcessor.from_pretrained(SIGLIP, revision=SIGLIP_REVISION)
    texts = ["a photo of %s %s." % (article(c), c) for c in CLASSES]
    with torch.no_grad():  # text embeddings are fixed per class set: computed once, as a deployment would
        txt = feats(siglip.get_text_features(**proc(text=texts, padding="max_length", return_tensors="pt")))
        txt = txt / txt.norm(dim=-1, keepdim=True)
    agent = load_vlm(CHECKPOINT, revision=REVISION, device="cpu")
    print("laya %s @ %s; siglip %s @ %s; dataset %s @ %s; threads %d" % (
        CHECKPOINT, agent.source["revision"], SIGLIP, SIGLIP_REVISION, DATASET, DATASET_REVISION, args.threads))

    rng = random.Random("neg-%d" % args.seed)
    out = open(args.out, "w") if args.out else None
    noul_rows, choice_rows, t_laya, t_sig = [], [], [], []
    for j, it in enumerate(items):
        c = it["cls"]
        neg = rng.choice([o for o in CLASSES if o != c])
        qs = {"pos": noul_q(c), "neg": noul_q(neg),
              "cls": {"type": "choice", "instructions": "What is this?", "criteria": list(CLASSES)}}
        t0 = time.perf_counter()
        a = agent.predict({"image": it["crop"]}, qs)["answers"]
        t_laya.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        with torch.no_grad():
            img = feats(siglip.get_image_features(**proc(images=it["crop"], return_tensors="pt")))
            img = img / img.norm(dim=-1, keepdim=True)
            logits = (img @ txt.T)[0] * siglip.logit_scale.exp() + siglip.logit_bias
        t_sig.append(time.perf_counter() - t0)
        sig_p = torch.sigmoid(logits).numpy()
        sig_soft = torch.softmax(logits, -1).numpy()

        row = {"i": j, "image_id": it["image_id"], "bbox_id": it["bbox_id"], "cls": c, "box": it["box"],
               "crop_size": list(it["crop"].size), "neg_cls": neg,
               "laya": {"p_pos": a["pos"]["noul"], "p_neg": a["neg"]["noul"], "probabilities": a["cls"]["probabilities"]},
               "siglip": {"logits": [round(float(v), 4) for v in logits],
                          "p_pos": float(sig_p[CLASSES.index(c)]), "p_neg": float(sig_p[CLASSES.index(neg)]),
                          "probabilities": {k: float(v) for k, v in zip(CLASSES, sig_soft)}},
               "latency_s": {"laya": round(t_laya[-1], 4), "siglip": round(t_sig[-1], 4)}}
        if out:
            out.write(json.dumps(row) + "\n")
            out.flush()
        noul_rows += [{"label": True, "laya": row["laya"]["p_pos"], "siglip": row["siglip"]["p_pos"]},
                      {"label": False, "laya": row["laya"]["p_neg"], "siglip": row["siglip"]["p_neg"]}]
        choice_rows.append({"cls": c, "laya": row["laya"]["probabilities"], "siglip": row["siglip"]["probabilities"]})
        if (j + 1) % 20 == 0:
            print("  %d/%d  laya %.2fs/crop  siglip %.3fs/crop" % (j + 1, len(items), np.mean(t_laya),
                                                                  np.mean(t_sig)), flush=True)
    if out:
        out.close()

    print("\nn crops = %d (%d verification answers)" % (len(items), len(noul_rows)))
    for m in ("laya", "siglip"):
        report_noul(m, noul_rows, m)
    for m in ("laya", "siglip"):
        report_choice(m, choice_rows, m)
    print("latency per crop (threads=%d): laya %.3fs (median %.3f; 3 questions, one predict)  siglip %.3fs "
          "(median %.3f; image tower only, text embeddings cached)" % (
              args.threads, np.mean(t_laya), np.median(t_laya), np.mean(t_sig), np.median(t_sig)))


if __name__ == "__main__":
    main()
