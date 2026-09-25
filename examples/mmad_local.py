# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#   "torch", "torchvision", "transformers>=4.45", "safetensors",
#   "huggingface_hub", "numpy", "pillow",
# ]
# ///
"""Run a small MMAD sample on this machine, without the 28 GB download.

MMAD ships its images as five zips totalling 28.3 GB, but the Hugging Face copy also serves them as
individual files, so a sample only needs the handful it actually looks at.

    uv run examples/mmad_local.py                     # 40 DS-MVTec images, 1-shot, Anomaly Detection
    uv run examples/mmad_local.py --n 200 --verbose   # bigger, and print every answer

This is a spot-check, not the benchmark result. The authoritative run is ``modal_mmad.py``, which works
from the complete zips and scores with MMAD's own ``summary.py`` over all 8,297 detection questions. Both
call the same mapping in ``laya/mmad.py``, so a disagreement between them is a bug rather than a
methodology difference.

Why it defaults to DS-MVTec only: the individual-file revision is an incomplete upload, and a 1-shot
question needs both its query image and its normal template. Counting against the repo manifest, of the
8,297 detection images only 1,060 (12.8%) have both:

    DS-MVTec    672/1670  40.2%        MVTec-LOCO   162/1565  10.4%
    MVTec-AD      8/  21  38.1%        VisA         119/2141   5.6%
                                       GoodsAD       99/2900   3.4%

Sampling across all five would therefore return something that is mostly DS-MVTec while looking like a
cross-dataset average, so one dataset stated plainly is the more honest number. ``--subsets`` overrides.
The gap does tilt the label balance a little (14.5% of defective images are usable against 10.1% of normal
ones), which is why the sample is stratified over normal/defective rather than drawn uniformly.
"""
import argparse
import json
import os
import random
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from laya.mmad import (  # noqa: E402
    DETECTION,
    MMAD_JSON_URL,
    MMAD_REF,
    NOTE_1SHOT,
    answer_record,
    balanced_accuracy,
    build_questions,
    calibration_extras,
    image_url,
    is_normal,
    select_conversations,
    to_letter,
)

RETRY_CODES = (429, 500, 502, 503, 504)


def _open(req, attempts=6):
    """Open a request, backing off on 429 and 5xx. Hugging Face rate-limits a burst of small downloads."""
    delay = 1.0
    for attempt in range(attempts):
        try:
            return urllib.request.urlopen(req)
        except urllib.error.HTTPError as e:
            if e.code not in RETRY_CODES or attempt == attempts - 1:
                raise
            wait = delay
            try:
                wait = max(delay, float(e.headers.get("Retry-After") or 0))
            except (TypeError, ValueError):
                pass
            time.sleep(min(wait, 30))
            delay *= 2


def fetch(url, dest):
    """Download to ``dest`` unless it is already cached. Returns bytes written, 0 if cached."""
    if os.path.exists(dest) and os.path.getsize(dest):
        return 0
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "laya-mmad-local/1.0"})
    # unique per call: several query images share one template, so two threads can want the same dest at
    # once, and a shared ".part" name means whichever renames second finds it already gone
    tmp = "%s.%d.part" % (dest, threading.get_ident())
    with _open(req) as r, open(tmp, "wb") as f:
        data = r.read()
        f.write(data)
    os.replace(tmp, dest)  # so an interrupted download is never mistaken for a cached file
    return len(data)


def manifest(cache):
    """Every path the individual-file revision serves, as a set, cached on disk.

    One listing call instead of a HEAD per candidate. Probing per file does not scale: a 200-image sample
    needs ~800 checks, and Hugging Face answers that burst with 429s no amount of backoff rides out.
    """
    path = os.path.join(cache, "repo_files.json")
    if os.path.exists(path):
        with open(path) as f:
            return set(json.load(f))
    from huggingface_hub import list_repo_files

    print("listing the MMAD file revision (one call, ~25k paths)", flush=True)
    files = list_repo_files("jiang-cc/MMAD", repo_type="dataset", revision=MMAD_REF.replace("%2F", "/"))
    os.makedirs(cache, exist_ok=True)
    with open(path, "w") as f:
        json.dump(sorted(files), f)
    return set(files)


def stratified(keys, n, seed):
    """Pick ``n`` keys spread over (sub-dataset, normal vs defective), so both classes are represented."""
    buckets = defaultdict(list)
    for k in keys:
        buckets[(k.split("/")[0], is_normal(k))].append(k)
    rng = random.Random(seed)
    for v in buckets.values():
        rng.shuffle(v)
    order = sorted(buckets)
    picked, i = [], 0
    while len(picked) < n and any(buckets[b] for b in order):
        b = order[i % len(order)]
        if buckets[b]:
            picked.append(buckets[b].pop())
        i += 1
    return picked


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=40, help="images to sample (default 40)")
    p.add_argument("--model", default="thaitea/laya-vision-smolvlm-256m", help="hub id or local checkpoint path")
    p.add_argument("--shots", type=int, default=1, help="normal reference images prepended (default 1)")
    p.add_argument("--template", default="random", choices=("random", "similar"))
    p.add_argument("--question-types", default=DETECTION, help='comma-separated, or "" for all nine')
    p.add_argument("--subsets", default="DS-MVTec",
                   help="default DS-MVTec: the only subset this revision covers well (see above)")
    p.add_argument("--n-permutations", type=int, default=1, help="average over K option orders")
    p.add_argument("--noul-detection", action="store_true", help="route yes/no through the noul head")
    p.add_argument("--cache", default=os.path.expanduser("~/.cache/mmad-local"))
    p.add_argument("--device", default=None, help="cuda / mps / cpu (default: best available)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--verbose", action="store_true", help="print every answer")
    args = p.parse_args()

    # -- the benchmark metadata --------------------------------------------------------------------------
    mmad_json = os.path.join(args.cache, "mmad.json")
    if not os.path.exists(mmad_json):
        print("fetching mmad.json (~29 MB) -> %s" % mmad_json, flush=True)
        fetch(MMAD_JSON_URL, mmad_json)
    with open(mmad_json) as f:
        chat_ad = json.load(f)
    served = manifest(args.cache)

    types = [t.strip() for t in args.question_types.split(",") if t.strip()]
    conv_of = select_conversations(chat_ad, [s for s in args.subsets.split(",") if s], types)

    # only the images whose every needed file is actually served
    paths_for, usable = {}, []
    for k in conv_of:
        paths = (chat_ad[k].get("%s_templates" % args.template) or [])[:args.shots] + [k]
        if len(paths) == args.shots + 1 and all(p in served for p in paths):
            paths_for[k] = paths
            usable.append(k)
    if not usable:
        raise SystemExit("no %d-shot image in %s is served individually; try --subsets DS-MVTec"
                         % (args.shots, args.subsets))
    print("%d of %d candidate images are fully served (%.1f%%) | questions: %s"
          % (len(usable), len(conv_of), 100 * len(usable) / len(conv_of), ", ".join(types) or "all nine"))

    keys = stratified(usable, args.n, args.seed)
    t0 = time.time()
    local = lambda rel: os.path.join(args.cache, "images", rel)  # noqa: E731
    need = list(dict.fromkeys(rel for k in keys for rel in paths_for[k]))  # templates are shared
    with ThreadPoolExecutor(max_workers=6) as pool:
        sizes = list(pool.map(lambda rel: fetch(image_url(rel), local(rel)), need))
    n_norm = sum(is_normal(k) for k in keys)
    print("%d images ready (%d normal, %d defective) over %d sub-datasets | %.1f MB in %.1f s"
          % (len(keys), n_norm, len(keys) - n_norm, len({k.split("/")[0] for k in keys}),
             sum(sizes) / 1e6, time.time() - t0), flush=True)
    if len(keys) < args.n:
        print("   only %d were available, not the %d requested" % (len(keys), args.n))

    # -- the model ---------------------------------------------------------------------------------------
    import torch
    from PIL import Image

    from laya.vlm import VLMAgent

    device = args.device or ("cuda" if torch.cuda.is_available()
                             else "mps" if torch.backends.mps.is_available() else "cpu")
    print("loading %s on %s" % (args.model, device), flush=True)
    agent = VLMAgent(args.model, device=device)

    records = []
    t0 = time.time()
    for i, key in enumerate(keys, 1):
        imgs = []
        for rel in paths_for[key]:
            with Image.open(local(rel)) as im:
                imgs.append(im.convert("RGB"))
        state = {"images": imgs}
        if args.shots:
            state["note"] = NOTE_1SHOT

        questions, meta = build_questions(conv_of[key], args.noul_detection)
        out = agent.predict(state, questions, n_permutations=args.n_permutations)
        for qid, m in meta.items():
            a = out["answers"][qid]
            letter, probs = to_letter(a, m)
            records.append(answer_record(key, m, letter, probs, a))
            if args.verbose:
                print("  %-58s %s  said %s want %s  %s" % (
                    key[-58:], "normal" if is_normal(key) else "defect", letter, m["answer"],
                    "ok " if letter == m["answer"] else "MISS"))
        if not args.verbose and i % 25 == 0:
            print("  %d/%d images (%.2f s/image)" % (i, len(keys), (time.time() - t0) / i), flush=True)
    secs = time.time() - t0

    # -- results -----------------------------------------------------------------------------------------
    out_path = os.path.join(args.cache, "sample_answers.json")
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2)

    extras = calibration_extras(records)
    print("\n%d questions over %d images in %.1f s (%.2f s/image)" % (len(records), len(keys), secs, secs / len(keys)))
    print("overall accuracy   %.4f" % extras["overall_accuracy"])
    if any(r["question_type"] == DETECTION for r in records):
        d = balanced_accuracy(records)
        print("\nAnomaly Detection, scored MMAD's way:")
        print("   balanced accuracy   %.4f      <- the headline metric ((normal + anomaly) / 2)" % d["balanced_acc"])
        print("   normal   accuracy   %.4f  (n=%d, overkill %.4f)" % (d["normal_acc"], d["n_normal"], d["overkill"]))
        print("   anomaly  accuracy   %.4f  (n=%d, miss     %.4f)" % (d["anomaly_acc"], d["n_anomalous"], d["miss"]))
        print("   precision %.4f   recall %.4f   F1 %.4f" % (d["precision"], d["recall"], d["f1"]))
        if "detection_auroc" in extras:
            print("   AUROC     %.4f   (from P(defective), which a letter-only scorer discards)"
                  % extras["detection_auroc"])
    print("   ECE       %.4f" % extras["ece"])
    print("\nchance is 0.5 on a balanced two-option question. answers: %s" % out_path)
    print("this is a %d-image spot-check, not the benchmark -- see docs/mmad.md" % len(keys))


if __name__ == "__main__":
    main()
