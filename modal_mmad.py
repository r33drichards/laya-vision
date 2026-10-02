"""Modal jobs that run the SmolVLM decision model on MMAD, the industrial anomaly-inspection benchmark.

    modal run --detach modal_mmad.py::prepare                                  # stage 28 GB of MMAD into /data/mmad
    modal run modal_mmad.py::bench --run-name all3-3ep/best --limit 50         # smoke test on 50 images
    modal run --detach modal_mmad.py::bench --run-name all3-3ep/best           # the full 1-shot sweep
    modal run modal_mmad.py::report --answers answers_1_shot_all3-3ep-best     # MMAD's own scorer

MMAD (Jiang et al., ICLR 2025, https://github.com/jam-cc/MMAD) asks 39,670 multiple-choice questions about
8,366 industrial images across five defect datasets and nine question types. Every question has 2 or 4
options, which is exactly a ``choice`` question, so the decision model answers it natively: the image is
encoded once per image and every question about it is scored in the same forward pass, no text generated and
no answer string to parse. ``argmax`` over the options *is* the letter.

Scoring is MMAD's own ``evaluation/examples/helper/summary.py``, downloaded with the benchmark rather than
copied in here (the MMAD repo ships no licence). The numbers are therefore theirs, not a reimplementation.

Volumes (created out of band; never ``modal deploy`` this app):
    laya-hf-cache     -> /cache/hf   (HF_HOME)
    laya-datasets     -> /data       (this app writes only under /data/mmad/)
    laya-checkpoints  -> /ckpt       (read-only)
"""
import json
import os
import shutil
import subprocess
import time

import modal

app = modal.App("laya-mmad")

hf_vol = modal.Volume.from_name("laya-hf-cache")
data_vol = modal.Volume.from_name("laya-datasets")
ckpt_vol = modal.Volume.from_name("laya-checkpoints")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl", "unzip")
    .pip_install(
        "torch==2.14.0",
        "torchvision==0.29.0",
        "transformers==5.17.0",
        "safetensors",
        "huggingface_hub",
        "numpy",
        "pillow",
        "pandas",
        "matplotlib",
        "seaborn",
    )
    .env({"HF_HOME": "/cache/hf", "TOKENIZERS_PARALLELISM": "false", "MPLBACKEND": "Agg"})
    .add_local_python_source("laya")
)

CKPT_ROOT = "/ckpt/smolvlm"

MMAD_DIR = "/data/mmad"
REPO_DIR = MMAD_DIR + "/MMAD-main"          # the benchmark's code: mmad.json, domain_knowledge.json, summary.py
DATA_PATH = REPO_DIR + "/dataset/MMAD"      # image roots live beside mmad.json, as MMAD's own scripts expect
RESULTS_DIR = MMAD_DIR + "/results"
ZIP_DIR = MMAD_DIR + "/_zips"
READY = MMAD_DIR + "/_READY"

REPO_TARBALL = "https://codeload.github.com/jam-cc/MMAD/tar.gz/refs/heads/main"
HF_ZIP = "https://huggingface.co/datasets/jiang-cc/MMAD/resolve/main/%s.zip"



def _mmad_json() -> dict:
    with open(os.path.join(DATA_PATH, "mmad.json")) as f:
        return json.load(f)


# ---------------------------------------------------------------------------------------------------------
# Staging
# ---------------------------------------------------------------------------------------------------------


@app.function(
    image=image,
    cpu=4,
    memory=16384,
    timeout=4 * 60 * 60,
    volumes={"/data": data_vol},
)
def stage_one(name: str):
    """Download one sub-dataset's zip and unpack it into the volume. One container per dataset.

    The zip goes to container-local scratch, not the volume: writing 28 GB of zip through the volume mount
    and reading it straight back is the slow way round, and the unpacked images have to cross that mount
    anyway. Datasets stage concurrently because each writes a disjoint subtree.
    """
    data_vol.reload()
    zp = "/tmp/%s.zip" % name
    t0 = time.time()
    subprocess.run(["curl", "-fsSL", "--retry", "5", "-o", zp, HF_ZIP % name], check=True)
    dl = time.time() - t0
    gb = os.path.getsize(zp) / 1e9
    print("%-12s downloaded %.1f GB in %.0f s (%.0f MB/s), unpacking" % (name, gb, dl, gb * 1000 / max(dl, 1)), flush=True)

    t1 = time.time()
    # the zips are Mac-made: they carry a __MACOSX sidecar and .DS_Store files that are not benchmark data
    subprocess.run(
        ["unzip", "-q", "-o", zp, "-d", DATA_PATH, "-x", "__MACOSX/*", "*/.DS_Store", "*.DS_Store"],
        check=True,
    )
    os.remove(zp)
    shutil.rmtree(os.path.join(DATA_PATH, "__MACOSX"), ignore_errors=True)
    data_vol.commit()
    print("%-12s staged (%.1f GB, %.0f s download + %.0f s unpack)" % (name, gb, dl, time.time() - t1), flush=True)
    return {"subset": name, "gb": round(gb, 2), "download_s": round(dl), "unpack_s": round(time.time() - t1)}


@app.function(image=image, cpu=4, memory=8192, timeout=6 * 60 * 60, volumes={"/data": data_vol})
def prepare(subsets: str = "", force: bool = False):
    """Download the MMAD code and images into /data/mmad, then verify every path mmad.json references."""
    from laya.mmad import SUBSETS

    os.makedirs(MMAD_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    shutil.rmtree(ZIP_DIR, ignore_errors=True)  # an older layout staged zips here; they are scratch, not data

    # -- the benchmark's own code and metadata ------------------------------------------------------------
    if force or not os.path.exists(os.path.join(DATA_PATH, "mmad.json")):
        print("fetching the MMAD repo (mmad.json, domain_knowledge.json, summary.py)")
        tar = "/tmp/mmad-repo.tar.gz"
        subprocess.run(["curl", "-fsSL", "--retry", "5", "-o", tar, REPO_TARBALL], check=True)
        subprocess.run(["tar", "xzf", tar, "-C", MMAD_DIR], check=True)  # extracts MMAD-main/
        os.remove(tar)
        data_vol.commit()  # the fan-out containers reload the volume to find DATA_PATH
    chat_ad = _mmad_json()
    print("mmad.json: %d images, %d questions" % (len(chat_ad), sum(len(v["conversation"]) for v in chat_ad.values())))

    # -- the images ----------------------------------------------------------------------------------------
    # "is it staged?" has to ask mmad.json for a real image path: the repo tarball creates every dataset
    # folder already, each holding a README that explains where to download the images to, so a directory
    # existing and being non-empty means nothing.
    probe = {}
    for key in chat_ad:
        probe.setdefault(key.split("/")[0], key)

    todo = []
    for name in [s for s in subsets.split(",") if s] or list(SUBSETS):
        if not force and name in probe and os.path.exists(os.path.join(DATA_PATH, probe[name])):
            print("%-12s already staged, skipping" % name)
        else:
            todo.append(name)
    if todo:
        print("staging %s concurrently, one container each" % ", ".join(todo), flush=True)
        for r in stage_one.map(todo):
            print("   done:", r, flush=True)
    data_vol.reload()

    # -- verify every path the benchmark will ask for ------------------------------------------------------
    print("\nverifying mmad.json paths against what is on disk")
    ok = miss_q = miss_t = 0
    per = {}
    for key, v in chat_ad.items():
        sub = key.split("/")[0]
        st = per.setdefault(sub, {"total": 0, "ok": 0, "no_query": 0, "no_template": 0})
        st["total"] += 1
        has_q = os.path.exists(os.path.join(DATA_PATH, key))
        tpls = (v.get("random_templates") or []) + (v.get("similar_templates") or [])
        has_t = any(os.path.exists(os.path.join(DATA_PATH, t)) for t in tpls[:1] + tpls[8:9])
        if not has_q:
            st["no_query"] += 1
            miss_q += 1
        if not has_t:
            st["no_template"] += 1
            miss_t += 1
        if has_q and has_t:
            st["ok"] += 1
            ok += 1
    for sub in sorted(per):
        st = per[sub]
        print("  %-12s %5d/%5d usable   (missing query %d, missing template %d)"
              % (sub, st["ok"], st["total"], st["no_query"], st["no_template"]))
    print("  %-12s %5d/%5d usable" % ("TOTAL", ok, len(chat_ad)))

    if ok:
        with open(READY, "w") as f:
            json.dump({"staged": [s for s in subsets.split(",") if s], "usable_images": ok, "total_images": len(chat_ad)}, f)
    data_vol.commit()
    if miss_q or miss_t:
        print("\n%d images and %d templates are missing; bench will skip those and say so." % (miss_q, miss_t))
    return {"usable": ok, "total": len(chat_ad), "missing_query": miss_q, "missing_template": miss_t}


# ---------------------------------------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------------------------------------


@app.function(
    image=image,
    gpu="L4",
    cpu=8,
    memory=32768,
    timeout=8 * 60 * 60,
    volumes={"/cache/hf": hf_vol, "/data": data_vol, "/ckpt": ckpt_vol.read_only()},
)
def bench(
    run_name: str = "all3-3ep/best",
    shots: int = 1,
    template: str = "random",
    subsets: str = "",
    question_types: str = "Anomaly Detection",
    limit: int = 0,
    n_permutations: int = 1,
    noul_detection: bool = False,
    prep_backend: str = "auto",
    out: str = "",
    resume: bool = True,
):
    """Answer every MMAD question about every staged image and write an answers file MMAD's scorer can read.

    ``shots=1`` (the benchmark's headline setting) prepends one known-good template image of the same product
    to the query image; ``template`` picks MMAD's ``random_templates`` or ``similar_templates`` list.

    ``question_types`` keeps only those MMAD subtasks; empty string means all nine. It defaults to Anomaly
    Detection ("Is there any defect in the object?"), which is the one subtask this harness answers under
    exactly the conditions the reference scripts use. Those scripts ask question i with questions 1..i-1 in
    the same completion, so a chat model answers a follow-up having just written its own answers to what
    came before -- and the follow-ups are phrased to presuppose the defect ("There is a defect in the
    object. What is the type of the defect?"). This model answers every question independently, so it never
    gets that conditioning. Anomaly Detection is exempt: it is always question index 0, one per image on
    8,297 of the 8,366 images, so the reference call for it carries no prior context either.

    ``prep_backend="auto"`` uses the Hugging Face processor whenever more than one image is in play. The
    checkpoint's own device-side path (``laya.preprocess``) stacks the images into one tensor and so requires
    them to share a resolution, which a query and its template often do not; the processor resizes each image
    independently and is the reference path that the device-side one reproduces to ~0.05 grey levels. Pass
    ``checkpoint`` to force the trained path (fine for ``shots=0``).
    """
    from dataclasses import replace

    import torch

    from laya.mmad import NOTE_1SHOT, SUBSETS, answer_record, build_questions, select_conversations, to_letter
    from laya.vlm import VLMAgent

    data_vol.reload()
    if not os.path.exists(os.path.join(DATA_PATH, "mmad.json")):
        raise SystemExit("MMAD is not staged; run `modal run --detach modal_mmad.py::prepare` first")

    chat_ad = _mmad_json()
    wanted = {s for s in subsets.split(",") if s} or set(SUBSETS)
    types = {t.strip() for t in question_types.split(",") if t.strip()}
    conv_of = select_conversations(chat_ad, wanted, types)
    keys = [k for k in chat_ad if k in conv_of]
    if types:
        print("question types: %s (%d of %d images have one, %d questions)"
              % (", ".join(sorted(types)), len(keys), len(chat_ad), sum(len(c) for c in conv_of.values())))
    if limit:
        keys = keys[:: max(1, len(keys) // limit)][:limit]  # spread the smoke test across products, not just bottles

    # every setting that changes an answer belongs in the name, so a smoke run and a sweep never share a file
    tag = out or "answers_%d_shot_%s" % (shots, run_name.replace("/", "-"))
    if not out:
        if types and types != {c["type"] for v in chat_ad.values() for c in v["conversation"]}:
            tag += "_" + "+".join(sorted(t.replace(" ", "") for t in types))
        if noul_detection:
            tag += "_noul"
        if n_permutations != 1:
            tag += "_perm%d" % n_permutations
        if limit:
            tag += "_smoke%d" % limit
    os.makedirs(RESULTS_DIR, exist_ok=True)
    jsonl_path = os.path.join(RESULTS_DIR, tag + ".jsonl")
    json_path = os.path.join(RESULTS_DIR, tag + ".json")

    done = set()
    if os.path.exists(jsonl_path):
        if not resume:
            os.remove(jsonl_path)  # --no-resume means redo it, not append to the last run's answers
        else:
            with open(jsonl_path) as f:
                for line in f:
                    try:
                        done.add(json.loads(line)["image"])
                    except Exception:
                        pass
            print("resuming: %d images already answered" % len(done))
    todo = [k for k in keys if k not in done]

    print("GPU:", torch.cuda.get_device_name(0))
    agent = VLMAgent(os.path.join(CKPT_ROOT, run_name), device="cuda")
    use_processor = prep_backend == "processor" or (prep_backend == "auto" and shots > 0)
    if use_processor and agent.prep.backend != "processor":
        agent.prep = replace(agent.prep, backend="processor")
        agent.prep.apply(agent.processor)
        agent.prep.check(agent.processor)
    print("checkpoint %s | image prep: %s @ %dpx | %d-shot (%s templates) | %d images, %d to answer"
          % (run_name, agent.prep.backend, agent.prep.image_size, shots, template, len(keys), len(todo)))

    def load_images(key):
        from PIL import Image

        paths = []
        if shots:
            tpls = chat_ad[key].get("%s_templates" % template) or []
            paths += [t for t in tpls if os.path.exists(os.path.join(DATA_PATH, t))][:shots]
            if len(paths) < shots:
                raise FileNotFoundError("no %s template on disk for %s" % (template, key))
        paths.append(key)
        imgs = []
        for p in paths:
            with Image.open(os.path.join(DATA_PATH, p)) as im:
                imgs.append(im.convert("RGB"))
        return imgs

    import collections
    from concurrent.futures import ThreadPoolExecutor

    pool = ThreadPoolExecutor(max_workers=8)
    pending = collections.deque()
    src = iter(todo)

    def refill():
        while len(pending) < 24:
            try:
                key = next(src)
            except StopIteration:
                return
            pending.append((key, pool.submit(load_images, key)))

    refill()
    n_img = n_q = n_correct = skipped = 0
    t0 = time.time()
    fh = open(jsonl_path, "a")
    try:
        while pending:
            key, fut = pending.popleft()
            refill()
            try:
                imgs = fut.result()
            except Exception as e:
                skipped += 1
                if skipped <= 10:
                    print("skip %s: %s" % (key, e))
                continue

            questions, meta = build_questions(conv_of[key], noul_detection)
            state = {"images": imgs}
            if shots:
                state["note"] = NOTE_1SHOT
            out_ = agent.predict(state, questions, n_permutations=n_permutations)

            for qid, m in meta.items():
                a = out_["answers"][qid]
                letter, probs = to_letter(a, m)
                n_q += 1
                n_correct += letter == m["answer"]
                fh.write(json.dumps(answer_record(key, m, letter, probs, a)) + "\n")
            n_img += 1
            if n_img % 200 == 0:
                fh.flush()
                data_vol.commit()
                rate = n_img / (time.time() - t0)
                eta = (len(todo) - n_img) / max(rate, 1e-6) / 60
                print("%5d/%d images | %6d questions | running acc %.3f | %.1f img/s | eta %.0f min"
                      % (n_img, len(todo), n_q, n_correct / max(n_q, 1), rate, eta), flush=True)
    finally:
        fh.close()
        pool.shutdown(wait=False)

    # MMAD's scorer wants one JSON list, so publish that alongside the resumable log
    records = []
    with open(jsonl_path) as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except Exception:
                pass
    with open(json_path, "w") as f:
        json.dump(records, f, indent=4)
    data_vol.commit()

    mins = (time.time() - t0) / 60
    print("\n%d images answered this run (%d skipped) in %.1f min | %d questions | raw accuracy %.4f"
          % (n_img, skipped, mins, n_q, n_correct / max(n_q, 1)))
    print("answers: %s (%d questions in total, including any resumed)" % (json_path, len(records)))
    return {"answers": tag, "images": n_img, "skipped": skipped, "questions": n_q, "total_questions": len(records),
            "raw_accuracy": round(n_correct / max(n_q, 1), 4), "minutes": round(mins, 1)}


# ---------------------------------------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------------------------------------


def _published(column: str = "Anomaly Detection"):
    """MMAD's own published results, read from the ``all_result/`` CSVs that ship with the benchmark.

    Yields ``(model, shots, {dataset_row: value})``. These are the numbers from the paper's tables, so a
    run can be placed against them without anyone copying figures by hand.
    """
    import csv
    import glob
    import re

    for path in sorted(glob.glob(os.path.join(REPO_DIR, "all_result", "*_accuracy*.csv"))):
        m = re.match(r"answers_(\d)_shot_(.+?)_accuracy(?:_test\d)?\.csv$", os.path.basename(path))
        if not m:
            continue
        with open(path) as f:
            table = {r[""]: r for r in csv.DictReader(f)}
        vals = {}
        for row, cols in table.items():
            try:
                vals[row] = float(cols[column])
            except (KeyError, TypeError, ValueError):
                pass
        if vals:
            yield m.group(2), int(m.group(1)), vals


def _leaderboard(our_csv: str, label: str, column: str = "Anomaly Detection", row: str = "Average",
                 published_row: str = ""):
    """Print the published models plus this run, sorted, so the result lands in context.

    ``row`` picks which line of our accuracy table to compare; ``published_row`` the line of theirs when
    the two are named differently. They are: ``summary.py`` folds DS-MVTec into an "MVTec-AD" row via
    ``normalize_dataset_name``, while the shipped CSVs were written before that and still say "DS-MVTec".
    Same images either way, ours plus the 21 native MVTec-AD ones.
    """
    import csv

    theirs = published_row or row
    entries = [(model, shots, vals[theirs]) for model, shots, vals in _published(column) if theirs in vals]
    if not entries:
        print("\n(no published %r row in %s/all_result to compare against)" % (theirs, REPO_DIR))
        return
    with open(our_csv) as f:
        ours = {r[""]: r for r in csv.DictReader(f)}
    if row not in ours:
        print("\n(this run has no %r row to place; it has %s)" % (row, ", ".join(sorted(ours))))
        return
    mine = float(ours[row][column])
    entries.append((label, 1, mine))
    entries.sort(key=lambda e: -e[2])

    print("\n%s (%s row), against MMAD's published results:" % (column, theirs))
    print("   %-42s %5s %9s" % ("model", "shot", "score"))
    for model, shots, val in entries:
        mark = "->" if model == label else "  "
        print("%s %-42s %5d %9.2f" % (mark, model, shots, val))
    better = sum(1 for _, _, v in entries if v < mine)
    print("   ranks %d of %d (ahead of %d published models)"
          % (len(entries) - better, len(entries), better))


@app.function(image=image, cpu=2, memory=8192, timeout=30 * 60, volumes={"/data": data_vol})
def report(answers: str, show_overkill_miss: bool = True):
    """Score an answers file with MMAD's own ``caculate_accuracy_mmad``, then add the calibration it discards.

    MMAD reports accuracy per (sub-dataset x question type), with Anomaly Detection as balanced accuracy over
    normal and anomalous images, and writes a CSV next to the answers file. The decision model also emits a
    probability per option, which a letter-only scorer throws away, so ECE and AUROC on the detection subtask
    are computed here as well.
    """
    import sys

    import matplotlib

    from laya.mmad import calibration_extras, threshold_sweep

    matplotlib.use("Agg")  # summary.py calls plt.show(); Agg makes that a no-op instead of a hang

    data_vol.reload()
    path = answers if answers.endswith(".json") else os.path.join(RESULTS_DIR, answers + ".json")
    if not os.path.exists(path):
        raise SystemExit("no answers file at %s" % path)

    sys.path.insert(0, os.path.join(REPO_DIR, "evaluation", "examples"))
    from helper.summary import caculate_accuracy_mmad

    caculate_accuracy_mmad(path, show_overkill_miss=show_overkill_miss)
    csv_path = path.replace(".json", "_accuracy.csv")
    data_vol.commit()

    # -- what the letter-only scorer drops -----------------------------------------------------------------
    with open(path) as f:
        records = json.load(f)
    extra = calibration_extras(records)
    print("\nbeyond MMAD's table (uses the probabilities a letter-only scorer discards):")
    for k, v in extra.items():
        print("   %-18s %s" % (k, v))
    sweep = threshold_sweep(records)
    if sweep:
        print("\nwhat the fixed 0.5 decision threshold costs (detection, n=%d):" % sweep["n"])
        print("   at 0.5, as MMAD scores it   %.4f" % sweep["at_half"])
        print("   best threshold %.3f          %.4f   <- chosen on this same data, so optimistic"
              % (sweep["oracle_threshold"], sweep["oracle"]))
        if sweep["cv"] is not None:
            print("   %d-fold cross-validated      %.4f   <- what retuning would actually deliver"
                  % (sweep["cv_folds"], sweep["cv"]))

    _leaderboard(csv_path, "laya-vision-smolvlm-256m", row="Average")
    for ours_row, their_row in (("MVTec-AD", "DS-MVTec"), ("MVTec-LOCO", "MVTec-LOCO"),
                                ("VisA", "VisA"), ("GoodsAD", "GoodsAD")):
        _leaderboard(csv_path, "laya-vision-smolvlm-256m", row=ours_row, published_row=their_row)
    print("\naccuracy table: %s" % csv_path)
    return extra


@app.local_entrypoint()
def main(run_name: str = "all3-3ep/best", shots: int = 1, limit: int = 0, subsets: str = "",
         question_types: str = "Anomaly Detection", noul_detection: bool = False, n_permutations: int = 1):
    """modal run modal_mmad.py --limit 50            # smoke test, then the full sweep with no --limit"""
    res = bench.remote(run_name=run_name, shots=shots, limit=limit, subsets=subsets,
                       question_types=question_types, noul_detection=noul_detection,
                       n_permutations=n_permutations)
    print(json.dumps(res, indent=2))
    if res["total_questions"]:
        print(json.dumps(report.remote(res["answers"]), indent=2))
