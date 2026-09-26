"""Two-tower / late-interaction scoring experiment (``laya/two_tower.py``), distilled from the cross-encoder.

    # train + evaluate one student (H100), results to /ckpt/two-tower/<run>/ (create-only)
    modal run --detach modal_two_tower.py::train --run tt-30m --mode tt --minutes 30
    modal run --detach modal_two_tower.py::train --run li-30m --mode li --minutes 30
    # the teacher's own numbers on the same rows, same GPU type (identity order, 3-order flip rate)
    modal run --detach modal_two_tower.py::eval_teacher
    # L4 bf16 latency per decision, teacher vs students, k options cached
    modal run modal_two_tower.py::latency --runs tt-30m,li-30m

Data: the autoresearch pool (``/data/autoresearch/pool-v2``): train on its ``train`` part (up to 6,000 examples per
Cauldron / score set, calibration tail excluded), fit per-type temperatures on its ``calib`` part (the last 100 train
records of each Cauldron / score set, never trained on), score its ``eval`` part (300 seeded val questions from each
of the 34 eval sets), as ``autoresearch/harness.py`` does. The teacher is ``cauldron-score-2ep-bidir-full/best``
(= thaitea/laya-vision); the student starts from its weights. Loss: ``w_kd`` x soft cross-entropy to the teacher's
calibrated probabilities + ``w_gt`` x soft cross-entropy to the label (vote histogram where there is one).
"""
import json
import math
import os
import time

import modal

REPO = os.path.dirname(os.path.abspath(__file__))
TEACHER = "/ckpt/smolvlm/cauldron-score-2ep-bidir-full/best"
POOL = "/data/autoresearch/pool-v2"
OUT = "/ckpt/two-tower"
MIX = {"score_vlfeedback": 3.0}   # the teacher run's sampling weights

app = modal.App("laya-two-tower")
hf_vol = modal.Volume.from_name("laya-hf-cache")
data_vol = modal.Volume.from_name("laya-datasets")
ckpt_vol = modal.Volume.from_name("laya-checkpoints")
VOLUMES = {"/cache/hf": hf_vol, "/data": data_vol, "/ckpt": ckpt_vol}
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.14.0", "torchvision==0.29.0", "transformers==5.17.0", "safetensors", "huggingface_hub",
                 "numpy", "pillow", "num2words")
    .env({"HF_HOME": "/cache/hf", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_python_source("laya")
)


def _log(t0, msg):
    print("[%7.1f s] %s" % (time.time() - t0, msg), flush=True)


def load_pool(kind, names=None):
    import pickle

    d = os.path.join(POOL, kind)
    out = []
    for f in sorted(os.listdir(d)):
        if f.endswith(".pkl") and (names is None or f[:-4] in names):
            with open(os.path.join(d, f), "rb") as fh:
                out += pickle.load(fh)
    return out


def _teacher_records(agent, examples, orders=None):
    from laya.vlm_train import collect_logits

    return collect_logits(agent.model, agent.processor, examples, batch_size=32, num_workers=8, orders=orders)


def _identity(records):
    return [dict(r, logits=r["logits_per_order"][0]) for r in records]


def three_orders(k):
    """Identity, reversed, and a cyclic shift by one (fewer distinct orders when k is 2)."""
    ident = list(range(k))
    return [ident, ident[::-1], [(i + 1) % k for i in range(k)]]


def summarize(calib_records, eval_records, teacher_eval=None):
    """Temperatures fit on the calib records; accuracy / ECE / NLL per set and pooled; the harness's quality;
    option-order flip rates (from ``logits_per_order`` when there are 3 orders); agreement with the teacher."""
    import numpy as np
    import torch

    from laya.vlm_train import fit_temperatures_from, metrics_from

    temps = fit_temperatures_from(_identity(calib_records))
    ev = _identity(eval_records)
    m = metrics_from(ev, temps)
    raw = metrics_from(ev, (1.0, 1.0, 1.0))
    hard = metrics_from([r for r in ev if float(r["target"].max()) >= 0.999], temps)
    names = [n for n in m if n != "all"]
    macro = float(np.mean([m[n]["acc"] for n in names]))
    out = {"temperatures": temps, "macro_acc": macro, "ece_hard": hard["all"]["ece"], "nll_hard": hard["all"]["nll"],
           "quality": macro - hard["all"]["ece"], "all": m["all"], "raw_all": raw["all"], "per_set": m}
    if len(eval_records[0]["logits_per_order"]) >= 3:
        rev = anyf = n = 0
        for r in eval_records:
            if len(r["target"]) < 2:
                continue
            a = [int(torch.argmax(z)) for z in r["logits_per_order"]]
            n += 1
            rev += a[0] != a[1]
            anyf += len(set(a)) > 1
        out["flip_reversed"], out["flip_any3"], out["flip_n"] = rev / n, anyf / n, n
        # permutation-averaged (3 orders) accuracy, as predict(n_permutations=3) would give
        avg = [dict(r, logits=torch.stack(r["logits_per_order"]).mean(0)) for r in eval_records]
        out["avg3_all"] = metrics_from(avg, temps)["all"]
    if teacher_eval is not None:
        agree = [int(torch.argmax(a["logits_per_order"][0])) == int(torch.argmax(b["logits_per_order"][0]))
                 for a, b in zip(eval_records, teacher_eval)]
        out["agree_teacher"] = float(np.mean(agree))
    return out


def _save_records(path, records):
    import torch

    torch.save([{"logits_per_order": [z.half() for z in r["logits_per_order"]], "target": r["target"],
                 "qtype": r["qtype"], "dataset": r["dataset"], "label": r["label"]} for r in records], path)


@app.function(image=image, gpu="H100", cpu=16, memory=65536, timeout=2 * 60 * 60, volumes=VOLUMES)
def train_remote(run: str, mode: str, minutes: float, lr_backbone: float, lr_head: float, lr_new: float,
                 batch_size: int, w_kd: float, w_gt: float, train_lm: str, eval_n: int, seed: int) -> dict:
    import functools
    import random

    import torch

    from laya import two_tower as tt
    from laya.vlm import VLMAgent
    from laya.vlm_train import ItemStream

    t0 = time.time()
    out_dir = os.path.join(OUT, run)
    if os.path.exists(os.path.join(out_dir, "student.pt")):
        raise FileExistsError("%s exists; results are create-only, pick a new --run" % out_dir)
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(seed)
    random.seed(seed)
    train = load_pool("train")
    _log(t0, "pool train: %d examples" % len(train))

    teacher = VLMAgent(TEACHER, device="cuda")
    teacher.model.eval().requires_grad_(False)
    T = torch.tensor(teacher.temperature, device="cuda")
    student = tt.build_from_agent(teacher, mode).cuda()
    student.requires_grad_(True)
    student.encoder.vision_model.requires_grad_(False)
    student.encoder.connector.requires_grad_(False)
    if train_lm != "full":  # "N": only the last N language-model layers (and the final norm)
        n = int(train_lm)
        tm = student.encoder.text_model
        tm.requires_grad_(False)
        for lay in tm.layers[-n:] if n else []:
            lay.requires_grad_(True)
        tm.norm.requires_grad_(n > 0)
    copied = ("head.", "scorer.", "type_emb.") if mode == "li" else ()
    groups = {"backbone": [], "copied": [], "new": []}
    for name, p in student.named_parameters():
        if not p.requires_grad:
            continue
        key = "backbone" if name.startswith("encoder.") else ("copied" if name.startswith(copied) else "new")
        groups[key].append(p)
    lrs = {"backbone": lr_backbone, "copied": lr_head, "new": lr_new}
    pg = [{"params": ps, "lr": lrs[k], "name": k} for k, ps in groups.items() if ps]
    opt = torch.optim.AdamW(pg, weight_decay=0.01)
    n_train = sum(p.numel() for g in pg for p in g["params"])
    _log(t0, "student %s: %.1fM trainable (%s)" % (mode, n_train / 1e6, ", ".join(
        "%s %.1fM" % (g["name"], sum(p.numel() for p in g["params"]) / 1e6) for g in pg)))

    ctx_ids, end_id = tt.option_context_ids(teacher.processor)
    stream = ItemStream(teacher.processor, train, seed, weights=MIX)
    loader = torch.utils.data.DataLoader(
        stream, batch_size=batch_size, num_workers=12, prefetch_factor=4, pin_memory=True,
        collate_fn=functools.partial(tt.collate_two_tower, pad_id=teacher.processor.tokenizer.pad_token_id,
                                     ctx_ids=ctx_ids, end_id=end_id))
    student.train()
    budget = minutes * 60
    warmup = 50
    step, t_train, hist = 0, time.time(), []
    stats = {"loss": 0.0, "kd": 0.0, "gt": 0.0, "agree": 0.0, "acc": 0.0, "tacc": 0.0, "n": 0}
    wait = 0.0
    t_wait = time.time()
    for batch in loader:
        wait += time.time() - t_wait
        el = time.time() - t_train
        if el >= budget:
            break
        prog = el / budget
        f = min(1.0, (step + 1) / warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog)))
        for g in opt.param_groups:
            g["lr"] = lrs[g["name"]] * f
        b = {k: (v.cuda(non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
        mask = b["marker_mask"]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            with torch.no_grad():
                ihs = None
                if b["pixel_values"] is not None:
                    ihs = teacher.model._image_features(b["pixel_values"].to(teacher.model.encoder.dtype),
                                                        b["pixel_attention_mask"])
                t_logits, _ = teacher.model(b["input_ids"], b["attention_mask"], b["marker_pos"], mask, b["qtype"],
                                            image_hidden_states=ihs, option_span=b["option_span"])
            s_logits = student(b, image_hidden_states=ihs)
        t_logits, s_logits = t_logits.float(), s_logits.float()
        p_t = torch.softmax((t_logits / T[b["qtype"]][:, None]).masked_fill(~mask, -1e4), -1)
        tgt = b["target"] / b["target"].sum(-1, keepdim=True).clamp_min(1e-9)
        logp = torch.log_softmax(s_logits.masked_fill(~mask, -1e4), -1)
        kd = -(p_t * logp).sum(-1).mean()
        gt = -(tgt * logp).sum(-1).mean()
        loss = w_kd * kd + w_gt * gt
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for g in pg for p in g["params"]], 1.0)
        opt.step()
        step += 1
        with torch.no_grad():
            sa, ta = s_logits.argmax(-1), t_logits.argmax(-1)
            stats["loss"] += float(loss); stats["kd"] += float(kd); stats["gt"] += float(gt)
            stats["agree"] += float((sa == ta).float().mean()); stats["acc"] += float((sa == b["label"]).float().mean())
            stats["tacc"] += float((ta == b["label"]).float().mean()); stats["n"] += 1
        if step % 50 == 0:
            n = stats.pop("n")
            row = {k: v / n for k, v in stats.items()}
            row.update(step=step, minutes=round(el / 60, 2), lr_f=round(f, 3), data_wait=round(wait / el, 3))
            if mode == "tt":
                row["scale"] = float(student.logit_scale.exp())
            hist.append(row)
            _log(t0, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}))
            stats = {k: 0.0 for k in stats}
            stats["n"] = 0
        t_wait = time.time()
    train_s = time.time() - t_train
    _log(t0, "trained %d steps in %.1f min (%.2f steps/s)" % (step, train_s / 60, step / train_s))
    student.eval()
    torch.save({"mode": mode, "state_dict": {k: v for k, v in student.state_dict().items()
                                             if not k.startswith("encoder.vision_model.")}}, os.path.join(out_dir, "student.pt"))
    args = dict(run=run, mode=mode, minutes=minutes, lr_backbone=lr_backbone, lr_head=lr_head, lr_new=lr_new,
                batch_size=batch_size, w_kd=w_kd, w_gt=w_gt, train_lm=train_lm, seed=seed, teacher=TEACHER, pool=POOL)
    meta = {"args": args, "steps": step, "train_s": train_s, "samples": step * batch_size, "history": hist,
            "trainable_m": n_train / 1e6}
    with open(os.path.join(out_dir, "train.json"), "w") as f:
        json.dump(meta, f, indent=1)
    ckpt_vol.commit()

    # -- evaluation on the same GPU -----------------------------------------------------------------------------
    calib, evals = load_pool("calib"), load_pool("eval")
    if eval_n:
        evals = [ex for i, ex in enumerate(evals) if i % max(1, len(evals) // eval_n) == 0][:eval_n]
        calib = calib[: max(eval_n, 50)]
    _log(t0, "eval: %d calib, %d eval rows" % (len(calib), len(evals)))
    s_cal = tt.collect_logits_tt(student, teacher.processor, calib, batch_size=32, num_workers=8)
    s_ev = tt.collect_logits_tt(student, teacher.processor, evals, orders=three_orders, batch_size=32, num_workers=8)
    t_ev = _teacher_records(teacher, evals)
    res = summarize(s_cal, s_ev, t_ev)
    res.update(meta={"gpu": torch.cuda.get_device_name(0), "n_eval": len(evals), "n_calib": len(calib)}, train=meta)
    _save_records(os.path.join(out_dir, "eval_records.pt"), s_ev)
    with open(os.path.join(out_dir, "eval.json"), "w") as f:
        json.dump(res, f, indent=1)
    ckpt_vol.commit()
    _log(t0, "eval: macro_acc %.4f ece_hard %.4f quality %.4f flip_any3 %.4f agree %.4f"
         % (res["macro_acc"], res["ece_hard"], res["quality"], res.get("flip_any3", -1), res["agree_teacher"]))
    return res


@app.function(image=image, gpu="H100", cpu=16, memory=65536, timeout=60 * 60, volumes=VOLUMES)
def eval_teacher_remote(name: str, eval_n: int) -> dict:
    import torch

    from laya.vlm import VLMAgent

    t0 = time.time()
    out_dir = os.path.join(OUT, name)
    if os.path.exists(os.path.join(out_dir, "eval.json")):
        raise FileExistsError("%s exists; pick a new --name" % out_dir)
    os.makedirs(out_dir, exist_ok=True)
    teacher = VLMAgent(TEACHER, device="cuda")
    calib, evals = load_pool("calib"), load_pool("eval")
    if eval_n:
        evals = [ex for i, ex in enumerate(evals) if i % max(1, len(evals) // eval_n) == 0][:eval_n]
        calib = calib[: max(eval_n, 50)]
    t_cal = _teacher_records(teacher, calib)
    t_ev = _teacher_records(teacher, evals, orders=three_orders)
    res = summarize(t_cal, t_ev)
    # also at the checkpoint's own temperatures (what predict returns)
    from laya.vlm_train import metrics_from

    res["ckpt_temp_all"] = metrics_from(_identity(t_ev), teacher.temperature)["all"]
    res["meta"] = {"gpu": torch.cuda.get_device_name(0), "n_eval": len(evals), "n_calib": len(calib), "teacher": TEACHER}
    _save_records(os.path.join(out_dir, "eval_records.pt"), t_ev)
    with open(os.path.join(out_dir, "eval.json"), "w") as f:
        json.dump(res, f, indent=1)
    ckpt_vol.commit()
    _log(t0, "teacher: macro_acc %.4f ece_hard %.4f quality %.4f flip_reversed %.4f flip_any3 %.4f"
         % (res["macro_acc"], res["ece_hard"], res["quality"], res["flip_reversed"], res["flip_any3"]))
    return res


@app.function(image=image, gpu="L4", cpu=4, memory=16384, timeout=40 * 60, volumes=VOLUMES)
def latency_remote(runs: list, n: int = 50) -> dict:
    """Median ms per decision on an L4 in bf16: one frame (processor pixels already on the GPU; the vision tower
    is timed), one choice question with k options. Teacher: the cross-encoder forward. Students: image features
    + state tower + scoring against option vectors cached beforehand (the one-off option encoding is timed
    separately)."""
    import numpy as np
    import torch

    from laya import two_tower as tt
    from laya.games import ATARI_ACTIONS, atari_question
    from laya.vlm import VLMAgent, _load_image, build_vlm_inputs, collate_vlm, vlm_prefix

    ckpt_vol.reload()
    teacher = VLMAgent(TEACHER, device="cuda", dtype="bf16")
    proc = teacher.processor
    students = {}
    for r in runs:
        ck = torch.load(os.path.join(OUT, r, "student.pt"), map_location="cpu")
        s = tt.build_from_agent(teacher, ck["mode"])
        missing, unexpected = s.load_state_dict(ck["state_dict"], strict=False)
        assert not unexpected and all(k.startswith("encoder.vision_model.") for k in missing), (missing, unexpected)
        students[r] = s.to("cuda").eval()  # backbone bf16 (copied from the bf16 teacher), heads fp32, as VLMAgent
    ex = load_pool("eval", ["cauldron_ai2d"])[0]
    img = ex["state"]["image"]
    prefix = vlm_prefix(proc, [_load_image(img)])
    acts = list(ATARI_ACTIONS)
    base_q = atari_question("Breakout", ["NOOP", "FIRE", "RIGHT", "LEFT"])["action"]

    def question(k):
        if k <= len(acts):
            crit = {a: ATARI_ACTIONS[a] for a in (["NOOP", "FIRE", "RIGHT", "LEFT"] if k == 4 else acts[:k])}
        else:
            crit = {"action %d" % i: "do thing number %d" % i for i in range(k)}
        return {"t": "choice", "ins": base_q["instructions"], "crit": crit}

    def timed(fn):
        for _ in range(10):
            fn()
        ts = []
        for _ in range(n):
            torch.cuda.synchronize()
            t = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            ts.append((time.perf_counter() - t) * 1000)
        return float(np.median(ts)), float(np.percentile(ts, 90))

    ctx_ids, end_id = tt.option_context_ids(proc)
    res = {"gpu": torch.cuda.get_device_name(0), "n": n, "rows": []}
    for k in (2, 4, 18, 64, 255, 1024):
        q = question(k)
        row = {"k": k}
        with torch.inference_mode():
            if k <= 255:
                it = build_vlm_inputs(proc, {}, q, prefix=prefix)
                it.update(qtype=0, target=[1.0] + [0.0] * (k - 1))
                b = collate_vlm([it], proc.tokenizer.pad_token_id)
                b = {kk: (v.cuda() if torch.is_tensor(v) else v) for kk, v in b.items()}
                pv = b["pixel_values"].to(torch.bfloat16)

                def teach():
                    return teacher.model(b["input_ids"], b["attention_mask"], b["marker_pos"], b["marker_mask"],
                                         b["qtype"], pixel_values=pv, pixel_attention_mask=b["pixel_attention_mask"],
                                         option_span=b["option_span"])
                row["teacher_ms"], row["teacher_p90"] = timed(teach)
                row["teacher_tokens"] = int(b["input_ids"].size(1))
                row["teacher_option_tokens"] = int(b["option_span"][0, 1] - b["option_span"][0, 0])
            else:
                it = build_vlm_inputs(proc, {}, question(4), prefix=prefix)
                it.update(qtype=0, target=[1.0, 0, 0, 0])
                b = collate_vlm([it], proc.tokenizer.pad_token_id)
                b = {kk: (v.cuda() if torch.is_tensor(v) else v) for kk, v in b.items()}
                pv = b["pixel_values"].to(torch.bfloat16)
            # the state prefix does not depend on k; options are the question's, tokenised as the cross-encoder does
            from laya.common import render_options
            tok = proc.tokenizer
            opts = [tuple(tok("- " + o.replace("\n", " "), add_special_tokens=False)["input_ids"][:48])
                    for o in render_options(q)]
            s0 = int(b["option_span"][0, 0])
            pids, pmask = b["input_ids"][:, :s0], b["attention_mask"][:, :s0]
            oids, omask = tt._pad(tt.option_sequences(opts, ctx_ids, end_id), tok.pad_token_id)
            oids, omask = oids.cuda(), omask.cuda()
            mm = torch.ones((1, k), dtype=torch.bool, device="cuda")
            qt = torch.zeros(1, dtype=torch.long, device="cuda")
            row["state_tokens"] = s0
            for r, s in students.items():
                def enc_opts():
                    return s.option_cache(s.encode_options(oids, omask))
                row[r + "_option_encode_ms"] = timed(enc_opts)[0]
                cache = enc_opts()[None]

                def step():
                    ihs = s.image_features(pv, b["pixel_attention_mask"])
                    h = s.encode_state(pids, pmask, ihs)
                    return s.score(h, pmask, qt, cache, mm)
                row[r + "_ms"], row[r + "_p90"] = timed(step)
                if r == list(students)[0]:
                    ihs = s.image_features(pv, b["pixel_attention_mask"])
                    h = s.encode_state(pids, pmask, ihs)
                    row["score_only_" + r + "_ms"] = timed(lambda: s.score(h, pmask, qt, cache, mm))[0]
        print(json.dumps(row), flush=True)
        res["rows"].append(row)
    return res


@app.local_entrypoint()
def train(run: str, mode: str = "tt", minutes: float = 30.0, lr_backbone: float = 1e-5, lr_head: float = 5e-5,
          lr_new: float = 3e-4, batch_size: int = 32, w_kd: float = 1.0, w_gt: float = 0.5, train_lm: str = "full",
          eval_n: int = 0, seed: int = 0, out: str = ""):
    res = train_remote.remote(run, mode, minutes, lr_backbone, lr_head, lr_new, batch_size, w_kd, w_gt, train_lm,
                              eval_n, seed)
    _write(out or os.path.join(REPO, "eval-results", "two-tower-%s.json" % run), res)


@app.local_entrypoint()
def eval_teacher(name: str = "teacher-eval", eval_n: int = 0, out: str = ""):
    res = eval_teacher_remote.remote(name, eval_n)
    _write(out or os.path.join(REPO, "eval-results", "two-tower-%s.json" % name), res)


@app.local_entrypoint()
def latency(runs: str, n: int = 50, out: str = ""):
    res = latency_remote.remote([r for r in runs.split(",") if r], n)
    _write(out or os.path.join(REPO, "eval-results", "two-tower-latency.json"), res)


def _write(path, res):
    if os.path.exists(path):
        path = path.replace(".json", "-%d.json" % int(time.time()))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(res, f, indent=1)
    print("wrote", path)
