"""Where laya's token limits come from, measured on the real checkpoint (L4, bf16).

    modal run benchmarks/modal_context_limits.py                      # all experiments -> benchmarks/data/
    modal run benchmarks/modal_context_limits.py --n 100 --out x.json

Experiments (each tests one claim about ``head_max_len`` / ``max_len`` / the backbone's context):

    trace     shapes and parameter counts at every stage of one ``predict`` (vision tower, connector, language
              model, decision head, scorer), from forward hooks
    tokens    tokens per image (1..8 images), per option and per instruction, from the sequence builder
    cost      latency and peak GPU memory of one ``predict`` as the sequence grows (256 .. 16k tokens), and what
              happens past the backbone's 8,192 positions
    state     A-OKVQA accuracy with N tokens of irrelevant state text before the question (N up to ~15k)
    head      A-OKVQA accuracy with the question block (``head_max_len``'s region) padded to 256 .. 4096 tokens
    position  A-OKVQA accuracy with no extra tokens, but the language model's position ids shifted by an offset:
              separates "far positions" from "more text to read" in the state / head results
    options   A-OKVQA accuracy with K options (the 4 real ones plus distractors drawn from the options of the first
              1,000 val questions), K 4 .. 250

A-OKVQA is the first ``--n`` questions of the prepared val split on the laya-datasets volume (4 options each).
The filler is a fixed neutral passage, cut to an exact token count. Every predict runs with ``strict=True``, so
no row is silently truncated; the budgets are raised per condition to fit.
"""
import json
import os
import time

import modal

app = modal.App("laya-context-limits")
hf_vol = modal.Volume.from_name("laya-hf-cache")
data_vol = modal.Volume.from_name("laya-datasets")

MODEL = "thaitea/laya-vision"
REVISION = "f2fe3c12cb6d04c59d8a190250bf3fb40fc828dc"
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.14.0", "torchvision==0.29.0", "transformers==5.17.0", "safetensors", "huggingface_hub",
                 "numpy", "pillow", "num2words")
    .env({"HF_HOME": "/cache/hf", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_python_source("laya")
)

FILLER = ("The municipal archive keeps its ledgers in a cool basement room. Clerks record each delivery by date, "
          "weight and the name of the carrier, and a second clerk checks every entry against the invoice before "
          "the ledger is shelved. Nothing in these records concerns the picture or the question that follows. ")


def _filler(tok, n: int) -> str:
    """Exactly-``n``-token neutral text (as the state or instruction is tokenized)."""
    if n <= 0:
        return ""
    ids = tok(FILLER * (n // 40 + 2), add_special_tokens=False)["input_ids"]
    return tok.decode(ids[:n])


def _set_budget(agent, max_len: int, head_max_len: int) -> None:
    agent.cfg["max_len"], agent.cfg["head_max_len"] = max_len, head_max_len
    agent.processor.laya_max_len = max_len


def _examples(n: int):
    from laya.vlm_train import load_jsonl_examples

    return load_jsonl_examples("/data/vqa", "aokvqa", "val", limit=n)


def _ask(agent, ex, ins_prefix: str = "", context: str = "", extra_opts=None, rng=None):
    """One A-OKVQA question through ``predict``: optional filler before the instructions or as state text,
    optional distractor options (shuffled in with the real ones). Returns (correct, p_correct, answer index)."""
    q = ex["q"]
    opts = list(q["crit"])
    label = ex["label"]
    if extra_opts:
        allopts = opts + [o for o in extra_opts if o not in opts]
        order = list(range(len(allopts)))
        rng.shuffle(order)
        label = order.index(label)
        opts = [allopts[i] for i in order]
    state = {"image": ex["state"]["image"]}
    if context:
        state["context"] = context
    question = {"type": "choice", "instructions": (ins_prefix + "\n" if ins_prefix else "") + q["ins"],
                "criteria": {o: "" for o in opts}}
    a = agent.predict(state, {"q": question}, strict=True)["answers"]["q"]
    p = [a["probabilities"][o] for o in opts]
    best = max(range(len(p)), key=p.__getitem__)
    return best == label, p[label], best


def _summary(rows):
    import numpy as np

    return {"n": len(rows), "acc": float(np.mean([r[0] for r in rows])),
            "p_correct": float(np.mean([r[1] for r in rows]))}


def _trace(agent) -> dict:
    """Shapes at each stage of one predict, and parameter counts per component."""
    import numpy as np
    import torch

    m, enc = agent.model, agent.model.encoder
    shapes = {}

    def hook(name):
        def f(_mod, inp, out):
            t = out[0] if isinstance(out, (tuple, list)) else getattr(out, "last_hidden_state", out)
            if hasattr(t, "shape"):
                shapes.setdefault(name, list(t.shape))
            i0 = inp[0] if inp and hasattr(inp[0], "shape") else None
            if i0 is not None:
                shapes.setdefault(name + ".in", list(i0.shape))
        return f

    hs = [enc.vision_model.register_forward_hook(hook("vision_tower")),
          enc.connector.register_forward_hook(hook("connector")),
          enc.text_model.register_forward_hook(hook("language_model")),
          m.head.register_forward_hook(hook("decision_head")),
          m.scorer.register_forward_hook(hook("scorer"))]
    img = np.full((480, 640, 3), 128, np.uint8)
    q = {"q": {"type": "choice", "instructions": "What colour is the picture?",
               "criteria": {"grey": "", "red": "", "blue": "", "green": ""}}}
    out = agent.predict({"image": img}, q)
    for h in hs:
        h.remove()
    count = lambda mod: sum(p.numel() for p in mod.parameters())  # noqa: E731
    tc = enc.config.text_config
    return {
        "shapes": shapes,
        "input_tokens": out["usage"]["input_tokens"],
        "params": {"vision_tower": count(enc.vision_model), "connector": count(enc.connector),
                   "language_model": count(enc.text_model), "decision_head": count(m.head),
                   "scorer": count(m.scorer), "act_head": count(m.act_head), "total": count(m)},
        "language_model": {"layers": len(enc.text_model.layers), "hidden": tc.hidden_size,
                           "heads": tc.num_attention_heads, "kv_heads": tc.num_key_value_heads,
                           "max_position_embeddings": tc.max_position_embeddings,
                           "rope_theta": getattr(tc, "rope_theta", None),
                           "attn_implementation": enc.config._attn_implementation},
        "vision": {"image_size": enc.config.vision_config.image_size, "patch": enc.config.vision_config.patch_size,
                   "scale_factor": enc.config.scale_factor},
        "head": {"layers": len(m.head.layers), "heads": m.head.layers[0].self_attn.num_heads,
                 "dtype": "fp32 (h.float() before the head)"},
        "option_attention": m.option_attention,
        "cfg": {k: agent.cfg.get(k) for k in ("max_len", "head_max_len", "image_size", "image_split_edge")},
        "gpu": torch.cuda.get_device_name(0),
    }


def _tokens(agent) -> dict:
    import numpy as np

    from laya.vlm import VLMAgent, build_vlm_inputs

    tok = agent.processor.tokenizer
    img = np.full((480, 640, 3), 128, np.uint8)
    q = VLMAgent._to_internal({"type": "choice", "instructions": "Which?", "criteria": {"a": "", "b": ""}})
    base = len(build_vlm_inputs(agent.processor, {}, q, 8192, 256)["ids"])
    per_images = {n: len(build_vlm_inputs(agent.processor, {"images": [img] * n}, q, 8192, 256)["ids"]) - base
                  for n in (1, 2, 4, 8)}
    words = ["left hand up", "close right gripper",
             "LEFT_HAND_FORWARD: move the left hand forward, away from the robot", "a red double-decker bus"]
    return {"image_tokens": per_images, "no_image_tokens": base,
            "option_tokens": {w: len(tok("\n- " + w, add_special_tokens=False)["input_ids"]) for w in words}}


def _cost(agent) -> list:
    """predict latency and peak memory vs total sequence length, via state-text filler (1 image, 4 options)."""
    import numpy as np
    import torch

    tok = agent.processor.tokenizer
    img = np.full((480, 640, 3), 128, np.uint8)
    q = {"q": {"type": "choice", "instructions": "What colour is the picture?",
               "criteria": {"grey": "", "red": "", "blue": "", "green": ""}}}
    rows = []
    for L in (256, 512, 1024, 2048, 4096, 8192, 12288, 16384):
        _set_budget(agent, L + 64, 256)
        ctx = _filler(tok, max(0, L - 150))
        try:
            agent.predict({"image": img, "context": ctx}, q, strict=True)  # warm
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            ts = []
            for _ in range(5):
                t0 = time.perf_counter()
                out = agent.predict({"image": img, "context": ctx}, q, strict=True)
                torch.cuda.synchronize()
                ts.append(time.perf_counter() - t0)
            a = out["answers"]["q"]
            rows.append({"target_len": L, "input_tokens": out["usage"]["input_tokens"],
                         "ms_median": 1000 * float(np.median(ts)),
                         "peak_mem_gb": torch.cuda.max_memory_allocated() / 2 ** 30,
                         "answer": a["choice"], "p_grey": a["probabilities"]["grey"], "error": None})
        except Exception as e:  # noqa: BLE001 - recording the failure is the point
            torch.cuda.empty_cache()
            rows.append({"target_len": L, "error": repr(e)[:300]})
        print(json.dumps(rows[-1]))
    _set_budget(agent, 1024, 256)
    return rows


def _state(agent, exs) -> list:
    tok = agent.processor.tokenizer
    out = []
    base_answers = None
    for n in (0, 500, 1000, 2000, 4000, 7000, 12000, 15000):
        _set_budget(agent, n + 1024, 256)
        ctx = _filler(tok, n)
        try:
            res = [_ask(agent, ex, context=ctx) for ex in exs]
        except Exception as e:  # noqa: BLE001
            out.append({"filler_tokens": n, "error": repr(e)[:300]})
            print(json.dumps(out[-1]))
            continue
        answers = [r[2] for r in res]
        if base_answers is None:
            base_answers = answers
        row = dict(_summary(res), filler_tokens=n,
                   agree_with_no_filler=sum(a == b for a, b in zip(answers, base_answers)) / len(answers))
        out.append(row)
        print(json.dumps(row))
    _set_budget(agent, 1024, 256)
    return out


def _head(agent, exs) -> list:
    tok = agent.processor.tokenizer
    out = []
    base_answers = None
    for h in (256, 512, 1024, 2048, 4096):
        pad = max(0, h - 200)  # the real question + 4 options fit in well under 200 tokens
        _set_budget(agent, h + 1024, h)
        prefix = _filler(tok, pad) if h > 256 else ""
        res = [_ask(agent, ex, ins_prefix=prefix) for ex in exs]
        answers = [r[2] for r in res]
        if base_answers is None:
            base_answers = answers
        pad_tokens = len(tok(prefix, add_special_tokens=False)["input_ids"]) if prefix else 0
        row = dict(_summary(res), head_max_len=h, instruction_filler_tokens=pad_tokens,
                   agree_with_256=sum(a == b for a, b in zip(answers, base_answers)) / len(answers))
        out.append(row)
        print(json.dumps(row))
    _set_budget(agent, 1024, 256)
    return out


def _position(agent, exs) -> list:
    """Same questions, same tokens; the text model sees position ids ``offset .. offset + L`` instead of ``0 .. L``."""
    import torch

    tm = agent.model.encoder.text_model
    offset = {"v": 0}

    def shift(_mod, args, kwargs):
        if offset["v"]:
            ref = kwargs.get("inputs_embeds")
            if ref is None:
                ref = kwargs.get("input_ids")
            L = ref.shape[1]
            kwargs["position_ids"] = torch.arange(offset["v"], offset["v"] + L, device=ref.device)[None]
        return args, kwargs

    h = tm.register_forward_pre_hook(shift, with_kwargs=True)
    out, base_answers = [], None
    try:
        for off in (0, 1000, 2000, 4000, 7000, 12000):
            offset["v"] = off
            res = [_ask(agent, ex) for ex in exs]
            answers = [r[2] for r in res]
            if base_answers is None:
                base_answers = answers
            row = dict(_summary(res), position_offset=off,
                       agree_with_no_offset=sum(a == b for a, b in zip(answers, base_answers)) / len(answers))
            out.append(row)
            print(json.dumps(row))
    finally:
        h.remove()
    return out


def _options(agent, exs) -> list:
    import random

    pool = sorted({o for ex in _examples(1000) for o in ex["q"]["crit"]})  # distractors: other questions' options
    out = []
    for k in (4, 8, 16, 32, 64, 128, 250):
        _set_budget(agent, 8192, 4096)
        rng = random.Random(k)
        res = []
        for ex in exs:
            own = list(ex["q"]["crit"])
            extra = rng.sample([o for o in pool if o not in own], k - len(own)) if k > len(own) else []
            res.append(_ask(agent, ex, extra_opts=extra, rng=rng) if extra else _ask(agent, ex))
        row = dict(_summary(res), options=k, chance=1.0 / k)
        out.append(row)
        print(json.dumps(row))
    _set_budget(agent, 1024, 256)
    return out


@app.function(image=image, gpu="L4", cpu=4, timeout=180 * 60,
              volumes={"/cache/hf": hf_vol, "/data": data_vol.read_only()})
def run(n: int = 200, parts: str = "trace,tokens,cost,state,head,position,options") -> dict:
    import torch

    from laya.vlm import VLMAgent

    agent = VLMAgent(MODEL, revision=REVISION, device="cuda", dtype="bf16")
    want = parts.split(",")
    res = {"model": MODEL, "revision": REVISION, "n": n, "torch": str(torch.__version__)}
    exs = _examples(n) if {"state", "head", "position", "options"} & set(want) else []
    for name, fn in (("trace", lambda: _trace(agent)), ("tokens", lambda: _tokens(agent)),
                     ("cost", lambda: _cost(agent)), ("state", lambda: _state(agent, exs)),
                     ("head", lambda: _head(agent, exs)), ("position", lambda: _position(agent, exs)),
                     ("options", lambda: _options(agent, exs))):
        if name in want:
            t0 = time.time()
            res[name] = fn()
            print("== %s done in %.0fs" % (name, time.time() - t0))
    return json.loads(json.dumps(res, default=str))  # plain JSON: the local client has no torch


@app.local_entrypoint()
def main(n: int = 200, parts: str = "trace,tokens,cost,state,head,position,options", out: str = ""):
    path = out or "benchmarks/data/context-limits-%s.json" % time.strftime("%Y%m%d-%H%M%S")
    if os.path.exists(path):
        raise SystemExit("%s exists; results are create-only" % path)
    res = run.remote(n, parts)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(res, f, indent=1)
    print("wrote", path)
