"""Serving throughput: how many ``predict`` requests per second one GPU container answers, and what a call costs.

A request is one image and one typed question, the shape a serving endpoint gets. The benchmark times four things
on the same real validation images and questions:

``sequential``  ``VLMAgent.predict`` called once per request, one after another: what a naive endpoint does. At
                batch 1 the GPU is launch-bound (site-docs/concepts/game-caching.md), so this is the floor.
``gpu``         Requests batched across users: B images through the vision tower in one pass, then B question
                rows through the language model in one pass. CPU preprocessing is done before the clock starts,
                so this is what the GPU alone can serve at batch B.
``cpu``         The CPU half of a request (the processor's resize and normalisation, tokenisation, row building)
                on a thread pool, without the GPU: what the container's cores can feed.
``pipelined``   Both at once, as a server would run them: a thread pool preprocesses requests while the main
                thread forwards batches of B. This is the end-to-end requests/second the cost column uses.

The batched path is the same computation as ``predict`` (``build_vlm_inputs`` rows, ``collate_vlm``, the model's
forward) with the rows of different requests stacked; ``agreement`` compares its raw logits and argmax with
``predict``'s on the same requests. Padding shorter rows can move bf16 logits by rounding, never the method.

Cost per million requests is the container's Modal list price (GPU + requested cores + memory, recorded in
``pricing``) divided by the measured requests/second, i.e. a container kept 100% busy. Real traffic that leaves
it idle part of the time costs proportionally more.

    modal run modal_app.py::serving_throughput --output results/raw/serving-throughput-l4.json
"""
from __future__ import annotations

import os
import platform
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from benchmarks.decision_vs_generation import code_revision  # noqa: E402

MODEL = "thaitea/laya-vision"
REVISION = "8b318c99d7ad3ce19c24369263463882eada9d1e"  # the 201M checkpoint the README recommends
VERSION = "laya-serving-throughput-v1"
PROVENANCE_KEYS = ("prompt_format_version", "checkpoint", "backbone", "dtype", "readout", "option_attention")

# Modal list prices, USD per second, read from https://modal.com/pricing on 2026-09-24.
MODAL_PRICES = {
    "source": "https://modal.com/pricing",
    "read_on": "2026-09-24",
    "gpu_per_second": {"L4": 0.000222, "A10G": 0.000306, "A100-40GB": 0.000583, "A100-80GB": 0.000694,
                       "H100": 0.001097},
    "cpu_core_per_second": 0.0000131,
    "memory_gib_per_second": 0.00000222,
}


def prepare(agent, state, question) -> dict:
    """The CPU half of one request: the image prefix and the question's one row, as ``predict`` builds them."""
    from laya.common import QTYPES, render_options
    from laya.vlm import build_vlm_inputs, split_state, vlm_prefix

    images, _ = split_state(state)
    prefix = vlm_prefix(agent.processor, images, agent.prep)
    q = agent._to_internal(question)
    k = len(render_options(q))
    row = build_vlm_inputs(agent.processor, state, q, agent.cfg.get("max_len", 1024),
                           agent.cfg.get("head_max_len", 256), option_order=list(range(k)), prefix=prefix)
    row.update(qtype=QTYPES[q["t"]])
    return {"prefix": prefix, "row": row, "k": k}


def forward(agent, reqs) -> list:
    """The GPU half for a batch of prepared requests: one vision pass, one language-model pass. Returns each
    request's raw option logits (before temperature), as ``predict``'s ``_raw_logits`` holds them."""
    import torch

    from laya.vlm import collate_vlm

    m, dev, dt = agent.model, agent.device, agent._torch_dtype()
    with torch.no_grad():
        prefixes = [r["prefix"] for r in reqs]
        if all(p["raw_images"] is not None for p in prefixes):  # device-side preprocessing checkpoints
            feats = torch.cat([m.encode_raw_images(p["raw_images"]) for p in prefixes])
        elif len({tuple(p["pixel_values"].shape) for p in prefixes}) == 1:
            pv = torch.cat([p["pixel_values"] for p in prefixes]).to(dev, dt)
            pam = torch.cat([p["pixel_attention_mask"] for p in prefixes]).to(dev)
            feats = m.encode_images(pv, pam)
        else:  # tiled images with different view counts: encode each, the rows still batch
            feats = torch.cat([m.encode_images(p["pixel_values"].to(dev, dt), p["pixel_attention_mask"].to(dev))
                               for p in prefixes])
        b = collate_vlm([r["row"] for r in reqs], agent.processor.tokenizer.pad_token_id, with_pixels=False)
        args = [b[k].to(dev) for k in ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")]
        logits, _ = m(*args, image_hidden_states=feats, option_span=b["option_span"].to(dev))
        logits = logits.float().cpu().numpy()
    return [logits[i, : r["k"]] for i, r in enumerate(reqs)]


def _sync():
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _stats(xs) -> dict:
    xs = sorted(xs)
    return {"median": statistics.median(xs), "p90": xs[min(len(xs) - 1, int(0.9 * len(xs)))], "n": len(xs)}


def run(cases, batch_sizes=(1, 2, 4, 8, 16, 32, 64, 128), n_sequential: int = 200, threads=(1, 2, 4, 8),
        gpu_name: str = "L4", cpu_cores: float = 8, memory_gib: float = 16, dtype: str = "bf16",
        device: str = "cuda", model: str = MODEL, revision: str = REVISION, dataset_note: str = "") -> dict:
    """``cases`` is a list of ``(state, question)`` with the state's image already decoded to PIL."""
    import numpy as np
    import torch
    import transformers

    from laya.vlm import VLMAgent

    agent = VLMAgent(model, device=device, dtype=dtype, revision=revision)
    n = len(cases)
    for state, q in cases[:5]:  # warm-up: kernels, allocator
        agent.predict(state, {"q": q})
    forward(agent, [prepare(agent, s, q) for s, q in cases[:8]])

    # sequential predict, and the reference logits for the agreement check
    seq_ms, ref, tokens = [], [], []
    for state, q in cases[:n_sequential]:
        raw = {}
        _sync()
        t0 = time.perf_counter()
        res = agent.predict(state, {"q": q}, _raw_logits=raw)
        _sync()
        seq_ms.append((time.perf_counter() - t0) * 1000)
        ref.append(raw["q"])
        tokens.append(res["usage"]["input_tokens"])
    sequential = {"latency_ms": _stats(seq_ms), "requests_per_second": 1000 * len(seq_ms) / sum(seq_ms)}

    # CPU preprocessing on a thread pool, no GPU
    cpu = {}
    for t in threads:
        with ThreadPoolExecutor(t) as ex:
            t0 = time.perf_counter()
            list(ex.map(lambda c: prepare(agent, *c), cases))
            cpu[str(t)] = {"requests_per_second": n / (time.perf_counter() - t0)}
    best_threads = max(threads, key=lambda t: cpu[str(t)]["requests_per_second"])

    prepared = [prepare(agent, s, q) for s, q in cases]
    gpu, pipelined, agreement = {}, {}, {}
    for bs in batch_sizes:
        if bs > n:
            break
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
        try:
            got, batch_ms = [], []
            for s in range(0, n - bs + 1, bs):  # full batches only, so every batch is size bs
                _sync()
                t0 = time.perf_counter()
                got += forward(agent, prepared[s: s + bs])
                _sync()
                batch_ms.append((time.perf_counter() - t0) * 1000)
        except torch.cuda.OutOfMemoryError:
            gpu[str(bs)] = {"oom": True}
            torch.cuda.empty_cache()
            break
        gpu[str(bs)] = {"batch_latency_ms": _stats(batch_ms),
                        "requests_per_second": 1000 * bs * len(batch_ms) / sum(batch_ms),
                        "peak_cuda_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None}
        k = min(len(got), len(ref))
        diffs = [float(np.abs(got[i] - ref[i]).max()) for i in range(k)]
        agreement[str(bs)] = {"n": k, "argmax_matches": sum(int(got[i].argmax() == ref[i].argmax()) for i in range(k)),
                              "max_abs_logit_diff": max(diffs), "median_abs_logit_diff": statistics.median(diffs)}

        # pipelined: preprocessing threads run ahead while the main thread forwards batches
        with ThreadPoolExecutor(best_threads) as ex:
            _sync()
            t0 = time.perf_counter()
            it = ex.map(lambda c: prepare(agent, *c), cases)
            batch = []
            for r in it:
                batch.append(r)
                if len(batch) == bs:
                    forward(agent, batch)
                    batch = []
            if batch:
                forward(agent, batch)
            _sync()
            pipelined[str(bs)] = {"requests_per_second": n / (time.perf_counter() - t0), "threads": best_threads}

    price_s = (MODAL_PRICES["gpu_per_second"][gpu_name] + cpu_cores * MODAL_PRICES["cpu_core_per_second"]
               + memory_gib * MODAL_PRICES["memory_gib_per_second"])

    def cost(rps):
        return {"usd_per_million_requests": 1e6 * price_s / rps}

    best_bs = max(pipelined, key=lambda b: pipelined[b]["requests_per_second"])
    return {
        "version": VERSION,
        "code": code_revision(),
        "hardware": {"gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                     "gpu_priced_as": gpu_name, "cpu_cores_requested": cpu_cores, "memory_gib_requested": memory_gib,
                     "os_cpu_count": os.cpu_count(), "cpu": platform.machine()},
        "software": {"torch": torch.__version__, "transformers": transformers.__version__,
                     "python": platform.python_version()},
        "dtype": dtype,
        "model": {"source": model, "revision": revision,
                  "provenance": {k: v for k, v in agent.provenance([], 1, {}).items() if k in PROVENANCE_KEYS}},
        "input": {"n_requests": n, "datasets": dataset_note, "questions_per_request": 1,
                  "mean_input_tokens": float(np.mean(tokens))},
        "pricing": dict(MODAL_PRICES, container_usd_per_second=price_s),
        "sequential": dict(sequential, **cost(sequential["requests_per_second"])),
        "cpu_preprocessing": cpu,
        "gpu_batched": {b: dict(v, **cost(v["requests_per_second"])) if "requests_per_second" in v else v
                        for b, v in gpu.items()},
        "pipelined": {b: dict(v, **cost(v["requests_per_second"])) for b, v in pipelined.items()},
        "agreement_with_predict": agreement,
        "best": {"batch_size": int(best_bs), **pipelined[best_bs], **cost(pipelined[best_bs]["requests_per_second"])},
    }


def summary(report: dict) -> dict:
    """The headline rows: requests/second and $ per million requests, sequential versus batched."""
    rows = {"sequential": {k: round(report["sequential"][k], 2) for k in ("requests_per_second",
                                                                          "usd_per_million_requests")}}
    for b, v in report["pipelined"].items():
        g = report["gpu_batched"][b]
        rows["batch %s" % b] = {"pipelined_rps": round(v["requests_per_second"], 1),
                                "gpu_only_rps": round(g["requests_per_second"], 1),
                                "batch_p50_ms": round(g["batch_latency_ms"]["median"], 1),
                                "usd_per_million_requests": round(v["usd_per_million_requests"], 2)}
    rows["cpu_preprocessing_rps"] = {t: round(v["requests_per_second"], 1)
                                     for t, v in report["cpu_preprocessing"].items()}
    rows["agreement"] = {b: "%d/%d argmax, max |dlogit| %.3g" % (a["argmax_matches"], a["n"], a["max_abs_logit_diff"])
                         for b, a in report["agreement_with_predict"].items()}
    return rows
