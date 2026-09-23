"""Repeat / batch invariance of the VLM decision model (``laya.vlm``): does a row's output depend on what else is
computed with it?

The model is deterministic in principle: a row's logits are a function of its own input ids and pixels. Right
padding is masked out of every attention (causal backbone, head transformer, readout), padded image slots are
all-zero and dropped by the vision tower, and the prefix cache replays the same keys and values. So every
condition below should match its reference to float rounding (the kernels a batch of a different shape picks may
sum in another order), and never flip the answer. Conditions, each compared with a reference per row:

Batch path (``laya.vlm_train.collect_logits``, the path the evals and ``laya.robustness`` use); reference
``alone``: every row in a batch of its own (``batch_size=1``).

* ``repeat``: the same again.
* ``batched``: the rows in batches of ``batch_size`` in the given order, so each is padded to its batch's longest.
* ``batched_reversed``: the same batches built from the reversed row list (other neighbours, other positions).
* ``hostile``: every chunk of rows batched behind ``hostile_rows``: a much longer text state (so every real row
  is mostly padding), a two-image state (ragged image counts), a text-only state (no image slot at all) and an
  eight-option question (ragged marker counts).

``VLMAgent.predict`` path (only when given a ``VLMAgent``); the rows are grouped by state, as a caller would send
them; reference ``predict_alone``: one question per call, ``prefix_cache=False``.

* ``predict_repeat``: the same again.
* ``predict_multi``: all of a state's questions in one call (``prefix_cache=False``), so they share batches.
* ``predict_multi_hostile``: the same plus a hostile eight-option question with long instructions and options.
* ``predict_prefix_cache``: one question per call with ``prefix_cache=True`` (the prefix pass plus a suffix pass).
* ``predict_multi_prefix_cache``: all of a state's questions in one call with ``prefix_cache=True``.

Cross-path and precision, reported, not expected to be zero-tolerance:

* ``predict_vs_batch``: ``predict_alone`` against ``alone``. Same ids (``predict`` encodes the images once and
  hands the features in, ``collect_logits`` runs the vision tower in the forward); on CUDA ``collect_logits``
  runs under bf16 autocast and ``predict`` does not, so this is a precision gap there, not a batching one.
* ``bf16`` (``bf16=True``): the batch path with a copy whose backbone is bf16 (as ``dtype="bf16"`` builds it; the
  head stays fp32), against the fp32 ``alone``.

For every condition and row: ``max_abs_dprob`` (max over options of |p - p_ref|, probabilities under the given
temperatures), ``max_abs_dlogit`` and ``flip`` (argmax differs); per condition a summary with the max and mean of
both and the flip count. Everything returned is JSON-able::

    python -m laya.robustness_invariance --data-root <root> --datasets sq --n 50 --out invariance.json
"""
import copy
import json
from typing import Dict, List, Optional, Sequence

import numpy as np

BATCH_CONDITIONS = ("repeat", "batched", "batched_reversed", "hostile")
PREDICT_CONDITIONS = ("predict_repeat", "predict_multi", "predict_multi_hostile", "predict_prefix_cache",
                      "predict_multi_prefix_cache")
REPORT_ONLY = ("predict_vs_batch", "bf16")

#: filler for the long-state hostile neighbour, repeated to ``LONG_STATE_WORDS`` words
_FILLER = ("The ledger lists every crate that came through the north gate that week, with its weight, its origin, "
           "the inspector's initials and a note on its condition. ")
LONG_STATE_WORDS = 300
_MANY = ["a weathered wooden signpost pointing towards the harbour", "three gulls perched on a railing",
         "a stack of blue fishing crates", "an orange lifebuoy hanging from a hook", "a bicycle leaning on a wall",
         "a chalkboard menu outside a cafe", "a coil of rope on the quay", "a lighthouse on a distant headland"]


# ---------------------------------------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------------------------------------


def _images_of(state) -> List:
    if not isinstance(state, dict):
        return []
    if state.get("image") is not None:
        return [state["image"]]
    return list(state.get("images") or [])


def _row(state, q: Dict, name: str) -> Dict:
    from .common import render_options

    k = len(render_options(q))
    return {"id": "hostile|" + name, "state": state, "q": q, "target": [1.0] + [0.0] * (k - 1), "label": 0,
            "dataset": "_hostile"}


def hostile_rows(rows: Sequence[Dict]) -> List[Dict]:
    """Neighbours built to stress padding and batching, from ``rows``' own images (none when no row has one)."""
    imgs = [im for r in rows for im in _images_of(r["state"])]
    long_text = " ".join((_FILLER * (LONG_STATE_WORDS // len(_FILLER.split()) + 1)).split()[:LONG_STATE_WORDS])
    noul = {"t": "noul", "ins": "The crates were inspected.", "crit": None}
    out = [_row(dict({"image": imgs[0]} if imgs else {}, context=long_text), noul, "long_state")]
    if imgs:
        out.append(_row({"images": [imgs[0], imgs[-1]]}, noul, "two_images"))
    out.append(_row({"context": "A short note."}, noul, "text_only"))
    many = {"t": "choice", "ins": "Which of these objects appears most prominently in the picture, looking at the "
                                  "foreground first and then the background?", "crit": {c: None for c in _MANY}}
    out.append(_row({"image": imgs[0]} if imgs else "", many, "many_options"))
    return out


def _public_question(q: Dict) -> Dict:
    """An internal ``{"t", "ins", "crit"}`` question in ``predict``'s input format."""
    crit = list(q["crit"]) if q["t"] == "choice" else q["crit"]
    return {"type": q["t"], "instructions": q["ins"], "criteria": crit}


def _prepare(rows: Sequence[Dict]) -> List[Dict]:
    """Rows with unique ids and identity option order (the ``predict`` path has no per-row order)."""
    out, seen = [], set()
    for i, r in enumerate(rows):
        rid = str(r.get("id", i))
        if rid in seen:
            raise ValueError("duplicate row id %r" % rid)
        seen.add(rid)
        out.append({k: v for k, v in r.items() if k != "order"} | {"id": rid})
    return out


# ---------------------------------------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------------------------------------


def _collect(model, processor, rows: List[Dict], batch_size: int) -> List[np.ndarray]:
    from .vlm_train import collect_logits

    return [rec["logits"].double().numpy() for rec in collect_logits(model, processor, rows, batch_size=batch_size)]


def _collect_hostile(model, processor, rows: List[Dict], hostile: List[Dict], batch_size: int) -> List[np.ndarray]:
    """Each chunk of ``max(1, batch_size - len(hostile))`` rows in one batch behind every hostile row."""
    c = max(1, batch_size - len(hostile))
    chunks = [rows[s: s + c] for s in range(0, len(rows), c)]
    out: List[np.ndarray] = []
    for ch in chunks:  # one call per chunk: the loader's batch is then exactly hostile + chunk
        z = _collect(model, processor, hostile + ch, len(hostile) + len(ch))
        out += z[len(hostile):]
    return out


def _by_state(rows: List[Dict]) -> Dict[str, List[Dict]]:
    groups: Dict[str, List[Dict]] = {}
    for r in rows:
        groups.setdefault(json.dumps(r["state"], sort_keys=True, default=str), []).append(r)
    return groups


def _predict(agent, rows: List[Dict], together: bool, prefix_cache: bool, batch_size: int,
             extra: Optional[Dict] = None) -> Dict[str, np.ndarray]:
    """Raw label-order logits per row id from ``agent.predict`` (one call per state, or per row)."""
    out: Dict[str, np.ndarray] = {}
    for group in _by_state(rows).values():
        calls = [group] if together else [[r] for r in group]
        for rs in calls:
            qs = {r["id"]: _public_question(r["q"]) for r in rs}
            if extra:
                qs.update({k: v for k, v in extra.items() if k not in qs})
            raw: Dict[str, np.ndarray] = {}
            agent.predict(rs[0]["state"], qs, batch_size=batch_size, prefix_cache=prefix_cache, _raw_logits=raw)
            for r in rs:
                out[r["id"]] = np.asarray(raw[r["id"]], dtype=np.float64)
    return out


# ---------------------------------------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------------------------------------


def _softmax(z: np.ndarray, t: float) -> np.ndarray:
    z = z / max(1e-3, float(t))
    p = np.exp(z - z.max())
    return p / p.sum()


def compare(rows: List[Dict], ref: List[np.ndarray], got: List[np.ndarray], temperatures: Sequence[float]) -> Dict:
    """Per-row deltas of ``got`` against ``ref`` (both label-order logits aligned with ``rows``) and their summary."""
    from .common import QTYPES

    per = []
    for r, a, b in zip(rows, ref, got):
        t = temperatures[QTYPES[r["q"]["t"]]]
        pa, pb = _softmax(a, t), _softmax(b, t)
        per.append({"id": r["id"], "max_abs_dprob": float(np.abs(pa - pb).max()),
                    "max_abs_dlogit": float(np.abs(a - b).max()), "flip": bool(pa.argmax() != pb.argmax()),
                    "pred_ref": int(pa.argmax()), "pred": int(pb.argmax())})
    return {"rows": per, "summary": summarize_rows(per)}


def summarize_rows(per: Sequence[Dict]) -> Dict:
    if not per:
        return {"n": 0}
    dp = np.array([r["max_abs_dprob"] for r in per])
    dz = np.array([r["max_abs_dlogit"] for r in per])
    return {"n": len(per), "max_abs_dprob": float(dp.max()), "mean_abs_dprob": float(dp.mean()),
            "max_abs_dlogit": float(dz.max()), "mean_abs_dlogit": float(dz.mean()),
            "n_flips": int(sum(r["flip"] for r in per)), "n_exact": int((dz == 0).sum())}


def run_invariance(model_or_agent, rows: Sequence[Dict], processor=None, temperatures: Optional[Sequence[float]] = None,
                   batch_size: int = 8, conditions: Optional[Sequence[str]] = None, bf16: bool = False) -> Dict:
    """Run every condition (``BATCH_CONDITIONS``, and ``PREDICT_CONDITIONS`` + ``predict_vs_batch`` when given a
    ``VLMAgent``; ``bf16`` on request) over ``rows`` (examples as ``load_jsonl_examples`` / ``laya.robustness``
    make them) and compare each with its reference. ``conditions`` restricts the set. Returns
    ``{"conditions": {name: {"reference", "rows", "summary"}}, "summary": {name: summary}, "meta": {...}}``."""
    import torch

    from .vlm import VLMAgent

    agent = model_or_agent if isinstance(model_or_agent, VLMAgent) else None
    model = agent.model if agent else model_or_agent
    processor = agent.processor if agent else processor
    if processor is None:
        raise ValueError("pass a VLMAgent, or a model and its processor")
    temps = list(temperatures if temperatures is not None else (agent.temperature if agent else (1.0, 1.0, 1.0)))
    wanted = set(conditions) if conditions else set(BATCH_CONDITIONS) | (
        set(PREDICT_CONDITIONS) | {"predict_vs_batch"} if agent else set()) | ({"bf16"} if bf16 else set())
    known = set(BATCH_CONDITIONS) | set(PREDICT_CONDITIONS) | set(REPORT_ONLY)
    if wanted - known:
        raise ValueError("unknown conditions %s (expected %s)" % (sorted(wanted - known), sorted(known)))
    if agent is None and wanted & (set(PREDICT_CONDITIONS) | {"predict_vs_batch"}):
        raise ValueError("the predict conditions need a VLMAgent")
    rows = _prepare(rows)
    hostile = hostile_rows(rows)
    res: Dict[str, Dict] = {}

    def add(name, reference, ref, got):
        res[name] = dict(compare(rows, ref, got, temps), reference=reference)

    alone = _collect(model, processor, rows, 1) if wanted & (set(BATCH_CONDITIONS) | set(REPORT_ONLY)) else None
    if "repeat" in wanted:
        add("repeat", "alone", alone, _collect(model, processor, rows, 1))
    if "batched" in wanted:
        add("batched", "alone", alone, _collect(model, processor, rows, batch_size))
    if "batched_reversed" in wanted:
        add("batched_reversed", "alone", alone, _collect(model, processor, rows[::-1], batch_size)[::-1])
    if "hostile" in wanted:
        add("hostile", "alone", alone, _collect_hostile(model, processor, rows, hostile, batch_size))
    if "bf16" in wanted:
        m16 = copy.deepcopy(model).eval()
        m16.encoder.to(torch.bfloat16)  # as ``dtype="bf16"`` builds it: the backbone in bf16, the head in fp32
        add("bf16", "alone", alone, _collect(m16, processor, rows, batch_size))
        del m16
    if agent is not None and wanted & (set(PREDICT_CONDITIONS) | {"predict_vs_batch"}):
        ids = [r["id"] for r in rows]
        p_alone = _predict(agent, rows, False, False, batch_size)
        ref = [p_alone[i] for i in ids]
        variants = {
            "predict_repeat": lambda: _predict(agent, rows, False, False, batch_size),
            "predict_multi": lambda: _predict(agent, rows, True, False, batch_size),
            "predict_multi_hostile": lambda: _predict(
                agent, rows, True, False, batch_size,
                extra={"hostile|many_options": _public_question(hostile[-1]["q"])}),
            "predict_prefix_cache": lambda: _predict(agent, rows, False, True, batch_size),
            "predict_multi_prefix_cache": lambda: _predict(agent, rows, True, True, batch_size),
        }
        for name in PREDICT_CONDITIONS:
            if name in wanted:
                got = variants[name]()
                add(name, "predict_alone", ref, [got[i] for i in ids])
        if "predict_vs_batch" in wanted:
            add("predict_vs_batch", "alone", alone, ref)
    dev = next(model.parameters()).device
    meta = {"n_rows": len(rows), "batch_size": batch_size, "temperatures": temps, "device": str(dev),
            "dtype": str(next(model.parameters()).dtype).replace("torch.", ""),
            "batch_path_autocast_bf16": dev.type == "cuda", "hostile": [h["id"] for h in hostile],
            "torch": torch.__version__, "report_only": [c for c in REPORT_ONLY if c in res]}
    return {"conditions": res, "summary": {k: v["summary"] for k, v in res.items()}, "meta": meta}


def format_table(result: Dict) -> str:
    """A markdown table: one line per condition."""
    lines = ["| condition | vs | rows | max abs dp | mean abs dp | max abs dlogit | flips | exact |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for name, c in result["conditions"].items():
        s = c["summary"]
        lines.append("| %s%s | %s | %d | %.2e | %.2e | %.2e | %d | %d |" % (
            name, " (report only)" if name in REPORT_ONLY else "", c["reference"], s["n"], s["max_abs_dprob"],
            s["mean_abs_dprob"], s["max_abs_dlogit"], s["n_flips"], s["n_exact"]))
    return "\n".join(lines)


def main(argv=None):
    import argparse

    import torch

    from . import robustness as R
    from .vlm import VLMAgent
    from .vlm_train import load_jsonl_examples

    ap = argparse.ArgumentParser(description="Repeat / batch / prefix-cache invariance of a VLM checkpoint.")
    ap.add_argument("--model", default="", help="checkpoint path or Hub id (default: a fresh untrained agent)")
    ap.add_argument("--backbone", default="HuggingFaceTB/SmolVLM-256M-Instruct", help="backbone of a fresh agent")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--data-root", required=True, help="directory holding <dataset>/<split>.jsonl")
    ap.add_argument("--datasets", default="", help="comma-separated dataset names under --data-root")
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=50, help="rows per dataset (seeded sample; 0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default=None)
    ap.add_argument("--bf16", action="store_true", help="also report a bf16 copy of the model against fp32")
    ap.add_argument("--conditions", default="", help="comma-separated subset of conditions")
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    if a.model:
        agent = VLMAgent(a.model, device=a.device, revision=a.revision)
    else:
        torch.manual_seed(a.seed)
        agent = VLMAgent(backbone=a.backbone, device=a.device, preprocess="processor")
    rows = []
    for name in [d for d in a.datasets.split(",") if d]:
        rows += R.source_rows(load_jsonl_examples(a.data_root, name, a.split), n=a.n, seed=a.seed, dataset=name)
    res = run_invariance(agent, rows, batch_size=a.batch_size, bf16=a.bf16,
                         conditions=[c for c in a.conditions.split(",") if c] or None)
    res["meta"].update(model=a.model or "fresh:" + a.backbone, datasets=a.datasets, split=a.split, n=a.n, seed=a.seed)
    print(format_table(res))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)
    return res


if __name__ == "__main__":
    main()
