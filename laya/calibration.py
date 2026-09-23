"""Post-hoc temperature calibration of a checkpoint on the user's own labelled data.

A checkpoint ships with temperatures fitted on its validation sets (``fit_temperatures_from`` in ``vlm_train.py``).
On a different workload the model can be over- or under-confident, and a probability threshold then means little.
This module refits the temperature on the user's rows and reports whether that actually helped, without touching
the model or the checkpoint:

* **One scalar ``T`` per question type** (``choice`` / ``score`` / ``noul``), or one pooled ``T``, fitted by minimising
  the mean negative log-likelihood of the labels under ``softmax(logits / T)``. A type needs ``min_rows`` labelled
  questions to get its own ``T``; types below that share a pooled ``T`` fitted on all rows, and if there are fewer
  than ``min_rows`` rows in total nothing is fitted and every type keeps the checkpoint's value.
* **Accuracy cannot change.** Dividing logits by a positive ``T`` is monotone, so the argmax (the answer) is the same;
  only the probabilities move. The report checks this rather than assuming it.
* **Honest evidence.** ECE of the fitted ``T`` is measured out-of-fold: rows are split into ``folds`` group-disjoint
  folds (by ``group_key``, e.g. the image id, so several questions about one image never sit on both sides of a
  split), ``T`` is refitted on the other folds with the same rules and scored on the held-out one. The raw (``T=1``)
  and checkpoint-temperature ECE are reported next to it, each with a 95% bootstrap interval that resamples groups,
  plus the paired interval of the improvement. ECE here is top-label: confidence is the largest option probability,
  15 equal-width bins (``laya.common.ece_score``, as in training).

The numeric part (fitting, folds, ECE, bootstrap) is plain numpy on per-question logit arrays and needs no model;
``VLMAgent.calibrate`` collects the logits and calls ``calibrate_records``. ``Calibration`` is the result: the
temperatures, the evidence and the identity of the checkpoint it was fitted for, saved as JSON and passed to
``VLMAgent.predict(..., calibration=cal)``. Recipe after SemIf's per-workload temperature scaling.
"""
import dataclasses
import hashlib
import json
import math
import time
import warnings
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .common import QTYPES, QTYPE_NAMES, ece_score

T_BOUNDS = (0.05, 20.0)
CONFIG_IGNORED = ("temperature", "temperature_by_options", "dtype")


# ---------------------------------------------------------------------------------------------------------
# Temperature overrides
# ---------------------------------------------------------------------------------------------------------


def check_temperature(t: Any, what: str = "temperature") -> float:
    """A temperature must be a finite number > 0 (dividing by zero or a negative flips or breaks the argmax)."""
    if isinstance(t, bool) or not isinstance(t, (int, float, np.floating, np.integer)):
        raise TypeError("%s must be a number, got %r" % (what, t))
    t = float(t)
    if not math.isfinite(t) or t <= 0:
        raise ValueError("%s must be finite and > 0, got %r" % (what, t))
    return t


def resolve_temperature(temperature: Any = None, calibration: Optional["Calibration"] = None) -> Dict[int, float]:
    """Per-call override for ``predict``: ``{qtype index: T}``; types absent keep the checkpoint's temperature.

    ``temperature`` is a number (every type) or a dict keyed by type name; ``calibration`` contributes its fitted
    temperatures. Passing both is ambiguous and rejected.
    """
    if temperature is not None and calibration is not None:
        raise ValueError("pass either temperature= or calibration=, not both")
    if calibration is not None:
        temperature = dict(calibration.temperatures)
    if temperature is None:
        return {}
    if isinstance(temperature, dict):
        bad = [k for k in temperature if k not in QTYPES]
        if bad:
            raise ValueError("temperature keys must be question types %s, got %r" % (sorted(QTYPES), bad))
        return {QTYPES[k]: check_temperature(v, "temperature[%r]" % k) for k, v in temperature.items()}
    t = check_temperature(temperature)
    return {i: t for i in QTYPES.values()}


# ---------------------------------------------------------------------------------------------------------
# Numerics on per-question logits (numpy only)
# ---------------------------------------------------------------------------------------------------------


def pad_logits(logits: Sequence[Sequence[float]]) -> np.ndarray:
    """Stack variable-length logit vectors into ``[N, Kmax]``, padding with ``-inf`` (probability 0)."""
    kmax = max(len(z) for z in logits)
    Z = np.full((len(logits), kmax), -np.inf)
    for i, z in enumerate(logits):
        Z[i, : len(z)] = z
    return Z


def log_probs(Z: np.ndarray, t) -> np.ndarray:
    """``log softmax(Z / t)`` row-wise; ``t`` is a scalar or one temperature per row."""
    t = np.broadcast_to(np.asarray(t, dtype=float), (Z.shape[0],))
    S = Z / t[:, None]
    m = S.max(1, keepdims=True)
    return S - m - np.log(np.exp(S - m).sum(1, keepdims=True))


def mean_nll(Z: np.ndarray, labels: np.ndarray, t) -> float:
    return float(-log_probs(Z, t)[np.arange(len(labels)), labels].mean())


def fit_temperature(Z: np.ndarray, labels: np.ndarray, bounds=T_BOUNDS, iterations: int = 80) -> float:
    """Minimise mean NLL over ``T`` by golden-section search on ``log T`` (NLL is convex in ``1/T``, so unimodal)."""
    lo, hi = math.log(bounds[0]), math.log(bounds[1])
    r = (math.sqrt(5) - 1) / 2
    f = lambda u: mean_nll(Z, labels, math.exp(u))  # noqa: E731
    a, b = hi - r * (hi - lo), lo + r * (hi - lo)
    fa, fb = f(a), f(b)
    for _ in range(iterations):
        if fa < fb:
            hi, b, fb = b, a, fa
            a = hi - r * (hi - lo)
            fa = f(a)
        else:
            lo, a, fa = a, b, fb
            b = lo + r * (hi - lo)
            fb = f(b)
    return float(math.exp((lo + hi) / 2))


def fit_rule(Z: np.ndarray, labels: np.ndarray, qtypes: np.ndarray, per_type: bool = True,
             min_rows: int = 30) -> Dict[str, Any]:
    """The fitting rule, shared by the shipped fit and every cross-validation fold.

    Returns ``{"temperatures": {type: T}, "sources": {type: "per_type" | "pooled"}, "fitted_on": {type: n}}`` for the
    types present; a type missing from ``temperatures`` keeps the checkpoint's value.
    """
    present = [int(q) for q in np.unique(qtypes)]
    out = {"temperatures": {}, "sources": {}, "fitted_on": {}}
    rest = []
    for q in present:
        sel = qtypes == q
        if per_type and sel.sum() >= min_rows:
            name = QTYPE_NAMES[q]
            out["temperatures"][name] = fit_temperature(Z[sel], labels[sel])
            out["sources"][name], out["fitted_on"][name] = "per_type", int(sel.sum())
        else:
            rest.append(q)
    if rest and len(labels) >= min_rows:
        pooled = fit_temperature(Z, labels)
        for q in rest:
            name = QTYPE_NAMES[q]
            out["temperatures"][name], out["sources"][name] = pooled, "pooled"
            out["fitted_on"][name] = int(len(labels))
    return out


def row_temperatures(fit: Dict[str, Any], qtypes: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    """One temperature per row: the fitted one for its type, else the row's checkpoint temperature."""
    t = np.array(fallback, dtype=float)
    for name, T in fit["temperatures"].items():
        t[qtypes == QTYPES[name]] = T
    return t


def group_folds(groups: Sequence, folds: int = 5, seed: int = 0) -> np.ndarray:
    """Fold index per row; every row of a group lands in the same fold (groups shuffled by ``seed``, dealt round-robin)."""
    uniq = sorted(set(groups), key=repr)
    order = np.random.default_rng(seed).permutation(len(uniq))
    fold_of = {uniq[g]: i % folds for i, g in enumerate(order)}
    return np.array([fold_of[g] for g in groups], dtype=int)


def out_of_fold(Z, labels, qtypes, fallback, fold, per_type=True, min_rows=30):
    """Held-out temperature per row: fitted on the other folds with ``fit_rule``. Also returns the fold fits."""
    t = np.empty(len(labels))
    fits = []
    for k in range(int(fold.max()) + 1):
        test = fold == k
        if not test.any():
            continue
        train = ~test
        fit = fit_rule(Z[train], labels[train], qtypes[train], per_type, min_rows) if train.any() else {"temperatures": {}}
        t[test] = row_temperatures(fit, qtypes[test], fallback[test])
        fits.append(fit["temperatures"])
    return t, fits


def conf_correct_nll(Z: np.ndarray, labels: np.ndarray, t) -> Dict[str, np.ndarray]:
    """Per-row top-label confidence, correctness and NLL under temperatures ``t``."""
    lp = log_probs(Z, t)
    return {"conf": np.exp(lp.max(1)), "correct": (lp.argmax(1) == labels).astype(float),
            "nll": -lp[np.arange(len(labels)), labels]}


def bootstrap_draws(groups: Sequence, n: int = 1000, seed: int = 0) -> List[np.ndarray]:
    """``n`` row-index resamples, each drawing whole groups with replacement (as many groups as there are)."""
    by_group = defaultdict(list)
    for i, g in enumerate(groups):
        by_group[g].append(i)
    members = [np.array(v) for _, v in sorted(by_group.items(), key=lambda kv: repr(kv[0]))]
    rng = np.random.default_rng(seed)
    return [np.concatenate([members[j] for j in rng.integers(len(members), size=len(members))]) for _ in range(n)]


def _ci(values: List[float]) -> Optional[List[float]]:
    if not values:
        return None
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def evidence(variants: Dict[str, Dict[str, np.ndarray]], groups: Sequence, bootstrap: int = 1000,
             seed: int = 0, paired=(("checkpoint", "calibrated_oof"), ("raw", "calibrated_oof"))) -> Dict[str, Any]:
    """ECE and NLL per variant (e.g. raw / checkpoint / calibrated_oof) with 95% group-bootstrap intervals.

    ``paired`` adds the interval of ``ECE(a) - ECE(b)`` computed on the same resamples (positive: ``b`` is better).
    """
    first = next(iter(variants.values()))
    acc = [float(v["correct"].mean()) for v in variants.values()]
    out = {"n": int(len(first["conf"])), "groups": len(set(groups)), "accuracy": acc[0],
           "accuracy_unchanged": bool(all(np.array_equal(v["correct"], first["correct"]) for v in variants.values())),
           "ece": {}, "nll": {}}
    draws = bootstrap_draws(groups, bootstrap, seed) if bootstrap else []
    boot = {name: [ece_score(v["conf"][d], v["correct"][d]) for d in draws] for name, v in variants.items()}
    for name, v in variants.items():
        out["ece"][name] = {"value": ece_score(v["conf"], v["correct"]), "ci95": _ci(boot[name])}
        out["nll"][name] = {"value": float(v["nll"].mean()),
                            "ci95": _ci([float(v["nll"][d].mean()) for d in draws])}
    out["ece_improvement"] = {}
    for a, b in paired:
        if a in variants and b in variants:
            out["ece_improvement"]["%s-%s" % (a, b)] = {
                "value": out["ece"][a]["value"] - out["ece"][b]["value"],
                "ci95": _ci([x - y for x, y in zip(boot[a], boot[b])])}
    return out


# ---------------------------------------------------------------------------------------------------------
# The result object
# ---------------------------------------------------------------------------------------------------------


@dataclasses.dataclass
class Calibration:
    """Temperatures fitted on the user's data, the evidence for them and the checkpoint they belong to.

    ``temperatures`` maps a question type to its ``T``; types absent keep the checkpoint's temperatures (including
    its per-option-count buckets). Pass to ``VLMAgent.predict(..., calibration=cal)``; ``save`` / ``load`` as JSON.
    """

    temperatures: Dict[str, float]
    sources: Dict[str, str]
    fitted_on: Dict[str, int]
    group_key: Optional[str]
    folds: int
    per_type: bool
    min_rows: int
    n_permutations: int
    seed: int
    bootstrap: int
    evidence: Dict[str, Any]
    fold_temperatures: List[Dict[str, float]]
    checkpoint: Dict[str, Any]
    created: str = dataclasses.field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    version: int = 1

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(dataclasses.asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "Calibration":
        with open(path) as f:
            d = json.load(f)
        names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})

    def check(self, identity: Dict[str, Any], strict: bool = False) -> bool:
        """Compare with a checkpoint identity (``checkpoint_identity(agent)``); warn, or raise when ``strict``."""
        diff = [k for k in ("config_sha256", "weights_sha256")
                if self.checkpoint.get(k) and identity.get(k) and self.checkpoint[k] != identity[k]]
        if not diff:
            return True
        msg = ("this calibration was fitted for a different checkpoint (%s: %s differs from the loaded %s); its "
               "temperatures may not fit this model" % (self.checkpoint.get("model"), ", ".join(diff), identity.get("model")))
        if strict:
            raise ValueError(msg)
        warnings.warn(msg, stacklevel=3)
        return False

    def summary(self) -> str:
        """A short plain-text report: temperatures, then ECE per variant with its interval."""
        fitted = ", ".join("%s=%.3f (%s, n=%d)" % (k, v, self.sources[k], self.fitted_on[k])
                           for k, v in self.temperatures.items())
        lines = ["temperatures: " + (fitted or "none fitted (checkpoint's kept)")]
        for scope, ev in self.evidence.items():
            parts = []
            for name, e in ev["ece"].items():
                ci = e["ci95"]
                parts.append("%s %.3f" % (name, e["value"]) + (" [%.3f, %.3f]" % tuple(ci) if ci else ""))
            lines.append("%s (n=%d, groups=%d, acc=%.3f): ECE %s" % (scope, ev["n"], ev["groups"], ev["accuracy"],
                                                                    "; ".join(parts)))
        return "\n".join(lines)


def checkpoint_identity(agent) -> Dict[str, Any]:
    """Model id/path plus hashes of the config and weights, so a calibration can tell which checkpoint it belongs to.

    The config hash leaves out the temperatures (what calibration replaces) and the inference dtype. The weights hash
    covers every head tensor and a strided sample of each backbone tensor, all rounded to bf16 so an fp32 and a bf16
    load of one checkpoint agree. It is cached until a parameter is modified in place (``Tensor._version``).
    """
    import torch

    cfg = {k: v for k, v in agent.cfg.items() if k not in CONFIG_IGNORED}
    sd = agent.model.state_dict()
    key = tuple((k, v.data_ptr(), v._version) for k, v in sd.items())
    cached = getattr(agent, "_identity_cache", None)
    if cached is None or cached[0] != key:
        h = hashlib.sha256()
        for k in sorted(sd):
            if k == "temperature" or not sd[k].is_floating_point():
                continue
            v = sd[k].detach().reshape(-1)
            if k.startswith("encoder."):
                v = v[:: max(1, v.numel() // 256)][:256]
            h.update(k.encode())
            h.update(v.to(torch.bfloat16).view(torch.int16).cpu().numpy().tobytes())
        cached = (key, h.hexdigest())
        agent._identity_cache = cached
    source = getattr(agent, "source", None) or {}
    return {"model": source.get("id") or agent.cfg.get("backbone"), "revision": source.get("revision"),
            "config_sha256": hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest(),
            "weights_sha256": cached[1]}


# ---------------------------------------------------------------------------------------------------------
# From records to a Calibration
# ---------------------------------------------------------------------------------------------------------


def calibrate_records(records: List[Dict[str, Any]], group_key: Optional[str] = "image_id", folds: int = 5,
                      per_type: bool = True, bootstrap: int = 1000, seed: int = 0, min_rows: int = 30,
                      checkpoint: Optional[Dict[str, Any]] = None, n_permutations: int = 1) -> Calibration:
    """Fit and evaluate temperatures on per-question records; no model needed.

    Each record is ``{"logits": raw averaged option logits (label order), "label": int, "qtype": int, "group": any,
    "checkpoint_t": the checkpoint's temperature for this question}``. The shipped temperatures are fitted on all
    records; ECE of the fitted temperature is out-of-fold over group-disjoint folds.
    """
    if not records:
        raise ValueError("no labelled questions to calibrate on")
    Z = pad_logits([np.asarray(r["logits"], dtype=float) for r in records])
    labels = np.array([int(r["label"]) for r in records])
    qtypes = np.array([int(r["qtype"]) for r in records])
    groups = [r["group"] for r in records]
    ckpt_t = np.array([float(r.get("checkpoint_t", 1.0)) for r in records])
    if not np.all(np.isfinite(Z[np.arange(len(labels)), labels])):
        raise ValueError("a label points at a missing option")

    fit = fit_rule(Z, labels, qtypes, per_type, min_rows)
    if not fit["temperatures"]:
        warnings.warn("only %d labelled questions (< min_rows=%d): nothing fitted, the checkpoint's temperatures are "
                      "kept" % (len(labels), min_rows), stacklevel=2)
    n_groups = len(set(groups))
    k = min(folds, n_groups)
    if k < folds:
        warnings.warn("only %d groups, using %d folds instead of %d" % (n_groups, k, folds), stacklevel=2)
    variants = {"raw": np.ones(len(labels)), "checkpoint": ckpt_t}
    fold_t = []
    if k >= 2:
        fold = group_folds(groups, k, seed)
        variants["calibrated_oof"], fold_t = out_of_fold(Z, labels, qtypes, ckpt_t, fold, per_type, min_rows)
    else:
        warnings.warn("fewer than 2 groups: no out-of-fold estimate", stacklevel=2)
    variants["calibrated_in_sample"] = row_temperatures(fit, qtypes, ckpt_t)

    scored = {name: conf_correct_nll(Z, labels, t) for name, t in variants.items()}
    ev = {"all": evidence(scored, groups, bootstrap, seed)}
    for q in sorted(set(qtypes.tolist())):
        sel = qtypes == q
        if sel.sum() and len(set(qtypes.tolist())) > 1:
            part = {name: {kk: vv[sel] for kk, vv in s.items()} for name, s in scored.items()}
            ev[QTYPE_NAMES[q]] = evidence(part, [g for g, s in zip(groups, sel) if s], bootstrap, seed)
    if not all(e["accuracy_unchanged"] for e in ev.values()):  # only a float tie can do this for T > 0
        warnings.warn("temperature scaling changed an argmax (a rounding tie between two options)", stacklevel=2)
    return Calibration(temperatures=fit["temperatures"], sources=fit["sources"], fitted_on=fit["fitted_on"],
                       group_key=group_key, folds=k, per_type=per_type, min_rows=min_rows,
                       n_permutations=n_permutations, seed=seed, bootstrap=bootstrap, evidence=ev,
                       fold_temperatures=fold_t, checkpoint=dict(checkpoint or {}))
