"""Question-form robustness families for the VLM decision model: the same judgment asked in another form.

Both families are built from the inputs alone (output-blind, as in ``laya.robustness``): each variant row keeps
its source row's ``group_id`` and ``cluster``, and its label is the source label mapped through a fixed rule,
never re-derived from the model. Unlike the ``laya.robustness`` families the question *type* or the label can
change, so a variant is not compared to its source row by plain argmax agreement; ``summarize_form`` gives each
family its own paired metrics.

* ``form_choice``: the judgment in the other primitive.
    - a ``noul`` row becomes a 2-option ``choice`` row with options ``("no", "yes")``, the question text
      unchanged (variant ``as_choice``). The options are in noul label order (false, true), so the label and
      target carry over as they are.
    - a ``choice`` row with k options becomes k ``noul`` rows ``"Is the answer to this question '<option>'?
      <question>"`` (variant ``opt<i>``), label ``int(i == gold)``, target ``[1 - t_i, t_i]`` from the source
      target ``t``. ``meta`` records ``option`` (i), ``k`` and ``choice_label`` so the k rows can be
      reassembled into a ranking of the options.
* ``negation`` (``noul`` only): a rule-based negated *frame*, the sentence itself untouched. A question (ends
  with ``?`` or starts with a question word) becomes ``"Decide whether the answer to this question is no:
  <question>"`` (variant ``question``), a statement ``"Is it false that <statement>?"`` (variant
  ``statement``). Label ``1 - label``, target reversed.

Skipped: ``score`` rows (no rule), empty questions, and ``noul`` rows with custom option descriptions (e.g.
``{"false": "the cat", "true": "the dog"}``), whose descriptions carry the meaning and would contradict a
``no``/``yes`` choice or a negated frame.

A coherent model gives P_yes(noul) = P_yes(choice), ranks the k options of a choice row the same way asked one at
a time, and has P_yes(x) + P_yes(negated x) = 1. ``summarize_form`` measures each, per dataset, from
``laya.robustness.score_rows`` predictions (it needs ``meta``, ``probs``, ``pred``, ``label``).
"""
import re
from typing import Dict, List, Optional, Sequence

import numpy as np

from .. import robustness as R
from ..common import render_options

FAMILIES = ("form_choice", "negation")

NOUL_AS_CHOICE = ("no", "yes")  # noul label order: 0 = false, 1 = true
COMPLEMENT_BAND = (0.9, 1.1)


def _custom_noul(q: Dict) -> bool:
    return bool(q.get("crit"))


def _is_question(s: str) -> bool:
    if s.endswith("?"):
        return True
    words = s.split()
    return bool(words) and words[0].lower().strip("\"'(") in R.QUESTION_WORDS


def _form_row(src: Dict, family: str, variant: str, q: Dict, target: List[float], label: int, **meta) -> Dict:
    row = R._variant(src, family, variant, q=q, target=target, label=label,
                     meta=dict(src.get("meta") or {}, **meta))
    for k in ("order", "shown_label"):  # a display order of the source's options means nothing for the new ones
        row.pop(k, None)
    return row


def form_choice_variants(rows: Sequence[Dict]) -> List[Dict]:
    """The ``form_choice`` family (see the module docstring)."""
    out = []
    for src in rows:
        q = src["q"]
        ins = q["ins"].strip()
        if not ins:
            continue
        if q["t"] == "noul" and not _custom_noul(q):
            nq = {"t": "choice", "ins": q["ins"], "crit": {o: None for o in NOUL_AS_CHOICE}}
            out.append(_form_row(src, "form_choice", "as_choice", nq, list(src["target"]), src["label"],
                                 form="noul_to_choice"))
        elif q["t"] == "choice":
            opts = render_options(q)
            k = len(opts)
            for i, o in enumerate(opts):
                nq = {"t": "noul", "ins": "Is the answer to this question '%s'? %s" % (o, ins), "crit": None}
                t = float(src["target"][i])
                out.append(_form_row(src, "form_choice", "opt%d" % i, nq, [1.0 - t, t], int(i == src["label"]),
                                     form="choice_to_noul", option=i, k=k, choice_label=src["label"]))
    return out


def negate_frame(ins: str) -> Optional[tuple]:
    """``(variant, negated instructions)`` for a noul question or statement, or ``None`` for an empty one."""
    s = ins.strip()
    if not s:
        return None
    if _is_question(s):
        return "question", "Decide whether the answer to this question is no: " + s
    first = s.split()[0]
    body = (s[0].lower() + s[1:]) if R._plain_word(re.sub(r"[^\w]+$", "", first)) else s
    return "statement", "Is it false that %s?" % body.rstrip(".!").rstrip()


def negation_variants(rows: Sequence[Dict]) -> List[Dict]:
    """The ``negation`` family (see the module docstring)."""
    out = []
    for src in rows:
        q = src["q"]
        if q["t"] != "noul" or _custom_noul(q):
            continue
        neg = negate_frame(q["ins"])
        if neg is None:
            continue
        variant, ins = neg
        out.append(_form_row(src, "negation", variant, dict(q, ins=ins), list(reversed(src["target"])),
                             1 - src["label"], form="negation"))
    return out


def build(rows: Sequence[Dict], families: Sequence[str] = FAMILIES, seed: int = 0) -> List[Dict]:
    """The variant rows (source rows not included) of ``families``, in family then source-row order.
    Deterministic; no rule is random, ``seed`` is accepted for the ``build_variants`` interface."""
    makers = {"form_choice": lambda: form_choice_variants(rows), "negation": lambda: negation_variants(rows)}
    unknown = set(families) - set(makers)
    if unknown:
        raise ValueError("unknown families %s (expected %s)" % (sorted(unknown), FAMILIES))
    out = []
    for fam in families:
        out += makers[fam]()
    ids = [r["id"] for r in out]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate row ids")
    return out


# ---------------------------------------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------------------------------------


def _p_yes(p: Dict) -> float:
    return float(p["probs"][1])


def _boot_ci(values: Sequence[float], clusters: Sequence[str], n_boot: int, rng: np.random.Generator) -> List[float]:
    """95% percentile interval of the mean of per-group ``values``, resampling clusters."""
    cl = sorted(set(clusters))
    if not n_boot or len(cl) < 2:
        return [float("nan")] * 2
    idx = {c: i for i, c in enumerate(cl)}
    gc = np.array([idx[c] for c in clusters])
    cnt = np.bincount(gc, minlength=len(cl)).astype(float)
    sums = np.bincount(gc, weights=np.asarray(values, dtype=float), minlength=len(cl))
    W = rng.multinomial(len(cl), np.full(len(cl), 1.0 / len(cl)), size=n_boot).astype(float)
    return R._ci((W @ sums) / np.maximum(W @ cnt, 1))


def _dist(v: Sequence[float]) -> Dict:
    a = np.asarray(v, dtype=float)
    return {"mean": float(a.mean()), "p50": float(np.percentile(a, 50)), "p90": float(np.percentile(a, 90)),
            "max": float(a.max())}


def _noul_to_choice(rows: Sequence[Dict], base: Dict[str, Dict], n_boot, rng) -> Optional[Dict]:
    pairs = [(p, base[p["group_id"]]) for p in rows if p["group_id"] in base]
    if not pairs:
        return None
    gap = [abs(_p_yes(p) - _p_yes(b)) for p, b in pairs]
    cls = [p["cluster"] for p, _ in pairs]
    return {"n_groups": len(pairs), "n_clusters": len(set(cls)), "abs_gap": _dist(gap),
            "abs_gap_ci": _boot_ci(gap, cls, n_boot, rng),
            "mean_signed_gap": float(np.mean([_p_yes(p) - _p_yes(b) for p, b in pairs])),  # choice minus noul
            "argmax_agree": float(np.mean([p["pred"] == b["pred"] for p, b in pairs])),
            "acc": float(np.mean([p["pred"] == p["label"] for p, _ in pairs])),
            "base_acc": float(np.mean([b["pred"] == b["label"] for _, b in pairs]))}


def _choice_to_noul(rows: Sequence[Dict], base: Dict[str, Dict], n_boot, rng) -> Optional[Dict]:
    by: Dict[str, Dict[int, Dict]] = {}
    for p in rows:
        by.setdefault(p["group_id"], {})[int(p["meta"]["option"])] = p
    groups = []
    incomplete = 0
    for gid, opts in by.items():
        k = int(next(iter(opts.values()))["meta"]["k"])
        b = base.get(gid)
        if sorted(opts) != list(range(k)) or b is None:
            incomplete += 1
            continue
        py = np.array([_p_yes(opts[i]) for i in range(k)])
        pc = np.asarray(b["probs"], dtype=float)
        top = int(py.argmax())
        groups.append({"cluster": b["cluster"], "agree": float(top == b["pred"]), "acc": float(top == b["label"]),
                       "base_acc": float(b["pred"] == b["label"]), "p_yes_sum": float(py.sum()),
                       "abs_gap": float(np.abs(py - pc).mean()),
                       "row_acc": float(np.mean([opts[i]["pred"] == opts[i]["label"] for i in range(k)])),
                       "chance": 1.0 / k})
    if not groups:
        return None
    col = lambda k: [g[k] for g in groups]  # noqa: E731
    cls = col("cluster")
    s = np.array(col("p_yes_sum"))
    return {"n_groups": len(groups), "n_clusters": len(set(cls)), "n_incomplete": incomplete,
            "argmax_agree": float(np.mean(col("agree"))), "argmax_agree_ci": _boot_ci(col("agree"), cls, n_boot, rng),
            "acc": float(np.mean(col("acc"))), "acc_ci": _boot_ci(col("acc"), cls, n_boot, rng),
            "base_acc": float(np.mean(col("base_acc"))), "delta_acc": float(np.mean(col("acc")) - np.mean(col("base_acc"))),
            "chance": float(np.mean(col("chance"))),
            "row_acc": float(np.mean(col("row_acc"))),  # yes/no accuracy per option row, group-averaged
            "p_yes_sum": {"mean": float(s.mean()), "min": float(s.min()), "max": float(s.max())},
            "abs_gap": _dist(col("abs_gap"))}  # mean over options of |P_yes(opt i) - P_choice(i)|


def _negation(rows: Sequence[Dict], base: Dict[str, Dict], n_boot, rng) -> Optional[Dict]:
    pairs = [(p, base[p["group_id"]]) for p in rows if p["group_id"] in base]
    if not pairs:
        return None
    s = np.array([_p_yes(p) + _p_yes(b) for p, b in pairs])
    cls = [p["cluster"] for p, _ in pairs]
    lo, hi = COMPLEMENT_BAND
    acc = [float(p["pred"] == p["label"]) for p, _ in pairs]
    base_acc = [float(b["pred"] == b["label"]) for _, b in pairs]
    # the negated row should give the opposite answer; a flip is one whose implied answer differs from the source
    flip = [float(p["pred"] != 1 - b["pred"]) for p, b in pairs]
    res = {"n_groups": len(pairs), "n_clusters": len(set(cls)),
           "complement_sum": {"mean": float(s.mean()), "min": float(s.min()), "max": float(s.max()),
                              "share_outside": float(np.mean((s < lo) | (s > hi))), "band": [lo, hi]},
           "complement_abs_dev": float(np.mean(np.abs(s - 1))),
           "complement_abs_dev_ci": _boot_ci(list(np.abs(s - 1)), cls, n_boot, rng),
           "acc": float(np.mean(acc)), "acc_ci": _boot_ci(acc, cls, n_boot, rng), "base_acc": float(np.mean(base_acc)),
           "delta_acc": float(np.mean(acc) - np.mean(base_acc)), "flip_rate": float(np.mean(flip)),
           "flip_rate_ci": _boot_ci(flip, cls, n_boot, rng), "variants": {}}
    for v in sorted({p["variant"] for p, _ in pairs}):
        vp = [(p, b) for p, b in pairs if p["variant"] == v]
        vs = np.array([_p_yes(p) + _p_yes(b) for p, b in vp])
        res["variants"][v] = {"n": len(vp), "acc": float(np.mean([p["pred"] == p["label"] for p, _ in vp])),
                              "flip_rate": float(np.mean([p["pred"] != 1 - b["pred"] for p, b in vp])),
                              "complement_sum_mean": float(vs.mean()),
                              "share_outside": float(np.mean((vs < lo) | (vs > hi)))}
    return res


def summarize_form(preds: Sequence[Dict], n_boot: int = 1000, seed: int = 0) -> Dict:
    """Per dataset: ``form_choice`` split into ``noul_to_choice`` (|P_yes(noul) - P_yes(choice)| distribution and
    argmax agreement with the source row) and ``choice_to_noul`` (the argmax of the k per-option P_yes against the
    source row's choice argmax, and its accuracy), and ``negation`` (the complement sum P_yes(orig) +
    P_yes(negated): mean / min / max / share outside ``COMPLEMENT_BAND``, plus accuracy, and flip rate = share of
    negated rows whose answer is not the opposite of the source row's). Each source row counts once; intervals
    are cluster-bootstrap 95%. ``"macro"`` averages the headline numbers over datasets."""
    rng = np.random.default_rng(seed)
    out: Dict[str, Dict] = {}
    for name in sorted({p["dataset"] for p in preds}):
        rows = [p for p in preds if p["dataset"] == name]
        base = {p["group_id"]: p for p in rows if p["family"] == "orig"}
        fc = [p for p in rows if p["family"] == "form_choice"]
        rep = {}
        n2c = _noul_to_choice([p for p in fc if p["meta"]["form"] == "noul_to_choice"], base, n_boot, rng)
        c2n = _choice_to_noul([p for p in fc if p["meta"]["form"] == "choice_to_noul"], base, n_boot, rng)
        if n2c or c2n:
            rep["form_choice"] = {k: v for k, v in (("noul_to_choice", n2c), ("choice_to_noul", c2n)) if v}
        neg = _negation([p for p in rows if p["family"] == "negation"], base, n_boot, rng)
        if neg:
            rep["negation"] = neg
        if rep:
            out[name] = rep
    macro: Dict[str, Dict] = {}
    heads = {("form_choice", "noul_to_choice"): lambda s: {"abs_gap_mean": s["abs_gap"]["mean"],
                                                           "argmax_agree": s["argmax_agree"]},
             ("form_choice", "choice_to_noul"): lambda s: {"argmax_agree": s["argmax_agree"], "acc": s["acc"],
                                                           "base_acc": s["base_acc"]},
             ("negation",): lambda s: {"complement_sum_mean": s["complement_sum"]["mean"],
                                       "share_outside": s["complement_sum"]["share_outside"], "acc": s["acc"],
                                       "flip_rate": s["flip_rate"]}}
    for path, fn in heads.items():
        per = []
        for rep in out.values():
            s = rep
            for k in path:
                s = s.get(k) if s else None
            if s:
                per.append(fn(s))
        if per:
            macro["/".join(path)] = dict({"n_datasets": len(per)},
                                         **{k: float(np.mean([d[k] for d in per])) for k in per[0]})
    return {"datasets": out, "macro": macro, "n_boot": n_boot, "seed": seed}
