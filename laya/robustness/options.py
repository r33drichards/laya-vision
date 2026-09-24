"""Option-set and abstention perturbations for ``choice`` rows, on top of ``laya.robustness``.

Same bookkeeping as ``laya.robustness``: every variant is built with ``robustness._variant`` from a source row
(``family == "orig"``, from ``robustness.source_rows``), keeps its ``group_id`` / ``cluster``, is fixed from the
inputs and ``seed`` alone before any model output is seen, and its label is never re-derived from the model.
Only ``choice`` rows get variants (a ``noul`` row's two options are fixed; ``score`` levels are ordered).

Families (``FAMILIES``):

* ``option_set``: the option list changes, the answer does not.
    - ``add_distractor``: one option of a *different* source row of the same dataset (and a different image
      cluster, so a sibling question's option that may be true of this image is never used), picked by
      ``seed``, appended last. A case-insensitive duplicate of an existing option is skipped for the next
      candidate. The label is unchanged.
    - ``drop_wrong`` (k >= 3): one seeded wrong option removed; the label is remapped.
  ``meta["orig_index"][j]`` is the source row's label index of the variant's option ``j`` (``None`` for the
  added distractor), so the options present in both can be compared.
* ``abstain``: can the model say "none of these"?
    - ``drop_gold`` (k >= 3): the gold option removed, so *no* option is right. ``meta["no_gold"]`` is true, the
      label is ``-1`` (never equal to an argmax) and the target is uniform over the remaining options, which
      keeps ``collect_logits`` working (it needs a k-long target; the label is only passed through). These rows
      carry no accuracy: only their confidence is read.
    - ``add_none``: ``"none of the above"`` appended last (``meta["none_index"]``); the label is unchanged, since
      the gold option is still there.
    - ``add_none_shuffled``: the same, on the mismatched image that ``robustness.shuffle_variants`` gives the
      row (the same seeded derangement as the ``image_shuffle`` control, so the donor is deterministic and
      identical to that family's). ``meta["mismatched_image"]`` is true; the stored label is still the source
      row's, but with the wrong picture "none" is a reasonable answer, so these rows are read for the
      none-rate, not accuracy.
  Rows that already have a "none of the above" option (case-insensitive) get no ``add_none*`` variant.

``summarize_options(preds)`` reads ``robustness.score_rows`` output (which copies ``meta`` into each prediction)
and reports, per dataset: for ``option_set``, the shift in pairwise log-odds between the original options kept
in both rows (variant minus source, from the raw logits and from the calibrated probabilities), argmax flips in
label space, and the paired accuracy change; for ``abstain``, the confidence on ``drop_gold`` rows (mean max
probability, share above 0.5 / 0.8) and how often "none" is picked on the real vs the mismatched image, plus
``add_none``'s accuracy cost.
"""
import math
import random
from typing import Dict, List, Optional, Sequence

import numpy as np

from .. import robustness as R

FAMILIES = ("option_set", "abstain")
VARIANTS = {"option_set": ("add_distractor", "drop_wrong"), "abstain": ("drop_gold", "add_none", "add_none_shuffled")}
NONE_OPTION = "none of the above"
MAIN_TABLE_EXCLUDE = {"abstain"}  # no gold (drop_gold) or a label the mismatched image voids (add_none_shuffled)

# ---------------------------------------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------------------------------------


def _is_choice(src: Dict) -> bool:
    return src["q"]["t"] == "choice" and isinstance(src["q"].get("crit"), dict)


def _items(src: Dict) -> List[tuple]:
    return list(src["q"]["crit"].items())


def _with_options(src: Dict, family: str, variant: str, items: List[tuple], target: List[float], label: int,
                  meta: Dict, **changes) -> Dict:
    """A variant with the choice options ``items`` (label order), its ``target`` / ``label`` and ``meta``. A
    display ``order`` or ``shown_label`` on the source is dropped: it indexes the old option list."""
    row = R._variant(src, family, variant, q=dict(src["q"], crit=dict(items)), target=target, label=label,
                     meta=dict(src.get("meta") or {}, **meta), **changes)
    row.pop("order", None)
    row.pop("shown_label", None)
    return row


def _renorm(t: List[float]) -> Optional[List[float]]:
    s = sum(t)
    return [p / s for p in t] if s > 0 else None


def add_distractor_variants(rows: Sequence[Dict], seed: int = 0) -> List[Dict]:
    """``option_set/add_distractor``: per choice row, the donor rows (same dataset, other group and image
    cluster) are shuffled by ``seed`` and the row's ``group_id``; the first donor option that is not a
    case-insensitive duplicate of one of the row's options is appended (with its description)."""
    by: Dict[str, List[Dict]] = {}
    for src in rows:
        if _is_choice(src):
            by.setdefault(src["dataset"], []).append(src)
    out = []
    for src in rows:
        if not _is_choice(src):
            continue
        items = _items(src)
        have = {k.strip().lower() for k, _ in items}
        donors = [d for d in by[src["dataset"]] if d["group_id"] != src["group_id"] and d["cluster"] != src["cluster"]]
        rng = random.Random(R._seed_for("distractor", seed, src["group_id"]))
        rng.shuffle(donors)
        pick = None
        for d in donors:
            cands = [kv for kv in _items(d) if kv[0].strip().lower() not in have]
            if cands:
                pick = (d, cands[rng.randrange(len(cands))])
                break
        if pick is None:
            continue
        d, kv = pick
        k = len(items)
        out.append(_with_options(src, "option_set", "add_distractor", items + [kv], list(src["target"]) + [0.0],
                                 src["label"], {"orig_index": list(range(k)) + [None], "added_index": k,
                                                "distractor_from": d["group_id"]}))
    return out


def _drop(src: Dict, i: int) -> tuple:
    """(items, target, orig_index) with option ``i`` removed."""
    keep = [j for j in range(len(src["target"])) if j != i]
    items = _items(src)
    return [items[j] for j in keep], [src["target"][j] for j in keep], keep


def drop_wrong_variants(rows: Sequence[Dict], seed: int = 0) -> List[Dict]:
    """``option_set/drop_wrong``: one seeded wrong option removed from each choice row with 3+ options; the label
    moves down by one if the removed option came before it."""
    out = []
    for src in rows:
        k = len(src["target"])
        if not _is_choice(src) or k < 3:
            continue
        wrong = [j for j in range(k) if j != src["label"]]
        i = wrong[random.Random(R._seed_for("drop_wrong", seed, src["group_id"])).randrange(len(wrong))]
        items, target, keep = _drop(src, i)
        target = _renorm(target)
        if target is None:
            continue
        out.append(_with_options(src, "option_set", "drop_wrong", items, target, keep.index(src["label"]),
                                 {"orig_index": keep, "dropped_index": i}))
    return out


def drop_gold_variants(rows: Sequence[Dict]) -> List[Dict]:
    """``abstain/drop_gold``: the gold option removed from each choice row with 3+ options. No option is right:
    label ``-1``, a uniform target, ``meta["no_gold"] = True``."""
    out = []
    for src in rows:
        k = len(src["target"])
        if not _is_choice(src) or k < 3:
            continue
        items, _, keep = _drop(src, src["label"])
        out.append(_with_options(src, "abstain", "drop_gold", items, [1.0 / (k - 1)] * (k - 1), -1,
                                 {"no_gold": True, "orig_index": keep, "dropped_index": src["label"]}))
    return out


def add_none_variants(rows: Sequence[Dict], seed: int = 0) -> List[Dict]:
    """``abstain/add_none`` and ``abstain/add_none_shuffled``: ``NONE_OPTION`` appended last, on the row's own image
    and on the donor image ``robustness.shuffle_variants`` gives it (from *all* ``rows``, so the donor matches the
    ``image_shuffle`` control's)."""
    shuffled = {v["group_id"]: v for v in R.shuffle_variants(rows, seed)}
    out = []
    for src in rows:
        if not _is_choice(src):
            continue
        items = _items(src)
        if NONE_OPTION in {k.strip().lower() for k, _ in items}:
            continue
        k = len(items)
        new = items + [(NONE_OPTION, None)]
        target = list(src["target"]) + [0.0]
        meta = {"orig_index": list(range(k)) + [None], "none_index": k}
        out.append(_with_options(src, "abstain", "add_none", new, target, src["label"], dict(meta, mismatched_image=False)))
        sh = shuffled.get(src["group_id"])
        if sh is not None:
            out.append(_with_options(src, "abstain", "add_none_shuffled", new, target, src["label"],
                                     dict(meta, mismatched_image=True), state=sh["state"],
                                     donor_image=sh["donor_image"]))
    return out


def build(rows: Sequence[Dict], families: Sequence[str] = FAMILIES, seed: int = 0) -> List[Dict]:
    """Every family's variant rows (not the source rows), in family then row order. Deterministic given ``rows``
    and ``seed``."""
    makers = {"option_set": lambda: add_distractor_variants(rows, seed) + drop_wrong_variants(rows, seed),
              "abstain": lambda: drop_gold_variants(rows) + add_none_variants(rows, seed)}
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


def _mean(xs) -> float:
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return float(np.mean(xs)) if xs else float("nan")


def _logodds_shift(p: Dict, b: Dict, key: str) -> Optional[List[float]]:
    """Per pair (i, j) of source options kept in the variant: (s_i - s_j) under the variant minus the same under
    the source row, where s is the raw logit (``key="logits"``) or the log calibrated probability
    (``key="probs"``, clipped at 1e-6 against the 5 d.p. rounding)."""
    f = (lambda v: float(v)) if key == "logits" else (lambda v: math.log(max(float(v), 1e-6)))
    sv, sb = [f(v) for v in p[key]], [f(v) for v in b[key]]
    kept = [(j, o) for j, o in enumerate(p["meta"]["orig_index"]) if o is not None]
    out = []
    for x, (j1, o1) in enumerate(kept):
        for j2, o2 in kept[x + 1:]:
            out.append((sv[j1] - sv[j2]) - (sb[o1] - sb[o2]))
    return out or None


def _option_set_stats(vr: List[Dict], base: Dict[str, Dict]) -> Dict:
    pairs = [(p, base[p["group_id"]]) for p in vr if p["group_id"] in base]
    raw = [_logodds_shift(p, b, "logits") for p, b in pairs]
    cal = [_logodds_shift(p, b, "probs") for p, b in pairs]
    # the variant's argmax in source label space (None: the added distractor)
    mapped = [p["meta"]["orig_index"][p["pred"]] for p, _ in pairs]
    kept = [b["pred"] in p["meta"]["orig_index"] for p, b in pairs]
    res = {"n": len(vr), "n_paired": len(pairs),
           "acc": _mean([float(p["pred"] == p["label"]) for p in vr]),
           "base_acc": _mean([float(b["pred"] == b["label"]) for _, b in pairs]),
           "mean_abs_logodds_shift": _mean([float(np.mean(np.abs(s))) for s in raw if s]),
           "mean_logodds_shift": _mean([float(np.mean(s)) for s in raw if s]),
           "mean_abs_logodds_shift_cal": _mean([float(np.mean(np.abs(s))) for s in cal if s]),
           "flip_rate": _mean([float(m != b["pred"]) for m, (_, b) in zip(mapped, pairs)]),
           # flips among rows whose source argmax is still on offer (drop_wrong can remove it)
           "flip_rate_kept": _mean([float(m != b["pred"]) for m, (_, b), k in zip(mapped, pairs, kept) if k])}
    res["delta_acc"] = _mean([float(p["pred"] == p["label"]) - float(b["pred"] == b["label"]) for p, b in pairs])
    if vr and vr[0]["variant"] == "add_distractor":
        res["pick_distractor_rate"] = _mean([float(m is None) for m in mapped])
    return res


def _pmax(rows: List[Dict], cut: float) -> float:
    return _mean([float(max(p["probs"]) > cut) for p in rows])


def _abstain_stats(fr: List[Dict], base: Dict[str, Dict]) -> Dict:
    out: Dict[str, Dict] = {}
    dg = [p for p in fr if p["variant"] == "drop_gold"]
    if dg:
        b = [base[p["group_id"]] for p in dg if p["group_id"] in base]
        out["drop_gold"] = {"n": len(dg), "mean_p_max": _mean([max(p["probs"]) for p in dg]),
                            "share_p_max_gt_0.5": _pmax(dg, 0.5), "share_p_max_gt_0.8": _pmax(dg, 0.8),
                            # the same rows with the gold option shown, for reference
                            "base_mean_p_max": _mean([max(r["probs"]) for r in b]),
                            "base_share_p_max_gt_0.8": _pmax(b, 0.8)}
    for v in ("add_none", "add_none_shuffled"):
        vr = [p for p in fr if p["variant"] == v]
        if not vr:
            continue
        pairs = [(p, base[p["group_id"]]) for p in vr if p["group_id"] in base]
        st = {"n": len(vr), "none_rate": _mean([float(p["pred"] == p["meta"]["none_index"]) for p in vr]),
              "mean_p_none": _mean([p["probs"][p["meta"]["none_index"]] for p in vr])}
        if v == "add_none":
            st["acc"] = _mean([float(p["pred"] == p["label"]) for p in vr])
            st["base_acc"] = _mean([float(b["pred"] == b["label"]) for _, b in pairs])
            st["delta_acc"] = _mean([float(p["pred"] == p["label"]) - float(b["pred"] == b["label"]) for p, b in pairs])
        out[v] = st
    if "add_none" in out and "add_none_shuffled" in out:
        # paired on the groups that have both
        a = {p["group_id"]: p for p in fr if p["variant"] == "add_none"}
        s = {p["group_id"]: p for p in fr if p["variant"] == "add_none_shuffled"}
        g = sorted(set(a) & set(s))
        out["none_rate_rise"] = {"n": len(g), "value": _mean([float(s[x]["pred"] == s[x]["meta"]["none_index"])
                                                              - float(a[x]["pred"] == a[x]["meta"]["none_index"])
                                                              for x in g])}
    return out


def summarize_options(preds: Sequence[Dict]) -> Dict:
    """Per dataset: ``{"option_set": {variant: stats}, "abstain": {...}}`` from ``robustness.score_rows``
    predictions (``"orig"`` rows are the paired baseline; other families are ignored). Every row is one group's
    only variant of its kind, so the means are already group means. ``delta_acc`` / flips are paired against the
    same groups' source rows; ``drop_gold`` rows never enter an accuracy."""
    out: Dict[str, Dict] = {}
    for name in sorted({p["dataset"] for p in preds}):
        rows = [p for p in preds if p["dataset"] == name]
        base = {p["group_id"]: p for p in rows if p["family"] == "orig"}
        rep: Dict[str, Dict] = {}
        os_rows = [p for p in rows if p["family"] == "option_set"]
        if os_rows:
            rep["option_set"] = {v: _option_set_stats([p for p in os_rows if p["variant"] == v], base)
                                 for v in VARIANTS["option_set"] if any(p["variant"] == v for p in os_rows)}
        ab_rows = [p for p in rows if p["family"] == "abstain"]
        if ab_rows:
            rep["abstain"] = _abstain_stats(ab_rows, base)
        if rep:
            out[name] = rep
    return out
