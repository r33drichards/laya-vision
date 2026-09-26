"""Rendering families: the same row with its state or its options *rendered* another way, for train/serve drift.

The released checkpoints were trained on one rendering of each input (``laya.prompt``): a record's text sits under
``{"context": ...}`` and the state's non-image keys are ``json.dumps``-ed, so the model reads
``{"context": "Lecture: ...\\n..."}`` with quotes, braces and escaped newlines; A-OKVQA options were lower-cased by
``laya.cauldron.parse_options``. A caller of ``predict`` can easily send something else. These families measure how
much that costs. Output-blind as in ``laya.robustness``: the label and target are the source row's.

* ``state_render`` (rows whose state has a string ``"context"``):
    - ``prose``: ``state_format="prose"`` (``laya.prompt.to_text``, CLM's rendering): ``context: <text>``.
    - ``text``: ``state_format="text"``: the text alone, no key (what ``predict("<text>", ...)`` sends
      without an image).
    - ``key_note``: the key renamed to ``"note"``, still JSON: the README's example state.
    - ``struct_json``: a context made of ``Key: value`` blocks separated by blank lines (``Question: ...\\n\\n
      Response: ...``, ``Prompt: ...``) split into those fields, JSON: a caller who sends the fields
      as a structured state. ``struct_prose``: the same fields in prose, kept only when its text differs from
      every earlier variant's (for these sets it is usually the ``text`` variant again).
* ``option_render`` (``choice`` rows with bare option names): ``lower`` (every name lower-cased, the A-OKVQA
  preparation) and ``title`` (first letter upper case); skipped when unchanged or when two names would collide.

A variant row carries its rendering as ``"state_format"`` (read by ``laya.vlm_train.make_item``) or as a changed
``state`` / ``q``. ``summarize_render`` reports, per dataset, family and variant, accuracy with the paired change
from the source rows (cluster-bootstrap 95% interval), the flip rate, the mean total-variation distance between
the variant's and the source row's probabilities, and the change in the gold option's log-probability.
"""
import re
from typing import Dict, List, Optional, Sequence

import numpy as np

from .. import robustness as R
from ..prompt import CONTEXT_KEY, normalize_choice_options, serialize_state

FAMILIES = ("state_render", "option_render")

_BLOCK = re.compile(r"^([A-Z][A-Za-z ]{0,30}):\s(.*)$", re.S)


def context_fields(text: str) -> Optional[Dict[str, str]]:
    """``"Question: a\\n\\nResponse: b"`` -> ``{"Question": "a", "Response": "b"}``; ``None`` unless every blank-line
    separated block starts with a distinct ``Key: ``."""
    out: Dict[str, str] = {}
    for block in text.split("\n\n"):
        m = _BLOCK.match(block.strip())
        if m is None or m.group(1) in out:
            return None
        out[m.group(1)] = m.group(2)
    return out or None


def _state_text(state, fmt: Optional[str] = None) -> str:
    rest = {k: v for k, v in state.items() if k not in ("image", "images")}
    return serialize_state(rest, fmt) if rest else ""


def state_variants(rows: Sequence[Dict]) -> List[Dict]:
    out = []
    for src in rows:
        st = src["state"]
        if not isinstance(st, dict) or not isinstance(st.get(CONTEXT_KEY), str) or not st[CONTEXT_KEY]:
            continue
        images = {k: v for k, v in st.items() if k in ("image", "images")}
        others = {k: v for k, v in st.items() if k not in ("image", "images", CONTEXT_KEY)}
        cands = [("prose", st, "prose"), ("text", st, "text"),
                 ("key_note", dict(images, note=st[CONTEXT_KEY], **others), None)]
        fields = context_fields(st[CONTEXT_KEY])
        if fields and not others:
            cands += [("struct_json", dict(images, **fields), None), ("struct_prose", dict(images, **fields), "prose")]
        seen = {_state_text(st)}
        for name, new, fmt in cands:
            txt = _state_text(new, fmt)
            if txt in seen:
                continue
            seen.add(txt)
            row = R._variant(src, "state_render", name, state=new, meta=dict(src.get("meta") or {}, state_text=txt[:200]))
            if fmt:
                row["state_format"] = fmt
            out.append(row)
    return out


def option_variants(rows: Sequence[Dict]) -> List[Dict]:
    out = []
    for src in rows:
        q = src["q"]
        if q["t"] != "choice" or not isinstance(q.get("crit"), dict) or any(q["crit"].values()):
            continue
        for case in ("lower", "title"):
            nq = normalize_choice_options(q, case)
            if list(nq["crit"]) == list(q["crit"]):
                continue
            out.append(R._variant(src, "option_render", case, q=nq))
    return out


def build(rows: Sequence[Dict], families: Sequence[str] = FAMILIES, seed: int = 0) -> List[Dict]:
    out = []
    if "state_render" in families:
        out += state_variants(rows)
    if "option_render" in families:
        out += option_variants(rows)
    return out


def _tv(p: Sequence[float], q: Sequence[float]) -> float:
    return 0.5 * float(np.abs(np.asarray(p) - np.asarray(q)).sum())


def summarize_render(preds: Sequence[Dict], n_boot: int = 1000, seed: int = 0) -> Dict:
    """Per dataset, family and variant: ``laya.robustness._stats`` (accuracy, paired change from the same source rows
    with its interval, flip rate, ECE) plus ``tv_mean`` (mean total-variation distance to the source row's
    probabilities) and ``dlogp_gold_mean`` (mean change of log P(gold)). ``"macro"`` averages ``delta_acc``,
    ``flip_rate`` and ``tv_mean`` per (family, variant) over the datasets that have it."""
    rng = np.random.default_rng(seed)
    out: Dict[str, Dict] = {}
    for name in sorted({p["dataset"] for p in preds}):
        rows = [p for p in preds if p["dataset"] == name]
        base = {p["group_id"]: p for p in rows if p["family"] == "orig"}
        rep: Dict[str, Dict] = {}
        for fam in FAMILIES:
            fr = [p for p in rows if p["family"] == fam and p["group_id"] in base]
            if not fr:
                continue
            rep[fam] = {}
            for v in sorted({p["variant"] for p in fr}):
                vr = [p for p in fr if p["variant"] == v]
                st = R._stats(vr, base, n_boot, rng)
                st["tv_mean"] = float(np.mean([_tv(p["probs"], base[p["group_id"]]["probs"]) for p in vr]))
                st["dlogp_gold_mean"] = float(np.mean([
                    np.log(max(p["probs"][p["label"]], 1e-9)) - np.log(max(base[p["group_id"]]["probs"][p["label"]], 1e-9))
                    for p in vr]))
                rep[fam][v] = st
        if rep:
            out[name] = rep
    macro: Dict[str, Dict] = {}
    for fam in FAMILIES:
        for v in sorted({v for rep in out.values() for v in rep.get(fam, {})}):
            per = [rep[fam][v] for rep in out.values() if v in rep.get(fam, {})]
            macro["%s/%s" % (fam, v)] = dict({"n_datasets": len(per)}, **{
                k: float(np.mean([s[k] for s in per])) for k in ("delta_acc", "flip_rate", "tv_mean", "dlogp_gold_mean")})
    return {"datasets": out, "macro": macro, "n_boot": n_boot, "seed": seed}


def format_table(summary: Dict) -> str:
    lines = ["| dataset | family | variant | rows | acc | vs orig [95% CI] | flip rate | mean TV | Δ log p(gold) |",
             "|---|---|---|---:|---:|---|---:|---:|---:|"]
    for name, rep in summary["datasets"].items():
        for fam, vs in rep.items():
            for v, s in vs.items():
                ci = s.get("delta_acc_ci", [float("nan")] * 2)
                lines.append("| %s | %s | %s | %d | %.3f | %+.3f [%+.3f, %+.3f] | %.3f | %.4f | %+.4f |" % (
                    name, fam, v, s["n_rows"], s["acc"], s["delta_acc"], ci[0], ci[1], s["flip_rate"], s["tv_mean"],
                    s["dlogp_gold_mean"]))
    return "\n".join(lines)
