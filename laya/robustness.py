"""Meaning-preserving robustness evaluations for the VLM decision model (``laya.vlm``).

Every perturbed row keeps a ``group_id`` back to the unperturbed source row it was built from, as in an
output-blind stability fixture: the perturbations are fixed from the inputs and a seed alone, before any
model output is seen, and the label is never re-derived from the model. Metrics are averaged within a group
first (so a source row with five variants counts once) and then across groups, and the 95% intervals come from a
bootstrap over *clusters* of source rows that share an image (a Cauldron image carries up to four questions and
a VQAv2 image several), so correlated questions are resampled together.

Families (``FAMILIES``), each deterministic given ``seed``:

* ``option_order``: the same row with its options shown in other orders: the cyclic shifts, the reversal, and
  one seeded random permutation, de-duplicated (a two-option ``noul`` row has one other order). The label is
  unchanged: ``collect_logits`` returns logits in label order whatever the display order (``"order"`` on the
  row), and ``shown_label`` records where the gold option was displayed, for accuracy by displayed position.
* ``text``: rule-based rewordings of the instructions and option descriptions (``TEXT_RULES``); a rule that
  does not apply to a row, or would leave it unchanged, produces no variant for it:
    - ``prefix``: prepend ``"Question: "``.
    - ``suffix``: append ``" Choose the correct option."``.
    - ``double_spaces``: collapse whitespace runs to one space, then double every space.
    - ``first_case``: toggle the case of the first letter, unless the first word is an acronym or mixed-case
      name (``"US"``, ``"iPhone"``), whose case carries meaning.
    - ``end_punct``: drop a trailing ``?`` / ``.`` / ``!``; add ``?`` (question word first) or ``.`` if none.
    - ``noul_frame`` (``noul``): a statement becomes ``"Is it true that <statement>?"``; a question is wrapped
      as ``"Decide whether the answer to this question is yes: <question>"``.
    - ``noul_options`` (``noul``): the option descriptions ``no`` / ``yes`` in place of the defaults
      (``"no, the statement does not hold"`` / ``"yes, the statement holds"``, ``laya.common.render_options``).
    - ``option_case`` (``choice``): toggle the first letter's case of every option that is a plain word (not a
      single character, a number, an acronym or a mixed-case name); skipped when two options differ only in case.
    - ``option_period`` (``choice``): end every option that has no final punctuation and is not a number with a
      period.
* ``image``: pixel-level changes that keep the content (``IMAGE_OPS``): JPEG re-encoding at quality 70/40/20,
  a centre crop to 95% of the area, a seeded random crop to 90% of the area, a downscale to half size and back,
  and brightness x0.9 / x1.1. Images are perturbed lazily inside the data loader (``realize``), from the row's
  ``"image_op"``; a row without an image gets no variant. The crops change the image size, which the
  ``preprocess="processor"`` path handles; the device-side ``"gpu"`` path cannot batch mixed sizes at all.
* ``image_shuffle``: the control. Each image-bearing row gets the image(s) of a *different* source image of the
  same dataset (a seeded derangement over distinct images, so a question never gets its own image back through a
  sibling question). Accuracy should fall towards the text-only prior; if it does not, the model is not reading
  the image (the SigLIP-projector failure in the README).
* ``text_only``: the images removed (the rest of the state, e.g. a ScienceQA hint, kept). With ``image_shuffle``
  this brackets what the image is worth.

``build_variants`` makes every row (``family == "orig"`` for the source rows); ``score_rows`` runs them through
``collect_logits`` and returns one prediction dict per row; ``summarize`` turns those into the report. Nothing
in ``summarize`` needs the model, so the committed per-row predictions can be re-summarised offline::

    python -m laya.robustness results/robustness/predictions.jsonl.gz
"""
import hashlib
import io
import json
import math
import random
import re
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from .common import ece_score, render_options

FAMILIES = ("option_order", "text", "image", "image_shuffle", "text_only")

# ---------------------------------------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------------------------------------


def _seed_for(*parts) -> int:
    """A stable 32-bit seed from strings / ints (``hash()`` is salted per process)."""
    return int(hashlib.sha256("/".join(str(p) for p in parts).encode()).hexdigest()[:8], 16)


def _image_key(state) -> Optional[tuple]:
    """The image(s) of a state as a hashable key (paths), or ``None`` for a text-only state."""
    if not isinstance(state, dict):
        return None
    if state.get("image") is not None:
        return (str(state["image"]),)
    if state.get("images"):
        return tuple(str(p) for p in state["images"])
    return None


def _with_images(state, key: Optional[tuple]):
    """``state`` with its image(s) replaced by ``key`` (``None``: removed; an empty state becomes ``""``)."""
    rest = {k: v for k, v in state.items() if k not in ("image", "images")} if isinstance(state, dict) else {}
    if key is None:
        return rest or ""
    if len(key) == 1:
        return dict(rest, image=key[0])
    return dict(rest, images=list(key))


def source_rows(examples: Sequence[Dict], n: int = 0, seed: int = 0, dataset: str = "") -> List[Dict]:
    """The unperturbed rows: ``n`` examples drawn by ``seed`` (all when ``n`` is 0 or larger than the set), in file
    order, each tagged ``group_id = "<dataset>/<file index>"``, ``cluster`` (its image, else its group) and
    ``family = variant = "orig"``."""
    idx = list(range(len(examples)))
    if n and n < len(idx):
        idx = sorted(random.Random(_seed_for("sample", dataset, seed)).sample(idx, n))
    out = []
    for i in idx:
        ex = examples[i]
        name = ex.get("dataset") or dataset or "_"
        gid = "%s/%06d" % (name, i)
        key = _image_key(ex["state"])
        out.append(dict(ex, dataset=name, group_id=gid, source_id=ex.get("id"), id=gid + "|orig",
                        cluster="%s/img:%s" % (name, "+".join(key)) if key else gid, family="orig", variant="orig"))
    return out


def _variant(src: Dict, family: str, variant: str, **changes) -> Dict:
    row = dict(src, family=family, variant=variant, id="%s|%s/%s" % (src["group_id"], family, variant))
    row.update(changes)
    return row


# ---------------------------------------------------------------------------------------------------------
# Option order
# ---------------------------------------------------------------------------------------------------------


def option_orders(k: int, seed: int, key: str = "") -> List[tuple]:
    """``(name, order)`` for the non-identity display orders of a ``k``-option row: ``shift<s>`` for each cyclic
    shift, ``reversed``, and ``perm`` (one random permutation seeded by ``seed`` and ``key``), de-duplicated."""
    seen, out = {tuple(range(k))}, []
    cands = [("shift%d" % s, [(i + s) % k for i in range(k)]) for s in range(1, k)]
    cands.append(("reversed", list(reversed(range(k)))))
    perm = list(range(k))
    random.Random(_seed_for("order", seed, key)).shuffle(perm)
    cands.append(("perm", perm))
    for name, order in cands:
        if tuple(order) not in seen:
            seen.add(tuple(order))
            out.append((name, order))
    return out


def order_variants(rows: Sequence[Dict], seed: int = 0, max_orders: int = 0) -> List[Dict]:
    """The ``option_order`` family (``max_orders`` > 0 keeps the first that many per row). ``order[j]`` is the
    label index of the option shown j-th, so the gold option is shown at ``order.index(label)``."""
    out = []
    for src in rows:
        k = len(src["target"])
        orders = option_orders(k, seed, src["group_id"])
        for name, order in orders[:max_orders or None]:
            out.append(_variant(src, "option_order", name, order=list(order), shown_label=order.index(src["label"])))
    return out


# ---------------------------------------------------------------------------------------------------------
# Text rewording
# ---------------------------------------------------------------------------------------------------------

QUESTION_WORDS = {"what", "which", "who", "whom", "whose", "where", "when", "why", "how", "is", "are", "was", "were",
                  "do", "does", "did", "can", "could", "will", "would", "should", "has", "have", "had", "may", "might"}


def _plain_word(w: str) -> bool:
    """A word whose first-letter case can be toggled without changing meaning: alphabetic, more than one letter,
    and at most its first letter upper case (not ``US``, ``iPhone``, ``McDonald``)."""
    return len(w) > 1 and w[0].isalpha() and not any(c.isupper() for c in w[1:])


def _toggle_first(s: str) -> Optional[str]:
    words = s.split()
    if not words or not _plain_word(re.sub(r"[^\w]+$", "", words[0])):
        return None
    lead = len(s) - len(s.lstrip())
    c = s[lead]
    return s[:lead] + (c.lower() if c.isupper() else c.upper()) + s[lead + 1:]


def _prefix(q):
    if q["ins"].lstrip().lower().startswith("question:"):
        return None
    return dict(q, ins="Question: " + q["ins"])


def _suffix(q):
    return dict(q, ins=q["ins"].rstrip() + " Choose the correct option.")


def _double_spaces(q):
    s = " ".join(q["ins"].split())
    return dict(q, ins=s.replace(" ", "  ")) if " " in s else None


def _first_case(q):
    s = _toggle_first(q["ins"])
    return dict(q, ins=s) if s is not None else None


def _end_punct(q):
    s = q["ins"].rstrip()
    if not s:
        return None
    if s[-1] in "?.!":
        return dict(q, ins=s[:-1].rstrip())
    first = s.split()[0].lower().strip("\"'(")
    return dict(q, ins=s + ("?" if first in QUESTION_WORDS else "."))


def _noul_frame(q):
    if q["t"] != "noul":
        return None
    s = q["ins"].strip()
    if not s:
        return None
    if s.endswith("?"):
        return dict(q, ins="Decide whether the answer to this question is yes: " + s)
    first = s.split()[0]
    body = (s[0].lower() + s[1:]) if _plain_word(re.sub(r"[^\w]+$", "", first)) else s
    return dict(q, ins="Is it true that %s?" % body.rstrip(".!").rstrip())


def _noul_options(q):
    if q["t"] != "noul" or q.get("crit"):
        return None
    return dict(q, crit={"false": "no", "true": "yes"})


def _choice_options(q, fn):
    if q["t"] != "choice" or not isinstance(q.get("crit"), dict):
        return None
    if len({k.lower() for k in q["crit"]}) < len(q["crit"]):
        return None  # options that differ only in case: a case or period change could blur which is which
    new = {}
    for k, v in q["crit"].items():
        new[fn(k) or k] = v
    if len(new) != len(q["crit"]) or list(new) == list(q["crit"]):
        return None  # a collision would drop an option; no change -> no variant
    return dict(q, crit=new)


def _option_case(q):
    return _choice_options(q, lambda s: _toggle_first(s) if _plain_word(s.split()[0] if s.split() else "") else None)


def _option_period(q):
    def one(s):
        s2 = s.rstrip()
        if not s2 or s2[-1] in ".?!:;,)]\"'" or re.fullmatch(r"[-+]?[\d.,/%]+", s2):
            return None
        return s2 + "."
    return _choice_options(q, one)


TEXT_RULES: Dict[str, Callable[[Dict], Optional[Dict]]] = {
    "prefix": _prefix, "suffix": _suffix, "double_spaces": _double_spaces, "first_case": _first_case,
    "end_punct": _end_punct, "noul_frame": _noul_frame, "noul_options": _noul_options,
    "option_case": _option_case, "option_period": _option_period,
}


def text_variants(rows: Sequence[Dict], rules: Sequence[str] = tuple(TEXT_RULES)) -> List[Dict]:
    """The ``text`` family: one row per (source row, applicable rule). Option order, and so the label and target,
    are unchanged by every rule (``option_case`` / ``option_period`` rename options in place)."""
    out = []
    for src in rows:
        for name in rules:
            q = TEXT_RULES[name](dict(src["q"]))
            if q is None or (q["ins"] == src["q"]["ins"] and render_options(q) == render_options(src["q"])):
                continue
            out.append(_variant(src, "text", name, q=q))
    return out


# ---------------------------------------------------------------------------------------------------------
# Image perturbations
# ---------------------------------------------------------------------------------------------------------

IMAGE_OPS: Dict[str, Dict] = {
    "jpeg70": {"op": "jpeg", "quality": 70},
    "jpeg40": {"op": "jpeg", "quality": 40},
    "jpeg20": {"op": "jpeg", "quality": 20},
    "crop_center95": {"op": "crop", "area": 0.95, "random": False},
    "crop_random90": {"op": "crop", "area": 0.90, "random": True},
    "rescale50": {"op": "rescale", "factor": 0.5},
    "bright90": {"op": "brightness", "factor": 0.9},
    "bright110": {"op": "brightness", "factor": 1.1},
}


def perturb_image(img, spec: Dict, seed: int = 0):
    """One ``IMAGE_OPS`` spec applied to a PIL image (RGB out, same size except for the crops)."""
    from PIL import Image, ImageEnhance

    img = img.convert("RGB")
    op = spec["op"]
    if op == "jpeg":
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=int(spec["quality"]))
        buf.seek(0)
        with Image.open(buf) as im:
            return im.convert("RGB")
    if op == "crop":
        w, h = img.size
        side = math.sqrt(spec["area"])
        cw, ch = max(1, round(w * side)), max(1, round(h * side))
        if spec.get("random"):
            rng = random.Random(seed)
            x0, y0 = rng.randint(0, w - cw), rng.randint(0, h - ch)
        else:
            x0, y0 = (w - cw) // 2, (h - ch) // 2
        return img.crop((x0, y0, x0 + cw, y0 + ch))
    if op == "rescale":
        w, h = img.size
        small = img.resize((max(1, round(w * spec["factor"])), max(1, round(h * spec["factor"]))), Image.BICUBIC)
        return small.resize((w, h), Image.BICUBIC)
    if op == "brightness":
        return ImageEnhance.Brightness(img).enhance(spec["factor"])
    raise ValueError("unknown image op %r" % op)


def image_variants(rows: Sequence[Dict], seed: int = 0, ops: Sequence[str] = tuple(IMAGE_OPS)) -> List[Dict]:
    """The ``image`` family: one row per (image-bearing source row, op). The pixels change at load time
    (``realize``); a random crop is seeded by ``seed`` and the row's *image*, so sibling questions on one image
    see the same crop."""
    out = []
    for src in rows:
        key = _image_key(src["state"])
        if key is None:
            continue
        for name in ops:
            spec = dict(IMAGE_OPS[name], seed=_seed_for("image", seed, name, *key))
            out.append(_variant(src, "image", name, image_op=spec))
    return out


def realize(ex: Dict) -> Dict:
    """``ex`` with its ``"image_op"`` applied to every image of its state (a no-op without one). Used as
    ``collect_logits(transform=...)`` so the perturbed pixels are made in the loader workers, never stored."""
    spec = ex.get("image_op")
    if not spec:
        return ex
    from PIL import Image

    def load(p, j):
        if isinstance(p, Image.Image):
            im = p
        else:
            with Image.open(p) as f:
                im = f.convert("RGB")
        return perturb_image(im, spec, spec.get("seed", 0) + j)

    st = dict(ex["state"])
    if st.get("image") is not None:
        st["image"] = load(st["image"], 0)
    if st.get("images"):
        st["images"] = [load(p, j) for j, p in enumerate(st["images"])]
    return dict(ex, state=st)


def shuffle_variants(rows: Sequence[Dict], seed: int = 0) -> List[Dict]:
    """The ``image_shuffle`` control: per dataset and image count, a seeded derangement of the distinct images
    (shuffle them, then give each the next one's), so no row gets its own image. Datasets with a single
    distinct image get no variant."""
    by = {}
    for src in rows:
        key = _image_key(src["state"])
        if key is not None:
            by.setdefault((src["dataset"], len(key)), []).append(src)
    out = []
    for (name, arity), srcs in sorted(by.items()):
        keys = sorted({_image_key(s["state"]) for s in srcs})
        if len(keys) < 2:
            continue
        random.Random(_seed_for("shuffle", seed, name, arity)).shuffle(keys)
        donor = {keys[i]: keys[(i + 1) % len(keys)] for i in range(len(keys))}
        for src in srcs:
            d = donor[_image_key(src["state"])]
            out.append(_variant(src, "image_shuffle", "shuffled", state=_with_images(src["state"], d),
                                donor_image="+".join(d)))
    return out


def text_only_variants(rows: Sequence[Dict]) -> List[Dict]:
    """The ``text_only`` control: every image-bearing row with its images removed."""
    return [_variant(src, "text_only", "no_image", state=_with_images(src["state"], None))
            for src in rows if _image_key(src["state"]) is not None]


def build_variants(rows: Sequence[Dict], families: Sequence[str] = FAMILIES, seed: int = 0) -> List[Dict]:
    """The source rows followed by every family's variants. Deterministic for a given ``rows`` and ``seed``."""
    makers = {"option_order": lambda: order_variants(rows, seed), "text": lambda: text_variants(rows),
              "image": lambda: image_variants(rows, seed), "image_shuffle": lambda: shuffle_variants(rows, seed),
              "text_only": lambda: text_only_variants(rows)}
    unknown = set(families) - set(makers)
    if unknown:
        raise ValueError("unknown families %s (expected %s)" % (sorted(unknown), FAMILIES))
    out = [dict(r) for r in rows]
    for fam in families:
        out += makers[fam]()
    ids = [r["id"] for r in out]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate row ids")
    return out


# ---------------------------------------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------------------------------------


def score_rows(model, processor, rows: Sequence[Dict], temperatures: Sequence[float] = (1.0, 1.0, 1.0),
               **kw) -> List[Dict]:
    """Run every row through ``collect_logits`` (each under its own display ``"order"``, images perturbed by
    ``realize``) and return one JSON-able prediction per row: raw logits and calibrated probabilities, both in
    label order."""
    import torch

    from .vlm_train import collect_logits

    recs = collect_logits(model, processor, list(rows), transform=realize, **kw)
    out = []
    for row, rec in zip(rows, recs):
        z = rec["logits"]
        p = torch.softmax(z / temperatures[rec["qtype"]], -1)
        pred = {k: row[k] for k in ("id", "group_id", "cluster", "dataset", "family", "variant", "label")}
        pred.update(qtype=int(rec["qtype"]), k=len(z), logits=[round(float(v), 4) for v in z],
                    probs=[round(float(v), 5) for v in p], pred=int(p.argmax()))
        for k in ("shown_label", "order", "donor_image"):
            if k in row:
                pred[k] = row[k]
        out.append(pred)
    return out


# ---------------------------------------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------------------------------------


def _groups(rows: Sequence[Dict], base: Dict[str, Dict]) -> List[Dict]:
    """Per source group: cluster, mean correctness, mean flip vs the source row, base correctness, 1/k."""
    by: Dict[str, List[Dict]] = {}
    for r in rows:
        by.setdefault(r["group_id"], []).append(r)
    out = []
    for gid, rs in by.items():
        b = base.get(gid)
        out.append({"cluster": rs[0]["cluster"], "acc": float(np.mean([r["pred"] == r["label"] for r in rs])),
                    "flip": float(np.mean([r["pred"] != b["pred"] for r in rs])) if b else float("nan"),
                    "any_flip": float(any(r["pred"] != b["pred"] for r in rs)) if b else float("nan"),
                    "base": float(b["pred"] == b["label"]) if b else float("nan"), "chance": 1.0 / rs[0]["k"],
                    "n": len(rs)})
    return out


def _balanced_acc(rows: Sequence[Dict]) -> float:
    """Mean per-label recall, each row weighted 1 / (its group's row count) so every source row counts once."""
    n_g: Dict[str, int] = {}
    for r in rows:
        n_g[r["group_id"]] = n_g.get(r["group_id"], 0) + 1
    num: Dict[int, float] = {}
    den: Dict[int, float] = {}
    for r in rows:
        w = 1.0 / n_g[r["group_id"]]
        num[r["label"]] = num.get(r["label"], 0.0) + w * (r["pred"] == r["label"])
        den[r["label"]] = den.get(r["label"], 0.0) + w
    return float(np.mean([num[c] / den[c] for c in den])) if den else float("nan")


def _ece(rows: Sequence[Dict]) -> float:
    if not rows:
        return float("nan")
    return ece_score(np.array([max(r["probs"]) for r in rows]), np.array([float(r["pred"] == r["label"]) for r in rows]))


def _ci(samples: np.ndarray) -> List[float]:
    s = samples[np.isfinite(samples)]
    return [float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))] if len(s) else [float("nan")] * 2


def _stats(rows: Sequence[Dict], base: Dict[str, Dict], n_boot: int, rng: np.random.Generator) -> Dict:
    """Group-averaged accuracy / balanced accuracy / flip rate / ECE for one (dataset, family) with cluster
    bootstrap 95% intervals, plus the same groups' unperturbed accuracy and the paired difference."""
    groups = _groups(rows, base)
    clusters = sorted({g["cluster"] for g in groups})
    cidx = {c: i for i, c in enumerate(clusters)}
    gc = np.array([cidx[g["cluster"]] for g in groups])
    A = {k: np.array([g[k] for g in groups]) for k in ("acc", "flip", "any_flip", "base", "chance")}
    res = {"n_groups": len(groups), "n_rows": len(rows), "n_clusters": len(clusters),
           "acc": float(A["acc"].mean()), "balanced_acc": _balanced_acc(rows), "flip_rate": float(np.nanmean(A["flip"])),
           "any_flip_rate": float(np.nanmean(A["any_flip"])), "ece": _ece(rows),
           "base_acc": float(np.nanmean(A["base"])), "chance": float(A["chance"].mean())}
    res["delta_acc"] = res["acc"] - res["base_acc"]
    if n_boot and len(clusters) > 1:
        # per-cluster sums, so one resample is a weighted sum over clusters
        ncl = len(clusters)
        cnt = np.bincount(gc, minlength=ncl).astype(float)
        sums = {k: np.bincount(gc, weights=np.nan_to_num(A[k]), minlength=ncl) for k in ("acc", "flip", "base")}
        row_c = np.array([cidx[r["cluster"]] for r in rows])
        rows_of = [np.flatnonzero(row_c == i) for i in range(ncl)]
        conf = np.array([max(r["probs"]) for r in rows])
        corr = np.array([float(r["pred"] == r["label"]) for r in rows])
        W = rng.multinomial(ncl, np.full(ncl, 1.0 / ncl), size=n_boot).astype(float)  # resample counts per cluster
        tot = W @ cnt
        boot = {k: (W @ sums[k]) / np.maximum(tot, 1) for k in sums}
        boot["delta"] = boot["acc"] - boot["base"]
        eces = []
        for w in W[: min(n_boot, 400)]:  # ECE needs the rows themselves; 400 resamples is plenty for its interval
            idx = np.concatenate([np.repeat(rows_of[i], int(c)) for i, c in enumerate(w) if c])
            eces.append(ece_score(conf[idx], corr[idx]))
        res.update(acc_ci=_ci(boot["acc"]), base_acc_ci=_ci(boot["base"]), delta_acc_ci=_ci(boot["delta"]),
                   flip_rate_ci=_ci(boot["flip"]), ece_ci=_ci(np.array(eces)))
    return res


def summarize(preds: Sequence[Dict], n_boot: int = 1000, seed: int = 0) -> Dict:
    """The report: per dataset and family (``"orig"`` included), ``_stats`` plus per-variant accuracy, paired change
    from the same rows unperturbed, and flip rate; for ``option_order`` the accuracy by variant (``orig``
    included), its spread over the orders every row has, and the accuracy by the gold option's displayed position; for the image controls the label prior (majority-label accuracy on the
    source rows) and how often the shuffled-image and no-image predictions agree. ``"macro"`` averages each
    family's point estimates over the datasets that have it."""
    rng = np.random.default_rng(seed)
    out: Dict[str, Dict] = {}
    for name in sorted({p["dataset"] for p in preds}):
        rows = [p for p in preds if p["dataset"] == name]
        base = {p["group_id"]: p for p in rows if p["family"] == "orig"}
        rep: Dict[str, Dict] = {}
        for fam in ("orig",) + FAMILIES:
            fr = [p for p in rows if p["family"] == fam]
            if not fr:
                continue
            st = _stats(fr, base, n_boot, rng)
            if fam != "orig":
                st["variants"] = {}
                for v in sorted({p["variant"] for p in fr}):
                    vr = [p for p in fr if p["variant"] == v]
                    vb = [(p, base[p["group_id"]]) for p in vr if p["group_id"] in base]
                    st["variants"][v] = {"n": len(vr), "acc": float(np.mean([p["pred"] == p["label"] for p in vr])),
                                         "delta_acc": float(np.mean([(p["pred"] == p["label"]) - (b["pred"] == b["label"])
                                                                     for p, b in vb])),
                                         "flip_rate": float(np.mean([p["pred"] != b["pred"] for p, b in vb]))}
            if fam == "option_order":
                accs = dict({"orig": float(np.mean([base[g]["pred"] == base[g]["label"] for g in {p["group_id"] for p in fr}]))},
                            **{v: s["acc"] for v, s in st["variants"].items()})
                st["acc_by_order"] = accs
                # the spread only over orders every row has: ``shift4`` exists only for 5+ options, so on a set
                # with mixed option counts it is measured on a different (and small) population
                common = ["orig"] + [v for v, s in st["variants"].items() if s["n"] == st["n_groups"]]
                st["acc_spread_orders"] = common
                st["acc_spread"] = max(accs[v] for v in common) - min(accs[v] for v in common)
                pos: Dict[int, List[float]] = {}
                for p in fr + [dict(b, shown_label=b["label"]) for g, b in base.items()]:
                    pos.setdefault(int(p["shown_label"]), []).append(float(p["pred"] == p["label"]))
                st["acc_by_gold_position"] = {str(k): {"n": len(v), "acc": float(np.mean(v))} for k, v in sorted(pos.items())}
            if fam in ("image_shuffle", "text_only"):
                labels = [base[g]["label"] for g in {p["group_id"] for p in fr}]
                st["majority_label_acc"] = max(labels.count(c) for c in set(labels)) / len(labels)
            rep[fam] = st
        sh = {p["group_id"]: p["pred"] for p in rows if p["family"] == "image_shuffle"}
        to = {p["group_id"]: p["pred"] for p in rows if p["family"] == "text_only"}
        common = sorted(set(sh) & set(to))
        if common:
            rep["image_shuffle"]["agree_with_text_only"] = float(np.mean([sh[g] == to[g] for g in common]))
        out[name] = rep
    macro: Dict[str, Dict] = {}
    for fam in ("orig",) + FAMILIES:
        per = [out[d][fam] for d in out if fam in out[d]]
        if per:
            macro[fam] = {"n_datasets": len(per)}
            for k in ("acc", "balanced_acc", "base_acc", "delta_acc", "flip_rate", "ece"):
                macro[fam][k] = float(np.nanmean([s[k] for s in per]))
    return {"datasets": out, "macro": macro, "n_boot": n_boot, "seed": seed}


def format_table(summary: Dict) -> str:
    """A markdown table: one line per dataset and family, accuracy with its interval, change from unperturbed,
    flip rate and ECE."""
    lines = ["| dataset | family | groups | acc [95% CI] | bal. acc | vs orig | flip rate | ECE |",
             "|---|---|---:|---|---:|---:|---:|---:|"]
    for name, rep in summary["datasets"].items():
        for fam, s in rep.items():
            ci = s.get("acc_ci", [float("nan")] * 2)
            lines.append("| %s | %s | %d | %.3f [%.3f, %.3f] | %.3f | %s | %s | %.3f |" % (
                name, fam, s["n_groups"], s["acc"], ci[0], ci[1], s["balanced_acc"],
                "" if fam == "orig" else "%+.3f" % s["delta_acc"], "" if fam == "orig" else "%.3f" % s["flip_rate"],
                s["ece"]))
    return "\n".join(lines)


def main(argv=None):
    import argparse
    import gzip

    ap = argparse.ArgumentParser(description="Re-summarise committed robustness predictions (jsonl or jsonl.gz).")
    ap.add_argument("predictions")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    opener = gzip.open if a.predictions.endswith(".gz") else open
    with opener(a.predictions, "rt") as f:
        preds = [json.loads(line) for line in f if line.strip()]
    s = summarize(preds, a.n_boot, a.seed)
    print(format_table(s))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(s, f, indent=1)


if __name__ == "__main__":
    main()
