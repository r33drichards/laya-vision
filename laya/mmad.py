"""Mapping the MMAD industrial-anomaly benchmark onto typed decision questions.

`MMAD <https://github.com/jam-cc/MMAD>`_ (Jiang et al., ICLR 2025) asks 39,670 multiple-choice questions
about 8,366 industrial images. Every question has 2 or 4 options, which is exactly a ``choice`` question, so
``predict`` answers it natively: no text is generated, so there is no reply to regex and no refusal to
retry, and ``argmax`` over the options *is* the answer letter.

This module is the pure part -- building the questions, recovering the letter, and the calibration numbers
MMAD's letter-only scorer throws away. ``modal_mmad.py`` runs it at scale on Modal and scores with MMAD's
own ``summary.py``; ``examples/mmad_local.py`` runs a small sample on a laptop. Both import from here so a
local spot-check and the full sweep cannot answer the same question differently.
"""
from typing import Dict, List, Sequence, Tuple

#: the five sub-datasets, in ascending download size
SUBSETS = ("DS-MVTec", "MVTec-AD", "VisA", "MVTec-LOCO", "GoodsAD")

#: MMAD's nine subtasks. Anomaly Detection is the one this mapping answers under the reference protocol:
#: it is always question index 0, so the reference scripts ask it with no prior context, exactly as we do.
#: The others are asked cumulatively there (question i carries questions 1..i-1 in the same completion) and
#: are phrased to presuppose the defect, which a one-pass model cannot reproduce. See docs/mmad.md.
DETECTION = "Anomaly Detection"

#: MMAD's scorer calls an image normal when its path contains this
NORMAL_FLAG = "good"

MMAD_REF = "refs%2Fpr%2F1"  # the HF revision that serves the images as individual files, not as zips
MMAD_JSON_URL = "https://raw.githubusercontent.com/jam-cc/MMAD/main/dataset/MMAD/mmad.json"


def image_url(rel_path: str) -> str:
    """URL of one MMAD image, e.g. ``DS-MVTec/bottle/image/broken_large/000.png``.

    A function rather than a ``%s`` template on purpose: the revision is percent-encoded (``refs%2Fpr%2F1``)
    and ``%2F`` is a valid printf conversion, so ``TEMPLATE % rel`` raises instead of interpolating.
    """
    return "https://huggingface.co/datasets/jiang-cc/MMAD/resolve/" + MMAD_REF + "/" + rel_path

#: what the query image and its reference template are, spelled out for the model the way MMAD's own prompt
#: spells it out for a chat model ("The last image is the query image")
NOTE_1SHOT = ("the first image is a normal reference sample of the same product; "
              "the last image is the query sample to inspect")


def build_questions(conversation: Sequence[Dict], noul_detection: bool = False) -> Tuple[Dict, Dict]:
    """MMAD conversation -> (``predict`` questions, per-question option letters and texts).

    The default is a uniform ``choice`` per question, in the benchmark's own option order. MMAD balances
    that order itself -- 4,195 of the yes/no questions put "Yes." at A and 4,077 put it at B -- so no
    de-biasing is needed to be fair to the model.

    ``noul_detection`` instead routes two-option yes/no questions through the ``noul`` head, the one head
    actually trained on yes/no data. It is a diagnostic, not the headline setting; the 27 four-option
    detection questions ("Maybe."/"Unknown.") stay on ``choice`` either way.
    """
    questions, meta = {}, {}
    for i, c in enumerate(conversation):
        qid = "q%d" % i
        letters = list(c["Options"].keys())
        texts = [c["Options"][l] for l in letters]
        if noul_detection and len(texts) == 2 and set(texts) == {"Yes.", "No."}:
            questions[qid] = {"type": "noul", "instructions": c["Question"]}
        else:
            questions[qid] = {"type": "choice", "instructions": c["Question"], "criteria": texts}
        meta[qid] = {"letters": letters, "texts": texts, "type": c["type"], "answer": c["Answer"],
                     "question": c["Question"], "options": c["Options"]}
    return questions, meta


def to_letter(answer: Dict, m: Dict) -> Tuple[str, Dict[str, float]]:
    """One ``predict`` answer -> (MMAD answer letter, probabilities keyed by letter).

    MMAD grades a letter, so this is where a silent off-by-one would turn into a plausible-looking accuracy
    rather than a crash. ``tests/test_mmad.py`` pins it for both yes/no letter orders.
    """
    letters, texts = m["letters"], m["texts"]
    if answer["type"] == "choice":
        probs = {letters[texts.index(t)]: p for t, p in answer["probabilities"].items()}
        return letters[texts.index(answer["choice"])], probs
    p_yes = float(answer["noul"])
    probs = {letters[texts.index("Yes.")]: round(p_yes, 4), letters[texts.index("No.")]: round(1 - p_yes, 4)}
    return letters[texts.index("Yes." if p_yes >= 0.5 else "No.")], probs


def question_text(m: Dict) -> Dict[str, str]:
    """The prompt text MMAD's own scripts record, so an answers file here is a drop-in for theirs."""
    opts = "".join("%s. %s\n" % (l, m["options"][l]) for l in m["letters"])
    return {"type": "text", "text": "Question: %s \n%s" % (m["question"], opts)}


def is_normal(image_path: str) -> bool:
    """MMAD's own test for a defect-free image."""
    return NORMAL_FLAG in image_path


def yes_letter(record: Dict) -> str:
    """The option letter holding "Yes." in a recorded detection question, or "" if there isn't exactly one."""
    hits = [ln.split(".")[0].strip() for ln in record["question"]["text"].splitlines() if ln.strip().endswith("Yes.")]
    return hits[0] if len(hits) == 1 else ""


def balanced_accuracy(records: Sequence[Dict]) -> Dict:
    """Detection scored the way MMAD scores it: (normal_acc + anomaly_acc) / 2, plus recall/precision/F1.

    This is MMAD's formula, not MMAD's code -- ``modal_mmad.report`` calls their ``summary.py`` for the
    authoritative table. It exists so a laptop sample can report the same quantity without the full repo.
    """
    tp = fp = fn = tn = 0
    for r in records:
        if r["question_type"] != DETECTION:
            continue
        right = r["gpt_answer"] == r["correct_answer"]
        if is_normal(r["image"]):
            tn += right
            fp += not right
        else:
            tp += right
            fn += not right
    normal_n, anom_n = tn + fp, tp + fn
    normal_acc = tn / normal_n if normal_n else 0.0
    anom_acc = tp / anom_n if anom_n else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = anom_acc
    return {
        "n_normal": normal_n,
        "n_anomalous": anom_n,
        "normal_acc": round(normal_acc, 4),
        "anomaly_acc": round(anom_acc, 4),
        "balanced_acc": round((normal_acc + anom_acc) / 2, 4),
        "overkill": round(1 - normal_acc, 4),
        "miss": round(1 - anom_acc, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(2 * precision * recall / (precision + recall), 4) if precision + recall else 0.0,
    }


def calibration_extras(records: Sequence[Dict], n_bins: int = 10) -> Dict:
    """Accuracy, ECE and detection AUROC from the per-option probabilities a letter-only scorer discards.

    ECE bins top-probability confidence against accuracy. AUROC is over the detection questions only,
    scoring P("Yes.") against MMAD's own ground truth for whether the image is defect-free. Tied scores get
    averaged ranks, so a model that answers uniformly scores 0.5 rather than whatever the sort order gives.
    """
    import numpy as np

    conf, hit, p_anom, y_anom = [], [], [], []
    for r in records:
        probs = r.get("probabilities") or {}
        if not probs:
            continue
        conf.append(max(probs.values()))
        hit.append(float(r["gpt_answer"] == r["correct_answer"]))
        if r["question_type"] == DETECTION:
            yes = yes_letter(r)
            if yes and yes in probs:
                p_anom.append(probs[yes])
                y_anom.append(0.0 if is_normal(r["image"]) else 1.0)
    if not hit:
        return {}

    c, h = np.array(conf), np.array(hit)
    out = {"overall_accuracy": round(float(h.mean()), 4), "n_questions": len(h)}
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (c > lo) & (c <= hi)
        if m.any():
            ece += abs(h[m].mean() - c[m].mean()) * m.mean()
    out["ece"] = round(float(ece), 4)

    if len(set(y_anom)) == 2:
        p, y = np.array(p_anom), np.array(y_anom)
        order = p.argsort(kind="mergesort")
        ranks = np.empty(len(p), dtype=float)
        ranks[order] = np.arange(1, len(p) + 1)
        _, inv, counts = np.unique(p, return_inverse=True, return_counts=True)
        sums = np.zeros(len(counts))
        np.add.at(sums, inv, ranks)
        ranks = (sums / counts)[inv]  # average the ranks of ties
        n1 = float(y.sum())
        n0 = float(len(y) - n1)
        out["detection_auroc"] = round(float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)), 4)
        out["detection_n"] = len(p)
    return out


def detection_scores(records: Sequence[Dict]):
    """(P(defective), is_defective) for every detection question that carries probabilities."""
    import numpy as np

    p, y = [], []
    for r in records:
        probs = r.get("probabilities") or {}
        if r["question_type"] != DETECTION or not probs:
            continue
        yes = yes_letter(r)
        if yes and yes in probs:
            p.append(probs[yes])
            y.append(0.0 if is_normal(r["image"]) else 1.0)
    return np.array(p), np.array(y)


def _bal_acc(p, y, t):
    pred = p >= t
    pos, neg = y == 1, y == 0
    if not pos.any() or not neg.any():
        return 0.0
    return 0.5 * (pred[pos].mean() + (~pred[neg]).mean())


def threshold_sweep(records: Sequence[Dict], folds: int = 5, seed: int = 0) -> Dict:
    """What the fixed 0.5 decision threshold costs, if anything.

    MMAD grades a letter, so the model's answer is ``P(defective) >= 0.5``. When AUROC comes out well above
    balanced accuracy, that threshold -- not the model's ranking -- is what is losing the defects. This
    reports three things:

    * ``at_half``: balanced accuracy at the 0.5 the benchmark actually scores.
    * ``oracle``: the best achievable threshold, *chosen on this same data*. Optimistic by construction and
      not a benchmark result -- it is an upper bound on what retuning could buy.
    * ``cv``: the honest version. The threshold is fitted on k-1 folds and scored on the held-out one, so
      it estimates what a threshold tuned on separate data would actually deliver.
    """
    import numpy as np

    p, y = detection_scores(records)
    if len(p) == 0 or len(set(y.tolist())) < 2:
        return {}

    cands = np.unique(p)
    cands = np.concatenate([[0.0], (cands[:-1] + cands[1:]) / 2 if len(cands) > 1 else [], cands, [1.01]])
    cands = np.unique(cands)
    scores = np.array([_bal_acc(p, y, t) for t in cands])
    best = int(scores.argmax())

    rng = np.random.default_rng(seed)
    fold_of = rng.permutation(len(p)) % folds
    held = []
    for f in range(folds):
        test = fold_of == f
        train = ~test
        if len(set(y[train].tolist())) < 2 or len(set(y[test].tolist())) < 2:
            continue
        t_star = cands[int(np.array([_bal_acc(p[train], y[train], t) for t in cands]).argmax())]
        held.append(_bal_acc(p[test], y[test], t_star))
    return {
        "n": int(len(p)),
        "at_half": round(float(_bal_acc(p, y, 0.5)), 4),
        "oracle": round(float(scores[best]), 4),
        "oracle_threshold": round(float(cands[best]), 4),
        "cv": round(float(np.mean(held)), 4) if held else None,
        "cv_folds": len(held),
    }


def answer_record(image_key: str, m: Dict, letter: str, probs: Dict[str, float], answer: Dict) -> Dict:
    """One row of the answers file, in the schema MMAD's ``caculate_accuracy_mmad`` reads."""
    return {
        "image": image_key,
        "question": question_text(m),
        "question_type": m["type"],
        "correct_answer": m["answer"],
        "gpt_answer": letter,
        "probabilities": probs,
        "confidence": answer["confidence"],
        "head": answer["type"],
    }


def select_conversations(chat_ad: Dict, subsets: Sequence[str], question_types: Sequence[str]) -> Dict[str, List[Dict]]:
    """Image key -> the questions of the wanted types, dropping images that end up with none."""
    want_sub, want_type = set(subsets), set(question_types)
    out = {}
    for key, v in chat_ad.items():
        if want_sub and key.split("/")[0] not in want_sub:
            continue
        conv = [c for c in v["conversation"] if not want_type or c["type"] in want_type]
        if conv:
            out[key] = conv
    return out
