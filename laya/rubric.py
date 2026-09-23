"""Rubric-scored image datasets as post-training data for ``score`` questions.

The released checkpoints never saw an ordinal question: every training set was ``choice`` or ``noul``, so
``score`` outputs are meaningless. This module turns four public datasets whose labels are graded levels into
prepared-dataset records (the jsonl format ``laya.vlm_train.load_jsonl_examples`` reads), each a ``score``
question whose ``criteria`` is a rubric written the way a user of ``predict`` would write one:

* **VLFeedback** (``MMInstruction/VLFeedback``): 80k image questions, 4 model responses each, rated 1-5 by GPT-4V
  on helpfulness and visual faithfulness (and ethics, which is nearly always 5 and is dropped). The image is the
  state, the question and the response go in ``state_text``, and the rating is the level. This is the rubric
  shape of ``laya.presets`` (``frustration``: "calm and neutral" ... "very angry"), applied to a response.
* **AVA** (``trojblue/AVA-aesthetics-10pct-min50-10bins``): 25k photos with the full histogram of 1-10 human
  aesthetic votes (at least 50 per image). The 10 ratings collapse to 5 levels and the histogram becomes a soft
  ``target``, which the ranked probability score in ``laya.common.proper_reward`` was built for.
* **RichHF-18K** (``Exploration/richhf_18k_with_images``): 17.8k generated images rated by three people on a
  5-point scale for plausibility (``artifact_score``), prompt alignment (``misalignment_score``), aesthetics and
  overall quality, stored as ``(mean - 1) / 4``; the level is that mean rounded back to 0-4. All four are read as
  "higher is better", as in the RichHF paper's tables. The prompt goes in ``state_text``.
* **CrisisMMD** (``QCRI/CrisisMMD``, ``damage``): 3.5k disaster photos labelled little / mild / severe damage.
  Image only; the tweet is dropped so the level has to come from the picture.

Cleanup that makes the records fit the model:

* ``state_text`` is capped (``max_chars``) because the causal sequence keeps the head of the state and drops the
  tail (``build_vlm_inputs`` with ``truncate_left=False``); the question is put first so it always survives.
* Each rubric level is one short clause: options are cut to 48 tokens each and the whole head to 256.
* Every dataset offers several phrasings of its instructions (``INSTRUCTIONS``); one is drawn per record so the
  head learns the rubric, not one sentence.
* ``balance_levels`` caps the count of any level at ``max_ratio`` times the median level count, since the ratings
  pile up at the top (VLFeedback helpfulness, RichHF) or the middle (AVA).
* At most ``max_texts`` records per image, sampled, so 4 responses x 2 aspects do not make one photo 8 rows.

The Modal job ``prepare_score`` in ``modal_app.py`` streams the sources, saves the images and writes the jsonl
files under ``/data/vqa/score_<name>``.
"""
import random
import re
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence

#: prepared dataset names (``score_<name>``) and their Hub sources
SOURCES = {
    "vlfeedback": "MMInstruction/VLFeedback",
    "ava": "trojblue/AVA-aesthetics-10pct-min50-10bins",
    "richhf": "Exploration/richhf_18k_with_images",
    "crisismmd": "QCRI/CrisisMMD",
}

# -- rubrics ----------------------------------------------------------------------------------------------------
# One criteria list per question (level 0 first); several instruction phrasings per question, drawn per record.

CRITERIA = {
    "helpfulness": [
        "unhelpful: ignores or misreads the question",
        "barely helpful: partly relevant, thin or partly wrong",
        "somewhat helpful: answers the question with gaps or minor errors",
        "helpful: answers the question correctly and clearly",
        "very helpful: complete, accurate and well explained",
    ],
    "faithfulness": [
        "unfaithful: describes things that are not in the image",
        "mostly unfaithful: several claims contradict the image",
        "partly faithful: mostly right with some invented or wrong details",
        "faithful: consistent with the image, minor imprecision at most",
        "fully faithful: everything stated matches the image",
    ],
    "aesthetics": [
        "very poor: unappealing, badly composed or exposed",
        "below average: weak composition, light or colour",
        "average: competent but unremarkable",
        "good: pleasing composition, light and colour",
        "excellent: striking, professional quality",
    ],
    "plausibility": [
        "implausible: obvious artifacts, broken anatomy or geometry",
        "mostly implausible: several clear artifacts",
        "somewhat plausible: a few visible flaws",
        "plausible: looks real at a glance, minor flaws",
        "fully plausible: no visible artifacts",
    ],
    "alignment": [
        "does not match: the prompt's subject is missing",
        "poor match: only a small part of the prompt is shown",
        "partial match: the main subject is there, details are wrong or missing",
        "good match: nearly everything in the prompt is shown",
        "exact match: every element of the prompt is shown",
    ],
    "overall": [
        "very poor",
        "poor",
        "acceptable",
        "good",
        "excellent",
    ],
    "quality": [
        "bad: heavily blurred, noisy, badly exposed or distorted",
        "poor: clearly visible blur, noise or exposure problems",
        "fair: acceptable, with some visible flaws",
        "good: sharp and well exposed, minor flaws at most",
        "excellent: technically flawless",
    ],
    "damage": [
        "little or no damage",
        "mild damage: some visible harm to buildings, roads or objects",
        "severe damage: destruction or heavy structural harm",
    ],
}

INSTRUCTIONS = {
    "helpfulness": [
        "How helpful is the response to the question about the image?",
        "Rate how well the response answers the user's question about this image.",
        "Grade the response's helpfulness for the question asked about the picture.",
    ],
    "faithfulness": [
        "How faithful is the response to what the image actually shows?",
        "Rate whether the response's claims about the image are true to the picture.",
        "Grade the visual faithfulness of the response: does it describe what is really in the image?",
    ],
    "aesthetics": [
        "How aesthetically pleasing is this photo?",
        "Rate the aesthetic quality of the image.",
        "Grade this picture's visual appeal as a photograph.",
    ],
    "aesthetics_generated": [
        "How aesthetically pleasing is this generated image?",
        "Rate the aesthetic quality of the image.",
        "Grade this picture's visual appeal.",
    ],
    "plausibility": [
        "How plausible does this generated image look? Consider artifacts, anatomy and geometry.",
        "Rate how free of visual artifacts this image is.",
        "Grade the image's plausibility: could it pass for a real, artifact-free picture?",
    ],
    "alignment": [
        "How well does the image match the prompt it was generated from?",
        "Rate how closely the picture follows the text prompt.",
        "Grade the image-prompt alignment: is everything the prompt asks for shown?",
    ],
    "overall": [
        "Rate the overall quality of this generated image for its prompt.",
        "How good is this image overall, given the prompt?",
        "Grade the image's overall quality, taking the prompt into account.",
    ],
    "quality": [
        "How good is the technical quality of this photo?",
        "Rate the image quality: sharpness, noise, exposure and distortions.",
        "Grade how technically clean this picture is.",
    ],
    "damage": [
        "How much damage does the scene in the photo show?",
        "Rate the level of physical damage visible in this image.",
        "Grade the severity of damage to buildings, roads or objects in the picture.",
    ],
}


def rubric_question(aspect: str, rng: Optional[random.Random] = None, instructions_key: Optional[str] = None) -> Dict:
    """A ``score`` question for ``aspect``: one of its instruction phrasings (the first without ``rng``) and its
    criteria. ``instructions_key`` picks a different phrasing set over the same criteria."""
    phrasings = INSTRUCTIONS[instructions_key or aspect]
    ins = rng.choice(phrasings) if rng is not None else phrasings[0]
    return {"type": "score", "instructions": ins, "criteria": list(CRITERIA[aspect])}


def clip_text(text: str, max_chars: int) -> str:
    """Whitespace-normalised ``text`` cut at ``max_chars`` on a word boundary, with an ellipsis when cut."""
    text = re.sub(r"[ \t]+", " ", text.strip())
    text = re.sub(r"\n{3,}", "\n\n", text)
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    if " " in cut[max_chars // 2:]:
        cut = cut[: cut.rfind(" ")]
    return cut.rstrip() + " ..."


# -- VLFeedback ---------------------------------------------------------------------------------------------------

VLFEEDBACK_ASPECTS = {"Helpfulness": "helpfulness", "Visual Faithfulness": "faithfulness"}


def _completions(completions) -> List[Dict]:
    """The row's completions as ``[{"model", "response", "annotations"}, ...]``: the parquet stores a struct of
    lists, older exports a list of structs."""
    if isinstance(completions, dict):
        n = len(completions.get("response") or [])
        return [{k: (completions[k][i] if completions.get(k) is not None and i < len(completions[k]) else None)
                 for k in completions} for i in range(n)]
    return list(completions or [])


def _rating(annotation) -> Optional[int]:
    """``{"Rating": "4", ...}`` -> 4 (1-5); anything else -> None."""
    if not isinstance(annotation, dict):
        return None
    m = re.match(r"^\s*([1-5])\s*$", str(annotation.get("Rating", "")))
    return int(m.group(1)) if m else None


def vlfeedback_records(row: Dict, row_id: str, rng: Optional[random.Random] = None, max_texts: int = 2,
                       max_chars: int = 1200, max_prompt_chars: int = 400) -> List[Dict]:
    """One VLFeedback row -> at most ``max_texts`` records: a (response, aspect) pair each, rated 1-5 -> level 0-4."""
    prompt = clip_text(str(row.get("prompt") or ""), max_prompt_chars)
    if not prompt:
        return []
    out = []
    for i, c in enumerate(_completions(row.get("completions"))):
        response = clip_text(str(c.get("response") or ""), max_chars)
        if not response:
            continue
        ann = c.get("annotations") or {}
        for key, aspect in VLFEEDBACK_ASPECTS.items():
            r = _rating(ann.get(key))
            if r is None:
                continue
            out.append({"id": "%s-%d-%s" % (row_id, i, aspect), "state_text": "Question: %s\n\nResponse: %s" % (prompt, response),
                        "question": rubric_question(aspect, rng), "label": r - 1, "source": "VLFeedback/%s" % (c.get("model") or "?")})
    if len(out) > max_texts:
        out = rng.sample(out, max_texts) if rng is not None else out[:max_texts]
    return out


# -- AVA ----------------------------------------------------------------------------------------------------------

AVA_LEVELS = 5


def collapse_counts(counts: Sequence[float], levels: int = AVA_LEVELS) -> List[float]:
    """Vote counts over ``len(counts)`` consecutive ratings summed into ``levels`` equal bins (10 -> 5: 1-2, 3-4, ...)."""
    n = len(counts)
    if n % levels:
        raise ValueError("%d ratings do not split into %d equal bins" % (n, levels))
    per = n // levels
    return [float(sum(counts[i * per:(i + 1) * per])) for i in range(levels)]


def ava_record(row: Dict, row_id: str, rng: Optional[random.Random] = None, min_votes: int = 10) -> Optional[Dict]:
    """One AVA row -> a ``score`` record with the collapsed vote histogram as a soft ``target``."""
    counts = list(row.get("rating_counts") or [])
    if len(counts) != 10 or sum(counts) < min_votes:
        return None
    target = collapse_counts(counts)
    total = sum(target)
    target = [c / total for c in target]
    label = max(range(len(target)), key=lambda i: (target[i], -i))  # ties go to the lower level
    return {"id": row_id, "state_text": None, "question": rubric_question("aesthetics", rng), "label": label,
            "target": [round(p, 4) for p in target], "source": "AVA"}


# -- RichHF-18K ---------------------------------------------------------------------------------------------------

RICHHF_ASPECTS = {"aesthetics_score": "aesthetics", "artifact_score": "plausibility",
                  "misalignment_score": "alignment", "overall_score": "overall"}


def richhf_level(score: float, levels: int = 5) -> Optional[int]:
    """``(mean rating - 1) / 4`` in [0, 1] -> the mean rating rounded to a level 0..levels-1."""
    if score is None or not 0.0 <= float(score) <= 1.0:
        return None
    return int(min(levels - 1, max(0, round(float(score) * (levels - 1)))))


def richhf_records(row: Dict, row_id: str, rng: Optional[random.Random] = None, max_texts: int = 2,
                   max_prompt_chars: int = 400) -> List[Dict]:
    """One RichHF row -> at most ``max_texts`` of its four aspect ratings. The prompt is the state for alignment
    and overall; aesthetics and plausibility are asked of the image alone."""
    caption = clip_text(str(row.get("caption") or ""), max_prompt_chars)
    out = []
    for col, aspect in RICHHF_ASPECTS.items():
        level = richhf_level(row.get(col))
        if level is None:
            continue
        if aspect in ("alignment", "overall"):
            if not caption:
                continue
            state_text = "Prompt: " + caption
        else:
            state_text = None
        q = rubric_question(aspect, rng, "aesthetics_generated" if aspect == "aesthetics" else None)
        out.append({"id": "%s-%s" % (row_id, aspect), "state_text": state_text, "question": q, "label": level, "source": "RichHF-18K"})
    if len(out) > max_texts:
        out = rng.sample(out, max_texts) if rng is not None else out[:max_texts]
    return out


# -- CrisisMMD ----------------------------------------------------------------------------------------------------

CRISISMMD_LABELS = ("little_or_no_damage", "mild_damage", "severe_damage")


def crisismmd_record(row: Dict, row_id: str, rng: Optional[random.Random] = None) -> Optional[Dict]:
    """One CrisisMMD ``damage`` row -> a 3-level ``score`` record. ``label`` may be the class index or its name."""
    label = row.get("label")
    if isinstance(label, str):
        if label not in CRISISMMD_LABELS:
            return None
        label = CRISISMMD_LABELS.index(label)
    if label is None or not 0 <= int(label) < 3:
        return None
    return {"id": row_id, "state_text": None, "question": rubric_question("damage", rng), "label": int(label),
            "source": "CrisisMMD/%s" % (row.get("event_name") or "?")}


# -- cleanup ------------------------------------------------------------------------------------------------------


def level_counts(recs: Iterable[Dict]) -> Counter:
    return Counter(int(r["label"]) for r in recs)


def balance_levels(recs: List[Dict], max_ratio: float = 3.0, rng: Optional[random.Random] = None,
                   floor: int = 200) -> List[Dict]:
    """Drop records so no level has more than ``max_ratio`` x the median level count (never below ``floor``).
    Keeps file order; the dropped ones are sampled with ``rng`` (else the last ones go)."""
    counts = level_counts(recs)
    if len(counts) < 2 or max_ratio <= 0:
        return list(recs)
    median = sorted(counts.values())[len(counts) // 2]
    cap = max(floor, int(round(max_ratio * median)))
    keep_idx = set()
    by_level: Dict[int, List[int]] = {}
    for i, r in enumerate(recs):
        by_level.setdefault(int(r["label"]), []).append(i)
    for level, idx in by_level.items():
        if len(idx) > cap:
            idx = rng.sample(idx, cap) if rng is not None else idx[:cap]
        keep_idx.update(idx)
    return [r for i, r in enumerate(recs) if i in keep_idx]


__all__ = ["SOURCES", "CRITERIA", "INSTRUCTIONS", "rubric_question", "clip_text", "vlfeedback_records", "ava_record",
           "collapse_counts", "richhf_records", "richhf_level", "crisismmd_record", "balance_levels", "level_counts"]
