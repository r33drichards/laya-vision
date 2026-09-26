"""The Cauldron (``HuggingFaceM4/the_cauldron``) as post-training data for the VLM decision model.

The Cauldron is the 50-subset instruction mixture both SmolVLM and ModernVBERT were aligned on. Each row is
``{"images": [...], "texts": [{"user", "assistant", "source"}, ...]}`` in free text. Laya answers *typed*
questions, so only the turns whose answer is a closed choice are usable. They come in four shapes:

* lettered choices, ``Question: ...\\nChoices:\\nA. ...\\nB. ...\\nAnswer with the letter.`` -> ``Answer: B``
  (ai2d, iconqa, intergps, scienceqa, tqa, visual7w) -> ``choice``; text before ``Question:`` (a ScienceQA
  lecture or hint) becomes the state's context.
* an options list, ``<question>\\n<instruction>\\nOptions: skateboarder, train, delivery, cab.`` -> ``Cab.`` or
  ``Answer: cab.\\nRationale: ...`` (aokvqa) -> ``choice``.
* yes / no, ``...\\nAnswer yes or no.`` -> ``Yes.`` (figureqa, hateful_memes, nlvr2, vsr, vqarad, and the yes/no
  share of clevr, dvqa, mapqa, ocrvqa, vqav2, chartqa, plotqa, textvqa) -> ``noul``.
* a bare letter with the candidates drawn in the image, ``Which figure should complete the logical sequence?``
  -> ``B`` (raven) -> ``choice`` over the letters A-H.

Everything else (numbers, free-form answers, captions, code) is skipped. ``cauldron_records`` turns one row's
turns into prepared-dataset records (the format ``laya.vlm_train.load_jsonl_examples`` reads); the Modal job
``prepare_cauldron`` in ``modal_app.py`` streams the subsets, saves the images and writes the jsonl files.
"""
import random
import re
from typing import Dict, List, Optional, Sequence

from .prompt import normalize_option_text

LETTERS = "ABCDEFGH"
RAVEN_OPTIONS = list(LETTERS)

#: subsets worth preparing: the closed-form ones, plus those with a large yes/no share
SUBSETS = (
    "ai2d", "aokvqa", "iconqa", "intergps", "scienceqa", "tqa", "visual7w", "raven",
    "figureqa", "hateful_memes", "nlvr2", "vsr", "vqarad",
    "clevr", "dvqa", "mapqa", "ocrvqa", "vqav2", "chartqa",
)

_YESNO_SUFFIX = re.compile(r"\s*Answer yes or no\.?\s*$")
_ANSWER_LETTER = re.compile(r"^Answer:\s*([A-Z])\.?$")
_CHOICE_LINE = re.compile(r"^([A-Z])\.\s?(.*)$")
_OPTIONS_LINE = re.compile(r"\nOptions:\s*(.+?)\s*$", re.S)
_RAVEN = re.compile(r"complete the (logical )?sequence", re.I)


def _split_context(head: str):
    """``"Lecture: ...\\nQuestion: What ..."`` -> ``("Lecture: ...", "What ...")``; no marker -> ``("", head)``."""
    head = head.strip()
    m = re.search(r"(?:^|\n)Question:\s*", head)
    if m is None:
        return "", head
    return head[: m.start()].strip(), head[m.end():].strip()


def parse_lettered(user: str, assistant: str) -> Optional[Dict]:
    """``Question: ...\\nChoices:\\nA. x\\nB. y\\nAnswer with the letter.`` -> ``Answer: B``."""
    if "\nChoices:\n" not in user:
        return None
    m = _ANSWER_LETTER.match(assistant.strip())
    if m is None:
        return None
    head, _, rest = user.partition("\nChoices:\n")
    choices: List[str] = []
    for line in rest.split("\n"):
        if not line.strip() or line.strip().startswith("Answer with the letter"):
            continue
        cm = _CHOICE_LINE.match(line)
        if cm and len(choices) < len(LETTERS) and cm.group(1) == LETTERS[len(choices)]:
            choices.append(cm.group(2).strip())
        elif choices:
            choices[-1] = (choices[-1] + " " + line.strip()).strip()  # a choice that wrapped onto a second line
        else:
            return None
    label = LETTERS.find(m.group(1))
    if len(choices) < 2 or not 0 <= label < len(choices) or len(set(choices)) != len(choices):
        return None
    context, question = _split_context(head)
    if not question:
        return None
    return {"question": {"type": "choice", "instructions": question, "criteria": choices}, "label": label,
            "state_text": context or None}


def parse_options(user: str, assistant: str) -> Optional[Dict]:
    """``<question>\\n<instruction>\\nOptions: a, b, c, d.`` -> ``B.`` / ``Answer: b.\\nRationale: ...`` (A-OKVQA)."""
    m = _OPTIONS_LINE.search(user)
    if m is None:
        return None
    opts = [normalize_option_text(o, "lower") for o in m.group(1).rstrip(".").split(",")]
    opts = [o for o in opts if o]
    answer = assistant.strip().split("\n")[0]
    answer = re.sub(r"^Answer:\s*", "", answer).strip().rstrip(".").lower()
    if len(opts) < 2 or len(set(opts)) != len(opts) or answer not in opts:
        return None
    question = user[: m.start()].strip().split("\n")[0].strip()
    if not question:
        return None
    return {"question": {"type": "choice", "instructions": question, "criteria": opts}, "label": opts.index(answer),
            "state_text": None}


def parse_raven(user: str, assistant: str, n_images: int) -> Optional[Dict]:
    """``Which figure should complete the logical sequence?`` -> ``B``: the candidates are panels in the image."""
    a = assistant.strip()
    if n_images != 1 or len(a) != 1 or a not in LETTERS or not _RAVEN.search(user):
        return None
    return {"question": {"type": "choice", "instructions": user.strip(), "criteria": list(RAVEN_OPTIONS)},
            "label": LETTERS.index(a), "state_text": None}


def parse_yesno(user: str, assistant: str) -> Optional[Dict]:
    """Any turn answered ``Yes.`` / ``No.``: the ``Answer yes or no.`` and brevity suffixes are dropped."""
    a = assistant.strip().rstrip(".").lower()
    if a not in ("yes", "no"):
        return None
    q = _YESNO_SUFFIX.sub("", user).strip()
    lines = [ln.strip() for ln in q.split("\n") if ln.strip()]
    # VQAv2-style turns carry a random brevity instruction on a last line ("Keep it brief.", "Quick response,
    # please."); the question itself is the line with the question mark
    if len(lines) > 1 and "?" not in lines[-1] and len(lines[-1].split()) <= 8:
        lines = lines[:-1]
    q = "\n".join(lines)
    if not q:
        return None
    return {"question": {"type": "noul", "instructions": q, "criteria": None}, "label": int(a == "yes"),
            "state_text": None}


def parse_turn(user: str, assistant: str, n_images: int = 1) -> Optional[Dict]:
    """One Cauldron turn -> ``{"question", "label", "state_text"}`` or ``None`` when its answer is open-ended."""
    return (parse_lettered(user, assistant) or parse_options(user, assistant)
            or parse_raven(user, assistant, n_images) or parse_yesno(user, assistant))


def cauldron_records(texts: Sequence[Dict], image_paths: Sequence[str], row_id: str, max_texts: int = 4,
                     rng: Optional[random.Random] = None) -> List[Dict]:
    """Prepared-dataset records for one row: at most ``max_texts`` usable turns (sampled with ``rng``, else the
    first ones), each pointing at the row's saved image(s) (``"image"`` for one, ``"images"`` for several)."""
    recs = []
    for k, t in enumerate(texts):
        parsed = parse_turn(t["user"], t["assistant"], len(image_paths))
        if parsed is None:
            continue
        rec = {"id": "%s-%d" % (row_id, k), "state_text": parsed["state_text"], "question": parsed["question"],
               "label": parsed["label"], "source": t.get("source")}
        if len(image_paths) == 1:
            rec["image"] = image_paths[0]
        else:
            rec["images"] = list(image_paths)
        recs.append(rec)
    if len(recs) > max_texts:
        recs = rng.sample(recs, max_texts) if rng is not None else recs[:max_texts]
    return recs


__all__ = ["SUBSETS", "cauldron_records", "parse_turn", "parse_lettered", "parse_options", "parse_raven", "parse_yesno"]
