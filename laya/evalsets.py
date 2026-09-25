"""Held-out evaluation sets for ``choice``, ``noul`` and ``score`` questions, as prepared-dataset records.

The training mixture (the Cauldron, the rubric sets) is scored on its own val splits. These sets measure
what those cannot: calibration against real human disagreement, abstention, hallucination, and rubric
scoring on data the model has never been tuned toward. Four of the six carry per-rater votes, which become a
soft ``target`` so the model can be scored against the spread of human answers, not just the argmax:

* **KonIQ-10k** (``chaofengc/IQA-PyTorch-Datasets``, ``koniq10k.tgz``): 10k in-the-wild photos, each rated by
  about 100 people on a 5-point quality scale. ``score`` over ``rubric.CRITERIA["quality"]``; the vote
  fractions ``c1..c5`` are the target. The official training / validation / test sets are kept.
* **EvalMuse-40K** (``DY-Evalab/EvalMuse``): 32.7k generated images with 3 or 6 raters each scoring how well
  the image matches its prompt, 1-5. ``score`` over ``rubric.CRITERIA["alignment"]`` (the RichHF alignment
  rubric) with the prompt as state text; the raters' histogram is the target. The upstream test labels are
  not released, so val is a seeded share of *prompts* (every image of a prompt stays on one side).
* **CIFAR-10H** (``MKZuziak/cifar10h``): the 10k CIFAR-10 test images with about 50 human guesses each.
  ``choice`` over the ten classes; the target is the human histogram, the label the CIFAR-10 class. Eval only.
* **FER+** (``microsoft/FERPlus`` votes on ``AutumnQiu/fer2013`` images): 48x48 faces, 10 taggers each over
  8 emotions. ``choice`` with the vote histogram as the target; faces the taggers mostly called unknown or
  not-a-face are dropped, as FER+ does. FER2013's Training / PublicTest / PrivateTest become train / val / test.
  The votes are joined to the images by row order, so ``ferplus_agreement`` checks the join before writing.
* **VizWiz** (``lmms-lab-encoder/VizWiz-VQA``, ``val``): photos taken by blind users with their question.
  ``noul`` "can the question be answered from this photo?"; the target is the share of the 10 crowd answers
  that are not "unanswerable". Eval only: a direct test of abstention.
* **POPE** (``lmms-lab-encoder/POPE``, ``Full``): "Is there a <object> in the image?" over 500 COCO images, in
  its three negative-sampling settings, each its own dataset (``pope_random``, ``pope_popular``,
  ``pope_adversarial``) since POPE is reported per setting. ``noul``, hard labels, eval only.

The Modal job ``prepare_eval`` in ``modal_app.py`` downloads the sources, saves the images and writes
``/data/vqa/eval_<name>/{train,val,test}.jsonl``; the helpers here have no network access and are unit tested.
"""
import hashlib
import random
import zipfile
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from laya.rubric import clip_text, rubric_question

#: prepared dataset names (``eval_<name>``) and their sources
SOURCES = {
    "koniq": "chaofengc/IQA-PyTorch-Datasets",
    "evalmuse": "DY-Evalab/EvalMuse",
    "cifar10h": "MKZuziak/cifar10h",
    "ferplus": "AutumnQiu/fer2013",
    "vizwiz": "lmms-lab-encoder/VizWiz-VQA",
    "pope_random": "lmms-lab-encoder/POPE",
    "pope_popular": "lmms-lab-encoder/POPE",
    "pope_adversarial": "lmms-lab-encoder/POPE",
}

FERPLUS_VOTES_URL = "https://raw.githubusercontent.com/microsoft/FERPlus/master/fer2013new.csv"

# -- shared helpers -----------------------------------------------------------------------------------------------


def histogram(values: Iterable[int], levels: int, lo: int = 1) -> Optional[List[float]]:
    """Ratings ``lo..lo+levels-1`` -> the fraction at each level; ``None`` when no rating is in range."""
    counts = [0] * levels
    for v in values:
        if lo <= int(v) < lo + levels:
            counts[int(v) - lo] += 1
    total = sum(counts)
    return [round(c / total, 4) for c in counts] if total else None


def mode_level(target: Sequence[float]) -> int:
    """The most-voted level; ties go to the lower level (as ``rubric.ava_record``)."""
    return max(range(len(target)), key=lambda i: (target[i], -i))


def stable_split(key: str, val_pct: float, seed: int = 0) -> str:
    """``"val"`` for a fixed ``val_pct`` percent of keys, else ``"train"``: a hash, so the split does not depend on
    stream order and every record sharing a key (all images of one prompt) lands on the same side."""
    h = int(hashlib.sha1(("%d:%s" % (seed, key)).encode()).hexdigest()[:8], 16)
    return "val" if h % 10000 < val_pct * 100 else "train"


# -- KonIQ-10k ----------------------------------------------------------------------------------------------------

KONIQ_SETS = {"training": "train", "validation": "val", "test": "test"}


def koniq_record(row: Dict, rng: Optional[random.Random] = None) -> Optional[Tuple[str, Dict]]:
    """A row of ``koniq10k_distributions_sets.csv`` -> ``(split, record)``; the vote fractions are the target."""
    split = KONIQ_SETS.get(str(row.get("set", "")).strip())
    try:
        votes = [float(row["c%d" % i]) for i in range(1, 6)]
    except (KeyError, TypeError, ValueError):
        return None
    if split is None or sum(votes) <= 0:
        return None
    target = [round(v / sum(votes), 4) for v in votes]
    name = str(row["image_name"])
    return split, {"id": "koniq-" + name.rsplit(".", 1)[0], "state_text": None, "question": rubric_question("quality", rng),
                   "label": mode_level(target), "target": target, "source": "KonIQ-10k"}


# -- EvalMuse-40K -------------------------------------------------------------------------------------------------


def evalmuse_record(row: Dict, rng: Optional[random.Random] = None, max_prompt_chars: int = 400) -> Optional[Dict]:
    """A ``train_list.json`` entry -> an alignment ``score`` record with the raters' 1-5 histogram as target.
    Prompts most raters marked meaningless (no subject to check the image against) are dropped."""
    prompt = clip_text(str(row.get("prompt") or ""), max_prompt_chars)
    meaningless = row.get("promt_meaningless") or []
    if not prompt or (meaningless and sum(meaningless) * 2 > len(meaningless)):
        return None
    target = histogram(row.get("total_score") or [], 5)
    if target is None:
        return None
    rid = "evalmuse-%s" % str(row["img_path"]).rsplit(".", 1)[0].replace("/", "-")
    return {"id": rid, "state_text": "Prompt: " + prompt, "question": rubric_question("alignment", rng),
            "label": mode_level(target), "target": target, "source": "EvalMuse/%s" % str(row["img_path"]).split("/")[0]}


class MultiPartZip:
    """Read single members of a zip archive split into byte-range parts (``images.zip.part-aa`` ...) without
    downloading it: ``fetch(part_index, start, end)`` returns bytes ``[start, end)`` of that part, e.g. an HTTP
    range request. ``open()`` gives a ``zipfile.ZipFile``; open one per thread, since a ZipFile is not
    thread-safe."""

    def __init__(self, sizes: Sequence[int], fetch: Callable[[int, int, int], bytes]):
        self.sizes, self.fetch = list(sizes), fetch

    def open(self) -> zipfile.ZipFile:
        return zipfile.ZipFile(_PartsFile(self.sizes, self.fetch))


class _PartsFile:
    """A seekable read-only file over the concatenated parts."""

    def __init__(self, sizes, fetch):
        self.sizes, self.fetch, self.pos, self.size = sizes, fetch, 0, sum(sizes)

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = max(0, min(n, self.size - self.pos))
        out, base = [], 0
        for i, size in enumerate(self.sizes):
            if n > 0 and self.pos < base + size:
                start = self.pos - base
                end = min(size, start + n)
                out.append(self.fetch(i, start, end))
                self.pos += end - start
                n -= end - start
            base += size
        return b"".join(out)

    def close(self):
        pass


# -- CIFAR-10H ----------------------------------------------------------------------------------------------------

CIFAR10_CLASSES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
CIFAR10_INSTRUCTIONS = (
    "What is the main object in this picture?",
    "Which of these is shown in the image?",
    "Classify the image.",
)


def cifar10h_record(row: Dict, index: int, rng: Optional[random.Random] = None) -> Optional[Dict]:
    """A ``MKZuziak/cifar10h`` row -> a 10-way ``choice`` with the human guesses as target."""
    counts = list(row.get("expert_counts") or [])
    label = row.get("label")
    if len(counts) != 10 or sum(counts) <= 0 or label is None:
        return None
    ins = rng.choice(CIFAR10_INSTRUCTIONS) if rng is not None else CIFAR10_INSTRUCTIONS[0]
    return {"id": "cifar10h-%05d" % index, "state_text": None,
            "question": {"type": "choice", "instructions": ins, "criteria": list(CIFAR10_CLASSES)},
            "label": int(label), "target": [round(c / sum(counts), 4) for c in counts], "source": "CIFAR-10H"}


# -- FER+ ---------------------------------------------------------------------------------------------------------

FERPLUS_EMOTIONS = ("neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear", "contempt")
FERPLUS_CRITERIA = ("neutral", "happy", "surprised", "sad", "angry", "disgusted", "afraid", "contemptuous")
FERPLUS_USAGE = {"Training": "train", "PublicTest": "val", "PrivateTest": "test"}
FERPLUS_INSTRUCTIONS = (
    "What emotion does this face show?",
    "Which expression is on the person's face?",
    "How does this person look like they feel?",
)
#: FER2013's labels in its own order, and the FER+ emotion each corresponds to (FER2013 has no contempt)
FER2013_LABELS = ("anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral")


def ferplus_record(votes: Dict, index: int, rng: Optional[random.Random] = None) -> Optional[Tuple[str, Dict]]:
    """A ``fer2013new.csv`` row -> ``(split, record)``, or ``None`` for a face FER+ drops: no image name, or at
    least half the votes are unknown / not a face. The 8 emotion counts are the target."""
    split = FERPLUS_USAGE.get(str(votes.get("Usage", "")).strip())
    if split is None or not str(votes.get("Image name") or "").strip():
        return None
    try:
        counts = [int(votes[e]) for e in FERPLUS_EMOTIONS]
        other = int(votes.get("unknown") or 0) + int(votes.get("NF") or 0)
    except (KeyError, TypeError, ValueError):
        return None
    if sum(counts) == 0 or other * 2 >= sum(counts) + other:
        return None
    target = [round(c / sum(counts), 4) for c in counts]
    ins = rng.choice(FERPLUS_INSTRUCTIONS) if rng is not None else FERPLUS_INSTRUCTIONS[0]
    return split, {"id": "ferplus-%05d" % index, "state_text": None,
                   "question": {"type": "choice", "instructions": ins, "criteria": list(FERPLUS_CRITERIA)},
                   "label": mode_level(target), "target": target, "source": "FER+"}


def ferplus_agreement(pairs: Iterable[Tuple[Dict, int]]) -> float:
    """Share of faces whose FER+ majority emotion equals the FER2013 label of the image it was joined to.

    FER+ relabelled FER2013, so the two agree on roughly 60-70% of faces when the rows line up and around 17%
    when they are off by one (checked on the Hub copy), which is what makes this a check of the join."""
    agree = n = 0
    for votes, fer_label in pairs:
        counts = {e: int(votes.get(e) or 0) for e in FER2013_LABELS}
        if sum(counts.values()) == 0:
            continue
        n += 1
        agree += max(counts, key=counts.get) == FER2013_LABELS[int(fer_label)]
    return agree / n if n else 0.0


# -- VizWiz -------------------------------------------------------------------------------------------------------

VIZWIZ_INSTRUCTIONS = (
    "Can the question be answered from this photo?",
    "Does the photo show enough to answer the question?",
    "Is the question answerable from what is visible in the picture?",
)


def vizwiz_record(row: Dict, rng: Optional[random.Random] = None, max_chars: int = 400) -> Optional[Dict]:
    """A VizWiz VQA row -> ``noul`` answerability; the target is the share of the crowd answers that are not
    "unanswerable", the label VizWiz's own answer type (``unanswerable`` or not)."""
    answers = [str(a).strip().lower() for a in (row.get("answers") or [])]
    question = clip_text(str(row.get("question") or ""), max_chars)
    if not answers or not question:
        return None
    p_yes = sum(a != "unanswerable" for a in answers) / len(answers)
    ins = rng.choice(VIZWIZ_INSTRUCTIONS) if rng is not None else VIZWIZ_INSTRUCTIONS[0]
    return {"id": "vizwiz-%s" % row["question_id"], "state_text": "Question: " + question,
            "question": {"type": "noul", "instructions": ins, "criteria": None},
            "label": int(row.get("category") != "unanswerable"), "target": [round(1 - p_yes, 4), round(p_yes, 4)],
            "source": "VizWiz/%s" % (row.get("category") or "?")}


# -- POPE ---------------------------------------------------------------------------------------------------------


def pope_record(row: Dict) -> Optional[Dict]:
    """A POPE row -> ``noul`` on its own question ("Is there a <object> in the image?")."""
    answer = str(row.get("answer") or "").strip().lower()
    question = str(row.get("question") or "").strip()
    if answer not in ("yes", "no") or not question:
        return None
    return {"id": "pope-%s-%s" % (row.get("category"), row["question_id"]), "state_text": None,
            "question": {"type": "noul", "instructions": question, "criteria": None},
            "label": int(answer == "yes"), "source": "POPE/%s" % row.get("category")}


# -- RF100-VL -----------------------------------------------------------------------------------------------------
#
# RF100-VL (Robicheaux et al. 2025, arXiv 2505.20612; https://github.com/roboflow/rf100-vl) is 100 object-detection
# datasets in 7 domains, scored by COCO box mAP. Laya Vision answers typed questions and draws no boxes, so its
# mAP is undefined; what it can be asked is the image-level half of detection: for every class of the image's
# dataset, is at least one instance of it in the image? The label is 1 when a ground-truth box of that class is
# there. ``benchmarks/rf100vl_presence.py`` scores the rows per dataset (presence AP, AUROC, balanced accuracy).

RF100VL_REPO = "probicheaux/rf100-vl"  # the paper's first author's parquet mirror; ids already 0-based per dataset
RF100VL_REVISION = "6b59bae252b2e68e5bf2fe1b9e1962df167f06f9"
RF100VL_CODE_COMMIT = "451c6ddbf0cd94528c36526d4e1bf1897ab5af38"  # roboflow/rf100-vl, for the domain map
RF100VL_DOMAINS_URL = ("https://raw.githubusercontent.com/roboflow/rf100-vl/%s/rf100vl/assets/"
                       "dataset_name_to_category.json" % RF100VL_CODE_COMMIT)
#: RF100-VL domain -> the suffix of its prepared dataset (``rf100vl_<suffix>``)
RF100VL_DOMAINS = {"Flora/Fauna": "flora_fauna", "Industrial": "industrial", "Misc": "misc",
                   "Lab Imaging": "lab_imaging", "Aerial": "aerial", "Document": "document", "Sport": "sport"}
RF100VL_INSTRUCTIONS = 'Is there at least one "%s" in the image?'


def rf100vl_records(row: Dict, class_names: Sequence[str], dataset: str, max_chars: int = 400) -> List[Dict]:
    """One RF100-VL image row (``image_id``, ``annotations.category_id``) -> a ``noul`` presence question per class
    of its dataset, in class-id order. The state text names the dataset and its label set, since many class names
    ("DIP", "0", "Bamboo 1") mean nothing on their own; the paper's zero-shot setting also gives the class names."""
    anns = row.get("annotations") or {}
    cats = anns.get("category_id") if isinstance(anns, dict) else [a.get("category_id") for a in anns]
    present = {int(c) for c in (cats or []) if c is not None}
    names = [str(n).strip() for n in class_names]
    if not names or not row.get("image_id") or not present <= set(range(len(names))):
        return []
    state = clip_text("An image from the %s dataset, labelled for: %s." % (dataset, ", ".join(names)), max_chars)
    return [{"id": "rf100vl-%s-%s-c%d" % (dataset, row["image_id"], k), "state_text": state,
             "question": {"type": "noul", "instructions": RF100VL_INSTRUCTIONS % name, "criteria": None},
             "label": int(k in present), "source": "RF100-VL/%s" % dataset}
            for k, name in enumerate(names)]


__all__ = ["SOURCES", "FERPLUS_VOTES_URL", "histogram", "mode_level", "stable_split", "koniq_record", "evalmuse_record",
           "MultiPartZip", "CIFAR10_CLASSES", "cifar10h_record", "FERPLUS_EMOTIONS", "FERPLUS_CRITERIA",
           "ferplus_record", "ferplus_agreement", "vizwiz_record", "pope_record", "RF100VL_REPO",
           "RF100VL_REVISION", "RF100VL_CODE_COMMIT", "RF100VL_DOMAINS_URL", "RF100VL_DOMAINS", "RF100VL_INSTRUCTIONS",
           "rf100vl_records"]
