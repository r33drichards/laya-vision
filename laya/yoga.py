"""The Kaggle yoga posture dataset as prepared ``choice`` records: which pose is the person doing?

Source: ``tr1gg3rtrash/yoga-posture-dataset`` on Kaggle (public, no API key needed for the download), one folder
of photos and drawings per pose, 47 poses and about 2.7k images, plus ``Poses.json`` with each pose's English
name. Each image becomes one ``choice`` whose options are the true pose and ``n_options - 1`` other poses drawn at
random, each written "English (Sanskrit)", e.g. "Reverse Warrior (Parsva Virabhadrasana)". Sampling the options
keeps the question as long as a Cauldron one; ``n_options=0`` asks over all 47.

Cleaning, done by ``prepare_yoga`` in ``modal_app.py`` with the helpers here:

* Byte-identical files are kept once. The same file filed under two poses (12 of them, mostly Cat / Cow and
  Crescent Lunge / Crescent Moon) is dropped: its label is a coin flip.
* Crescent Lunge (Alanasana) and Crescent Moon (Ashta Chandrasana) are the same high lunge in this dataset's
  photos, so they are never offered as options of the same question (``CONFUSABLE``).
* The split is by content hash (``evalsets.stable_split``), so a file never sits on both sides.

The helpers here have no network access and are unit tested.
"""
import hashlib
import random
from typing import Dict, List, Optional, Sequence

KAGGLE_DATASET = "tr1gg3rtrash/yoga-posture-dataset"
KAGGLE_URL = "https://www.kaggle.com/api/v1/datasets/download/" + KAGGLE_DATASET

#: folder name in the archive -> English name, as ``Poses.json`` gives it (one folder name is misspelt upstream)
POSES = {
    "Adho Mukha Svanasana": "Downward-Facing Dog",
    "Adho Mukha Vrksasana": "Handstand",
    "Alanasana": "Crescent Lunge",
    "Anjaneyasana": "Low Lunge",
    "Ardha Chandrasana": "Half-Moon",
    "Ardha Matsyendrasana": "Half Lord of the Fishes",
    "Ardha Navasana": "Half-Boat",
    "Ardha Pincha Mayurasana": "Dolphin",
    "Ashta Chandrasana": "Crescent Moon",
    "Baddha Konasana": "Butterfly",
    "Bakasana": "Crow",
    "Balasana": "Child's Pose",
    "Bitilasana": "Cow",
    "Camatkarasana": "Wild Thing",
    "Dhanurasana": "Bow",
    "Eka Pada Rajakapotasana": "King Pigeon",
    "Garudasana": "Eagle",
    "Halasana": "Plow",
    "Hanumanasana": "Splits",
    "Malasana": "Squat",
    "Marjaryasana": "Cat",
    "Navasana": "Boat",
    "Padmasana": "Lotus",
    "Parsva Virabhadrasana": "Reverse Warrior",
    "Parsvottanasana": "Pyramid",
    "Paschimottanasana": "Seated Forward Bend",
    "Phalakasana": "Plank",
    "Pincha Mayurasana": "Forearm Stand",
    "Salamba Bhujangasana": "Sphinx",
    "Salamba Sarvangasana": "Shoulder Stand",
    "Setu Bandha Sarvangasana": "Bridge",
    "Sivasana": "Corpse",
    "Supta Kapotasana": "Pigeon",
    "Trikonasana": "Triangle",
    "Upavistha Konasana": "Side Splits",
    "Urdhva Dhanurasana": "Wheel",
    "Urdhva Mukha Svsnssana": "Upward-Facing Dog",
    "Ustrasana": "Camel",
    "Utkatasana": "Chair",
    "Uttanasana": "Standing Forward Bend",
    "Utthita Hasta Padangusthasana": "Extended Hand to Toe",
    "Utthita Parsvakonasana": "Extended Side Angle",
    "Vasisthasana": "Side Plank",
    "Virabhadrasana One": "Warrior One",
    "Virabhadrasana Three": "Warrior Three",
    "Virabhadrasana Two": "Warrior Two",
    "Vrksasana": "Tree",
}
_SANSKRIT_FIX = {"Urdhva Mukha Svsnssana": "Urdhva Mukha Svanasana"}

#: poses never offered together: indistinguishable in this dataset's images
CONFUSABLE = ({"Alanasana", "Ashta Chandrasana"},)

INSTRUCTIONS = (
    "Which yoga pose is the person doing?",
    "What yoga pose is shown in the image?",
    "Name the yoga posture in this picture.",
    "Which asana is this?",
)


def pose_label(folder: str) -> str:
    """``"Parsva Virabhadrasana"`` -> ``"Reverse Warrior (Parsva Virabhadrasana)"``."""
    return "%s (%s)" % (POSES[folder], _SANSKRIT_FIX.get(folder, folder))


def _confusable(a: str, b: str) -> bool:
    return any(a in s and b in s for s in CONFUSABLE)


def pose_options(folder: str, n_options: int, rng: random.Random) -> List[str]:
    """The true pose plus ``n_options - 1`` random others (none confusable with it), shuffled; 0: every pose
    except the ones confusable with it."""
    others = [p for p in POSES if p != folder and not _confusable(p, folder)]
    if n_options:
        others = rng.sample(others, min(n_options - 1, len(others)))
    opts = [folder] + others
    rng.shuffle(opts)
    return opts


def yoga_record(folder: str, key: str, rng: random.Random, n_options: int = 12) -> Optional[Dict]:
    """One image of ``folder``'s pose -> a ``choice`` record (``image`` is filled in by the caller)."""
    if folder not in POSES:
        return None
    opts = pose_options(folder, n_options, rng)
    return {"id": "yoga-" + key, "state_text": None,
            "question": {"type": "choice", "instructions": rng.choice(INSTRUCTIONS),
                         "criteria": [pose_label(p) for p in opts]},
            "label": opts.index(folder), "source": "Kaggle yoga/" + folder}


def unique_files(files: Sequence[tuple]) -> Dict[str, Dict]:
    """``[(folder, filename, bytes), ...]`` -> ``{sha1: {"folder", "name", "data"}}``: one entry per distinct file,
    in first-seen order; a file found under two or more poses is left out."""
    seen: Dict[str, Dict] = {}
    conflicted = set()
    for folder, name, data in files:
        h = hashlib.sha1(data).hexdigest()
        if h in seen and seen[h]["folder"] != folder:
            conflicted.add(h)
        seen.setdefault(h, {"folder": folder, "name": name, "data": data})
    return {h: v for h, v in seen.items() if h not in conflicted}


__all__ = ["KAGGLE_DATASET", "KAGGLE_URL", "POSES", "CONFUSABLE", "INSTRUCTIONS", "pose_label", "pose_options",
           "yoga_record", "unique_files"]
