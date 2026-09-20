"""Convert Open CaptchaWorld puzzles into typed Laya decisions.

Open CaptchaWorld (Luo et al., 2025, arXiv:2505.24878) scores browser agents that click, drag and type
their way through a CAPTCHA. This model does none of those things: it answers typed questions
(``choice`` / ``score`` / ``noul``) about images in one forward pass. So the benchmark is re-expressed
here as decisions, and graded offline against the bundled ``ground_truth.json`` files rather than by
driving the Flask app.

Only the types that are genuinely discrete decisions are covered -- 13 of 20. The six coordinate-click
types (Geometry_Click, Pick_Area, Place_Dot, Click_Order, Slide_Puzzle, Misleading_Click) need
pixel-precise output this architecture has no head for, and Hold_Button has no perception content at all.

Every decision is kept small and self-contained, which is the whole trick:

- **k-way** (reference image + option images): one ``noul`` per option -- "does this option match the
  reference?" -- over exactly two images, then argmax over the k yes-probabilities. Asking instead for a
  single ``choice`` over options named "option 1".."option k" would measure whether a 256M model can bind
  an option label to an image position, not whether it can see; pairwise comparison avoids that entirely.
- **grid select** (one image cut into r x c cells): one ``noul`` per cell over that cell's crop alone.
  Single-answer grids grade by argmax, multi-answer grids by thresholding at 0.5 and comparing sets, which
  is what ``app.py`` does to a human's clicks.
- **count**: one ``choice`` over a deterministic ladder of candidate sums.

Grading follows the benchmark: pass@1 per puzzle, exact match. Each decision also yields a
(confidence, correct) pair so per-decision accuracy and ECE come out of the same run.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
import json
import os
import random

# ---------------------------------------------------------------------------------------------------------
# Type specs
# ---------------------------------------------------------------------------------------------------------

#: reference + option images -> argmax over per-option ``noul``. Values are the ground-truth key names.
KWAY_SPECS = {
    "Connect_icon":   {"ref": "reference_image", "opts": "options",       "correct": "correct_option"},
    "Coordinates":    {"ref": "reference_image", "opts": "option_images", "correct": "correct_option_index"},
    "Dart_Count":     {"ref": "reference_image", "opts": "option_images", "correct": "correct_option_index"},
    "Image_Matching": {"ref": "reference_image", "opts": "option_images", "correct": "correct_option_index"},
    "Object_Match":   {"ref": "reference_image", "opts": "option_images", "correct": "correct_option_index"},
    "Path_Finder":    {"ref": "reference_image", "opts": "options",       "correct": "correct_option"},
}

#: one image cut row-major into ``grid_size = [rows, cols]`` cells (matches ``static/js/script.js``).
GRID_SPECS = {
    "Select_Animal":     {"answer": "correct_patches",   "grid": [2, 3], "multi": False},
    "Unusual_Detection": {"answer": "answer",            "grid": [2, 3], "multi": True},
    "Patch_Select":      {"answer": "correct_patches",   "grid": [5, 5], "multi": True},
}

ROTATION_ANGLES = (0, 45, 90, 135, 180, 225, 270, 315)

#: types this module deliberately does not cover, and why.
EXCLUDED = {
    "Geometry_Click":  "coordinate click: needs a pixel output head",
    "Pick_Area":       "coordinate click: needs a pixel output head",
    "Place_Dot":       "coordinate click: needs a pixel output head",
    "Click_Order":     "coordinate click sequence: needs a pixel output head",
    "Slide_Puzzle":    "drag to a 10px tolerance: needs a pixel output head",
    "Misleading_Click": "coordinate click: needs a pixel output head",
    "Hold_Button":     "no perception content (answer is always 'completed')",
}

MAX_EDGE = 512  # pre-shrink before the processor; the backbone works far below this anyway


# ---------------------------------------------------------------------------------------------------------
# Puzzle representation
# ---------------------------------------------------------------------------------------------------------


@dataclass
class Decision:
    """One typed question, with the images it is asked about and what the right answer would be."""

    qid: str
    images: List[str]           # file paths, or ("path", box) crops resolved at load time
    question: Dict[str, Any]
    truth: Any                  # bool for noul, option key for choice
    crops: List[Optional[Tuple[int, int, int, int]]] = field(default_factory=list)

    def state(self) -> Dict[str, Any]:
        return {"images": [_open(p, c) for p, c in zip(self.images, self.crops or [None] * len(self.images))]}


@dataclass
class Puzzle:
    """One benchmark puzzle, as a list of decisions plus how to grade them."""

    ctype: str
    pid: str
    prompt: str
    mode: str                   # "argmax" | "subset" | "choice"
    decisions: List[Decision]
    truth: Any                  # int index, set of ints, or option key
    n_options: int

    def grade(self, answers: Dict[str, Dict]) -> Tuple[bool, List[Dict[str, Any]]]:
        """Return (passed, per-decision records).

        A record carries the confidence and whether the decision was right, plus -- for ``noul`` -- the
        yes-probability and the true label, so a caller can tell a calibrated model from one that simply
        answers "yes" to everything.
        """
        recs = []
        for d in self.decisions:
            a = answers[d.qid]
            if a["type"] == "noul":
                recs.append({"conf": float(a["confidence"]), "correct": (a["noul"] >= 0.5) == bool(d.truth),
                             "prob": float(a["noul"]), "truth": bool(d.truth)})
            else:
                recs.append({"conf": float(a["confidence"]), "correct": a["choice"] == d.truth,
                             "prob": None, "truth": None})

        if self.mode == "choice":
            return answers[self.decisions[0].qid]["choice"] == self.truth, recs
        probs = [float(answers[d.qid]["noul"]) for d in self.decisions]
        if self.mode == "argmax":
            pick = max(range(len(probs)), key=probs.__getitem__)
            return (pick in self.truth if isinstance(self.truth, (set, frozenset)) else pick == self.truth), recs
        picked = {i for i, p in enumerate(probs) if p >= 0.5}
        return picked == set(self.truth), recs


def _open(path: str, box: Optional[Tuple[int, int, int, int]] = None):
    from PIL import Image

    im = Image.open(path).convert("RGB")
    if box is not None:
        im = im.crop(box)
    if max(im.size) > MAX_EDGE:
        im.thumbnail((MAX_EDGE, MAX_EDGE), Image.LANCZOS)
    return im


def _cells(path: str, rows: int, cols: int) -> List[Tuple[int, int, int, int]]:
    """Row-major crop boxes, matching how the benchmark's frontend indexes grid cells."""
    from PIL import Image

    with Image.open(path) as im:
        w, h = im.size
    cw, ch = w / cols, h / rows
    return [
        (int(round((i % cols) * cw)), int(round((i // cols) * ch)),
         int(round((i % cols + 1) * cw)), int(round((i // cols + 1) * ch)))
        for i in range(rows * cols)
    ]


# ---------------------------------------------------------------------------------------------------------
# Question builders
# ---------------------------------------------------------------------------------------------------------


def match_question(prompt: str) -> Dict:
    """`noul` over [reference, candidate]: is the candidate the one the puzzle asks for?"""
    return {
        "type": "noul",
        "instructions": "The first image is the reference and the second image is a candidate answer. "
                        "Task: %s Is the second image the correct answer to that task?" % prompt.rstrip(),
        "criteria": {"true": "yes, the second image is the correct answer",
                     "false": "no, the second image is not the correct answer"},
    }


def cell_question(prompt: str, target: str) -> Dict:
    """`noul` over one cropped grid cell: does this cell show the thing being asked for?"""
    return {
        "type": "noul",
        "instructions": "This image is one tile of a CAPTCHA grid. Task: %s Does this tile show %s?"
                        % (prompt.rstrip(), target),
        "criteria": {"true": "yes, this tile shows it", "false": "no, this tile does not show it"},
    }


def count_question(prompt: str, candidates: Sequence[int]) -> Dict:
    return {
        "type": "choice",
        "instructions": prompt.rstrip(),
        "criteria": {str(c): "the total is %d" % c for c in candidates},
    }


# ---------------------------------------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------------------------------------


def _gt(root: str, ctype: str) -> Dict:
    with open(os.path.join(root, ctype, "ground_truth.json")) as fh:
        return json.load(fh)


def _kway(root: str, ctype: str) -> List[Puzzle]:
    spec, d, out = KWAY_SPECS[ctype], os.path.join(root, ctype), []
    for pid, v in _gt(root, ctype).items():
        ref, opts = v[spec["ref"]], list(v[spec["opts"]])
        prompt = v.get("prompt", "Pick the matching image.")
        if ctype == "Dart_Count":  # the reference is a number rendered as an image; say it in words too
            prompt = "%s The target number is %s." % (prompt, v.get("reference_number"))
        decs = [
            Decision(qid="opt%d" % i, images=[os.path.join(d, ref), os.path.join(d, o)],
                     question=match_question(prompt), truth=(i == v[spec["correct"]]), crops=[None, None])
            for i, o in enumerate(opts)
        ]
        out.append(Puzzle(ctype, pid, prompt, "argmax", decs, v[spec["correct"]], len(opts)))
    return out


def _rotation(root: str) -> List[Puzzle]:
    d, out = os.path.join(root, "Rotation_Match"), []
    for pid, v in _gt(root, "Rotation_Match").items():
        base = v["object_base_image"].rsplit(".", 1)[0]
        prompt = v.get("prompt", "Rotate the object to face the reference direction.")
        decs = [
            Decision(qid="ang%d" % a, images=[os.path.join(d, v["reference_image"]), os.path.join(d, "%s_%d.png" % (base, a))],
                     question=match_question(prompt), truth=(a == v["correct_angle"]), crops=[None, None])
            for a in ROTATION_ANGLES
        ]
        out.append(Puzzle("Rotation_Match", pid, prompt, "argmax", decs,
                          ROTATION_ANGLES.index(v["correct_angle"]), len(ROTATION_ANGLES)))
    return out


def _grid(root: str, ctype: str) -> List[Puzzle]:
    spec, d, out = GRID_SPECS[ctype], os.path.join(root, ctype), []
    for pid, v in _gt(root, ctype).items():
        rows, cols = v.get("grid_size", spec["grid"])
        path = os.path.join(d, pid)
        prompt = v.get("prompt", "Select the matching tiles.")
        target = v.get("target_object") or "what the task asks for"
        boxes = _cells(path, rows, cols)
        truth = set(v[spec["answer"]])
        decs = [
            Decision(qid="cell%d" % i, images=[path], question=cell_question(prompt, target),
                     truth=(i in truth), crops=[box])
            for i, box in enumerate(boxes)
        ]
        mode = "subset" if spec["multi"] else "argmax"
        out.append(Puzzle(ctype, pid, prompt, mode, decs,
                          truth if spec["multi"] else next(iter(truth)), rows * cols))
    return out


def _image_recognition(root: str) -> List[Puzzle]:
    d, out = os.path.join(root, "Image_Recognition"), []
    for pid, v in _gt(root, "Image_Recognition").items():
        prompt, sub = v.get("question", v["prompt"]), v["subfolder"]
        truth = set(v["correct_selections"])
        decs = [
            Decision(qid="img%d" % i, images=[os.path.join(d, sub, im)],
                     question=cell_question(prompt, "what the task asks for"), truth=(i in truth), crops=[None])
            for i, im in enumerate(v["images"])
        ]
        out.append(Puzzle("Image_Recognition", pid, prompt, "subset", decs, truth, len(v["images"])))
    return out


def _bingo(root: str) -> List[Puzzle]:
    """3x3 tile swap. Each decision is one candidate (i, j) swap, scored over the whole grid image."""
    d, out = os.path.join(root, "Bingo"), []
    for pid, v in _gt(root, "Bingo").items():
        rows, cols = v.get("grid_size", [3, 3])
        n = rows * cols
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        truth = {tuple(sorted(a)) for a in v["answer"]}
        prompt = v.get("prompt", "Swap two tiles to line up three matching images.")
        decs = []
        for i, j in pairs:
            q = {
                "type": "noul",
                "instructions": "%s Would swapping the tile at row %d column %d with the tile at row %d column %d "
                                "line up three matching tiles?"
                                % (prompt.rstrip(), i // cols + 1, i % cols + 1, j // cols + 1, j % cols + 1),
                "criteria": {"true": "yes, that swap completes a line", "false": "no, that swap does not"},
            }
            decs.append(Decision(qid="swap%d_%d" % (i, j), images=[os.path.join(d, pid)],
                                 question=q, truth=((i, j) in truth), crops=[None]))
        out.append(Puzzle("Bingo", pid, prompt, "argmax", decs,
                          {pairs.index(p) for p in truth if p in pairs}, len(pairs)))
    return out


def _dice(root: str, n_options: int = 10, seed: int = 0) -> List[Puzzle]:
    """Counting has no natural option list, so build a deterministic ladder around the true sum.

    Distractors are drawn from the range the benchmark's own sums span, spaced so the answer is not
    guessable from magnitude alone. Chance is 1/n_options; the ladder is seeded per puzzle so the
    numbers are identical on every run.
    """
    d, gt, out = os.path.join(root, "Dice_Count"), _gt(root, "Dice_Count"), []
    lo = min(v["sum"] for v in gt.values())
    hi = max(v["sum"] for v in gt.values())
    for pid, v in gt.items():
        rng = random.Random("%s/%d" % (pid, seed))
        truth = v["sum"]
        cands = {truth}
        while len(cands) < n_options:
            cands.add(rng.randint(max(1, lo - 5), hi + 5))
        cands = sorted(cands)
        prompt = v.get("prompt", "Sum up the numbers on all the dice")
        decs = [Decision(qid="sum", images=[os.path.join(d, pid)],
                         question=count_question(prompt, cands), truth=str(truth), crops=[None])]
        out.append(Puzzle("Dice_Count", pid, prompt, "choice", decs, str(truth), len(cands)))
    return out


LOADERS = {
    **{t: (lambda r, t=t: _kway(r, t)) for t in KWAY_SPECS},
    **{t: (lambda r, t=t: _grid(r, t)) for t in GRID_SPECS},
    "Rotation_Match": _rotation,
    "Image_Recognition": _image_recognition,
    "Bingo": _bingo,
    "Dice_Count": _dice,
}


def load_puzzles(root: str, types: Optional[Sequence[str]] = None) -> List[Puzzle]:
    """Load every supported puzzle from a checkout of ``OpenCaptchaWorld/captcha_data``.

    ``root`` must be the ``captcha_data`` directory with its git-lfs objects resolved.
    """
    names = list(types) if types else sorted(LOADERS)
    unknown = [t for t in names if t not in LOADERS]
    if unknown:
        raise ValueError("unsupported captcha types %r (known: %s)" % (unknown, ", ".join(sorted(LOADERS))))
    out = []
    for t in names:
        if not os.path.isdir(os.path.join(root, t)):
            raise FileNotFoundError("missing captcha type directory: %s" % os.path.join(root, t))
        out.extend(LOADERS[t](root))
    return out
