"""Behaviour-cloning and probe records from BiGym demo-follower runs (``laya.bigymdemos.follow``).

A kept follower run is a seeded episode plus the primitive chosen at every decision; replaying it with the head
camera (``modal_bigym.py::bigym_bc_episode``) gives one frame per decision. This module turns such a run into
the jsonl records of three datasets (pure Python, no BiGym or Modal needed):

- ``bc_f1``: the control question with one frame (``bigym_question(task, 1)``), ``"image"`` = the current frame;
- ``bc_f4``: the control question with four frames (``bigym_question(task, 4)``), ``"images"`` = the last four
  decision frames oldest first, the first repeated at the start of the episode (``BiGymGame.frames(4)``);
- ``probe``: the perception questions (``probe_questions(task)``) on every ``PROBE_EVERY``-th decision frame and on
  the final (successful) frame, labelled from the simulator's ground truth (``bigymgames.labels``).

The control label is the chosen primitive's index in ``PRIMITIVES`` order, which is its option's position in the
question's criteria. Control ids are ``<task>-<seed>-<decision>`` (``vlm_train.episode_step`` reads episode
``<task>-<seed>``); probe ids are ``<task>-<seed>-<question>-<decision>`` so each (episode, step) key is unique.
Train and val are split by demo: ``val_seeds`` holds out about ``VAL_FRAC`` of each task's kept demos.
"""
import hashlib
from typing import Dict, Iterable, List, Sequence

from .bigymgames import PRIMITIVES, bigym_question, probe_questions

PROBE_EVERY = 3  # probe records on every third decision frame (plus the final frame)
VAL_FRAC = 0.1
_ORDER = {p: i for i, p in enumerate(PRIMITIVES)}


def label_index(primitive: str) -> int:
    """The control label: ``primitive``'s index in ``PRIMITIVES`` order (= its option in ``bigym_question``)."""
    if primitive not in _ORDER:
        raise ValueError("unknown primitive %r" % primitive)
    return _ORDER[primitive]


def episode_name(task: str, seed: int) -> str:
    return "%s-%d" % (task, seed)


def record_id(task: str, seed: int, decision: int) -> str:
    return "%s-%d" % (episode_name(task, seed), decision)


def image_path(task: str, seed: int, decision: int) -> str:
    """The frame of ``decision`` (the view the primitive was chosen on), relative to a dataset root."""
    return "images/%s-%d-%04d.jpg" % (task, seed, decision)


def frame_window(decision: int, k: int) -> List[int]:
    """The decisions whose frames ``BiGymGame.frames(k)`` returns at ``decision``: the last ``k``, oldest first,
    with decision 0 repeated at the start of the episode."""
    if k < 1:
        raise ValueError("k must be >= 1")
    return [max(0, decision - (k - 1) + j) for j in range(k)]


def probe_decisions(n_decisions: int, every: int = PROBE_EVERY) -> List[int]:
    """The decision frames that get probe records: every ``every``-th (from 0) plus the final frame
    ``n_decisions`` (the state after the last primitive, where the task is done)."""
    out = list(range(0, n_decisions, every))
    return out + [n_decisions] if n_decisions not in out else out


def val_seeds(task: str, seeds: Iterable[int], frac: float = VAL_FRAC) -> List[int]:
    """About ``frac`` of ``seeds`` (at least one when there are two or more), chosen by a hash of task and seed so
    the split does not depend on which other demos were kept or on their order."""
    seeds = sorted(set(int(s) for s in seeds))
    n = 0 if len(seeds) < 2 else max(1, int(round(frac * len(seeds))))
    ranked = sorted(seeds, key=lambda s: hashlib.sha1(("%s-%d" % (task, s)).encode()).hexdigest())
    return sorted(ranked[:n])


def control_records(task: str, seed: int, primitives: Sequence[str], frames: int) -> List[Dict]:
    """One record per decision of a kept run: the control question over ``frames`` head frames, labelled with the
    primitive chosen there."""
    q = bigym_question(task, frames)["action"]
    out = []
    for d, p in enumerate(primitives):
        rec = {"id": record_id(task, seed, d), "state_text": None, "question": q, "label": label_index(p),
               "primitive": p}
        if frames == 1:
            rec["image"] = image_path(task, seed, d)
        else:
            rec["images"] = [image_path(task, seed, i) for i in frame_window(d, frames)]
        out.append(rec)
    return out


def probe_records(task: str, seed: int, probes: Sequence[Dict]) -> List[Dict]:
    """``probes``: ``[{"decision", "labels": bigymgames.labels(task, truth)}]``; one record per (frame, question)."""
    qs = probe_questions(task)
    out = []
    for fr in probes:
        d = int(fr["decision"])
        for qid, q in qs.items():
            out.append({"id": "%s-%s-%d" % (episode_name(task, seed), qid, d), "image": image_path(task, seed, d),
                        "state_text": None, "question": q, "label": int(fr["labels"][qid])})
    return out
