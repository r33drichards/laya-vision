"""Label cleaning and probe rebalancing for the BiGym behaviour-cloning sets (``laya.bigymdata``), pure Python.

The demo follower (``laya.bigymdemos.follow``) picks each primitive greedily, so near a waypoint it often dithers:
``RIGHT_HAND_LEFT RIGHT_HAND_RIGHT RIGHT_HAND_LEFT ...`` until its patience runs out. Those decisions are real
labels (the lookahead's try-and-undo steps are not recorded), but a policy cloned from them learns to oscillate.
Measured on ``bigym_v2_bc_f1`` (23,020 train records, 134 episodes): about 35% of the records sit in such
alternating chains, most of them 5 to 7 long; cancelling only adjacent pairs removes within 0.4% of what a full
stack reduction (``A B B' A'`` too) would; STAY never occurs (``lookahead`` breaks STAY ties by moving on to the
next waypoint), so the STAY rule is a no-op on that data.

Control rules (``clean_sequence``), applied to each episode's recorded sequence (grouped by ``<task>-<seed>``,
ordered by decision), both read off the original sequence (no cascading after a removal):

- **undo**: ``B`` undoes ``A`` when ``inverse(A) == B`` (``*_FORWARD``/``*_BACK``, ``*_LEFT``/``*_RIGHT``,
  ``*_UP``/``*_DOWN``, ``*_WRIST_CW``/``*_WRIST_CCW``, ``*_GRIPPER_CLOSE``/``*_GRIPPER_OPEN``, same hand or the base,
  which covers the hand moves, tilts, base steps / sidesteps / turns / crouch). A maximal chain of consecutive
  undoes ``A A' A A' ...`` of length ``L >= 2`` nets out to nothing when ``L`` is even (all ``L`` dropped) and to one
  ``A`` when it is odd: the chain's last record is kept (the move the follower left the chain with) and the other
  ``L - 1`` are dropped. So ``A A'`` and ``A A' A A'`` go entirely, ``A A' A`` keeps its last ``A``, and
  ``A A A'`` keeps its first ``A``.
- **stay**: of a run of consecutive ``STAY`` records only the first is kept.

Everything else is kept as recorded, frames included (the four-frame windows of ``bc_f4`` still show the frames of
dropped decisions; they are what the policy saw). Ids are not renumbered, so ``vlm_train.episode_step`` still
places each record at its original decision and ``with_next_targets`` only links records that were consecutive
decisions.

Probe rule (``oversample``): in a split, per task, ``done=1`` records are duplicated (ids suffixed ``-dup<N>``,
N = 1, 2, ...; such ids no longer parse as ``<episode>-<step>``, so the copies never collide as game frames) until
they make up ``target`` of that task's ``done`` records: ``round(target * n0 / (1 - target))`` positives in total,
spread as evenly as possible over the originals (every original gets ``m`` or ``m + 1`` copies). The other
questions are untouched.
"""
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

UNDO_PAIRS = (("FORWARD", "BACK"), ("LEFT", "RIGHT"), ("UP", "DOWN"), ("CW", "CCW"), ("CLOSE", "OPEN"))
_FLIP = {a: b for a, b in UNDO_PAIRS} | {b: a for a, b in UNDO_PAIRS}
MAX_DROP_FRAC = 0.4  # episodes losing more than this are listed in the report (the rule still applies)
DONE_TARGET = 0.15
CONTROL_RULES = {  # recorded in the cleaned datasets' meta.json
    "undo": "B undoes A when inverse(A) == B (FORWARD/BACK, LEFT/RIGHT, UP/DOWN, WRIST_CW/CCW, GRIPPER_CLOSE/OPEN "
            "of the same hand, or of the base: hand moves, tilts, base steps / sidesteps / turns / crouch). A maximal "
            "chain of consecutive undoes A A' A A' ... of length L >= 2 is dropped entirely when L is even; when L "
            "is odd its last record (the net move) is kept and the other L - 1 dropped",
    "stay": "of a run of consecutive STAY records only the first is kept",
    "scope": "train and val alike, per episode (<task>-<seed>, ordered by decision); both rules read the recorded "
             "sequence (no cascading after a removal); frames and 4-frame windows kept as recorded; ids not "
             "renumbered",
}


def inverse(primitive: str) -> Optional[str]:
    """The primitive that undoes ``primitive`` (same hand / the base, opposite direction), or None (``STAY``)."""
    head, _, last = primitive.rpartition("_")
    return head + "_" + _FLIP[last] if head and last in _FLIP else None


def undo_drops(seq: Sequence[str]) -> List[bool]:
    """Per decision, whether the undo rule drops it (see the module docstring)."""
    out = [False] * len(seq)
    i = 0
    while i < len(seq):
        j = i
        while j + 1 < len(seq) and inverse(seq[j]) == seq[j + 1]:
            j += 1
        n = j - i + 1
        if n >= 2:
            for x in range(i, j + 1 if n % 2 == 0 else j):
                out[x] = True
        i = j + 1
    return out


def stay_drops(seq: Sequence[str]) -> List[bool]:
    """Per decision, whether the STAY rule drops it: every STAY that directly follows a STAY."""
    return [p == "STAY" and i > 0 and seq[i - 1] == "STAY" for i, p in enumerate(seq)]


def clean_sequence(seq: Sequence[str]) -> Dict:
    """``{"keep": [bool per decision], "undo": n dropped by the undo rule, "stay": n dropped by the STAY rule}``
    (the rules never overlap: STAY has no inverse)."""
    u, s = undo_drops(seq), stay_drops(seq)
    return {"keep": [not (a or b) for a, b in zip(u, s)], "undo": sum(u), "stay": sum(s)}


def split_control_id(rec_id: str) -> Tuple[str, int, int]:
    """``"<task>-<seed>-<decision>"`` -> (task, seed, decision)."""
    task, seed, decision = rec_id.rsplit("-", 2)
    return task, int(seed), int(decision)


def clean_control(records: List[Dict], primitives: Sequence[str],
                  max_drop_frac: float = MAX_DROP_FRAC) -> Tuple[List[Dict], Dict]:
    """Apply both rules per episode to control records (``bigymdata.control_records``; ``label`` indexes
    ``primitives``). Returns the kept records in input order and a report: per task the records before / after,
    the counts each rule dropped and the label counts before / after, plus the episodes that lost more than
    ``max_drop_frac`` (``flagged``). Episodes must be complete runs of decisions ``0..n-1``."""
    eps = defaultdict(list)
    for i, r in enumerate(records):
        task, seed, d = split_control_id(r["id"])
        eps[(task, seed)].append((d, i))
    keep = [False] * len(records)
    tasks: Dict[str, Dict] = {}
    flagged = []
    for (task, seed), rows in sorted(eps.items()):
        rows.sort()
        if [d for d, _ in rows] != list(range(len(rows))):
            raise ValueError("%s-%d: decisions are not 0..%d" % (task, seed, len(rows) - 1))
        seq = [primitives[records[i]["label"]] for _, i in rows]
        c = clean_sequence(seq)
        for (_, i), k in zip(rows, c["keep"]):
            keep[i] = k
        t = tasks.setdefault(task, {"episodes": 0, "before": 0, "after": 0, "undo": 0, "stay": 0,
                                    "labels_before": Counter(), "labels_after": Counter()})
        t["episodes"] += 1
        t["before"] += len(seq)
        t["after"] += sum(c["keep"])
        t["undo"] += c["undo"]
        t["stay"] += c["stay"]
        t["labels_before"].update(seq)
        t["labels_after"].update(p for p, k in zip(seq, c["keep"]) if k)
        frac = 1 - sum(c["keep"]) / len(seq)
        if frac > max_drop_frac:
            flagged.append({"episode": "%s-%d" % (task, seed), "decisions": len(seq), "dropped": len(seq) - sum(
                c["keep"]), "frac": round(frac, 3)})
    for t in tasks.values():
        for k in ("labels_before", "labels_after"):
            t[k] = dict(t[k].most_common())
    report = {"before": len(records), "after": sum(keep), "undo": sum(t["undo"] for t in tasks.values()),
              "stay": sum(t["stay"] for t in tasks.values()), "per_task": tasks, "max_drop_frac": max_drop_frac,
              "flagged": sorted(flagged, key=lambda f: -f["frac"])}
    return [r for r, k in zip(records, keep) if k], report


def probe_question(rec_id: str) -> Tuple[str, str]:
    """``"<task>-<seed>-<question>-<decision>"`` -> (task, question)."""
    task, _seed, question, _decision = rec_id.rsplit("-", 3)
    return task, question


def oversample(records: List[Dict], question: str = "done", positive: int = 1,
               target: float = DONE_TARGET) -> Tuple[List[Dict], Dict]:
    """Duplicate the ``question`` records labelled ``positive`` per task until they are ``target`` of that task's
    ``question`` records (never fewer than there are). Copies follow the originals at the end of the list, ids
    ``<id>-dup<N>``. Returns the records and ``{task: {"records", "positive", "negative", "positive_after",
    "records_after", "frac_before", "frac_after"}}``."""
    if not 0 < target < 1:
        raise ValueError("target must be in (0, 1)")
    pos, neg = defaultdict(list), Counter()
    for r in records:
        task, q = probe_question(r["id"])
        if q != question:
            continue
        if int(r["label"]) == positive:
            pos[task].append(r)
        else:
            neg[task] += 1
    out, report = list(records), {}
    for task in sorted(set(pos) | set(neg)):
        p, n = pos[task], neg[task]
        want = max(len(p), int(round(target * n / (1 - target)))) if p else 0
        extra = want - len(p)
        for j, r in enumerate(p):  # copies per original: extra // len(p), one more for the first extra % len(p)
            for c in range(extra // len(p) + (1 if j < extra % len(p) else 0)):
                out.append(dict(r, id="%s-dup%d" % (r["id"], c + 1)))
        report[task] = {"records": len(p) + n, "positive": len(p), "negative": n, "positive_after": want,
                        "records_after": want + n, "frac_before": round(len(p) / max(1, len(p) + n), 4),
                        "frac_after": round(want / max(1, want + n), 4)}
    return out, report
