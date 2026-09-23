"""Atari-HEAD human play -> the shared Atari training-data format (``site-docs/reference/atari-data-format.md``).

Atari-HEAD v4 (Zhang et al. 2019, "Atari-HEAD: Atari Human Eye-Tracking and Demonstration Dataset",
arXiv:1903.06754) is on Zenodo as record 3451402 under CC-BY-4.0: one ``<game>.zip`` per game plus
``meta_data.csv`` and ``action_enums.txt``. Each zip holds, per trial (one play session), a
``<trial>_<subject>_<n>_<date>.tar.bz2`` of full-colour 210x160 PNG frames and a ``.txt`` label file with columns
``frame_id,episode_id,score,duration(ms),unclipped_reward,action,gaze_positions...``. ``action`` is the ALE default
18-action enum (``action_enums.txt``: NOOP=0 ... DOWNLEFTFIRE=17, identical to ``ale_py.Action``); gaze positions
trail the row as ``x0,y0,x1,y1,...`` or ``null``. ``<game>/highscore/`` holds the best player's 2-hour trials.

Trials that continue a saved game (``load_trial`` in ``meta_data.csv``) are grouped with the trial they resume, and
train/val are split by those groups, so no game episode straddles the split. The record ``episode`` is the
Atari-HEAD ``trial_id``.
"""
import json
import os
import random
import tarfile
import time
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Sequence

ZENODO_RECORD = 3451402
FILE_URL = "https://zenodo.org/api/records/%d/files/%%s/content" % ZENODO_RECORD
ORIGIN = "Atari-HEAD v4, https://zenodo.org/records/%d (doi:10.5281/zenodo.%d)" % (ZENODO_RECORD, ZENODO_RECORD)
LICENSE = "CC-BY-4.0"
CITATION = ("Zhang, Walshe, Liu, Guan, Muller, Whritner, Zhang, Hayhoe, Ballard. Atari-HEAD: Atari Human "
            "Eye-Tracking and Demonstration Dataset. arXiv:1903.06754, 2019.")
SOURCE = "atari_head"

# action_enums.txt ("ALE default enums"), index = the label file's action integer.
FULL_ACTIONS = ["NOOP", "FIRE", "UP", "RIGHT", "LEFT", "DOWN", "UPRIGHT", "UPLEFT", "DOWNRIGHT", "DOWNLEFT",
                "UPFIRE", "RIGHTFIRE", "LEFTFIRE", "DOWNFIRE", "UPRIGHTFIRE", "UPLEFTFIRE", "DOWNRIGHTFIRE",
                "DOWNLEFTFIRE"]

# Atari-HEAD game name (zip name, meta_data.csv GameName without "_highscore") -> ALE v5 name.
GAMES = {
    "alien": "Alien", "asterix": "Asterix", "bank_heist": "BankHeist", "berzerk": "Berzerk",
    "breakout": "Breakout", "centipede": "Centipede", "demon_attack": "DemonAttack", "enduro": "Enduro",
    "freeway": "Freeway", "frostbite": "Frostbite", "hero": "Hero", "montezuma_revenge": "MontezumaRevenge",
    "ms_pacman": "MsPacman", "name_this_game": "NameThisGame", "phoenix": "Phoenix", "riverraid": "Riverraid",
    "road_runner": "RoadRunner", "seaquest": "Seaquest", "space_invaders": "SpaceInvaders", "venture": "Venture",
}

_PNG_SIG = b"\x89PNG\r\n\x1a\n"
_PNG_COLOR = {0: "gray", 2: "rgb", 3: "palette", 4: "gray_alpha", 6: "rgba"}


def download(url: str, path: str, n_conn: int = 6, retries: int = 20) -> int:
    """Fetch ``url`` to ``path`` over ``n_conn`` parallel HTTP range requests, resuming each range on failure."""
    import requests

    size = int(requests.head(url, allow_redirects=True, timeout=60).headers["Content-Length"])
    with open(path, "wb") as f:
        f.truncate(size)
    bounds = [(size * i // n_conn, size * (i + 1) // n_conn) for i in range(n_conn)]

    def fetch(lo_hi):
        pos, hi = lo_hi
        with open(path, "r+b") as f:
            for attempt in range(retries):
                try:
                    with requests.get(url, headers={"Range": "bytes=%d-%d" % (pos, hi - 1)}, stream=True,
                                      timeout=120) as r:
                        if r.status_code == 429:
                            raise IOError("rate limited")
                        r.raise_for_status()
                        f.seek(pos)
                        for chunk in r.iter_content(1 << 20):
                            f.write(chunk)
                            pos += len(chunk)
                    if pos >= hi:
                        return
                except Exception as e:  # noqa: BLE001 - network errors of every kind get retried
                    print("  range %d-%d: %s (attempt %d)" % (pos, hi, e, attempt + 1), flush=True)
                    time.sleep(min(60, 5 * (attempt + 1)))
        raise IOError("download failed: %s bytes %d-%d" % (url, pos, hi))

    with ThreadPoolExecutor(n_conn) as ex:
        list(ex.map(fetch, bounds))
    return size


def trial_groups(meta_csv: str, dataset_game: str) -> Dict[int, int]:
    """trial_id -> id of the first trial in its chain of saved-game continuations (``load_trial``)."""
    rows = [l.split(",") for l in meta_csv.splitlines()[1:] if l.strip()]
    parent = {int(r[1]): int(r[3]) for r in rows if r[0] in (dataset_game, dataset_game + "_highscore")}

    def root(t):
        seen = set()
        while parent.get(t, 0) and t not in seen:
            seen.add(t)
            t = parent[t]
        return t

    return {t: root(t) for t in parent}


def meta_rows(meta_csv: str, dataset_game: str) -> Dict[int, Dict[str, str]]:
    lines = [l.split(",") for l in meta_csv.splitlines() if l.strip()]
    head = [h.strip() for h in lines[0]]
    return {int(r[1]): dict(zip(head, (x.strip() for x in r))) for r in lines[1:]
            if r[0] in (dataset_game, dataset_game + "_highscore")}


def parse_labels(text: str) -> List[Dict]:
    """Rows of one trial's label file: frame_id, step, raw action (int or None), in-trial episode, has_gaze."""
    rows = []
    for line in text.splitlines()[1:]:
        parts = line.strip().split(",")
        if len(parts) < 6 or not parts[0]:
            continue
        try:
            action = int(parts[5])
        except ValueError:
            action = None
        if action is not None and not 0 <= action < len(FULL_ACTIONS):
            action = None
        try:
            step = int(parts[0].rsplit("_", 1)[1])
        except (IndexError, ValueError):
            step = len(rows) + 1
        gaze = parts[6:]
        rows.append({"frame_id": parts[0], "step": step, "action": action,
                     "trial_episode": None if parts[1] == "null" else parts[1],
                     "has_gaze": len(gaze) >= 2 and gaze[0] not in ("null", "")})
    return rows


def evenly(items: Sequence, k: int) -> List:
    """``k`` items spread evenly over ``items`` (all of them if there are no more than ``k``)."""
    n = len(items)
    if n <= k:
        return list(items)
    return [items[(2 * i + 1) * n // (2 * k)] for i in range(k)]


def split_groups(frames_per_group: Dict[int, int], val_frac: float = 0.05, seed: int = 0) -> List[int]:
    """Pick val groups (seeded) until they hold ~``val_frac`` of the frames, skipping any that would overshoot 2x."""
    groups = sorted(frames_per_group)
    random.Random(seed).shuffle(groups)
    total = sum(frames_per_group.values())
    val, n = [], 0
    for g in groups[:-1]:  # always leave at least one group for train
        if n >= val_frac * total:
            break
        if val and n + frames_per_group[g] > 2 * val_frac * total:
            continue
        if not val and frames_per_group[g] > 4 * val_frac * total and len(groups) > 2:
            continue
        val.append(g)
        n += frames_per_group[g]
    if not val:  # every group is large: take the smallest one
        val = [min(groups, key=lambda g: frames_per_group[g])]
    return sorted(val)


def _png_format(b: bytes) -> str:
    if b[:8] != _PNG_SIG or b[12:16] != b"IHDR":
        return "not_png"
    w, h = int.from_bytes(b[16:20], "big"), int.from_bytes(b[20:24], "big")
    return "%s_%dx%d" % (_PNG_COLOR.get(b[25], "color%d" % b[25]), h, w)


def _extract(zf_path: str, tar_name: str, wanted: Dict[str, str], img_dir: str) -> Dict[str, str]:
    """Stream one trial's tar.bz2 and write the ``wanted`` frames (member basename -> record id). Returns id -> format."""
    found = {}
    with zipfile.ZipFile(zf_path) as z, z.open(tar_name) as f, tarfile.open(fileobj=f, mode="r|bz2") as tf:
        for m in tf:
            rid = wanted.get(os.path.basename(m.name)) if m.isfile() else None
            if rid is None:
                continue
            b = tf.extractfile(m).read()
            with open(os.path.join(img_dir, rid + ".png"), "wb") as o:
                o.write(b)
            found[rid] = _png_format(b)
    return found


def convert_game(zip_path: str, dataset_game: str, out_dir: str, meta_csv: str, actions: Sequence[str],
                 max_train: int = 20000, max_val: int = 1000, val_frac: float = 0.05, seed: int = 0,
                 workers: int = 8) -> Dict:
    """Write ``train.jsonl``, ``val.jsonl``, ``images/`` and ``meta.json`` for one game into ``out_dir``.

    ``actions`` is the game's ALE v5 minimal action set. Labels index into it; frames whose action is outside it,
    null/invalid, or whose image is missing from the tar are dropped and counted.
    """
    from laya.games import atari_question

    game = GAMES[dataset_game]
    actions = list(actions)
    question = atari_question(game, actions)["action"]
    groups = trial_groups(meta_csv, dataset_game)
    trial_meta = meta_rows(meta_csv, dataset_game)

    z = zipfile.ZipFile(zip_path)
    names = [i.filename for i in z.infolist()]
    stems = sorted(n[:-4] for n in names if n.endswith(".txt"))
    trials = {}
    for stem in stems:
        tid = int(os.path.basename(stem).split("_")[0])
        if stem + ".tar.bz2" not in names:
            print("  %s: no tar.bz2 for %s, skipped" % (dataset_game, stem))
            continue
        trials[tid] = stem

    dropped = Counter()
    raw_counts, outside = Counter(), Counter()
    valid = defaultdict(list)  # trial -> [(step, frame_id, label, has_gaze)]
    label_rows, gaze_rows = 0, 0
    for tid, stem in sorted(trials.items()):
        for r in parse_labels(z.read(stem + ".txt").decode("utf-8", "replace")):
            label_rows += 1
            gaze_rows += r["has_gaze"]
            if r["action"] is None:
                dropped["invalid_action"] += 1
                continue
            name = FULL_ACTIONS[r["action"]]
            raw_counts[name] += 1
            if name not in actions:
                dropped["not_in_minimal_set"] += 1
                outside[name] += 1
                continue
            valid[tid].append((r["step"], r["frame_id"], actions.index(name), r["has_gaze"]))
    z.close()

    per_group = Counter()
    for tid, fr in valid.items():
        per_group[groups.get(tid, tid)] += len(fr)
    val_groups = set(split_groups(dict(per_group), val_frac, seed))
    split_trials = {"train": sorted(t for t in valid if groups.get(t, t) not in val_groups),
                    "val": sorted(t for t in valid if groups.get(t, t) in val_groups)}

    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    chosen = {}
    for split, cap in (("train", max_train), ("val", max_val)):
        pool = [(t,) + f for t in split_trials[split] for f in sorted(valid[t])]
        chosen[split] = evenly(pool, cap)

    wanted = defaultdict(dict)  # trial -> {png name: record id}
    for split in chosen:
        for t, step, frame_id, _, _ in chosen[split]:
            wanted[t][frame_id + ".png"] = "ah-%s-t%03d-s%06d" % (game, t, step)
    formats = {}
    with ThreadPoolExecutor(workers) as ex:
        for found in ex.map(lambda t: _extract(zip_path, trials[t] + ".tar.bz2", wanted[t], img_dir), sorted(wanted)):
            formats.update(found)

    meta = {"source": SOURCE, "game": game, "dataset_game": dataset_game, "actions": actions}
    for split in ("train", "val"):
        labels, n, eps, gz = Counter(), 0, set(), 0
        with open(os.path.join(out_dir, split + ".jsonl"), "w") as f:
            for t, step, frame_id, label, has_gaze in chosen[split]:
                rid = "ah-%s-t%03d-s%06d" % (game, t, step)
                if rid not in formats:
                    dropped["missing_image"] += 1
                    continue
                f.write(json.dumps({"id": rid, "image": "images/%s.png" % rid, "game": game, "actions": actions,
                                    "label": label, "question": question, "source": SOURCE, "episode": t,
                                    "step": step, "frame_id": frame_id}) + "\n")
                labels[actions[label]] += 1
                eps.add(t)
                gz += has_gaze
                n += 1
        meta[split] = {"records": n, "episodes": len(eps), "labels": dict(labels.most_common()),
                       "trials": sorted(eps), "available_frames": sum(len(valid[t]) for t in split_trials[split]),
                       "frames_with_gaze": gz}
    fmt = Counter(formats.values())
    meta["frame_format"] = "rgb_210x160" if set(fmt) == {"rgb_210x160"} else "mixed:" + json.dumps(dict(fmt))
    meta["frame_formats_seen"] = dict(fmt)
    meta["dropped"] = {"not_in_minimal_set": dropped["not_in_minimal_set"], "invalid_action": dropped["invalid_action"],
                       "missing_image": dropped["missing_image"]}
    meta["not_in_minimal_set_actions"] = dict(outside.most_common())
    meta["raw_action_counts"] = dict(raw_counts.most_common())
    meta["label_rows"] = label_rows
    meta["trials"] = len(trials)
    meta["frame_averaging_trials"] = sorted(t for t in trials if trial_meta.get(t, {}).get("frame_averaging") == "TRUE")
    meta["gaze"] = {"available": gaze_rows > 0, "label_rows_with_gaze": gaze_rows,
                    "note": "per-frame gaze_positions (x0,y0,...; origin top-left, 160x210 screen) are in the source "
                            "label files; not included in records"}
    meta["action_encoding"] = "ALE default 18-action enum (action_enums.txt), mapped by name to the minimal set"
    meta["episode_field"] = "Atari-HEAD trial_id; train/val split by load_trial chains of trials"
    meta["val_groups"] = sorted(val_groups)
    meta["origin"] = ORIGIN
    meta["license"] = LICENSE
    meta["citation"] = CITATION
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return meta
