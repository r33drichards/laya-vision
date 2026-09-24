"""Convert the Atari subsets of ``jat-project/jat-dataset`` into the shared format of ``site-docs/reference/atari-data-format.md``.

How JAT recorded Atari (``huggingface/jat`` ``data/envs/atari/create_atari_dataset.py``): a Sample Factory agent on
envpool (``envpool.make(..., episodic_life=True, reward_clip=True)``, defaults ``full_action_space=False``,
``gray_scale=True``, 84x84, ``stack_num=4``, ``frame_skip=4``). Each dataset row is one complete episode:

- ``image_observations[t]``: the observation the action was taken on, saved with
  ``Image.fromarray(obs.transpose(1, 2, 0))``, i.e. a 4-channel 84x84 PNG whose channels are the 4 stacked gray
  frames, oldest first. We keep only the last channel (the current frame). Two games (``SCRAMBLED``) hold the
  same stack with its bytes in channel-last order (see ``decode_stack``).
- ``discrete_actions[t]``: an index into envpool's action set, which is ``getMinimalActionSet()`` (envpool
  ``atari_env.h``), the same list and order as ``gym.make("ALE/<Game>-v5").unwrapped.get_action_meanings()``.

JAT's own ``train``/``test`` splits are disjoint sets of episodes; we use them as our ``train``/``val``.
"""
import io
import json
import os
from collections import Counter
from typing import Dict, List, Optional

import numpy as np

REPO = "jat-project/jat-dataset"
ORIGIN = "https://huggingface.co/datasets/jat-project/jat-dataset"
LICENSE = "apache-2.0"
FRAME_FORMAT = "gray_84x84"

# JAT config suffix -> ALE v5 name, from TASK_NAME_TO_ENV_ID in huggingface/jat jat/eval/rl/core.py.
JAT_TO_ALE = {
    "alien": "Alien", "amidar": "Amidar", "assault": "Assault", "asterix": "Asterix", "asteroids": "Asteroids",
    "atlantis": "Atlantis", "bankheist": "BankHeist", "battlezone": "BattleZone", "beamrider": "BeamRider",
    "berzerk": "Berzerk", "bowling": "Bowling", "boxing": "Boxing", "breakout": "Breakout", "centipede": "Centipede",
    "choppercommand": "ChopperCommand", "crazyclimber": "CrazyClimber", "defender": "Defender",
    "demonattack": "DemonAttack", "doubledunk": "DoubleDunk", "enduro": "Enduro", "fishingderby": "FishingDerby",
    "freeway": "Freeway", "frostbite": "Frostbite", "gopher": "Gopher", "gravitar": "Gravitar", "hero": "Hero",
    "icehockey": "IceHockey", "jamesbond": "Jamesbond", "kangaroo": "Kangaroo", "krull": "Krull",
    "kungfumaster": "KungFuMaster", "montezumarevenge": "MontezumaRevenge", "mspacman": "MsPacman",
    "namethisgame": "NameThisGame", "phoenix": "Phoenix", "pitfall": "Pitfall", "pong": "Pong",
    "privateeye": "PrivateEye", "qbert": "Qbert", "riverraid": "Riverraid", "roadrunner": "RoadRunner",
    "robotank": "Robotank", "seaquest": "Seaquest", "skiing": "Skiing", "solaris": "Solaris",
    "spaceinvaders": "SpaceInvaders", "stargunner": "StarGunner", "surround": "Surround", "tennis": "Tennis",
    "timepilot": "TimePilot", "tutankham": "Tutankham", "upndown": "UpNDown", "venture": "Venture",
    "videopinball": "VideoPinball", "wizardofwor": "WizardOfWor", "yarsrevenge": "YarsRevenge", "zaxxon": "Zaxxon",
}


def minimal_actions(game: str) -> List[str]:
    import ale_py
    import gymnasium as gym

    gym.register_envs(ale_py)
    env = gym.make("ALE/%s-v5" % game)
    try:
        return list(env.unwrapped.get_action_meanings())
    finally:
        env.close()


def parquet_files(jat_name: str, split: str) -> List[str]:
    """Repo paths of ``atari-<jat_name>/<split>-*.parquet``, in shard order."""
    from huggingface_hub import HfApi

    prefix = "atari-%s/%s-" % (jat_name, split)
    return sorted(f for f in HfApi().list_repo_files(REPO, repo_type="dataset")
                  if f.startswith(prefix) and f.endswith(".parquet"))


def download(paths: List[str], local_dir: str) -> List[str]:
    from huggingface_hub import hf_hub_download

    return [hf_hub_download(REPO, p, repo_type="dataset", local_dir=local_dir) for p in paths]


# In these configs each 84x84x4 PNG holds the bytes of an (84, 84, 4) channel-last stack written as if it were
# channel-first, so ``a.transpose(2, 0, 1)`` is the flat stack. Found with ``inspect_game``: the raw channels fail the
# frame-shift test and look like noise, the decoded ones pass it on every step and show the game.
SCRAMBLED = {"kungfumaster", "montezumarevenge"}


def decode_stack(png: bytes, scrambled: bool = False) -> np.ndarray:
    """A stored JAT observation as a (4, 84, 84) uint8 frame stack, oldest frame first."""
    from PIL import Image

    a = np.asarray(Image.open(io.BytesIO(png)))
    if a.ndim == 2:
        return a[None]
    if scrambled:
        return np.ascontiguousarray(a.transpose(2, 0, 1)).reshape(a.shape).transpose(2, 0, 1)
    return a.transpose(2, 0, 1)


def stack_shift_match(stacks: np.ndarray) -> float:
    """Fraction of steps t where frames 1..3 of stack t equal frames 0..2 of stack t+1 (1.0 for a real stack)."""
    return float(np.mean([(stacks[t, 1:] == stacks[t + 1, :-1]).all() for t in range(len(stacks) - 1)]))


def check_stacks(file: str, scrambled: bool, steps: int = 64) -> float:
    """``stack_shift_match`` over the first ``steps`` observations of the first episode in ``file``."""
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(file)
    frames = pf.read_row_group(0, columns=["image_observations"]).column(0)[0].values.field("bytes")
    n = min(steps, len(frames))
    return stack_shift_match(np.stack([decode_stack(frames[i].as_py(), scrambled) for i in range(n)]))


def even_indices(total: int, k: int) -> np.ndarray:
    """``min(k, total)`` distinct indices spread evenly over ``range(total)``."""
    if total <= k:
        return np.arange(total)
    return np.unique(np.linspace(0, total - 1, k).round().astype(np.int64))


def read_actions(files: List[str]) -> List[np.ndarray]:
    """Per-episode action arrays, reading only the ``discrete_actions`` column."""
    import pyarrow.parquet as pq

    eps = []
    for f in files:
        for a in pq.read_table(f, columns=["discrete_actions"]).column(0).to_pylist():
            eps.append(np.asarray(a, dtype=np.int64))
    return eps


def iter_episode_images(files: List[str], wanted: Dict[int, List[int]]):
    """Yield ``(episode, step, png_bytes)`` for the steps in ``wanted[episode]``, one row group at a time."""
    import pyarrow.parquet as pq

    ep = 0
    for f in files:
        pf = pq.ParquetFile(f)
        for g in range(pf.num_row_groups):
            n = pf.metadata.row_group(g).num_rows
            if not any(ep + r in wanted for r in range(n)):
                ep += n
                continue
            col = pf.read_row_group(g, columns=["image_observations"]).column(0)
            for r in range(n):
                steps = wanted.get(ep + r)
                if steps:
                    frames = col[r].values.field("bytes")
                    for s in steps:
                        yield ep + r, s, frames[s].as_py()
            ep += n
            del col


def convert_split(game: str, actions: List[str], question: Dict, files: List[str], cap: int, ep_offset: int,
                  out_dir: str, scrambled: bool = False) -> Dict:
    """Write one split's images and records; returns its stats (and the records under ``"_records"``)."""
    from PIL import Image

    eps = read_actions(files)
    lengths = np.array([len(a) for a in eps], dtype=np.int64)
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    n = len(actions)

    # Drop out-of-range actions before subsampling, so the cap counts kept frames.
    valid = np.concatenate([(a >= 0) & (a < n) for a in eps]) if eps else np.zeros(0, bool)
    dropped = int((~valid).sum())
    pick = np.flatnonzero(valid)[even_indices(int(valid.sum()), cap)]
    ep_of = np.searchsorted(starts, pick, side="right") - 1
    wanted: Dict[int, List[int]] = {}
    for g, e in zip(pick.tolist(), ep_of.tolist()):
        wanted.setdefault(e, []).append(g - int(starts[e]))

    records = []
    for e, s, png in iter_episode_images(files, wanted):
        episode = ep_offset + e
        rid = "jat-%s-e%06d-s%06d" % (game, episode, s)
        frame = decode_stack(png, scrambled)[-1]
        Image.fromarray(frame, mode="L").save(os.path.join(out_dir, "images", rid + ".png"))
        records.append({"id": rid, "image": "images/%s.png" % rid, "game": game, "actions": actions,
                        "label": int(eps[e][s]), "question": question, "source": "jat", "episode": episode,
                        "step": s})
    labels = Counter(actions[r["label"]] for r in records)
    all_actions = np.concatenate(eps) if eps else np.zeros(0, np.int64)
    return {"records": len(records), "episodes": len({r["episode"] for r in records}),
            "labels": dict(labels.most_common()), "source_episodes": len(eps), "source_frames": int(lengths.sum()),
            "source_action_max": int(all_actions.max()) if len(all_actions) else None,
            "dropped": dropped, "_records": records}


def convert_game(jat_name: str, out_root: str, train_cap: int = 20_000, val_cap: int = 1_000,
                 download_dir: str = "/tmp/jat", commit=None) -> Dict:
    """Convert ``atari-<jat_name>`` to ``<out_root>/<Game>/``. ``commit`` (e.g. ``vol.commit``) runs before ``_READY``."""
    from laya.games import atari_question

    game = JAT_TO_ALE[jat_name]
    actions = minimal_actions(game)
    question = atari_question(game, actions)["action"]
    out_dir = os.path.join(out_root, game)
    ready = os.path.join(out_dir, "_READY")
    if os.path.exists(ready):
        os.remove(ready)
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)

    train_files = download(parquet_files(jat_name, "train"), download_dir)
    val_files = download(parquet_files(jat_name, "test"), download_dir)
    scrambled = jat_name in SCRAMBLED
    stack_check = {"train": check_stacks(train_files[0], scrambled), "val": check_stacks(val_files[0], scrambled)}
    if min(stack_check.values()) < 0.9:
        raise ValueError("%s: frame stacks don't decode (shift match %s)" % (jat_name, stack_check))
    train = convert_split(game, actions, question, train_files, train_cap, 0, out_dir, scrambled)
    val = convert_split(game, actions, question, val_files, val_cap, train["source_episodes"], out_dir, scrambled)
    for f in train_files + val_files:
        os.remove(f)

    for name, st in (("train", train), ("val", val)):
        with open(os.path.join(out_dir, name + ".jsonl"), "w") as fh:
            for r in st.pop("_records"):
                fh.write(json.dumps(r) + "\n")
    meta = {"source": "jat", "game": game, "frame_format": FRAME_FORMAT, "actions": actions,
            "train": train, "val": val,
            "dropped": {"not_in_minimal_set": train["dropped"] + val["dropped"]},
            "origin": "%s (config atari-%s; train split -> train, test split -> val)" % (ORIGIN, jat_name),
            "license": LICENSE, "stack_check": stack_check, "stack_layout": "scrambled" if scrambled else "chw",
            "notes": "Frames are the most recent frame of envpool's 4-frame gray 84x84 stack (frame_skip 4, "
                     "no sticky actions). Actions are indices into ALE's minimal action set."}
    with open(os.path.join(out_dir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    if commit is not None:
        commit()
    open(ready, "w").close()
    if commit is not None:
        commit()
    return meta


def inspect_game(jat_name: str, download_dir: str = "/tmp/jat_inspect", max_frames: Optional[int] = 1000) -> Dict:
    """Stream the first test episode: image mode/size, frame-stack order, action range vs. the minimal set."""
    import pyarrow.parquet as pq
    from PIL import Image

    game = JAT_TO_ALE[jat_name]
    actions = minimal_actions(game)
    f = download(parquet_files(jat_name, "test")[:1], download_dir)[0]
    pf = pq.ParquetFile(f)
    t = pf.read_row_group(0)
    acts = np.asarray(t.column("discrete_actions")[0].as_py())
    frames = t.column("image_observations")[0].values.field("bytes")
    n = min(len(frames), max_frames or len(frames))
    im0 = Image.open(io.BytesIO(frames[0].as_py()))
    stack = np.stack([np.asarray(Image.open(io.BytesIO(frames[i].as_py()))) for i in range(n)])
    # If channel c+1 at step t equals channel c at step t+1, channels run oldest -> newest.
    shift = [float(np.mean([(stack[i, ..., c + 1] == stack[i + 1, ..., c]).all() for i in range(n - 1)]))
             for c in range(stack.shape[-1] - 1)] if stack.ndim == 4 else []
    decoded = np.stack([decode_stack(frames[i].as_py(), jat_name in SCRAMBLED) for i in range(n)])
    all_acts = np.concatenate([np.asarray(a) for a in pq.read_table(f, columns=["discrete_actions"])
                              .column(0).to_pylist()])
    os.remove(f)
    return {"game": game, "schema": str(pf.schema_arrow), "row_groups": pf.num_row_groups,
            "rows": pf.metadata.num_rows, "episode_len": int(len(acts)), "image_mode": im0.mode,
            "image_size": im0.size, "array_shape": list(stack.shape[1:]), "stack_shift_match": shift,
            "decoded_shift_match": stack_shift_match(decoded),
            "minimal_set": actions, "n_minimal": len(actions), "action_max_test_split": int(all_acts.max()),
            "distinct_actions_test_split": int(len(np.unique(all_acts))),
            "sample_stack": stack.tobytes(), "sample_shape": list(stack.shape), "sample_actions": acts[:n].tolist()}
