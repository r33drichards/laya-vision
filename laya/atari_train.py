"""Game-only Atari training helpers for ``modal_atari_train.py``: load the shared Atari data layout, hold out
calibration frames, score per game, and play ALE games with a checkpoint.

Data layout (``docs/atari-data-format.md``): ``<root>/<source>/<Game>/{train,val}.jsonl`` + ``images/`` +
``meta.json`` + ``_READY``. Each record becomes a ``choice`` example tagged ``dataset = "<source>/<Game>"`` and
``game``. A record's optional soft ``target`` is the training target; otherwise it is one-hot from ``label``.
Training balances games, not sources: every game gets an equal share of samples and, within a game, the frames
of all its sources are pooled. With ``frames=2`` the state is ``{"images": [prev_image, image]}`` (oldest first;
records without ``prev_image`` repeat ``image``), and play keeps the previous observation by the same rule.

Local smoke test (a tiny synthetic dataset from real ALE frames, 2 CPU training steps, a few play steps):
    python -m laya.atari_train --smoke /tmp/atari_synth [--frames 2]
"""
import argparse
import json
import os
import random
from collections import Counter
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .common import QTYPES, render_options, temp_bucket
from .games import atari_question
from .preprocess import FrameFeatureCache
from .vlm import VLMAgent, build_vlm_inputs, collate_vlm, vlm_prefix
from .vlm_train import fit_temperatures_from, jsonl_example, metrics_from

ATARI_ROOT = "/data/atari"
SOURCES = ("expert", "atari_head", "jat")

# ---------------------------------------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------------------------------------


def ready_datasets(root: str = ATARI_ROOT, sources: Optional[Sequence[str]] = None,
                   games: Optional[Sequence[str]] = None) -> List[Dict]:
    """Every ``<root>/<source>/<Game>`` with a ``_READY`` marker, optionally filtered by source and game."""
    out = []
    for source in sorted(os.listdir(root)) if os.path.isdir(root) else []:
        sdir = os.path.join(root, source)
        if (sources and source not in sources) or not os.path.isdir(sdir):
            continue
        for game in sorted(os.listdir(sdir)):
            d = os.path.join(sdir, game)
            if (games and game not in games) or not os.path.exists(os.path.join(d, "_READY")):
                continue
            try:
                with open(os.path.join(d, "meta.json")) as f:
                    meta = json.load(f)
            except (OSError, ValueError):
                meta = {}
            out.append({"source": source, "game": game, "dir": d, "name": "%s/%s" % (source, game), "meta": meta,
                        "frame_format": meta.get("frame_format", "?")})
    return out


def even_subsample(items: List, n: Optional[int]) -> List:
    """``n`` items spread evenly over the list (records are grouped by episode, so this spans episodes)."""
    if not n or len(items) <= n:
        return items
    return [items[int(i * len(items) / n)] for i in range(n)]


def load_split(ds: Dict, split: str, limit: Optional[int] = None, frames: int = 1) -> List[Dict]:
    """Examples from one ready dataset's ``<split>.jsonl``; ``limit`` subsamples evenly. ``frames=2`` makes the
    state ``{"images": [prev_image, image]}``."""
    path = os.path.join(ds["dir"], split + ".jsonl")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        recs = [json.loads(line) for line in f if line.strip()]
    out = []
    for r in even_subsample(recs, limit):
        ex = jsonl_example(r, ds["dir"], ds["name"])
        if ex is not None:
            ex.update(game=ds["game"], source=ds["source"], episode=r.get("episode"),
                      taken=r.get("taken", ex["label"]), rtg=r.get("return_to_go"),
                      episode_score=r.get("episode_score"))
            if frames == 2:
                cur = ex["state"]["image"]
                ex["state"] = {"images": [os.path.join(ds["dir"], r["prev_image"]) if r.get("prev_image") else cur, cur]}
            out.append(ex)
    return out


def hold_out_episodes(examples: List[Dict], n: int, seed: int = 0, max_frac: float = 0.1) -> Tuple[List[Dict], List[Dict]]:
    """Split one dataset's train examples into (train, calib) with ``n`` calibration frames.

    Calibration frames come from whole held-out episodes, so they are not near-duplicates of training frames.
    When the episodes are too long for that (holding them out would drop more than ``max_frac`` of the data),
    random frames are held out instead.
    """
    if n <= 0:
        return examples, []
    rng = random.Random(seed)
    per_ep = Counter(ex.get("episode") for ex in examples)
    eps = sorted(per_ep, key=str)
    rng.shuffle(eps)
    held, count = set(), 0
    for e in eps:
        if count >= n:
            break
        held.add(e)
        count += per_ep[e]
    if None not in held and count <= max(n, max_frac * len(examples)) and count < len(examples):
        return ([ex for ex in examples if ex.get("episode") not in held],
                even_subsample([ex for ex in examples if ex.get("episode") in held], n))
    idx = set(rng.sample(range(len(examples)), min(n, len(examples) // 10)))
    return [ex for i, ex in enumerate(examples) if i not in idx], [ex for i, ex in enumerate(examples) if i in idx]


def load_atari(root: str = ATARI_ROOT, sources: Optional[Sequence[str]] = None, games: Optional[Sequence[str]] = None,
               n_calib: int = 60, val_limit: Optional[int] = None, train_limit: Optional[int] = None,
               seed: int = 0, frames: int = 1) -> Dict:
    """``{"datasets", "train", "calib", "val"}`` over every ready (source, game); prints what was found."""
    found = ready_datasets(root, sources, games)
    train, calib, val = [], [], []
    print("%-28s %-12s %8s %6s %6s %6s" % ("source/game", "frame_format", "train", "calib", "val", "soft"))
    for ds in found:
        tr = load_split(ds, "train", train_limit, frames)
        tr, ca = hold_out_episodes(tr, n_calib, seed)
        va = load_split(ds, "val", val_limit, frames)
        n_soft = sum(max(ex["target"]) < 1.0 for ex in tr)
        ds.update(n_train=len(tr), n_calib=len(ca), n_val=len(va), n_soft=n_soft)
        print("%-28s %-12s %8d %6d %6d %6d" % (ds["name"], ds["frame_format"], len(tr), len(ca), len(va), n_soft))
        train += tr
        calib += ca
        val += va
    games_ = sorted({ds["game"] for ds in found})
    print("%d datasets, %d games, %d train / %d calib / %d val frames" % (len(found), len(games_), len(train), len(calib), len(val)))
    return {"datasets": found, "games": games_, "train": train, "calib": calib, "val": val}


# ---------------------------------------------------------------------------------------------------------
# Metrics and calibration
# ---------------------------------------------------------------------------------------------------------


def game_of(name: str) -> str:
    return name.split("/", 1)[-1]


def per_game_metrics(records: List[Dict], temperatures: Sequence[float] = (1.0, 1.0, 1.0)) -> Dict:
    """``metrics_from`` per game (sources pooled) and per source/game; ``mean`` averages acc / ECE / NLL over games."""
    by_game = metrics_from([dict(r, dataset=game_of(r["dataset"])) for r in records], temperatures)
    by_ds = metrics_from(records, temperatures)
    games = sorted(k for k in by_game if k != "all")
    mean = {m: float(np.mean([by_game[g][m] for g in games])) for m in ("acc", "ece", "nll")}
    return {"mean": mean, "all": by_ds["all"], "games": {g: by_game[g] for g in games},
            "datasets": {k: v for k, v in by_ds.items() if k != "all"}}


def outcome_targets(examples: List[Dict], strength: float = 1.0, key: str = "rtg") -> int:
    """Blend each example's target toward the action actually taken, by how good that frame's outcome was.

    This is the game form of the text model's TD(1) target (``laya.common.td_lambda_targets`` only covers the
    two-option case): an episode is a trajectory, ``rtg`` (``return_to_go``) is the realised outcome from that
    frame on, and ``g`` is its percentile among the same game's frames, so the scale of each game's score does
    not matter. The target becomes ``(1 - strength*g) * expert + strength*g * onehot(taken)``: frames that led
    to good outcomes pull toward what was really done, frames that led to bad ones keep the expert's
    distribution. Returns how many examples were changed; examples without ``rtg`` are left alone.
    """
    if strength <= 0:
        return 0
    by_game: Dict[str, List[Dict]] = {}
    for ex in examples:
        if ex.get("rtg") is not None:
            by_game.setdefault(ex["game"], []).append(ex)
    n = 0
    for exs in by_game.values():
        rtgs = np.array([float(ex["rtg"]) for ex in exs])
        ranks = rtgs.argsort().argsort() / max(1, len(rtgs) - 1)  # percentile of the return-to-go
        for ex, g in zip(exs, ranks):
            w = strength * float(g)
            t = [(1 - w) * p for p in ex["target"]]
            taken = int(ex.get("taken", ex["label"]))
            if not 0 <= taken < len(t):
                continue
            t[taken] += w
            total = sum(t)
            ex["target"] = [p / total for p in t]
            n += 1
    return n


def filter_top_episodes(examples: List[Dict], frac: float) -> List[Dict]:
    """Keep only frames from the best episodes per game, by ``episode_score`` (all frames if it is missing)."""
    if not 0 < frac < 1:
        return examples
    keep, by_game = [], {}
    for ex in examples:
        by_game.setdefault(ex["game"], []).append(ex)
    for game, exs in by_game.items():
        scores = {ex["episode"]: ex.get("episode_score") for ex in exs}
        scored = {e: v for e, v in scores.items() if v is not None}
        if not scored:
            keep += exs
            continue
        cutoff = sorted(scored.values())[max(0, int((1 - frac) * len(scored)) - 1)] if len(scored) > 1 else -float("inf")
        good = {e for e, v in scored.items() if v >= cutoff}
        keep += [ex for ex in exs if ex["episode"] in good or scores.get(ex["episode"]) is None]
    return keep


def fit_option_temperatures(records: List[Dict], min_n: int = 200) -> Dict[str, float]:
    """Temperatures per ``temp_bucket`` (question type x option count) with at least ``min_n`` records.

    Atari games have 3 to 18 actions; ``VLMAgent.predict`` prefers these over the per-type temperature.
    """
    groups: Dict[str, List[Dict]] = {}
    for r in records:
        groups.setdefault(temp_bucket(r["qtype"], len(r["logits"])), []).append(r)
    return {b: fit_temperatures_from(rs)[rs[0]["qtype"]] for b, rs in sorted(groups.items()) if len(rs) >= min_n}


def scale_records(records: List[Dict], temperatures: Sequence[float], by_options: Optional[Dict[str, float]] = None) -> List[Dict]:
    """Records with logits divided by the temperature ``VLMAgent.predict`` would use."""
    out = []
    for r in records:
        t = (by_options or {}).get(temp_bucket(r["qtype"], len(r["logits"])), temperatures[r["qtype"]])
        out.append(dict(r, logits=r["logits"] / t))
    return out


# ---------------------------------------------------------------------------------------------------------
# Playing
# ---------------------------------------------------------------------------------------------------------


@torch.no_grad()
def encode_frames(agent: VLMAgent, frames: Sequence[np.ndarray]) -> torch.Tensor:
    """Vision tower + connector over a list of raw RGB frames -> ``[n, image_seq_len, d]``, either path."""
    if agent.prep.on_gpu:
        return agent.model.encode_raw_images(list(frames))
    dev, dtype = agent.device, agent.model.encoder.dtype
    per = [vlm_prefix(agent.processor, [f], agent.prep) for f in frames]
    pv = torch.stack([p["pixel_values"][0] for p in per]).to(dev, dtype)
    pam = torch.stack([p["pixel_attention_mask"][0] for p in per]).to(dev)
    return agent.model.encode_images(pv, pam)


@torch.no_grad()
def action_probs(agent: VLMAgent, frames: Sequence[np.ndarray], question: Dict,
                 prev_frames: Optional[Sequence[np.ndarray]] = None, return_act: bool = False,
                 cache: Optional["FrameFeatureCache"] = None):
    """Calibrated action probabilities (actions order) for several RGB frames in one forward pass.

    Same sequence, option order and temperature as ``agent.predict({"image": frame}, ...)`` with one permutation,
    or ``{"images": [prev, frame]}`` when ``prev_frames`` is given. ``return_act`` also returns the act head's
    P(act) per frame (the decide-or-escalate gate).

    With a ``cache`` the vision tower runs only on frames it has not seen: in two-frame play this step's current
    frame is next step's previous frame, so the image encoder does half the work. The language model still sees
    the same features, so the answer is unchanged (``tests/test_vlm.py`` checks that).
    """
    q = VLMAgent._to_internal(question)
    k = len(render_options(q))
    items = []
    for j, fr in enumerate(frames):
        state = {"image": fr} if prev_frames is None else {"images": [prev_frames[j], fr]}
        it = build_vlm_inputs(agent.processor, state, q, agent.cfg.get("max_len", 1024), agent.cfg.get("head_max_len", 256))
        it["qtype"] = QTYPES["choice"]
        items.append(it)
    b = collate_vlm(items, agent.processor.tokenizer.pad_token_id)
    dev, dtype = agent.device, agent.model.encoder.dtype
    if cache is not None:
        flat = [f for j in range(len(frames)) for f in ((frames[j],) if prev_frames is None
                                                        else (prev_frames[j], frames[j]))]
        pix = dict(image_hidden_states=torch.stack(cache.features(lambda fs: encode_frames(agent, fs), flat)))
    elif b["pixel_values"] is not None:
        pix = dict(pixel_values=b["pixel_values"].to(dev, dtype), pixel_attention_mask=b["pixel_attention_mask"].to(dev))
    else:
        pix = dict(raw_pixels=b["raw_pixels"].to(dev), image_mask=b["image_mask"].to(dev))
    logits, act = agent.model(
        b["input_ids"].to(dev), b["attention_mask"].to(dev), b["marker_pos"].to(dev), b["marker_mask"].to(dev),
        b["qtype"].to(dev), option_span=b["option_span"].to(dev), **pix,
    )
    t = agent.temperature_by_options.get(temp_bucket(QTYPES["choice"], k), agent.temperature[QTYPES["choice"]])
    p = torch.softmax(logits[:, :k].float() / max(1e-3, float(t)), -1).cpu().numpy()
    return (p, torch.softmax(act.float(), -1)[:, 0].cpu().numpy()) if return_act else p


def play(game: str, policy: Callable[[List[np.ndarray], List[np.ndarray], List[int]], Sequence[int]], episodes: int = 3,
         max_steps: int = 4500, seed: int = 0, auto_fire: bool = True) -> Dict:
    """Play ``episodes`` of ``ALE/<game>-v5`` (default settings) in lockstep and report the scores.

    ``policy(frames, prevs, ids)`` gets the raw RGB observations of the episodes still running, each one's
    observation at the previous decision step, and their episode indices, and returns an action index for each. As in
    ``examples/atari_live.py``, FIRE is pressed on reset and after each lost life (not counted as agent steps).
    The previous frame is a copy of the current one on an episode's first step and on the first step after an
    auto-FIRE (the ``expert2f`` data rule). Episodes stop at game over or after ``max_steps`` agent steps.
    """
    import ale_py
    import gymnasium as gym

    gym.register_envs(ale_py)
    envs = [gym.make("ALE/%s-v5" % game) for _ in range(episodes)]
    actions = envs[0].unwrapped.get_action_meanings()
    fire = actions.index("FIRE") if auto_fire and "FIRE" in actions else None
    obs, lives, score, steps, done, capped = [], [], [0.0] * episodes, [0] * episodes, [False] * episodes, [False] * episodes
    for i, env in enumerate(envs):
        o, info = env.reset(seed=seed + i)
        if fire is not None:
            o, r, _, _, info = env.step(fire)
            score[i] += r
        obs.append(o)
        lives.append(info.get("lives", 0))
    prev = list(obs)
    counts = Counter()
    while not all(done):
        live = [i for i in range(episodes) if not done[i]]
        for i, a in zip(live, policy([obs[i] for i in live], [prev[i] for i in live], live)):
            prev[i] = obs[i]
            o, r, term, trunc, info = envs[i].step(int(a))
            counts[actions[int(a)]] += 1
            score[i] += r
            steps[i] += 1
            if fire is not None and not (term or trunc) and info.get("lives", lives[i]) < lives[i]:
                o, r, term, trunc, info = envs[i].step(fire)
                score[i] += r
                prev[i] = o
            lives[i] = info.get("lives", lives[i])
            obs[i] = o
            if term or trunc or steps[i] >= max_steps:
                done[i], capped[i] = True, not (term or trunc)
    for env in envs:
        env.close()
    return {"game": game, "episodes": episodes, "scores": score, "mean_score": float(np.mean(score)),
            "steps": steps, "capped": sum(capped), "actions": dict(counts), "action_names": actions}


def model_policy(agent: VLMAgent, game: str, actions: Sequence[str], sample: bool = False, seed: int = 0,
                 frames: int = 1, gate: bool = False, stats: Optional[Dict] = None,
                 cache_features: Optional[bool] = None, cuda_graph: bool = False) -> Callable:
    """Greedy (the most likely action, as ``predict``'s ``choice``) or sampled from the calibrated probabilities.

    ``frames=2`` gives the model ``[previous, current]``, and reuses the encoder output each frame already earned
    as the step before's current frame (see ``FrameFeatureCache``). With ``gate``, the act head decides: when it
    says escalate rather than act, the previous action is repeated instead of taking the model's choice
    (``stats``, if given, counts the gated steps).

    ``cache_features`` defaults to on for two frames on the device-side path and off otherwise. It is a loss on the
    Hugging Face processor path: caching means preprocessing frames one at a time, and the processor costs ~33 ms
    of CPU per frame either way, so the lost batching outweighs the halved encoder work (measured on an L4:
    11.2 decisions/s uncached against 8.7 cached, versus 38.4 against 53.1 on the device-side path).

    ``cuda_graph`` runs each decision as one captured CUDA graph (``laya.static_step.StaticStep``, one per number
    of episodes still running) instead of eager ``action_probs``; it replaces the feature cache. See
    docs/game-caching.md.
    """
    q = atari_question(game, actions)["action"]
    rng = np.random.default_rng(seed)
    last: Dict[int, int] = {}
    if cache_features is None:
        cache_features = agent.prep.on_gpu
    cache = FrameFeatureCache() if frames == 2 and cache_features and not cuda_graph else None
    steps: Dict[int, "StaticStep"] = {}

    def pick(row):
        return int(row.argmax()) if not sample else int(rng.choice(len(row), p=row / row.sum()))

    def policy(obs, prevs, ids=None):
        ids = list(range(len(obs))) if ids is None else ids
        if cuda_graph:
            if len(obs) not in steps:
                from .static_step import StaticStep

                steps[len(obs)] = StaticStep(agent, q, frames=frames, batch=len(obs))
            out = steps[len(obs)].probs(obs, prevs if frames == 2 else None, return_act=gate)
        else:
            out = action_probs(agent, obs, q, prevs if frames == 2 else None, return_act=gate, cache=cache)
        p, act = out if gate else (out, None)
        acts = []
        for j, (row, i) in enumerate(zip(p, ids)):
            a = pick(row)
            if gate and act[j] < 0.5 and i in last:  # escalate: hold this episode's previous action
                a = last[i]
                if stats is not None:
                    stats["gated"] = stats.get("gated", 0) + 1
            last[i] = a
            acts.append(a)
        if stats is not None:
            stats["steps"] = stats.get("steps", 0) + len(acts)
            if gate:
                stats["act_p_sum"] = stats.get("act_p_sum", 0.0) + float(act.sum())
        return acts

    return policy


def random_policy(n_actions: int, seed: int = 0) -> Callable:
    rng = random.Random(seed)
    return lambda frames, prevs=None, ids=None: [rng.randrange(n_actions) for _ in frames]


def game_actions(game: str) -> List[str]:
    import ale_py
    import gymnasium as gym

    gym.register_envs(ale_py)
    env = gym.make("ALE/%s-v5" % game)
    actions = env.unwrapped.get_action_meanings()
    env.close()
    return actions


def play_check(agent: VLMAgent, games: Sequence[str], baselines: Dict[str, Dict], episodes: int = 3,
               max_steps: int = 400, frames: int = 1, seed: int = 500_000) -> Dict:
    """Short play check for in-run checkpoint selection: mean normalised score over a few games.

    Frame accuracy has not predicted play strength, so training can select on this instead. ``baselines`` maps a
    game to ``{"expert", "random"}`` (the capped baselines); a game without them is scored raw and skipped in the
    mean. The episodes are much shorter than a real evaluation, so this is a noisy proxy.
    """
    out, norms = {}, []
    for g in games:
        actions = game_actions(g)
        res = play(g, model_policy(agent, g, actions, False, seed, frames), episodes, max_steps, seed)
        b = baselines.get(g) or {}
        n = None
        if b.get("expert") is not None and b.get("random") is not None and b["expert"] != b["random"]:
            n = (res["mean_score"] - b["random"]) / (b["expert"] - b["random"])
            norms.append(n)
        out[g] = {"score": res["mean_score"], "normalized": n}
    out["mean_normalized"] = float(np.mean(norms)) if norms else float("nan")
    return out


# ---------------------------------------------------------------------------------------------------------
# Synthetic data for pipeline tests
# ---------------------------------------------------------------------------------------------------------


def write_synthetic(root: str, games: Sequence[str] = ("Breakout", "Pong"), sources: Sequence[str] = ("expert", "jat"),
                    n_train: int = 48, n_val: int = 12, seed: int = 0, force: bool = False) -> List[str]:
    """Write a tiny dataset in the shared layout from random ALE play. ``expert`` gets full-colour frames with
    soft targets and baseline scores, other sources 84x84 grayscale frames with hard labels. Every record also has
    a ``prev_image`` (a copy of ``image`` on an episode's first step). Labels are random. A game that already has
    ``_READY`` is skipped unless ``force``."""
    import ale_py
    import gymnasium as gym
    from PIL import Image

    gym.register_envs(ale_py)
    rng = random.Random(seed)
    written = []
    for source in sources:
        for game in games:
            d = os.path.join(root, source, game)
            if os.path.exists(os.path.join(d, "_READY")) and not force:
                written.append("%s/%s (already ready)" % (source, game))
                continue
            os.makedirs(os.path.join(d, "images"), exist_ok=True)
            env = gym.make("ALE/%s-v5" % game)
            actions = env.unwrapped.get_action_meanings()
            q = atari_question(game, actions)["action"]
            rgb = source == "expert"
            meta = {"source": source, "game": game, "frame_format": "rgb_210x160" if rgb else "gray_84x84",
                    "actions": actions, "origin": "synthetic random play (pipeline test)"}
            for split, n, ep0 in (("train", n_train, 0), ("val", n_val, 1000)):
                with open(os.path.join(d, split + ".jsonl"), "w") as f:
                    t, ep = 0, ep0
                    obs, _ = env.reset(seed=seed + ep)
                    prev = obs
                    for s in range(n):
                        a = rng.randrange(len(actions))
                        rid = "%s-%s-e%06d-s%06d" % (source, game, ep, t)
                        img = Image.fromarray(obs) if rgb else Image.fromarray(obs).convert("L").resize((84, 84))
                        img.save(os.path.join(d, "images", rid + ".png"))
                        pimg = Image.fromarray(prev) if rgb else Image.fromarray(prev).convert("L").resize((84, 84))
                        pimg.save(os.path.join(d, "images", rid + "_prev.png"))
                        rec = {"id": rid, "image": "images/%s.png" % rid, "prev_image": "images/%s_prev.png" % rid,
                               "game": game, "actions": actions, "label": a,
                               "question": q, "source": source, "episode": ep, "step": t}
                        if rgb:
                            p = np.full(len(actions), 0.2 / max(1, len(actions) - 1))
                            p[a] = 0.8
                            rec["target"] = p.tolist()
                        f.write(json.dumps(rec) + "\n")
                        prev = obs
                        obs, _, term, trunc, _ = env.step(a)
                        t += 1
                        if term or trunc or (s + 1) % (n // 4 or 1) == 0:  # several short episodes per split
                            ep, t = ep + 1, 0
                            obs, _ = env.reset(seed=seed + ep)
                            prev = obs
                meta[split] = {"records": n}
            if rgb:
                meta.update(expert_score=10.0, random_score=1.0)
            env.close()
            with open(os.path.join(d, "meta.json"), "w") as f:
                json.dump(meta, f, indent=2)
            open(os.path.join(d, "_READY"), "w").close()
            written.append("%s/%s" % (source, game))
    return written


def main(argv=None):
    from .vlm_train import collect_logits, train

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", required=True, help="directory for the synthetic dataset")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--frames", type=int, choices=[1, 2], default=1)
    args = ap.parse_args(argv)
    write_synthetic(args.smoke, n_train=16, n_val=4)
    data = load_atari(args.smoke, n_calib=4, frames=args.frames)
    agent = VLMAgent(backbone="HuggingFaceTB/SmolVLM-256M-Instruct", device=args.device)
    losses = train(agent.model, agent.processor, [dict(ex, dataset=ex["game"]) for ex in data["train"]], steps=2,
                   batch_size=2, freeze="full", device=args.device)
    print("losses", losses)
    recs = collect_logits(agent.model, agent.processor, data["val"], batch_size=4)
    print("val", json.dumps(per_game_metrics(recs)["mean"]))
    print("option temps", fit_option_temperatures(collect_logits(agent.model, agent.processor, data["calib"]), min_n=4))
    actions = game_actions("Breakout")
    print(play("Breakout", model_policy(agent, "Breakout", actions, frames=args.frames), episodes=2, max_steps=3))
    print(play("Breakout", random_policy(len(actions)), episodes=2, max_steps=50))


if __name__ == "__main__":
    main()
