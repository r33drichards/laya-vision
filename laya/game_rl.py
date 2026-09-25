"""Reinforcement learning from game rewards: GRPO over the games, the game's own score as the reward.

Imitation (``toolkit`` expert frames) caps the model at its expert and only ever shows it the expert's states; the
model then drifts off them (LunarLander agrees with its expert on 70% of expert states but 7% of its own). Here the
model plays, and the score it gets is the training signal: RL with verifiable rewards (arXiv 2506.14245) in the GRPO
form (DeepSeekMath, arXiv 2402.03300), with no expert and no value network.

One RL phase (``GameRL.phase``):

1. **Rollout.** For each game, ``episodes_per_phase / group`` training seeds are drawn from
   ``[seed_lo, TRAIN_SEED_MAX)`` (below 100,000; the benchmark plays 700,000+). Each seed is played ``group`` times
   (G episodes, one *group*), all episodes of all games in lockstep: every round is one batched forward over every
   live episode (``ModelPolicy.act``). Each episode keeps its own frame history (``laya.frames.History``) and its
   state is ``laya.frames.state`` of it in the checkpoint's frame mode (``mode_for(cfg, family)``), exactly as
   ``games_eval`` builds it; the vision tower runs once per distinct frame (``FrameFeatureCache``, as ``batched_probs``
   with a cache does) and the language model reads the cached features. The action is *sampled* from the model's
   calibrated option distribution divided by ``temperature``: ``softmax(z / (T_cal * temperature))``. Per step the
   trainer keeps the image features (what the update needs to rebuild the input: the text ids depend only on the game
   and the image count), the action and its log-probability.
2. **Advantage** (``advantages``), group-relative. ``returns="episode"``: every step of episode i gets
   ``(R_i - mean_group R) / std_group R`` (GRPO's outcome supervision). ``returns="togo"``: step t of episode i gets
   ``G_{i,t} - mean_j G_{j,t}`` with ``G`` the discounted return-to-go (0 once an episode has ended: an episode that
   crashed has no more reward to come, one that reached the goal no more cost), divided by the RMS of those centred
   values over the group; the baseline is the group's own return-to-go at the same step, which the shared seed makes a
   fair comparison. A group whose episodes all score the same carries no signal (advantage 0).
3. **Update** (``pg_loss``): the PPO clipped surrogate on the log-probability of the action taken under the *same*
   distribution it was sampled from (calibrated temperature times ``temperature``), so the gradient is the on-policy
   gradient of the policy that played; the calibrated temperature is a constant, so it only rescales the logits'
   gradient and never changes which action is greedy. The loss averages over steps (every step one vote, as Dr. GRPO /
   DAPO do, rather than GRPO's per-episode length normalisation). The ratio is 1 on the first minibatch of the first
   epoch; clipping matters for later minibatches and ``epochs > 1``. ``kl`` adds the exact KL(pi || pi_start) over the
   options toward a frozen copy of the starting model, ``entropy`` an entropy bonus. Gradients reach the language model
   and the head; the image features are the rollout's (the vision tower is frozen in every recipe; the connector then
   learns from the supervised batches only). The next-move and value heads are not used.

Reward normalisation: none beyond the group's. Group normalisation divides out each game's scale (CartPole's +1 per
step, LunarLander's hundreds, Snake's food count), which is what makes the games comparable; an extra per-game affine
map (the benchmark's random / expert baselines) cancels exactly under it for ``returns="episode"`` and would only shift
return-to-go by a step-count term. The baselines, if given, are used for logging only.

Interleaving: ``laya.vlm_train.train(..., step_hook=rl.hook)`` calls the hook after every supervised step; the hook runs
a phase every ``every`` steps (``every >= 1``) or whenever RL has had less than a share ``every`` (``0 < every < 1``)
of the wall clock. The RL optimizer is its own AdamW over the same trainable parameters (so the RL gradients, whose
scale and noise differ, do not disturb the supervised optimizer's moment estimates), with its learning rate ``lr``
times the supervised schedule's current factor (warmup, then cosine), so both anneal together. RL time counts against
``train``'s time budget; a phase is skipped when the previous one would not fit in what is left, and a rollout is cut
at the deadline. ``GameRL.run(minutes)`` is the RL-only loop.

Games (``RL_GAMES``): CartPole, Acrobot, MountainCar, LunarLander (Gymnasium), Maze4, Maze6, Snake10 (pure Python),
with the benchmark's step caps, plus Atari (``Freeway``, ``Breakout``; ``ALE/<game>-v5`` with ``laya.atari_train.play``'s
auto-FIRE and history reset, cap 1,000) when ``ale-py`` is installed. Rewards: the environment's reward; Maze 1 on
reaching the goal; Snake 1 per food.
"""
import copy
import random
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from . import frames as F

TRAIN_SEED_MAX = 100_000   # every training seed stays below this; the games benchmark plays 700,000+
FORWARD_BATCH = 64         # sequences per forward in rollouts


@dataclass(frozen=True)
class RLGame:
    """A game RL can play. ``cap``: the most agent steps an episode may take (``None``: the game's own limit)."""
    name: str
    family: str
    cap: Optional[int]
    params: Dict = field(default_factory=dict)


# the games benchmark's games with its caps (``autoresearch/games_eval.py`` SUITE; the tests check they agree)
GAMES: Dict[str, RLGame] = {g.name: g for g in (
    RLGame("Maze4", "grid", None, {"game": "maze", "size": 4}),
    RLGame("Maze6", "grid", None, {"game": "maze", "size": 6}),
    RLGame("Snake10", "grid", 200, {"game": "snake", "size": 10}),
    RLGame("CartPole", "control", 200, {"game": "CartPole"}),
    RLGame("Acrobot", "control", 200, {"game": "Acrobot"}),
    RLGame("MountainCar", "control", 200, {"game": "MountainCar"}),
    RLGame("LunarLander", "control", 300, {"game": "LunarLander"}),
    RLGame("Freeway", "atari", 1000, {"game": "Freeway"}),
    RLGame("Breakout", "atari", 1000, {"game": "Breakout"}),
)}


def check_seed(seed: int) -> int:
    if not 0 <= seed < TRAIN_SEED_MAX:
        raise ValueError("RL training seed %d is outside [0, %d): the benchmark's seeds must never be trained on"
                         % (seed, TRAIN_SEED_MAX))
    return seed


# ---------------------------------------------------------------------------------------------------------------
# Environments: actions, frame(), step(name) -> reward, done, score, steps, close()
# ---------------------------------------------------------------------------------------------------------------


class ControlEnv:
    family = "control"

    def __init__(self, game: str, seed: int, cap: Optional[int]):
        from .controlgames import ControlGame

        self.game, self.cap = ControlGame(game, seed), cap
        self.actions = tuple(self.game.actions)

    @property
    def done(self) -> bool:
        return self.game.done or (self.cap is not None and self.game.steps >= self.cap)

    @property
    def score(self) -> float:
        return self.game.score

    @property
    def steps(self) -> int:
        return self.game.steps

    def frame(self) -> np.ndarray:
        return self.game.frame()

    def step(self, action: str) -> float:
        return self.game.step(action)

    def close(self) -> None:
        self.game.close()


class GridEnv:
    """Maze (reward 1 when the goal is reached) or Snake (1 per food), the benchmark's score as a reward."""
    family = "grid"

    def __init__(self, game: str, size: int, seed: int, cap: Optional[int]):
        from .gridgames import ACTIONS, make_game

        self.env, self.actions = make_game(game, size, seed, cap or 0), tuple(ACTIONS)

    @property
    def done(self) -> bool:
        return self.env.done

    @property
    def score(self) -> float:
        return float(self.env.solved) if hasattr(self.env, "solved") else float(self.env.eaten)

    @property
    def steps(self) -> int:
        return self.env.steps

    def frame(self) -> np.ndarray:
        return np.asarray(self.env.render())

    def step(self, action: str) -> float:
        before = self.score
        self.env.step(action)
        return self.score - before

    def close(self) -> None:
        pass


class AtariEnv:
    """One ``ALE/<game>-v5`` episode as ``laya.atari_train.play`` plays it: FIRE on reset and after a lost life (not
    agent steps; their reward counts), the frame history restarting after an auto-FIRE (``reset_history``)."""
    family = "atari"

    def __init__(self, game: str, seed: int, cap: Optional[int]):
        import ale_py
        import gymnasium as gym

        gym.register_envs(ale_py)
        self.env, self.cap = gym.make("ALE/%s-v5" % game), cap
        meanings = self.env.unwrapped.get_action_meanings()
        self.actions = tuple(meanings)
        self.fire = meanings.index("FIRE") if "FIRE" in meanings else None
        self.obs, info = self.env.reset(seed=seed)
        self.score, self.steps, self.over, self.reset_history = 0.0, 0, False, False
        if self.fire is not None:
            self.obs, r, _, _, info = self.env.step(self.fire)
            self.score += float(r)
        self.lives = info.get("lives", 0)

    @property
    def done(self) -> bool:
        return self.over or (self.cap is not None and self.steps >= self.cap)

    def frame(self) -> np.ndarray:
        return self.obs

    def step(self, action: str) -> float:
        o, r, term, trunc, info = self.env.step(self.actions.index(action))
        reward = float(r)
        self.steps += 1
        if self.fire is not None and not (term or trunc) and info.get("lives", self.lives) < self.lives:
            o, r, term, trunc, info = self.env.step(self.fire)
            reward += float(r)
            self.reset_history = True
        self.lives = info.get("lives", self.lives)
        self.obs, self.over = o, bool(term or trunc)
        self.score += reward
        return reward

    def close(self) -> None:
        self.env.close()


def make_env(name: str, seed: int, cap: Any = "default"):
    """Episode ``seed`` of game ``name`` (a ``GAMES`` key); ``cap`` overrides the game's step cap."""
    g = GAMES[name]
    check_seed(seed)
    cap = g.cap if cap == "default" else cap
    if g.family == "control":
        return ControlEnv(g.params["game"], seed, cap)
    if g.family == "grid":
        return GridEnv(g.params["game"], g.params["size"], seed, cap)
    if g.family == "atari":
        return AtariEnv(g.params["game"], seed, cap)
    raise ValueError("no RL environment for %s" % name)


def question_for(name: str, mode: str = "single") -> Dict:
    """The benchmark's question for ``name`` (``games_eval.question_for`` / ``atari_model_policy``)."""
    from .games import atari_question, control_question, maze_question, snake_question

    g = GAMES[name]
    if g.family == "control":
        return control_question(g.params["game"], mode)
    if g.family == "grid":
        return maze_question() if g.params["game"] == "maze" else snake_question()
    if g.family == "atari":
        from .atari_train import game_actions

        return atari_question(g.params["game"], game_actions(g.params["game"]))
    raise ValueError(name)


# ---------------------------------------------------------------------------------------------------------------
# Rollouts
# ---------------------------------------------------------------------------------------------------------------


@dataclass
class Episode:
    game: str
    seed: int
    group: int
    actions: List[int] = field(default_factory=list)
    logps: List[float] = field(default_factory=list)
    inputs: List[Any] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    score: float = 0.0
    steps: int = 0
    cut: bool = False     # stopped by the time budget before the game ended

    @property
    def ret(self) -> float:
        return float(sum(self.rewards))


def rollout(policy, groups: Sequence[Tuple[str, int]], group_size: int, modes: Dict[str, str],
            make: Callable = make_env, deadline: Optional[float] = None, families: Optional[Dict[str, str]] = None
            ) -> List[Episode]:
    """Play ``group_size`` episodes of every ``(game, seed)`` in ``groups``, all in lockstep.

    Each round every live episode pushes its current frame to its own history, its state is built in its game's mode
    (``modes[game]``), and ``policy.act([(game, images), ...]) -> (actions, logps, inputs)`` answers all of them in one
    call; every live episode then steps once. An env that sets ``reset_history`` (Atari after an auto-FIRE) restarts
    its history. At ``deadline`` (``time.time()``) the remaining episodes stop and are marked ``cut``."""
    envs, eps, hists = [], [], []
    for gi, (name, seed) in enumerate(groups):
        check_seed(seed)
        for _ in range(group_size):
            env = make(name, seed)
            fam = (families or {}).get(name) or env.family
            envs.append((env, fam))
            eps.append(Episode(name, seed, gi))
            hists.append(F.History(max(1, F.frames_needed(modes[name], fam))))
    while True:
        live = [i for i, (e, _) in enumerate(envs) if not e.done]
        if not live:
            break
        if deadline is not None and time.time() >= deadline:
            for i in live:
                eps[i].cut = True
            break
        states = []
        for i in live:
            env, fam = envs[i]
            if getattr(env, "reset_history", False):
                hists[i].reset()
                env.reset_history = False
            hists[i].push(env.frame())
            st = F.state(hists[i].frames, modes[eps[i].game], fam)
            states.append((eps[i].game, [np.asarray(x) for x in F.state_images(st)]))
        acts, logps, inputs = policy.act(states)
        for i, a, lp, x in zip(live, acts, logps, inputs):
            env = envs[i][0]
            r = env.step(env.actions[int(a)])
            ep = eps[i]
            ep.actions.append(int(a))
            ep.logps.append(float(lp))
            ep.inputs.append(x)
            ep.rewards.append(float(r))
    for ep, (env, _) in zip(eps, envs):
        ep.score, ep.steps = float(env.score), int(env.steps)
        if hasattr(env, "close"):
            env.close()
    return eps


# ---------------------------------------------------------------------------------------------------------------
# Advantages and loss
# ---------------------------------------------------------------------------------------------------------------


def group_advantages(returns: Sequence[float], eps: float = 1e-6) -> np.ndarray:
    """GRPO's outcome advantage: ``(R - mean R) / std R`` over one group; zeros when every return is the same."""
    r = np.asarray(returns, np.float64)
    s = r.std()
    return np.zeros_like(r) if s < eps else (r - r.mean()) / s


def returns_to_go(rewards: Sequence[float], gamma: float) -> np.ndarray:
    out, acc = np.zeros(len(rewards)), 0.0
    for t in range(len(rewards) - 1, -1, -1):
        acc = rewards[t] + gamma * acc
        out[t] = acc
    return out


def togo_advantages(rewards: Sequence[Sequence[float]], gamma: float, eps: float = 1e-6) -> List[np.ndarray]:
    """Per-step advantages for one group: ``G_{i,t} - mean_j G_{j,t}`` (return-to-go, 0 after an episode's end),
    divided by the RMS of the centred values over every real step of the group; zeros when that is 0."""
    n, T = len(rewards), max((len(r) for r in rewards), default=0)
    G = np.zeros((n, T))
    valid = np.zeros((n, T), bool)
    for i, r in enumerate(rewards):
        G[i, :len(r)] = returns_to_go(r, gamma)
        valid[i, :len(r)] = True
    C = G - G.mean(0, keepdims=True)
    s = float(np.sqrt((C[valid] ** 2).mean())) if valid.any() else 0.0
    if s < eps:
        return [np.zeros(len(r)) for r in rewards]
    return [C[i, :len(r)] / s for i, r in enumerate(rewards)]


def advantages(episodes: Sequence[Episode], returns: str = "episode", gamma: float = 0.99) -> List[np.ndarray]:
    """One advantage array per episode (a value per step), normalised within each ``group``."""
    if returns not in ("episode", "togo"):
        raise ValueError("returns must be 'episode' or 'togo', not %r" % returns)
    out: List[Optional[np.ndarray]] = [None] * len(episodes)
    by_group: Dict[Any, List[int]] = {}
    for i, ep in enumerate(episodes):
        by_group.setdefault((ep.game, ep.group), []).append(i)
    for idx in by_group.values():
        if returns == "episode":
            a = group_advantages([episodes[i].ret for i in idx])
            for i, v in zip(idx, a):
                out[i] = np.full(len(episodes[i].rewards), v)
        else:
            for i, v in zip(idx, togo_advantages([episodes[i].rewards for i in idx], gamma)):
                out[i] = v
    return out


def pg_loss(logits: torch.Tensor, actions: torch.Tensor, adv: torch.Tensor, old_logp: torch.Tensor,
            clip: float = 0.2, ref_logits: Optional[torch.Tensor] = None, kl: float = 0.0, entropy: float = 0.0):
    """PPO-clip policy gradient over option logits ``[B, K]`` (already divided by the sampling temperature; masked
    options at -1e4-ish). Returns ``(loss, stats)``. ``kl`` weights the exact KL(pi || pi_ref) over the options,
    ``entropy`` an entropy bonus."""
    logp_all = torch.log_softmax(logits.float(), -1)
    logp = logp_all.gather(1, actions[:, None]).squeeze(1)
    ratio = torch.exp(logp - old_logp)
    surr = torch.minimum(ratio * adv, ratio.clamp(1 - clip, 1 + clip) * adv)
    loss = -surr.mean()
    p = logp_all.exp()
    ent = -(p * logp_all).sum(-1).mean()
    stats = {"entropy": float(ent.detach()), "clip_frac": float(((ratio - 1).abs() > clip).float().mean()),
             "ratio_dev": float((ratio.detach() - 1).abs().mean())}
    if entropy:
        loss = loss - entropy * ent
    if ref_logits is not None:
        ref = torch.log_softmax(ref_logits.float(), -1)
        kl_ref = (p * (logp_all - ref)).sum(-1).mean()
        stats["kl_start"] = float(kl_ref.detach())
        if kl:
            loss = loss + kl * kl_ref
    return loss, stats


# ---------------------------------------------------------------------------------------------------------------
# The model as a policy
# ---------------------------------------------------------------------------------------------------------------


class ModelPolicy:
    """A ``VLMAgent`` as a sampling policy over the games, with batched forwards on cached image features.

    ``act`` encodes each distinct frame once (``FrameFeatureCache``, reset per rollout by ``begin``), builds each game's
    text ids once (they depend only on the question and the image count, ``laya.preprocess.prefix_ids``; the same
    sequence ``games_eval.batched_probs`` runs with a cache) and samples from ``softmax(z / (T_cal * temperature))``.
    ``scaled_logits(inputs)`` is that same forward, with gradients, for the update. Needs ``image_split_edge`` 0 (one
    view per image), as stacked play does."""

    def __init__(self, agent, games: Sequence[str], modes: Dict[str, str], temperature: float = 1.0,
                 seed: int = 0, amp: Optional[bool] = None):
        from .common import QTYPES, render_options, temp_bucket
        from .vlm import VLMAgent

        if agent.prep.split_edge:
            raise ValueError("RL on cached image features needs one view per image (image_split_edge 0)")
        self.agent, self.modes, self.temperature = agent, dict(modes), float(temperature)
        self.q = {g: VLMAgent._to_internal(question_for(g, modes[g])["action"]) for g in games}
        self.k = {g: len(render_options(self.q[g])) for g in games}
        qt = QTYPES["choice"]
        self.t_cal = {g: float(agent.temperature_by_options.get(temp_bucket(qt, self.k[g]), agent.temperature[qt]))
                      for g in games}
        self.keep = max(F.images_per_state(modes[g], GAMES[g].family) for g in games)
        self.amp = agent.device.type == "cuda" if amp is None else amp
        self.gen = torch.Generator().manual_seed(seed)
        self._items: Dict[Tuple[str, int], Dict] = {}
        self._pool: Optional[ThreadPoolExecutor] = None
        self.cache = None

    def begin(self) -> None:
        """A new rollout: the weights may have changed, so no cached feature carries over."""
        from .preprocess import FrameFeatureCache

        self.cache = FrameFeatureCache(keep=self.keep)

    def _item(self, game: str, n_images: int) -> Dict:
        key = (game, n_images)
        if key not in self._items:
            from .common import QTYPES
            from .preprocess import prefix_ids
            from .vlm import MASK_PREFIX_TEXT, PREFIX_TEXT, build_vlm_inputs, processor_readout

            a = self.agent
            text = MASK_PREFIX_TEXT if processor_readout(a.processor) == "mask" else PREFIX_TEXT
            prefix = {"ids": prefix_ids(a.processor, text, n_images, a.prep.image_seq_len), "pixel_values": None,
                      "pixel_attention_mask": None, "raw_images": None, "n_images": n_images}
            it = build_vlm_inputs(a.processor, {}, self.q[game], a.cfg.get("max_len", 1024),
                                  a.cfg.get("head_max_len", 256), prefix=prefix)
            it["qtype"] = QTYPES["choice"]
            self._items[key] = it
        return self._items[key]

    @torch.no_grad()
    def _encode(self, frames: Sequence[np.ndarray]) -> torch.Tensor:
        """Vision tower + connector (``laya.atari_train.encode_frames``), the processor's CPU work on a thread pool."""
        from .vlm import vlm_prefix

        a = self.agent
        if a.prep.on_gpu:  # one batch per frame size (the games' screens differ)
            by_shape: Dict[Tuple, List[int]] = {}
            for i, f in enumerate(frames):
                by_shape.setdefault(np.shape(f), []).append(i)
            out: List[Any] = [None] * len(frames)
            for idx in by_shape.values():
                for i, x in zip(idx, a.model.encode_raw_images([frames[i] for i in idx])):
                    out[i] = x
            return torch.stack(out)
        if self._pool is None:
            import os

            self._pool = ThreadPoolExecutor(max_workers=max(1, min(16, os.cpu_count() or 1)))
        per = list(self._pool.map(lambda f: vlm_prefix(a.processor, [f], a.prep), frames))
        pv = torch.stack([p["pixel_values"][0] for p in per]).to(a.device, a.model.encoder.dtype)
        pam = torch.stack([p["pixel_attention_mask"][0] for p in per]).to(a.device)
        return a.model.encode_images(pv, pam)

    def scaled_logits(self, inputs: Sequence[Tuple[str, Tuple[torch.Tensor, ...]]], model=None) -> torch.Tensor:
        """Option logits ``[B, kmax]`` divided by each game's ``T_cal * temperature``; ``inputs`` as ``act`` returns."""
        from .vlm import collate_vlm

        model = model if model is not None else self.agent.model
        dev = self.agent.device
        items = [self._item(g, len(fs)) for g, fs in inputs]
        b = collate_vlm(items, self.agent.processor.tokenizer.pad_token_id, with_pixels=False)
        feats = torch.stack([f for _, fs in inputs for f in fs])
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=self.amp):
            logits, _ = model(b["input_ids"].to(dev), b["attention_mask"].to(dev), b["marker_pos"].to(dev),
                              b["marker_mask"].to(dev), b["qtype"].to(dev), image_hidden_states=feats,
                              option_span=b["option_span"].to(dev))
        scale = torch.tensor([1.0 / max(1e-3, self.t_cal[g] * self.temperature) for g, _ in inputs], device=dev)
        return logits.float() * scale[:, None]

    @torch.no_grad()
    def act(self, states: Sequence[Tuple[str, Sequence[np.ndarray]]]):
        if self.cache is None:
            self.begin()
        flat = [f for _, imgs in states for f in imgs]
        feats = self.cache.features(self._encode, flat)
        inputs, off = [], 0
        for g, imgs in states:
            inputs.append((g, tuple(feats[off:off + len(imgs)])))
            off += len(imgs)
        acts, logps = [], []
        for s in range(0, len(inputs), FORWARD_BATCH):
            z = self.scaled_logits(inputs[s:s + FORWARD_BATCH]).cpu()
            logp = torch.log_softmax(z, -1)
            a = torch.multinomial(logp.exp(), 1, generator=self.gen).squeeze(1)
            acts += a.tolist()
            logps += logp.gather(1, a[:, None]).squeeze(1).tolist()
        return acts, logps, inputs

    def end(self) -> None:
        self.cache = None


# ---------------------------------------------------------------------------------------------------------------
# The trainer
# ---------------------------------------------------------------------------------------------------------------


@dataclass
class RLConfig:
    games: Tuple[str, ...] = ("CartPole", "Acrobot", "MountainCar", "LunarLander", "Maze4", "Snake10")
    group: int = 8                 # G: episodes per seed
    episodes_per_phase: int = 16   # per game per phase (rounded to whole groups, at least one)
    temperature: float = 1.0       # sampling temperature on top of the calibrated one
    lr: float = 2e-6               # times the supervised schedule's factor when run as a train() hook
    kl: float = 0.0                # KL(pi || pi_start) weight (a frozen copy of the starting model when > 0)
    entropy: float = 0.0           # entropy bonus
    returns: str = "episode"       # "episode" (GRPO outcome) or "togo" (discounted return-to-go)
    gamma: float = 0.99            # for "togo"
    clip: float = 0.2              # PPO ratio clip
    epochs: int = 1                # passes over a phase's steps
    minibatch: int = 64            # steps per optimizer step
    max_update_steps: int = 0      # subsample a phase's steps to at most this many (0 = all)
    every: float = 200             # >= 1: supervised steps between phases; in (0, 1): RL's share of the wall clock
    seed: int = 0
    seed_lo: int = 10_000          # training seeds are drawn from [seed_lo, TRAIN_SEED_MAX)
    caps: Dict[str, Optional[int]] = field(default_factory=dict)   # per-game step cap overrides


class GameRL:
    """GRPO phases over ``cfg.games`` for ``agent`` (see the module docstring). ``policy``, ``params``, ``modes`` and
    ``make`` default to the agent's (``ModelPolicy``, its trainable parameters, its frame modes, ``make_env``); tests
    pass small stand-ins. ``baselines`` (``{game: {"random", "expert"}}``) only adds normalised scores to the log."""

    def __init__(self, agent, cfg: RLConfig, policy=None, params: Optional[List[torch.nn.Parameter]] = None,
                 modes: Optional[Dict[str, str]] = None, make: Optional[Callable] = None,
                 baselines: Optional[Dict[str, Dict]] = None, log: Callable[[str], None] = print,
                 ref_model=None):
        unknown = [g for g in cfg.games if g not in GAMES and make is None]
        if unknown:
            raise ValueError("no RL game %s (%s)" % (unknown, ", ".join(GAMES)))
        check_seed(cfg.seed_lo)
        if cfg.returns not in ("episode", "togo"):
            raise ValueError("returns must be 'episode' or 'togo'")
        self.agent, self.cfg, self.log_fn, self.baselines = agent, cfg, log, baselines or {}
        if modes is None:
            modes = {g: F.mode_for(getattr(agent, "cfg", None), GAMES[g].family) for g in cfg.games}
        self.modes = modes
        self.policy = policy if policy is not None else ModelPolicy(agent, cfg.games, modes, cfg.temperature,
                                                                    seed=cfg.seed)
        self._make = make or (lambda name, seed: make_env(name, seed, cfg.caps.get(name, "default")))
        self._params = params
        self.model = getattr(agent, "model", None)
        self.ref = ref_model
        if self.ref is None and cfg.kl and self.model is not None:
            self.ref = copy.deepcopy(self.model).eval()
            self.ref.requires_grad_(False)
        self.opt = None
        self.rng = random.Random(cfg.seed)
        self.history: List[Dict] = []
        self.t_rollout = self.t_update = 0.0
        self.episodes = self.decisions = 0
        self._last_phase_s = 0.0

    # -- bookkeeping ------------------------------------------------------------------------------------------------

    def params(self) -> List[torch.nn.Parameter]:
        if self._params is None:
            self._params = [p for p in self.model.parameters() if p.requires_grad]
        return self._params

    def _optimizer(self):
        if self.opt is None:
            self.opt = torch.optim.AdamW(self.params(), lr=self.cfg.lr, weight_decay=0.0)
        return self.opt

    def groups(self) -> List[Tuple[str, int]]:
        n = max(1, self.cfg.episodes_per_phase // max(1, self.cfg.group))
        return [(g, check_seed(self.rng.randrange(self.cfg.seed_lo, TRAIN_SEED_MAX)))
                for g in self.cfg.games for _ in range(n)]

    # -- one phase --------------------------------------------------------------------------------------------------

    def phase(self, lr_scale: float = 1.0, deadline: Optional[float] = None, step: Optional[int] = None) -> Dict:
        """Rollout then update; returns (and logs) the phase's stats."""
        t0 = time.time()
        was_training = self.model.training if self.model is not None else False
        if self.model is not None:
            self.model.eval()  # the same (dropout-free) forward in rollout and update, so the PPO ratio starts at 1
        try:
            if hasattr(self.policy, "begin"):
                self.policy.begin()
            eps = rollout(self.policy, self.groups(), self.cfg.group, self.modes, self._make, deadline)
            t1 = time.time()
            adv = advantages(eps, self.cfg.returns, self.cfg.gamma)
            upd = self.update(eps, adv, lr_scale, deadline)
            if hasattr(self.policy, "end"):
                self.policy.end()
        finally:
            if self.model is not None and was_training:
                self.model.train()
        t2 = time.time()
        self.t_rollout += t1 - t0
        self.t_update += t2 - t1
        self.episodes += len(eps)
        n_dec = sum(len(ep.actions) for ep in eps)
        self.decisions += n_dec
        self._last_phase_s = t2 - t0
        per_game = {}
        for g in self.cfg.games:
            mine = [ep for ep in eps if ep.game == g]
            if not mine:
                continue
            m = float(np.mean([ep.score for ep in mine]))
            row = {"mean_score": m, "mean_return": float(np.mean([ep.ret for ep in mine])),
                   "episodes": len(mine), "mean_steps": float(np.mean([ep.steps for ep in mine])),
                   "cut": sum(ep.cut for ep in mine)}
            b = self.baselines.get(g)
            if b and b.get("expert") != b.get("random"):
                row["normalized"] = (m - b["random"]) / (b["expert"] - b["random"])
            per_game[g] = row
        rec = {"phase": len(self.history), "step": step, "time": t2, "rollout_s": t1 - t0, "update_s": t2 - t1,
               "rollout_frac": (t1 - t0) / max(1e-9, t2 - t0), "episodes": len(eps), "decisions": n_dec,
               "episodes_per_s": len(eps) / max(1e-9, t1 - t0), "decisions_per_s": n_dec / max(1e-9, t1 - t0),
               "games": per_game, "lr": self.cfg.lr * lr_scale, **upd}
        self.history.append(rec)
        self.log_fn("rl phase %d%s | %s | rollout %.1f s (%.0f%%) update %.1f s | %.2f ep/s %.0f dec/s | loss %.4f "
                    "ent %.3f clip %.2f%s" % (
                        rec["phase"], "" if step is None else " @ step %d" % step,
                        " ".join("%s %.1f%s" % (g, r["mean_score"], "" if "normalized" not in r else
                                                " (%.2f)" % r["normalized"]) for g, r in per_game.items()),
                        rec["rollout_s"], 100 * rec["rollout_frac"], rec["update_s"], rec["episodes_per_s"],
                        rec["decisions_per_s"], rec.get("loss", float("nan")), rec.get("entropy", float("nan")),
                        rec.get("clip_frac", float("nan")),
                        "" if "kl_start" not in rec else " kl %.4f" % rec["kl_start"]))
        return rec

    def update(self, eps: Sequence[Episode], adv: Sequence[np.ndarray], lr_scale: float = 1.0,
               deadline: Optional[float] = None) -> Dict:
        cfg = self.cfg
        rows = [(x, a, lp, float(v)) for ep, av in zip(eps, adv)
                for x, a, lp, v in zip(ep.inputs, ep.actions, ep.logps, av)]
        if not (cfg.kl or cfg.entropy):
            rows = [r for r in rows if r[3] != 0.0]  # zero advantage and no regulariser: no gradient
        out = {"update_steps": 0, "rows": len(rows), "mean_abs_adv": float(np.mean([abs(r[3]) for r in rows]))
               if rows else 0.0}
        if not rows:
            return out
        if cfg.max_update_steps and len(rows) > cfg.max_update_steps:
            rows = self.rng.sample(rows, cfg.max_update_steps)
        opt = self._optimizer()
        for g in opt.param_groups:
            g["lr"] = cfg.lr * lr_scale
        agg: Dict[str, List[float]] = {}
        for _ in range(max(1, cfg.epochs)):
            order = list(range(len(rows)))
            self.rng.shuffle(order)
            for s in range(0, len(order), cfg.minibatch):
                if deadline is not None and time.time() >= deadline:
                    break
                mb = [rows[j] for j in order[s:s + cfg.minibatch]]
                inputs = [r[0] for r in mb]
                z = self.policy.scaled_logits(inputs)
                ref = None
                if self.ref is not None:
                    with torch.no_grad():
                        ref = self.policy.scaled_logits(inputs, model=self.ref)
                loss, st = pg_loss(z, torch.tensor([r[1] for r in mb], device=z.device),
                                   torch.tensor([r[3] for r in mb], device=z.device, dtype=torch.float32),
                                   torch.tensor([r[2] for r in mb], device=z.device, dtype=torch.float32),
                                   cfg.clip, ref, cfg.kl, cfg.entropy)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.params(), 1.0)
                opt.step()
                out["update_steps"] += 1
                for k, v in dict(st, loss=float(loss.detach())).items():
                    agg.setdefault(k, []).append(v)
        out.update({k: float(np.mean(v)) for k, v in agg.items()})
        return out

    # -- drivers ----------------------------------------------------------------------------------------------------

    def due(self, step: int, elapsed_s: float) -> bool:
        every = self.cfg.every
        if every >= 1:
            return step > 0 and step % int(every) == 0
        return self.t_rollout + self.t_update < every * elapsed_s

    def hook(self, step: int, ctx: Dict) -> bool:
        """``laya.vlm_train.train(..., step_hook=rl.hook)``: a phase when one is due and fits in the time left."""
        if not self.due(step, ctx.get("elapsed_s", 0.0)):
            return False
        deadline = ctx.get("deadline")
        if deadline is not None and time.time() + 1.2 * self._last_phase_s >= deadline:
            return False
        self.phase(ctx.get("lr_scale", 1.0), deadline, step)
        return True

    def run(self, minutes: float, lr_scale: float = 1.0, max_phases: int = 0) -> List[Dict]:
        """RL only: phases until ``minutes`` have passed (or ``max_phases``)."""
        deadline = time.time() + minutes * 60
        while time.time() + 1.2 * self._last_phase_s < deadline and not (max_phases and len(self.history) >= max_phases):
            self.phase(lr_scale, deadline)
        return self.history

    def summary(self) -> Dict:
        total = self.t_rollout + self.t_update
        return {"phases": len(self.history), "episodes": self.episodes, "decisions": self.decisions,
                "rollout_s": self.t_rollout, "update_s": self.t_update,
                "rollout_frac": self.t_rollout / total if total else 0.0,
                "episodes_per_s": self.episodes / self.t_rollout if self.t_rollout else 0.0,
                "decisions_per_s": self.decisions / self.t_rollout if self.t_rollout else 0.0}


__all__ = ["TRAIN_SEED_MAX", "GAMES", "RLGame", "RLConfig", "GameRL", "ModelPolicy", "Episode", "rollout",
           "make_env", "question_for", "group_advantages", "returns_to_go", "togo_advantages", "advantages",
           "pg_loss", "check_seed"]
