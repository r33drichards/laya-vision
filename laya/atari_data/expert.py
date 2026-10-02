"""Expert-labelled Atari frames: a pretrained agent plays each game, and every full-colour frame it acts on is saved
with the agent's action probabilities, in the layout of ``site-docs/reference/atari-data-format.md`` (source ``expert``).

Agents are CleanRL's JAX PPO checkpoints on the Hugging Face Hub (``cleanrl/<Game>-v5-<exp>-seed<n>``), which cover
all 57 games. SB3's Hub agents (``sb3/*NoFrameskip-v4``) cover only 10 games and score lower. All three CleanRL
families used here were trained in envpool with the same IMPALA Atari settings: the minimal action set, 4-frame skip
with a max-pool over the last two frames, ALE grayscale resized to 84x84 with INTER_AREA, a 4-frame stack, no
sticky actions, FIRE after reset and after each lost life (``episodic_life``), and 27,000 steps per episode at most.

``Player`` rebuilds that observation on ``ALE/<Game>-v5`` created with ``frameskip=1``. Each decision takes 4 env
steps, so the dynamics, including the default 0.25 sticky actions, are those of the default v5 env. Each record
saves the raw 210x160 RGB screen the policy acted on. That matches what ``examples/atari_live.py`` feeds the model at
play time.

Run on Modal with ``modal_atari_expert.py``.
"""
import io
import json
import os
import shutil
import time
from collections import Counter
from typing import Callable, Dict, List, Optional

import numpy as np

FRAMESKIP = 4
MAX_EPISODE_STEPS = 27_000  # agent steps (108,000 frames), as in training and ALE v5's own frame cap
EPSILON = 0.1               # uniform-random actions mixed into the behaviour policy
VAL_SEED, TRAIN_SEED, EXPERT_EVAL_SEED, RANDOM_EVAL_SEED = 1_000, 2_000, 900_000, 950_000

# Best of seeds 1-3 of three CleanRL PPO families by the score on each model card (envpool eval, no sticky
# actions). All three share the training env above. "sebulba_*" and "cleanba_*" use the IMPALA CNN.
# "*_naturecnn" uses the Nature DQN CNN. The value is (experiment-seed, model-card mean return).
AGENTS = {
    "Alien": ("sebulba_ppo_envpool_impala_atari_wrapper-seed1", 5939.0),
    "Amidar": ("sebulba_ppo_envpool_impala_atari_wrapper-seed1", 1969.5),
    "Assault": ("sebulba_ppo_envpool_impala_atari_wrapper-seed1", 25571.3),
    "Asterix": ("cleanba_ppo_envpool_impala_atari_wrapper-seed3", 359500.0),
    "Asteroids": ("sebulba_ppo_envpool_impala_atari_wrapper-seed2", 105438.0),
    "Atlantis": ("cleanba_ppo_envpool_impala_atari_wrapper_naturecnn-seed2", 1002840.0),
    "BankHeist": ("sebulba_ppo_envpool_impala_atari_wrapper-seed2", 1302.0),
    "BattleZone": ("sebulba_ppo_envpool_impala_atari_wrapper-seed1", 73300.0),
    "BeamRider": ("sebulba_ppo_envpool_impala_atari_wrapper-seed1", 46892.8),
    "Berzerk": ("sebulba_ppo_envpool_impala_atari_wrapper-seed2", 5506.0),
    "Bowling": ("cleanba_ppo_envpool_impala_atari_wrapper-seed2", 52.9),
    "Boxing": ("cleanba_ppo_envpool_impala_atari_wrapper-seed2", 100.0),
    "Breakout": ("cleanba_ppo_envpool_impala_atari_wrapper-seed2", 828.9),
    "Centipede": ("sebulba_ppo_envpool_impala_atari_wrapper-seed3", 12005.4),
    "ChopperCommand": ("sebulba_ppo_envpool_impala_atari_wrapper-seed3", 52100.0),
    "CrazyClimber": ("sebulba_ppo_envpool_impala_atari_wrapper-seed3", 133760.0),
    "Defender": ("sebulba_ppo_envpool_impala_atari_wrapper-seed1", 449965.0),
    "DemonAttack": ("sebulba_ppo_envpool_impala_atari_wrapper-seed3", 133519.0),
    "DoubleDunk": ("cleanba_ppo_envpool_impala_atari_wrapper-seed2", 0.6),
    "Enduro": ("sebulba_ppo_envpool_impala_atari_wrapper-seed2", 2331.5),
    "FishingDerby": ("cleanba_ppo_envpool_impala_atari_wrapper-seed1", 41.0),
    "Freeway": ("cleanba_ppo_envpool_impala_atari_wrapper-seed1", 34.0),
    "Frostbite": ("cleanba_ppo_envpool_impala_atari_wrapper_naturecnn-seed2", 5139.0),
    "Gopher": ("cleanba_ppo_envpool_impala_atari_wrapper_naturecnn-seed2", 29770.0),
    "Gravitar": ("sebulba_ppo_envpool_impala_atari_wrapper-seed1", 2245.0),
    "Hero": ("sebulba_ppo_envpool_impala_atari_wrapper-seed1", 37326.0),
    "IceHockey": ("cleanba_ppo_envpool_impala_atari_wrapper-seed1", 15.5),
    "Jamesbond": ("sebulba_ppo_envpool_impala_atari_wrapper-seed3", 18450.0),
    "Kangaroo": ("sebulba_ppo_envpool_impala_atari_wrapper-seed1", 14480.0),
    "Krull": ("sebulba_ppo_envpool_impala_atari_wrapper-seed3", 10281.0),
    "KungFuMaster": ("sebulba_ppo_envpool_impala_atari_wrapper-seed1", 45980.0),
    "MontezumaRevenge": ("cleanba_ppo_envpool_impala_atari_wrapper-seed3", 290.0),
    "MsPacman": ("cleanba_ppo_envpool_impala_atari_wrapper-seed3", 4857.0),
    "NameThisGame": ("sebulba_ppo_envpool_impala_atari_wrapper-seed3", 15449.0),
    "Phoenix": ("sebulba_ppo_envpool_impala_atari_wrapper-seed2", 327976.0),
    "Pitfall": ("cleanba_ppo_envpool_impala_atari_wrapper-seed1", 0.0),
    "Pong": ("cleanba_ppo_envpool_impala_atari_wrapper-seed1", 21.0),
    "PrivateEye": ("cleanba_ppo_envpool_impala_atari_wrapper-seed1", 100.0),
    "Qbert": ("sebulba_ppo_envpool_impala_atari_wrapper-seed3", 21352.5),
    "Riverraid": ("sebulba_ppo_envpool_impala_atari_wrapper-seed3", 31214.0),
    "RoadRunner": ("sebulba_ppo_envpool_impala_atari_wrapper-seed1", 75260.0),
    "Robotank": ("cleanba_ppo_envpool_impala_atari_wrapper-seed1", 59.5),
    "Seaquest": ("cleanba_ppo_envpool_impala_atari_wrapper_naturecnn-seed3", 1838.0),
    "Skiing": ("cleanba_ppo_envpool_impala_atari_wrapper_naturecnn-seed1", -8987.2),
    "Solaris": ("sebulba_ppo_envpool_impala_atari_wrapper-seed3", 2488.0),
    "SpaceInvaders": ("cleanba_ppo_envpool_impala_atari_wrapper-seed3", 49396.5),
    "StarGunner": ("sebulba_ppo_envpool_impala_atari_wrapper-seed1", 275080.0),
    "Surround": ("cleanba_ppo_envpool_impala_atari_wrapper-seed2", 8.1),
    "Tennis": ("cleanba_ppo_envpool_impala_atari_wrapper-seed1", 21.0),
    "TimePilot": ("sebulba_ppo_envpool_impala_atari_wrapper-seed1", 68760.0),
    "Tutankham": ("cleanba_ppo_envpool_impala_atari_wrapper-seed2", 306.7),
    "UpNDown": ("sebulba_ppo_envpool_impala_atari_wrapper-seed2", 370396.0),
    "Venture": ("cleanba_ppo_envpool_impala_atari_wrapper_naturecnn-seed1", 1190.0),
    "VideoPinball": ("sebulba_ppo_envpool_impala_atari_wrapper-seed2", 632621.7),
    "WizardOfWor": ("sebulba_ppo_envpool_impala_atari_wrapper-seed2", 21120.0),
    "YarsRevenge": ("sebulba_ppo_envpool_impala_atari_wrapper-seed2", 127249.0),
    "Zaxxon": ("sebulba_ppo_envpool_impala_atari_wrapper-seed2", 41460.0),
}


def agent_repo(game: str) -> str:
    return "cleanrl/%s-v5-%s" % (game, AGENTS[game][0])


class Policy:
    """A CleanRL PPO actor: ``probs(obs)`` maps uint8 ``(N, 4, 84, 84)`` frame stacks to action probabilities."""

    def __init__(self, repo: str):
        import flax.linen as nn
        import jax
        import jax.numpy as jnp
        from flax import serialization
        from huggingface_hub import hf_hub_download

        exp = repo.split("-v5-", 1)[1].rsplit("-seed", 1)[0]
        with open(hf_hub_download(repo, exp + ".cleanrl_model"), "rb") as f:
            # saved as to_bytes([vars(args), [network_params, actor_params, critic_params]])
            state = serialization.msgpack_restore(f.read())
        net_params, actor_params = state["1"]["0"], state["1"]["1"]
        self.repo, self.n_actions = repo, int(actor_params["params"]["Dense_0"]["kernel"].shape[-1])
        self.arch = "impala_cnn" if "ConvSequence_0" in net_params["params"] else "nature_cnn"

        # Module names and layouts match the training scripts, so the checkpoint's parameter tree applies as is.
        class ResidualBlock(nn.Module):
            channels: int

            @nn.compact
            def __call__(self, x):
                y = nn.Conv(self.channels, kernel_size=(3, 3))(nn.relu(x))
                y = nn.Conv(self.channels, kernel_size=(3, 3))(nn.relu(y))
                return x + y

        class ConvSequence(nn.Module):
            channels: int

            @nn.compact
            def __call__(self, x):
                x = nn.Conv(self.channels, kernel_size=(3, 3))(x)
                x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding="SAME")
                x = ResidualBlock(self.channels)(x)
                return ResidualBlock(self.channels)(x)

        impala = self.arch == "impala_cnn"

        class Network(nn.Module):
            @nn.compact
            def __call__(self, x):
                x = jnp.transpose(x, (0, 2, 3, 1)) / 255.0
                if impala:
                    for channels in (16, 32, 32):
                        x = ConvSequence(channels)(x)
                    x = nn.relu(x)
                else:
                    for c, k, s in ((32, 8, 4), (64, 4, 2), (64, 3, 1)):
                        x = nn.relu(nn.Conv(c, kernel_size=(k, k), strides=(s, s), padding="VALID")(x))
                x = x.reshape((x.shape[0], -1))
                return nn.relu(nn.Dense(256 if impala else 512)(x))

        class Actor(nn.Module):
            action_dim: int

            @nn.compact
            def __call__(self, x):
                return nn.Dense(self.action_dim)(x)

        net, actor = Network(), Actor(self.n_actions)
        self._probs = jax.jit(lambda obs: jax.nn.softmax(actor.apply(actor_params, net.apply(net_params, obs)), -1))

    def probs(self, obs: np.ndarray) -> np.ndarray:
        p = np.asarray(self._probs(obs), dtype=np.float64)
        return p / p.sum(-1, keepdims=True)


class Player:
    """``ALE/<Game>-v5`` with default settings, stepped 4 frames per decision. It exposes the raw RGB screen
    (``rgb``) and the envpool-style 84x84x4 max-pooled grayscale stack the agents were trained on (``obs``).

    FIRE is pressed after reset and after each lost life, as in training. Those steps are not decisions, so they
    are never recorded.
    """

    def __init__(self, game: str, repeat_action_probability: float = 0.25):
        import ale_py
        import gymnasium as gym

        gym.register_envs(ale_py)
        self.env = gym.make("ALE/%s-v5" % game, frameskip=1, repeat_action_probability=repeat_action_probability)
        self.ale = self.env.unwrapped.ale
        self.actions = list(self.env.unwrapped.get_action_meanings())
        self.fire = self.actions.index("FIRE") if "FIRE" in self.actions else None

    def minimal_action_set(self) -> List[str]:
        """The ALE's own minimal action set, which envpool's ``full_action_space=False`` exposes to the agents."""
        import ale_py

        return [ale_py.Action(a).name for a in self.ale.getMinimalActionSet()]

    def _gray(self) -> np.ndarray:
        return np.asarray(self.ale.getScreenGrayscale()).reshape(210, 160)

    def _push(self, frame: np.ndarray, fill: bool = False):
        import cv2

        small = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)
        self.stack = [small] * 4 if fill else self.stack[1:] + [small]

    def reset(self, seed: int):
        self.rgb, info = self.env.reset(seed=seed)
        self.lives, self.done, self.score, self.steps = info.get("lives", 0), False, 0.0, 0
        self.last_gray = self._gray()
        self._push(self.last_gray, fill=True)
        self._fire()

    @property
    def obs(self) -> np.ndarray:
        return np.stack(self.stack)[None]

    def _fire(self):
        if self.fire is not None and not self.done:
            self._act(self.fire)

    def step(self, action: int) -> bool:
        """Play one decision. Returns True when a life was lost and FIRE was pressed automatically (a game
        without FIRE never resets its frame history this way)."""
        lives = self.lives
        self._act(action)
        if not self.done and self.lives < lives:
            self._fire()
            return self.fire is not None
        return False

    def _act(self, action: int):
        grays = [self.last_gray]
        for _ in range(FRAMESKIP):
            self.rgb, r, term, trunc, info = self.env.step(action)
            self.score += float(r)
            grays.append(self._gray())
            if term or trunc:
                self.done = True
                break
        self.last_gray = grays[-1]
        self.lives = info.get("lives", self.lives)
        self._push(np.maximum(grays[-1], grays[-2]))
        self.steps += 1
        if self.steps >= MAX_EPISODE_STEPS:
            self.done = True


class EvenSample:
    """Evenly spaced subset of a stream of unknown length. It keeps every ``stride``-th step, and when the buffer
    reaches ``2 * cap`` it doubles the stride and drops every other kept item. Memory stays bounded, and PNGs are
    encoded only for steps that might be kept."""

    def __init__(self, cap: int):
        self.cap, self.stride, self.items = cap, 1, []

    def wants(self, step: int) -> bool:
        return step % self.stride == 0

    def add(self, item: Dict):
        self.items.append(item)
        if len(self.items) >= 2 * self.cap:
            self.stride *= 2
            self.items = [it for it in self.items if it["step"] % self.stride == 0]

    def result(self) -> List[Dict]:
        return even(self.items, self.cap)


def even(items: List, n: int) -> List:
    if len(items) <= n:
        return list(items)
    return [items[i] for i in np.unique(np.linspace(0, len(items) - 1, n).round().astype(int))]


def _png(rgb: np.ndarray) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return buf.getvalue()


def play(player: Player, policy: Optional[Policy], mode: str, seed: int, rng: np.random.Generator,
         keep: Optional[EvenSample] = None, max_steps: Optional[int] = None, history: int = 0) -> float:
    """One episode. ``mode`` is ``greedy`` (argmax), ``random`` (uniform), or ``behaviour`` (sample from the
    policy, with an ``EPSILON`` chance of a uniform-random action instead). ``max_steps`` ends the episode after
    that many decisions (auto-FIRE steps not counted). Returns the episode score.

    ``history`` > 0 also gives each kept item ``"hist"``: ``[(step, png), ...]``, the RGB screens of up to
    ``history`` previous decisions, oldest first, as ``laya.atari_train.play`` keeps them at play time: the
    history starts empty on reset and again after an auto-FIRE (a life lost), so it never reaches back past one.
    Recording it draws nothing from ``rng``: the trajectory is the same as without it."""
    n = len(player.actions)
    player.reset(seed)
    t = 0
    past: List = []   # (step, rgb) of the previous decisions since the last reset / auto-FIRE
    pngs: Dict[int, bytes] = {}  # step -> encoded screen, for the steps in ``past``
    while not player.done and (max_steps is None or t < max_steps):
        if mode == "random":
            a = int(rng.integers(n))
        else:
            p = policy.probs(player.obs)[0]
            if mode == "greedy":
                a = int(p.argmax())
            else:
                a = int(rng.integers(n)) if rng.random() < EPSILON else int(rng.choice(n, p=p))
                if keep is not None and keep.wants(t):
                    item = {"step": t, "png": _png(player.rgb), "target": p, "taken": a}
                    if history:
                        for s, rgb in past:
                            if s not in pngs:
                                pngs[s] = _png(rgb)
                        item["hist"] = [(s, pngs[s]) for s, _ in past]
                    keep.add(item)
        rgb = player.rgb
        if player.step(a):
            past, pngs = [], {}
        elif history:
            past = (past + [(t, np.array(rgb))])[-history:]
            pngs = {s: pngs[s] for s, _ in past if s in pngs}
        t += 1
    return player.score


def collect(player: Player, policy: Policy, n_frames: int, per_episode: int, first_episode: int, seed_base: int,
            rng: np.random.Generator, max_steps: int, log: Callable = print, history: int = 0):
    """Play behaviour episodes until ``n_frames`` are kept (at most ``per_episode`` per episode, spread evenly over
    the episode) or ``max_steps`` agent steps are used. Then subsample evenly across episodes down to ``n_frames``.
    Returns ``(records, episodes_played, scores)``, where ``records`` is a list of ``(episode, item)``."""
    kept, scores, steps, ep = [], [], 0, first_episode
    while sum(len(k) for _, k in kept) < n_frames and steps < max_steps:
        keep = EvenSample(per_episode)
        t0 = time.time()
        scores.append(play(player, policy, "behaviour", seed_base + ep, rng, keep, history=history))
        steps += player.steps
        kept.append((ep, keep.result()))
        log("  episode %d: score %.0f, %d steps, kept %d (%.0fs)"
            % (ep, scores[-1], player.steps, len(kept[-1][1]), time.time() - t0))
        ep += 1
    flat = [(e, it) for e, items in kept for it in items]
    return even(flat, n_frames), ep - first_episode, scores


def baseline_scores(game: str, max_steps: int, episodes: int = 5, sticky: float = 0.25) -> Dict:
    """Greedy-expert and uniform-random mean scores with episodes capped at ``max_steps`` decisions, on the same
    episode seeds as ``generate``'s uncapped baselines, to match an evaluation that uses the same cap."""
    player = Player(game, sticky)
    policy = Policy(agent_repo(game))
    rng = np.random.default_rng(0)
    expert = [play(player, policy, "greedy", EXPERT_EVAL_SEED + i, rng, max_steps=max_steps) for i in range(episodes)]
    random_ = [play(player, None, "random", RANDOM_EVAL_SEED + i, rng, max_steps=max_steps) for i in range(episodes)]
    return {"game": game, "expert": expert, "random": random_}


def generate(game: str, out_root: str, train_frames: int = 20_000, val_frames: int = 1_000, eval_episodes: int = 5,
             seed: int = 0, sticky: float = 0.25, log: Callable = print, source: str = "expert", history: int = 0,
             baselines: Optional[Dict] = None, create_only: bool = False) -> Dict:
    """Measure expert and random scores for ``game``. If ``train_frames`` > 0, also write
    ``<out_root>/<game>/{train,val}.jsonl, images/, meta.json``. The caller commits and then writes ``_READY``.
    Returns the summary that also goes in ``meta.json``. ``sticky`` is ALE's repeat_action_probability. Keep the
    v5 default (0.25) for data. 0 reproduces the agents' training env, which is useful only as a pipeline check.

    ``history`` > 0 also stores, per record, ``"history"``: the image paths of up to ``history`` previous decision
    screens of the same episode, oldest first (fewer after a reset or an auto-FIRE; see ``play``). Each screen is
    written once as ``images/<source>-<Game>-e<episode>-s<step>.png`` and shared by every record that needs it.

    ``baselines`` (the score fields of an earlier run's meta.json) are copied in instead of re-measured. The greedy
    expert episodes draw nothing from the RNG, so they are skipped; the random ones are still played (and their
    scores dropped) so the RNG reaches the data episodes in the state an ordinary run leaves it in: with the same
    ``seed`` the recording then replays the earlier run's trajectories exactly. ``create_only`` refuses an
    existing ``<out_root>/<game>`` instead of replacing it."""
    from laya.games import atari_question

    out = os.path.join(out_root, game)
    if create_only and os.path.exists(out):
        raise FileExistsError("%s exists; this recording is create-only" % out)
    repo = agent_repo(game)
    player = Player(game, sticky)
    actions = player.actions
    policy = Policy(repo)
    ale_minimal = player.minimal_action_set()
    check = {"agent_n_actions": policy.n_actions, "v5_n_actions": len(actions),
             "v5_meanings_equal_ale_minimal_set": ale_minimal == actions,
             "ok": policy.n_actions == len(actions) and ale_minimal == actions}
    log("%s: %s (%s), actions %s, check %s" % (game, repo, policy.arch, actions, check))
    if not check["ok"]:
        raise ValueError("%s: action-set mismatch %s (ALE minimal set %s)" % (game, check, ale_minimal))

    rng = np.random.default_rng(seed)
    t0 = time.time()
    if baselines is None:
        expert = [play(player, policy, "greedy", EXPERT_EVAL_SEED + i, rng) for i in range(eval_episodes)]
        random_ = [play(player, None, "random", RANDOM_EVAL_SEED + i, rng) for i in range(eval_episodes)]
        log("%s: expert %s | random %s (%.0fs)" % (game, expert, random_, time.time() - t0))
        baselines = {
            "expert_score": float(np.mean(expert)), "expert_scores": expert,
            "random_score": float(np.mean(random_)), "random_scores": random_,
            "eval": {"env": "ALE/%s-v5" % game,
                     "settings": "defaults: frameskip 4, repeat_action_probability %s, minimal action set, max "
                                 "108000 frames; run as frameskip=1 x 4 so the agent sees max-pooled frames" % sticky,
                     "repeat_action_probability": sticky, "sticky_actions": sticky > 0,
                     "score": "sum of raw (unclipped) ALE v5 rewards until game over or the step cap",
                     "episodes": eval_episodes, "expert_policy": "greedy (argmax of the PPO policy)",
                     "random_policy": "uniform over the minimal action set",
                     "fire_after_reset_and_life_loss": player.fire is not None,
                     "max_episode_steps": MAX_EPISODE_STEPS},
        }
    else:
        random_ = [play(player, None, "random", RANDOM_EVAL_SEED + i, rng) for i in range(eval_episodes)]
        log("%s: baselines copied (expert %s, random %s); random episodes replayed for the RNG: %s (%.0fs)"
            % (game, baselines.get("expert_score"), baselines.get("random_score"), random_, time.time() - t0))
    meta = {
        "source": source, "game": game, "frame_format": "rgb_210x160", "actions": actions,
        "origin": "https://huggingface.co/" + repo,
        "license": "agent weights: CleanRL (MIT); frames rendered with ale-py (ROMs bundled under GPL-2.0)",
        "agent": {"repo": repo, "algorithm": "PPO", "network": policy.arch, "framework": "JAX/Flax (CleanRL)",
                  "model_card_score": AGENTS[game][1],
                  "trained_env": "envpool %s-v5: minimal action set, frameskip 4 + max-pool of last 2 frames, ALE "
                                 "grayscale 84x84 INTER_AREA, 4-frame stack, repeat_action_probability 0, "
                                 "noop_max 30, episodic_life, FIRE on reset, reward clipping" % game},
        **baselines,
        "action_check": check,
    }
    if train_frames <= 0:
        return meta

    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(os.path.join(out, "images"))
    budget = 20 * (train_frames + val_frames) + 10 * MAX_EPISODE_STEPS
    t0 = time.time()
    log("%s: val episodes" % game)
    val, n_val_eps, val_scores = collect(player, policy, val_frames, max(1, val_frames // 2), 0, VAL_SEED, rng,
                                         budget // 20, log, history)
    log("%s: train episodes" % game)
    train, n_train_eps, train_scores = collect(player, policy, train_frames, max(1, train_frames // 10),
                                               n_val_eps, TRAIN_SEED, rng, budget, log, history)
    question = atari_question(game, actions)["action"]
    for split, recs, n_eps, scores in (("val", val, n_val_eps, val_scores), ("train", train, n_train_eps,
                                                                           train_scores)):
        labels, written = Counter(), set()

        def save(ep, step, png):
            name = "%s-%s-e%06d-s%06d.png" % (source, game, ep, step)
            if name not in written:
                with open(os.path.join(out, "images", name), "wb") as img:
                    img.write(png)
                written.add(name)
            return "images/" + name

        with open(os.path.join(out, split + ".jsonl"), "w") as f:
            for ep, it in recs:
                rid = "%s-%s-e%06d-s%06d" % (source, game, ep, it["step"])
                image = save(ep, it["step"], it["png"])
                target = [round(float(p), 6) for p in it["target"]]
                label = int(np.argmax(it["target"]))
                labels[actions[label]] += 1
                rec = {"id": rid, "image": image, "game": game, "actions": actions,
                       "label": label, "target": target, "question": question, "source": source,
                       "episode": ep, "step": it["step"], "taken": it["taken"]}
                if history:
                    rec["history"] = [save(ep, s, png) for s, png in it["hist"]]
                f.write(json.dumps(rec) + "\n")
        meta[split] = {"records": len(recs), "episodes": len({ep for ep, _ in recs}), "episodes_played": n_eps,
                       "labels": dict(labels.most_common()), "behaviour_scores": scores}
    meta["dropped"] = {"not_in_minimal_set": 0}
    meta["behaviour"] = {"policy": "sample from the agent's policy; with probability %.2f take a uniform-random "
                                   "action instead" % EPSILON, "epsilon": EPSILON,
                         "target": "agent's action probabilities before the random mixing",
                         "label": "argmax(target)", "taken": "action actually played (index into actions)",
                         "frames": "per episode, an even subsample of decision steps (at most %d train / %d val); "
                                   "then an even subsample across episodes" % (max(1, train_frames // 10),
                                                                               max(1, val_frames // 2)),
                         "episode_seeds": {"val": VAL_SEED, "train": TRAIN_SEED}}
    if history:
        meta["history"] = ("per record, the image paths of up to %d previous decision screens of the same episode "
                           "(4 emulator frames apart), oldest first, saved at decision time; empty on the first "
                           "decision and restarted after an auto-FIRE (reset or life loss), exactly the frames "
                           "laya.atari_train.play keeps at play time" % history)
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    log("%s: wrote %d train / %d val frames (%.0fs)" % (game, len(train), len(val), time.time() - t0))
    return meta
