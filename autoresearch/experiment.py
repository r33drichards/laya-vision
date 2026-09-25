"""The autoresearch experiment: the one file the agent edits (upstream's ``train.py``).

``harness.py`` calls ``build(ctx)`` (model + data loading; not timed), then ``train(agent, ctx)``, which must return
within ``ctx.time_budget_s`` seconds (15 minutes). The harness then fits temperatures, saves the model, reloads it and
measures quality, parameter count and L4 latency the same way for every experiment. Anything is fair game here:
where to start from, what to cut, what to train on, the objective and the optimizer. The only rules are the
harness's: train only on ``ctx.train_examples()``, stay inside the time budget, and make whatever you change survive
``agent.save`` / reload (an architecture change has to be written into the backbone config, as the helpers below do).

``ctx`` has ``time_budget_s``, ``device``, ``ckpt_path(run)`` (a run on the laya-checkpoints volume) and
``train_examples(names=...)`` (the data pool's up to 6,000 examples per Cauldron and score train split, never the
calibration tail; images are in memory as encoded bytes) and ``game_examples(names=...)`` (the pool's Atari
Freeway / Breakout and ViZDoom basic expert frames).

Game play is the ``games`` objective (``games_eval.py``). ``import toolkit`` generates more game training data on the
fly: ``toolkit.maze_examples(n)`` and ``toolkit.snake_examples(n)`` (soft targets over every shortest-path move, a
``value`` target, the board's 8 symmetries), ``toolkit.control_examples(game, n)`` for CartPole, Acrobot,
MountainCar and LunarLander, and ``toolkit.game_mix(ctx.train_examples(), games, frac, base_weights=MIX)`` to give
games ``frac`` of the draws. A model built with ``"value_head": True`` in its config learns the ``value`` targets
(``train(..., w_value=...)``); ``"next_head": True`` adds an auxiliary head trained on the examples' ``next_target``
(the expert's move at the next step; ``train(..., w_next=...)``), never used for play. The games are played greedy:
no test-time search.

``GAME_FRAMES`` is the game frame mode (``laya.frames``): ``"single"`` (one frame; the control games ghost the
previous frame in), ``"trail-N"`` (the last N frames blended into one image) or ``"stack-N"`` (the last N frames as
N images, N <= 5). ``build`` passes it to every game-data call and writes it into ``agent.cfg["game_frames"]``, so the
saved checkpoint carries it: the games benchmark plays in it and the latency job times a game move in it
(``latency_x`` is the larger of the question and game-move ratios, so a stack's extra per-move cost counts).
"""
from typing import Dict, Optional

# -- the recipe ---------------------------------------------------------------------------------------------------

INIT = "cauldron-score-2ep-bidir-full/best"   # start from this checkpoint (= thaitea/laya-vision); "" = fresh BACKBONE
BACKBONE = "HuggingFaceTB/SmolVLM-256M-Instruct"
OPTION_ATTENTION = "bidirectional"            # for a fresh BACKBONE only; a checkpoint keeps its own

# size and latency: 0 keeps what the checkpoint has
KEEP_TEXT_LAYERS = 20      # keep the first N language-model decoder layers (SmolVLM-256M has 30)
KEEP_VISION_LAYERS = 0    # keep the first N vision-tower layers (SmolVLM-256M has 12)
IMAGE_SIZE = 0            # square side fed to the vision tower, a multiple of 64 (the checkpoint uses 512)

# training
TRAIN_SETS = None         # None = every trainable set (ctx.train_examples() default)
MIX: Optional[Dict[str, float]] = {"score_vlfeedback": 3.0}   # per-dataset sampling weights, as in the checkpoint's run
FREEZE = "full"           # "head", "last_n" or "full" (everything but the vision tower)
TRAIN_VISION = False      # with "full": train the vision tower too (at LR_VISION)
LR_VISION = None          # the vision tower's LR when it trains; None = LR_BACKBONE
LR_HEAD = 5e-5
LR_BACKBONE = 1e-5
BATCH_SIZE = 64
WARMUP_STEPS = 20

# games: share of training draws given to game examples (toolkit-generated + the pool's expert frames); 0 = none
GAME_FRAC = 0.45
# how game states are shown (laya.frames): "single", "trail-N" or "stack-N"; saved in the checkpoint
GAME_FRAMES = "stack-4"
# auxiliary next-move head (KataGo 1902.10565 sec. 3.4): predicts the expert's move at t+1, training only
NEXT_HEAD = False
W_NEXT = 0.5
CONTROL_GAMES = ("CartPole", "Acrobot", "MountainCar", "LunarLander")

# RL from game rewards (laya.game_rl: GRPO, the game's own score as the reward, rollouts on training seeds < 100,000),
# interleaved with the supervised batches through train()'s step_hook. RL_GAMES = () is off: the recipe is unchanged.
# Games: CartPole, Acrobot, MountainCar, LunarLander, Maze4, Maze6, Snake10 (Atari needs ale-py, not in this image).
RL_GAMES = ()
RL_EVERY = 200             # supervised steps between RL phases; a value in (0, 1) is RL's share of the wall clock
RL_GROUP = 8               # G: episodes per seed (the group the advantage is normalised over)
RL_EPISODES_PER_PHASE = 16  # per game per phase (whole groups)
RL_TEMPERATURE = 1.0       # sampling temperature on top of the checkpoint's calibrated one
RL_LR = 1e-5               # RL optimizer LR, times the supervised schedule's factor (smoke: 1e-5 > 2e-6)
RL_KL = 0.0                # KL toward the starting policy (keeps a frozen copy of the model on the GPU when > 0)
RL_RETURN = "episode"      # "episode" (GRPO outcome advantage) or "togo" (discounted return-to-go)
RL_GAMMA = 0.99


# -- helpers ------------------------------------------------------------------------------------------------------

def keep_text_layers(agent, n: int) -> None:
    """Drop all but the first ``n`` decoder layers of the language model, in the module and in the saved config."""
    tm = agent.model.encoder.text_model
    tm.layers = tm.layers[:n]
    cfg = agent.model.encoder.config.text_config
    cfg.num_hidden_layers = n
    if getattr(cfg, "layer_types", None):
        cfg.layer_types = list(cfg.layer_types)[:n]


def keep_vision_layers(agent, n: int) -> None:
    """Drop all but the first ``n`` layers of the vision tower, in the module and in the saved config."""
    enc = agent.model.encoder.vision_model.encoder
    enc.layers = enc.layers[:n]
    agent.model.encoder.config.vision_config.num_hidden_layers = n


def set_image_size(agent, size: int) -> None:
    """Feed the vision tower ``size`` x ``size`` images (fewer image tokens per question)."""
    from laya.preprocess import ImagePrep

    prep = ImagePrep.from_config(dict(agent.cfg, image_size=size), default_backend=agent.prep.backend)
    prep.apply(agent.processor)
    agent.prep = agent.model.prep = prep
    agent.cfg.update(prep.to_config())


# -- the two entry points the harness calls ----------------------------------------------------------------------

def build(ctx):
    from laya.vlm import VLMAgent

    if INIT:
        agent = VLMAgent(ctx.ckpt_path(INIT), device=ctx.device, **({"next_head": True} if NEXT_HEAD else {}))
    else:
        agent = VLMAgent(backbone=BACKBONE, device=ctx.device, option_attention=OPTION_ATTENTION)
    if KEEP_TEXT_LAYERS:
        keep_text_layers(agent, KEEP_TEXT_LAYERS)
    if KEEP_VISION_LAYERS:
        keep_vision_layers(agent, KEEP_VISION_LAYERS)
    if IMAGE_SIZE:
        set_image_size(agent, IMAGE_SIZE)
    agent.cfg["game_frames"] = GAME_FRAMES  # the frame mode is part of the model: the benchmark reads it back
    ctx.data = ctx.train_examples(TRAIN_SETS) if TRAIN_SETS else ctx.train_examples()
    ctx.mix = MIX
    if GAME_FRAC:
        import toolkit

        games = (toolkit.maze_examples(20000, frames=GAME_FRAMES) + toolkit.snake_examples(20000, frames=GAME_FRAMES)
                 + ctx.game_examples(frames=GAME_FRAMES))
        for g in CONTROL_GAMES:
            games += toolkit.control_examples(g, 5000, frames=GAME_FRAMES)
        ctx.data, ctx.mix = toolkit.game_mix(ctx.data, games, GAME_FRAC, base_weights=MIX)
    return agent


def train(agent, ctx):
    import laya.vlm_train as vt
    from laya.vlm_train import train as train_loop

    if TRAIN_VISION:
        base = vt.set_trainable
        vt.set_trainable = lambda model, mode="head", n_last=4: base(model, mode, n_last=n_last, train_vision=True)

    extra = {}
    if RL_GAMES:
        extra["step_hook"] = rl_trainer(agent).hook
    if TRAIN_VISION and LR_VISION is not None:
        extra["lr_vision"] = LR_VISION
    train_loop(agent.model, agent.processor, ctx.data, steps=10**9, batch_size=BATCH_SIZE, freeze=FREEZE,
               lr_head=LR_HEAD, lr_backbone=LR_BACKBONE, warmup=WARMUP_STEPS,
               mix_weights=ctx.mix,
               w_next=W_NEXT if NEXT_HEAD else 0.0, max_minutes=ctx.time_budget_s / 60, num_workers=12, log_every=50, device=ctx.device,
               **extra)


def rl_trainer(agent):
    """The ``laya.game_rl.GameRL`` the RL_* knobs describe (the benchmark's baselines only label its log)."""
    from laya.game_rl import GameRL, RLConfig

    try:
        import games_eval

        baselines = games_eval.load_baselines()
    except Exception:  # noqa: BLE001 - logging only
        baselines = None
    cfg = RLConfig(games=tuple(RL_GAMES), group=RL_GROUP, episodes_per_phase=RL_EPISODES_PER_PHASE,
                   temperature=RL_TEMPERATURE, lr=RL_LR, kl=RL_KL, returns=RL_RETURN, gamma=RL_GAMMA, every=RL_EVERY)
    return GameRL(agent, cfg, baselines=baselines)
