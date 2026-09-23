"""The autoresearch experiment: the one file the agent edits (upstream's ``train.py``).

``harness.py`` calls ``build(ctx)`` (model + data loading; not timed), then ``train(agent, ctx)``, which must return
within ``ctx.time_budget_s`` seconds (5 minutes). The harness then fits temperatures, saves the model, reloads it and
measures quality, parameter count and L4 latency the same way for every experiment. Anything is fair game here:
where to start from, what to cut, what to train on, the objective and the optimizer. The only rules are the
harness's: train only on ``ctx.train_examples()``, stay inside the time budget, and make whatever you change survive
``agent.save`` / reload (an architecture change has to be written into the backbone config, as the helpers below do).

``ctx`` has ``time_budget_s``, ``device``, ``ckpt_path(run)`` (a run on the laya-checkpoints volume) and
``train_examples(names=...)`` (the data pool's 2,000 examples per Cauldron and score train split, never the
calibration tail; images are in memory as encoded bytes) and ``game_examples(names=...)`` (the pool's Atari
Freeway / Breakout and ViZDoom basic expert frames).

Game play is the ``games`` objective (``games_eval.py``). ``import toolkit`` generates more game training data on the
fly: ``toolkit.maze_examples(n)`` and ``toolkit.snake_examples(n)`` (soft targets over every shortest-path move, a
``value`` target, the board's 8 symmetries), ``toolkit.control_examples(game, n)`` for CartPole, Acrobot,
MountainCar and LunarLander, and ``toolkit.game_mix(ctx.train_examples(), games, frac, base_weights=MIX)`` to give
games ``frac`` of the draws. A model built with ``"value_head": True`` in its config learns the ``value`` targets
(``train(..., w_value=...)``); setting ``agent.cfg["search"]`` makes the games benchmark plan with ``laya.search``
on the deterministic games.
"""
from typing import Dict, Optional

# -- the recipe ---------------------------------------------------------------------------------------------------

INIT = "cauldron-score-2ep-bidir-full/best"   # start from this checkpoint (= thaitea/laya-vision); "" = fresh BACKBONE
BACKBONE = "HuggingFaceTB/SmolVLM-256M-Instruct"
OPTION_ATTENTION = "bidirectional"            # for a fresh BACKBONE only; a checkpoint keeps its own

# size and latency: 0 keeps what the checkpoint has
KEEP_TEXT_LAYERS = 15      # keep the first N language-model decoder layers (SmolVLM-256M has 30)
KEEP_VISION_LAYERS = 0    # keep the first N vision-tower layers (SmolVLM-256M has 12)
IMAGE_SIZE = 0            # square side fed to the vision tower, a multiple of 64 (the checkpoint uses 512)

# training
TRAIN_SETS = None         # None = every trainable set (ctx.train_examples() default)
MIX: Optional[Dict[str, float]] = {"score_vlfeedback": 3.0}   # per-dataset sampling weights, as in the checkpoint's run
FREEZE = "full"           # "head", "last_n" or "full" (everything but the vision tower)
LR_HEAD = 1e-4
LR_BACKBONE = 2e-5
BATCH_SIZE = 32
WARMUP_STEPS = 20

# games: share of training draws given to game examples (toolkit-generated + the pool's expert frames); 0 = none
GAME_FRAC = 0.25
CONTROL_GAMES = ("CartPole", "Acrobot", "MountainCar", "LunarLander")


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
        agent = VLMAgent(ctx.ckpt_path(INIT), device=ctx.device)
    else:
        agent = VLMAgent(backbone=BACKBONE, device=ctx.device, option_attention=OPTION_ATTENTION)
    if KEEP_TEXT_LAYERS:
        keep_text_layers(agent, KEEP_TEXT_LAYERS)
    if KEEP_VISION_LAYERS:
        keep_vision_layers(agent, KEEP_VISION_LAYERS)
    if IMAGE_SIZE:
        set_image_size(agent, IMAGE_SIZE)
    ctx.data = ctx.train_examples(TRAIN_SETS) if TRAIN_SETS else ctx.train_examples()
    ctx.mix = MIX
    if GAME_FRAC:
        import toolkit

        games = toolkit.maze_examples(20000) + toolkit.snake_examples(20000) + ctx.game_examples()
        for g in CONTROL_GAMES:
            games += toolkit.control_examples(g, 5000)
        ctx.data, ctx.mix = toolkit.game_mix(ctx.data, games, GAME_FRAC, base_weights=MIX)
    return agent


def train(agent, ctx):
    from laya.vlm_train import train as train_loop

    train_loop(agent.model, agent.processor, ctx.data, steps=10**9, batch_size=BATCH_SIZE, freeze=FREEZE,
               lr_head=LR_HEAD, lr_backbone=LR_BACKBONE, warmup=WARMUP_STEPS, mix_weights=ctx.mix,
               max_minutes=ctx.time_budget_s / 60, num_workers=12, log_every=50, device=ctx.device)
