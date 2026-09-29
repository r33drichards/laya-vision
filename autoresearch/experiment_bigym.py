"""The BiGym autoresearch experiment: the file the agent edits under ``--profile bigym`` (see ``program_bigym.md``).

    modal run autoresearch/harness.py --tag <tag> --profile bigym --experiment autoresearch/experiment_bigym.py

The harness calls ``build(ctx)`` (not timed) and ``train(agent, ctx)`` (``ctx.time_budget_s``, 15 minutes), then
calibrates, saves, reloads and measures quality, games and the BiGym benchmark (``bigym_eval.py``) on the saved
checkpoint. What the benchmark plays is read from the checkpoint: ``agent.cfg["bigym_frames"]`` (1 or 4 head frames
per decision) and its token budgets (``head_max_len``: the 4-frame 37-option question needs 320 to be safe).

Besides ``experiment.py``'s context (``train_examples``, ``game_examples``, ``toolkit``), ``ctx`` has:

* ``ctx.bigym_examples(names=...)``: the cleaned BiGym train records, ``bigym_v2c_bc_f4`` (control on the last 4
  head frames, the demo follower's primitive as the label), ``bigym_v2c_bc_f1`` (the same on 1 frame) and
  ``bigym_v2c_probe`` (done / progress / side questions with simulator labels); the four cupboard tasks only;
* ``ctx.bigym_demos(task)``: the cupboard tasks' train demos as waypoints, for ``laya.bigymdemos.follow`` /
  ``lookahead`` (label the model's own states: DAgger);
* ``ctx.bigym_game(task, seed, cameras=True, env=None)``: a ``laya.bigymgames.BiGymGame`` for rollouts. Only
  training seeds: below 100,000 or a train demo's (``ctx.bigym_seed_ok``); the eval's seeds (780,000+) raise.

This starting recipe is the fine-tune that beat random on ReachTarget (``modal_app.py`` bigym-ft, 20% vs 5%; 0% on
the cupboards), cut to the 15-minute H100 budget: the 20-layer long-sep24-b64 checkpoint, vision tower frozen,
head_max_len 320, about half the draws BiGym (4-frame control mostly, 1-frame control, a little probe), 22.5% the
game replay, the rest the VQA / score pool.
"""
from typing import Dict

# -- the recipe ---------------------------------------------------------------------------------------------------

INIT = "autoresearch/full/long-sep24-b64/best"   # the 20-layer model (ctx.ckpt_path: under /ckpt/smolvlm, then /ckpt)
HEAD_MAX_LEN = 320        # question + options budget: the 37-option BiGym question with room to spare
BIGYM_FRAMES = 4          # head frames the benchmark shows the model per decision (written to the saved config)

# sampling weights within the non-game draws (as the fine-tune: BiGym 45 of 70 -> ~50% of all draws)
MIX: Dict[str, float] = {"bigym_v2c_bc_f4": 30.0, "bigym_v2c_bc_f1": 10.0, "bigym_v2c_probe": 5.0,
                         "score_vlfeedback": 3.0}
GAME_FRAC = 0.225         # share of draws for the game replay (toolkit + the pool's expert frames)
CONTROL_GAMES = ("CartPole", "Acrobot", "MountainCar", "LunarLander")

FREEZE = "full"           # everything but the vision tower
LR_HEAD = 5e-5
LR_BACKBONE = 1e-5
BATCH_SIZE = 64           # fits an H100's 80 GB with 4-image records (a 40 GB A100 does not)
WARMUP_STEPS = 40
NUM_WORKERS = 14
PREFETCH = 2


def _loader_fit(batch_size: int, workers: int, prefetch: int, images: int = 4, side: int = 512):
    """(workers, prefetch) so the batches in flight fit in half of /dev/shm (a batch pads every example to its
    most images: float32 pixels, ``images`` x 3 x side^2 each), as modal_app._loader_fit does."""
    import shutil

    per_batch = batch_size * images * 3 * side * side * 4
    try:
        fit = int(shutil.disk_usage("/dev/shm").total * 0.5 // per_batch)
    except OSError:
        return workers, prefetch
    while prefetch > 1 and workers * prefetch > fit:
        prefetch //= 2
    return max(2, min(workers, fit // max(1, prefetch))), prefetch


# -- the two entry points the harness calls ----------------------------------------------------------------------

def build(ctx):
    import toolkit
    from laya.vlm import VLMAgent

    agent = VLMAgent(ctx.ckpt_path(INIT), device=ctx.device, head_max_len=HEAD_MAX_LEN)
    agent.cfg["bigym_frames"] = BIGYM_FRAMES
    data = ctx.train_examples() + ctx.bigym_examples()
    games = toolkit.maze_examples(20000) + toolkit.snake_examples(20000) + ctx.game_examples()
    for g in CONTROL_GAMES:
        games += toolkit.control_examples(g, 5000)
    ctx.data, ctx.mix = toolkit.game_mix(data, games, GAME_FRAC, base_weights=MIX)
    return agent


def train(agent, ctx):
    from laya.vlm_train import train as train_loop

    workers, prefetch = _loader_fit(BATCH_SIZE, NUM_WORKERS, PREFETCH)
    print("loader: %d workers, prefetch %d" % (workers, prefetch), flush=True)
    stats = {}
    train_loop(agent.model, agent.processor, ctx.data, steps=10**9, batch_size=BATCH_SIZE, freeze=FREEZE,
               lr_head=LR_HEAD, lr_backbone=LR_BACKBONE, warmup=WARMUP_STEPS, mix_weights=ctx.mix,
               max_minutes=ctx.time_budget_s / 60, num_workers=workers, prefetch_factor=prefetch, log_every=50,
               device=ctx.device, stats=stats)
    print("train stats: %s" % {k: v for k, v in stats.items() if k != "samples_per_dataset"}, flush=True)
    print("samples: %s" % stats.get("samples_per_dataset"), flush=True)
