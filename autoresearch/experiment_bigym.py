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
BIGYM_FRAMES = 1          # head frames the benchmark shows the model per decision (written to the saved config)

# sampling weights within the non-game draws (as the fine-tune: BiGym 45 of 70 -> ~50% of all draws)
MIX: Dict[str, float] = {"bigym_v2c_bc_f1": 40.0, "bigym_v2c_probe": 5.0,
                         "score_vlfeedback": 3.0}
GAME_FRAC = 0.225         # share of draws for the game replay (toolkit + the pool's expert frames)
CONTROL_GAMES = ("CartPole", "Acrobot", "MountainCar", "LunarLander")

FREEZE = "full"           # everything but the vision tower
LR_HEAD = 1e-4
LR_BACKBONE = 1.5e-5
BATCH_SIZE = 64           # fits an H100's 80 GB with 4-image records (a 40 GB A100 does not)
WARMUP_STEPS = 40
NUM_WORKERS = 26
PREFETCH = 2

# reach tasks have no BC data: roll the privileged reach oracle out on training seeds inside the budget, with some
# random moves so it also sees off-path states (every visited state labelled with the oracle's move)
REACH_TASKS = ("ReachTarget", "ReachTargetSingle")
REACH_ROLLOUT_S = 150     # wall seconds of rollouts, counted in the 15-minute budget
REACH_PROCS = 30          # rollout processes (OSMesa rendering, CPU only)
REACH_EXPLORE = 0.3       # chance of playing a random primitive instead of the oracle's (the label stays the oracle's)
REACH_WEIGHT = 1.0        # sampling weight relative to bigym_v2c_bc_f1

_REACH_WORKER = r"""
import io, os, pickle, random, sys, time
os.environ["MUJOCO_GL"] = os.environ["PYOPENGL_PLATFORM"] = "osmesa"
from PIL import Image
from laya import bigymgames as bg
from laya.bigymdata import label_index

wid, deadline, frames, explore, out_path = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
jobs = pickle.loads(bytes.fromhex(sys.argv[6]))
rng = random.Random(wid)
envs, out, episodes = {}, [], 0
for task, seed in jobs:
    if time.time() > deadline:
        break
    if task not in envs:
        envs[task] = bg.make_env(task, cameras=True)
    g = bg.BiGymGame(task, seed, env=envs[task])
    enc = {}
    while not g.done and g.decisions < bg.TASKS[task]["max_decisions"] and time.time() < deadline:
        a = bg.oracle_action(g)
        imgs = []
        for f in g.frames(frames):
            if id(f) not in enc:
                buf = io.BytesIO()
                Image.fromarray(f).save(buf, format="JPEG", quality=90)
                enc[id(f)] = buf.getvalue()
            imgs.append(enc[id(f)])
        out.append((task, seed, g.decisions, imgs, label_index(a)))
        g.step(a if rng.random() >= explore else rng.choice(g.actions))
    episodes += 1
with open(out_path, "wb") as fh:
    pickle.dump({"records": out, "episodes": episodes}, fh)
"""


def reach_rollouts(ctx, seconds: float, procs: int, frames: int, explore: float):
    """Oracle-labelled reach examples from ``procs`` subprocesses rolling out for ``seconds`` on training seeds."""
    import os
    import pickle
    import random
    import subprocess
    import sys
    import tempfile
    import time

    from laya.bigymgames import bigym_question
    from laya.vlm import VLMAgent

    rng = random.Random(0)
    tmp = tempfile.mkdtemp()
    script = os.path.join(tmp, "reach_worker.py")
    with open(script, "w") as f:
        f.write(_REACH_WORKER)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in sys.path if p))
    deadline = time.time() + seconds
    runs = []
    for w in range(procs):
        jobs = [(REACH_TASKS[(w + i) % 2], rng.randrange(1, 100_000)) for i in range(400)]
        assert all(ctx.bigym_seed_ok(s) for _, s in jobs)
        out = os.path.join(tmp, "out-%d.pkl" % w)
        runs.append((out, subprocess.Popen([sys.executable, script, str(w), str(deadline), str(frames), str(explore),
                                            out, pickle.dumps(jobs).hex()], env=env)))
    qs = {t: VLMAgent._to_internal(bigym_question(t, frames)["action"]) for t in REACH_TASKS}
    exs, episodes = [], 0
    for out, p in runs:
        if p.wait(timeout=seconds + 300) != 0:
            print("reach worker %s failed (exit %s)" % (out, p.returncode), flush=True)
            continue
        with open(out, "rb") as f:
            got = pickle.load(f)
        episodes += got["episodes"]
        for task, seed, d, imgs, label in got["records"]:
            k = len(qs[task]["crit"])
            state = {"images": imgs} if frames > 1 else {"image": imgs[-1]}
            exs.append({"state": state, "q": qs[task], "target": [float(i == label) for i in range(k)],
                        "label": label, "dataset": "bigym_reach_oracle", "id": "%s-%d-%d" % (task, seed, d)})
    print("reach rollouts: %d examples from %d episodes in %.0f s" % (len(exs), episodes,
                                                                      seconds - (deadline - time.time())), flush=True)
    return exs


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
    data = ctx.train_examples() + ctx.bigym_examples(names=("bigym_v2c_bc_f1", "bigym_v2c_probe"))
    games = toolkit.maze_examples(20000) + toolkit.snake_examples(20000) + ctx.game_examples()
    for g in CONTROL_GAMES:
        games += toolkit.control_examples(g, 5000)
    ctx.data, ctx.mix = toolkit.game_mix(data, games, GAME_FRAC, base_weights=MIX)
    return agent


def train(agent, ctx):
    import time

    from laya.vlm_train import train as train_loop

    t0 = time.time()
    reach = reach_rollouts(ctx, REACH_ROLLOUT_S, REACH_PROCS, BIGYM_FRAMES, REACH_EXPLORE)
    data, mix = list(ctx.data), dict(ctx.mix)
    if reach:
        data += reach
        mix["bigym_reach_oracle"] = REACH_WEIGHT * mix["bigym_v2c_bc_f1"]
    left_min = (ctx.time_budget_s - (time.time() - t0)) / 60
    workers, prefetch = _loader_fit(BATCH_SIZE, NUM_WORKERS, PREFETCH, images=BIGYM_FRAMES)
    print("loader: %d workers, prefetch %d" % (workers, prefetch), flush=True)
    stats = {}
    train_loop(agent.model, agent.processor, data, steps=10**9, batch_size=BATCH_SIZE, freeze=FREEZE,
               lr_head=LR_HEAD, lr_backbone=LR_BACKBONE, warmup=WARMUP_STEPS, mix_weights=mix,
               max_minutes=left_min, num_workers=workers, prefetch_factor=prefetch, log_every=50,
               device=ctx.device, stats=stats)
    print("train stats: %s" % {k: v for k, v in stats.items() if k != "samples_per_dataset"}, flush=True)
    print("samples: %s" % stats.get("samples_per_dataset"), flush=True)
