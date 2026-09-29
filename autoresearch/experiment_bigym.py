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
LR_BACKBONE = 2e-5
BATCH_SIZE = 64           # fits an H100's 80 GB with 4-image records (a 40 GB A100 does not)
WARMUP_STEPS = 40
NUM_WORKERS = 26
PREFETCH = 2

# DART-style data for the cupboard tasks, made while the model trains: subprocesses follow the train demos with
# laya.bigymdemos.follow, but play a random primitive instead of the follower's with probability DART_EPS, and record
# every visited state (4 head frames) labelled with the follower's choice there: states off the demo path, with
# the move that recovers. Phase 1 trains on the pool while they run; phase 2 adds their frames.
DART_TASKS = ("DrawerTopOpen", "DrawerTopClose", "WallCupboardOpen", "WallCupboardClose")
DART_S = 420              # wall seconds of rollouts (phase 1 trains meanwhile)
DART_PROCS = 10           # rollout processes (OSMesa), next to the loader's workers
DART_EPS = 0.25
DART_WEIGHT = 1.0         # sampling weight relative to bigym_v2c_bc_f1
DART_TEMP = 0.005         # soft target softmax(-cost / T) over the lookahead's costs (a 3 cm step moves cost ~0.03)

_DART_WORKER = r"""
import io, os, pickle, random, sys, time
os.environ["MUJOCO_GL"] = os.environ["PYOPENGL_PLATFORM"] = "osmesa"
from PIL import Image
from laya import bigymgames as bg
from laya import bigymdemos as bd
from laya.bigymdata import label_index

wid, deadline, frames, eps, out_path, jobs_path, temp = (int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]),
                                                         float(sys.argv[4]), sys.argv[5], sys.argv[6], float(sys.argv[7]))
with open(jobs_path, "rb") as fh:
    jobs = pickle.load(fh)
rng = random.Random(wid)
out, episodes, successes = [], 0, 0
cur = {}
orig_step = bg.BiGymGame.step


class Out(Exception):
    pass


orig_look = bd.lookahead
inlook = [False]


import numpy as np
last_scores = {}


def lookahead(game, target, banned=(), part=None):
    # bd.lookahead, keeping every primitive's cost (the soft target); the trial steps need no camera, so they get
    # the last observation instead of rendering 37 frames
    env, get = game.env, game.env.get_observation
    env.get_observation = lambda: game.obs
    inlook[0] = True
    try:
        snap = game.snapshot()
        scores = {}
        for p in bg.PRIMITIVES:
            if p in banned:
                continue
            game.step(p)
            scores[p] = bd.cost(bd._state(game), target) - (1e-4 if p == "STAY" else 0.0)
            if part is not None and bd.W_PART:
                scores[p] += bd.W_PART * float(np.abs(np.array(bd._part(game.env, game.task)) - part).sum())
            game.restore(snap)
    finally:
        inlook[0] = False
        env.get_observation = get
    last_scores.clear()
    last_scores.update(scores)
    return min(scores, key=scores.get)


def step(game, name):
    if inlook[0]:
        return orig_step(game, name)
    if time.time() > deadline:
        raise Out()
    enc, imgs = cur.setdefault("enc", {}), []
    for f in game.frames(frames):
        if id(f) not in enc:
            buf = io.BytesIO()
            Image.fromarray(f).save(buf, format="JPEG", quality=90)
            enc[id(f)] = buf.getvalue()
        imgs.append(enc[id(f)])
    c = np.array([last_scores.get(p, np.inf) for p in bg.PRIMITIVES])
    z = np.exp(-(c - c.min()) / temp)
    out.append((game.task, game_seed[0], game.decisions, imgs, label_index(name), (z / z.sum()).tolist()))
    return orig_step(game, name if rng.random() >= eps else rng.choice(game.actions))


bg.BiGymGame.step = step
bd.lookahead = lookahead
envs = {}
game_seed = [0]
for task, demo in jobs:
    if time.time() > deadline:
        break
    if task not in envs:
        envs[task] = bg.make_env(task, cameras=True)
    cur.clear()
    game_seed[0] = demo["seed"]
    try:
        r = bd.follow(task, demo, max_decisions=bg.TASKS[task]["max_decisions"], env=envs[task])
        successes += r["success"]
    except Out:
        break
    episodes += 1
with open(out_path, "wb") as fh:
    pickle.dump({"records": out, "episodes": episodes, "successes": successes}, fh)
"""


def start_dart(ctx, seconds: float, procs: int, frames: int, eps: float):
    """Launch the rollout subprocesses; returns what ``collect_dart`` needs."""
    import os
    import pickle
    import random
    import subprocess
    import sys
    import tempfile
    import time

    rng = random.Random(0)
    demos = [(t, d) for t in DART_TASKS for d in ctx.bigym_demos(t)]
    assert all(ctx.bigym_seed_ok(d["seed"]) for _, d in demos)
    tmp = tempfile.mkdtemp()
    script = os.path.join(tmp, "dart_worker.py")
    with open(script, "w") as f:
        f.write(_DART_WORKER)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in sys.path if p))
    deadline = time.time() + seconds
    runs = []
    for w in range(procs):
        jobs = [demos[rng.randrange(len(demos))] for _ in range(200)]
        jobs_path, out = os.path.join(tmp, "jobs-%d.pkl" % w), os.path.join(tmp, "out-%d.pkl" % w)
        with open(jobs_path, "wb") as f:
            pickle.dump(jobs, f)
        runs.append((out, subprocess.Popen([sys.executable, script, str(w), str(deadline), str(frames), str(eps),
                                            out, jobs_path, str(DART_TEMP)], env=env)))
    return runs, seconds, frames


def collect_dart(started):
    from laya.bigymgames import bigym_question
    from laya.vlm import VLMAgent

    runs, seconds, frames = started
    qs = {t: VLMAgent._to_internal(bigym_question(t, frames)["action"]) for t in DART_TASKS}
    exs, episodes, succ = [], 0, 0
    for out, p in runs:
        if p.wait(timeout=seconds + 300) != 0:
            print("dart worker %s failed (exit %s)" % (out, p.returncode), flush=True)
            continue
        import pickle

        with open(out, "rb") as f:
            got = pickle.load(f)
        episodes, succ = episodes + got["episodes"], succ + got["successes"]
        for task, seed, d, imgs, label, soft in got["records"]:
            state = {"images": imgs} if frames > 1 else {"image": imgs[-1]}
            exs.append({"state": state, "q": qs[task], "target": soft,
                        "label": label, "dataset": "bigym_dart", "id": "%s-%d-%d" % (task, seed, d)})
    print("dart rollouts: %d examples from %d finished episodes (%d successes)" % (len(exs), episodes, succ),
          flush=True)
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
    started = start_dart(ctx, DART_S, DART_PROCS, BIGYM_FRAMES, DART_EPS)
    workers, prefetch = _loader_fit(BATCH_SIZE, NUM_WORKERS, PREFETCH, images=BIGYM_FRAMES)
    print("loader: %d workers, prefetch %d" % (workers, prefetch), flush=True)

    def phase(data, mix, minutes, warmup):
        stats = {}
        train_loop(agent.model, agent.processor, data, steps=10**9, batch_size=BATCH_SIZE, freeze=FREEZE,
                   lr_head=LR_HEAD, lr_backbone=LR_BACKBONE, warmup=warmup, mix_weights=mix,
                   max_minutes=minutes, num_workers=workers, prefetch_factor=prefetch, log_every=50,
                   device=ctx.device, stats=stats)
        print("train stats: %s" % {k: v for k, v in stats.items() if k != "samples_per_dataset"}, flush=True)
        print("samples: %s" % stats.get("samples_per_dataset"), flush=True)

    phase(ctx.data, ctx.mix, DART_S / 60, WARMUP_STEPS)
    dart = collect_dart(started)
    data, mix = list(ctx.data), dict(ctx.mix)
    if dart:
        data += dart
        mix["bigym_dart"] = DART_WEIGHT * mix["bigym_v2c_bc_f1"]
    left = (ctx.time_budget_s - (time.time() - t0)) / 60
    if left > 0.5:
        phase(data, mix, left, 10)
