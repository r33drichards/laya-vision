"""Gradio Space: Laya Vision drives a simulated Microduck from its own camera, in quackd's 2D or 3D simulator.

Each step the model sees the duck's camera frame and answers one `choice` question over four actions. The simulators
are quackd's (0.14.0 on PyPI): `microduck:sim2d`, the flat cartoon, and `microduck:mujoco`, MuJoCo physics with
upstream's Microduck model walking on upstream's trained policy. The action space, timings and question are the ones
`scripts/eval_microduck.py` in r33drichards/quackd scores and the checkpoint was trained on, so what plays here is
what was measured there.

Real time: an action plays out at wall-clock speed, the view streaming as it goes, and the simulator waits only for
the model's decision between actions, as the eval does (there, a decision takes no simulated time). How the physics
is chunked to keep pace never changes where the duck goes: commands are re-sent on the simulator's own clock, exactly
as the eval's transport does.

One episode is one ZeroGPU call. `play_episode` is a generator under `@spaces.GPU`: ZeroGPU runs it in a forked
worker that holds the GPU, and it builds its own world from the seed there, plays to the end and streams frames back.
Calling the GPU once per decision instead counted every decision as a ZeroGPU run, and a visitor ran out of runs a
few decisions into the first episode. Reset only draws the starting position, on the CPU, in this process.

MuJoCo renders through OSMesa, whose context belongs to the thread that made it, and Gradio runs handlers on a pool
of threads, so each game sends all of its 3D calls through one thread of its own (`game["gl"]`): `on_gl` for the
preview, and a thread the episode starts for itself inside the GPU worker.

quackd is installed at start-up without its dependencies: the Gradio SDK installs `gradio[mcp]`, which pins mcp<2,
and quackd 0.14.0 declares mcp>=2 for a server this Space never runs. What the simulators import (mujoco,
onnxruntime, numpy, pillow, pydantic) is in requirements.txt.
"""
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import spaces  # before torch: ZeroGPU patches it

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
# OSMesa's rasteriser (llvmpipe) keeps a pool of render threads; a process forked after the first render (the GPU
# worker, forked after the page's preview) inherits the pool's state but not its threads, and its first render waits
# for them forever. With no pool it renders on the calling thread.
os.environ.setdefault("LP_NUM_THREADS", "0")
_MAIN_PID = os.getpid()

_QUACKD = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".quackd-pkgs")
try:
    import quackd  # noqa: F401
    import quackd_microduck  # noqa: F401
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "--no-deps", "--target", _QUACKD,
                           "quackd==0.14.0", "quackd-microduck==0.14.0"])
    sys.path.insert(0, _QUACKD)

import gradio as gr  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

import laya  # noqa: E402
from quackd.sim2d.render import render_duckcam, render_topdown  # noqa: E402
from quackd.sim2d.world import DT as DT_2D  # noqa: E402
from quackd.sim2d.world import KICK_CONE_DEG, KICK_RANGE_M, World  # noqa: E402

MODEL_ID = os.environ.get("LAYA_MODEL", "thaitea/laya-vision-microduck-kick")
# Pinned: the Hub id can later name another checkpoint, and the Space only loads at startup. "" follows the id.
REVISION = os.environ.get("LAYA_REVISION", "7505ee2f2cf211cdff7faa2c66ef0c1472088e0b") or None


def usable_cpus():
    """The CPUs this container may use. ``os.cpu_count()`` is the host's: on a Space's 2 vCPUs it can report
    dozens, and torch running that many threads on two cores made every decision several times slower."""
    n = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    try:
        with open("/sys/fs/cgroup/cpu.max") as f:  # cgroup v2: "<quota> <period>" or "max <period>"
            quota, period = f.read().split()
        if quota != "max":
            n = min(n, max(1, math.ceil(int(quota) / int(period))))
    except (OSError, ValueError):
        pass
    return max(1, n)


torch.set_num_threads(usable_cpus())
agent = laya.load_vlm(MODEL_ID, device="cpu", revision=REVISION)
ON_GPU = spaces.config.Config.zero_gpu
if ON_GPU:  # ZeroGPU records the move here and makes it in the GPU worker
    agent.model.to("cuda")
    agent.device = torch.device("cuda")


def decide(image):
    """One decision: the model's answer for one duck-camera frame. Called inside `play_episode`, so on ZeroGPU it
    runs on the GPU worker's copy of the model; anywhere else, on the CPU."""
    answer = agent.predict({"image": image}, QUESTION)["answers"]["action"]
    return answer["choice"], {k: float(v) for k, v in answer["probabilities"].items()}


_gl = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mujoco")


def on_gl(fn, *args):
    """Run ``fn`` on the one thread that owns every MuJoCo world and its OSMesa context."""
    return _gl.submit(fn, *args).result()


def _prefetch_microduck():
    # the model and policies download on first use (about 10 MB); do it while the Space starts, not on a click
    from quackd_microduck.sim3d.assets import ensure_microduck

    ensure_microduck()


on_gl(_prefetch_microduck)

# ── the task, exactly as eval_microduck.py defines it ─────────────────────────────────────────────────────
ACTIONS = {
    "FORWARD": "walk forward",
    "LEFT": "turn left",
    "RIGHT": "turn right",
    "KICK": "kick the ball in front of you",
}
QUESTION = {"action": {
    "type": "choice",
    "instructions": (
        "You are a small duck robot looking through your own camera. Somewhere in the arena is "
        "an orange ball. Turn to find it, walk up to it until it is close and in front of you, "
        "then kick it. Which action should you take now?"
    ),
    "criteria": dict(ACTIONS),
}}
RESEND_S = 0.2      # the worlds' deadman stops a duck whose last command is older than 0.3 s
KICK_SETTLE_S = 1.5  # after a kick, time for the ball to roll
SUCCESS_M = 0.3     # find-and-kick.duck: the ball moved at least 0.3 m, by a kick that connected
MAX_STEPS = 60
FRAME_EVERY_S = 0.2  # wall seconds between redraws while an action plays out (a 3D frame costs ~0.1 s of CPU)
SIZE = 320

SIMS = {
    # eval_microduck.SIM2D
    "2D cartoon": {"twist": {"FORWARD": (0.25, 0.0, 0.0), "LEFT": (0.0, 0.0, 0.6), "RIGHT": (0.0, 0.0, -0.6)},
                   "action_s": {"FORWARD": 0.5, "LEFT": 0.5, "RIGHT": 0.5}, "dt": DT_2D},
    # eval_microduck.mujoco_sim("microduck")
    "3D physics": {"twist": {"FORWARD": (0.3, 0.0, 0.0), "LEFT": (0.0, 0.0, 0.8), "RIGHT": (0.0, 0.0, -0.8)},
                   "action_s": {"FORWARD": 1.0, "LEFT": 0.6, "RIGHT": 0.6}, "dt": 0.02},
}


# ── the two worlds behind one interface ───────────────────────────────────────────────────────────────────


def is_3d(game):
    return game["sim"] == "3D physics"


def ball_xy(world):
    return (world.ball.x, world.ball.y) if hasattr(world, "ball") else (world.ball_x, world.ball_y)


def fallen(world):
    d = world.ducks[0] if hasattr(world, "ducks") else world
    return d.posture == "fallen"


def cam(game, size):
    w = game["world"]
    if is_3d(game):
        from quackd_microduck.sim3d.render import render_headcam

        return game["gl"](render_headcam, w, size)
    return render_duckcam(w, size)


def top(game, size):
    w = game["world"]
    if is_3d(game):
        return game["gl"](_overview_3d, w, size)
    return render_topdown(w, size)


def _overview_3d(world, size):
    """quackd's ``render_overview`` without the floor's reflection, which doubled its cost (the robot's meshes are
    drawn twice). The viewer's picture only: the duck camera, which is what the model sees, is untouched."""
    import mujoco
    import numpy as np
    from quackd_microduck.sim3d import render as R

    cam = R._free_camera((world.x, world.y, R.OVERVIEW_HEIGHT), R.OVERVIEW_DISTANCE, R.OVERVIEW_AZIMUTH_DEG,
                         R.OVERVIEW_ELEVATION_DEG)
    was = float(world.model.vis.global_.fovy)
    world.model.vis.global_.fovy = R.OVERVIEW_FOV_DEG
    try:
        renderer = world.renderer(size)
        renderer.update_scene(world.data, camera=cam)
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 1
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
        return Image.fromarray(np.asarray(renderer.render()), "RGB")
    finally:
        world.model.vis.global_.fovy = was


def world_call(game, fn, *args):
    return game["gl"](fn, *args) if is_3d(game) else fn(*args)


def new_game(seed, sim="2D cartoon", gl=on_gl):
    try:  # the page can send the number as text
        seed = int(float(seed))
    except (TypeError, ValueError):
        seed = 0
    if sim == "3D physics":
        from quackd_microduck.sim3d.world import MujocoWorld

        world = gl(lambda: MujocoWorld(seed=seed, body="microduck"))
    else:
        world = World(seed=seed)
    return {"world": world, "sim": sim, "seed": seed, "steps": 0, "probs": {}, "last": None, "done": False, "gl": gl}


def close_game(game):
    if game and is_3d(game):
        game["gl"](game["world"].close)


def truth(world):
    dist, bearing = world.relative(*ball_xy(world))
    b = math.degrees(bearing)
    return dist, b, dist <= KICK_RANGE_M and abs(b) <= KICK_CONE_DEG


def won(world):
    return world.kicks_connected > 0 and world.ball_displacement_m >= SUCCESS_M


# ── drawing ───────────────────────────────────────────────────────────────────────────────────────────────


def draw(game, caption=""):
    world = game["world"]
    frame = Image.new("RGB", (2 * SIZE + 8, SIZE + 28), (24, 24, 28))
    frame.paste(top(game, SIZE), (0, 28))
    frame.paste(cam(game, SIZE), (SIZE + 8, 28))
    d = ImageDraw.Draw(frame)
    view = "arena, from a corner" if is_3d(game) else "arena, from above"
    d.text((6, 8), "%s   t=%.1fs   %s" % (view, world.t, caption), fill=(235, 235, 235))
    d.text((SIZE + 14, 8), "duck camera: what the model sees", fill=(170, 170, 170))
    return frame


def panel(game):
    world = game["world"]
    dist, bearing, kickable = world_call(game, truth, world)
    if won(world):
        status = "**Kicked it.** The ball moved %.2f m after %d steps." % (world.ball_displacement_m, game["steps"])
    elif fallen(world):
        status = "**The duck fell over** after %d steps. Nothing in these four actions stands it up; press Reset." % (
            game["steps"])
    elif game["steps"] >= MAX_STEPS:
        status = "**Out of steps.** %d is the eval's limit; press Reset to try again." % MAX_STEPS
    else:
        status = "Step %d of %d." % (game["steps"], MAX_STEPS)
    rows = ["| action | P(action) |", "|---|---:|"]
    best = max(game["probs"], key=game["probs"].get) if game["probs"] else None
    for name in ACTIONS:
        p = game["probs"].get(name)
        bar = "█" * int(round((p or 0) * 20))
        cell = "-" if p is None else "%.2f %s" % (p, bar)
        rows.append("| %s | %s |" % ("**%s**" % name if name == best else name, cell))
    truth_md = (
        "| ground truth (the model never sees this) | |\n|---|---:|\n"
        "| distance to ball | %.2f m |\n| bearing | %+.0f° |\n| kickable now | %s |\n"
        "| kicks connected / tried | %d / %d |\n| ball moved | %.2f m |"
        % (dist, bearing, "yes" if kickable else "no", world.kicks_connected, world.kicks,
           world.ball_displacement_m)
    )
    last = "" if game["last"] is None else "Last action: **%s** (%s)." % game["last"]
    return "%s %s\n\n%s\n\n%s" % (status, last, "\n".join(rows), truth_md)


# ── one step ──────────────────────────────────────────────────────────────────────────────────────────────


def _advance(world, twist, steps, dt, since_cmd):
    """Step ``world`` ``steps`` times, re-sending ``twist`` every RESEND_S of simulated time (None sends nothing);
    returns since_cmd. Where the chunks fall does not matter: the re-sends follow the simulator's clock."""
    for _ in range(steps):
        if twist is not None and since_cmd >= RESEND_S - 1e-9:
            world.set_velocity(*twist)
            since_cmd = 0.0
        world.step(dt)
        since_cmd += dt
    return since_cmd


def play_action(game, action):
    """Run one action at wall-clock speed, yielding at most every FRAME_EVERY_S so the view can redraw. The physics
    catches up with the clock before each frame, so a slow frame lowers the frame rate, never the duck's speed."""
    world, spec = game["world"], SIMS[game["sim"]]
    if action == "KICK":
        world_call(game, world.kick, "right")
        duration, twist = KICK_SETTLE_S, None
    else:
        duration, twist = spec["action_s"][action], spec["twist"][action]
    dt = spec["dt"]
    total = int(round(duration / dt))
    done, since_cmd = 0, RESEND_S
    start = last_frame = time.perf_counter()
    while done < total:
        due = min(total, int((time.perf_counter() - start) / dt) + 1)
        if due > done:
            since_cmd = world_call(game, _advance, world, twist, due - done, dt, since_cmd)
            done = due
        if fallen(world):
            break
        now = time.perf_counter()
        if now - last_frame >= FRAME_EVERY_S:
            last_frame = now
            yield
        else:
            time.sleep(min(dt, FRAME_EVERY_S - (now - last_frame)))
    world_call(game, world.stop)


def take_turn(game, override=None):
    """One decision: the model's, or the button the viewer pressed. Yields (image, panel) as it plays out."""
    if game["done"]:
        yield draw(game, "press Reset"), panel(game)
        return
    if override is None:
        t0 = time.perf_counter()
        action, game["probs"] = decide(cam(game, 256))
        ms = (time.perf_counter() - t0) * 1000
        game["last"] = (action, "model on %s, %.0f ms" % ("GPU" if ON_GPU else "CPU", ms))
    else:
        action = override
        game["probs"] = {}
        game["last"] = (action, "you")
    game["steps"] += 1
    caption = "step %d: %s" % (game["steps"], action)
    for _ in play_action(game, action):
        yield draw(game, caption), panel(game)
    world = game["world"]
    game["done"] = won(world) or fallen(world) or game["steps"] >= MAX_STEPS
    yield draw(game, caption), panel(game)


# ── the page ──────────────────────────────────────────────────────────────────────────────────────────────


def on_reset(seed, sim, game):
    close_game(game)
    game = new_game(seed, sim)
    return game, draw(game, "ready"), panel(game)


def episode_seconds(seed, sim):
    """GPU time to ask for: the 60-step limit at each simulator's pace, and no more, so a visitor's quota covers it."""
    return 80 if sim == "3D physics" else 45


@spaces.GPU(duration=episode_seconds)
def play_episode(seed, sim):
    """A whole episode in one GPU call: a fresh world from the seed, played in real time to a kick, a fall or the step
    limit. Yields (frame, panel) as it goes; nothing unpicklable leaves, because on ZeroGPU this runs in a worker."""
    if os.getpid() != _MAIN_PID:
        # a forked GPU worker: torch's CPU thread pool (used here while the model loaded) did not survive the fork,
        # and the image preprocessing's first parallel op would wait on it forever. One thread never touches it.
        torch.set_num_threads(1)
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="episode-gl") as ex:
        game = new_game(seed, sim, gl=lambda fn, *args: ex.submit(fn, *args).result())
        try:
            yield draw(game, "starting"), panel(game)
            while not game["done"]:
                for img, md in take_turn(game):
                    yield img, md
        finally:
            close_game(game)


INTRO = """
# Laya Vision plays find-and-kick

A simulated [Microduck](https://github.com/pollen-robotics/microduck) in [quackd](https://github.com/r33drichards/quackd),
driven by [thaitea/laya-vision-microduck-kick](https://huggingface.co/thaitea/laya-vision-microduck-kick).
Each step the model sees only the duck's camera (the right-hand picture) and chooses one of four actions. The view of
the arena and the ground-truth numbers are there for you; the model never sees them.

**3D physics** is MuJoCo with upstream's Microduck model walking on upstream's trained policy: the walk is slow and
lurching, and the duck can fall. **2D cartoon** is quackd's flat simulator. **Play** runs a whole episode on a GPU in
real time, until the duck kicks the ball, falls or uses its 60 steps: typically 10 to 40 seconds. Pick a simulator and
a seed, then press Play.
"""

with gr.Blocks(title="Laya Vision: Microduck find-and-kick") as demo:
    gr.Markdown(INTRO)
    game = gr.State()
    with gr.Row():
        sim = gr.Radio(list(SIMS), value="3D physics", label="Simulator", scale=2)
        seed = gr.Textbox(value="0", label="Seed (0-9 are the eval's; any integer works)", scale=2)
        reset = gr.Button("Reset", scale=1)
    view = gr.Image(type="pil", format="webp", label="Simulator", interactive=False, show_label=False)
    with gr.Row():
        play = gr.Button("▶ Play", variant="primary")
        stop = gr.Button("■ Stop")
    info = gr.Markdown()

    outs = [game, view, info]
    demo.load(on_reset, [seed, sim, game], outs)
    reset.click(on_reset, [seed, sim, game], outs)
    sim.change(on_reset, [seed, sim, game], outs)
    seed.submit(on_reset, [seed, sim, game], outs)
    play_ev = play.click(play_episode, [seed, sim], [view, info])
    stop.click(None, cancels=[play_ev])

if __name__ == "__main__":
    # No server-side rendering: under it the page sent the seed as a string that gr.Number could not round, so every
    # visit failed at load, and it puts a Node proxy in front of every streamed frame.
    demo.queue(max_size=16, default_concurrency_limit=4).launch(show_error=True, ssr_mode=False)
