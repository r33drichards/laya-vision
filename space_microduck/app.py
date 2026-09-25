"""Gradio Space: Laya Vision drives a simulated Microduck from its own camera, in quackd's 2D simulator.

Each step the model sees the duck's camera frame and answers one `choice` question over four actions. The
simulator is quackd's `microduck:sim2d` world (quackd on PyPI); the action space, timings and question are the
ones `scripts/eval_microduck.py` in r33drichards/quackd scores and the checkpoint was trained on, so what plays
here is what was measured there.
"""
import math
import os
import time

import gradio as gr
import torch
from PIL import Image, ImageDraw

import laya
from quackd.sim2d.render import render_duckcam, render_topdown
from quackd.sim2d.world import DT, KICK_CONE_DEG, KICK_RANGE_M, World

MODEL_ID = os.environ.get("LAYA_MODEL", "thaitea/laya-vision-microduck-kick")
REVISION = os.environ.get("LAYA_REVISION", "") or None
torch.set_num_threads(max(1, os.cpu_count() or 1))
agent = laya.load_vlm(MODEL_ID, device="cpu", revision=REVISION)

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
TWIST = {"FORWARD": (0.25, 0.0, 0.0), "LEFT": (0.0, 0.0, 0.6), "RIGHT": (0.0, 0.0, -0.6)}
ACTION_S = 0.5      # one move: 0.125 m forward or about 17 degrees of turn
RESEND_S = 0.2      # the world's deadman stops a duck whose last command is older than 0.3 s
KICK_SETTLE_S = 1.5  # after a kick, time for the ball to roll
SUCCESS_M = 0.3     # find-and-kick.duck: the ball moved at least 0.3 m, by a kick that connected
MAX_STEPS = 60
FRAME_EVERY_S = 0.1  # how often the view redraws while an action plays out
SIZE = 320


def new_game(seed):
    return {"world": World(seed=int(seed)), "seed": int(seed), "steps": 0, "log": [], "probs": {}, "last": None,
            "done": False}


def truth(world):
    dist, bearing = world.relative(world.ball.x, world.ball.y)
    b = math.degrees(bearing)
    return dist, b, dist <= KICK_RANGE_M and abs(b) <= KICK_CONE_DEG


def won(world):
    return world.kicks_connected > 0 and world.ball_displacement_m >= SUCCESS_M


def draw(game, caption=""):
    world = game["world"]
    frame = Image.new("RGB", (2 * SIZE + 8, SIZE + 28), (24, 24, 28))
    frame.paste(render_topdown(world, SIZE), (0, 28))
    frame.paste(render_duckcam(world, SIZE), (SIZE + 8, 28))
    d = ImageDraw.Draw(frame)
    d.text((6, 8), "arena, from above   t=%.1fs   %s" % (world.t, caption), fill=(235, 235, 235))
    d.text((SIZE + 14, 8), "duck camera: what the model sees", fill=(170, 170, 170))
    return frame


def panel(game):
    world = game["world"]
    dist, bearing, kickable = truth(world)
    if won(world):
        status = "**Kicked it.** The ball moved %.2f m after %d steps." % (world.ball_displacement_m, game["steps"])
    elif game["steps"] >= MAX_STEPS:
        status = "**Out of steps.** %d is the eval's limit; press Reset to try again." % MAX_STEPS
    else:
        status = "Step %d of %d." % (game["steps"], MAX_STEPS)
    rows = ["| action | P(action) |", "|---|---:|"]
    top = max(game["probs"], key=game["probs"].get) if game["probs"] else None
    for name in ACTIONS:
        p = game["probs"].get(name)
        bar = "█" * int(round((p or 0) * 20))
        cell = "–" if p is None else "%.2f %s" % (p, bar)
        rows.append("| %s | %s |" % ("**%s**" % name if name == top else name, cell))
    truth_md = (
        "| ground truth (the model never sees this) | |\n|---|---:|\n"
        "| distance to ball | %.2f m |\n| bearing | %+.0f° |\n| kickable now | %s |\n"
        "| kicks connected / tried | %d / %d |\n| ball moved | %.2f m |"
        % (dist, bearing, "yes" if kickable else "no", world.kicks_connected, world.kicks,
           world.ball_displacement_m)
    )
    last = "" if game["last"] is None else "Last action: **%s** (%s)." % game["last"]
    return "%s %s\n\n%s\n\n%s" % (status, last, "\n".join(rows), truth_md)


def play_action(game, action):
    """Run one action through the world, yielding a redraw every FRAME_EVERY_S of sim time."""
    world = game["world"]
    if action == "KICK":
        world.kick("right")
        duration, twist = KICK_SETTLE_S, None
    else:
        duration, twist = ACTION_S, TWIST[action]
    t, since_cmd, since_frame = 0.0, RESEND_S, 0.0
    while t < duration - 1e-9:
        if twist is not None and since_cmd >= RESEND_S - 1e-9 and t < ACTION_S - 1e-9:
            world.set_velocity(*twist)
            since_cmd = 0.0
        world.step(DT)
        t += DT
        since_cmd += DT
        since_frame += DT
        if since_frame >= FRAME_EVERY_S - 1e-9:
            since_frame = 0.0
            yield
    world.stop()


def take_turn(game, override=None):
    """One decision: the model's, or the button the viewer pressed. Yields (image, panel) as it plays out."""
    world = game["world"]
    if game["done"]:
        yield draw(game, "press Reset"), panel(game)
        return
    if override is None:
        t0 = time.perf_counter()
        answer = agent.predict({"image": render_duckcam(world, 256)}, QUESTION)["answers"]["action"]
        ms = (time.perf_counter() - t0) * 1000
        action = answer["choice"]
        game["probs"] = {k: float(v) for k, v in answer["probabilities"].items()}
        game["last"] = (action, "model, %.0f ms" % ms)
    else:
        action = override
        game["probs"] = {}
        game["last"] = (action, "you")
    game["steps"] += 1
    caption = "step %d: %s" % (game["steps"], action)
    for _ in play_action(game, action):
        yield draw(game, caption), panel(game)
    game["done"] = won(world) or game["steps"] >= MAX_STEPS
    yield draw(game, caption), panel(game)


def on_reset(seed):
    game = new_game(seed)
    return game, draw(game, "ready"), panel(game)


def on_step(game):
    for img, md in take_turn(game):
        yield game, img, md


def on_play(game):
    while not game["done"]:
        for img, md in take_turn(game):
            yield game, img, md


def manual(action):
    def run(game):
        for img, md in take_turn(game, override=action):
            yield game, img, md
    return run


INTRO = """
# Laya Vision plays find-and-kick

A simulated [Microduck](https://github.com/pollen-robotics/microduck) in [quackd](https://github.com/r33drichards/quackd)'s
2D simulator, driven by [thaitea/laya-vision-microduck-kick](https://huggingface.co/thaitea/laya-vision-microduck-kick).
Each step the model sees only the duck's camera (the right-hand picture) and chooses one of four actions. It never
sees the arena from above. That view, and the ground-truth numbers, are there for you.

**Play** lets it run until it kicks the ball or uses its 60 steps. **Step** asks it for one action. The arrow buttons
let you take a step yourself, then hand back to the model. A step takes about a second of model time on this CPU.

On 100 seeds it never trained on, it kicks the ball in 95, the same as the scripted teacher it learned from.
It has only seen this 2D cartoon. It has not been tried on quackd's 3D MuJoCo simulator or a real camera.
"""

with gr.Blocks(title="Laya Vision: Microduck find-and-kick") as demo:
    gr.Markdown(INTRO)
    game = gr.State()
    with gr.Row():
        seed = gr.Number(value=0, precision=0, label="Seed (0-9 are the eval's seeds; any integer works)", scale=3)
        reset = gr.Button("Reset", scale=1)
    view = gr.Image(type="pil", label="Simulator", interactive=False, show_label=False)
    with gr.Row():
        play = gr.Button("▶ Play", variant="primary")
        step = gr.Button("Step (model)")
        stop = gr.Button("■ Stop")
    with gr.Row():
        left = gr.Button("↰ Left")
        fwd = gr.Button("↑ Forward")
        right = gr.Button("↱ Right")
        kick = gr.Button("⚽ Kick")
    info = gr.Markdown()

    outs = [game, view, info]
    demo.load(on_reset, seed, outs)
    reset.click(on_reset, seed, outs)
    seed.submit(on_reset, seed, outs)
    play_ev = play.click(on_play, game, outs)
    step.click(on_step, game, outs)
    for button, action in ((left, "LEFT"), (fwd, "FORWARD"), (right, "RIGHT"), (kick, "KICK")):
        button.click(manual(action), game, outs)
    stop.click(None, cancels=[play_ev])

if __name__ == "__main__":
    demo.queue(max_size=16, default_concurrency_limit=4).launch(show_error=True)
