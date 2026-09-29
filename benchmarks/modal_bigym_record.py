"""Record a checkpoint playing BiGym tasks as GIFs: the head camera the model sees next to a third-person view, with
the chosen primitive, the decision count and the task state on every frame (one frame per 0.1 s decision).

    timeout 3600 modal run benchmarks/modal_bigym_record.py --model autoresearch/bigym-sep29/a17717e --out <dir>

Plays greedy, one ``predict`` per decision, on the autoresearch benchmark's eval seeds (``780000 + 1000 * task
index + i``, ``bigym_eval.task_seeds``) under its caps. Per task it plays up to ``--tries`` episodes and keeps the
first that succeeds, else the first one; the GIF's caption says which (so a success shown is not a success rate:
read that from the benchmark). Writes ``<out>/<task>.gif`` and ``<out>/record.json``. Nothing is written to volumes.
"""
import json
import os
import sys

import modal

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from modal_bigym import TASKS, _agent, _pick_gl, ckpt_vol, hf_vol  # noqa: E402
from modal_bigym import image as _image  # noqa: E402

image = _image.add_local_python_source("modal_bigym")  # the container imports this file, and it imports modal_bigym

app = modal.App("laya-bigym-record")
SEED_BASE = 780_000  # autoresearch/bigym_eval.py
CAPS = {"ReachTarget": 60, "ReachTargetSingle": 60}  # others: 150 (bigym_eval.CAPS)


def _episode(agent, game, frames, cam, font, small):
    import numpy as np
    from PIL import Image, ImageDraw

    from laya import bigymgames as bg

    import mujoco

    task = game.task
    renderer = mujoco.Renderer(game._model, 256, 320)
    policy = bg.model_policy(agent, task, frames)
    cap = CAPS.get(task, 150)
    wall = bg.TASKS[task].get("part") == "wall"
    shots, moves = [], []

    def state_text():
        t = game.ground_truth()
        if "distance" in t:
            return "wrist to target %.0f cm" % (100 * t["distance"])
        return "%s %.0f%% open" % ("doors" if wall else "drawer", 100 * t["open"])

    def snap(move):
        pelvis = game._data.xpos[game._pelvis]
        cam.lookat[:] = [pelvis[0] + 0.45, pelvis[1], 1.35 if wall else 0.95]
        renderer.update_scene(game._data, camera=cam)
        third = renderer.render()
        head = np.array(Image.fromarray(game.frame()).resize((256, 256)))
        c = Image.new("RGB", (256 + 320 + 12, 256 + 60), (20, 26, 34))
        c.paste(Image.fromarray(head), (4, 56))
        c.paste(Image.fromarray(third), (264, 56))
        d = ImageDraw.Draw(c)
        d.text((8, 6), "%s  -  %s" % (task, bg.OPTION_WORDS[move] if move else "start"), font=font,
               fill=(255, 255, 255))
        status = "SUCCESS" if game.success else state_text()
        d.text((8, 30), "decision %d/%d  |  t = %.1f s  |  %s  |  %s" % (
            game.decisions, cap, game.decisions * bg.HOLD / 50, move or "-", status), font=small,
            fill=(120, 220, 140) if game.success else (160, 175, 190))
        d.text((8, 60), "head camera (the model's input)", font=small, fill=(255, 255, 255))
        d.text((268, 60), "third-person view", font=small, fill=(255, 255, 255))
        shots.append(c)

    snap(None)
    while not game.done and game.decisions < cap:
        move = policy(game)
        moves.append(move)
        game.step(move)
        snap(move)
    for _ in range(8):  # hold the last frame
        shots.append(shots[-1])
    ok = bool(game.success)
    renderer.close()
    game.close()
    return ok, shots, moves


@app.function(image=image, gpu="L4", cpu=4, timeout=60 * 60,
              volumes={"/cache/hf": hf_vol, "/ckpt": ckpt_vol.read_only()})
def record(task: str, model: str, tries: int = 6) -> dict:
    import io

    _pick_gl()
    import mujoco
    from PIL import Image, ImageFont

    from laya import bigymgames as bg

    agent = _agent(model, "")
    frames = int(agent.cfg.get("bigym_frames", 4))
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 15)
        small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except OSError:
        font = small = ImageFont.load_default()
    cam = mujoco.MjvCamera()
    cam.distance, cam.azimuth, cam.elevation = 2.4, -40, -22
    played, keep = [], None
    for i in range(tries):
        seed = SEED_BASE + 1000 * TASKS.index(task) + i
        ok, shots, moves = _episode(agent, bg.BiGymGame(task, seed), frames, cam, font, small)
        played.append({"seed": seed, "success": ok, "decisions": len(moves)})
        if keep is None or (ok and not keep[1]):
            keep = (seed, ok, shots, moves)
        if ok:
            break
    seed, ok, shots, moves = keep
    shots = [s.quantize(colors=128, method=Image.Quantize.MEDIANCUT) for s in shots]
    buf = io.BytesIO()
    shots[0].save(buf, format="GIF", save_all=True, append_images=shots[1:], duration=100, loop=0, optimize=True)
    return {"task": task, "gif": buf.getvalue(), "shown": {"seed": seed, "success": ok, "decisions": len(moves)},
            "played": played, "frames": frames}


@app.local_entrypoint()
def main(model: str = "autoresearch/bigym-sep29/a17717e", out: str = "bigym-recordings", tasks: str = ",".join(TASKS),
         tries: int = 6):
    os.makedirs(out, exist_ok=True)
    names = [t for t in tasks.split(",") if t]
    summary = {"model": model}
    for res in record.starmap([(t, model, tries) for t in names]):
        with open(os.path.join(out, res["task"] + ".gif"), "wb") as f:
            f.write(res["gif"])
        summary[res["task"]] = {k: res[k] for k in ("shown", "played", "frames")}
        print(res["task"], res["shown"], "of", len(res["played"]), "played", flush=True)
    with open(os.path.join(out, "record.json"), "w") as f:
        json.dump(summary, f, indent=1)
