"""Record a MuJoCo pendulum episode (``laya.mujocogames``) to a video: the screen, the chosen push and, for a model,
its probabilities over the pushes.

    pip install -e . torchvision "gymnasium[mujoco]" "imageio[ffmpeg]"
    python examples/mujoco_video.py --policy expert --out expert.webm
    python examples/mujoco_video.py --game InvertedDoublePendulum --policy random --out random.webm
    python examples/mujoco_video.py --policy model --model thaitea/laya-vision \\
        --revision 8b318c99d7ad3ce19c24369263463882eada9d1e --out model.webm

The model sees what a games-suite run sees: the rendered screen with the previous frame ghosted in, and
``laya.games.control_question``. On a machine with no display, rendering uses EGL (``apt-get install libegl1``) or
``MUJOCO_GL=osmesa`` (``libosmesa6``). The suffix of ``--out`` picks the container: .webm (VP9) or .mp4 (H.264).
"""
import argparse

import numpy as np
from PIL import Image, ImageDraw

from laya.games import control_question
from laya.mujocogames import GAMES, MujocoGame, random_policy

PANEL_W = 220


def draw(frame: Image.Image, env: MujocoGame, action: str, probs, label: str) -> np.ndarray:
    scale = 2
    w, h = frame.width * scale, frame.height * scale
    img = Image.new("RGB", (w + PANEL_W, h), (24, 24, 28))
    img.paste(frame.resize((w, h), Image.NEAREST), (0, 0))
    d = ImageDraw.Draw(img)
    x = w + 12
    d.text((x, 12), "%s  (%s)" % (env.game, label), fill=(230, 230, 230))
    d.text((x, 30), "step %d   return %.0f" % (env.steps, env.score), fill=(170, 170, 170))
    for i, a in enumerate(env.actions):
        y = 60 + 26 * i
        chosen = a == action
        d.text((x, y), a, fill=(255, 200, 80) if chosen else (200, 200, 200))
        p = probs.get(a) if probs else (1.0 if chosen else 0.0)
        d.rectangle((x + 100, y + 2, x + 100 + int(100 * p), y + 14), fill=(255, 200, 80) if chosen else (90, 90, 110))
    return np.asarray(img)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--game", default="InvertedPendulum", choices=sorted(GAMES))
    ap.add_argument("--policy", default="expert", choices=("expert", "random", "model"))
    ap.add_argument("--model", default="thaitea/laya-vision", help="checkpoint dir or Hub id (--policy model)")
    ap.add_argument("--revision", default=None, help="Hub commit to load --model at")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=200_000)
    ap.add_argument("--max-steps", type=int, default=300, help="stop the recording here (0 = the episode's end)")
    ap.add_argument("--out", default="mujoco.webm")
    args = ap.parse_args()

    import imageio.v2 as imageio

    env = MujocoGame(args.game, args.seed)
    if args.policy == "model":
        from laya.vlm import load_vlm

        agent = load_vlm(args.model, device=args.device, revision=args.revision)
        question = control_question(args.game)

        def choose(e):
            ans = agent.predict({"image": e.render()}, question)["answers"]["action"]
            return ans["choice"], ans.get("probabilities")
    elif args.policy == "random":
        pick = random_policy(args.seed)
        choose = lambda e: (pick(e), None)  # noqa: E731
    else:
        choose = lambda e: (e.expert(), None)  # noqa: E731

    codec = {"codec": "libvpx-vp9", "ffmpeg_params": ["-b:v", "0", "-crf", "32"]} if args.out.endswith(".webm") else {}
    fps = round(1 / env.env.unwrapped.dt)  # real time: 25 for the single pole, 20 for the double
    with imageio.get_writer(args.out, fps=fps, macro_block_size=1, **codec) as out:
        while not env.done and not (args.max_steps and env.steps >= args.max_steps):
            frame = env.render()  # the ghosted screen, as the model sees it
            action, probs = choose(env)
            out.append_data(draw(frame, env, action, probs, args.policy))
            env.step(action)
        out.append_data(draw(env.render(), env, "", None, args.policy + (" - fell" if env.terminated else "")))
    print("%s %s: %d steps, return %.1f, %s -> %s" % (args.game, args.policy, env.steps, env.score,
                                                    "fell" if env.terminated else "still up", args.out))
    env.close()


if __name__ == "__main__":
    main()
