"""Record a MuJoCo episode (``laya.mujocogames``) to a video: the screen, the chosen push and, for a model, its
probabilities over the pushes.

    pip install -e . torchvision "gymnasium[mujoco]" "imageio[ffmpeg]"
    python examples/mujoco_video.py --policy expert --out expert.webm
    python examples/mujoco_video.py --game Hopper --policy random --out random.webm
    python examples/mujoco_video.py --game HalfCheetah --policy model --model thaitea/laya-vision \\
        --revision 8b318c99d7ad3ce19c24369263463882eada9d1e --out model.webm

The model sees what a games-suite run sees: the rendered screen with the previous frame ghosted in, and
``laya.games.control_question``. On a machine with no display, rendering uses EGL (``apt-get install libegl1``) or
``MUJOCO_GL=osmesa`` (``libosmesa6``). The suffix of ``--out`` picks the container: .webm (VP9) or .mp4 (H.264).
The video plays in real time (the environment's step length). ``examples/mujoco_baseline.py`` records and scores
every game.
"""
import argparse

import numpy as np
from PIL import Image, ImageDraw

from laya.games import control_question
from laya.mujocogames import GAMES, MujocoGame, random_policy, still_policy

PANEL_W = 300
CHOSEN, OTHER = (255, 200, 80), (90, 90, 110)


def draw(frame: Image.Image, env: MujocoGame, action: str, probs, label: str) -> np.ndarray:
    scale = 2
    w, h = frame.width * scale, frame.height * scale
    img = Image.new("RGB", (w + PANEL_W, h), (24, 24, 28))
    img.paste(frame.resize((w, h), Image.NEAREST), (0, 0))
    d = ImageDraw.Draw(img)
    x = w + 12
    d.text((x, 12), "%s  (%s)" % (env.game, label), fill=(230, 230, 230))
    d.text((x, 30), "step %d   return %.1f" % (env.steps, env.score), fill=(170, 170, 170))
    row = min(26, (h - 60) // len(env.actions))  # 35 options for Humanoid
    for i, a in enumerate(env.actions):
        y = 56 + row * i
        chosen = a == action
        d.text((x, y), a, fill=CHOSEN if chosen else (200, 200, 200))
        p = probs.get(a, 0.0) if probs else (1.0 if chosen else 0.0)
        d.rectangle((x + 160, y + 1, x + 160 + max(1, int(120 * p)), y + row - 3), fill=CHOSEN if chosen else OTHER)
    return np.asarray(img)


def model_chooser(agent, game: str):
    """``choose(env) -> (action, probabilities)`` from the model's answer on the ghosted screen."""
    question = control_question(game)

    def choose(env):
        ans = agent.predict({"image": env.render()}, question)["answers"]["action"]
        return ans["choice"], ans.get("probabilities")
    return choose


def record(env: MujocoGame, choose, label: str, out_path: str, max_steps: int = 0) -> None:
    """Play ``env`` to its end (or ``max_steps``) with ``choose(env) -> (action, probabilities or None)``, writing
    each screen the policy saw to ``out_path``."""
    import imageio.v2 as imageio

    webm = out_path.endswith(".webm")
    codec = {"codec": "libvpx-vp9", "ffmpeg_params": ["-b:v", "0", "-crf", "32"]} if webm else {}
    fps = round(1 / env.env.unwrapped.dt)
    with imageio.get_writer(out_path, fps=fps, macro_block_size=1, **codec) as out:
        while not env.done and not (max_steps and env.steps >= max_steps):
            frame = env.render()  # the ghosted screen, as the model sees it
            action, probs = choose(env)
            out.append_data(draw(frame, env, action, probs, label))
            env.step(action)
        out.append_data(draw(env.render(), env, "", None, label + (" - ended" if env.terminated else "")))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--game", default="InvertedPendulum", choices=sorted(GAMES))
    ap.add_argument("--policy", default="expert", choices=("expert", "random", "still", "model"))
    ap.add_argument("--model", default="thaitea/laya-vision", help="checkpoint dir or Hub id (--policy model)")
    ap.add_argument("--revision", default=None, help="Hub commit to load --model at")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=200_000)
    ap.add_argument("--max-steps", type=int, default=300, help="stop the recording here (0 = the episode's end)")
    ap.add_argument("--out", default="mujoco.webm")
    args = ap.parse_args()

    env = MujocoGame(args.game, args.seed)
    if args.policy == "model":
        from laya.vlm import load_vlm

        choose = model_chooser(load_vlm(args.model, device=args.device, revision=args.revision), args.game)
    else:
        pick = {"random": random_policy(args.seed), "still": still_policy,
                "expert": lambda e: e.expert()}[args.policy]
        choose = lambda e: (pick(e), None)  # noqa: E731
    record(env, choose, args.policy, args.out, args.max_steps)
    print("%s %s: %d steps, return %.1f, %s -> %s" % (args.game, args.policy, env.steps, env.score,
                                                    "ended" if env.terminated else "running", args.out))
    env.close()


if __name__ == "__main__":
    main()
