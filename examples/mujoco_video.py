"""Record a MuJoCo episode (``laya.mujocogames``) to a video: the screen, the chosen push and, for a model, its
probabilities over the pushes.

    pip install -e . torchvision "gymnasium[mujoco]" "imageio[ffmpeg]" stable-baselines3 sb3-contrib
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

from laya.mujocogames import GAMES, MujocoGame, model_chooser, random_policy, record, still_policy


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
    ap.add_argument("--views", default=None,
                    help="cameras from laya.mujocogames.VIEWS, comma-separated, or 'all' (default: the single view)")
    args = ap.parse_args()

    env = MujocoGame(args.game, args.seed, views=args.views)
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
