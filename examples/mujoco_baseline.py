"""A model's baseline on every MuJoCo game (``laya.mujocogames``): a video of its first episode per game, and its
mean return against random play, doing nothing (always ``NONE``) and the expert, on the same seeded episodes.

    pip install -e . torchvision "gymnasium[mujoco]" "imageio[ffmpeg]" stable-baselines3 sb3-contrib
    python examples/mujoco_baseline.py --model thaitea/laya-vision \\
        --revision 8b318c99d7ad3ce19c24369263463882eada9d1e --out mujoco-baseline/

Writes ``<out>/<game>-model.webm`` and ``<out>/baseline.json`` (one row per game, with the model's resolved
revision), and prints a table. ``modal run modal_app.py::mujoco_eval`` runs every game at once on GPUs. ``--max-steps`` caps every episode (the model runs about 1 s a step on a CPU, and
most games run to 1000 steps); ``normalized`` = (model - random) / (expert - random), blank where a game has no
expert. See ``examples/mujoco_video.py`` for one game at a time.
"""
import argparse
import json
import os

from laya.mujocogames import GAMES, baseline, baseline_table


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="thaitea/laya-vision", help="checkpoint dir or Hub id")
    ap.add_argument("--revision", default=None, help="Hub commit to load --model at")
    ap.add_argument("--device", default=None)
    ap.add_argument("--games", default=",".join(GAMES))
    ap.add_argument("--episodes", type=int, default=3, help="model episodes per game (the first is recorded)")
    ap.add_argument("--baseline-episodes", type=int, default=10, help="random / still / expert episodes per game")
    ap.add_argument("--seed", type=int, default=200_000)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--out", default="mujoco-baseline")
    args = ap.parse_args()

    from laya.vlm import load_vlm

    agent = load_vlm(args.model, device=args.device, revision=args.revision)
    os.makedirs(args.out, exist_ok=True)
    rows = []
    for game in [g for g in args.games.split(",") if g]:
        video = os.path.join(args.out, "%s-model.webm" % game)
        row = baseline(agent, game, video, args.episodes, args.baseline_episodes, args.seed, args.max_steps)
        row.update(model=args.model, revision=agent.source.get("revision"), video=video)  # the resolved commit
        rows.append(row)
        print(json.dumps(row), flush=True)
        with open(os.path.join(args.out, "baseline.json"), "w") as f:
            json.dump(rows, f, indent=1)
    print("\n" + baseline_table(rows))


if __name__ == "__main__":
    main()
