"""A model's baseline on every MuJoCo game (``laya.mujocogames``): a video of its first episode per game, and its
mean return against random play, doing nothing (always ``NONE``) and the expert, on the same seeded episodes.

    pip install -e . torchvision "gymnasium[mujoco]" "imageio[ffmpeg]"
    python examples/mujoco_baseline.py --model thaitea/laya-vision \\
        --revision 8b318c99d7ad3ce19c24369263463882eada9d1e --out mujoco-baseline/

Writes ``<out>/<game>-model.webm`` and ``<out>/baseline.json`` (one row per game, with the model's resolved
revision), and prints a table. ``--max-steps`` caps every episode (the model runs about 1 s a step on a CPU, and
most games run to 1000 steps); ``normalized`` = (model - random) / (expert - random), blank where a game has no
expert. See ``examples/mujoco_video.py`` for one game at a time.
"""
import argparse
import json
import os
import time

from mujoco_video import model_chooser, record

from laya.mujocogames import (GAMES, MujocoGame, expert_policy, has_expert, normalized, play_episodes,
                              random_policy, still_policy)


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
        t0 = time.time()
        choose = model_chooser(agent, game)
        video = os.path.join(args.out, "%s-model.webm" % game)
        env = MujocoGame(game, args.seed)
        record(env, choose, "model", video, args.max_steps)  # episode 1, also scored below
        first = {"score": round(env.score, 3), "steps": env.steps, "terminated": env.terminated}
        env.close()
        cap = {"max_steps": args.max_steps}
        rest = play_episodes(game, lambda e: choose(e)[0], args.episodes - 1, args.seed + 1, **cap) \
            if args.episodes > 1 else {"results": [], "actions": {}}
        model = [first] + rest["results"]
        base = {"episodes": args.baseline_episodes, "seed": args.seed, **cap}
        rnd = play_episodes(game, random_policy(args.seed), **base)
        still = play_episodes(game, still_policy, **base)
        exp = play_episodes(game, expert_policy, **base) if has_expert(game) else None
        m = sum(e["score"] for e in model) / len(model)
        row = {"game": game, "model": args.model, "revision": agent.source.get("revision"),  # the resolved commit
               "model_score": round(m, 3), "model_scores": [e["score"] for e in model],
               "model_steps": [e["steps"] for e in model], "random_score": round(rnd["mean_score"], 3),
               "still_score": round(still["mean_score"], 3),
               "expert_score": None if exp is None else round(exp["mean_score"], 3),
               "normalized": None if exp is None else normalized(m, rnd["mean_score"], exp["mean_score"]),
               "max_steps": args.max_steps, "seed": args.seed, "video": video, "seconds": round(time.time() - t0, 1)}
        rows.append(row)
        print(json.dumps(row), flush=True)
        with open(os.path.join(args.out, "baseline.json"), "w") as f:
            json.dump(rows, f, indent=1)

    print("\n%-22s %10s %10s %10s %10s %7s" % ("game", "model", "random", "still", "expert", "norm"))
    for r in rows:
        ex = "-" if r["expert_score"] is None else "%.1f" % r["expert_score"]
        nm = "-" if r["normalized"] is None else "%.2f" % r["normalized"]
        print("%-22s %10.1f %10.1f %10.1f %10s %7s" % (r["game"], r["model_score"], r["random_score"],
                                                      r["still_score"], ex, nm))


if __name__ == "__main__":
    main()
