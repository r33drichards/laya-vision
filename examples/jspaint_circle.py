"""Ask Laya Vision to draw a circle in JSPaint with the mouse only, and score it.

Each step the model sees two screenshots of the JSPaint canvas (8 steps ago and now, the cursor marked on both) plus
its pen state and last 12 actions as text, and a `choice` question over mouse actions (a 6 px move toward each of 32
compass directions, pen down, pen up, done) picks the next move, which is sent to the real app as pointer events.
The model is loaded with raised token budgets and asked with strict=True, so the full question is never cut. At the
end the circle verifier scores the true canvas pixels. The scripted expert and a random policy play the same seeds as
reference points, and the model's score is also reported normalized, (model - random) / (expert - random), as in the
games suite.

    pip install -e . playwright pillow
    git clone https://github.com/r33drichards/jspaint ../jspaint
    python examples/jspaint_circle.py --jspaint ../jspaint --policies expert,random     # no model needed
    python examples/jspaint_circle.py --jspaint ../jspaint --episodes 5                 # + thaitea/laya-vision
    python examples/jspaint_circle.py --model checkpoints/my-run --revision <sha> --device cuda

Chromium: set LAYA_CHROMIUM (or --chromium) to an existing binary instead of running `playwright install`.
Each run writes a new directory under --out (create-only): summary.json, and per policy and episode a
trajectory.jsonl (action, cursor, pen, and the model's action probabilities), final.png and episode.gif.
"""
import argparse
import json
import os
import time

from laya.paintenv import (PAINT_HEAD_MAX_LEN, PAINT_MAX_LEN, JSPaintEnv, JSPaintServer, circle_expert, model_policy,
                           play_episodes, random_policy)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jspaint", default=os.path.join(os.path.dirname(__file__), "..", "..", "jspaint"),
                    help="path to a JSPaint checkout (default: a sibling of this repo)")
    ap.add_argument("--policies", default="model,expert,random", help="comma list of model, expert, random")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--seed", type=int, default=900_000)
    ap.add_argument("--max-steps", type=int, default=260)
    ap.add_argument("--step-px", type=int, default=6)
    ap.add_argument("--directions", type=int, default=32, choices=(8, 16, 32))
    ap.add_argument("--model", default="thaitea/laya-vision")
    ap.add_argument("--revision", default=None, help="pin the Hub revision of --model")
    ap.add_argument("--device", default=None)
    ap.add_argument("--chromium", default=None, help="Chromium executable (default: LAYA_CHROMIUM or Playwright's)")
    ap.add_argument("--canvas-size", type=int, default=None,
                    help="square canvas side (default: the model's input image_size, 512 for the released checkpoints)")
    ap.add_argument("--headed", action="store_true", help="show the browser window")
    ap.add_argument("--out", default="results/jspaint")
    args = ap.parse_args()

    out_dir = os.path.join(args.out, time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(out_dir, exist_ok=False)
    names = [p.strip() for p in args.policies.split(",") if p.strip()]
    summary = {"task": "circle", "jspaint": os.path.abspath(args.jspaint), "args": vars(args), "policies": {}}

    agent = None
    if "model" in names:
        from laya import load_vlm

        # raised token budgets: the full question and a two-frame state fit with nothing cut
        agent = load_vlm(args.model, revision=args.revision, device=args.device, head_max_len=PAINT_HEAD_MAX_LEN,
                         max_len=PAINT_MAX_LEN)
    # A square canvas the size of the model's input, so the screenshot reaches the vision tower unscaled.
    canvas_size = args.canvas_size or (agent.prep.image_size if agent is not None else 512)
    summary["canvas_size"] = canvas_size

    with JSPaintServer(args.jspaint) as server, JSPaintEnv(server.url, max_steps=args.max_steps,
                                                           step_px=args.step_px, directions=args.directions,
                                                           canvas_size=canvas_size, headless=not args.headed,
                                                           executable_path=args.chromium, keep_frames=True) as env:
        for name in names:
            if name == "model":
                policy = model_policy(agent)
            elif name == "expert":
                policy = circle_expert()
            elif name == "random":
                policy = random_policy(args.seed)
            else:
                raise SystemExit("unknown policy %r" % name)

            rows = []

            def on_step(env, action):
                row = dict(env.trajectory[-1], episode=env.seed)
                if name == "model":
                    row["probabilities"] = policy.last["probabilities"]
                rows.append(row)
                if env.done:
                    ep_dir = os.path.join(out_dir, name, "seed%d" % env.seed)
                    os.makedirs(ep_dir)
                    with open(os.path.join(ep_dir, "trajectory.jsonl"), "w") as f:
                        f.writelines(json.dumps(r) + "\n" for r in rows)
                    env.frames[-1].save(os.path.join(ep_dir, "final.png"))
                    env.frames[0].save(os.path.join(ep_dir, "episode.gif"), save_all=True,
                                       append_images=env.frames[1:], duration=120, loop=0)
                    rows.clear()

            t = time.time()
            res = play_episodes(env, policy, args.episodes, seed=args.seed, on_step=on_step)
            res["seconds"] = round(time.time() - t, 1)
            if name == "model":
                res["model"], res["provenance"] = args.model, policy.provenance
            summary["policies"][name] = res
            print("%-7s mean score %.3f  pass rate %.2f  mean steps %.1f  (%.0fs)" % (
                name, res["mean_score"], res["pass_rate"], res["mean_steps"], res["seconds"]))

    pol = summary["policies"]
    if {"model", "expert", "random"} <= set(pol):
        e, r = pol["expert"]["mean_score"], pol["random"]["mean_score"]
        summary["normalized_model_score"] = (pol["model"]["mean_score"] - r) / (e - r) if e > r else None
        print("model normalized score (model - random) / (expert - random): %s" % summary["normalized_model_score"])
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print("wrote %s" % out_dir)


if __name__ == "__main__":
    main()
