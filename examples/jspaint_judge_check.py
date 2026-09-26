"""Check whether Laya Vision's judgements track what is really happening on the JSPaint canvas.

The scripted circle expert and a random policy play; every --every steps the model answers the judgement questions
(`laya.games.paint_judgements`: progress 0-4, on track or off track, what is drawn) about the same two-frame state
the drawing policy sees. The truth for progress is the share of the 36 angular sectors around the target circle's
centre that the cursor has passed through with the pen down (0 before drawing, 1 for a full turn). Reported:

* how judged progress rises with true progress along the expert's circles (Spearman rank correlation, and the mean
  judged progress per quarter of true progress);
* judged progress and P(on track) for the random policy, which should stay low;
* what the model says the clean final canvas shows, per policy.

    python examples/jspaint_judge_check.py --jspaint ../jspaint --episodes 3

Writes a new directory under --out: judgements.jsonl (one row per check) and summary.json.
"""
import argparse
import json
import math
import os
import time

import numpy as np
from PIL import Image

from laya.games import paint_judgements
from laya.paintenv import (PAINT_HEAD_MAX_LEN, PAINT_MAX_LEN, JSPaintEnv, JSPaintServer, circle_expert, judge_canvas,
                           random_policy, summarize_judgements)

SECTORS = 36


def true_progress(env) -> float:
    """Share of the 36 sectors around the canvas centre that pen-down cursor positions have covered."""
    cx, cy = env.width / 2.0, env.height / 2.0
    hit = {int((math.atan2(t["y"] - cy, t["x"] - cx) + math.pi) / (2 * math.pi) * SECTORS) % SECTORS
           for t in env.trajectory if t["pen"]}
    return len(hit) / SECTORS


def spearman(a, b) -> float:
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1]) if len(a) > 2 and np.std(ra) and np.std(rb) else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jspaint", default=os.path.join(os.path.dirname(__file__), "..", "..", "jspaint"))
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--every", type=int, default=10, help="ask the judgements every N steps")
    ap.add_argument("--seed", type=int, default=900_000)
    ap.add_argument("--model", default="thaitea/laya-vision")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--chromium", default=None)
    ap.add_argument("--out", default="results/jspaint-judge")
    args = ap.parse_args()

    from laya import load_vlm

    out_dir = os.path.join(args.out, time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(out_dir, exist_ok=False)
    agent = load_vlm(args.model, revision=args.revision, device=args.device, head_max_len=PAINT_HEAD_MAX_LEN,
                     max_len=PAINT_MAX_LEN)
    questions = paint_judgements("circle")
    rows, finals = [], {}
    with JSPaintServer(args.jspaint) as server, JSPaintEnv(server.url, canvas_size=agent.prep.image_size,
                                                           executable_path=args.chromium) as env:
        for name, make in (("expert", lambda: circle_expert()), ("random", lambda: random_policy(args.seed))):
            policy = make()
            finals[name] = []
            for i in range(args.episodes):
                env.reset(args.seed + i)
                while not env.done:
                    if env.steps % args.every == 0:
                        ans = agent.predict(env.state(), questions, strict=True)["answers"]
                        rows.append({"policy": name, "seed": args.seed + i, "step": env.steps, "pen": env.pen,
                                     "true_progress": round(true_progress(env), 4), **summarize_judgements(ans)})
                    env.step(policy(env))
                final = judge_canvas(agent, Image.fromarray(env.canvas_pixels()))
                finals[name].append({"seed": args.seed + i, "verifier": env.result["score"], **final})
                rows.append({"policy": name, "seed": args.seed + i, "step": env.steps, "final": True,
                             "true_progress": round(true_progress(env), 4), **final})
                print("%-6s seed %d  verifier %.2f  final canvas judged: %s (progress %.2f, on track %.2f)" % (
                    name, args.seed + i, env.result["score"], final["drawn"], final["progress"], final["on_track"]))

    exp = [r for r in rows if r["policy"] == "expert" and not r.get("final")]
    rnd = [r for r in rows if r["policy"] == "random"]
    tp, jp = [r["true_progress"] for r in exp], [r["progress"] for r in exp]
    quarters = {}
    for lo, hi in ((0, 0.001), (0.001, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01)):
        sel = [r for r in exp if lo <= r["true_progress"] < hi]
        if sel:
            quarters["%.2f-%.2f" % (lo, min(hi, 1.0))] = {
                "n": len(sel), "judged_progress": round(float(np.mean([r["progress"] for r in sel])), 3),
                "p_on_track": round(float(np.mean([r["on_track"] for r in sel])), 3)}
    summary = {
        "model": args.model, "provenance_revision": getattr(agent, "source", {}).get("revision"),
        "every": args.every, "episodes": args.episodes, "seed": args.seed,
        "expert_spearman_true_vs_judged_progress": round(spearman(tp, jp), 3),
        "expert_by_true_progress": quarters,
        "random_mean_judged_progress": round(float(np.mean([r["progress"] for r in rnd])), 3),
        "random_mean_p_on_track": round(float(np.mean([r["on_track"] for r in rnd])), 3),
        "final_canvas": finals,
    }
    with open(os.path.join(out_dir, "judgements.jsonl"), "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "final_canvas"}, indent=2))
    print("wrote %s" % out_dir)


if __name__ == "__main__":
    main()
