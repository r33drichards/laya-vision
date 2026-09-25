"""Score the DrivingBench driving policy offline, on the published DrivingBench v1 runs (drivingbench.com).

DrivingBench v1 had four frontier models drive a real Toyota Corolla through a cone course (11 attempts, one
finished). Its release has, for every attempt, the road-camera video and a 5 Hz GPS track in course coordinates with
`progress` along the course centerline. This asks `laya.driving`'s maneuver question about frames of that video and
checks the answer's direction against where the car actually went:

- Frames: one every --every seconds of each attempt's road video (480p copy), while the car moves (>= 0.3 m/s).
- Label: the car's turn over the next --ahead metres it drove, from the GPS track: left or right if the direction of
  travel changes by more than --turn-deg, else straight. A frame is kept only if course progress rose over that
  stretch by at least --min-progress (the car was advancing along the course, not leaving it), so every label is a
  direction a driver took on course; most come from the one finished attempt.
- Metrics: accuracy of the chosen maneuver's direction (a stop counts wrong), accuracy of the likeliest direction
  once the maneuvers are summed per direction, mean P(label), the stop rate, and the same accuracies for three
  baselines: always the majority label, keep turning the way the car's measured steering points, and the direction
  of the frontier driver's command in force at that moment. The labels come from where those commands took the car,
  so the last one is a reference point, not a fair opponent.

    pip install -e . torchvision imageio-ffmpeg
    python benchmarks/drivingbench_offline.py --out drivingbench-laya-vision.jsonl        # ~10 min on a CPU

Downloads (~45 MB, sha256-checked against the release manifest) are cached in --cache. The release is CC BY 4.0 by
Simon Mahns, Tobias Gessler and Aditya Ramabadran; nothing from it is written to --out except derived labels.
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time
import urllib.request
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from laya.driving import (CONE_COURSE, DIRECTIONS, MANEUVERS, direction_probs, direction_question,  # noqa: E402
                          question, state)

RELEASE = "https://xttlbkfgeupajhyo.public.blob.vercel-storage.com/releases/2026-09-21/artifact-manifest.json"
COLLECTION = "dataset/2026-09-17-cone-course-trial/"
MIN_SPEED = 0.3


def fetch(url: str, path: str, sha256: str = None) -> str:
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".part"
        urllib.request.urlretrieve(url, tmp)
        os.replace(tmp, path)
    if sha256:
        with open(path, "rb") as f:
            got = hashlib.sha256(f.read()).hexdigest()
        if got != sha256:
            raise ValueError("%s: sha256 %s, manifest says %s" % (path, got, sha256))
    return path


def attempts(cache: str):
    """{"<group>/<attempt>": {"track": path, "timeline": path, "video": path}} for every attempt in the release."""
    manifest = json.load(open(fetch(RELEASE, os.path.join(cache, "artifact-manifest.json"))))
    runs = defaultdict(dict)
    for e in manifest:
        p = e["path"]
        if p.startswith(COLLECTION) and p.endswith(("/track.json", "/timeline.json")):
            key = "/".join(p[len(COLLECTION):].split("/")[:2])
            runs[key][p.rsplit("/", 1)[1][:-5]] = e
        elif p.startswith("runs/") and p.endswith("/road-480p.mp4"):
            runs["/".join(p.split("/")[1:3])]["video"] = e
    out = {}
    for key, files in sorted(runs.items()):
        if {"track", "timeline", "video"} <= set(files):
            out[key] = {k: fetch(e["url"], os.path.join(cache, key, e["path"].rsplit("/", 1)[1]), e["sha256"])
                        for k, e in files.items()}
    return out


def turn_deg(points, i: int, ahead: float):
    """Signed change in direction of travel (degrees, positive left) over the `ahead` metres driven after point i,
    each direction taken over the first / last metre of that stretch; with the index of the stretch's end. None if
    the attempt ends first. The track's x, y are metres in a right-handed frame (x east, y north)."""
    xy = np.array([[p["x"], p["y"]] for p in points], dtype=float)
    d = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])

    def at(dist):
        j = int(np.searchsorted(d, d[i] + dist))
        return j if j < len(points) else None

    a1, b0, b1 = at(1.0), at(ahead - 1.0), at(ahead)
    if a1 is None or b1 is None:
        return None, None
    v0, v1 = xy[a1] - xy[i], xy[b1] - xy[b0]
    return math.degrees(math.atan2(v0[0] * v1[1] - v0[1] * v1[0], v0 @ v1)), b1


def sign_direction(value: float, deadband: float) -> str:
    return "left" if value > deadband else "right" if value < -deadband else "straight"


def command_at(calls, t: float) -> str:
    """The direction of the driver's command in force at video time t ("stop" if none)."""
    current = "stop"
    for c in calls:
        if c["t"] > t:
            break
        if c["tool"] == "set_motion" and c.get("outcome", {}).get("status", "accepted") == "accepted":
            a = c["arguments"]
            current = "straight" if a["steering_percent"] == 0 else a["direction"]
        elif c["tool"] == "stop_now":
            current = "stop"
    return current


def label_frames(run: dict, every: float, ahead: float, turn: float, min_progress: float):
    """The attempt's labelled sample times: dicts with t, label, turn_deg, progress, speed, steering, command."""
    track, timeline = json.load(open(run["track"])), json.load(open(run["timeline"]))
    points, tel = track["points"], timeline["telemetry"]
    scale = timeline["video"].get("steering_scale_deg", 180)
    tel_t = np.array([s["t"] for s in tel])
    t, rows = track["window"]["t_start"], []
    pt = np.array([p["t"] for p in points])
    while t <= min(track["window"]["t_end"], timeline["video"]["duration_s"] - 0.1):
        i = int(np.argmin(np.abs(pt - t)))
        p = points[i]
        if p["speed_mps"] >= MIN_SPEED:
            deg, j = turn_deg(points, i, ahead)
            if deg is not None and points[j]["progress"] - p["progress"] >= min_progress:
                s = tel[int(np.argmin(np.abs(tel_t - t)))]
                steer = None if s.get("steering_deg") is None else round(100 * s["steering_deg"] / scale, 1)
                rows.append({"t": round(t, 2), "label": sign_direction(deg, turn), "turn_deg": round(deg, 1),
                             "progress": p["progress"], "speed_mps": round(p["speed_mps"], 2),
                             "steering_percent": steer, "command": command_at(timeline["calls"], t)})
        t += every
    return rows, timeline["video"]["fps"]


def video_frames(path: str, times, fps: float):
    """PIL frames of the video at the given times (seconds)."""
    import imageio_ffmpeg
    from PIL import Image

    want = {int(round(t * fps)): t for t in times}
    reader = imageio_ffmpeg.read_frames(path)
    meta = next(reader)
    w, h = meta["size"]
    out = {}
    for n, buf in enumerate(reader):
        if n in want:
            out[want[n]] = Image.fromarray(np.frombuffer(buf, np.uint8).reshape(h, w, 3))
        if len(out) == len(want):
            break
    reader.close()
    return out


def summarize(rows):
    def acc(key):
        return round(float(np.mean([r[key] == r["label"] for r in rows])), 4) if rows else None

    labels = Counter(r["label"] for r in rows)
    majority = labels.most_common(1)[0][0] if rows else None
    return {
        "n": len(rows),
        "labels": dict(labels),
        "laya_choice_acc": acc("laya_direction"),
        "laya_direction_acc": acc("laya_top_direction"),
        "laya_mean_p_label": round(float(np.mean([r["laya_p"][r["label"]] for r in rows])), 4) if rows else None,
        "laya_stop_rate": round(float(np.mean([r["laya_direction"] == "stop" for r in rows])), 4) if rows else None,
        "majority_acc": round(labels[majority] / len(rows), 4) if rows else None,
        "keep_steering_acc": acc("keep_steering"),
        "driver_command_acc": acc("command"),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="thaitea/laya-vision")
    ap.add_argument("--revision", default=None, help="pin the checkpoint to a Hub commit")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", required=True, help="new JSONL file: one row per frame, then a summary row")
    ap.add_argument("--cache", default=os.path.expanduser("~/.cache/drivingbench-v1"))
    ap.add_argument("--every", type=float, default=1.0, help="seconds between sampled frames")
    ap.add_argument("--ahead", type=float, default=5.0, help="metres of driving the label looks ahead")
    ap.add_argument("--turn-deg", type=float, default=10.0, help="turn over --ahead metres that counts as a turn")
    ap.add_argument("--min-progress", type=float, default=0.01,
                    help="course progress (0-1) the car must gain over the stretch for the frame to count")
    ap.add_argument("--steer-deadband", type=float, default=10.0,
                    help="measured steering_percent within +-this is straight, for the keep-steering baseline")
    ap.add_argument("--directions", action="store_true",
                    help="ask laya.driving.direction_question (left / straight / right, no stop), not the maneuvers")
    ap.add_argument("--no-telemetry", action="store_true", help="give the model the image only (an ablation)")
    ap.add_argument("--limit", type=int, default=0, help="score at most this many frames (0 = all)")
    ap.add_argument("--labels-only", action="store_true", help="print the label counts per attempt and exit")
    args = ap.parse_args()

    runs = attempts(args.cache)
    labelled = {}
    for key, run in runs.items():
        labelled[key] = label_frames(run, args.every, args.ahead, args.turn_deg, args.min_progress)
        print("%-45s %4d frames %s" % (key, len(labelled[key][0]), dict(Counter(r["label"] for r in labelled[key][0]))))
    if args.labels_only:
        return

    import laya

    agent = laya.load_vlm(args.model, revision=args.revision, device=args.device)
    qs = direction_question(CONE_COURSE) if args.directions else question(CONE_COURSE)
    with open(args.out, "x") as out:  # create-only: never overwrite an earlier run's results
        rows, t0 = [], time.perf_counter()
        for key, (frames, fps) in labelled.items():
            if args.limit and len(rows) >= args.limit:
                break
            frames = frames[: args.limit - len(rows)] if args.limit else frames
            images = video_frames(runs[key]["video"], [r["t"] for r in frames], fps)
            for r in frames:
                s = state(images[r["t"]]) if args.no_telemetry else \
                    state(images[r["t"]], r["speed_mps"], r["steering_percent"], "executing")
                answer = agent.predict(s, qs, strict=True)["answers"]["action"]
                if args.directions:
                    options = qs["action"]["criteria"]
                    p = {"stop": 0.0, **{d: answer["probabilities"][o] for d, o in zip(DIRECTIONS, options)}}
                    target = (DIRECTIONS[options.index(answer["choice"])], None)
                else:
                    p = direction_probs(answer)
                    target = MANEUVERS[answer["choice"]]
                r = dict(r, attempt=key, laya_choice=answer["choice"],
                         laya_direction="stop" if target is None else target[0],
                         laya_top_direction=max(DIRECTIONS, key=p.get),
                         laya_p={k: round(v, 4) for k, v in p.items()},
                         keep_steering="straight" if r["steering_percent"] is None
                         else sign_direction(r["steering_percent"], args.steer_deadband))
                rows.append(r)
                out.write(json.dumps(r) + "\n")
            print("%-45s done, %d frames in %.0f s" % (key, len(rows), time.perf_counter() - t0), flush=True)
        by_attempt = defaultdict(list)
        for r in rows:
            by_attempt[r["attempt"]].append(r)
        summary = {
            "summary": True, "model": args.model, "revision": agent.source["revision"], "release": RELEASE,
            "settings": {k: getattr(args, k) for k in ("every", "ahead", "turn_deg", "min_progress",
                                                         "steer_deadband", "directions", "no_telemetry", "limit")},
            "question": qs["action"],
            "all": summarize(rows),
            "first_corner": summarize([r for r in rows if r["progress"] < 0.15]),
            "by_attempt": {k: summarize(v) for k, v in by_attempt.items()},
        }
        out.write(json.dumps(summary) + "\n")
    print(json.dumps({k: summary[k] for k in ("all", "first_corner")}, indent=1))


if __name__ == "__main__":
    main()
