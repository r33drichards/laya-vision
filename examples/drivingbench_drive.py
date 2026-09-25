"""Let Laya Vision drive through the DrivingBench harness (github.com/aditya-ramabadran/drivingbench_harness_v1).

The harness exposes a car as one MCP server, `drivingbench_sandbox`, with three tools: `observe`, `set_motion` and
`stop_now`. This script is the chat app: it spawns that server over stdio, like Codex or Claude Code do, and loops

    observe -> one `choice` question over a few maneuvers -> set_motion (or stop_now)

The camera frame is the image; the operator's objective and the car's reported speed and steering are the state
text. Every command carries a `reason` naming the maneuver and its probability, so the harness trace shows why the
car moved. Zero-shot: the model was trained on photo/diagram questions and games, never on driving.

    pip install -e . torchvision "mcp>=1.12,<2"
    # dry run (default): observes and prints what it would do, never commands the car
    python examples/drivingbench_drive.py --server ~/path/to/runtime/bin/drivingbench-sandbox \\
        --objective "Drive straight through the cone gate and stop past it."
    # drive: only with the harness README's safety setup in place (driver in the seat, foot over the brake)
    python examples/drivingbench_drive.py --server ... --objective ... --drive --speed 0.5

It stops (stop_now) when the model picks "stop", when the top maneuver's probability is below --min-prob, when
observe reports an error, a camera problem or an unready car, and on exit (Ctrl-C included). None of that is a
safety function; the human on the brake is.
"""
import argparse
import asyncio
import base64
import io
import json
import time
from datetime import datetime, timezone

# maneuver -> (direction, steering_percent); None means stop_now
MANEUVERS = {
    "stop: a cone, person, vehicle or wall is close ahead, or the way is unclear": None,
    "go straight": ("straight", 0),
    "turn gently left": ("left", 25),
    "turn sharply left": ("left", 60),
    "turn gently right": ("right", 25),
    "turn sharply right": ("right", 60),
}


def question(objective: str):
    return {
        "action": {
            "type": "choice",
            "instructions": "You are driving a car at walking pace in an empty parking lot, seen from its "
                            "windshield camera. Objective: %s What should the car do next?" % objective,
            "criteria": list(MANEUVERS),
        }
    }


def decide(answer, min_prob: float):
    """The model's answer -> ("stop", None, reason) or ("motion", (direction, steering_percent), reason)."""
    choice = answer["choice"]
    p = answer["probabilities"][choice]
    reason = "laya: %s (p=%.2f)" % (choice.split(":")[0], p)
    if MANEUVERS[choice] is None:
        return "stop", None, reason
    if p < min_prob:
        return "stop", None, "laya: unsure, best %s (p=%.2f < %.2f)" % (choice, p, min_prob)
    return "motion", MANEUVERS[choice], reason


def unready(summary) -> str:
    """Why this observation must not be acted on, or '' if it can be."""
    if summary.get("error"):
        return summary["error"]
    if summary.get("camera_reason"):
        return summary["camera_reason"]
    if summary.get("state") == "unavailable":
        return summary.get("reason") or "unavailable"
    return ""


def parse_observe(result):
    """An MCP observe result -> (state summary dict, list of PIL images)."""
    from PIL import Image

    summary, images = {}, []
    for block in result.content:
        if block.type == "text":
            summary = json.loads(block.text)
        elif block.type == "image":
            images.append(Image.open(io.BytesIO(base64.b64decode(block.data))).convert("RGB"))
    return summary, images


async def run(args):
    import laya
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    print("loading %s ..." % args.model)
    agent = laya.load_vlm(args.model, revision=args.revision, device=args.device)
    qs = question(args.objective)
    server = StdioServerParameters(command=args.server,
                                   args=["--gateway-url", args.gateway_url, "--client", "laya-vision"])
    log = open(args.log, "x") if args.log else None  # create-only: never overwrite an earlier drive's log
    async with stdio_client(server) as (read, write), ClientSession(read, write) as mcp:
        await mcp.initialize()

        async def call(name, **kw):
            result = await mcp.call_tool(name, kw)
            return json.loads(result.content[0].text) if result.content else {}

        async def stop(reason):
            if args.drive:
                return await call("stop_now", reason=reason[:200])
            return {"status": "dry_run"}

        print("mode: %s" % ("DRIVE" if args.drive else "dry run (pass --drive to command the car)"))
        step = 0
        try:
            while not args.steps or step < args.steps:
                t0 = time.perf_counter()
                summary, images = parse_observe(await mcp.call_tool("observe", {}))
                why = unready(summary) or ("" if images else "no_image")
                answer = None
                if why:
                    kind, target, reason = "stop", None, "laya: not acting, " + why
                else:
                    state = {"images": images, "note": {
                        "speed_mps": summary.get("speed_mps"),
                        "steering_percent_positive_left": summary.get("steering_percent"),
                        "command_state": summary.get("state"),
                    }}
                    answer = agent.predict(state, qs)["answers"]["action"]
                    kind, target, reason = decide(answer, args.min_prob)
                ms = (time.perf_counter() - t0) * 1000
                if kind == "stop":
                    status = await stop(reason)
                elif args.drive:
                    status = await call("set_motion", direction=target[0], steering_percent=target[1],
                                        speed_mps=args.speed, duration_s=args.duration, reason=reason[:200])
                else:
                    status = {"status": "dry_run"}
                print("%4d  %-7s %-50s speed=%s steer=%s  %4.0f ms  %s" % (
                    step, kind, reason, summary.get("speed_mps"), summary.get("steering_percent"), ms,
                    status.get("error") or status.get("status")))
                if log:
                    log.write(json.dumps({
                        "time": datetime.now(timezone.utc).isoformat(), "step": step, "observe": summary,
                        "probabilities": answer and answer["probabilities"], "decision": kind,
                        "target": target, "reason": reason, "ms": round(ms, 1), "result": status,
                    }) + "\n")
                    log.flush()
                step += 1
                await asyncio.sleep(max(0.0, args.period - (time.perf_counter() - t0)))
        finally:
            print("stopping: %s" % await stop("laya: driver loop exited"))
            if log:
                log.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", default="drivingbench-sandbox",
                    help="the harness's MCP executable (the one `drivingbench install` registers in chat apps)")
    ap.add_argument("--gateway-url", default="http://127.0.0.1:8766")
    ap.add_argument("--objective", default="Follow the operator-designated course and stop at the destination.",
                    help="the task, as in the harness's shared settings (observe does not return it)")
    ap.add_argument("--model", default="thaitea/laya-vision")
    ap.add_argument("--revision", default=None, help="pin the checkpoint to a Hub commit")
    ap.add_argument("--device", default=None)
    ap.add_argument("--drive", action="store_true", help="actually send set_motion; without it, a dry run")
    ap.add_argument("--speed", type=float, default=0.5, help="m/s, 0.5-3.5 and at most the operator's ceiling")
    ap.add_argument("--duration", type=float, default=5.0,
                    help="s each command lasts before the car brakes, 5-60; the loop replaces it every --period")
    ap.add_argument("--period", type=float, default=1.0, help="s between observations")
    ap.add_argument("--min-prob", type=float, default=0.5,
                    help="stop unless the chosen maneuver has at least this probability")
    ap.add_argument("--steps", type=int, default=0, help="quit after this many observations (0 = until Ctrl-C)")
    ap.add_argument("--log", default=None, help="write one JSON line per step to this new file")
    args = ap.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
