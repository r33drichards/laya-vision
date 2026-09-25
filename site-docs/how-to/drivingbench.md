# Drive through the DrivingBench harness

[DrivingBench Harness](https://github.com/aditya-ramabadran/drivingbench_harness_v1) lets a model drive a
comma-equipped Toyota at parking-lot speed through one MCP server, `drivingbench_sandbox`, with three tools:
`observe`, `set_motion` and `stop_now`. Chat apps such as Codex or Claude Code are its usual clients.
[`examples/drivingbench_drive.py`](https://github.com/r33drichards/laya-vision/blob/main/examples/drivingbench_drive.py)
makes Laya Vision a client too.

**This moves a real car.** Read the harness README's disclaimer and safety section first, and follow it: a sober,
licensed driver in the seat with a foot over the brake, private ground empty of people, walking pace. Laya Vision
has never been trained on driving; this is a zero-shot experiment.

## How it drives

Each step it calls `observe`, then asks one `choice` question about the camera frame. The question gives the
objective and asks which of six maneuvers to take: stop, straight, gentle left or right (25% steering), sharp left
or right (60%). The state text carries the reported speed, steering and command state. The script then calls:

- `stop_now` if the model picks stop, if the chosen maneuver's probability is below `--min-prob` (default 0.5), if
  `observe` returns an error, a camera problem, no image or an unavailable car, and when the loop exits (Ctrl-C
  included).
- `set_motion` otherwise, at `--speed` (default 0.5 m/s, the harness minimum) for `--duration` (default 5 s, the
  harness minimum). The next step replaces the command about every `--period` seconds (default 1).

Every command's `reason` names the maneuver and its probability, so the harness trace shows why the car moved.
Without `--drive` it is a dry run: it observes and prints what it would do and never calls `set_motion` or
`stop_now`.

## Run it

Set up the harness and bring the car online (its README: install, deploy, calibrate, supervised check). Then, on
the laptop running the gateway:

```bash
pip install -e . torchvision "mcp>=1.12,<2"
python examples/drivingbench_drive.py \
    --server <harness runtime>/bin/drivingbench-sandbox \
    --objective "Drive straight through the cone gate and stop past it." \
    --log drive-$(date +%Y%m%d-%H%M%S).jsonl                   # dry run
# then, with the car engaged and the driver ready:
python examples/drivingbench_drive.py --server ... --objective ... --drive
```

`--server` is the executable `drivingbench install` registers in chat apps (look up the `drivingbench_sandbox`
entry in `~/.claude.json` or `~/.codex/config.toml`). `--gateway-url` defaults to the harness's
`http://127.0.0.1:8766`. The objective in the harness's shared settings is not returned by `observe`, so pass it
with `--objective`. `--log` writes one JSON line per step (the observation summary, the six probabilities, the
decision and the harness's reply) to a new file; it refuses to overwrite one.

To benchmark, start a labeled session in the harness UI first with model `laya-vision` and harness `other`. The
MCP calls arrive with client label `laya-vision`. `drivingbench sync` attaches the transcript that drove a segment
by finding its `observe` timestamps in the files it searches. The `--log` file records them, so add a glob matching
your log files to `extra_transcripts` in the harness's laptop config (`~/.config/drivingbench-v01/config.json`) and
the log is attached like a chat transcript.
