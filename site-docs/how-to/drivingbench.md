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

## Score it offline on the published runs

[DrivingBench v1](https://drivingbench.com/) had four frontier models drive the harness's Corolla through a cone
course: 11 attempts, one finished (GPT-6 Astra, second attempt). Its
[release](https://drivingbench.com/downloads/) (CC BY 4.0, by Simon Mahns, Tobias Gessler and Aditya Ramabadran)
has every attempt's road video and a 5 Hz GPS track with progress along the course.
[`benchmarks/drivingbench_offline.py`](https://github.com/r33drichards/laya-vision/blob/main/benchmarks/drivingbench_offline.py)
asks the driver's question about those frames and checks the answer against where the car went. No car is needed.

```bash
pip install -e . torchvision imageio-ffmpeg
python benchmarks/drivingbench_offline.py --out drivingbench-maneuvers.jsonl               # ~6 min on a CPU
python benchmarks/drivingbench_offline.py --out drivingbench-directions.jsonl --directions
```

- **Frames:** one per second of each attempt's road video (the 480p copy), while the car moves at 0.3 m/s or more.
  About 45 MB is downloaded once, checked against the release manifest's SHA-256s, and cached.
- **Label:** which way the car turned over the next 5 m it drove, from the GPS track: left or right past 10°,
  otherwise straight. A frame counts only if the car gained course progress over those 5 m, so every label is a
  direction a driver took while advancing along the course. This gives 371 frames, 188 of them from the finished
  attempt: 183 straight, 96 left, 92 right.
- **Question:** the driver's six maneuvers, scored by the direction of the chosen maneuver (a stop counts as wrong).
  With `--directions`, a balanced question with one option each for left, straight and right. `--no-telemetry`
  leaves out the speed and steering text and gives the model only the image.

Results for `thaitea/laya-vision` (Hub commit `8b318c9`), zero-shot, against three baselines:

| | All 371 frames | First corner (progress < 15%, 153 frames) |
|---|---:|---:|
| Laya Vision, six maneuvers | 16.7% (stops on 28%) | 15.0% |
| Laya Vision, six maneuvers, likeliest direction | 22.9% | 22.2% |
| Laya Vision, balanced left / straight / right | 25.3% | 38.6% |
| Same, image only | 24.3% | 37.9% |
| Always the majority label (straight) | 49.3% | 50.3% |
| Keep turning the way the measured steering points | 60.9% | 56.2% |
| The frontier driver's command in force | 62.0% | 56.2% |

Over all frames Laya Vision is below chance (33%) on every variant. It does not tell left from right: with the balanced question it
answers "steer left" on 340 of the 371 frames, whichever way the car turned, which is also why it does better on the
first corner, a left turn. The frontier drivers' own commands agree with the label 62% of the time. That is a
reference point rather than a fair opponent, since the labels come from where those commands took the car. The
model was never trained on driving, and nothing here suggests it can steer this course zero-shot. The per-frame
predictions and summaries are in
[`eval-results/drivingbench-v1/`](https://github.com/r33drichards/laya-vision/tree/main/eval-results/drivingbench-v1).
