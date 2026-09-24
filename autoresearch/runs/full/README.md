# Full-length runs

Recipes from the autoresearch frontier trained for longer with `autoresearch/full_run.py` (full train splits,
`pool-full-v1`), and measured exactly as the harness measures a 15-minute run, so the numbers sit on the same
frontier. Each `<name>.json` is the run's result; the full evaluation suite for it is in
`eval-results/autoresearch-full-<name>*.json`.

| Run | Recipe | Minutes | Quality | Games | Params | Latency | Status |
|---|---|---:|---:|---:|---:|---:|---|
| `long-sep24-b64` | `dba11d5`: 20 text layers, 45% game data, batch 64 | 120 | 0.686 | 0.346 | 201.2M | 0.828× | published as `thaitea/laya-vision` and `thaitea/laya-vision-201m` |
| `long-sep24-vision-halflr` | `09704d2`: 20 text layers, 45% game data, vision tower trained at LR 5e-6 | 120 | 0.655 | 0.536 | 201.2M | 0.834× | not published: loses several points on visual-reasoning sets |
| `long-sep24-b64-next05` | `56328f0`: `dba11d5` + next-move head, weight 0.5 | 120 | | | | | training |
| `smoke-sep24-a` | `5e9ef40`, 4 minutes, a simulated preemption | 4 | 0.678 | 0.133 | 201.2M | 0.829× | tooling smoke test |

Quality is macro accuracy over 34 eval sets minus calibration error; games is the mean normalized score over the
10-game benchmark (0 = random play, 1 = expert); latency is relative to the previous checkpoint timed on the same L4.
