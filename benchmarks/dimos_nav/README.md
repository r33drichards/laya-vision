# Dimensional System One navigation

[System One navigation](https://research.dimensional.org/system-one-navigation/) asks a policy to drive a robot
to a named object in a furnished house (Habitat-Sim, HSSD scenes), from a text-only WorldState. Each step,
the policy answers six typed questions: `drive.x`, `drive.y`, `drive.yaw`, `stop`, `task` and `target`.
Those are Laya's own question types, so a Laya checkpoint can take the place of the page's "TypeSafe Jev"
arm unchanged. dimos's `TypeSafeAgent` builds the state, asks the questions and steers. Its call to TypeSafe's
`POST /v1/systemone` goes to [`systemone_server.py`](systemone_server.py), which answers with
`VLMAgent.predict`.

| File | What |
|---|---|
| [`../../modal_dimos_nav.py`](../../modal_dimos_nav.py) | Image (dimos at a pinned commit, habitat-sim, the MLS planner, a laya venv), `prepare_hssd`, one L4 per case |
| [`systemone_server.py`](systemone_server.py) | `/v1/systemone` over a Laya checkpoint; logs latency, truncation and answers per call |
| [`summarize.py`](summarize.py) | Arrival, SPL, SoftSPL, per difficulty and scene, and the model's picks over every call |
| [`probe_options.py`](probe_options.py) | Re-scores logged states under other option renderings, to tell adapter effects from the checkpoint's |

```bash
modal run modal_dimos_nav.py::prepare_hssd                  # once: the HSSD files the suite's scenes use
modal run modal_dimos_nav.py::main --out <new dir>          # 84 cases in parallel, ~15 min
python benchmarks/dimos_nav/summarize.py <dir> --scenes <dimos checkout>/dimos/evals/suites/scenes/habitat
```

## Result: `thaitea/laya-vision` @ `f2fe3c1`, 2026-09-29

[`eval-results/dimos-nav/laya-vision-f2fe3c1-600s.json`](../../eval-results/dimos-nav/laya-vision-f2fe3c1-600s.json):
84 of 84 cases graded, no errors, 600 s per case (the published runs' cut-off), on an L4.

| | laya-vision (this run, 84 public cases) | TypeSafe Jev (page, 327 tasks) |
|---|---|---|
| Arrival | **0 / 84 (0%)** | 45.9% |
| Mean SPL | **0.000** | 0.263 |
| Mean SoftSPL | 0.076 | 0.325 |
| Model call | 0.22 s median, ~2 calls/s | |

**The checkpoint does not navigate.** Over 80,361 calls it answered `drive.x = backward` on every call.
`drive.yaw` was `none` on all but 149. `stop` never reached the agent's 0.7 threshold, and `task` said
`finished` once. The robot only ever reverses, and the little SoftSPL it earns is from runs where reversing
happened to bring it closer.

**It is not the adapter.** The option-rendering probe re-scored 168 logged states, 2 per case
([`laya-vision-f2fe3c1-option-probe.json`](../../eval-results/dimos-nav/laya-vision-f2fe3c1-option-probe.json)).
Changing only the option text moves the constant pick: `backward` becomes `forward`, and `none` becomes
`turn_left`. So the model reads the options' wording, not the state. None of the variants gives a usable
policy: full JSON criteria, labels only, order-averaged over 6 permutations, or a state with no truncation.
Labels only would declare `finished` at once. Full JSON would always drive forward, turning left. On the 33
sampled states where the way is clear, labels-only `drive.yaw` matches the target's bearing 25 times,
barely above always answering `turn_right` (22).

This is expected. The checkpoint is a 201M image-QA model with a 1,024-token budget. This eval is a
text-only control prompt it never trained on, and one that overflows that budget on every call. Every call lost 190–380 tokens from the end of
the state (`open_sides`, `free_space`). Goal, target bearing and distance fit.

**Caveats.** The public dimos suite is 84 cases over 11 scenes (10 HSSD, plus the HM3D example house). The
page's 327 tasks are a different set; even the ids both use (e.g. `102344193_chair`) have other spawns and
goals. So the two columns above are not the same cases. The option renderer passes each criterion's `what`
text; see the probe for the others. Raw per-case output (the agent's traces, dimos run dirs, the server's call
logs) is on the `dimos-habitat` Modal volume under `runs/laya-vision-f2fe3c1-600s-20260929`. Its SHA-256s are
in the results file.
