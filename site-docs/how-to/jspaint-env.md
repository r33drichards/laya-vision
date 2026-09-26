# Draw in JSPaint with the mouse

This environment gives the model a real paint program, [JSPaint](https://github.com/r33drichards/jspaint), and a
task such as "draw a circle". The model can only use the mouse. Each step it sees a screenshot of the canvas and
picks one mouse action, which is sent to the app as real pointer events. When the episode ends, a verifier scores
the pixels on the canvas.

The environment is in `laya/paintenv.py`, the verifier in `laya/circle_verifier.py`, and the runner in
[`examples/jspaint_circle.py`](https://github.com/r33drichards/laya-vision/blob/main/examples/jspaint_circle.py).

## Run it

```bash
pip install -e ".[paint]"
git clone https://github.com/r33drichards/jspaint ../jspaint
python examples/jspaint_circle.py --jspaint ../jspaint --policies expert,random   # reference points, no model
python examples/jspaint_circle.py --jspaint ../jspaint --episodes 5 --revision <sha>   # + thaitea/laya-vision
```

JSPaint needs no build step. `JSPaintServer` serves the checkout as static files, and Playwright drives it in
headless Chromium. Set `LAYA_CHROMIUM` (or pass `--chromium`) to use an existing Chromium binary instead of
running `playwright install`.

Each run writes a new directory under `results/jspaint/`. It contains:

- `summary.json`: per-policy mean score, pass rate, and the verifier components for every episode;
- one folder per policy and episode, each holding:
    - `trajectory.jsonl`: action, cursor, pen state, and the model's probabilities over the actions;
    - `final.png`: the last frame;
    - `episode.gif`: every frame of the episode.

## Observation and actions

Each episode starts the same way. The canvas is cleared to white, the Brush tool is selected (4 px strokes), and
the cursor is placed at a seeded point in the middle of the canvas.

| | |
|---|---|
| Observation | A screenshot of the 683×384 canvas with the cursor drawn on it: a red ring and cross when the button is up, a filled red dot when it is held down. The `note` text gives the task, the pen state and the step count. |
| Actions (`choice`) | `UP` `DOWN` `LEFT` `RIGHT` `UP_LEFT` `UP_RIGHT` `DOWN_LEFT` `DOWN_RIGHT` each move 24 px (`step_px`) and draw while the pen is down. The other three are `PEN_DOWN`, `PEN_UP` and `DONE`. |
| End | The episode ends on `DONE` or after 80 steps (`max_steps`). The pen is released automatically at the end. |
| Reward | The verifier score, paid at the end. With `reward="shaped"`, each step instead earns the change in score. |

The model can only pick from a list of options, so the actions are relative moves. The screenshot has no OS
cursor, so the cursor is drawn onto it. This is the same approach as Maze and Snake.

The discrete actions are built on a mouse-only tool API: `move_mouse(dx, dy)`, `mouse_down()`, `mouse_up()` and
`screenshot()`. `paintenv.TOOLS` describes these tools as JSON-schema tool definitions, and
`JSPaintEnv.call_tool(name, args)` runs them. A tool-calling agent can therefore drive the same environment with
free coordinates.

## The circle verifier

`score_circle(pixels)` reads the canvas pixels from the page with `toDataURL`, so the cursor overlay is never
counted as ink. It fits a least-squares circle to the ink and computes five components, each between 0 and 1:

- **fit**: how tightly the ink hugs the fitted circle, `1 - rms(|d - r|) / (0.2 r)`;
- **coverage**: the share of 36 angular sectors that contain ink on the ring;
- **clean**: the share of ink within `0.2 r` of the ring, so scribbles and fills count against the drawing;
- **closure**: 1 if no gap in the ring is wider than 30°, falling to 0 at a 120° gap;
- **size**: 1 if the radius is between 8% and 50% of the canvas height and the centre is on the canvas, else 0.

`score` is the product of the five. An episode `passed` if the score is at least 0.6 and the circle is closed.

`tests/test_circle_verifier.py` checks both directions:

- A drawn ring scores 0.95, and an octagon (what the eight moves can draw) scores 0.87. Both pass.
- These all fail: a line, a dot, a tiny ring, a half arc, a three-quarter arc, a square, a flat ellipse, a filled
  disc, a ring with a stray line, and random scribbles.

## Reference points

`circle_expert()` is a scripted policy. It walks with the pen up to the edge of a circle centred on the canvas,
presses the button, and traces one full turn with the eight moves. `random_policy` picks actions uniformly. On seeds
900000–900002:

| Policy | Mean score | Pass rate | Mean steps |
|---|---|---|---|
| Scripted expert | 0.76 | 3/3 | 37.7 |
| Random | 0.00 | 0/3 | 8.7 |
| `thaitea/laya-vision`, zero-shot | 0.00 | 0/3 | 2.0 |

The zero-shot checkpoint's action probabilities are nearly uniform, 0.10–0.15 for the top action against 1/11 ≈
0.09. It chooses `DONE` within 1–3 steps. The model was never trained on this task. The expert's trajectories
(screenshot, action) are the obvious data to train it on, as the game checkpoints were trained on expert frames.
