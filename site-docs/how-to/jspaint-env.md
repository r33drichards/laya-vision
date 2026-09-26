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
| Canvas | 512×512, the size of laya-vision's input image (`--canvas-size`, default: the model's `image_size`). |
| Observation | A screenshot of the canvas with the cursor drawn on it: a red ring and cross when the button is up, a filled red dot when it is held down. The `note` text gives the task, the pen state and the step count. |
| Actions (`choice`) | 32 compass moves, `N` `NbE` `NNE` `NEbN` `NE` … `NbW`, one every 11.25°. Each moves 6 px (`step_px`) and draws while the pen is down. Each option's text gives its bearing in degrees. The other three are `PEN_DOWN`, `PEN_UP` and `DONE`. |
| End | The episode ends on `DONE` or after 260 steps (`max_steps`). The pen is released automatically at the end. |
| Reward | The verifier score, paid at the end. With `reward="shaped"`, each step instead earns the change in score. |

### Why the canvas is 512×512

The released checkpoints read every image as a single 512×512 view, and the processor resizes to that square
without keeping the aspect ratio. JSPaint's default 683×384 canvas was therefore squashed: a drawn circle reached the
model as a tall ellipse (165×293 px). On that canvas laya-vision called a perfect ring "blank", with P(circle) 0.05.

A 512×512 canvas reaches the vision tower pixel for pixel. On it, the model recognizes the scripted expert's circles
without any cropping, and still tells them apart from a square drawn the same way:

| Canvas (JSPaint, 512×512) | "What is drawn?" (6 options) | "Which shape?" (5 options) | Verifier |
|---|---|---|---|
| Expert circle, seed 0 | circle (0.67) | circle (0.94) | 0.97 |
| Expert circle, seed 1 | circle (0.84) | circle (0.94) | 0.97 |
| Expert circle, seed 2 | circle (0.82) | circle (0.94) | 0.97 |
| Square drawn with the same moves | square (P(circle) 0.09) | square (0.03) | 0.00 |
| Blank | blank (0.08) | heart (0.15) | 0.00 |

Each cell is the top answer, with the probability of "circle" in brackets. These are spot checks of
`thaitea/laya-vision` at revision `f2fe3c1`, not a benchmark. Two caveats. The yes/no question "Does the image
show a circle?" stays below 0.35 on all of them, so ask it as a choice. And the red cursor marker in the
observation sometimes pulls the 6-option answer to "line".

JSPaint reads its canvas size from `localStorage` at start-up, so the environment writes `width` and `height`
there before the page loads. JSPaint itself is unmodified. The browser viewport is 900×720 so the whole canvas is
on screen.

### Move size

The model can only pick from a list of options, so the actions are relative moves. They are fine-grained because
coarse moves can't draw a round circle. Scripted-expert scores over 3 episodes, measured on the earlier 683×384
canvas (radius 115 px):

| Directions | Step | Expert score | Steps per circle |
|---|---|---|---|
| 8 | 24 px | 0.39 (an octagon; fails) | 42 |
| 16 | 12 px | 0.83 | 78 |
| 16 | 8 px | 0.91 | 116 |
| 32 | 8 px | 0.94 | 116 |
| **32** | **6 px (default)** | **0.95** | **139** |
| 32 | 4 px | 0.96 | 227 |

More directions matter most, because the heading error per step falls from 22.5° to 5.6°. Steps below 6 px add
little and make episodes longer. Both settings are configurable (`directions=8|16|32`, `step_px`; `--directions`,
`--step-px` on the command line).

The screenshot has no OS cursor, so the cursor is drawn onto it. This is the same approach as Maze and Snake.

The discrete actions are built on a mouse-only tool API: `move_mouse(dx, dy)`, `mouse_down()`, `mouse_up()` and
`screenshot()`. `paintenv.TOOLS` describes these tools as JSON-schema tool definitions, and
`JSPaintEnv.call_tool(name, args)` runs them. A tool-calling agent can therefore drive the same environment with
free coordinates.

## The circle verifier

`score_circle(pixels)` reads the canvas pixels from the page with `toDataURL`, so the cursor overlay is never
counted as ink. It fits a least-squares circle to the ink and computes five components, each between 0 and 1:

- **roundness**: how constant the radius is around the ring. It is `1 - std(r_k) / (0.05 r)`, where `r_k` is the
  mean radius of the ink in each of 36 angular sectors. Averaging within a sector cancels the stroke width, so this
  measures the shape itself: a clean ring scores 0.99, a 16-sided polygon 0.88, an octagon 0.56, and a square or a
  4:3 ellipse 0;
- **coverage**: the share of 36 angular sectors that contain ink on the ring;
- **clean**: the share of ink within `0.2 r` of the ring, so scribbles and fills count against the drawing;
- **closure**: 1 if no gap in the ring is wider than 30°, falling to 0 at a 120° gap;
- **size**: 1 if the radius is between 8% and 50% of the canvas height and the centre is on the canvas, else 0.

`score` is the product of the five. An episode `passed` if the score is at least 0.7 and the circle is closed.

`tests/test_circle_verifier.py` checks both directions:

- These pass: a drawn ring, a small ring, a coloured ring, and 16- and 32-sided polygons (what the fine moves draw).
- These all fail: an octagon, a line, a dot, a tiny ring, a half arc, a three-quarter arc, a square, a flat ellipse,
  a filled disc, a ring with a stray line, and random scribbles.

## Reference points

`circle_expert()` is a scripted policy. It walks with the pen up straight to the nearest point of a circle centred
on the canvas, and presses the button once it is within half a step of the radius. It then traces one full turn,
each step taking the forward move that lands closest to the radius. `random_policy` picks actions uniformly. On
seeds 900000–900002, on the 512×512 canvas with the default 32 directions and 6 px steps:

| Policy | Mean score | Pass rate | Mean steps |
|---|---|---|---|
| Scripted expert | 0.97 | 3/3 | 181.7 |
| Random | 0.00 | 0/3 | 22.0 |
| `thaitea/laya-vision`, zero-shot | 0.00 | 0/3 | 174.0 |

The zero-shot checkpoint picks `PEN_UP` on 521 of 522 steps, which does nothing while the pen is already up, and
never draws. Recognising a circle is not the same as knowing how to draw one: it was never trained on this task. The expert's trajectories (screenshot, action) are the obvious data to
train it on, as the game checkpoints were trained on expert frames.
