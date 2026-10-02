# Decide over a detector's boxes

A detector says where things are; Laya Vision scores a closed set of options. The layer in
[`laya/regions.py`](https://github.com/r33drichards/laya-vision/blob/main/laya/regions.py) connects them in two
ways:

- **select** asks one question over all the boxes, for example "which element should I click?". Each box is one option.
- **map** asks the same questions of each box's crop, for example "is this person wearing a helmet?", or a rubric.

Both can run over a stream of frames. This is a draft; the design is still being discussed.

> **Note:** No released checkpoint was trained on numbered marks or on detector crops, so nobody has measured how
> accurate these modes are on your task. Label a few hundred examples and measure it. After that,
> [calibrate](calibrate.md) if you act on thresholds.

## Plug in a detector

A detector is any callable `image -> output`. `laya.regions.as_regions` converts the output of these detector
families into a list of `Region` objects. Each `Region` has a pixel box `(x0, y0, x1, y1)`, and optionally a label,
a score and a mask:

| Detector | What to pass |
|---|---|
| OmniParser (screens and browsers) | the parsed content list (`bbox` given as fractions of the image; `content` becomes the label) |
| YOLO, RT-DETR, SAM or FastSAM via `ultralytics` | `model(img)[0]` (a `Results`) |
| RF-DETR, or anything in `supervision` | an `sv.Detections` |
| Segment Anything / SAM 2 automatic masks | the list of mask dicts (`bbox` is `xywh`) |
| `transformers` object-detection pipelines (DETR, OWL-ViT, Grounding DINO) | the pipeline output |
| anything else | a list of `Region`, `{"box": [x0, y0, x1, y1], "label": ...}` dicts, or `[x0, y0, x1, y1]` lists |

None of these libraries is a dependency. The conversion is duck-typed.

```python
import laya
from laya.regions import RegionPipeline

agent = laya.load_vlm("thaitea/laya-vision")

from ultralytics import YOLO
yolo = YOLO("yolo11n.pt")
pipe = RegionPipeline(agent, detector=lambda img: yolo(img, verbose=False)[0])
```

To swap in a different detector, change the lambda. For example, with RF-DETR:
`detector=lambda img: rfdetr.predict(img, threshold=0.5)`. `filter=` runs on every detection, so it can keep only
OmniParser's interactable elements, drop low scores or cap the count:

```python
pipe = RegionPipeline(agent, detector=omniparser_fn,
                      filter=lambda rs: [r for r in rs if r.meta.get("interactivity")][:30])
```

## Pick one box (select)

```python
out = pipe.select(screenshot, "Which element opens the account settings?",
                  none_option="none of the marked boxes", max_options=10)
out["index"], out["click"], out["confidence"]   # the chosen region, its centre, and how sure the model is
out["probabilities"]                             # {"1": 0.02, "2": 0.81, ..., "none": 0.05}, keyed by mark number
```

> **Caution:** on the released checkpoint, numbered marks mostly are not read. In an exploratory probe it read an
> 18 px digit 99% of the time, but picked the right one of 4 marked boxes only 40.5% of the time (chance 25%) and
> mostly answered "box 2"; 64 px tags raised that to 62.5%. The probe's options were the numbers only, as with
> `describe=False`. See [Detector regions and zoom search](../reference/results/detector-regions-and-zoom.md).

Each box is drawn on the image as a numbered outline. Each box also becomes one option of a `choice` question,
written `box N: <label>`, where the label is the detector's class or OmniParser's OCR text or caption. To send
the numbers only, pass `describe=False`.

- **More boxes than `max_options`** start a knockout. Each group of up to `max_options` boxes is asked about on
  its own, with only that group's marks drawn. The winners go on to the next round. `rounds` records every group,
  and `probabilities` come from the final round. Small groups keep the marks readable at the model's input
  resolution (512 px unless the checkpoint uses `image_split_edge`) and keep the options within `head_max_len`.
- **`none_option`** adds an abstain option to the final round, so the model can say that no box fits.
- **`state_text={"goal": ...}`** adds text to the state, next to the image.
- **Other keywords** go to `predict`, for example `n_permutations=4` to average out option-order bias.

## Ask about every box (map)

```python
checks = {
    "helmet": {"type": "noul", "instructions": "Is this person wearing a hard hat?"},
    "pose": {"type": "choice", "instructions": "What is the person doing?",
             "criteria": ["standing", "climbing a ladder", "kneeling", "lying down"]},
}
for rec in pipe.map(photo, checks, pad=0.15):
    print(rec["region"].box, rec["answers"]["helmet"]["noul"], rec["answers"]["pose"]["choice"])
```

Each crop is the box padded by `pad` of its own size and grown to at least `min_size` pixels. The answers are
exactly what `predict` returns for that crop. `with_label=True` puts the detector's label in the state text. That
helps with a rubric about "this forklift", but it can also pull the answer toward the detector's guess.
`laya.regions.imap_regions` yields the records one at a time instead of building a list.

## Zoom search

The checkpoint sees an image as one 512 px tile, so small objects in a large image are lost. On V*Bench, asking
`noul` "Is there a &lt;target&gt; in this image?" over quadrants and then sub-quadrants, and answering on the best crop,
raised attribute accuracy from 22% to 48% in an exploratory probe. Asking "where is the &lt;target&gt;?" as a `choice`
did not work. There is no API for this in `laya/regions.py` yet (the probe scores the crops with
`laya.search.score_states`); see
[Detector regions and zoom search](../reference/results/detector-regions-and-zoom.md) for the probes and scripts.

## Run on a stream

```python
import cv2
def frames():
    cap = cv2.VideoCapture(0)
    while True:
        ok, bgr = cap.read()
        if not ok:
            return
        yield bgr[:, :, ::-1].copy()          # RGB

for rec in pipe.stream(frames(), map=checks, drop_stale=True):
    print(rec["index"], rec["dropped"], rec["detect_ms"], rec["decide_ms"], rec["result"])
```

A background thread detects the next frame while the decision model works on the current one. `prefetch=0` runs
both steps in turn instead. With `drop_stale=True`, when the decision model falls behind a live source, the newest
detected frame replaces the one waiting in the queue, and `dropped` counts the frames skipped. Give `stream`
either `select="..."` or `map={...}`. Detector errors are raised from the loop.

## Without a pipeline

The functions take the regions directly: `laya.regions.select(agent, image, regions, instructions)`,
`map_regions(agent, image, regions, questions)`, and
`stream(frames, detector, decide=lambda frame, regions: ...)`. Every result carries `regions_format`, the version
of how regions become model inputs (mark drawing, option texts, crop padding), alongside `predict`'s `provenance`.
