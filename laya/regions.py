"""Detector -> decision: put the regions a detector finds in front of the decision model.

A detector (OmniParser on a screenshot, YOLO, RF-DETR, a Segment Anything mask generator, a ``transformers``
object-detection pipeline, anything that returns boxes) says *where* things are. ``VLMAgent.predict`` scores a
closed set of options with calibrated probabilities. This module joins the two in the two shapes that come up:

* **select** (one decision over all regions): which region should be clicked / is the target / answers the
  question? Every region is drawn on the image as a numbered box (Set-of-Mark style) and becomes one option of a
  ``choice`` question, keyed by its number and described by the detector's label or OCR text when there is one.
  The answer is a region, with a probability for every region (and for ``none_option`` if given).
* **map** (one decision per region): crop every region and ask the same questions of each crop, e.g. a
  ``choice`` over classes, a ``noul`` check or a ``score`` rubric. The answers are ``predict``'s, one per region.

``stream`` runs either over a sequence of frames, detecting the next frame on a background thread while the
decision model works on the current one (``drop_stale`` skips frames the decision model cannot keep up with).

Detectors are swappable: ``as_regions`` turns the output of each supported family into a list of ``Region``
(pixel ``xyxy`` box, optional label, score and mask), duck-typed so none of those libraries is imported here.
A detector is any callable ``image -> output`` whose output ``as_regions`` accepts (or a list of ``Region``).

Caveat: no released checkpoint was trained on numbered marks or on crops chosen by a detector, so ``select`` and
``map`` accuracy on your task is unmeasured until you measure it; ``VLMAgent.calibrate`` on a few labelled
examples fixes the probabilities, not the ranking. The model sees the image at the checkpoint's input resolution
(512 px unless it was trained with ``image_split_edge``), so marks on a full-size screenshot can be too small to
read: fewer options per round (``max_options``) and ``map``'s crops both help.
"""
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

#: Version of how regions become model inputs here (mark drawing, option keys and texts, crop padding). Reported
#: in every ``select`` / ``map_regions`` result as ``"regions_format"``; bump it with any change to those.
REGIONS_FORMAT_VERSION = "1"
#: characters of a region's label kept in its option text (options are cut to 48 tokens by ``predict`` anyway)
MAX_LABEL_CHARS = 80
NONE_KEY = "none"
_PALETTE = [(230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48), (145, 30, 180), (70, 240, 240),
            (240, 50, 230), (210, 245, 60), (0, 128, 128), (170, 110, 40), (128, 0, 0), (0, 0, 128)]


@dataclass
class Region:
    """One detection: ``box`` is ``(x0, y0, x1, y1)`` in pixels of the image it was found in."""

    box: Tuple[float, float, float, float]
    label: Optional[str] = None
    score: Optional[float] = None
    mask: Optional[np.ndarray] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def center(self) -> Tuple[float, float]:
        x0, y0, x1, y1 = self.box
        return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)

    @property
    def area(self) -> float:
        x0, y0, x1, y1 = self.box
        return max(0.0, x1 - x0) * max(0.0, y1 - y0)

    def crop_box(self, size: Tuple[int, int], pad: float = 0.1, min_size: int = 32) -> Tuple[int, int, int, int]:
        """The integer crop around the box in an image of ``size`` (w, h): padded by ``pad`` of the box's own
        width/height on each side, grown to at least ``min_size`` px per side, clamped to the image."""
        w, h = size
        x0, y0, x1, y1 = self.box
        bw, bh = max(1.0, x1 - x0), max(1.0, y1 - y0)
        px, py = max(bw * pad, (min_size - bw) / 2.0), max(bh * pad, (min_size - bh) / 2.0)
        cx0, cy0 = int(max(0, np.floor(x0 - px))), int(max(0, np.floor(y0 - py)))
        cx1, cy1 = int(min(w, np.ceil(x1 + px))), int(min(h, np.ceil(y1 + py)))
        return cx0, cy0, max(cx1, cx0 + 1), max(cy1, cy0 + 1)

    def crop(self, image, pad: float = 0.1, min_size: int = 32):
        image = _pil(image)
        return image.crop(self.crop_box(image.size, pad, min_size))

    def to_dict(self) -> Dict[str, Any]:
        d = {"box": [round(float(v), 2) for v in self.box]}
        if self.label is not None:
            d["label"] = self.label
        if self.score is not None:
            d["score"] = round(float(self.score), 4)
        if self.meta:
            d["meta"] = self.meta
        return d


# ---------------------------------------------------------------------------------------------------------
# Detector adapters
# ---------------------------------------------------------------------------------------------------------


def _np(x) -> Optional[np.ndarray]:
    if x is None:
        return None
    if hasattr(x, "detach"):  # a torch tensor
        x = x.detach().cpu()
    return np.asarray(x)


def _pil(image):
    from PIL import Image

    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, np.ndarray):
        return Image.fromarray(image).convert("RGB")
    with Image.open(image) as im:  # a path
        return im.convert("RGB")


def _image_size(image) -> Tuple[int, int]:
    if hasattr(image, "size") and not isinstance(image, np.ndarray):
        return tuple(image.size)
    a = np.asarray(image)
    return a.shape[1], a.shape[0]


def from_xyxy(boxes, labels: Optional[Sequence] = None, scores: Optional[Sequence] = None,
              masks: Optional[Sequence] = None) -> List[Region]:
    """Regions from parallel arrays: pixel ``xyxy`` boxes and optional labels, scores and masks."""
    boxes = _np(boxes)
    if boxes is None or len(boxes) == 0:
        return []
    scores, masks = _np(scores), _np(masks) if masks is not None else None
    return [Region(tuple(float(v) for v in b[:4]),
                   None if labels is None or labels[i] is None else str(labels[i]),
                   None if scores is None else float(scores[i]),
                   None if masks is None else masks[i]) for i, b in enumerate(boxes)]


def from_supervision(det) -> List[Region]:
    """``supervision.Detections`` (what ``rfdetr`` returns, and ``from_ultralytics`` / ``from_sam`` build)."""
    names = None
    data = getattr(det, "data", None) or {}
    if "class_name" in data:
        names = list(data["class_name"])
    elif getattr(det, "class_id", None) is not None:
        names = [str(int(c)) for c in det.class_id]
    return from_xyxy(det.xyxy, names, getattr(det, "confidence", None), getattr(det, "mask", None))


def from_ultralytics(result) -> List[Region]:
    """One ``ultralytics`` ``Results`` (YOLO detect/segment, RT-DETR, SAM, FastSAM): ``model(img)[0]``."""
    boxes = result.boxes
    if boxes is None:
        return []
    cls = _np(getattr(boxes, "cls", None))
    names = getattr(result, "names", None) or {}
    labels = None if cls is None else [names.get(int(c), str(int(c))) for c in cls]
    masks = getattr(result, "masks", None)
    return from_xyxy(boxes.xyxy, labels, getattr(boxes, "conf", None), None if masks is None else _np(masks.data))


def from_sam(masks: Sequence[Dict]) -> List[Region]:
    """``segment_anything`` / SAM 2 automatic mask generator output: dicts with an ``xywh`` ``bbox``."""
    out = []
    for m in masks:
        x, y, w, h = (float(v) for v in m["bbox"])
        score = m.get("predicted_iou", m.get("stability_score"))
        out.append(Region((x, y, x + w, y + h), None, None if score is None else float(score), m.get("segmentation"),
                          {k: m[k] for k in ("area", "stability_score") if k in m}))
    return out


def from_omniparser(items: Sequence[Dict], image_size: Optional[Tuple[int, int]] = None) -> List[Region]:
    """OmniParser's parsed content list: dicts with ``bbox`` (``xyxy``, as fractions of the image), ``type``
    (``"text"`` / ``"icon"``), ``content`` (OCR text or icon caption) and ``interactivity``. ``image_size`` (w, h)
    scales fractional boxes to pixels; boxes already in pixels (any coordinate > 1) are kept."""
    out = []
    for it in items:
        b = [float(v) for v in it["bbox"]]
        if max(b) <= 1.0:
            if image_size is None:
                raise ValueError("OmniParser boxes are fractions of the image: pass image_size=(w, h)")
            w, h = image_size
            b = [b[0] * w, b[1] * h, b[2] * w, b[3] * h]
        content = it.get("content")
        meta = {k: it[k] for k in ("type", "interactivity", "source") if k in it}
        out.append(Region(tuple(b), None if content in (None, "") else str(content).strip(), None, None, meta))
    return out


def from_hf_pipeline(outputs: Sequence[Dict]) -> List[Region]:
    """A ``transformers`` object-detection / zero-shot-object-detection pipeline's output (DETR, OWL-ViT,
    Grounding DINO...): dicts with ``score``, ``label`` and ``box = {xmin, ymin, xmax, ymax}``."""
    return [Region((float(o["box"]["xmin"]), float(o["box"]["ymin"]), float(o["box"]["xmax"]), float(o["box"]["ymax"])),
                   o.get("label"), o.get("score")) for o in outputs]


def as_regions(output: Any, image_size: Optional[Tuple[int, int]] = None) -> List[Region]:
    """Whatever a supported detector returned, as ``Region`` s. Recognised: a list of ``Region``, supervision
    ``Detections``, an ultralytics ``Results`` (or a one-element list of them), SAM mask dicts, OmniParser items,
    ``transformers`` pipeline dicts, dicts with ``box``/``bbox`` in pixel ``xyxy``, and ``[x0, y0, x1, y1]`` lists."""
    if output is None:
        return []
    if hasattr(output, "xyxy") and hasattr(output, "confidence"):
        return from_supervision(output)
    if hasattr(output, "boxes") and hasattr(output, "names"):
        return from_ultralytics(output)
    items = list(output)
    if not items:
        return []
    first = items[0]
    if isinstance(first, Region):
        return items
    if hasattr(first, "boxes") and hasattr(first, "names"):
        if len(items) != 1:
            raise ValueError("got %d ultralytics results; pass one image's result" % len(items))
        return from_ultralytics(first)
    if isinstance(first, dict):
        if "segmentation" in first and "bbox" in first:
            return from_sam(items)
        if "bbox" in first and ("content" in first or "interactivity" in first):
            return from_omniparser(items, image_size)
        if isinstance(first.get("box"), dict):
            return from_hf_pipeline(items)
        return [Region(tuple(float(v) for v in (d.get("box") or d["bbox"])[:4]), d.get("label"), d.get("score"),
                       d.get("mask"), d.get("meta") or {}) for d in items]
    return from_xyxy(items)


def detect(detector: Callable, image) -> List[Region]:
    """Run ``detector(image)`` and convert its output (``as_regions``, with the image's size for fractional boxes)."""
    return as_regions(detector(image), _image_size(image))


# ---------------------------------------------------------------------------------------------------------
# select: one choice over the regions
# ---------------------------------------------------------------------------------------------------------


def draw_marks(image, regions: Sequence[Region], numbers: Sequence[int], width: Optional[int] = None,
               font_size: Optional[int] = None, color: Optional[Tuple[int, int, int]] = None):
    """A copy of ``image`` with each region outlined and tagged with its number (Set-of-Mark style). Marks cycle
    through a palette unless ``color`` fixes one; the tag's text is ``font_size`` px (default 1/30 of the short side)."""
    from PIL import ImageDraw, ImageFont

    img = _pil(image).copy()
    w, h = img.size
    lw = width or max(2, min(w, h) // 200)
    size = font_size or max(12, min(w, h) // 30)
    try:
        font = ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1: fixed-size bitmap font
        font = ImageFont.load_default()
    draw = ImageDraw.Draw(img)
    for r, n in zip(regions, numbers):
        c = color or _PALETTE[(n - 1) % len(_PALETTE)]
        x0, y0, x1, y1 = r.box
        draw.rectangle([x0, y0, x1, y1], outline=c, width=lw)
        tag = str(n)
        tx0, ty0, tx1, ty1 = draw.textbbox((0, 0), tag, font=font)
        tw, th = tx1 - tx0 + 2 * lw, ty1 - ty0 + 2 * lw
        ax, ay = max(0, min(x0, w - tw)), max(0, min(y0, h - th))
        draw.rectangle([ax, ay, ax + tw, ay + th], fill=c)
        draw.text((ax + lw - tx0, ay + lw - ty0), tag, fill=(255, 255, 255), font=font)
    return img


def _describe(r: Region) -> Optional[str]:
    if not r.label:
        return None
    label = " ".join(r.label.split())
    return label if len(label) <= MAX_LABEL_CHARS else label[: MAX_LABEL_CHARS - 3] + "..."


def select_question(regions: Sequence[Region], numbers: Sequence[int], instructions: str,
                    none_option: Optional[str] = None, describe: bool = True) -> Dict[str, Any]:
    """The ``choice`` question ``select`` asks: one option per region keyed by its mark number, described by
    ``"box N: <label>"`` (just ``"box N"`` without a label or with ``describe=False``), plus ``"none"``."""
    crit = {}
    for r, n in zip(regions, numbers):
        text = _describe(r) if describe else None
        crit[str(n)] = "box %d: %s" % (n, text) if text else "box %d" % n
    if none_option:
        crit[NONE_KEY] = none_option
    return {"type": "choice", "instructions": instructions, "criteria": crit}


def select(agent, image, regions: Sequence[Region], instructions: str, max_options: int = 10,
           none_option: Optional[str] = None, describe: bool = True, marks: bool = True,
           state_text: Optional[Dict[str, Any]] = None, **predict_kwargs) -> Dict[str, Any]:
    """Pick one of ``regions`` for ``instructions`` ("Which element opens the settings?").

    Regions are numbered 1..N in the order given (sort them first if order should matter, e.g. by detector score).
    With more than ``max_options`` regions it runs a knockout: each group of up to ``max_options`` is asked on
    its own (only that group's marks drawn), the group winners go to the next round, until one round is left;
    ``probabilities`` are that final round's, so a region knocked out earlier has none. ``none_option`` (text,
    e.g. "none of the marked boxes") adds a ``"none"`` option to the final round. ``marks=False`` sends the
    image unmarked (only useful with ``describe`` and labels). ``state_text`` adds keys to the state, which
    ``predict`` serializes as text next to the image. Other keywords go to ``agent.predict``.

    Returns ``{"index", "region", "number", "click", "probabilities", "confidence", "rounds", "answer",
    "provenance", "regions_format"}``: ``index`` into ``regions`` (None when ``none`` wins or there are no
    regions), ``click`` the region's centre, ``probabilities`` keyed by mark number (and ``"none"``).
    """
    if max_options < 2:
        raise ValueError("max_options must be at least 2")
    base = _pil(image)
    alive = list(range(len(regions)))
    rounds: List[List[Dict[str, Any]]] = []
    final = None
    if not alive:
        return {"index": None, "region": None, "number": None, "click": None, "probabilities": {},
                "confidence": None, "rounds": [], "answer": None, "provenance": None,
                "regions_format": REGIONS_FORMAT_VERSION}
    while True:
        last = len(alive) <= max_options
        groups = [alive[s: s + max_options] for s in range(0, len(alive), max_options)]
        results, winners = [], []
        for g in groups:
            if len(g) == 1 and not last:  # a bye
                winners.append(g[0])
                results.append({"candidates": [g[0]], "winner": g[0], "probabilities": None})
                continue
            nums = [i + 1 for i in g]
            q = select_question([regions[i] for i in g], nums, instructions, none_option if last else None, describe)
            state = dict(state_text or {})
            state["image"] = draw_marks(base, [regions[i] for i in g], nums) if marks else base
            res = agent.predict(state, {"region": q}, **predict_kwargs)
            ans = res["answers"]["region"]
            key = ans["choice"]
            win = None if key == NONE_KEY else int(key) - 1
            winners.append(win)
            results.append({"candidates": list(g), "winner": win, "probabilities": ans["probabilities"]})
            final = (ans, res)
        rounds.append(results)
        if last:
            break
        alive = [w for w in winners if w is not None]
    ans, res = final
    idx = rounds[-1][0]["winner"]
    r = None if idx is None else regions[idx]
    return {
        "index": idx,
        "region": r,
        "number": None if idx is None else idx + 1,
        "click": None if r is None else r.center,
        "probabilities": ans["probabilities"],
        "confidence": ans["confidence"],
        "rounds": rounds,
        "answer": ans,
        "provenance": res.get("provenance"),
        "regions_format": REGIONS_FORMAT_VERSION,
    }


_select = select  # ``RegionPipeline`` methods take a ``select=`` argument that shadows the function


# ---------------------------------------------------------------------------------------------------------
# map: the same questions of every region
# ---------------------------------------------------------------------------------------------------------


def imap_regions(agent, image, regions: Sequence[Region], questions: Dict[str, Dict[str, Any]], pad: float = 0.1,
                 min_size: int = 32, with_label: bool = False, **predict_kwargs) -> Iterator[Dict[str, Any]]:
    """Lazily ask ``questions`` of each region's crop, yielding ``{"index", "region", "crop_box", "answers",
    "provenance", "regions_format"}`` in region order as each finishes.

    The crop is the box padded by ``pad`` of its size and grown to ``min_size`` px (``Region.crop_box``).
    ``with_label=True`` puts the detector's label in the state text (``{"detector_label": ...}``), which helps a
    rubric about "this <label>" and can also lean the answer toward the detector's guess.
    """
    base = _pil(image)
    for i, r in enumerate(regions):
        cb = r.crop_box(base.size, pad, min_size)
        state: Dict[str, Any] = {"image": base.crop(cb)}
        if with_label and r.label:
            state["detector_label"] = r.label
        res = agent.predict(state, questions, **predict_kwargs)
        yield {"index": i, "region": r, "crop_box": cb, "answers": res["answers"],
               "provenance": res.get("provenance"), "regions_format": REGIONS_FORMAT_VERSION}


def map_regions(agent, image, regions: Sequence[Region], questions: Dict[str, Dict[str, Any]],
                **kwargs) -> List[Dict[str, Any]]:
    """``imap_regions`` collected into a list (one entry per region, in order)."""
    return list(imap_regions(agent, image, regions, questions, **kwargs))


# ---------------------------------------------------------------------------------------------------------
# streaming
# ---------------------------------------------------------------------------------------------------------

_DONE = object()


def stream(frames: Iterable, detector: Callable, decide: Callable[[Any, List[Region]], Any], prefetch: int = 1,
           drop_stale: bool = False) -> Iterator[Dict[str, Any]]:
    """Run ``detector`` then ``decide(frame, regions)`` over ``frames``, yielding one record per decided frame:
    ``{"index", "frame", "regions", "result", "detect_ms", "decide_ms", "dropped"}``.

    Detection runs on a background thread up to ``prefetch`` frames ahead of the decision model (``prefetch=0``
    runs both in turn on the caller's thread). ``drop_stale=True`` is for live sources: when the decision model
    falls behind, a detected frame waiting in the queue is replaced by the newest one, so decisions stay current;
    ``dropped`` counts the frames skipped since the previous record. An exception in the detector or the frame
    source is raised from the generator. Closing the generator stops the thread at its next frame.
    """
    if prefetch <= 0:
        for i, frame in enumerate(frames):
            t0 = time.perf_counter()
            regions = detect(detector, frame)
            t1 = time.perf_counter()
            result = decide(frame, regions)
            yield {"index": i, "frame": frame, "regions": regions, "result": result,
                   "detect_ms": (t1 - t0) * 1e3, "decide_ms": (time.perf_counter() - t1) * 1e3, "dropped": 0}
        return

    q: "queue.Queue" = queue.Queue(maxsize=prefetch)
    stop = threading.Event()

    def put(item):
        while not stop.is_set():
            if drop_stale and item is not _DONE and not isinstance(item, BaseException):
                try:
                    q.put_nowait(item)
                    return
                except queue.Full:
                    try:
                        q.get_nowait()  # drop the oldest waiting frame (unless the consumer just took it)
                    except queue.Empty:
                        pass
                continue
            try:
                q.put(item, timeout=0.1)
                return
            except queue.Full:
                continue

    def produce():
        try:
            for i, frame in enumerate(frames):
                if stop.is_set():
                    return
                t0 = time.perf_counter()
                regions = detect(detector, frame)
                put((i, frame, regions, (time.perf_counter() - t0) * 1e3))
            put(_DONE)
        except BaseException as e:  # handed to the consumer
            put(e)

    th = threading.Thread(target=produce, name="laya-regions-detect", daemon=True)
    th.start()
    last = -1
    try:
        while True:
            item = q.get()
            if item is _DONE:
                return
            if isinstance(item, BaseException):
                raise item
            i, frame, regions, detect_ms = item
            t1 = time.perf_counter()
            result = decide(frame, regions)
            yield {"index": i, "frame": frame, "regions": regions, "result": result, "detect_ms": detect_ms,
                   "decide_ms": (time.perf_counter() - t1) * 1e3, "dropped": i - last - 1}
            last = i
    finally:
        stop.set()


class RegionPipeline:
    """A detector and a decision model together.

    >>> pipe = RegionPipeline(agent, detector=lambda img: yolo(img)[0])      # any detector
    >>> pipe.select(screenshot, "Which element opens the settings?")        # one of N regions
    >>> pipe.map(photo, {"helmet": {"type": "noul", "instructions": "Is this person wearing a helmet?"}})
    >>> for rec in pipe.stream(frames, select="Which car is closest?", drop_stale=True): ...

    ``filter`` (``regions -> regions``) runs after every detection, e.g. to keep OmniParser's interactable
    elements, drop low scores or cap the count.
    """

    def __init__(self, agent, detector: Callable, filter: Optional[Callable[[List[Region]], List[Region]]] = None):
        self.agent = agent
        self.detector = detector
        self.filter = filter

    def _detector(self, image):
        regions = detect(self.detector, image)
        return self.filter(regions) if self.filter else regions

    def detect(self, image) -> List[Region]:
        return self._detector(image)

    def select(self, image, instructions: str, regions: Optional[List[Region]] = None, **kwargs) -> Dict[str, Any]:
        regions = self._detector(image) if regions is None else regions
        return select(self.agent, image, regions, instructions, **kwargs)

    def map(self, image, questions: Dict[str, Dict[str, Any]], regions: Optional[List[Region]] = None,
            **kwargs) -> List[Dict[str, Any]]:
        regions = self._detector(image) if regions is None else regions
        return map_regions(self.agent, image, regions, questions, **kwargs)

    def stream(self, frames: Iterable, select: Optional[str] = None,
               map: Optional[Dict[str, Dict[str, Any]]] = None, prefetch: int = 1, drop_stale: bool = False,
               **kwargs) -> Iterator[Dict[str, Any]]:
        """``stream`` with this pipeline's detector; give exactly one of ``select`` (instructions) or ``map``
        (questions). Other keywords go to that call."""
        if (select is None) == (map is None):
            raise ValueError("give exactly one of select= or map=")
        if select is not None:
            decide = lambda frame, regions: _select(self.agent, frame, regions, select, **kwargs)  # noqa: E731
        else:
            decide = lambda frame, regions: map_regions(self.agent, frame, regions, map, **kwargs)  # noqa: E731
        return stream(frames, self._detector, decide, prefetch=prefetch, drop_stale=drop_stale)
