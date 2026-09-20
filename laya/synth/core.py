"""Synthetic typed-question datasets from existing labelled image sets.

No public dataset is labelled as Laya questions (``choice`` / ``score`` / ``noul`` over an image plus a note), so
this package makes them: each *source* (``laya.synth.sources``) turns a labelled row of an existing dataset into
one or more typed questions, and ``run_source`` streams a source from the Hugging Face Hub into the prepared
dataset layout that ``laya.vlm_train.load_jsonl_examples`` reads::

    <root>/<name>/
        train.jsonl, val.jsonl      {"id", "image" | "images", "state_text", "question", "label", "target"?,
                                     "family", "source"}
        images/<key>.jpg            one file per source image, shared by all of its questions
        meta.json                   origin, licence, counts per split and question family
        _READY                      written last

Rules every source follows:

* **Soft targets whenever the source has votes.** ``target`` is the annotators' (or raters') distribution over the
  options; ``label`` is its argmax and is what accuracy is scored on.
* **Distractors come from the same dataset**, sampled from a pool of other rows' answers (``Pool``), so they are
  plausible for the question rather than random strings. They never match any accepted answer of the row.
* **All questions on one image land in one split.** Sources with an official validation split use it; the rest
  are split by a hash of the image key (``split_of``).
"""
import hashlib
import json
import os
import random
import re
import shutil
import time
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

STOPWORDS = frozenset("""a an the of in on at to for with and or is are be this that these those it its from by as
page screen app application display displaying displays shows showing show option options window image images
photo picture website site web""".split())


def norm(s: str) -> str:
    """Lower-case, collapse whitespace and strip punctuation at the ends, for comparing answers."""
    return re.sub(r"\s+", " ", str(s).strip().lower()).strip(" .,!?'\"")


def content_words(s: str) -> set:
    return {w for w in re.findall(r"[a-z0-9]+", norm(s)) if len(w) > 3 and w not in STOPWORDS}


def pick(rng: random.Random, phrasings: Sequence[str], **fmt) -> str:
    """One of several phrasings of the same question, so the model reads instructions rather than a template."""
    return rng.choice(list(phrasings)).format(**fmt)


def split_of(key: str, val_frac: float, seed: int = 0) -> str:
    """Deterministic train/val assignment by image key: every question on an image goes to the same split."""
    h = hashlib.sha1(("%d:%s" % (seed, key)).encode()).digest()
    return "val" if int.from_bytes(h[:4], "big") / 2**32 < val_frac else "train"


# ---------------------------------------------------------------------------------------------------------
# Question builders (the JSONL ``question`` dicts, i.e. the public ``predict`` format)
# ---------------------------------------------------------------------------------------------------------


def choice_q(instructions: str, options: Sequence[str]) -> Dict:
    return {"type": "choice", "instructions": instructions, "criteria": [str(o) for o in options]}


def noul_q(instructions: str) -> Dict:
    return {"type": "noul", "instructions": instructions}


def score_q(instructions: str, levels: Sequence[str]) -> Dict:
    return {"type": "score", "instructions": instructions, "criteria": [str(l) for l in levels]}


class Example:
    """One typed question about a row's image(s).

    ``target`` is optional: a probability per option in ``label`` order (choice: criteria order; score: level
    order; noul: [false, true]). ``images`` is optional and overrides the row's own image(s).
    """

    __slots__ = ("family", "question", "label", "target", "state_text", "images")

    def __init__(self, family: str, question: Dict, label: int, target: Optional[Sequence[float]] = None,
                 state_text: Optional[str] = None, images: Optional[List[Any]] = None):
        self.family, self.question, self.label = family, question, int(label)
        self.target = [float(p) for p in target] if target is not None else None
        self.state_text, self.images = state_text, images

    def n_options(self) -> int:
        q = self.question
        return len(q["criteria"]) if q["type"] in ("choice", "score") else 2

    def valid(self) -> bool:
        q, k = self.question, self.n_options()
        if not isinstance(q.get("instructions"), str) or not q["instructions"].strip():
            return False
        if q["type"] == "choice" and len({norm(c) for c in q["criteria"]}) != len(q["criteria"]):
            return False  # duplicate options collapse when rendered
        if q["type"] in ("choice", "score") and k < 2:
            return False
        if not 0 <= self.label < k:
            return False
        if self.target is not None and (len(self.target) != k or min(self.target) < 0 or sum(self.target) <= 0):
            return False
        return True


def soft_target(votes: Dict[str, float], options: Sequence[str]) -> List[float]:
    """Vote counts keyed by (normalised) answer -> a distribution over ``options``; unmatched votes are dropped."""
    t = [float(votes.get(norm(o), 0.0)) for o in options]
    s = sum(t)
    return [p / s for p in t] if s > 0 else [1.0 / len(options)] * len(options)


# ---------------------------------------------------------------------------------------------------------
# Distractor pool
# ---------------------------------------------------------------------------------------------------------


class Pool:
    """Unique strings per key from rows seen so far, sampled as distractors for later rows.

    ``sample`` never returns a string that normalises to one in ``exclude``, and, with ``disjoint=True``, none that
    shares a content word with any excluded string (keeps "Fashion Brand" from being a distractor for
    "Fashion Retailer"). Each key keeps at most ``cap`` strings; later additions replace random earlier ones so
    the pool keeps drifting over a long stream.
    """

    def __init__(self, cap: int = 4000, seed: int = 0):
        self.items: Dict[str, List[str]] = defaultdict(list)
        self.seen: Dict[str, set] = defaultdict(set)
        self.cap, self.rng = cap, random.Random(seed)

    def add(self, key: str, s: str) -> None:
        s = str(s).strip()
        n = norm(s)
        if not n or n in self.seen[key]:
            return
        self.seen[key].add(n)
        bucket = self.items[key]
        if len(bucket) < self.cap:
            bucket.append(s)
        else:
            bucket[self.rng.randrange(self.cap)] = s

    def size(self, key: str) -> int:
        return len(self.items.get(key, ()))

    def sample(self, key: str, k: int, rng: random.Random, exclude: Iterable[str] = (), disjoint: bool = False,
               fallback: Union[None, str, Sequence[str]] = None) -> List[str]:
        """Up to ``k`` distinct distractors for ``key``, then from the ``fallback`` key(s) in order if short.
        May return fewer."""
        ex_norm = {norm(e) for e in exclude}
        ex_words = set().union(*(content_words(e) for e in exclude)) if disjoint else set()
        out, out_norm = [], set()
        keys = [key] + ([] if fallback is None else [fallback] if isinstance(fallback, str) else list(fallback))
        for kk in keys:
            bucket = self.items.get(kk, [])
            if not bucket:
                continue
            if len(bucket) <= 40 * k:  # small pool: one shuffled pass sees every entry
                order = list(bucket)
                rng.shuffle(order)
            else:
                order = (bucket[rng.randrange(len(bucket))] for _ in range(40 * k))
            for s in order:
                n = norm(s)
                if n in ex_norm or n in out_norm:
                    continue
                if disjoint and content_words(s) & ex_words:
                    continue
                out.append(s)
                out_norm.add(n)
                if len(out) >= k:
                    return out
        return out


# ---------------------------------------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------------------------------------


class DatasetWriter:
    """Writes ``<root>/<name>/`` atomically: everything goes to ``<name>.tmp`` and is renamed at ``finish``."""

    SPLITS = ("train", "val")

    def __init__(self, root: str, name: str, max_side: int = 1024, jpeg_quality: int = 90):
        self.root, self.name = root, name
        self.final = os.path.join(root, name)
        self.tmp = self.final + ".tmp"
        self.max_side, self.jpeg_quality = max_side, jpeg_quality
        shutil.rmtree(self.tmp, ignore_errors=True)
        os.makedirs(os.path.join(self.tmp, "images"))
        self.files = {s: open(os.path.join(self.tmp, s + ".jsonl"), "w") for s in self.SPLITS}
        self.records = {s: 0 for s in self.SPLITS}
        self.families = {s: Counter() for s in self.SPLITS}
        self.labels = {s: defaultdict(Counter) for s in self.SPLITS}
        self.images = {s: set() for s in self.SPLITS}
        self._saved: Dict[str, str] = {}

    def save_image(self, key: str, image, split: str) -> str:
        """Save one PIL image once per key (resized so its longest side is ``max_side``); returns its relative path."""
        if key in self._saved:
            return self._saved[key]
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", key)
        rel = "images/%s.jpg" % safe
        img = image.convert("RGB")
        w, h = img.size
        scale = self.max_side / max(w, h)
        if scale < 1:
            img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))))
        img.save(os.path.join(self.tmp, rel), quality=self.jpeg_quality)
        self._saved[key] = rel
        self.images[split].add(key)
        return rel

    def write(self, split: str, rec: Dict) -> None:
        self.files[split].write(json.dumps(rec) + "\n")
        self.records[split] += 1
        self.families[split][rec.get("family", "?")] += 1
        self.labels[split][rec.get("family", "?")][str(rec["label"])] += 1

    def summary(self) -> Dict:
        return {s: {"records": self.records[s], "images": len(self.images[s]), "families": dict(self.families[s]),
                    "labels": {f: dict(c) for f, c in self.labels[s].items()}} for s in self.SPLITS}

    def finish(self, meta: Dict) -> Dict:
        for f in self.files.values():
            f.close()
        meta = dict(meta, **self.summary())
        with open(os.path.join(self.tmp, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        shutil.rmtree(self.final, ignore_errors=True)
        os.rename(self.tmp, self.final)
        open(os.path.join(self.final, "_READY"), "w").close()
        return meta

    def abort(self) -> None:
        for f in self.files.values():
            f.close()
        shutil.rmtree(self.tmp, ignore_errors=True)


# ---------------------------------------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------------------------------------


def _hf_rows(repo: str, split: str, config: Optional[str], columns: Optional[Sequence[str]] = None,
             data_files: Optional[Dict[str, str]] = None) -> Iterator[Dict]:
    """Stream one Hub split in file order, keeping only ``columns``. ``data_files`` maps split names to parquet
    globs for repos whose card does not declare the split (VQAv2's train files).

    No ``.shuffle``: on these image datasets even ``buffer_size=1`` took about 10 GB of RSS against 3 GB for a
    plain stream (measured on VizWiz), so a run takes the first rows of the split. Streams are read one parquet
    row group at a time, which is where the 3 GB goes.
    """
    from datasets import load_dataset

    kw = {"name": config} if config else {}
    if data_files:
        kw["data_files"] = {split: data_files[split]}
    ds = load_dataset(repo, split=split, streaming=True, **kw)
    if columns:
        ds = ds.select_columns([c for c in columns if c in (ds.column_names or columns)])
    return iter(ds)


def run_source(source, writer: DatasetWriter, n_train: int, n_val: int, seed: int = 0, val_frac: float = 0.05,
               rows: Optional[Dict[str, Iterable[Dict]]] = None, log=print, min_pool: int = 200) -> Dict:
    """Stream ``source`` into ``writer`` until each split has its target number of records (or the stream ends).

    ``rows`` overrides the Hugging Face streams per HF split name (tests). Rows seen before the source's pools
    hold ``min_pool`` entries only feed the pools, so distractors are never sampled from a near-empty pool.
    """
    rng = random.Random(seed)
    targets = {"train": n_train, "val": n_val}
    t0 = time.time()
    plan: List[Tuple[str, Optional[str]]] = [(source.train_split, None if source.val_split is None else "train")]
    if source.val_split is not None:
        plan.append((source.val_split, "val"))
    dropped = Counter()
    for hf_split, fixed_split in plan:
        stream = rows[hf_split] if rows is not None else _hf_rows(source.repo, hf_split, source.config, source.columns,
                                                                  source.data_files)
        for idx, row in enumerate(stream):
            if all(writer.records[s] >= targets[s] for s in writer.SPLITS if fixed_split in (None, s)):
                break
            key = "%s-%s" % (source.name, source.key(row, idx))
            split = fixed_split or split_of(key, val_frac, seed)
            source.observe(row)
            if writer.records[split] >= targets[split]:
                continue
            if not source.ready(min_pool):
                dropped["warmup"] += 1
                continue
            examples = [ex for ex in source.examples(row, rng) if ex.valid()]
            if not examples:
                dropped["no_examples"] += 1
                continue
            rec_images = None
            for j, ex in enumerate(examples):
                imgs = ex.images if ex.images is not None else source.images(row)
                if ex.images is None:
                    if rec_images is None:
                        rec_images = [writer.save_image(key if len(imgs) == 1 else "%s-%d" % (key, i), im, split)
                                      for i, im in enumerate(imgs)]
                    paths = rec_images
                else:
                    paths = [writer.save_image("%s-%d" % (key, i), im, split) for i, im in enumerate(imgs)]
                rec = {"id": "%s-%s-%d" % (key, ex.family, j), "state_text": ex.state_text, "question": ex.question,
                       "label": ex.label, "family": ex.family, "source": source.name}
                if len(paths) == 1:
                    rec["image"] = paths[0]
                else:
                    rec["images"] = paths
                if ex.target is not None:
                    rec["target"] = ex.target
                writer.write(split, rec)
            if idx % 2000 == 0 and idx:
                log("%s %s: row %d, train %d, val %d (%.0fs)" % (source.name, hf_split, idx, writer.records["train"],
                                                                writer.records["val"], time.time() - t0))
    meta = {"source": source.name, "repo": source.repo, "config": source.config, "license": source.license,
            "origin": source.origin, "hf_splits": {"train": source.train_split, "val": source.val_split or "hash %.2f of train" % val_frac},
            "seed": seed, "max_side": writer.max_side, "dropped": dict(dropped), "seconds": round(time.time() - t0, 1)}
    return writer.finish(meta)
