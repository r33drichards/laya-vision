"""Typed readout versus generated JSON: the same questions about the same image, answered two ways.

Path A (``typed``) is ``VLMAgent.predict`` on the published checkpoint: every question is one row, each option is
read from one hidden state, and the answers come back as probabilities. Nothing is generated.

Path B (``compact_array``) is the frozen backbone the checkpoint was fine-tuned from, SmolVLM-256M-Instruct at a
pinned revision, shown the same image and state text and asked for ONE compact JSON array with one value per
question in order: an option id for ``choice``, ``true``/``false`` for ``noul``, a level number for ``score``. No
keys, confidences or explanations. Greedy decoding. Path B' (``per_question``) asks the same backbone one question
per ``generate`` call, the way a caller without batching would.

What this measures is a systems comparison, not a quality one: the two paths do not run the same weights (path A's
backbone was fine-tuned with its head), so the generated answers are not what the trained head would say. The
agreement column says how often the base model's generated value matches path A's argmax, nothing more.

Both paths start from a decoded PIL image and end with Python values, so both clocks include the processor's CPU
preprocessing. The image is shown to both paths as the one 512-pixel view the checkpoint uses (64 image tokens);
``compact_array_shipped`` repeats path B with the backbone's shipped processor settings (image splitting on, the
image upscaled to 2048 and cut into 16 tiles plus a global view), which is how the base model is normally run.
Timings are CUDA-synchronized wall clock, median over ``--repeats`` runs after warm-up runs of every path.

    python benchmarks/decision_vs_generation.py --write-fixture          # regenerate the owned test image
    python benchmarks/decision_vs_generation.py --output results/raw/decision-vs-generation-l4.json
    modal run modal_app.py::decision_vs_generation                        # the same on an L4, writes results/raw/

The fixture (``benchmarks/data/dvg_card.png``) is drawn by ``draw_fixture`` below, so it is owned by this repo and
Apache 2.0 like the code; the questions and the state text are in ``benchmarks/data/dvg_questions.json``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIXTURE_IMAGE = HERE / "data" / "dvg_card.png"
FIXTURE_QUESTIONS = HERE / "data" / "dvg_questions.json"

TYPED_MODEL = "thaitea/laya-vision"
TYPED_REVISION = "d1fbdc0612fbe3b3d8ec6f54d328b195d35bb338"
BASE_MODEL = "HuggingFaceTB/SmolVLM-256M-Instruct"
BASE_REVISION = "7e3e67edbbed1bf9888184d9df282b700a323964"
VERSION = "laya-decision-vs-generation-v1"

INSTRUCTION = (
    "Answer each question about the image independently. Return exactly one JSON array with one value per "
    "question, in the order given. For a choice question the value is one of its option ids as a string, for a "
    "true/false question it is true or false, for a level question it is the level number. Emit no keys, "
    "confidence values, markdown or explanation."
)
ONE_INSTRUCTION = (
    "Answer the question about the image. Return exactly one JSON value: for a choice question one of its option "
    "ids as a string, for a true/false question true or false, for a level question the level number. Emit no "
    "markdown or explanation."
)


# ---------------------------------------------------------------------------------------------------------
# Fixture, prompt, parsing and agreement (pure Python; unit-tested on CPU in tests/test_decision_vs_generation.py)
# ---------------------------------------------------------------------------------------------------------


def draw_fixture(path: Path = FIXTURE_IMAGE) -> Path:
    """A 512x512 test card: a red circle top left, a larger blue square top right, a green triangle bottom left
    and a yellow sign reading EXIT bottom right, on a light gray background. Deterministic; drawn here so the
    image is owned by the repo."""
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (512, 512), (225, 225, 225))
    d = ImageDraw.Draw(img)
    d.ellipse((40, 40, 190, 190), fill=(210, 30, 30))
    d.rectangle((270, 30, 480, 240), fill=(30, 60, 200))
    d.polygon([(40, 470), (200, 470), (120, 300)], fill=(30, 160, 60))
    d.rectangle((280, 330, 480, 450), fill=(245, 205, 40), outline=(20, 20, 20), width=4)
    try:
        font = ImageFont.load_default(size=64)
    except TypeError:  # Pillow < 10.1 has only the fixed bitmap font
        font = ImageFont.load_default()
    d.text((380, 390), "EXIT", fill=(20, 20, 20), font=font, anchor="mm")
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, format="PNG", optimize=False)
    return path


def load_fixture(path: Path = FIXTURE_QUESTIONS) -> dict:
    """``{"state": {...text keys...}, "questions": {qid: predict-schema question}}``."""
    with open(path) as f:
        return json.load(f)


def option_ids(q: dict) -> list:
    """Allowed answer values in label order: option ids, ``[False, True]``, or level numbers."""
    if q["type"] == "choice":
        return list(q["criteria"])
    if q["type"] == "score":
        return list(range(len(q["criteria"])))
    return [False, True]


def request_items(questions: dict) -> list:
    """The questions as the generation prompt lists them: text plus what the answer may be."""
    items = []
    for q in questions.values():
        if q["type"] == "choice":
            crit = q["criteria"]
            options = crit if isinstance(crit, list) else ["%s: %s" % kv if kv[1] else kv[0] for kv in crit.items()]
            items.append({"question": q["instructions"], "kind": "choice", "option_ids": options})
        elif q["type"] == "score":
            items.append({"question": q["instructions"], "kind": "level",
                          "levels": {str(i): c for i, c in enumerate(q["criteria"])}})
        else:
            items.append({"question": q["instructions"], "kind": "true/false"})
    return items


def compact_prompt(state_text: str, questions: dict) -> str:
    """The user turn for path B: instruction, then the state and the ordered questions as one JSON object."""
    request = {"state": state_text, "questions_in_output_order": request_items(questions)}
    return INSTRUCTION + "\n" + json.dumps(request, ensure_ascii=False)


def one_prompt(state_text: str, q: dict) -> str:
    """The user turn for path B': one question."""
    request = {"state": state_text, "question": request_items({"q": q})[0]}
    return ONE_INSTRUCTION + "\n" + json.dumps(request, ensure_ascii=False)


_TRUE, _FALSE = ("true", "yes"), ("false", "no")


def normalize(q: dict, value, strict: bool = True):
    """``value`` as the canonical answer for ``q`` (option id str, bool, level int), or None if it is not one.

    Strict takes only the requested JSON type: an exact option id string, a JSON boolean, a JSON integer level.
    Lenient also takes the obvious near misses a small model makes: option ids in another case or with the
    ``id: text`` suffix, ``"true"``/``"yes"``/``"no"`` strings, ``1``/``0`` for true/false, ``"2"`` or
    ``"level 2"`` for a level."""
    allowed = option_ids(q)
    if q["type"] == "choice":
        if isinstance(value, str) and value in allowed:
            return value
        if strict or not isinstance(value, str):
            return None
        v = value.strip().lower()
        for opt in allowed:
            if v == opt.lower() or v.startswith(opt.lower() + ":"):
                return opt
        return None
    if q["type"] == "noul":
        if isinstance(value, bool):
            return value
        if strict:
            return None
        if isinstance(value, int) and value in (0, 1):
            return bool(value)
        if isinstance(value, str):
            v = value.strip().lower()
            return True if v in _TRUE else False if v in _FALSE else None
        return None
    if isinstance(value, int) and not isinstance(value, bool) and value in allowed:
        return value
    if strict:
        return None
    if isinstance(value, float) and value.is_integer() and int(value) in allowed:
        return int(value)
    if isinstance(value, str):
        m = re.fullmatch(r"\s*(?:level\s*)?(\d+)\s*(?::.*)?", value.lower())
        if m and int(m.group(1)) in allowed:
            return int(m.group(1))
    return None


def _first_json(text: str, opener: str):
    """The first JSON value in ``text`` that starts at ``opener`` (``[`` for an array, any for a scalar), or
    raise ``ValueError``. Skips markdown fences and prose around it."""
    dec = json.JSONDecoder()
    starts = [i for i, ch in enumerate(text) if ch == opener] if opener else [0]
    for i in starts:
        try:
            return dec.raw_decode(text[i:].lstrip())[0]
        except json.JSONDecodeError:
            continue
    raise ValueError("no JSON value found")


def parse_array(text: str, questions: dict) -> dict:
    """Parse a generated compact array.

    ``strict_valid``: the whole output (whitespace stripped) is one JSON array of the right length and every value
    has exactly the requested type, the standard SemIf's benchmark holds its baseline to. ``lenient``: the first
    JSON array anywhere in the text, each position normalized leniently (None where it is missing or unusable)."""
    qs = list(questions.values())
    out = {"strict_valid": False, "strict": None, "json_found": False, "length": None, "lenient": [None] * len(qs)}
    try:
        whole = json.loads(text.strip())
    except json.JSONDecodeError:
        whole = None
    if isinstance(whole, list) and len(whole) == len(qs):
        vals = [normalize(q, v, strict=True) for q, v in zip(qs, whole)]
        if all(v is not None for v in vals):
            out.update(strict_valid=True, strict=vals)
    try:
        arr = whole if isinstance(whole, list) else _first_json(text, "[")
    except ValueError:
        return out
    if not isinstance(arr, list):
        return out
    out.update(json_found=True, length=len(arr))
    out["lenient"] = [normalize(q, arr[i], strict=False) if i < len(arr) else None for i, q in enumerate(qs)]
    return out


def parse_one(text: str, q: dict) -> dict:
    """Parse a one-question answer: strict takes the whole output as one JSON value of the requested type; lenient
    also takes the first JSON scalar or bare word in the text."""
    t = text.strip()
    try:
        whole = json.loads(t)
    except json.JSONDecodeError:
        whole = None
    strict = normalize(q, whole, strict=True) if whole is not None else None
    lenient = normalize(q, whole, strict=False) if whole is not None else None
    if lenient is None:
        for tok in re.findall(r'"[^"]*"|[\w.:-]+', t):
            tok = tok.rstrip(".,;:!?") or tok
            try:
                v = json.loads(tok)
            except json.JSONDecodeError:
                v = tok
            lenient = normalize(q, v, strict=False)
            if lenient is not None:
                break
    return {"strict_valid": strict is not None, "strict": strict, "lenient": lenient}


def typed_argmax(answer: dict, q: dict):
    """Path A's top answer in the same canonical form: the top option id, P(true) > 0.5, the most likely level."""
    if q["type"] == "choice":
        return answer["choice"]
    if q["type"] == "noul":
        return answer["noul"] > 0.5
    probs = answer["probabilities"]
    return int(max(probs, key=lambda k: probs[k]))


def agreement(reference: list, generated: list) -> dict:
    """Per-question match between path A's argmax and a generated answer; a missing answer is a mismatch."""
    per = [g is not None and g == r for r, g in zip(reference, generated)]
    return {"per_question": per, "matches": sum(per), "n": len(per), "rate": sum(per) / len(per) if per else None}


# ---------------------------------------------------------------------------------------------------------
# Timed paths
# ---------------------------------------------------------------------------------------------------------


class TimelineStreamer:
    """Records the wall time of every generated token (``generate`` first sends the prompt ids, skipped)."""

    def __init__(self, tokenizer, started: float, sync):
        self.tokenizer, self.started, self.sync = tokenizer, started, sync
        self.initial = True
        self.events = []

    def put(self, value):
        if self.initial:
            self.initial = False
            return
        self.sync()
        elapsed = time.perf_counter() - self.started
        for token_id in value.detach().cpu().reshape(-1).tolist():
            self.events.append({"seconds": elapsed, "token_id": token_id,
                                "text": self.tokenizer.decode([token_id], skip_special_tokens=False)})

    def end(self):
        pass


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def code_revision() -> dict:
    """The git sha of the code being measured, from ``LAYA_GIT_SHA`` (set by the Modal job) or ``git``."""
    sha, dirty = os.environ.get("LAYA_GIT_SHA"), os.environ.get("LAYA_GIT_DIRTY")
    if not sha:
        try:
            root = str(HERE.parent)
            sha = subprocess.check_output(["git", "-C", root, "rev-parse", "HEAD"], text=True).strip()
            dirty = str(bool(subprocess.check_output(["git", "-C", root, "status", "--porcelain"], text=True).strip()))
        except (OSError, subprocess.CalledProcessError):
            sha = None
    return {"git_sha": sha, "dirty": None if dirty is None else dirty.lower() == "true"}


def run(output: Path, repeats: int = 5, warmup: int = 2, max_new_tokens: int = 256, device: str = "cuda",
        dtype: str = "bf16", typed_path: str = "", per_question: bool = True, shipped: bool = True) -> dict:
    """Run every path and write the report to ``output`` (which must not exist). ``typed_path`` is a local
    checkpoint directory; empty means ``TYPED_MODEL`` at ``TYPED_REVISION`` from the Hub."""
    import torch
    import transformers
    from huggingface_hub import snapshot_download
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    from laya.common import serialize_state
    from laya.vlm import VLMAgent

    if output.exists():
        raise SystemExit("%s exists; results are never overwritten" % output)
    dev = torch.device(device)
    cuda = dev.type == "cuda"
    sync = torch.cuda.synchronize if cuda else (lambda: None)
    torch_dtype = torch.bfloat16 if dtype == "bf16" else torch.float32

    fixture = load_fixture()
    questions = fixture["questions"]
    qs = list(questions.values())
    with Image.open(FIXTURE_IMAGE) as im:
        image = im.convert("RGB")
    state = dict(fixture["state"], image=image)
    state_text = serialize_state(fixture["state"])

    typed_source = typed_path or snapshot_download(TYPED_MODEL, revision=TYPED_REVISION)
    agent = VLMAgent(typed_source, device=device, dtype=dtype)
    processor = AutoProcessor.from_pretrained(BASE_MODEL, revision=BASE_REVISION)
    base = AutoModelForImageTextToText.from_pretrained(BASE_MODEL, revision=BASE_REVISION, dtype=torch_dtype)
    base.to(dev).eval()
    tok = processor.tokenizer

    def timed(fn):
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        sync()
        t0 = time.perf_counter()
        out = fn()
        sync()
        total = time.perf_counter() - t0
        return out, {"total_seconds": total, "peak_cuda_bytes": torch.cuda.max_memory_allocated() if cuda else None}

    def typed(batch_size):
        return agent.predict(state, questions, batch_size=batch_size)

    def generate(text, split, budget):
        started = time.perf_counter()
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=prompt, images=[image], do_image_splitting=split, return_tensors="pt").to(dev)
        inputs["pixel_values"] = inputs["pixel_values"].to(torch_dtype)
        n_in = int(inputs["input_ids"].shape[-1])
        streamer = TimelineStreamer(tok, started, sync)
        with torch.inference_mode():
            out = base.generate(**inputs, do_sample=False, max_new_tokens=budget, streamer=streamer, use_cache=True)
        sync()
        total = time.perf_counter() - started
        gen = out[0, n_in:].detach().cpu().tolist()
        return {"prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "input_tokens": n_in,
                "image_views": int(inputs["pixel_values"].shape[1]), "output_tokens": len(gen),
                "hit_token_budget": len(gen) >= budget,
                "time_to_first_token_seconds": streamer.events[0]["seconds"] if streamer.events else None,
                "total_seconds": total, "output_text": tok.decode(gen, skip_special_tokens=True),
                "timeline": streamer.events}

    def compact(split):
        r = generate(compact_prompt(state_text, questions), split, max_new_tokens)
        r["parsed"] = parse_array(r["output_text"], questions)
        return r

    def one_by_one():
        calls = []
        for q in qs:
            r = generate(one_prompt(state_text, q), False, 16)
            r["parsed"] = parse_one(r["output_text"], q)
            calls.append(r)
        return calls

    n = len(qs)
    paths = {"typed_default_batch": lambda: typed(8), "typed_one_batch": lambda: typed(n),
             "compact_array": lambda: compact(False)}
    if shipped:
        paths["compact_array_shipped"] = lambda: compact(True)
    if per_question:
        paths["per_question"] = one_by_one
    for _ in range(warmup):  # kernels, allocator, the processor's caches; excluded from every duration
        for fn in paths.values():
            fn()
    runs = {name: [] for name in paths}
    for _ in range(repeats):  # interleaved so drift in clocks or thermals hits every path alike
        for name, fn in paths.items():
            out, timing = timed(fn)
            runs[name].append((out, timing))

    reference_out = runs["typed_default_batch"][-1][0]
    reference = [typed_argmax(reference_out["answers"][qid], q) for qid, q in questions.items()]
    report = {
        "version": VERSION,
        "code": code_revision(),
        "hardware": {"gpu": torch.cuda.get_device_name(0) if cuda else None, "device": device,
                     "cpu": platform.processor() or platform.machine()},
        "software": {"torch": str(torch.__version__), "transformers": str(transformers.__version__),
                     "python": platform.python_version(), "cuda": str(torch.version.cuda)},
        "dtype": dtype,
        "models": {"typed": {"source": typed_path or TYPED_MODEL, "revision": None if typed_path else TYPED_REVISION,
                             "option_attention": agent.cfg.get("option_attention"),
                             "temperature": agent.temperature},
                   "generation": {"source": BASE_MODEL, "revision": BASE_REVISION}},
        "input": {"image": FIXTURE_IMAGE.name, "image_sha256": _sha(FIXTURE_IMAGE), "image_size": list(image.size),
                  "questions": FIXTURE_QUESTIONS.name, "questions_sha256": _sha(FIXTURE_QUESTIONS), "n_questions": n,
                  "types": {t: sum(q["type"] == t for q in qs) for t in ("choice", "noul", "score")},
                  "state_text": state_text},
        "protocol": {"repeats": repeats, "warmup_rounds": warmup, "max_new_tokens": max_new_tokens,
                     "per_question_max_new_tokens": 16, "decoding": "greedy",
                     "timer": "perf_counter around the call with torch.cuda.synchronize before and after; runs "
                              "interleaved across paths; includes CPU preprocessing from a decoded PIL image"},
        "scope": ("Systems comparison. Path A is the fine-tuned checkpoint's typed readout; path B is its frozen base "
                  "backbone generating a compact JSON array from the same image, state text and questions. They do "
                  "not share weights, so agreement is agreement with path A's argmax, not a correctness measure, "
                  "and nothing here says the generated answers are what the trained head would produce."),
        "reference_argmax": reference,
        "paths": {},
    }
    for name, rs in runs.items():
        entry = {"median_total_seconds": statistics.median(t["total_seconds"] for _, t in rs)}
        if name.startswith("typed"):
            entry["runs"] = [dict(t, input_tokens=o["usage"]["input_tokens"]) for o, t in rs]
            entry["answers"] = rs[-1][0]["answers"]
            entry["argmax"] = [typed_argmax(rs[-1][0]["answers"][qid], q) for qid, q in questions.items()]
            entry["output_tokens"] = 0
        elif name == "per_question":
            entry["runs"] = [dict(t, output_tokens=sum(c["output_tokens"] for c in o),
                                  time_to_first_token_seconds=o[0]["time_to_first_token_seconds"],
                                  calls=[{k: c[k] for k in c if k != "timeline"} for c in o]) for o, t in rs]
            entry["runs"][0]["calls"] = [dict(c) for c in rs[0][0]]  # keep one full token timeline
            entry["median_output_tokens"] = statistics.median(r["output_tokens"] for r in entry["runs"])
            last = rs[-1][0]
            entry["all_strict_valid"] = all(c["parsed"]["strict_valid"] for c in last)
            entry["strict_valid_count"] = sum(c["parsed"]["strict_valid"] for c in last)
            entry["agreement_lenient"] = agreement(reference, [c["parsed"]["lenient"] for c in last])
            entry["outputs_identical_across_runs"] = len({json.dumps([c["output_text"] for c in o]) for o, _ in rs}) == 1
        else:
            entry["runs"] = [dict({k: o[k] for k in o if k != "timeline" or i == 0}, **t) for i, (o, t) in enumerate(rs)]
            entry["prompt_text"] = compact_prompt(state_text, questions)
            entry["median_output_tokens"] = statistics.median(o["output_tokens"] for o, _ in rs)
            entry["median_time_to_first_token_seconds"] = statistics.median(
                o["time_to_first_token_seconds"] for o, _ in rs if o["time_to_first_token_seconds"] is not None)
            parsed = rs[-1][0]["parsed"]
            entry["all_runs_strict_valid"] = all(o["parsed"]["strict_valid"] for o, _ in rs)
            entry["agreement_strict"] = agreement(reference, parsed["strict"] or [None] * n)
            entry["agreement_lenient"] = agreement(reference, parsed["lenient"])
            entry["outputs_identical_across_runs"] = len({o["output_text"] for o, _ in rs}) == 1
        report["paths"][name] = entry
    a = report["paths"]["typed_default_batch"]["median_total_seconds"]
    report["median_wall_ratio_vs_typed_default_batch"] = {k: v["median_total_seconds"] / a for k, v in report["paths"].items()}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False, default=_json_default) + "\n")
    print(json.dumps(summary(report), indent=2))
    return report


def _json_default(v):
    try:
        import numpy as np

        if isinstance(v, np.generic):
            return v.item()
    except ImportError:
        pass
    raise TypeError("not JSON serializable: %r" % type(v).__name__)


def summary(report: dict) -> dict:
    """The headline numbers: per path, median seconds, output tokens, validity and agreement."""
    out = {"code": report["code"], "gpu": report["hardware"]["gpu"]}
    for name, p in report["paths"].items():
        row = {"median_s": round(p["median_total_seconds"], 4),
               "ratio": round(report["median_wall_ratio_vs_typed_default_batch"][name], 2)}
        if "median_output_tokens" in p:
            row["output_tokens"] = p["median_output_tokens"]
        if "median_time_to_first_token_seconds" in p:
            row["ttft_s"] = round(p["median_time_to_first_token_seconds"], 4)
        if "all_runs_strict_valid" in p:
            row["strict_valid"] = p["all_runs_strict_valid"]
            row["agree_strict"] = "%d/%d" % (p["agreement_strict"]["matches"], p["agreement_strict"]["n"])
        if "strict_valid_count" in p:
            row["strict_valid"] = "%d/%d" % (p["strict_valid_count"], len(report["reference_argmax"]))
        if "agreement_lenient" in p:
            row["agree_lenient"] = "%d/%d" % (p["agreement_lenient"]["matches"], p["agreement_lenient"]["n"])
        out[name] = row
    return out


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--write-fixture", action="store_true", help="redraw benchmarks/data/dvg_card.png and exit")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bf16", choices=("bf16", "fp32"))
    parser.add_argument("--typed-path", default="", help="local checkpoint dir instead of the pinned Hub revision")
    parser.add_argument("--no-per-question", action="store_true")
    parser.add_argument("--no-shipped", action="store_true")
    args = parser.parse_args(argv)
    if args.write_fixture:
        print(draw_fixture(), _sha(FIXTURE_IMAGE))
        return
    if args.output is None or args.repeats < 1 or args.warmup < 0 or args.max_new_tokens < 1:
        parser.error("--output is required and the numeric limits must be positive")
    run(args.output, args.repeats, args.warmup, args.max_new_tokens, args.device, args.dtype, args.typed_path,
        per_question=not args.no_per_question, shipped=not args.no_shipped)


if __name__ == "__main__":
    sys.path.insert(0, str(HERE.parent))
    main()
