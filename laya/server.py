"""Laya Vision HTTP API (FastAPI): ``VLMAgent.predict`` over the network.

    pip install "laya[serve]"
    laya-serve --port 8710                                   # serves thaitea/laya-vision (pinned) as "laya-vision"
    laya-serve --model prod=./runs/a --model next=./runs/b   # several checkpoints, each under its own name
    laya-serve --mock                                        # fake model (laya.mock), no weights, no torch
    export LAYA_API_KEY=...      # optional; then requests need "Authorization: Bearer <key>"

    POST /v1/systemone   {"state": ..., "questions": {id: question}, "model": "laya-vision",
                          "temperature": T | {"choice": T, ...}, "n_permutations": 1, "strict": false}
                         -> predict's result: {"model", "answers": {id: answer}, "usage", "provenance", "generation"}
    POST /v1/rank        {"candidates": [...], "instructions": ..., "state": ..., "model": ..., "chunk_size": 255}
                         -> {"model", "ranked": [{rank, candidate, prob, round}], "tournament", "usage"}
    GET  /v1/models      -> {"models": [{"name", "source", "generation", "loaded_at", "mock"}]}
    GET  /health         -> {"ok": true, "models": {name: generation}, "mock": bool}

The request's ``state`` and ``questions`` are exactly ``predict``'s (see site-docs/reference/predict.md), except
that an image in ``state["image"]`` / ``state["images"]`` travels as a string: base64, a ``data:image/...;base64,``
URL, or an ``http(s)://`` URL the server fetches (``--no-image-urls`` turns fetching off). A server never opens a
local path named in a request.

Every served checkpoint that is a local directory is watched: when its config or weights change (by mtime and
size, and unchanged for ``--reload-settle`` seconds), the next request loads it again beside the old one, swaps it in
and bumps that model's ``generation``, which every response carries. A reload that fails keeps the old model.
"""
import argparse
import asyncio
import base64
import binascii
import hmac
import os
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

DEFAULT_MODEL_NAME = "laya-vision"
DEFAULT_MODEL_ID = "thaitea/laya-vision"
# the checkpoint the Space pins (space/app.py); bump both together
DEFAULT_REVISION = "8b318c99d7ad3ce19c24369263463882eada9d1e"
QUESTION_TYPES = ("choice", "score", "noul")
MAX_OPTIONS = 255  # the act head's option-count feature is k / 255
MAX_PERMUTATIONS = 8
MAX_IMAGE_BYTES = 20 * 1024 * 1024
# the files a checkpoint directory is judged by (laya.vlm.CONFIG_NAME / WEIGHTS_NAME / HEAD_WEIGHTS_NAME)
WATCHED_FILES = ("vlm_agent_config.json", "model.safetensors", "head.safetensors")


class ModelNotFound(KeyError):
    pass


class RequestError(ValueError):
    """A malformed request: the server answers 422 with this message."""


class ImageFetchError(RuntimeError):
    """An image URL could not be fetched: the server answers 502."""


class ModelUnavailable(RuntimeError):
    """A model failed to load: the server answers 503."""


# ---------------------------------------------------------------------------------------------------------
# Request validation
# ---------------------------------------------------------------------------------------------------------


def validate_questions(questions: Any, max_options: int = MAX_OPTIONS) -> Dict[str, Dict[str, Any]]:
    """Check ``questions`` against predict's schema and return them; ``RequestError`` names the first problem."""
    if not isinstance(questions, dict) or not questions:
        raise RequestError("questions must be a non-empty object of {id: question}")
    for qid, q in questions.items():
        if not isinstance(q, dict):
            raise RequestError("question %r must be an object" % qid)
        t = q.get("type")
        if t not in QUESTION_TYPES:
            raise RequestError("question %r: type must be one of %s, got %r" % (qid, list(QUESTION_TYPES), t))
        if q.get("instructions") is None:
            raise RequestError("question %r: instructions are required" % qid)
        crit = q.get("criteria")
        if t == "choice":
            if not isinstance(crit, (list, dict)) or len(crit) < 2:
                raise RequestError("question %r: choice needs criteria, a list or object of at least 2 options" % qid)
            names = list(crit)
            if not all(isinstance(c, str) and c for c in names) or len(set(names)) != len(names):
                raise RequestError("question %r: choice options must be distinct non-empty strings" % qid)
            if isinstance(crit, dict) and not all(v is None or isinstance(v, str) for v in crit.values()):
                raise RequestError("question %r: choice option descriptions must be strings" % qid)
        elif t == "score":
            if not isinstance(crit, list) or len(crit) < 2 or not all(isinstance(c, str) for c in crit):
                raise RequestError("question %r: score needs criteria, a list of at least 2 level descriptions" % qid)
        elif crit is not None and (not isinstance(crit, dict) or set(crit) - {"true", "false"}):
            raise RequestError("question %r: noul criteria may only hold 'true' and 'false' descriptions" % qid)
        if t != "noul" and len(crit) > max_options:
            raise RequestError("question %r has %d options; at most %d per question (POST /v1/rank ranks more)"
                               % (qid, len(crit), max_options))
    return questions


def validate_temperature(value: Any) -> Any:
    """``None``, a positive number, or ``{type: positive number}``."""
    if value is None:
        return None
    items = value.items() if isinstance(value, dict) else [(None, value)]
    for t, v in items:
        if t is not None and t not in QUESTION_TYPES:
            raise RequestError("temperature keys must be question types %s, got %r" % (list(QUESTION_TYPES), t))
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 < v <= 100:
            raise RequestError("temperature must be a number in (0, 100] or {type: number}")
    return value


def decode_image(value: Any, allow_urls: bool = True, max_bytes: int = MAX_IMAGE_BYTES,
                 fetch: Optional[Callable[[str, int], bytes]] = None) -> bytes:
    """An image from a request, as encoded bytes (``predict`` decodes those itself).

    ``value`` is base64 (optionally a ``data:...;base64,`` URL) or an ``http(s)://`` URL. Never a local path.
    """
    if not isinstance(value, str) or not value:
        raise RequestError("an image must be a string: base64, a data: URL or an http(s):// URL")
    if value.startswith(("http://", "https://")):
        if not allow_urls:
            raise RequestError("image URLs are disabled on this server; send the image as base64")
        return (fetch or fetch_url)(value, max_bytes)
    if value.startswith("data:"):
        head, _, value = value.partition(",")
        if not head.endswith(";base64"):
            raise RequestError("a data: URL image must be base64 encoded")
    if len(value) * 3 // 4 > max_bytes:
        raise RequestError("image is larger than %d bytes" % max_bytes)
    try:
        return base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as e:
        raise RequestError("image is not valid base64 (nor a data: or http(s):// URL): %s" % e) from e


def fetch_url(url: str, max_bytes: int = MAX_IMAGE_BYTES, timeout: float = 15.0) -> bytes:
    import requests

    try:
        with requests.get(url, stream=True, timeout=timeout, headers={"User-Agent": "laya-serve"}) as r:
            if r.status_code != 200:
                raise ImageFetchError("fetching image %s: HTTP %d" % (url, r.status_code))
            data = bytearray()
            for chunk in r.iter_content(1 << 16):
                data += chunk
                if len(data) > max_bytes:
                    raise ImageFetchError("image at %s is larger than %d bytes" % (url, max_bytes))
            return bytes(data)
    except requests.RequestException as e:
        raise ImageFetchError("fetching image %s: %s" % (url, e)) from e


def decode_state(state: Any, allow_urls: bool = True, max_bytes: int = MAX_IMAGE_BYTES,
                 fetch: Optional[Callable[[str, int], bytes]] = None) -> Any:
    """The request's state with ``image`` / ``images`` decoded to bytes; everything else as sent."""
    if not isinstance(state, dict) or ("image" not in state and "images" not in state):
        return state
    out = dict(state)
    if out.get("image") is not None:
        out["image"] = decode_image(out["image"], allow_urls, max_bytes, fetch)
    if out.get("images") is not None:
        if not isinstance(out["images"], list):
            raise RequestError("state.images must be a list")
        out["images"] = [decode_image(v, allow_urls, max_bytes, fetch) for v in out["images"]]
    return out


# ---------------------------------------------------------------------------------------------------------
# Engine: named checkpoints, hot reload, rank tournaments
# ---------------------------------------------------------------------------------------------------------


def checkpoint_signature(path: Optional[str]) -> Optional[Tuple]:
    """(mtime_ns, size) of the files that make a checkpoint, or None for a Hub id (not watched)."""
    if not path or not os.path.exists(path):
        return None
    files = [os.path.join(path, f) for f in WATCHED_FILES] if os.path.isdir(path) else [path]
    sig = []
    for f in files:
        try:
            st = os.stat(f)
        except OSError:
            continue
        sig.append((os.path.basename(f), st.st_mtime_ns, st.st_size))
    return tuple(sig)


class ModelSlot:
    """One served name: its agent, the checkpoint it came from and how many times it was (re)loaded."""

    def __init__(self, name: str, path: Optional[str], loader: Callable[[Optional[str]], Any]):
        self.name, self.path, self.loader = name, path, loader
        self.agent = None
        self.generation = 0
        self.loaded_at = None
        self.signature = None
        self.last_error = None
        self.infer_lock = threading.Lock()  # predict keeps per-call state on the model (e.g. last_value)
        self.reload_lock = threading.Lock()
        self._next_check = 0.0

    def load(self):
        sig = checkpoint_signature(self.path)
        agent = self.loader(self.path)
        self.agent, self.signature = agent, sig
        self.generation += 1
        self.loaded_at = time.time()
        self.last_error = None
        return agent

    def maybe_reload(self, interval: float, settle: float) -> None:
        """Reload when the checkpoint changed and has been still for ``settle`` seconds; never blocks on a reload
        another request is doing (that request keeps serving the old model meanwhile)."""
        now = time.time()
        if self.signature is None or now < self._next_check:
            return
        self._next_check = now + interval
        sig = checkpoint_signature(self.path)
        if not sig or sig == self.signature:
            return
        if now - max(m for _, m, _ in sig) / 1e9 < settle:
            self._next_check = now  # still being written: look again next request
            return
        if not self.reload_lock.acquire(blocking=False):
            return
        try:
            try:
                agent = self.loader(self.path)
            except Exception as e:  # noqa: BLE001 - keep serving the old model
                self.signature, self.last_error = sig, "reload failed: %s: %s" % (type(e).__name__, e)
                print("[laya] %s: %s" % (self.name, self.last_error), flush=True)
                return
            with self.infer_lock:
                self.agent, self.signature = agent, sig
                self.generation += 1
                self.loaded_at = time.time()
                self.last_error = None
            print("[laya] %s reloaded from %s (generation %d)" % (self.name, self.path, self.generation), flush=True)
        finally:
            self.reload_lock.release()

    def info(self) -> Dict[str, Any]:
        out = {"name": self.name, "source": self.path, "generation": self.generation, "loaded_at": self.loaded_at,
               "watched": self.signature is not None, "mock": bool(getattr(self.agent, "mock", False))}
        if self.last_error:
            out["last_error"] = self.last_error
        return out


class Engine:
    """Named agents behind the API. ``models`` maps a name to a checkpoint path or Hub id; ``loader(path)`` builds
    the agent (``laya.load_vlm`` by default, ``MockAgent.load`` for the mock). The first model is the default."""

    def __init__(self, models: Dict[str, Optional[str]], loader: Callable[[Optional[str]], Any],
                 reload: bool = True, reload_interval: float = 2.0, reload_settle: float = 2.0,
                 max_options: int = MAX_OPTIONS):
        if not models:
            raise ValueError("no models to serve")
        self.slots = {name: ModelSlot(name, path, loader) for name, path in models.items()}
        self.default = next(iter(self.slots))
        self.reload, self.reload_interval, self.reload_settle = reload, reload_interval, reload_settle
        self.max_options = max_options
        for slot in self.slots.values():
            slot.load()
            if not reload:
                slot.signature = None

    @property
    def mock(self) -> bool:
        return any(getattr(s.agent, "mock", False) for s in self.slots.values())

    def models(self) -> List[Dict[str, Any]]:
        return [s.info() for s in self.slots.values()]

    def slot(self, name: Optional[str]) -> ModelSlot:
        name = name or self.default
        if name not in self.slots:
            raise ModelNotFound("unknown model %r; available: %s" % (name, list(self.slots)))
        slot = self.slots[name]
        if self.reload:
            slot.maybe_reload(self.reload_interval, self.reload_settle)
        if slot.agent is None:
            raise ModelUnavailable("model %r is not loaded: %s" % (name, slot.last_error))
        return slot

    def predict(self, state: Any, questions: Dict, model: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        slot = self.slot(model)
        with slot.infer_lock:
            out = slot.agent.predict(state, questions, **kwargs)
            generation = slot.generation
        return dict(out, model=slot.name, generation=generation)

    def rank(self, candidates: List[str], instructions: Any = None, state: Any = "", model: Optional[str] = None,
             chunk_size: Optional[int] = None, **kwargs) -> Dict[str, Any]:
        """Candidates best first, as one choice question whose options are the candidates verbatim.

        More than ``chunk_size`` candidates run as a tournament: the pool is split into near-equal chunks, every
        chunk is one question (all of a round's chunks go in one ``predict`` call), the top ``chunk_size //
        n_chunks`` of each chunk go through, and rounds repeat until one final question holds the rest. A
        candidate's ``prob`` comes from the last round it played (``round``; the final is the highest), so probs
        compare within a round; the order puts later rounds first.
        """
        cap = min(int(chunk_size or self.max_options), self.max_options)
        if cap < 2:
            raise RequestError("chunk_size must be at least 2")
        ins = instructions if instructions is not None else "Which option is the best?"
        slot = self.slot(model)
        placed: Dict[str, Tuple[int, float, float]] = {}  # candidate -> (round, prob, logit) of its last round
        pool, rnd, n_questions, usage = list(candidates), 0, 0, {"input_tokens": 0, "output_tokens": 0}
        truncated: List[str] = []  # the questions predict had to cut to fit the checkpoint's token budgets
        with slot.infer_lock:  # one generation for the whole tournament
            generation = slot.generation
            while True:
                n_chunks = -(-len(pool) // cap)
                bounds = [round(i * len(pool) / n_chunks) for i in range(n_chunks + 1)]
                chunks = [pool[bounds[i]:bounds[i + 1]] for i in range(n_chunks)]
                scored = self._ask_chunks(slot, state, ins, chunks, usage, truncated, rnd, **kwargs)
                n_questions += sum(1 for c in chunks if len(c) > 1)
                if n_chunks == 1:
                    for c, (p, z) in scored[0].items():
                        placed[c] = (rnd, p, z)
                    break
                keep, advanced = max(1, cap // n_chunks), []
                for sc in scored:
                    order = sorted(sc, key=lambda c: -sc[c][1])
                    advanced += order[:keep]
                    for c in order[keep:]:
                        placed[c] = (rnd, sc[c][0], sc[c][1])
                pool, rnd = advanced, rnd + 1
        order = sorted(candidates, key=lambda c: (-placed[c][0], -placed[c][1], -placed[c][2]))
        ranked = [{"rank": i + 1, "candidate": c, "prob": placed[c][1], "round": placed[c][0]}
                  for i, c in enumerate(order)]
        return {"model": slot.name, "generation": generation, "ranked": ranked,
                "tournament": {"rounds": rnd + 1, "chunk_size": cap, "questions": n_questions, "truncated": truncated},
                "usage": usage}

    @staticmethod
    def _ask_chunks(slot: ModelSlot, state, ins, chunks: List[List[str]], usage: Dict, truncated: List[str],
                    rnd: int, **kwargs) -> List[Dict]:
        """One predict call for a round: per chunk ``{candidate: (prob, logit)}``. Candidates are ordered by the
        unrounded logits (``_raw_logits``), since predict rounds probabilities to 4 places and a 255-option question
        has many ties there."""
        questions = {"chunk%d" % i: {"type": "choice", "instructions": ins, "criteria": list(c)}
                     for i, c in enumerate(chunks) if len(c) > 1}
        raw: Dict[str, Any] = {}
        out = slot.agent.predict(state, questions, _raw_logits=raw, **kwargs) if questions else {"answers": {}}
        for k in usage:
            usage[k] += int((out.get("usage") or {}).get(k) or 0)
        scored = []
        for i, c in enumerate(chunks):
            if len(c) == 1:
                scored.append({c[0]: (1.0, 0.0)})
                continue
            answer = out["answers"]["chunk%d" % i]
            if answer.get("truncated"):
                truncated.append("round %d chunk %d" % (rnd, i))
            probs = answer["probabilities"]
            scored.append({x: (float(probs[x]), float(z)) for x, z in zip(c, raw["chunk%d" % i])})
        return scored


# ---------------------------------------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------------------------------------


def create_app(engine: Engine, api_key: Optional[str] = None, cors: bool = False, allow_image_urls: bool = True,
               max_image_bytes: int = MAX_IMAGE_BYTES, fetch: Optional[Callable[[str, int], bytes]] = None):
    """The API over ``engine``. ``cors=True`` lets browsers call it from any origin; it is off by default because a
    browser would then send the API key header from any page."""
    from fastapi import FastAPI, Header, HTTPException, Request
    from fastapi.responses import JSONResponse

    app = FastAPI(title="Laya Vision API", version="0.1.0")
    app.state.engine = engine
    if cors:
        from fastapi.middleware.cors import CORSMiddleware

        app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST", "OPTIONS"],
                           allow_headers=["*"], allow_credentials=False, expose_headers=["X-Laya-Latency-Ms"])

    def auth(authorization: Optional[str]):
        if api_key and not hmac.compare_digest((authorization or "").encode(), ("Bearer %s" % api_key).encode()):
            raise HTTPException(401, "invalid or missing API key (Authorization: Bearer <key>)",
                                headers={"WWW-Authenticate": "Bearer"})

    async def body_of(request: Request) -> Dict[str, Any]:
        try:
            body = await request.json()
        except Exception as e:  # noqa: BLE001
            raise HTTPException(422, "body is not JSON: %s" % e) from e
        if not isinstance(body, dict):
            raise HTTPException(422, "body must be a JSON object")
        return body

    async def run(fn: Callable[[], Dict[str, Any]]) -> JSONResponse:
        t0 = time.perf_counter()
        try:
            out = await asyncio.get_running_loop().run_in_executor(None, fn)
        except (ModelNotFound, RequestError) as e:
            raise HTTPException(422, str(e.args[0])) from e
        except ImageFetchError as e:
            raise HTTPException(502, str(e)) from e
        except ModelUnavailable as e:
            raise HTTPException(503, str(e)) from e
        except (ValueError, KeyError, TypeError) as e:  # predict rejects what validation let through
            raise HTTPException(422, "invalid request: %s: %s" % (type(e).__name__, e)) from e
        return JSONResponse(out, headers={"X-Laya-Latency-Ms": "%.1f" % ((time.perf_counter() - t0) * 1000)})

    def options(body: Dict[str, Any]) -> Dict[str, Any]:
        kw = {}
        if body.get("temperature") is not None:
            kw["temperature"] = validate_temperature(body["temperature"])
        if body.get("n_permutations") is not None:
            n = body["n_permutations"]
            if isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= MAX_PERMUTATIONS:
                raise RequestError("n_permutations must be an integer in [1, %d]" % MAX_PERMUTATIONS)
            kw["n_permutations"] = n
        if body.get("strict") is not None:
            kw["strict"] = bool(body["strict"])
        return kw

    def model_name(body: Dict[str, Any]) -> Optional[str]:
        m = body.get("model")
        if m is not None and not isinstance(m, str):
            raise RequestError("model must be a string")
        return m

    @app.get("/health")
    def health():
        return {"ok": all(s.agent is not None for s in engine.slots.values()), "mock": engine.mock,
                "default_model": engine.default, "models": {s.name: s.generation for s in engine.slots.values()}}

    @app.get("/v1/models")
    def models(authorization: Optional[str] = Header(default=None)):
        auth(authorization)
        return {"default": engine.default, "models": engine.models()}

    @app.post("/v1/systemone")
    async def systemone(request: Request, authorization: Optional[str] = Header(default=None)):
        auth(authorization)
        body = await body_of(request)

        def go():
            if "state" not in body:
                raise RequestError("body must be {state, questions[, model, temperature, n_permutations, strict]}")
            questions = validate_questions(body.get("questions"), engine.max_options)
            kw, name = options(body), model_name(body)
            state = decode_state(body["state"], allow_image_urls, max_image_bytes, fetch)
            return engine.predict(state, questions, name, **kw)

        return await run(go)

    @app.post("/v1/rank")
    async def rank(request: Request, authorization: Optional[str] = Header(default=None)):
        auth(authorization)
        body = await body_of(request)

        def go():
            cands = body.get("candidates")
            if not isinstance(cands, list) or not cands:
                raise RequestError("body must be {candidates: [..][, instructions, state, model, chunk_size]}")
            if not all(isinstance(c, str) and c for c in cands) or len(set(cands)) != len(cands):
                raise RequestError("candidates must be distinct non-empty strings")
            chunk = body.get("chunk_size")
            if chunk is not None and (isinstance(chunk, bool) or not isinstance(chunk, int)
                                      or not 2 <= chunk <= engine.max_options):
                raise RequestError("chunk_size must be an integer in [2, %d]" % engine.max_options)
            kw, name = options(body), model_name(body)
            state = decode_state(body.get("state") or "", allow_image_urls, max_image_bytes, fetch)
            return engine.rank(cands, body.get("instructions"), state, name, chunk, **kw)

        return await run(go)

    return app


# ---------------------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------------------


def parse_model_spec(spec: str) -> Tuple[str, str, Optional[str]]:
    """``NAME=PATH`` or ``NAME=HUB_ID@REVISION`` -> (name, path, revision)."""
    name, sep, path = spec.partition("=")
    if not sep or not name or not path:
        raise SystemExit("--model expects NAME=PATH or NAME=HUB_ID@REVISION, got %r" % spec)
    revision = None
    if not os.path.exists(path) and "@" in path:
        path, _, revision = path.rpartition("@")
    return name, path, revision


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(prog="laya-serve", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=int(os.environ.get("LAYA_PORT", 8710)))
    ap.add_argument("--model", action="append", default=[], metavar="NAME=PATH",
                    help="serve a checkpoint (local directory, or Hub id with an optional @revision) as NAME; "
                         "repeat for several, the first is the default (default: %s=%s@%s)"
                         % (DEFAULT_MODEL_NAME, DEFAULT_MODEL_ID, DEFAULT_REVISION[:12]))
    ap.add_argument("--mock", action="store_true", help="serve laya.mock.MockAgent: fake numbers, no weights")
    ap.add_argument("--device", default=None, help="torch device (default: the best available)")
    ap.add_argument("--dtype", default=None, choices=["fp32", "bf16"], help="override the checkpoint's dtype")
    ap.add_argument("--no-reload", action="store_true", help="do not reload checkpoints whose files change")
    ap.add_argument("--reload-interval", type=float, default=2.0, help="seconds between checkpoint checks")
    ap.add_argument("--reload-settle", type=float, default=2.0,
                    help="a changed checkpoint must be this many seconds old before it is loaded")
    ap.add_argument("--max-options", type=int, default=MAX_OPTIONS,
                    help="options per question, and the default /v1/rank chunk size")
    ap.add_argument("--no-image-urls", action="store_true", help="refuse http(s) image URLs (base64 only)")
    ap.add_argument("--max-image-mb", type=float, default=MAX_IMAGE_BYTES / 2 ** 20)
    ap.add_argument("--cors", action="store_true", help="allow browser requests from any origin")
    args = ap.parse_args(argv)

    specs = [parse_model_spec(s) for s in args.model]
    if not specs:
        specs = [("mock", None, None)] if args.mock else [(DEFAULT_MODEL_NAME, DEFAULT_MODEL_ID, DEFAULT_REVISION)]
    if len({n for n, _, _ in specs}) != len(specs):
        raise SystemExit("--model names must be distinct")
    revisions = {p: r for _, p, r in specs}
    if args.mock:
        from .mock import MockAgent

        loader = MockAgent.load
    else:
        from .vlm import load_vlm

        def loader(path):
            return load_vlm(path, device=args.device, dtype=args.dtype, revision=revisions.get(path))

    engine = Engine({n: p for n, p, _ in specs}, loader, reload=not args.no_reload,
                    reload_interval=args.reload_interval, reload_settle=args.reload_settle,
                    max_options=args.max_options)
    api_key = os.environ.get("LAYA_API_KEY")
    app = create_app(engine, api_key, cors=args.cors, allow_image_urls=not args.no_image_urls,
                     max_image_bytes=int(args.max_image_mb * 2 ** 20))
    for m in engine.models():
        print("[laya] model %s from %s%s" % (m["name"], m["source"], " (watched)" if m["watched"] else ""), flush=True)
    if engine.mock:
        print("[laya] MOCK MODEL: the numbers are n-gram noise, not predictions", flush=True)
    print("[laya] auth %s; POST http://%s:%d/v1/systemone" % ("on" if api_key else "off", args.host, args.port),
          flush=True)
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
