"""An Ollama-compatible ``/v1/systemone`` decision server for Laya Vision checkpoints.

Ollama 0.35 serves decision models (``ollama pull nimble``) at ``POST /v1/systemone``, TypeSafe's Jev API: a
``state``, up to 64 named ``choice`` / ``noul`` / ``score`` questions, and typed answers with probabilities. Laya's
``predict(state, questions)`` already takes the same questions and returns the same answer schema, so this module
puts a checkpoint behind the same endpoint. A client written for Ollama (curl, or ``typesafe-sdk`` with
``TYPESAFE_BASE_URL``) talks to it by changing the base URL and the model name::

    python -m laya.serve --model thaitea/laya-vision --revision <commit>      # listens on 127.0.0.1:11435

Text requests are the same as Ollama's. Ollama's endpoint takes no images; this one also reads base64 images (plain
or as a ``data:`` URL) from ``state["image"]``, ``state["images"]`` or a top-level ``images`` list, the field
Ollama's ``/api/generate`` uses. Image strings are only ever decoded as base64, never opened as paths.

A question that does not fit the checkpoint's token budgets is a 400, as a prompt over the loaded context is in
Ollama, unless the server runs with ``--allow-truncation`` (answers then carry ``truncated``).
"""
import argparse
import base64
import binascii
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Iterable, Optional, Tuple

DEFAULT_PORT = 11435  # Ollama's 11434 plus one, so both can run on one machine
DEFAULT_NAME = "laya-vision"
DEFAULT_MAX_BODY = 16 * 1024 * 1024  # Ollama caps text requests at 64 KiB; images need more
MAX_QUESTIONS = 64
MIN_OPTIONS, MAX_OPTIONS = 2, 26
MAX_IMAGES = 16

REQUEST_KEYS = {"model", "state", "questions", "keep_alive", "images"}
QUESTION_KEYS = {"type", "instructions", "criteria"}
ANSWER_KEYS = {
    "choice": ("type", "choice", "probabilities", "confidence"),
    "noul": ("type", "noul"),
    "score": ("type", "score", "legend", "probabilities", "confidence"),
}


class RequestError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


def _content(value: Any, where: str) -> None:
    """Jev's SystemOneContent: a nonempty string, or an object or array (serialised as JSON text)."""
    if isinstance(value, str):
        if not value.strip():
            raise RequestError(400, "%s must not be empty" % where)
    elif not isinstance(value, (dict, list)):
        raise RequestError(400, "%s must be a string, an object or an array" % where)


def _decode_image(value: Any, where: str) -> bytes:
    if not isinstance(value, str):
        raise RequestError(400, "%s must be a base64 string" % where)
    data = value
    if data.startswith("data:"):
        head, sep, data = data.partition(",")
        if not sep or not head.endswith(";base64"):
            raise RequestError(400, "%s: only base64 data URLs are supported" % where)
    try:
        raw = base64.b64decode("".join(data.split()), validate=True)
    except (binascii.Error, ValueError):
        raise RequestError(400, "%s is not valid base64" % where) from None
    if not raw:
        raise RequestError(400, "%s is empty" % where)
    return raw


def _validate_question(qid: str, q: Any) -> Dict[str, Any]:
    where = "questions.%s" % qid
    if not isinstance(q, dict):
        raise RequestError(400, "%s must be an object" % where)
    unknown = set(q) - QUESTION_KEYS
    if unknown:
        raise RequestError(400, "%s: unknown field(s) %s" % (where, ", ".join(sorted(unknown))))
    t = q.get("type")
    if t not in ANSWER_KEYS:
        raise RequestError(400, "%s.type must be one of choice, noul, score" % where)
    if "instructions" not in q:
        raise RequestError(400, "%s.instructions is required" % where)
    _content(q["instructions"], where + ".instructions")
    crit = q.get("criteria")
    if t == "choice":
        if isinstance(crit, list):  # Laya's shorthand: option names only
            if not all(isinstance(c, str) and c for c in crit) or len(set(crit)) != len(crit):
                raise RequestError(400, "%s.criteria names must be distinct nonempty strings" % where)
        elif isinstance(crit, dict):
            if not all(k and (v is None or isinstance(v, str)) for k, v in crit.items()):
                raise RequestError(400, "%s.criteria must map option names to descriptions" % where)
        else:
            raise RequestError(400, "%s.criteria must be an object of option name -> description" % where)
    elif t == "score":
        if not isinstance(crit, list) or not all(isinstance(c, str) and c for c in crit):
            raise RequestError(400, "%s.criteria must be an array of level descriptions, lowest first" % where)
    elif crit is not None:  # noul
        if not isinstance(crit, dict) or set(crit) - {"true", "false"} or \
                not all(isinstance(v, str) for v in crit.values()):
            raise RequestError(400, "%s.criteria may only describe \"true\" and \"false\"" % where)
    if t != "noul" and not MIN_OPTIONS <= len(crit) <= MAX_OPTIONS:
        raise RequestError(400, "%s.criteria must have %d to %d entries" % (where, MIN_OPTIONS, MAX_OPTIONS))
    out = {"type": t, "instructions": q["instructions"]}
    if crit is not None:
        out["criteria"] = crit
    return out


def parse_request(body: bytes, model_names: Iterable[str]) -> Tuple[str, Any, Dict[str, Dict[str, Any]]]:
    """Validate a ``/v1/systemone`` body. Returns (model, state for ``predict``, questions); raises RequestError."""
    try:
        req = json.loads(body)
    except (UnicodeDecodeError, ValueError):
        raise RequestError(400, "request body must be JSON") from None
    if not isinstance(req, dict):
        raise RequestError(400, "request body must be a JSON object")
    unknown = set(req) - REQUEST_KEYS
    if unknown:
        raise RequestError(400, "unsupported field(s) %s (streaming, tools and generation controls are not supported)"
                           % ", ".join(sorted(unknown)))
    model = req.get("model")
    if not isinstance(model, str) or not model.strip():
        raise RequestError(400, "model is required")
    keep_alive = req.get("keep_alive")
    if keep_alive is not None and (isinstance(keep_alive, bool) or not isinstance(keep_alive, (str, int, float))):
        raise RequestError(400, "keep_alive must be a duration string or a number of seconds")
    names = set(model_names)
    if model not in names:
        raise RequestError(404, "model %r not found, try one of: %s" % (model, ", ".join(sorted(names))))

    questions = req.get("questions")
    if not isinstance(questions, dict) or not 1 <= len(questions) <= MAX_QUESTIONS:
        raise RequestError(400, "questions must be an object with 1 to %d entries" % MAX_QUESTIONS)
    parsed = {qid: _validate_question(qid, q) for qid, q in questions.items()}

    if "state" not in req:
        raise RequestError(400, "state is required")
    state = req["state"]
    images = []
    if isinstance(state, dict):
        state = dict(state)
        if state.get("image") is not None:
            images.append(_decode_image(state.pop("image"), "state.image"))
        else:
            state.pop("image", None)
        extra = state.pop("images", None)
        if extra is not None:
            if not isinstance(extra, list):
                raise RequestError(400, "state.images must be an array of base64 strings")
            images += [_decode_image(v, "state.images[%d]" % i) for i, v in enumerate(extra)]
    top = req.get("images")
    if top is not None:
        if not isinstance(top, list):
            raise RequestError(400, "images must be an array of base64 strings")
        images += [_decode_image(v, "images[%d]" % i) for i, v in enumerate(top)]
    if len(images) > MAX_IMAGES:
        raise RequestError(400, "at most %d images per request" % MAX_IMAGES)
    if isinstance(state, dict) and not state and not images:
        raise RequestError(400, "state must not be empty")
    if images:
        if isinstance(state, dict):
            state["images"] = images
        else:  # text or an array, read the way predict reads them, beside the images
            _content(state, "state")
            state = {"images": images, "text": state}
    else:
        _content(state, "state")
    return model, state, parsed


def to_response(model: str, result: Dict[str, Any]) -> Dict[str, Any]:
    """``predict``'s result in Jev's response shape: the documented answer fields and token usage."""
    answers = {}
    for qid, a in result["answers"].items():
        out = {k: a[k] for k in ANSWER_KEYS[a["type"]]}
        if "truncated" in a:
            out["truncated"] = a["truncated"]
        answers[qid] = out
    usage = result.get("usage", {})
    return {"model": model, "answers": answers,
            "usage": {"input_tokens": int(usage.get("input_tokens", 0)),
                      "output_tokens": int(usage.get("output_tokens", 0))}}


class SystemOneService:
    """One loaded agent behind ``/v1/systemone``. ``predict`` calls are serialised: the model is not thread-safe."""

    def __init__(self, agent, names: Iterable[str] = (DEFAULT_NAME,), allow_truncation: bool = False,
                 n_permutations: int = 1, max_body: int = DEFAULT_MAX_BODY):
        self.agent = agent
        self.names = list(dict.fromkeys(n for n in names if n))
        if not self.names:
            raise ValueError("at least one model name is needed")
        self.allow_truncation = allow_truncation
        self.n_permutations = n_permutations
        self.max_body = max_body
        self._lock = threading.Lock()

    def systemone(self, body: bytes) -> Tuple[int, Dict[str, Any]]:
        try:
            model, state, questions = parse_request(body, self.names)
        except RequestError as e:
            return e.status, {"error": str(e)}
        try:
            with self._lock:
                result = self.agent.predict(state, questions, n_permutations=self.n_permutations,
                                            strict=not self.allow_truncation)
        except ValueError as e:  # a question over the token budgets (strict), or an option the head cannot fit
            return 400, {"error": str(e)}
        except Exception as e:  # noqa: BLE001 - e.g. an image PIL cannot decode
            from PIL import UnidentifiedImageError

            if isinstance(e, UnidentifiedImageError):
                return 400, {"error": "cannot decode image: %s" % e}
            return 500, {"error": "scoring failed: %s: %s" % (type(e).__name__, e)}
        return 200, to_response(model, result)

    def tags(self) -> Dict[str, Any]:
        source = getattr(self.agent, "source", None) or {}
        details = {"family": "laya-vision", "format": "safetensors",
                   "checkpoint": source.get("id"), "revision": source.get("revision")}
        return {"models": [{"name": n, "model": n, "details": details} for n in self.names]}


def make_handler(service: SystemOneService):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "laya-serve"

        def _send(self, status: int, payload: Any, content_type: str = "application/json") -> None:
            data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(data)

        def do_GET(self):  # noqa: N802 - http.server's naming
            path = self.path.split("?", 1)[0]
            if path == "/":
                self._send(200, b"Laya is running", "text/plain; charset=utf-8")
            elif path == "/api/tags":
                self._send(200, service.tags())
            else:
                self._send(404, {"error": "not found"})

        do_HEAD = do_GET

        def do_POST(self):  # noqa: N802
            if self.path.split("?", 1)[0] != "/v1/systemone":
                self._send(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", ""))
            except ValueError:
                self.close_connection = True
                self._send(411, {"error": "Content-Length is required"})
                return
            if length > service.max_body:
                self.close_connection = True  # the unread body would otherwise be parsed as the next request
                self._send(413, {"error": "request body must not exceed %d bytes" % service.max_body})
                return
            status, payload = service.systemone(self.rfile.read(length))
            self._send(status, payload)

        def log_message(self, fmt, *args):
            sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    return Handler


def make_server(service: SystemOneService, host: str = "127.0.0.1", port: int = DEFAULT_PORT) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), make_handler(service))


def main(argv: Optional[list] = None) -> None:
    p = argparse.ArgumentParser(prog="laya-serve", description=__doc__.split("\n\n")[0])
    p.add_argument("--model", default="thaitea/laya-vision", help="Hub id or local checkpoint directory")
    p.add_argument("--revision", default=None, help="pin the checkpoint's Hub commit")
    p.add_argument("--backbone-revision", default=None, help="pin the backbone's Hub commit (head-only checkpoints)")
    p.add_argument("--name", action="append", default=None,
                   help="model name clients send (repeatable; default %r, and the --model id is always accepted)"
                   % DEFAULT_NAME)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--device", default=None)
    p.add_argument("--n-permutations", type=int, default=1, help="option orders averaged per question")
    p.add_argument("--allow-truncation", action="store_true",
                   help="cut inputs over the token budgets and report it in 'truncated', instead of a 400")
    p.add_argument("--max-body-bytes", type=int, default=DEFAULT_MAX_BODY)
    args = p.parse_args(argv)

    from .vlm import load_vlm

    agent = load_vlm(args.model, device=args.device, revision=args.revision, backbone_revision=args.backbone_revision)
    service = SystemOneService(agent, names=(args.name or [DEFAULT_NAME]) + [args.model],
                               allow_truncation=args.allow_truncation, n_permutations=args.n_permutations,
                               max_body=args.max_body_bytes)
    server = make_server(service, args.host, args.port)
    print("serving %s (revision %s) as %s on http://%s:%d/v1/systemone"
          % (args.model, agent.source.get("revision"), ", ".join(service.names), args.host, server.server_port),
          file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
