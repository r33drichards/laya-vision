"""TypeSafe System One's HTTP API (`POST /v1/systemone`) served by a Laya Vision checkpoint.

dimos's `TypeSafeAgent` (dimos/agents/typesafe) sends `{"state", "model", "questions"}` and reads
`{"answers", "model", "usage"}`. Its questions are Laya's own schema (`choice` / `noul`, `instructions`,
`criteria`), so this only adapts two things:

- an option's criterion may be a JSON object (`{"what", "not_for", "examples"}`); Laya takes text, so it is
  rendered as its `what` field (the option's meaning) when present, else as compact JSON;
- the state is the WorldState dict, serialized the way `predict` serializes any dict state.

Every call is appended to `--log` (one JSON line: latency, truncation per question, answers) so a run can
say how much of each WorldState the checkpoint actually saw.

    python benchmarks/dimos_nav/systemone_server.py --model thaitea/laya-vision --revision <sha> --port 8765 \
        [--max-len 4096 --head-max-len 1024 --option-max-len 256]   # the checkpoint's budgets (1024 / 256 / 48) cut
                                                                      # every WorldState and the long option texts
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import time


def criterion_text(v):
    if v is None or isinstance(v, str):
        return v
    if isinstance(v, dict) and isinstance(v.get("what"), str):
        return v["what"]
    return json.dumps(v, separators=(",", ":"))


def to_laya(questions: dict) -> dict:
    out = {}
    for qid, q in questions.items():
        crit = q.get("criteria")
        if isinstance(crit, dict):
            crit = {k: criterion_text(v) for k, v in crit.items()}
        out[qid] = {"type": q["type"], "instructions": q["instructions"], "criteria": crit}
    return out


def to_systemone(answers: dict) -> dict:
    out = {}
    for qid, a in answers.items():
        if a["type"] == "choice":
            out[qid] = {"type": "choice", "choice": a["choice"], "confidence": a["confidence"],
                        "probabilities": a["probabilities"]}
        elif a["type"] == "noul":
            out[qid] = {"type": "noul", "noul": a["noul"]}
        else:
            out[qid] = a
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="thaitea/laya-vision")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--log", default=None)
    ap.add_argument("--max-len", type=int, default=None, help="override the checkpoint's sequence budget")
    ap.add_argument("--head-max-len", type=int, default=None, help="override its question+options budget")
    ap.add_argument("--option-max-len", type=int, default=None, help="override the tokens each option keeps (48)")
    args = ap.parse_args()

    import laya

    overrides = {k: v for k, v in (("max_len", args.max_len), ("head_max_len", args.head_max_len),
                                   ("option_max_len", args.option_max_len)) if v}
    agent = laya.load_vlm(args.model, revision=args.revision, device=args.device, **overrides)
    lock = threading.Lock()  # one GPU, one predict at a time
    log = open(args.log, "a") if args.log else None
    name = "laya:%s@%s" % (args.model, args.revision or "main") + "".join(
        ",%s=%d" % kv for kv in sorted(overrides.items()))

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code: int, body: dict) -> None:
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self._send(200, {"ok": True, "model": name})

        def do_POST(self):
            if self.path.rstrip("/") != "/v1/systemone":
                return self._send(404, {"error": "not found"})
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                t0 = time.monotonic()
                with lock:
                    res = agent.predict(body["state"], to_laya(body["questions"]))
                dt = time.monotonic() - t0
                answers = to_systemone(res["answers"])
                if log is not None:
                    trunc = {q: a["truncated"] for q, a in res["answers"].items() if "truncated" in a}
                    log.write(json.dumps({"t": time.time(), "latency_s": dt, "truncated": trunc,
                                          "answers": answers}) + "\n")
                    log.flush()
                self._send(200, {"answers": answers, "model": name, "usage": {}})
            except Exception as e:  # the agent retries 5xx and logs the rest
                self._send(500, {"error": repr(e)})

    print("serving %s on :%d" % (name, args.port), flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
