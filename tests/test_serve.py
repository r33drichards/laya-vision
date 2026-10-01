"""The Ollama-compatible /v1/systemone server (laya.serve)."""
import base64
import io
import json
import threading
import urllib.error
import urllib.request

import pytest
from PIL import Image

from laya.serve import SystemOneService, make_server, parse_request, to_response

# the request from Ollama's announcement post, verbatim
OLLAMA_EXAMPLE = {
    "model": "nimble",
    "state": {"ticket": "I was charged twice. Please refund the extra payment."},
    "questions": {
        "team": {
            "type": "choice",
            "instructions": "Which team should handle this ticket?",
            "criteria": {"billing": "Payments and refunds", "technical": "Bugs and integrations",
                         "other": "None of the above"},
        },
        "refund": {"type": "noul", "instructions": "Does the customer explicitly ask for a refund?"},
        "urgency": {"type": "score", "instructions": "How urgent is this ticket?",
                    "criteria": ["Routine", "Soon", "Urgent"]},
    },
}


def png_b64(color=(200, 10, 10)):
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), color).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def body(**over):
    req = json.loads(json.dumps(OLLAMA_EXAMPLE))
    req.update(over)
    return json.dumps(req).encode()


class FakeAgent:
    """Returns predict's result shape (extra fields included) and records what it was called with."""

    source = {"id": "thaitea/laya-vision", "revision": "abc"}

    def __init__(self, exc=None):
        self.calls, self.exc = [], exc

    def predict(self, state, questions, **kw):
        self.calls.append((state, questions, kw))
        if self.exc:
            raise self.exc
        answers = {}
        for qid, q in questions.items():
            ext = {"act_probability": 0.5}
            if q["type"] == "choice":
                names = list(q["criteria"])
                answers[qid] = {"type": "choice", "choice": names[0], "confidence": 0.9, "action": ext,
                                "probabilities": {n: 1.0 / len(names) for n in names}}
            elif q["type"] == "score":
                k = len(q["criteria"])
                answers[qid] = {"type": "score", "score": 1.0, "confidence": 0.1, "action": ext,
                                "legend": {str(i): c for i, c in enumerate(q["criteria"])},
                                "probabilities": {str(i): 1.0 / k for i in range(k)}}
            else:
                answers[qid] = {"type": "noul", "noul": 0.97, "confidence": 0.97, "action": ext}
        return {"model": "laya-vlm", "answers": answers, "provenance": {},
                "usage": {"input_tokens": 123, "output_tokens": 0, "images": 0}}


def service(agent=None, **kw):
    return SystemOneService(agent or FakeAgent(), names=["laya-vision", "nimble"], **kw)


def test_ollama_example_round_trips_in_jev_shape():
    agent = FakeAgent()
    status, out = service(agent).systemone(body())
    assert status == 200
    assert out["model"] == "nimble"
    assert out["usage"] == {"input_tokens": 123, "output_tokens": 0}
    assert set(out["answers"]["team"]) == {"type", "choice", "probabilities", "confidence"}
    assert out["answers"]["refund"] == {"type": "noul", "noul": 0.97}
    assert set(out["answers"]["urgency"]) == {"type", "score", "legend", "probabilities", "confidence"}
    state, questions, kw = agent.calls[0]
    assert state == OLLAMA_EXAMPLE["state"] and questions == OLLAMA_EXAMPLE["questions"]
    assert kw["strict"] is True  # over-budget is a 400, as an over-context prompt is in Ollama


def test_truncated_is_kept_when_allowed():
    res = FakeAgent().predict("x", {"q": {"type": "noul", "instructions": "y"}})
    res["answers"]["q"]["truncated"] = {"state_tokens_dropped": 3}
    assert to_response("m", res)["answers"]["q"]["truncated"] == {"state_tokens_dropped": 3}
    agent = FakeAgent()
    service(agent, allow_truncation=True).systemone(body())
    assert agent.calls[0][2]["strict"] is False


@pytest.mark.parametrize("over, status, needle", [
    ({"model": "llama3"}, 404, "not found"),
    ({"model": " "}, 400, "model"),
    ({"stream": False}, 400, "stream"),
    ({"state": ""}, 400, "state"),
    ({"state": 3}, 400, "state"),
    ({"state": {}}, 400, "state"),
    ({"questions": {}}, 400, "questions"),
    ({"questions": {str(i): {"type": "noul", "instructions": "x"} for i in range(65)}}, 400, "questions"),
    ({"questions": {"q": {"type": "rank", "instructions": "x"}}}, 400, "type"),
    ({"questions": {"q": {"type": "noul"}}}, 400, "instructions"),
    ({"questions": {"q": {"type": "noul", "instructions": "  "}}}, 400, "instructions"),
    ({"questions": {"q": {"type": "choice", "instructions": "x", "criteria": {"a": "only one"}}}}, 400, "2 to 26"),
    ({"questions": {"q": {"type": "choice", "instructions": "x",
                          "criteria": {chr(97 + i) * 2: None for i in range(27)}}}}, 400, "2 to 26"),
    ({"questions": {"q": {"type": "score", "instructions": "x", "criteria": {"a": "b", "c": "d"}}}}, 400, "array"),
    ({"questions": {"q": {"type": "noul", "instructions": "x", "criteria": {"maybe": "?"}}}}, 400, "true"),
    ({"questions": {"q": {"type": "noul", "instructions": "x", "temperature": 0}}}, 400, "temperature"),
    ({"keep_alive": True}, 400, "keep_alive"),
    ({"images": ["not base64!"]}, 400, "base64"),
    ({"images": ["/etc/passwd"]}, 400, "base64"),
    ({"state": {"image": 5}}, 400, "base64"),
])
def test_invalid_requests(over, status, needle):
    agent = FakeAgent()
    got, out = service(agent).systemone(body(**over))
    assert got == status and needle in out["error"]
    assert not agent.calls


def test_accepted_variants():
    ok = [
        {"keep_alive": "5m"}, {"keep_alive": 0}, {"keep_alive": None},
        {"state": "plain text"}, {"state": ["turn 1", "turn 2"]},
        {"questions": {"q": {"type": "choice", "instructions": {"ask": "json"}, "criteria": ["a", "b"]}}},
        {"questions": {"q": {"type": "choice", "instructions": "x", "criteria": {"a": None, "b": "desc"}}}},
        {"questions": {"q": {"type": "noul", "instructions": "x", "criteria": {"true": "yes it is"}}}},
    ]
    for over in ok:
        status, out = service().systemone(body(**over))
        assert status == 200, (over, out)


def test_images_are_decoded_to_bytes_never_paths():
    b64 = png_b64()
    _, state, _ = parse_request(body(state={"image": "data:image/png;base64," + b64, "note": "hi"},
                                     images=[b64]), ["nimble"])
    assert state["note"] == "hi" and "image" not in state
    assert len(state["images"]) == 2 and all(isinstance(i, bytes) for i in state["images"])
    _, state, _ = parse_request(body(state="what is this?", images=[b64]), ["nimble"])
    assert state == {"text": "what is this?", "images": [base64.b64decode(b64)]}
    _, state, _ = parse_request(body(state={"image": b64}), ["nimble"])  # an image alone is a state
    assert list(state) == ["images"]


def test_errors_from_predict():
    status, out = service(FakeAgent(ValueError("question 'q' would be truncated"))).systemone(body())
    assert status == 400 and "truncated" in out["error"]
    status, out = service(FakeAgent(RuntimeError("boom"))).systemone(body())
    assert status == 500 and "boom" in out["error"]


@pytest.fixture
def http_server():
    srv = make_server(service(max_body=4096), "127.0.0.1", 0)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield "http://127.0.0.1:%d" % srv.server_port
    srv.shutdown()
    srv.server_close()


def post(url, data):
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_over_http(http_server):
    status, out = post(http_server + "/v1/systemone", body())
    assert status == 200 and out["answers"]["team"]["choice"] == "billing"
    assert post(http_server + "/v1/systemone", body(model="nope"))[0] == 404
    status, out = post(http_server + "/v1/systemone", b"x" * 5000)
    assert status == 413 and "4096" in out["error"]
    assert post(http_server + "/v1/chat/completions", body())[0] == 404
    with urllib.request.urlopen(http_server + "/api/tags") as r:
        tags = json.loads(r.read())
    assert [m["name"] for m in tags["models"]] == ["laya-vision", "nimble"]
    assert tags["models"][0]["details"]["revision"] == "abc"
    with urllib.request.urlopen(http_server + "/") as r:
        assert r.read() == b"Laya is running"


def test_real_agent_end_to_end():
    """A fresh SmolVLM agent (untrained head) answers an image request in the Jev shape."""
    torch = pytest.importorskip("torch")
    from laya.vlm import VLMAgent

    torch.manual_seed(0)
    agent = VLMAgent(backbone="HuggingFaceTB/SmolVLM-256M-Instruct", device="cpu")
    status, out = SystemOneService(agent).systemone(body(
        model="laya-vision", state={"note": "a product photo"}, images=[png_b64()],
        questions={**OLLAMA_EXAMPLE["questions"],
                   "red": {"type": "noul", "instructions": "Is it red?", "criteria": {"true": "it is red"}}}))
    assert status == 200, out
    assert out["answers"]["team"]["choice"] in {"billing", "technical", "other"}
    assert abs(sum(out["answers"]["urgency"]["probabilities"].values()) - 1) < 1e-2
    assert 0 <= out["answers"]["red"]["noul"] <= 1 and out["usage"]["input_tokens"] > 0
