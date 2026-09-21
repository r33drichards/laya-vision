"""Tests for the HTTP API (``laya.serve``) against a stub agent: no model download, no GPU."""
import io
import json

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from laya.serve import MAX_IMAGES, PredictResponse, decode_image, make_app

QUESTIONS = {
    "color": {"type": "choice", "instructions": "What color is the square?", "criteria": ["red", "blue"]},
    "size": {"type": "score", "instructions": "How big?", "criteria": ["tiny", "huge"]},
    "is_red": {"type": "noul", "instructions": "Is it red?"},
}


class StubAgent:
    """Records every call and answers in the ``VLMAgent.predict`` schema."""

    def __init__(self):
        self.calls = []

    def predict(self, state, questions, n_permutations=1, batch_size=8):
        imgs = state.get("images") or [state["image"]]
        self.calls.append({"n_images": len(imgs), "sizes": [im.size for im in imgs], "text": state.get("text"),
                           "questions": questions, "n_permutations": n_permutations})
        act = {"act_probability": 0.5}
        answers = {}
        for qid, q in questions.items():
            if q["type"] == "choice":
                keys = list(q["criteria"])
                answers[qid] = {"type": "choice", "choice": keys[0], "probabilities": {k: 1.0 / len(keys) for k in keys},
                                "confidence": 0.5, "action": act}
            elif q["type"] == "score":
                answers[qid] = {"type": "score", "score": 0.5, "legend": {str(i): c for i, c in enumerate(q["criteria"])},
                                "probabilities": {"0": 0.5, "1": 0.5}, "confidence": 0.5, "action": act}
            else:
                answers[qid] = {"type": "noul", "noul": 0.7, "confidence": 0.7, "action": act}
        return {"model": "stub", "answers": answers, "usage": {"input_tokens": 10, "output_tokens": 0, "images": len(imgs)}}


def png(color=(220, 20, 20), size=(64, 64)):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def jpeg(size):
    buf = io.BytesIO()
    Image.new("RGB", size, (10, 200, 30)).save(buf, format="JPEG", quality=80)
    return buf.getvalue()


@pytest.fixture
def client():
    agent = StubAgent()
    return TestClient(make_app(agent, run_name="stub/best")), agent


def files(*blobs, names=None):
    return [("images", (names[i] if names else "img%d.png" % i, b, "image/png")) for i, b in enumerate(blobs)]


def test_predict_one_image_per_result(client):
    c, agent = client
    r = c.post("/predict", files=files(png(), png((0, 0, 255))), data={"questions": json.dumps(QUESTIONS), "text": "hello"})
    assert r.status_code == 200, r.text
    out = PredictResponse.model_validate(r.json())
    assert out.run == "stub/best"
    assert len(out.results) == 2
    assert [c_["n_images"] for c_ in agent.calls] == [1, 1]
    assert agent.calls[0]["text"] == "hello"
    assert agent.calls[0]["questions"]["color"]["criteria"] == ["red", "blue"]
    a = out.results[0].answers
    assert a["color"].type == "choice" and a["color"].choice == "red"
    assert a["size"].type == "score" and a["size"].legend == {"0": "tiny", "1": "huge"}
    assert a["is_red"].type == "noul" and a["is_red"].noul == 0.7
    assert out.timing_ms >= 0


def test_predict_joint_is_one_state(client):
    c, agent = client
    r = c.post("/predict", files=files(png(), png(), png()),
               data={"questions": json.dumps(QUESTIONS), "joint": "true", "n_permutations": 3})
    assert r.status_code == 200, r.text
    assert len(r.json()["results"]) == 1
    assert len(agent.calls) == 1 and agent.calls[0]["n_images"] == 3 and agent.calls[0]["n_permutations"] == 3


def test_questions_are_validated(client):
    c, agent = client
    bad = {"q": {"type": "guess", "instructions": "?"}}
    r = c.post("/predict", files=files(png()), data={"questions": json.dumps(bad)})
    assert r.status_code == 422 and "questions" in r.text
    r = c.post("/predict", files=files(png()), data={"questions": "not json"})
    assert r.status_code == 422
    r = c.post("/predict", files=files(png()), data={"questions": "{}"})
    assert r.status_code == 422
    assert agent.calls == []


def test_bad_image_and_limits(client):
    c, agent = client
    r = c.post("/predict", files=files(png(), b"definitely not an image", names=["a.png", "b.png"]),
               data={"questions": json.dumps(QUESTIONS)})
    assert r.status_code == 422 and "images[1]" in r.text and "b.png" in r.text
    r = c.post("/predict", files=files(*[png()] * (MAX_IMAGES + 1)), data={"questions": json.dumps(QUESTIONS)})
    assert r.status_code == 413
    r = c.post("/predict", data={"questions": json.dumps(QUESTIONS)})
    assert r.status_code == 422


def test_jpeg_draft_decodes_small():
    big = decode_image(jpeg((4000, 3000)))
    assert big.mode == "RGB"
    assert max(big.size) <= 2000 and max(big.size) >= 1000  # DCT-scaled down, never below the draft floor
    small = decode_image(jpeg((300, 200)))
    assert small.size == (300, 200)  # nothing below the floor is touched
    assert decode_image(png(size=(4000, 3000))).size == (4000, 3000)  # draft is JPEG-only


def test_health_and_openapi(client):
    c, _ = client
    assert c.get("/health").json() == {"status": "ok", "run": "stub/best"}
    schema = c.get("/openapi.json").json()
    comps = schema["components"]["schemas"]
    for name in ("Question", "Questions", "ChoiceAnswer", "ScoreAnswer", "NoulAnswer", "Result", "PredictResponse"):
        assert name in comps, name
    post = schema["paths"]["/predict"]["post"]
    assert post["operationId"] == "predict"
    assert "multipart/form-data" in post["requestBody"]["content"]
    assert post["responses"]["200"]["content"]["application/json"]["schema"]["$ref"].endswith("PredictResponse")
    body = comps[post["requestBody"]["content"]["multipart/form-data"]["schema"]["$ref"].split("/")[-1]]
    assert body["properties"]["images"]["type"] == "array"
    assert body["properties"]["questions"]["contentSchema"] == {"$ref": "#/components/schemas/Questions"}


def test_openapi_without_model():
    schema = make_app(None).openapi()
    assert "Questions" in schema["components"]["schemas"]
    c = TestClient(make_app(None))
    assert c.post("/predict", files=files(png()), data={"questions": json.dumps(QUESTIONS)}).status_code == 503
