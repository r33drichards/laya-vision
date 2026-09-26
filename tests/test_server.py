"""``laya-serve`` end to end on the fake model (``laya.mock``): no weights, no torch, CPU only."""
import base64
import io
import json
import os
import subprocess
import sys

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from laya.client import (ChoiceAnswer, LayaClient, LayaError, NoulAnswer, ScoreAnswer,  # noqa: E402
                         encode_image, parse_answer)
from laya.mock import MockAgent  # noqa: E402
from laya.server import (Engine, ImageFetchError, create_app, decode_image, parse_model_spec,  # noqa: E402
                         validate_questions)

QUESTIONS = {
    "weather": {"type": "choice", "instructions": "What is the weather?", "criteria": ["sunny", "rain", "snow"]},
    "team": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "charges", "tech": "bugs"}},
    "risk": {"type": "score", "instructions": "How risky is this?", "criteria": ["none", "some", "high"]},
    "person": {"type": "noul", "instructions": "Is there a person?"},
}
MANY = ["option %d" % i for i in range(300)]  # more than one question may hold


def _png(color=(200, 30, 30), size=(8, 6)) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def _serve(api_key=None, models=None, fetch=None, **engine_kw):
    engine = Engine(models or {"mock": None}, MockAgent.load, **engine_kw)
    return engine, TestClient(create_app(engine, api_key, fetch=fetch))


# ---------------------------------------------------------------------------------------------------------
# the mock itself


def test_mock_predict_schema_matches_vlm_agent():
    out = MockAgent().predict({"note": "a sunny beach", "image": _png()}, QUESTIONS)
    a = out["answers"]
    assert set(a) == set(QUESTIONS) and out["mock"] is True and out["usage"]["images"] == 1
    assert a["weather"]["choice"] in ("sunny", "rain", "snow")
    assert set(a["team"]["probabilities"]) == {"billing", "tech"}
    assert abs(sum(a["weather"]["probabilities"].values()) - 1) < 1e-3
    assert set(a["risk"]["probabilities"]) == {"0", "1", "2"} and a["risk"]["legend"]["2"] == "high"
    assert 0 <= a["risk"]["score"] <= 2 and 0 <= a["person"]["noul"] <= 1
    assert all("confidence" in v and "act_probability" in v["action"] for v in a.values())
    again = MockAgent().predict({"note": "a sunny beach", "image": _png()}, QUESTIONS)
    assert again["answers"] == a  # deterministic
    other = MockAgent().predict({"note": "a sunny beach", "image": _png((0, 0, 255))}, QUESTIONS)
    assert other["answers"] != a  # the image matters


def test_mock_temperature_flattens_but_keeps_the_argmax():
    q = {"w": QUESTIONS["weather"]}
    sharp = MockAgent().predict("rain all day, rain", q, temperature=0.5)["answers"]["w"]
    flat = MockAgent().predict("rain all day, rain", q, temperature={"choice": 5.0})["answers"]["w"]
    assert sharp["choice"] == flat["choice"]
    assert max(flat["probabilities"].values()) < max(sharp["probabilities"].values())


# ---------------------------------------------------------------------------------------------------------
# /health, /v1/models, /v1/systemone


def test_health_reports_mock_and_generations():
    _, c = _serve(api_key="k")
    h = c.get("/health")  # never needs the key
    assert h.status_code == 200
    assert h.json() == {"ok": True, "mock": True, "default_model": "mock", "models": {"mock": 1}}


def test_systemone_matches_predict_for_base64_data_url_and_url_images():
    png = _png()
    fetched = []

    def fetch(url, max_bytes):
        fetched.append(url)
        return png

    _, c = _serve(fetch=fetch)
    want = MockAgent().predict({"image": png, "note": "x"}, QUESTIONS)["answers"]
    b64 = base64.b64encode(png).decode()
    for img in (b64, "data:image/png;base64," + b64, "https://example.com/cat.png"):
        r = c.post("/v1/systemone", json={"state": {"image": img, "note": "x"}, "questions": QUESTIONS})
        assert r.status_code == 200, r.text
        j = r.json()
        assert j["answers"] == want and j["model"] == "mock" and j["generation"] == 1
        assert float(r.headers["X-Laya-Latency-Ms"]) >= 0
    assert fetched == ["https://example.com/cat.png"]
    r = c.post("/v1/systemone", json={"state": {"images": [b64, b64]}, "questions": {"p": QUESTIONS["person"]}})
    assert r.status_code == 200 and r.json()["usage"]["images"] == 2


def test_systemone_passes_temperature_and_permutations():
    _, c = _serve()
    body = {"state": "rain", "questions": {"w": QUESTIONS["weather"]}, "temperature": {"choice": 2.0},
            "n_permutations": 3}
    j = c.post("/v1/systemone", json=body).json()
    assert j["provenance"]["temperatures"] == {"w": 2.0} and j["provenance"]["n_permutations"] == 3


def test_a_local_path_is_never_opened(tmp_path):
    p = tmp_path / "secret.png"
    p.write_bytes(_png())
    _, c = _serve()
    r = c.post("/v1/systemone", json={"state": {"image": str(p)}, "questions": {"p": QUESTIONS["person"]}})
    assert r.status_code == 422 and "base64" in r.json()["detail"]


@pytest.mark.parametrize("body, needle", [
    ({"questions": QUESTIONS}, "state"),
    ({"state": "x", "questions": {}}, "non-empty"),
    ({"state": "x", "questions": {"q": {"type": "maybe", "instructions": "?"}}}, "type must be"),
    ({"state": "x", "questions": {"q": {"type": "noul"}}}, "instructions"),
    ({"state": "x", "questions": {"q": {"type": "choice", "instructions": "?", "criteria": ["only"]}}}, "at least 2"),
    ({"state": "x", "questions": {"q": {"type": "choice", "instructions": "?", "criteria": ["a", "a"]}}}, "distinct"),
    ({"state": "x", "questions": {"q": {"type": "score", "instructions": "?", "criteria": "abc"}}}, "score needs"),
    ({"state": "x", "questions": {"q": {"type": "noul", "instructions": "?", "criteria": {"maybe": "?"}}}}, "noul"),
    ({"state": "x", "questions": {"q": {"type": "choice", "instructions": "?", "criteria": MANY}}}, "/v1/rank"),
    ({"state": "x", "questions": QUESTIONS, "model": "nope"}, "unknown model"),
    ({"state": "x", "questions": QUESTIONS, "temperature": 0}, "temperature"),
    ({"state": "x", "questions": QUESTIONS, "temperature": {"bogus": 1.0}}, "temperature keys"),
    ({"state": "x", "questions": QUESTIONS, "n_permutations": 99}, "n_permutations"),
    ({"state": {"image": "not base64!"}, "questions": QUESTIONS}, "base64"),
    ({"state": {"images": "abc"}, "questions": QUESTIONS}, "list"),
])
def test_systemone_422s_name_the_problem(body, needle):
    _, c = _serve()
    r = c.post("/v1/systemone", json=body)
    assert r.status_code == 422 and needle in r.json()["detail"], r.text


def test_non_json_body_is_422():
    _, c = _serve()
    r = c.post("/v1/systemone", content=b"{nope", headers={"Content-Type": "application/json"})
    assert r.status_code == 422 and "not JSON" in r.json()["detail"]


def test_image_fetch_failure_is_502_and_urls_can_be_disabled():
    def fetch(url, max_bytes):
        raise ImageFetchError("fetching image %s: HTTP 404" % url)

    _, c = _serve(fetch=fetch)
    r = c.post("/v1/systemone", json={"state": {"image": "http://x/y.png"}, "questions": QUESTIONS})
    assert r.status_code == 502 and "HTTP 404" in r.json()["detail"]
    with pytest.raises(ValueError, match="disabled"):
        decode_image("http://x/y.png", allow_urls=False)


def test_bearer_auth():
    _, c = _serve(api_key="s3cret")
    body = {"state": "x", "questions": {"p": QUESTIONS["person"]}}
    assert c.post("/v1/systemone", json=body).status_code == 401
    assert c.post("/v1/systemone", json=body, headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert c.get("/v1/models").status_code == 401
    assert c.post("/v1/rank", json={"candidates": ["a", "b"]}).status_code == 401
    ok = {"Authorization": "Bearer s3cret"}
    assert c.post("/v1/systemone", json=body, headers=ok).status_code == 200
    assert c.get("/v1/models", headers=ok).status_code == 200


def test_cors_is_off_by_default():
    engine = Engine({"mock": None}, MockAgent.load)
    pre = {"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"}
    off = TestClient(create_app(engine)).options("/v1/systemone", headers=pre)
    assert "access-control-allow-origin" not in off.headers
    on = TestClient(create_app(engine, cors=True)).options("/v1/systemone", headers=pre)
    assert on.headers.get("access-control-allow-origin") == "*"


# ---------------------------------------------------------------------------------------------------------
# several checkpoints, hot reload


def _ckpt(path, seed):
    path.mkdir(exist_ok=True)
    cfg = path / "vlm_agent_config.json"
    cfg.write_text(json.dumps({"mock_seed": seed}))
    st = cfg.stat()
    os.utime(cfg, ns=(st.st_atime_ns, st.st_mtime_ns - 10 ** 10 - seed))  # "written" 10 s ago, distinct per seed


def test_several_models_by_name(tmp_path):
    _ckpt(tmp_path / "a", 1)
    _ckpt(tmp_path / "b", 2)
    _, c = _serve(models={"prod": str(tmp_path / "a"), "next": str(tmp_path / "b")})
    ms = c.get("/v1/models").json()
    assert ms["default"] == "prod" and [m["name"] for m in ms["models"]] == ["prod", "next"]
    assert all(m["watched"] and m["mock"] for m in ms["models"])
    body = {"state": "a red car", "questions": QUESTIONS}
    default, prod, nxt = (c.post("/v1/systemone", json=dict(body, **kw)).json() for kw in ({}, {"model": "prod"},
                                                                                          {"model": "next"}))
    assert default["model"] == prod["model"] == "prod" and nxt["model"] == "next"
    assert default["answers"] == prod["answers"] != nxt["answers"]
    assert nxt["provenance"]["mock_seed"] == 2


def test_hot_reload_bumps_the_generation_and_survives_a_bad_checkpoint(tmp_path):
    ck = tmp_path / "ck"
    _ckpt(ck, 0)
    engine, c = _serve(models={"m": str(ck)}, reload_interval=0, reload_settle=1)
    body = {"state": "a red car", "questions": QUESTIONS}
    first = c.post("/v1/systemone", json=body).json()
    assert first["generation"] == 1 and first["provenance"]["mock_seed"] == 0

    _ckpt(ck, 1)
    second = c.post("/v1/systemone", json=body).json()
    assert second["generation"] == 2 and second["provenance"]["mock_seed"] == 1
    assert second["answers"] != first["answers"]
    assert c.get("/health").json()["models"] == {"m": 2}

    (ck / "vlm_agent_config.json").write_text("{half written")  # fresh: not loaded until it settles
    assert c.post("/v1/systemone", json=body).json()["generation"] == 2
    st = (ck / "vlm_agent_config.json").stat()
    os.utime(ck / "vlm_agent_config.json", ns=(st.st_atime_ns, st.st_mtime_ns - 5 * 10 ** 9))
    third = c.post("/v1/systemone", json=body)  # the load fails: the old model keeps serving
    assert third.status_code == 200 and third.json()["generation"] == 2
    assert third.json()["answers"] == second["answers"]
    assert "reload failed" in c.get("/v1/models").json()["models"][0]["last_error"]

    _ckpt(ck, 3)
    assert c.post("/v1/systemone", json=body).json()["generation"] == 3
    assert "last_error" not in c.get("/v1/models").json()["models"][0]


def test_reload_can_be_turned_off(tmp_path):
    ck = tmp_path / "ck"
    _ckpt(ck, 0)
    _, c = _serve(models={"m": str(ck)}, reload=False, reload_interval=0, reload_settle=0)
    _ckpt(ck, 1)
    assert c.post("/v1/systemone", json={"state": "x", "questions": QUESTIONS}).json()["generation"] == 1


# ---------------------------------------------------------------------------------------------------------
# /v1/rank


def _global_order(cands, instructions, state=""):
    """The mock scores each option on its own, so ranking any subset keeps this order: a tournament must agree."""
    raw = {}
    MockAgent().predict(state, {"q": {"type": "choice", "instructions": instructions, "criteria": cands}},
                        _raw_logits=raw)
    return [c for _, c in sorted(zip(raw["q"], cands), key=lambda t: -t[0])]


def test_rank_small_list_is_one_question():
    _, c = _serve()
    cands = ["a red sports car", "a blue city bus", "a green tractor"]
    j = c.post("/v1/rank", json={"candidates": cands, "instructions": "Which is a car?", "state": "red car"}).json()
    assert [r["candidate"] for r in j["ranked"]] == _global_order(cands, "Which is a car?", "red car")
    assert [r["rank"] for r in j["ranked"]] == [1, 2, 3] and j["tournament"] == {"rounds": 1, "chunk_size": 255,
                                                                                  "questions": 1}
    assert abs(sum(r["prob"] for r in j["ranked"]) - 1) < 1e-3


@pytest.mark.parametrize("n, chunk, rounds, questions", [(600, None, 2, 4), (20, 3, 3, 7 + 3 + 1), (256, None, 2, 3)])
def test_rank_tournament(n, chunk, rounds, questions):
    _, c = _serve()
    cands = ["candidate number %d about %s" % (i, ("cats", "dogs", "cars", "boats")[i % 4]) for i in range(n)]
    body = {"candidates": cands, "instructions": "Which one is about boats?"}
    if chunk:
        body["chunk_size"] = chunk
    j = c.post("/v1/rank", json=body).json()
    ranked = [r["candidate"] for r in j["ranked"]]
    assert sorted(ranked) == sorted(cands) and [r["rank"] for r in j["ranked"]] == list(range(1, n + 1))
    assert j["tournament"]["rounds"] == rounds and j["tournament"]["questions"] == questions
    rounds_seen = [r["round"] for r in j["ranked"]]
    assert rounds_seen == sorted(rounds_seen, reverse=True) and rounds_seen[0] == rounds - 1
    order = _global_order(cands, "Which one is about boats?")
    assert ranked[0] == order[0]  # the true best always wins
    finalists = [r["candidate"] for r in j["ranked"] if r["round"] == rounds - 1]
    assert finalists == [x for x in order if x in finalists]
    final_mass = sum(r["prob"] for r in j["ranked"] if r["round"] == rounds - 1)
    assert abs(final_mass - 1) < 5e-5 * len(finalists)  # predict rounds each probability to 4 places


@pytest.mark.parametrize("body, needle", [
    ({}, "candidates"),
    ({"candidates": ["a", "a"]}, "distinct"),
    ({"candidates": ["a", ""]}, "distinct"),
    ({"candidates": ["a", "b"], "chunk_size": 1}, "chunk_size"),
    ({"candidates": ["a", "b"], "chunk_size": 999}, "chunk_size"),
    ({"candidates": ["a", "b"], "model": "nope"}, "unknown model"),
])
def test_rank_422s(body, needle):
    _, c = _serve()
    r = c.post("/v1/rank", json=body)
    assert r.status_code == 422 and needle in r.json()["detail"], r.text


# ---------------------------------------------------------------------------------------------------------
# the client


def test_client_parses_typed_answers(tmp_path):
    _, tc = _serve(api_key="k")
    client = LayaClient(base_url="http://testserver", api_key="k", session=tc)
    png = _png()
    p = tmp_path / "img.png"
    p.write_bytes(png)
    from PIL import Image

    rs = [client.system_one({"image": img, "note": "x"}, QUESTIONS) for img in (png, str(p), p, Image.open(p))]
    r = rs[0]
    assert isinstance(r.answers["weather"], ChoiceAnswer) and isinstance(r.answers["risk"], ScoreAnswer)
    assert isinstance(r.answers["person"], NoulAnswer) and r.mock and r.model == "mock" and r.generation == 1
    assert r.usage.images == 1 and r.latency_ms is not None
    assert r.answers["person"].probabilities["true"] == r.answers["person"].noul
    assert r.answers["risk"].legend["0"] == "none"
    assert all(x.answers == r.answers for x in rs[1:3])  # bytes, path string and Path send the same image
    assert rs[3].answers["weather"].choice in ("sunny", "rain", "snow")  # PIL: re-encoded, so maybe other bytes
    assert client.predict == client.system_one
    assert client.health()["mock"] is True and client.models()[0]["name"] == "mock"
    rk = client.rank(["a", "b", "c"], "Pick one", chunk_size=2)
    assert rk.best == rk.ranked[0].candidate and rk.tournament["rounds"] == 2


def test_client_errors_and_env(monkeypatch):
    _, tc = _serve(api_key="k")
    monkeypatch.setenv("LAYA_BASE_URL", "http://testserver/")
    monkeypatch.setenv("LAYA_API_KEY", "wrong")
    client = LayaClient(session=tc)
    assert client.base_url == "http://testserver"
    with pytest.raises(LayaError) as e:
        client.system_one("x", QUESTIONS)
    assert e.value.status == 401
    client = LayaClient(api_key="k", session=tc, base_url="http://testserver")
    with pytest.raises(LayaError) as e:
        client.system_one("x", QUESTIONS, model="nope")
    assert e.value.status == 422 and "unknown model" in e.value.message
    with pytest.raises(LayaError) as e:
        LayaClient(base_url="http://127.0.0.1:9", timeout=2).health()
    assert e.value.status == 0


def test_parse_answer_and_encode_image():
    assert parse_answer({"type": "noul", "noul": 0.25, "confidence": 0.75}).probabilities == {"false": 0.75,
                                                                                             "true": 0.25}
    with pytest.raises(ValueError):
        parse_answer({"type": "bogus"})
    assert encode_image(b"\x89PNG") == base64.b64encode(b"\x89PNG").decode()
    assert encode_image("https://x/y.png") == "https://x/y.png"


def test_client_import_loads_no_torch():
    code = "import sys, laya.client, laya.mock, laya.server; sys.exit(int('torch' in sys.modules))"
    assert subprocess.run([sys.executable, "-c", code], cwd=os.path.dirname(os.path.dirname(__file__))).returncode == 0


def test_public_api_still_resolves_lazily():
    import laya

    assert "VLMAgent" in dir(laya) and laya.LayaClient is LayaClient
    with pytest.raises(AttributeError):
        laya.not_a_name


def test_parse_model_spec_and_validate(tmp_path):
    assert parse_model_spec("prod=thaitea/laya-vision@abc123") == ("prod", "thaitea/laya-vision", "abc123")
    assert parse_model_spec("x=%s" % tmp_path) == ("x", str(tmp_path), None)
    with pytest.raises(SystemExit):
        parse_model_spec("no-equals")
    assert validate_questions(QUESTIONS) is QUESTIONS
