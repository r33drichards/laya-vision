"""Python client for ``laya-serve`` (``laya/server.py``). Needs only ``requests``: importing it loads no torch.

    from laya.client import LayaClient

    client = LayaClient()                      # LAYA_BASE_URL (default http://127.0.0.1:8710), LAYA_API_KEY
    r = client.system_one(
        state={"image": "photo.jpg", "note": "front door camera"},   # a path, bytes, PIL image, URL or base64
        questions={
            "person": {"type": "noul", "instructions": "Is there a person at the door?"},
            "weather": {"type": "choice", "instructions": "What is the weather?", "criteria": ["sun", "rain", "snow"]},
            "risk": {"type": "score", "instructions": "How risky is this?", "criteria": ["none", "some", "high"]},
        },
    )
    r.answers["person"].noul                   # P(true), 0..1
    r.answers["weather"].choice                # "sunny"
    r.answers["weather"].probabilities         # {"sunny": ..., "rain": ..., "snow": ...}
    r.answers["risk"].score                    # expected level, 0..2
    client.rank(["a red car", "a blue bus"], "Which caption fits the image?", state={"image": "photo.jpg"})

The request and answers are ``VLMAgent.predict``'s (site-docs/reference/predict.md). Images in ``state["image"]`` /
``state["images"]`` that are bytes, a PIL image or an existing file path are sent as base64; any other string (an
``http(s)://`` URL, a ``data:`` URL or base64) is sent as is.
"""
import base64
import io
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import requests

DEFAULT_BASE_URL = "http://127.0.0.1:8710"


class LayaError(RuntimeError):
    """A non-200 answer from the server: ``status`` (401, 422, 502, ...) and its ``message``."""

    def __init__(self, status: int, message: str):
        super().__init__("%s: %s" % (status, message))
        self.status, self.message = status, message


# ---------------------------------------------------------------------------------------------------------
# answers
# ---------------------------------------------------------------------------------------------------------


@dataclass
class NoulAnswer:
    noul: float  # P(true)
    confidence: float
    action: Dict[str, Any] = field(default_factory=dict)
    truncated: Optional[Dict[str, Any]] = None
    value: Optional[float] = None
    type: str = "noul"

    @property
    def probabilities(self) -> Dict[str, float]:
        return {"false": 1.0 - self.noul, "true": self.noul}


@dataclass
class ChoiceAnswer:
    choice: str
    probabilities: Dict[str, float]
    confidence: float
    action: Dict[str, Any] = field(default_factory=dict)
    truncated: Optional[Dict[str, Any]] = None
    value: Optional[float] = None
    type: str = "choice"


@dataclass
class ScoreAnswer:
    score: float  # expected level
    probabilities: Dict[str, float]
    confidence: float
    legend: Dict[str, str] = field(default_factory=dict)
    action: Dict[str, Any] = field(default_factory=dict)
    truncated: Optional[Dict[str, Any]] = None
    value: Optional[float] = None
    type: str = "score"


Answer = Union[NoulAnswer, ChoiceAnswer, ScoreAnswer]


def parse_answer(d: Dict[str, Any]) -> Answer:
    t = d.get("type")
    extra = dict(action=dict(d.get("action") or {}), truncated=d.get("truncated"),
                 value=float(d["value"]) if d.get("value") is not None else None)
    if t == "noul":
        return NoulAnswer(noul=float(d["noul"]), confidence=float(d["confidence"]), **extra)
    if t == "choice":
        return ChoiceAnswer(choice=d["choice"], probabilities={k: float(v) for k, v in d["probabilities"].items()},
                            confidence=float(d["confidence"]), **extra)
    if t == "score":
        return ScoreAnswer(score=float(d["score"]), probabilities={k: float(v) for k, v in d["probabilities"].items()},
                           confidence=float(d["confidence"]), legend=dict(d.get("legend") or {}), **extra)
    raise ValueError("unknown answer type %r" % t)


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    images: int = 0


@dataclass
class SystemOneResponse:
    model: str
    answers: Dict[str, Answer]
    usage: Usage
    generation: Optional[int] = None  # how many times the server has (re)loaded this model
    provenance: Dict[str, Any] = field(default_factory=dict)
    value: Optional[float] = None
    mock: bool = False
    latency_ms: Optional[float] = None


@dataclass
class RankedCandidate:
    rank: int
    candidate: str
    prob: float
    round: int = 0  # the tournament round the prob comes from; the final is the highest


@dataclass
class RankResponse:
    model: str
    ranked: List[RankedCandidate]
    tournament: Dict[str, Any]
    usage: Usage
    generation: Optional[int] = None
    latency_ms: Optional[float] = None

    @property
    def best(self) -> str:
        return self.ranked[0].candidate


def _usage(u: Optional[Dict[str, Any]]) -> Usage:
    u = u or {}
    return Usage(int(u.get("input_tokens") or 0), int(u.get("output_tokens") or 0), int(u.get("images") or 0))


def encode_image(img: Any) -> str:
    """An image for the wire: bytes, a PIL image or an existing file path become base64; other strings pass."""
    if isinstance(img, (bytes, bytearray, memoryview)):
        return base64.b64encode(bytes(img)).decode()
    if isinstance(img, os.PathLike) or (isinstance(img, str) and not img.startswith(("http://", "https://", "data:"))
                                         and len(img) < 4096 and os.path.isfile(img)):
        with open(img, "rb") as f:
            return base64.b64encode(f.read()).decode()
    if isinstance(img, str):
        return img
    if hasattr(img, "save"):  # PIL image
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()
    raise TypeError("unsupported image type %r (expected bytes, a PIL image, a path, a URL or base64)"
                    % type(img).__name__)


def encode_state(state: Any) -> Any:
    if not isinstance(state, dict) or ("image" not in state and "images" not in state):
        if hasattr(state, "save") and hasattr(state, "convert"):  # a bare PIL image
            return {"image": encode_image(state)}
        return state
    out = dict(state)
    if out.get("image") is not None:
        out["image"] = encode_image(out["image"])
    if out.get("images") is not None:
        out["images"] = [encode_image(i) for i in out["images"]]
    return out


# ---------------------------------------------------------------------------------------------------------
# client
# ---------------------------------------------------------------------------------------------------------


class LayaClient:
    """``base_url`` / ``api_key`` default to ``LAYA_BASE_URL`` / ``LAYA_API_KEY``; ``model`` to the server's default.
    ``session`` may be any object with ``requests.Session``'s ``get`` / ``post`` (the tests pass a TestClient)."""

    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None, timeout: float = 300.0,
                 model: Optional[str] = None, session: Any = None):
        self.base_url = (base_url or os.environ.get("LAYA_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.api_key = api_key or os.environ.get("LAYA_API_KEY")
        self.timeout, self.model = timeout, model
        self._s = session if session is not None else requests.Session()
        self._headers = {"Authorization": "Bearer %s" % self.api_key} if self.api_key else {}

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    def close(self) -> None:
        if hasattr(self._s, "close"):
            self._s.close()

    def _request(self, method: str, path: str, body: Optional[Dict] = None):
        kw = {"headers": self._headers, "timeout": self.timeout}
        if body is not None:
            kw["json"] = body
        try:
            r = getattr(self._s, method)(self.base_url + path, **kw)
        except requests.RequestException as e:
            raise LayaError(0, "cannot reach %s: %s" % (self.base_url, e)) from e
        if r.status_code != 200:
            try:
                msg = r.json().get("detail", r.text)
            except ValueError:
                msg = r.text
            raise LayaError(r.status_code, str(msg))
        lat = r.headers.get("X-Laya-Latency-Ms")
        return r.json(), float(lat) if lat else None

    def system_one(self, state: Any, questions: Dict[str, Dict[str, Any]], model: Optional[str] = None,
                   temperature: Any = None, n_permutations: Optional[int] = None,
                   strict: Optional[bool] = None) -> SystemOneResponse:
        """Every question answered against one state, as ``VLMAgent.predict`` would."""
        body: Dict[str, Any] = {"state": encode_state(state), "questions": questions}
        for k, v in (("model", model or self.model), ("temperature", temperature),
                     ("n_permutations", n_permutations), ("strict", strict)):
            if v is not None:
                body[k] = v
        j, lat = self._request("post", "/v1/systemone", body)
        return SystemOneResponse(model=j["model"], answers={k: parse_answer(a) for k, a in j["answers"].items()},
                                 usage=_usage(j.get("usage")), generation=j.get("generation"),
                                 provenance=dict(j.get("provenance") or {}), value=j.get("value"),
                                 mock=bool(j.get("mock")), latency_ms=lat)

    predict = system_one

    def rank(self, candidates: List[str], instructions: Any = None, state: Any = None, model: Optional[str] = None,
             temperature: Any = None, chunk_size: Optional[int] = None) -> RankResponse:
        """``candidates`` best first for ``state`` + ``instructions``; large lists run as a server-side tournament."""
        body: Dict[str, Any] = {"candidates": list(candidates)}
        for k, v in (("instructions", instructions), ("state", encode_state(state) if state is not None else None),
                     ("model", model or self.model), ("temperature", temperature), ("chunk_size", chunk_size)):
            if v is not None:
                body[k] = v
        j, lat = self._request("post", "/v1/rank", body)
        return RankResponse(model=j["model"], ranked=[RankedCandidate(**r) for r in j["ranked"]],
                            tournament=dict(j.get("tournament") or {}), usage=_usage(j.get("usage")),
                            generation=j.get("generation"), latency_ms=lat)

    def models(self) -> List[Dict[str, Any]]:
        return self._request("get", "/v1/models")[0]["models"]

    def health(self) -> Dict[str, Any]:
        """The server's ``/health`` (``{"ok", "mock", "models"}``); raises ``LayaError`` when unreachable."""
        return self._request("get", "/health")[0]
