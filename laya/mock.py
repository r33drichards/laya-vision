"""A fake ``VLMAgent`` for CPU tests and for working on the server or a client without weights.

``MockAgent.predict`` takes the same arguments and returns the same schema as ``VLMAgent.predict`` (the answer
types, ``probabilities`` / ``confidence`` / ``action``, ``usage`` and a ``provenance`` block), but its logits are
character n-gram overlap between the state (plus a digest of each image's bytes) and each option's rendered text.
The numbers are deterministic lexical-overlap noise, not Laya predictions: every result carries ``"mock": true``
and the server reports ``{"mock": true}`` on ``/health``.

    laya-serve --mock                          # the real server, fake model, no torch needed
    from laya.mock import MockAgent
    MockAgent().predict({"image": png_bytes, "note": "a red car"}, {"q": {"type": "noul", "instructions": "Red?"}})

It imports neither torch nor transformers, so the server tests run anywhere ``fastapi`` does. A checkpoint
directory for it is any directory; ``MockAgent.load(path)`` reads an optional ``vlm_agent_config.json`` whose
``mock_seed`` changes every answer, which is what the hot-reload tests rewrite.
"""
import hashlib
import json
import math
import os
from typing import Any, Dict, List, Optional

QTYPES = {"choice": 0, "score": 1, "noul": 2}  # laya.common.QTYPES, without importing torch
DIM = 512
SCALE = 12.0
CONFIG_NAME = "vlm_agent_config.json"  # laya.vlm.CONFIG_NAME


def _features(text: str, seed: int) -> List[float]:
    """Hashed character 3/4-grams, L2 normalised: deterministic across runs and platforms."""
    v = [0.0] * DIM
    t = " " + " ".join(text.lower().split()) + " "
    salt = str(seed).encode()
    for n in (3, 4):
        for i in range(len(t) - n + 1):
            h = int.from_bytes(hashlib.blake2b(t[i:i + n].encode(), digest_size=4, salt=salt).digest(), "big")
            v[h % DIM] += 1.0
    norm = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / norm for x in v]


def _image_digest(img: Any) -> str:
    """A short stable tag for an image, so different images give different answers."""
    if isinstance(img, (bytes, bytearray, memoryview)):
        return hashlib.sha256(bytes(img)).hexdigest()[:16]
    if isinstance(img, (str, os.PathLike)) and os.path.exists(img):
        with open(img, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()[:16]
    if hasattr(img, "tobytes"):  # PIL image or numpy array
        return hashlib.sha256(img.tobytes()).hexdigest()[:16]
    raise TypeError("unsupported image type %r (expected PIL.Image, uint8 array, path or encoded bytes)"
                    % type(img).__name__)


def _split_state(state: Any):
    """(image digests, text) like ``laya.vlm.split_state``."""
    if not isinstance(state, dict):
        if hasattr(state, "convert") and hasattr(state, "tobytes"):  # a bare PIL image
            return [_image_digest(state)], ""
        return [], state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
    images = []
    if state.get("image") is not None:
        images.append(_image_digest(state["image"]))
    for img in state.get("images") or []:
        images.append(_image_digest(img))
    rest = {k: v for k, v in state.items() if k not in ("image", "images")}
    return images, json.dumps(rest, ensure_ascii=False) if rest else ""


def _render_options(t: str, crit: Any) -> List[str]:
    """``laya.common.render_options`` for a public question definition."""
    if t == "choice":
        crit = {c: None for c in crit} if isinstance(crit, list) else crit
        return [k if not v else "%s: %s" % (k, v) for k, v in crit.items()]
    if t == "score":
        return ["level %d: %s" % (i, c) for i, c in enumerate(crit)]
    crit = crit or {}
    return ["false: " + (crit.get("false") or "no, the statement does not hold"),
            "true: " + (crit.get("true") or "yes, the statement holds")]


def _confidence(p: List[float]) -> float:
    """``laya.common.confidence_from_probs``: 1 - H(p) / log(k)."""
    k = len(p)
    if k < 2:
        return 1.0
    ent = -sum(x * math.log(min(1.0, max(x, 1e-12))) for x in p)
    return float(min(1.0, max(0.0, 1.0 - ent / math.log(k))))


class MockAgent:
    """Same ``predict`` / ``system_one`` surface as ``laya.VLMAgent``; see the module docstring."""

    mock = True

    def __init__(self, seed: int = 0, temperature: Optional[List[float]] = None, source: Optional[str] = None):
        self.seed = int(seed)
        self.temperature = list(temperature or [1.0, 1.0, 1.0])
        self.source = {"id": source, "revision": None}
        self.calls = 0

    @classmethod
    def load(cls, path: Optional[str] = None, **_) -> "MockAgent":
        """A mock "checkpoint": ``path`` may hold a ``vlm_agent_config.json`` with ``mock_seed``."""
        cfg = {}
        cfg_path = os.path.join(path, CONFIG_NAME) if path and os.path.isdir(path) else path
        if cfg_path and os.path.isfile(cfg_path):
            with open(cfg_path) as f:
                cfg = json.load(f)
        return cls(seed=cfg.get("mock_seed", 0), temperature=cfg.get("temperature"), source=path)

    def predict(self, state: Any, questions: Dict[str, Dict[str, Any]], n_permutations: int = 1, batch_size: int = 8,
                prefix_cache: Optional[bool] = None, temperature: Any = None, calibration: Any = None,
                strict_calibration: bool = False, _raw_logits: Optional[Dict] = None, strict: bool = False
                ) -> Dict[str, Any]:
        self.calls += 1
        images, text = _split_state(state)
        s_vec = _features(" ".join(images) + " " + text, self.seed)
        answers, temps, n_tokens = {}, {}, 0
        for qid, qdef in questions.items():
            t = qdef["type"]
            if t not in QTYPES:
                raise KeyError(t)
            ins = qdef["instructions"]
            ins = ins if isinstance(ins, str) else json.dumps(ins)
            opts = _render_options(t, qdef.get("criteria"))
            q_vec = _features(ins, self.seed)
            logits = []
            for o in opts:
                o_vec = _features(o, self.seed)
                logits.append(SCALE * sum(a * (b + c) for a, b, c in zip(o_vec, s_vec, q_vec)))
            n_tokens += (len(text) + len(ins) + sum(len(o) for o in opts)) // 4 + 64 * len(images)
            if _raw_logits is not None:
                _raw_logits[qid] = list(logits)
            if isinstance(temperature, dict):
                ts = float(temperature.get(t, self.temperature[QTYPES[t]]))
            elif temperature is not None:
                ts = float(temperature)
            else:
                ts = self.temperature[QTYPES[t]]
            temps[qid] = ts
            z = [v / max(1e-3, ts) for v in logits]
            m = max(z)
            e = [math.exp(v - m) for v in z]
            p = [v / sum(e) for v in e]
            j = max(range(len(p)), key=p.__getitem__)
            ext = {"action": {"act_probability": 0.5}}
            if t == "choice":
                crit = qdef["criteria"]
                keys = list(crit) if isinstance(crit, (list, dict)) else []
                answers[qid] = dict(type="choice", choice=keys[j],
                                    probabilities={k: round(v, 4) for k, v in zip(keys, p)},
                                    confidence=round(_confidence(p), 4), **ext)
            elif t == "score":
                answers[qid] = dict(type="score", score=round(sum(i * v for i, v in enumerate(p)), 4),
                                    legend={str(i): c for i, c in enumerate(qdef["criteria"])},
                                    probabilities={str(i): round(v, 4) for i, v in enumerate(p)},
                                    confidence=round(_confidence(p), 4), **ext)
            else:
                answers[qid] = dict(type="noul", noul=round(p[1], 4), confidence=round(max(p[1], 1.0 - p[1]), 4),
                                    **ext)
        return {
            "model": "laya-vlm-mock",
            "answers": answers,
            "usage": {"input_tokens": n_tokens, "output_tokens": 0, "images": len(images)},
            "provenance": {"mock": True, "mock_seed": self.seed, "checkpoint": dict(self.source),
                           "n_permutations": max(1, n_permutations), "temperatures": temps},
            "mock": True,
        }

    system_one = predict
