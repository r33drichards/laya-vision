"""Parity between the browser demo's JavaScript port (``web-demo/laya.js``) and ``laya.vlm``.

The token-id half is a committed fixture: ``web-demo/fixtures/parity.json`` holds the ids, markers and option spans
``build_vlm_inputs`` produces for a few fixed inputs. ``test_fixture_matches_python`` regenerates them from Python
(tokenizer only, no model weights), so a change to the sequence format fails here until the fixture is refreshed:

    python tests/test_web_demo.py        # rewrite web-demo/fixtures/parity.json

``test_js_matches_fixture`` runs ``web-demo/test_parity.mjs`` under Node against the same fixture, and
``test_js_pixels_match_processor`` compares the JavaScript resize with the Hugging Face processor on real pixels.
Both need Node and ``npm install`` in ``web-demo/`` (for ``@huggingface/tokenizers``) and skip otherwise.
"""
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB = os.path.join(ROOT, "web-demo")
FIXTURE = os.path.join(WEB, "fixtures", "parity.json")
BACKBONE = "HuggingFaceTB/SmolVLM-256M-Instruct"

sys.path[:0] = [ROOT, os.path.join(ROOT, "scripts")]

LONG = " ".join("word%d" % i for i in range(1500))
CASES = [
    {"name": "readme-example", "n_images": 1, "state": {"note": "customer says it arrived broken"},
     "questions": {
         "damage": {"type": "score", "instructions": "How much damage does the item show?",
                    "criteria": ["none", "cosmetic: scratches or dents", "functional: parts broken or missing",
                                 "destroyed"]},
         "category": {"type": "choice", "instructions": "What kind of item is this?",
                      "criteria": ["electronics", "clothing", "furniture", "food", "other"]},
         "outdoors": {"type": "noul", "instructions": "Was the photo taken outdoors?"}}},
    {"name": "text-only-string", "n_images": 0, "state": "Plain text state.\nSecond line with émoji 🚀.",
     "questions": {
         "route": {"type": "choice", "instructions": {"goal": "pick a route", "notes": ["fast", "café"]},
                   "criteria": {"north": "via the bridge\nthen left", "south": "", "stay": None}},
         "ok": {"type": "noul", "instructions": "Is it fine?<end_of_utterance> really?",
                "criteria": {"true": "it is fine", "false": "it is broken"}}}},
    {"name": "two-images-no-text", "n_images": 2, "state": {},
     "questions": {"same": {"type": "noul", "instructions": "Do both images show the same scene?"},
                   "q": {"type": "score", "instructions": "Rate the lighting.", "criteria": ["dark", "ok", "bright"]}}},
    {"name": "state-truncated", "n_images": 1, "state": {"log": LONG, "n": 3, "flag": True, "x": None},
     "questions": {"c": {"type": "choice", "instructions": "Which?", "criteria": ["a", "b"]}}},
    {"name": "many-long-options", "n_images": 1, "state": {"k": "v"},
     "questions": {"many": {"type": "choice", "instructions": "Pick the best description. " * 40,
                            "criteria": ["option %d: " % i + "a rather long description of it " * 6
                                         for i in range(12)]}}},
]


def _processor():
    from transformers import AutoProcessor

    from laya.preprocess import ImagePrep

    proc = AutoProcessor.from_pretrained(BACKBONE)
    prep = ImagePrep(backend="processor")
    prep.apply(proc)
    return proc, prep


def build_fixture():
    from export_onnx import sequence_config

    from laya.preprocess import prefix_ids
    from laya.vlm import PREFIX_TEXT, VLMAgent, _permutations, build_vlm_inputs

    proc, prep = _processor()
    cfg = sequence_config(proc, prep)
    out = {"backbone": BACKBONE, "cfg": cfg, "cases": []}
    for case in CASES:
        prefix = {"ids": prefix_ids(proc, PREFIX_TEXT, case["n_images"], prep.image_seq_len), "pixel_values": None,
                  "pixel_attention_mask": None, "raw_images": None, "n_images": case["n_images"]}
        rows = []
        for qid, qdef in case["questions"].items():
            q = VLMAgent._to_internal(qdef)
            k = len(q["crit"]) if q["t"] != "noul" else 2
            for order in _permutations(k, 2):
                it = build_vlm_inputs(proc, case["state"], q, cfg["max_len"], cfg["head_max_len"], option_order=order,
                                      prefix=prefix)
                rows.append({"qid": qid, "order": order, "ids": it["ids"], "markers": it["markers"],
                             "option_span": list(it["option_span"])})
        out["cases"].append(dict(case, prefix_ids=prefix["ids"], rows=rows))
    return out


@pytest.fixture(scope="module")
def fixture():
    with open(FIXTURE) as f:
        return json.load(f)


def test_fixture_matches_python(fixture):
    assert build_fixture() == fixture, "sequence format changed: run `python tests/test_web_demo.py` to refresh"


def _node_ready():
    return shutil.which("node") and os.path.isdir(os.path.join(WEB, "node_modules", "@huggingface", "tokenizers"))


def _tokenizer_dir():
    from huggingface_hub import snapshot_download

    return snapshot_download(BACKBONE, allow_patterns=["tokenizer.json", "tokenizer_config.json"])


@pytest.mark.skipif(not _node_ready(), reason="needs node and `npm install` in web-demo/")
def test_js_matches_fixture():
    r = subprocess.run(["node", os.path.join(WEB, "test_parity.mjs"), "ids", _tokenizer_dir()],
                       capture_output=True, text=True, cwd=WEB, timeout=300)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "rows match" in r.stdout


def _test_images():
    from PIL import Image

    rng = np.random.RandomState(0)
    y, x = np.mgrid[0:480, 0:640]
    smooth = np.stack([x * 255 // 639, y * 255 // 479, (x + y) % 256], -1).astype(np.uint8)
    return {"smooth-640x480": smooth, "noise-210x160": rng.randint(0, 256, (210, 160, 3), dtype=np.uint8),
            "tall-100x300": rng.randint(0, 256, (300, 100, 3), dtype=np.uint8),
            "big-2600x1300": np.asarray(Image.fromarray(smooth).resize((2600, 1300)))}


@pytest.mark.skipif(not _node_ready(), reason="needs node and `npm install` in web-demo/")
def test_js_pixels_match_processor(tmp_path):
    proc, prep = _processor()
    from PIL import Image

    for name, arr in _test_images().items():
        raw = tmp_path / (name + ".rgb")
        raw.write_bytes(arr.tobytes())
        got = tmp_path / (name + ".out")
        r = subprocess.run(["node", os.path.join(WEB, "test_parity.mjs"), "pixels", str(raw), str(arr.shape[0]),
                            str(arr.shape[1]), str(got)], capture_output=True, text=True, cwd=WEB, timeout=300)
        assert r.returncode == 0, r.stdout + r.stderr
        js = np.frombuffer(got.read_bytes(), dtype=np.float32).reshape(3, prep.image_size, prep.image_size)
        ref = proc(text=[proc.image_token], images=[[Image.fromarray(arr)]], do_image_splitting=False,
                   return_tensors="pt")["pixel_values"][0, 0].numpy()
        levels = np.abs(js - ref) * 127.5
        print("%s: mean %.3f max %.1f grey levels" % (name, levels.mean(), levels.max()))
        assert levels.mean() < 0.05 and levels.max() <= 3.0, name


if __name__ == "__main__":
    os.makedirs(os.path.dirname(FIXTURE), exist_ok=True)
    with open(FIXTURE, "w") as f:
        json.dump(build_fixture(), f, ensure_ascii=False)
    print("wrote", FIXTURE)


def _toposort(model):
    """The fp16 converter appends its casts after the nodes that read them; onnxruntime reorders, the reference
    evaluator does not."""
    g = model.graph
    known = {i.name for i in g.input} | {t.name for t in g.initializer} | {""}
    todo, done = list(g.node), []
    while todo:
        ready = [n for n in todo if all(i in known for i in n.input)]
        assert ready, "graph has a missing input or a cycle"
        for n in ready:
            known.update(n.output)
        done += ready
        todo = [n for n in todo if n not in ready]
    del g.node[:]
    g.node.extend(done)
    return model


def test_fp16_keeps_rmsnorm_in_fp32(tmp_path):
    """The fp16 variant must not square the residual stream in float16 (``export_onnx.rmsnorm_in_fp32``). A Llama
    RMSNorm fed values of a few thousand, as SmolVLM's last layers produce, run with real float16 arithmetic
    (onnx's reference evaluator; onnxruntime's CPU backend would silently upcast and hide the bug)."""
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from onnx.reference import ReferenceEvaluator
    from onnxruntime.transformers.float16 import convert_float_to_float16
    from transformers.models.llama.modeling_llama import LlamaRMSNorm

    from export_onnx import rmsnorm_in_fp32

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(64, 64)
            self.norm = LlamaRMSNorm(64)

        def forward(self, x):
            return self.norm(self.proj(x))

    torch.manual_seed(0)
    block = Block().eval()
    x = torch.randn(1, 8, 64) * 3000
    path = str(tmp_path / "block.onnx")
    torch.onnx.export(block, (x,), path, input_names=["x"], output_names=["y"], opset_version=18, dynamo=False)
    want = block(x).detach().numpy()
    m = onnx.load(path)
    # the plain conversion (what the first published fp16 files were): all zeros in real float16
    naive = ReferenceEvaluator(_toposort(convert_float_to_float16(onnx.load(path), keep_io_types=True))).run(
        None, {"x": x.numpy()})[0]
    assert np.abs(naive).max() == 0, "expected the plain conversion to overflow to zeros"
    fixed = rmsnorm_in_fp32(convert_float_to_float16(m, keep_io_types=True))
    got = ReferenceEvaluator(_toposort(fixed)).run(None, {"x": x.numpy()})[0]
    np.testing.assert_allclose(got, want, atol=0.05)
