"""Parity between the iOS app's Swift core (``ios/LayaCore``) and ``laya.vlm``.

The same checks as ``tests/test_web_demo.py`` runs on the JavaScript port, against the same committed fixture
(``web-demo/fixtures/parity.json``): token ids, markers and option spans for every row, the tokenizer on strings that
exercise each branch of the GPT-2 pre-tokenizer, and the image resize against the Hugging Face processor. They build
``ios/LayaCore`` with ``swift build`` (Linux or macOS; the package uses no Apple-only frameworks) and skip when there
is no Swift toolchain.
"""
import json
import os
import shutil
import subprocess

import numpy as np
import pytest

from test_web_demo import BACKBONE, FIXTURE, _processor, _test_images, _tokenizer_dir

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(ROOT, "ios", "LayaCore")

pytestmark = pytest.mark.skipif(not shutil.which("swift"), reason="needs a Swift toolchain")

TEXTS = [
    "Hello world", "  two leading spaces", "trailing spaces   ", "tabs\tand\nnewlines\n\n  indented",
    "don't we'll they're I'm you've he'd it's", "shouting DON'T 'quoted' '' 'S", "numbers 123 4567 3.14 1e-05 ½ ²",
    "punctuation!!! ...?? (brackets) [x] {y} <z> -- -> => ::", "émoji 🚀🚀 café naïve Ünïcödé", "日本語のテキスト 中文 한국어",
    "mixed123abc 12ab 1,000,000", "<end_of_utterance>inside<|im_start|>text<image><image>", " <image> spaced ",
    " non-breaking em space　ideographic", "a\r\nwindows\rline", "x" * 300, "",
    "user@example.com https://example.org/a?b=c&d=e#f", "C'est l'été, n'est-ce pas?", "\t\t\t", "a  b   c    d",
]


@pytest.fixture(scope="module")
def check_bin():
    r = subprocess.run(["swift", "build", "-c", "release", "--package-path", PKG], capture_output=True, text=True,
                       timeout=1200)
    assert r.returncode == 0, r.stdout + r.stderr
    path = subprocess.run(["swift", "build", "-c", "release", "--package-path", PKG, "--show-bin-path"],
                          capture_output=True, text=True, timeout=300).stdout.strip()
    return os.path.join(path, "laya-core-check")


def test_swift_ids_match_fixture(check_bin):
    tok = os.path.join(_tokenizer_dir(), "tokenizer.json")
    r = subprocess.run([check_bin, "ids", FIXTURE, tok], capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "rows match" in r.stdout


def test_swift_tokenizer_matches_python(check_bin, tmp_path):
    from transformers import AutoTokenizer

    tok_dir = _tokenizer_dir()
    ref = AutoTokenizer.from_pretrained(BACKBONE)
    texts = tmp_path / "texts.json"
    texts.write_text(json.dumps(TEXTS))
    r = subprocess.run([check_bin, "encode", str(texts), os.path.join(tok_dir, "tokenizer.json")], capture_output=True,
                       text=True, timeout=300)
    assert r.returncode == 0, r.stdout + r.stderr
    got = [json.loads(line) for line in r.stdout.splitlines()]
    for text, ids in zip(TEXTS, got):
        assert ids == ref(text, add_special_tokens=False)["input_ids"], text
    assert len(got) == len(TEXTS)


def test_swift_pixels_match_processor(check_bin, tmp_path):
    from PIL import Image

    proc, prep = _processor()
    for name, arr in _test_images().items():
        raw = tmp_path / (name + ".rgb")
        raw.write_bytes(arr.tobytes())
        out = tmp_path / (name + ".out")
        r = subprocess.run([check_bin, "pixels", FIXTURE, str(raw), str(arr.shape[0]), str(arr.shape[1]), str(out)],
                           capture_output=True, text=True, timeout=600)
        assert r.returncode == 0, r.stdout + r.stderr
        got = np.frombuffer(out.read_bytes(), dtype=np.float32).reshape(3, prep.image_size, prep.image_size)
        ref = proc(text=[proc.image_token], images=[[Image.fromarray(arr)]], do_image_splitting=False,
                   return_tensors="pt")["pixel_values"][0, 0].numpy()
        levels = np.abs(got - ref) * 127.5
        print("%s: mean %.3f max %.1f grey levels" % (name, levels.mean(), levels.max()))
        assert levels.mean() < 0.05 and levels.max() <= 3.0, name


@pytest.mark.skipif(not os.environ.get("LAYA_ONNX_DIR"), reason="set LAYA_ONNX_DIR to a scripts/export_onnx.py export")
def test_swift_predictor_matches_pytorch(check_bin, tmp_path):
    """The whole Swift pipeline (tokenizer, sequence, resize, the loop over questions and option orders, answers) on
    the exported fp32 graphs, run by onnxruntime through ios/tools/ort_server.py, against ``VLMAgent.predict``."""
    import sys

    from PIL import Image

    from laya.vlm import VLMAgent

    out_dir = os.environ["LAYA_ONNX_DIR"]
    with open(os.path.join(out_dir, "laya_web.json")) as f:
        web = json.load(f)
    agent = VLMAgent(web["source"], device="cpu", dtype="fp32", revision=web["checkpoint"].get("revision"))
    img = Image.open(os.path.join(ROOT, "site-docs", "tutorials", "example.jpg")).convert("RGB")
    arr = np.asarray(img)
    questions = {
        "damage": {"type": "score", "instructions": "How much damage does the item show?",
                   "criteria": ["none", "cosmetic: scratches or dents", "functional: parts broken or missing", "destroyed"]},
        "category": {"type": "choice", "instructions": "What kind of item is this?",
                     "criteria": ["electronics", "clothing", "furniture", "food", "other"]},
        "outdoors": {"type": "noul", "instructions": "Was the photo taken outdoors?"},
        "count": {"type": "choice", "instructions": "How many turntables are there?", "criteria": {"1": "one", "2": "two", "3": None}},
    }
    state = {"note": "customer says it arrived broken", "order": 12345}
    (tmp_path / "img.rgb").write_bytes(arr.tobytes())
    (tmp_path / "q.json").write_text(json.dumps(questions))
    (tmp_path / "s.json").write_text(json.dumps(state))
    proc, _ = _processor()
    px = proc(text=[proc.image_token], images=[[img]], do_image_splitting=False, return_tensors="pt")["pixel_values"]
    (tmp_path / "px.f32").write_bytes(px[0, 0].numpy().astype(np.float32).tobytes())
    refs = {n: agent.predict(dict(state, image=img), questions, n_permutations=n) for n in (1, 2)}
    # With the processor's own pixels everything else must match predict up to its 4-place rounding; with the Swift
    # resize (about 1.4% of pixels one grey level off on this photo) probabilities move by up to ~0.006.
    for pixels, tol in (("processor", 1e-3), ("swift", 1e-2)):
        env = dict(os.environ, LAYA_PIXELS=str(tmp_path / "px.f32")) if pixels == "processor" else None
        r = subprocess.run([check_bin, "predict", out_dir, str(tmp_path / "img.rgb"), str(arr.shape[0]), str(arr.shape[1]),
                            str(tmp_path / "q.json"), str(tmp_path / "s.json"), "", sys.executable],
                           capture_output=True, text=True, timeout=900, env=env)
        assert r.returncode == 0, r.stdout + r.stderr
        print(r.stderr)
        got = [json.loads(line) for line in r.stdout.splitlines()]
        for n, swift in zip([1, 2], got):
            for qid, a in refs[n]["answers"].items():
                b = swift["answers"][qid]
                assert a["type"] == b["type"]
                if a["type"] == "noul":
                    pa, pb = [a["noul"]], [b["noul"]]
                else:
                    pa, pb = list(a["probabilities"].values()), list(b["probabilities"].values())
                    assert list(a["probabilities"]) == list(b["probabilities"]), qid
                diff = np.abs(np.array(pa) - np.array(pb)).max()
                print("%-9s pixels, orders=%d %-9s max |dp| %.4f" % (pixels, n, qid, diff))
                assert diff < tol, (pixels, n, qid, a, b)
            assert a.get("choice") == b.get("choice"), (n, qid)
