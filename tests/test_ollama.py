"""laya.ollama: Ollama's /v1/systemone prompts, checked against Ollama's own Go compiler.

tests/fixtures/ollama_systemone_compiled.jsonl was written by ollama/ollama's ``decision.Compile`` (the main branch
at 1abe35e) from ollama_systemone_requests.jsonl, one line per request.
"""
import json
import os

import pytest
from PIL import Image

from laya.ollama import (GO_TEMPLATE, SYSTEM_PROMPT, compile_request, go_json, image_views, input_ids, letter_ids,
                         prompts, render_prompt)

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def load(name):
    with open(os.path.join(FIXTURES, name)) as f:
        return [json.loads(line) for line in f if line.strip()]


@pytest.mark.parametrize("i", range(len(load("ollama_systemone_requests.jsonl"))))
def test_compile_matches_ollama_byte_for_byte(i):
    req = load("ollama_systemone_requests.jsonl")[i]
    want = load("ollama_systemone_compiled.jsonl")[i]
    got = compile_request(req["state"], req["questions"])
    assert [u for u, _ in got] == want["users"]
    assert [c for _, c in got] == want["candidates"]


def test_go_json_escapes_like_go():
    assert go_json({"a": "<b> &   é"}) == '{"a":"\\u003cb\\u003e \\u0026 \\u2028 é"}'
    assert go_json([False, True, "0"]) == '[false,true,"0"]'


def test_rendered_prompt_and_template_agree():
    p = render_prompt('{"x":1}\n\nRequested field: "q"', n_images=2)
    assert p == ("<|im_start|>System: " + SYSTEM_PROMPT + "<end_of_utterance>\n"
                 'User:[img-0][img-1]{"x":1}\n\nRequested field: "q"<end_of_utterance>\nAssistant:')
    # the Go template's literal text, with its two actions filled the same way
    assert GO_TEMPLATE.startswith("<|im_start|>") and GO_TEMPLATE.endswith("{{ end }}{{ end }}Assistant:")
    assert render_prompt("u", system=None) == "<|im_start|>User:u<end_of_utterance>\nAssistant:"


def test_invalid_requests():
    with pytest.raises(ValueError):
        compile_request("", {"q": {"type": "noul", "instructions": "x"}})
    with pytest.raises(ValueError):
        compile_request("s", {"q": {"type": "choice", "instructions": "x", "criteria": {"a": None}}})
    with pytest.raises(ValueError):
        compile_request("s", {"q": {"type": "noul", "instructions": " "}})


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("thaitea/laya-vision", subfolder="processor",
                                         revision="f2fe3c12cb6d04c59d8a190250bf3fb40fc828dc")


def test_letters_append_one_token(tokenizer):
    """Ollama refuses a candidate unless prompt+letter tokenizes as the prompt's tokens plus that letter."""
    (prompt, letters), = prompts("s", {"q": {"type": "score", "instructions": "x", "criteria": list("abcdefghijklmnopqrstuvwxyz")}})
    base = tokenizer.encode(prompt, add_special_tokens=False)
    ids = letter_ids(tokenizer, letters)
    for c, i in zip(letters, ids):
        assert tokenizer.encode(prompt + c, add_special_tokens=False) == base + [i]
    assert len(set(ids)) == 26


def test_image_tokens_follow_mtmd(tokenizer):
    (prompt, _), = prompts({"note": "hi"}, {"q": {"type": "noul", "instructions": "red?"}}, n_images=1)
    ids = input_ids(tokenizer, prompt, 1)
    img = tokenizer.convert_tokens_to_ids("<image>")
    assert ids.count(img) == 128
    first = ids.index(img)
    tok = tokenizer.convert_ids_to_tokens
    assert tok(ids[first - 2:first]) == ["<fake_token_around_image>", "<row_1_col_1>"]
    mid = ids[first + 64:first + 67]
    assert tok(mid)[1:] == ["<fake_token_around_image>", "<global-img>"] and tokenizer.decode(mid[:1]) == "\n\n"
    assert tok(ids[first + 131:first + 132]) == ["<fake_token_around_image>"]
    text_only = input_ids(tokenizer, prompt.replace("[img-0]", ""), 0)
    assert ids[:first - 2] == text_only[:first - 2]  # "<|im_start|>System: ... User:" is unchanged
    assert img not in text_only
    views = image_views(Image.new("RGB", (300, 120), (255, 0, 0)))
    assert [v.size for v in views] == [(512, 512), (512, 512)]


def _ex(i, image, q, label, text=None, dataset="toy"):
    from laya.vlm_train import jsonl_example

    rec = {"id": str(i), "question": q, "label": label}
    if image:
        rec["image"] = image
    if text:
        rec["state_text"] = text
    return jsonl_example(rec, "/imgs", dataset)


def test_requests_group_questions_about_one_image():
    import random

    from laya.ollama_train import requests_from

    color = {"type": "choice", "instructions": "What color?", "criteria": ["red", "blue", "green", "black"]}
    red = {"type": "noul", "instructions": "Is it red?"}
    exs = [_ex(0, "a.png", color, 2), _ex(1, "a.png", red, 0), _ex(2, "b.png", red, 1),
           _ex(3, None, {"type": "score", "instructions": "Urgent?", "criteria": ["no", "yes"]}, 1, text="refund")]
    rows = requests_from(exs, random.Random(0), p_single=0.0, max_questions=4)
    assert len(rows) == 4
    by_image = {}
    for r in rows:
        by_image.setdefault(tuple(r["images"]), []).append(r)
        q = r["questions"][r["name"]]
        if q["type"] == "choice":  # options shuffled; the target follows its option
            assert list(q["criteria"])[r["label"]] == "green" and sorted(q["criteria"]) == ["black", "blue", "green", "red"]
        if q["type"] == "noul":
            assert r["label"] == (0 if r["images"] == ["/imgs/a.png"] else 1)
    a = by_image[("/imgs/a.png",)]
    assert len(a) == 2 and a[0]["questions"] == a[1]["questions"] and len(a[0]["questions"]) == 2
    text_row = by_image[()][0]
    assert text_row["state"] in ("refund", {"context": "refund"})
    assert by_image[("/imgs/b.png",)][0]["state"] in __import__("laya.ollama_train", fromlist=["x"]).STATES_WITHOUT_TEXT
    # every row compiles into an Ollama prompt
    for r in rows:
        assert prompts(r["state"], r["questions"], n_images=len(r["images"]))


def test_val_rows_keep_option_order():
    import random

    from laya.ollama_train import requests_from

    q = {"type": "choice", "instructions": "Pick", "criteria": ["w", "x", "y", "z"]}
    rows = requests_from([_ex(i, "%d.png" % i, q, i % 4) for i in range(8)], random.Random(1), p_single=1.0, shuffle=False)
    assert all(list(r["questions"][r["name"]]["criteria"]) == ["w", "x", "y", "z"] for r in rows)
    assert sorted(int(r["id"]) % 4 == r["label"] for r in rows) == [True] * 8
