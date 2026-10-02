"""``laya.prompt``, the shared (state, question) -> text module.

The first tests need nothing but the standard library. The ``processor`` tests load the SmolVLM-256M processor
(tokenizer and image processor, no weights) and pin the input ids built for a battery of states and questions
through the three paths that build them (``build_vlm_inputs`` as ``predict`` calls it, and ``jsonl_example`` ->
``make_item`` as training and every eval call it) to a sha256 recorded before ``laya.prompt`` existed: the default
rendering must keep every released checkpoint's input ids (AGENTS.md: a change to them bumps
``PROMPT_FORMAT_VERSION``)."""
import json
import os
import random

import pytest
from PIL import Image

from laya import prompt as P
from laya.agent import Agent
from laya.cauldron import parse_turn
from laya.common import render_options, serialize_state

# ---------------------------------------------------------------------------------------------------------
# Pure text
# ---------------------------------------------------------------------------------------------------------

STATES = [
    "plain text",
    "",
    {"context": "Lecture: two lines\nsecond \"quoted\" line, café, 東京"},
    {"note": "customer says it arrived broken", "order": {"id": 17, "items": ["mug", "plate"], "paid": True}},
    {"a": None, "b": 1.5, "c": [], "d": {}, "e": [{"x": 1}, [2, 3]]},
    [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}],
    ["one", 2, False],
]


def test_json_default_is_the_old_serialization():
    for st in STATES:
        old = st if isinstance(st, str) else json.dumps(st, ensure_ascii=False)
        assert serialize_state(st) == old
        assert P.serialize_state(st, "json") == old
        assert P.serialize_state(st, None) == old


def test_to_text_matches_clm():
    """Byte-for-byte the rendering of ``clm.schema.to_text`` (CLM repository) on these inputs."""
    assert P.to_text({"note": "hi", "order": {"id": 17, "items": ["mug", "plate"], "paid": True}}) == (
        "note: hi\n\norder:\n  id: 17\n  items:\n    - mug\n    - plate\n  paid: true")
    assert P.to_text(["one", 2, False, None]) == "- one\n- 2\n- false\n- "
    assert P.to_text({"a": None, "c": [], "d": {}}) == "a: \n\nc: \n\nd: "
    assert P.to_text([{"x": 1}, [2, 3]]) == "-\n  x: 1\n-\n  - 2\n  - 3"
    assert P.to_text("as is\n") == "as is\n" and P.to_text(None) == ""


def test_prose_and_text_formats():
    st = {"context": "Question: q?\n\nResponse: r"}
    assert P.serialize_state(st, "prose") == "context: Question: q?\n\nResponse: r"
    assert P.serialize_state(st, "text") == "Question: q?\n\nResponse: r"
    assert P.serialize_state({"a": "x", "b": {"c": 1}}, "text") == "x\n\nc: 1"
    assert P.serialize_state("raw", "prose") == P.serialize_state("raw", "text") == "raw"
    with pytest.raises(ValueError):
        P.serialize_state(st, "yaml")


def _old_to_internal(qdef):  # VLMAgent._to_internal / Agent._to_internal before laya.prompt
    t = qdef["type"]
    crit = qdef.get("criteria")
    if t == "choice" and isinstance(crit, list):
        crit = {c: None for c in crit}
    ins = qdef["instructions"]
    if not isinstance(ins, str):
        ins = json.dumps(ins)
    return {"t": t, "ins": ins, "crit": crit}


QDEFS = [
    {"type": "choice", "instructions": "Which?", "criteria": ["a", "B", "a"]},
    {"type": "choice", "instructions": "Which?", "criteria": {"UP": "move up", "DOWN": None}},
    {"type": "score", "instructions": "How good?", "criteria": ["bad", "ok", "good"]},
    {"type": "noul", "instructions": "Is it red?"},
    {"type": "noul", "instructions": {"claim": "red", "strict": True}, "criteria": {"true": "red", "false": "not red"}},
]


def test_question_to_internal_is_shared_and_unchanged():
    from laya.vlm import VLMAgent

    assert VLMAgent._to_internal is P.question_to_internal or VLMAgent._to_internal(QDEFS[0]) == P.question_to_internal(QDEFS[0])
    for qd in QDEFS:
        assert P.question_to_internal(qd) == _old_to_internal(qd) == VLMAgent._to_internal(qd) == Agent._to_internal(qd)


def _old_record_state(rec, root):  # jsonl_example's inline state before laya.prompt
    state = {}
    if rec.get("image"):
        state["image"] = os.path.join(root, rec["image"])
    elif rec.get("images"):
        state["images"] = [os.path.join(root, p) for p in rec["images"]]
    if rec.get("state_text"):
        state["context"] = rec["state_text"]
    return state or ""


def test_record_state_and_make_state():
    recs = [{"image": "i/a.jpg", "state_text": "Prompt: x"}, {"images": ["a", "b"]}, {"state_text": "t"}, {},
            {"image": "i/a.jpg", "state_text": None}, {"image": "", "images": ["c"], "state_text": ""}]
    for rec in recs:
        assert P.record_state(rec, "/root") == _old_record_state(rec, "/root")
    assert P.make_state(image="img", context="Prompt: x") == {"image": "img", "context": "Prompt: x"}
    assert P.make_state(images=["a", "b"]) == {"images": ["a", "b"]}
    assert P.make_state() == ""


def test_option_normalisation():
    assert P.normalize_option_text("  Cab ", "lower") == "cab"
    assert P.normalize_option_text("drive-in movie", "title") == "Drive-in movie"
    assert P.normalize_option_text(" x ") == "x"
    q = {"t": "choice", "ins": "?", "crit": {"Cab": None, "train": None}}
    assert list(P.normalize_choice_options(q, "lower")["crit"]) == ["cab", "train"]
    assert list(P.normalize_choice_options(q, "title")["crit"]) == ["Cab", "Train"]
    clash = {"t": "choice", "ins": "?", "crit": {"Cab": None, "cab": None}}
    assert P.normalize_choice_options(clash, "lower") is clash
    # the A-OKVQA preparation still lower-cases through laya.prompt
    p = parse_turn("What is it?\nPick one.\nOptions: Skateboarder, Train , delivery, cab.", "Cab.")
    assert p["question"]["criteria"] == ["skateboarder", "train", "delivery", "cab"] and p["label"] == 3


def test_render_options_is_the_common_one():
    from laya import common

    assert common.render_options is P.render_options and common.option_labels is P.option_labels
    assert render_options(P.question_to_internal(QDEFS[3])) == ["false: no, the statement does not hold",
                                                                "true: yes, the statement holds"]


# ---------------------------------------------------------------------------------------------------------
# Input ids (SmolVLM processor, no weights)
# ---------------------------------------------------------------------------------------------------------

#: sha256 (``laya.vlm.input_ids_sha256``) of ``battery_ids``' rows, computed with the code before ``laya.prompt``
#: (commit d4075b0). Changing it means the default rendering changed: bump ``PROMPT_FORMAT_VERSION``.
GOLDEN_IDS_SHA256 = "6e9834ab50ffd4fa1a9c9529570ef95786ee661063c37f2e9ea660ecac655297"


def _png(path, color):
    Image.new("RGB", (40, 30), color).save(path)
    return path


def battery_ids(vlm, vlm_train, processor, tmp):
    """Token-id rows for every (state, question, option order) in the battery through ``build_vlm_inputs`` (as
    ``predict`` builds them, and with the image prefix built once as ``predict`` passes it) and through
    ``jsonl_example`` -> ``make_item`` (training / eval). Uses only APIs that existed before ``laya.prompt``."""
    img = _png(os.path.join(tmp, "a.png"), (200, 30, 30))
    img2 = _png(os.path.join(tmp, "b.png"), (30, 30, 200))
    states = [s for s in STATES if not isinstance(s, str) or s] + [
        {"image": img}, {"image": img, "context": "Lecture: two lines\nsecond \"quoted\" line, café"},
        {"images": [img, img2], "note": "pair"}, {"image": img, "context": "word " * 1500}]
    rows = []
    for st in states:
        for qd in QDEFS:
            q = vlm.VLMAgent._to_internal(qd)
            k = len(render_options(q))
            for order in (None, list(reversed(range(k)))):
                rows.append(vlm.build_vlm_inputs(processor, st, q, 1024, 256, option_order=order)["ids"])
            images, _ = vlm.split_state(st)
            prefix = vlm.vlm_prefix(processor, images)
            rows.append(vlm.build_vlm_inputs(processor, st, q, 1024, 256, prefix=prefix)["ids"])
    recs = [{"id": "r0", "image": "a.png", "state_text": "Question: q?\n\nResponse: \"r\" é",
             "question": {"type": "score", "instructions": "Rate it.", "criteria": ["bad", "ok", "good"]}, "label": 1},
            {"id": "r1", "images": ["a.png", "b.png"], "state_text": None,
             "question": {"type": "noul", "instructions": "Same colour?", "criteria": None}, "label": 0},
            {"id": "r2", "state_text": "Hint only",
             "question": {"type": "choice", "instructions": "Pick", "criteria": ["x", "Y", "z"]}, "label": 2}]
    for rec in recs:
        ex = vlm_train.jsonl_example(rec, tmp, "d")
        k = len(ex["target"])
        for order in (list(range(k)), list(reversed(range(k)))):
            rows.append(vlm_train.make_item(processor, ex, random.Random(0), shuffle=False, order=order)["ids"])
    return rows


@pytest.fixture(scope="module")
def processor():
    from transformers import AutoProcessor

    from laya.vlm import DEFAULT_BACKBONE

    return AutoProcessor.from_pretrained(DEFAULT_BACKBONE)


def test_default_rendering_keeps_the_input_ids(processor, tmp_path):
    from laya import vlm, vlm_train

    rows = battery_ids(vlm, vlm_train, processor, str(tmp_path))
    assert vlm.input_ids_sha256(rows) == GOLDEN_IDS_SHA256


def test_state_format_changes_only_the_state_text(processor, tmp_path):
    from laya.vlm import build_vlm_inputs, split_state, vlm_prefix

    img = _png(str(tmp_path / "a.png"), (200, 30, 30))
    st = {"image": img, "context": "Question: q?\n\nResponse: r"}
    q = P.question_to_internal(QDEFS[3])
    tok = processor.tokenizer
    base = build_vlm_inputs(processor, st, q)
    assert build_vlm_inputs(processor, st, q, state_format="json")["ids"] == base["ids"]
    for fmt, text in (("prose", "context: Question: q?\n\nResponse: r"), ("text", "Question: q?\n\nResponse: r")):
        assert split_state(st, fmt)[1] == text
        it = build_vlm_inputs(processor, st, q, state_format=fmt)
        assert it["ids"] != base["ids"]
        n_old = len(tok(json.dumps({"context": st["context"]}), add_special_tokens=False)["input_ids"])
        n_new = len(tok(text, add_special_tokens=False)["input_ids"])
        # the image prefix and the question/option tail are unchanged; only the state tokens in between differ
        n_prefix = len(vlm_prefix(processor, split_state(st)[0])["ids"])
        assert it["ids"][:n_prefix] == base["ids"][:n_prefix]
        tail = len(base["ids"]) - n_prefix - n_old
        assert it["ids"][-tail:] == base["ids"][-tail:] and len(it["ids"]) == len(base["ids"]) - n_old + n_new
    # the processor carries a checkpoint's choice, as it carries max_len
    processor.laya_state_format = "prose"
    try:
        assert build_vlm_inputs(processor, st, q)["ids"] == build_vlm_inputs(processor, st, q, state_format="prose")["ids"]
    finally:
        del processor.laya_state_format

