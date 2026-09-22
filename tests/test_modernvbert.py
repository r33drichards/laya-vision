"""Tests for the ModernVBERT (bidirectional, ``[MASK]`` readout) backbone. Downloads ModernVBERT/modernvbert (~1 GB).

Shares fixtures and questions with ``test_vlm.py``; that file covers what is backbone-independent (the
preprocessing operators, the feature cache, the prepared-dataset format). Here: the sequence, the readout, the
freezing stages, save/load and latency for the second family.
"""
import math
import random
import time

import pytest
import torch
from test_vlm import DEVICE, QUESTIONS, check_schema, frame, square, sync

from laya.common import render_options
from laya.preprocess import ImagePrep, prefix_ids
from laya.vlm import (
    MASK_PREFIX_TEXT,
    MODERNVBERT_BACKBONE,
    VLMAgent,
    VLMDecisionModel,
    build_vlm_inputs,
    collate_vlm,
    processor_readout,
    readout_for,
    set_trainable,
    vlm_prefix,
)
from laya.vlm_train import collect_logits, cyclic_orders, metrics_from, synthetic_examples, train


@pytest.fixture(scope="module")
def agent():
    torch.manual_seed(0)
    return VLMAgent(backbone=MODERNVBERT_BACKBONE, device=DEVICE)


class _Cfg:
    def __init__(self, model_type):
        self.model_type = model_type


def test_readout_follows_the_backbone(agent):
    """The family is decided by the backbone's model_type, recorded in the config and left on the processor."""
    assert readout_for(_Cfg("modernvbert")) == "mask"
    assert readout_for(_Cfg("idefics3")) == readout_for(_Cfg("smolvlm")) == "terminator"
    assert agent.model.readout == agent.cfg["readout"] == processor_readout(agent.processor) == "mask"
    assert agent.model.encoder.config.model_type == "modernvbert"
    with pytest.raises(ValueError):
        VLMDecisionModel(agent.model.encoder, readout="mask", option_attention="block")
    with pytest.raises(ValueError):
        VLMDecisionModel(agent.model.encoder, readout="eos")


def test_predict_image_and_text(agent):
    for state in ({"image": square((220, 20, 20)), "caption": "a test card"}, {"images": [square((220, 20, 20)), square((20, 40, 220))]}):
        res = agent.predict(state, QUESTIONS)
        check_schema(res, QUESTIONS)
        assert res["usage"]["images"] == len(state.get("images", [None]))
    res = agent.predict("Customer: I was billed twice, please refund.", QUESTIONS)
    check_schema(res, QUESTIONS)
    assert res["usage"]["images"] == 0
    check_schema(agent.predict({"image": square((20, 40, 220))}, QUESTIONS, n_permutations=3), QUESTIONS)


def test_sequence_format(agent):
    """[CLS]User:<image run> <type> question: <ins>[SEP][MASK] opt0[MASK] opt1 ...[SEP]<state>[SEP]"""
    proc = agent.processor
    tok = proc.tokenizer
    for state, n_img in (({"image": square((0, 0, 255)), "note": "x"}, 1), ("plain text state", 0),
                         ({"images": [frame(), frame(1)]}, 2)):
        for qdef in QUESTIONS.values():
            q = VLMAgent._to_internal(qdef)
            opts = render_options(q)
            order = list(range(len(opts)))
            random.Random(1).shuffle(order)
            it = build_vlm_inputs(proc, state, q, option_order=order)
            ids, markers = it["ids"], it["markers"]
            assert it["n_images"] == n_img and ids.count(proc.image_token_id) == 64 * n_img
            assert ids[0] == tok.cls_token_id and ids[-1] == tok.sep_token_id
            assert len(markers) == len(opts)
            start, end = it["option_span"]
            assert markers[0] == start and ids[start - 1] == tok.sep_token_id and ids[end] == tok.sep_token_id
            bounds = markers + [end]
            for j, m in enumerate(markers):
                assert ids[m] == tok.mask_token_id
                assert tok.decode(ids[m + 1 : bounds[j + 1]]) == " " + opts[order[j]]
            head = tok.decode(ids[: start - 1])
            assert head.startswith("[CLS]User:") and head.endswith("%s question: %s" % (q["t"], qdef["instructions"]))
            tail = tok.decode(ids[end + 1 :])
            if isinstance(state, str):
                assert tail == state + "[SEP]"  # the state follows the options: only a bidirectional readout can use it
            elif "note" in state:
                assert tail == '{"note": "x"}[SEP]'
            else:
                assert tail == ""


def test_prefix_ids_match_the_processor(agent):
    """The device-side path writes the same ``[CLS]User:<image run>`` the processor does, minus the pixels."""
    proc = agent.processor
    out = proc(text=[MASK_PREFIX_TEXT + proc.image_token], images=[[frame()]], do_image_splitting=False,
               return_tensors="pt", add_special_tokens=False)
    assert prefix_ids(proc, MASK_PREFIX_TEXT, 1, 64) == out["input_ids"][0].tolist()
    gpu = vlm_prefix(proc, [frame()], ImagePrep(backend="gpu"))
    cpu = vlm_prefix(proc, [frame()], ImagePrep(backend="processor"))
    assert gpu["ids"] == cpu["ids"] == out["input_ids"][0].tolist()
    assert gpu["raw_images"].shape == (1, 3, 210, 160) and tuple(cpu["pixel_values"].shape) == (1, 3, 512, 512)
    assert vlm_prefix(proc, [])["ids"] == proc.tokenizer(MASK_PREFIX_TEXT, add_special_tokens=False)["input_ids"]


@torch.no_grad()
def test_markers_see_the_state_after_them(agent):
    """The state text comes after the option block; a causal readout could not react to it, this one must."""
    q = VLMAgent._to_internal(QUESTIONS["is_red"])
    items = [dict(build_vlm_inputs(agent.processor, "The square is %s." % c, q), qtype=2) for c in ("red", "blue")]
    assert items[0]["markers"] == items[1]["markers"]  # same positions, only the trailing state differs
    b = collate_vlm(items, agent.processor.tokenizer.pad_token_id)
    logits, _ = agent.model(*(b[k].to(agent.device) for k in ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")))
    assert float((logits[0] - logits[1]).abs().max()) > 1e-4


def test_train_head_only(agent):
    model = agent.model
    enc_names = ["vision_model.embeddings.patch_embedding.weight", "connector.modality_projection.weight",
                 "text_model.embeddings.tok_embeddings.weight", "text_model.layers.21.mlp.Wo.weight", "text_model.final_norm.weight"]
    enc_params = dict(model.encoder.named_parameters())
    enc_before = {n: enc_params[n].detach().clone() for n in enc_names}
    head_before = {n: p.detach().clone() for n, p in model.named_parameters() if not n.startswith("encoder.")}

    losses = train(model, agent.processor, synthetic_examples(4), steps=3, batch_size=2, freeze="head", device=DEVICE)

    assert len(losses) == 3 and all(math.isfinite(x) for x in losses)
    assert all(p.grad is None for p in model.encoder.parameters())
    for n in enc_names:
        assert torch.equal(enc_params[n], enc_before[n]), n
    head_now = dict(model.named_parameters())
    changed = [n for n, v in head_before.items() if not torch.equal(head_now[n], v)]
    assert any(n.startswith("scorer.") for n in changed)
    assert any(n.startswith("head.") for n in changed)
    assert not model.training


def test_freezing_stages_find_the_modernbert_tower():
    """``last_n`` must reach ``text_model.layers`` and ``final_norm`` here (SmolVLM's norm is called ``norm``)."""
    torch.manual_seed(0)
    a = VLMAgent(backbone=MODERNVBERT_BACKBONE, device=DEVICE)  # fresh: this trains backbone layers
    enc = a.model.encoder
    n_layers = len(enc.text_model.layers)
    set_trainable(a.model, "last_n", n_last=2)
    on = {n for n, p in enc.named_parameters() if p.requires_grad}
    assert "text_model.final_norm.weight" in on
    assert all(n.startswith(("text_model.layers.%d." % (n_layers - 1), "text_model.layers.%d." % (n_layers - 2))) or n == "text_model.final_norm.weight" for n in on)
    assert not any(n.startswith("vision_model.") for n in on)
    n_full = set_trainable(a.model, "full")
    assert not any(p.requires_grad for p in enc.vision_model.parameters())
    assert n_full > sum(p.numel() for n, p in a.model.named_parameters() if not n.startswith("encoder."))
    losses = train(a.model, a.processor, synthetic_examples(4), steps=1, batch_size=2, freeze="last_n", n_last=2, device=DEVICE)
    assert all(math.isfinite(x) for x in losses)


@pytest.mark.parametrize("include_backbone", [True, False])
def test_save_load_roundtrip(agent, tmp_path, include_backbone):
    agent.temperature = [1.3, 0.8, 1.1]
    state = {"image": square((220, 20, 20)), "caption": "a test card"}
    before = [agent.predict(state, QUESTIONS), agent.predict("plain text", QUESTIONS)]
    agent.save(str(tmp_path), include_backbone=include_backbone)
    loaded = VLMAgent(str(tmp_path), device=DEVICE)
    assert loaded.cfg["readout"] == loaded.model.readout == processor_readout(loaded.processor) == "mask"
    assert loaded.cfg["backbone"] == MODERNVBERT_BACKBONE
    after = [loaded.predict(state, QUESTIONS), loaded.predict("plain text", QUESTIONS)]
    assert after == before
    assert loaded.cfg["temperature"] == [1.3, 0.8, 1.1]


def test_eval_path(agent):
    """collect_logits / metrics_from run on this readout, with option orders permuted (harmless here)."""
    ex = [dict(e, dataset="synthetic") for e in synthetic_examples(2)]
    records = collect_logits(agent.model, agent.processor, ex, batch_size=3, orders=cyclic_orders(2))
    assert len(records) == len(ex) and all(len(r["logits_per_order"]) == min(2, len(r["target"])) for r in records)
    m = metrics_from(records)
    assert m["all"]["n"] == len(ex) and 0.0 <= m["all"]["acc"] <= 1.0 and math.isfinite(m["all"]["nll"])


def test_train_and_predict_at_256(tmp_path):
    """ModernVBERT's SigLIP tower has a fixed 512-pixel position grid; smaller tiles need it interpolated."""
    torch.manual_seed(0)
    a = VLMAgent(backbone=MODERNVBERT_BACKBONE, device=DEVICE, image_size=256, preprocess="gpu")
    assert a.model._vision_kw == {"interpolate_pos_encoding": True}
    assert a.prep.image_seq_len == 16 and a.processor.image_seq_len == 16
    assert a.model.encode_raw_images([frame(), frame(1)]).shape[:2] == (2, 16)
    losses = train(a.model, a.processor, synthetic_examples(4), steps=2, batch_size=2, freeze="head", device=DEVICE)
    assert all(math.isfinite(x) for x in losses)
    state = {"images": [square((220, 20, 20)), square((20, 40, 220))]}
    res = a.predict(state, QUESTIONS)
    check_schema(res, QUESTIONS)
    ids = build_vlm_inputs(a.processor, state, VLMAgent._to_internal(QUESTIONS["color"]), prep=a.prep)["ids"]
    assert sum(i == a.processor.image_token_id for i in ids) == 32  # 2 images x 16, not 2 x 64
    a.save(str(tmp_path))
    loaded = VLMAgent(str(tmp_path), device=DEVICE)
    assert loaded.cfg["image_size"] == 256 and loaded.prep.backend == "gpu" and loaded.model.readout == "mask"
    assert loaded.predict(state, QUESTIONS) == res


def test_latency(agent):
    q = {"is_red": QUESTIONS["is_red"]}
    for name, state in (("image", {"image": square((220, 20, 20))}), ("text", "Customer: I was billed twice, please refund.")):
        agent.predict(state, q)  # warm-up
        ts = []
        for _ in range(5):
            t0 = time.perf_counter()
            agent.predict(state, q)
            sync()
            ts.append((time.perf_counter() - t0) * 1000)
        ts.sort()
        print("\nmodernvbert latency %s state, 1 noul question, %s: median %.1f ms (min %.1f, max %.1f)" % (name, DEVICE, ts[2], ts[0], ts[-1]))


def test_mask_builder_and_text_sequence_report_what_they_cut(agent):
    """The ``[MASK]`` sequence and the text-only ``build_sequence`` it copies report the same kinds of cut."""
    from test_vlm import LONG, NO_CUT, TRUNCATION_QUESTIONS

    from laya.common import build_sequence, truncation_answer

    proc, tok = agent.processor, agent.processor.tokenizer
    for qid, check in (
        ("long_option", lambda t: t == dict(NO_CUT, options=[1])),
        ("same_after_cut", lambda t: t["options"] == [0, 1] and t["indistinguishable"] == [[0, 1]]),
        ("many_options", lambda t: t["options"] == list(range(30)) and not t["indistinguishable"]),
        ("long_instructions", lambda t: t["instructions_tokens_dropped"] > 100 and not t["options"]),
    ):
        q = VLMAgent._to_internal(TRUNCATION_QUESTIONS[qid])
        order = list(reversed(range(len(render_options(q)))))
        assert check(build_vlm_inputs(proc, "x", q, option_order=order)["truncation"]), qid
        report = {}
        build_sequence(tok, "x", q, 512, 192, option_order=order, report=report)
        assert check(report), qid
    q = VLMAgent._to_internal(QUESTIONS["is_red"])
    state = {"image": square((0, 0, 255)), "note": "word " * 3000}
    for left in (False, True):
        it = build_vlm_inputs(proc, state, q, truncate_left=left)
        assert it["truncation"]["state_tokens_dropped"] > 1500 and len(it["ids"]) == 1024
        report = {}
        ids, _ = build_sequence(tok, state, q, 512, 192, truncate_left=left, report=report)
        assert report["state_tokens_dropped"] > 2000 and len(ids) == 512
    report = {}
    build_sequence(tok, "a short state", VLMAgent._to_internal(QUESTIONS["color"]), report=report)
    assert report == NO_CUT and truncation_answer(report, VLMAgent._to_internal(QUESTIONS["color"])) is None
    t = agent.predict({"image": square((0, 0, 255))}, {"s": TRUNCATION_QUESTIONS["same_after_cut"]})["answers"]["s"]
    assert t["truncated"]["indistinguishable"] == [[LONG + "alpha", LONG + "beta"]]
    with pytest.raises(ValueError, match="'s' would be truncated"):
        agent.predict("x", {"s": TRUNCATION_QUESTIONS["same_after_cut"]}, strict=True)
