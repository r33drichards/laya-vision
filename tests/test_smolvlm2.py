"""Tests for the SmolVLM2 backbone and image splitting. Downloads HuggingFaceTB/SmolVLM2-256M-Video-Instruct (~0.5 GB).

Shares fixtures and questions with ``test_vlm.py``; that file covers the SmolVLM (v1) path in depth. Here: the
second causal backbone builds and runs through the same code, and ``image_split_edge`` produces exactly the tiles
and prompt tokens the processor would, trains and survives save/load.
"""
import math

import pytest
import torch
from PIL import Image
from test_vlm import DEVICE, QUESTIONS, check_schema, frame, square

from laya.preprocess import ImagePrep, prefix_ids
from laya.vlm import PREFIX_TEXT, SMOLVLM2_BACKBONE, VLMAgent, build_vlm_inputs, collate_vlm, default_max_len, vlm_prefix
from laya.vlm_train import collect_logits, synthetic_examples, train


@pytest.fixture(scope="module")
def agent():
    torch.manual_seed(0)
    return VLMAgent(backbone=SMOLVLM2_BACKBONE, device=DEVICE)


@pytest.fixture(scope="module")
def split_agent():
    torch.manual_seed(0)
    return VLMAgent(backbone=SMOLVLM2_BACKBONE, device=DEVICE, image_split_edge=1024)


def photo(w=700, h=400):
    img = Image.new("RGB", (w, h), (255, 255, 255))
    img.paste(Image.new("RGB", (w // 3, h // 3), (220, 20, 20)), (w // 4, h // 4))
    return img


def test_smolvlm2_is_a_causal_terminator_backbone(agent):
    assert agent.model.readout == "terminator"
    assert agent.cfg["max_len"] == 1024 and agent.processor.laya_max_len == 1024
    assert agent.prep.split_edge == 0 and agent.prep.max_tiles == 1
    check_schema(agent.predict({"image": square((220, 20, 20))}, QUESTIONS), QUESTIONS)


def test_smolvlm2_prefix_cache_matches_full_path(agent):
    """The shared-prefix path goes through SmolVLM2's model for the image prefill and its text model after."""
    state = {"image": square((220, 20, 20)), "caption": "a test card"}
    for attention in ("causal", "block"):
        agent.model.option_attention = attention
        try:
            ref = agent.predict(state, QUESTIONS, n_permutations=2, prefix_cache=False)
            got = agent.predict(state, QUESTIONS, n_permutations=2, prefix_cache=True)
        finally:
            agent.model.option_attention = "causal"
        for qid, a in ref["answers"].items():
            b = got["answers"][qid]
            pa, pb = a.get("probabilities", {"p": a.get("noul")}), b.get("probabilities", {"p": b.get("noul")})
            assert all(abs(pa[k] - pb[k]) <= 2e-4 for k in pa), (attention, qid, a, b)
            assert abs(a["action"]["act_probability"] - b["action"]["act_probability"]) <= 2e-4


def test_smolvlm2_fast_path_ids_match_the_processor(agent):
    """``prefix_ids`` rebuilds SmolVLMProcessor's unsplit image run (its global tag is ``global_image_token``)."""
    proc = agent.processor
    out = proc(text=[PREFIX_TEXT + proc.image_token], images=[[frame()]], do_image_splitting=False, return_tensors="pt",
               add_special_tokens=False)
    assert prefix_ids(proc, PREFIX_TEXT, 1) == out["input_ids"][0].tolist()


def test_split_config_round_trip():
    prep = ImagePrep.from_config({"image_split_edge": 2048})
    assert prep.backend == "processor" and prep.max_tiles == 17
    assert ImagePrep.from_config(prep.to_config()) == prep
    assert ImagePrep.from_config({}).split_edge == 0  # configs written before the key existed
    assert ImagePrep(backend="processor", split_edge=1024).max_tiles == 5
    with pytest.raises(ValueError):
        ImagePrep(backend="gpu", split_edge=1024)
    with pytest.raises(ValueError):
        ImagePrep(backend="processor", split_edge=256)
    assert default_max_len(ImagePrep()) == 1024
    assert default_max_len(ImagePrep(backend="processor", split_edge=1024)) == 1536
    assert default_max_len(ImagePrep(backend="processor", split_edge=2048)) == 2304


def test_split_prefix_matches_the_processor(split_agent):
    """Tiles and ids come straight from the processor, and the vision side agrees with the prompt."""
    proc, prep = split_agent.processor, split_agent.prep
    assert split_agent.cfg["max_len"] == default_max_len(prep) == proc.laya_max_len
    img = photo()  # 700x400 -> 1024x586 -> a 2x2 grid of 512 tiles + the global view
    p = vlm_prefix(proc, [img], prep)
    assert p["n_images"] == 5 and tuple(p["pixel_values"].shape) == (5, 3, 512, 512)
    assert sum(t == proc.image_token_id for t in p["ids"]) == 5 * prep.image_seq_len
    assert len(p["ids"]) > len(prefix_ids(proc, PREFIX_TEXT, 1))
    feats = split_agent.model.encode_images(p["pixel_values"].to(split_agent.device), p["pixel_attention_mask"].to(split_agent.device))
    assert feats.shape[:2] == (5, prep.image_seq_len)
    small = vlm_prefix(proc, [square((20, 40, 220))], prep)  # upscaled to 1024 first, as the processor does
    assert small["n_images"] == 5


def test_split_batches_pad_ragged_tile_counts(split_agent):
    proc = split_agent.processor
    q = VLMAgent._to_internal(QUESTIONS["color"])
    items = [dict(build_vlm_inputs(proc, {"image": img}, q), qtype=0) for img in (photo(), photo(500, 500), photo(1500, 300))]
    counts = [it["n_images"] for it in items]
    assert counts == [5, 5, 3]  # 1500x300 -> 1024x204 -> one row of two tiles + the global view
    b = collate_vlm(items, proc.tokenizer.pad_token_id)
    assert b["pixel_values"].shape[:2] == (3, max(counts))
    for i, n in enumerate(counts):
        assert bool((b["pixel_values"][i, n:] == 0).all())


def test_split_predict_train_save_load(split_agent, tmp_path):
    img = photo()
    res = split_agent.predict({"image": img}, QUESTIONS)
    check_schema(res, QUESTIONS)
    assert res["usage"]["input_tokens"] > 3 * 5 * split_agent.prep.image_seq_len
    exs = synthetic_examples(2)
    losses = train(split_agent.model, split_agent.processor, exs, steps=2, batch_size=2, freeze="head",
                   device=str(split_agent.device), log_every=0)
    assert all(math.isfinite(x) for x in losses)
    split_agent.model.eval()
    recs = collect_logits(split_agent.model, split_agent.processor, exs[:3], batch_size=3)
    assert len(recs) == 3
    split_agent.save(str(tmp_path), include_backbone=False)
    back = VLMAgent(str(tmp_path), device=DEVICE)
    assert back.prep == split_agent.prep and back.cfg["max_len"] == split_agent.cfg["max_len"]
    assert back.processor.laya_max_len == split_agent.cfg["max_len"]
    a, b = back.predict({"image": img}, QUESTIONS), split_agent.predict({"image": img}, QUESTIONS)
    for qid in QUESTIONS:
        assert a["answers"][qid]["confidence"] == pytest.approx(b["answers"][qid]["confidence"], abs=2e-3)
