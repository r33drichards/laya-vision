"""Tests for the experimental LocateAnything-3B backbone (``laya.locate_anything``).

Most of this runs a tiny randomly initialised LocateAnything (the checkpoint's own tokenizer and ``modeling_vit.py``
at the pinned commit, a few MB, with small dimensions), which is enough for the plumbing: the image run, the
patchify, the readout, prefix caching, save/load. ``test_checkpoint_keys_cover_the_model`` checks the real
checkpoint's weight names against the full-size model on the meta device. ``test_real_checkpoint`` downloads and
runs the real weights (7.6 GB, bf16); set ``LAYA_TEST_LOCATE_ANYTHING=1`` to run it.
"""
import json
import os

import numpy as np
import pytest
import torch
from test_vlm import QUESTIONS, check_schema, square

import laya.vlm as vlm
from laya.locate_anything import (LOCATE_ANYTHING_BACKBONE, LOCATE_ANYTHING_REVISION, LocateAnythingBackbone,
                                  LocateAnythingConfig, LocateAnythingProcessor, checkpoint_key, is_locate_anything)
from laya.preprocess import ImagePrep, prefix_ids
from laya.vlm import PREFIX_TEXT, VLMAgent, VLMDecisionModel, build_vlm_inputs, collate_vlm, set_trainable, vlm_prefix

TINY = {
    "model_type": "locateanything",
    "image_token_index": 151665,
    "text_config": {"architectures": ["Qwen2ForCausalLM"], "model_type": "qwen2", "hidden_size": 64,
                    "intermediate_size": 128, "num_attention_heads": 4, "num_key_value_heads": 2,
                    "num_hidden_layers": 2, "vocab_size": 152681, "max_position_embeddings": 4096,
                    "rms_norm_eps": 1e-6, "rope_theta": 1000000.0, "tie_word_embeddings": True},
    "vision_config": {"model_type": "moonvit", "hidden_size": 32, "intermediate_size": 64, "num_attention_heads": 2,
                      "num_hidden_layers": 2, "patch_size": 14, "init_pos_emb_height": 8, "init_pos_emb_width": 8,
                      "merge_kernel_size": [2, 2]},
}


def probs(res, qid):
    a = res["answers"][qid]
    return a.get("probabilities") or {"noul": a["noul"]}


def tiny_backbone():
    torch.manual_seed(0)
    return LocateAnythingBackbone.from_config(LocateAnythingConfig(TINY, commit=LOCATE_ANYTHING_REVISION))


@pytest.fixture(scope="module")
def agent():
    """A fresh agent on the tiny backbone: ``VLMAgent(backbone=LOCATE_ANYTHING_BACKBONE)`` with the 3.8B download
    swapped for the tiny model; everything else (processor, prep, config) is the real path."""
    real = LocateAnythingBackbone.from_pretrained
    LocateAnythingBackbone.from_pretrained = classmethod(lambda cls, *a, **k: tiny_backbone())
    try:
        return VLMAgent(backbone=LOCATE_ANYTHING_BACKBONE, device="cpu", dtype="fp32", image_size=112)
    finally:
        LocateAnythingBackbone.from_pretrained = real


def test_backbone_detection_and_defaults(agent):
    assert is_locate_anything(LOCATE_ANYTHING_BACKBONE) and is_locate_anything("someone/locate-anything-3b-copy")
    assert not is_locate_anything("HuggingFaceTB/SmolVLM-256M-Instruct") and not is_locate_anything(None)
    assert agent.model.readout == "terminator"
    assert agent.cfg["backbone_revision"] == LOCATE_ANYTHING_REVISION
    assert (agent.prep.patch_size, agent.prep.scale_factor, agent.prep.image_seq_len) == (14, 2, 16)
    assert agent.cfg["image_patch_size"] == 14 and agent.cfg["image_scale_factor"] == 2
    assert isinstance(agent.processor, LocateAnythingProcessor)
    assert ImagePrep.from_config(agent.cfg) == agent.prep


def test_smolvlm_config_keys_unchanged():
    """The new geometry keys are written only off SmolVLM's defaults, so its saved configs do not change."""
    assert set(ImagePrep().to_config()) == {"image_size", "preprocess", "image_interpolation", "image_split_edge"}


def test_image_run_and_both_backends_agree(agent):
    proc, img = agent.processor, square((255, 0, 0))
    gpu = vlm_prefix(proc, [img, img], ImagePrep(image_size=112, backend="gpu", interpolation="bicubic",
                                                 patch_size=14, scale_factor=2))
    cpu = vlm_prefix(proc, [img, img], ImagePrep(image_size=112, backend="processor", interpolation="bicubic",
                                                 patch_size=14, scale_factor=2))
    tok = proc.tokenizer
    run = [tok.convert_tokens_to_ids("<img>")] + [151665] * 16 + [tok.convert_tokens_to_ids("</img>")]
    head = tok(PREFIX_TEXT, add_special_tokens=False)["input_ids"]
    assert gpu["ids"] == cpu["ids"] == head + run + run
    assert cpu["ids"] == prefix_ids(proc, PREFIX_TEXT, 2)
    assert cpu["pixel_values"].shape == (2, 3, 112, 112)
    pv, _ = agent.prep.pixel_values(gpu["raw_images"])
    assert torch.allclose(pv, cpu["pixel_values"])


def test_images_of_different_sizes(agent):
    from PIL import Image

    out = vlm_prefix(agent.processor, [square((255, 0, 0)), Image.new("RGB", (200, 50), (0, 0, 255))],
                     ImagePrep(image_size=112, backend="processor", interpolation="bicubic", patch_size=14,
                               scale_factor=2))
    assert out["pixel_values"].shape == (2, 3, 112, 112) and out["n_images"] == 2


def test_patchify_matches_the_checkpoint_image_processor(agent):
    """Our batched patchify packs patches in the order the checkpoint's ``LocateAnythingImageProcessor.patchify``
    does, so MoonViT sees the same sequence."""
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    cls = get_class_from_dynamic_module("image_processing_locateanything.LocateAnythingImageProcessor",
                                        LOCATE_ANYTHING_BACKBONE, revision=LOCATE_ANYTHING_REVISION)
    x = torch.randn(3, 56, 84)
    theirs, grid = cls().patchify(x)
    enc = agent.model.encoder
    seen = {}
    real = enc.vision_model.forward
    enc.vision_model.forward = lambda p, g: seen.update(p=p, g=g) or real(p, g)
    try:
        feats = enc.get_image_features(x[None, None]).pooler_output
    finally:
        del enc.vision_model.forward
    assert torch.equal(seen["p"], theirs) and seen["g"].tolist() == [list(grid)]
    assert feats.shape == (1, (56 // 28) * (84 // 28), 64)


def test_padded_image_slots_are_dropped(agent):
    enc = agent.model.encoder
    x = torch.rand(1, 3, 3, 56, 56) * 2 - 1
    x[0, 1] = 0
    assert enc.get_image_features(x).pooler_output.shape[0] == 2


def test_predict_schema(agent):
    res = agent.predict({"image": square((255, 0, 0)), "note": "a test"}, QUESTIONS)
    check_schema(res, QUESTIONS)
    assert res["provenance"]["backbone"] == {"id": LOCATE_ANYTHING_BACKBONE, "revision": LOCATE_ANYTHING_REVISION}


@pytest.mark.parametrize("attention", ["causal", "block"])
def test_prefix_cache_matches_full_path(agent, attention):
    """The native Qwen2 takes the 4D masks and the KV cache ``laya.vlm`` builds, like SmolVLM's Llama does."""
    agent.model.option_attention = attention
    try:
        state = {"image": square((0, 0, 255)), "note": "shared prefix"}
        full = agent.predict(state, QUESTIONS, prefix_cache=False)
        cached = agent.predict(state, QUESTIONS, prefix_cache=True)
    finally:
        agent.model.option_attention = "causal"
    for qid in QUESTIONS:
        for k, v in probs(full, qid).items():
            assert abs(v - probs(cached, qid)[k]) < 1e-4


def test_freezing_stages(agent):
    m = agent.model
    n_head = set_trainable(m, "head")
    assert not any(p.requires_grad for p in m.encoder.parameters())
    n_last = set_trainable(m, "last_n", n_last=1)
    assert n_last > n_head and m.encoder.text_model.layers[-1].self_attn.q_proj.weight.requires_grad
    set_trainable(m, "full")
    assert not any(p.requires_grad for p in m.encoder.vision_model.parameters())
    assert m.encoder.mlp1[1].weight.requires_grad
    set_trainable(m, "head")


def test_batched_forward_with_padding(agent):
    q = vlm.VLMAgent._to_internal(QUESTIONS["color"])
    items = [build_vlm_inputs(agent.processor, {"image": square((255, 0, 0))}, q, prep=agent.prep),
             build_vlm_inputs(agent.processor, {"images": [square((0, 255, 0))] * 2, "t": "longer " * 20}, q,
                              prep=agent.prep)]
    for it in items:
        it["qtype"] = 0
    b = collate_vlm(items, agent.processor.tokenizer.pad_token_id)
    with torch.no_grad():
        logits, act = agent.model(b["input_ids"], b["attention_mask"], b["marker_pos"], b["marker_mask"],
                                  torch.zeros(2, dtype=torch.long), raw_pixels=b.get("raw_pixels"),
                                  pixel_values=b.get("pixel_values"), pixel_attention_mask=b.get("pixel_attention_mask"),
                                  image_mask=b.get("image_mask"))
    assert logits.shape == (2, 3) and torch.isfinite(logits).all() and act.shape == (2, 2)


@pytest.mark.parametrize("include_backbone", [True, False])
def test_save_load_roundtrip(agent, tmp_path, include_backbone):
    state = {"image": square((0, 255, 0))}
    before = agent.predict(state, QUESTIONS)
    agent.save(str(tmp_path), include_backbone=include_backbone)
    with open(tmp_path / "vlm_agent_config.json") as f:
        cfg = json.load(f)
    assert cfg["backbone"] == LOCATE_ANYTHING_BACKBONE and cfg["image_patch_size"] == 14
    if not include_backbone:
        return  # a head-only reload fetches the 3.8B backbone; covered by test_real_checkpoint
    assert json.load(open(tmp_path / "backbone" / "config.json"))["laya_code_revision"] == LOCATE_ANYTHING_REVISION
    again = VLMAgent(str(tmp_path), device="cpu")
    assert isinstance(again.processor, LocateAnythingProcessor)
    after = again.predict(state, QUESTIONS)
    for qid in QUESTIONS:
        for k, v in probs(before, qid).items():
            assert abs(v - probs(after, qid)[k]) < 1e-5


def test_checkpoint_keys_cover_the_model():
    """Every weight of the full-size adapter comes from the checkpoint, and only the LM head is left over."""
    from huggingface_hub import hf_hub_download

    cfg = LocateAnythingConfig.from_pretrained(LOCATE_ANYTHING_BACKBONE)
    assert cfg._commit_hash == LOCATE_ANYTHING_REVISION and cfg.text_config.hidden_size == 2048
    with torch.device("meta"):
        model = LocateAnythingBackbone(cfg)
    with open(hf_hub_download(LOCATE_ANYTHING_BACKBONE, "model.safetensors.index.json",
                              revision=LOCATE_ANYTHING_REVISION)) as f:
        keys = list(json.load(f)["weight_map"])
    mapped = {checkpoint_key(k) for k in keys} - {None}
    assert mapped == set(model.state_dict())
    assert [k for k in keys if checkpoint_key(k) is None] == ["language_model.lm_head.weight"]


@pytest.mark.skipif(os.environ.get("LAYA_TEST_LOCATE_ANYTHING") != "1",
                    reason="downloads the 7.6 GB checkpoint; set LAYA_TEST_LOCATE_ANYTHING=1")
def test_real_checkpoint(tmp_path):
    """The real weights load, run, and give an image-dependent readout (the head itself is untrained)."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    a = VLMAgent(backbone=LOCATE_ANYTHING_BACKBONE, device=dev)
    assert a.model.encoder.dtype == torch.bfloat16
    red, blue = square((255, 0, 0)), square((0, 0, 255))
    q = {"color": QUESTIONS["color"]}
    check_schema(a.predict({"image": red}, q), q)
    with torch.no_grad():
        f = [a.model.encode_raw_images(np.asarray(img)[None]) for img in (red, blue)]
    assert f[0].shape == (1, 256, 2048) and not torch.allclose(f[0], f[1])
    a.save(str(tmp_path), include_backbone=False)
    del a, f  # two 3.8B models at once do not fit a 16 GB machine
    import gc

    gc.collect()
    b = VLMAgent(str(tmp_path), device=dev)
    assert b.cfg["backbone_revision"] == LOCATE_ANYTHING_REVISION
