"""Tests for the experimental SmolVLM-backed decision model. Downloads HuggingFaceTB/SmolVLM-256M-Instruct (~0.5 GB)."""
import json
import math
import random
import time

import numpy as np
import pytest
import torch
from PIL import Image

from laya.common import render_options
from laya.preprocess import FrameFeatureCache, ImagePrep, _axis_weights, prefix_ids, stage1_size
from laya.vlm import (OPTION_BULLET, OPTION_END, PREFIX_TEXT, PROMPT_FORMAT_VERSION, VLMAgent, build_vlm_inputs, collate_vlm,
                      input_ids_sha256, snapshot_revision, split_state, vlm_prefix)
from laya.vlm_train import collect_logits, fit_temperatures_from, load_jsonl_examples, metrics_from, synthetic_examples, train

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def sync():
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    elif DEVICE == "mps":
        torch.mps.synchronize()

QUESTIONS = {
    "color": {
        "type": "choice",
        "instructions": "What color is the square?",
        "criteria": {"red": "the square is red", "blue": "the square is blue", "green": "the square is green"},
    },
    "size": {"type": "score", "instructions": "How much of the image does the square fill?", "criteria": ["tiny", "about half", "almost all"]},
    "is_red": {"type": "noul", "instructions": "Is the square red?"},
}


def square(color):
    img = Image.new("RGB", (96, 96), (255, 255, 255))
    img.paste(Image.new("RGB", (48, 48), color), (24, 24))
    return img


@pytest.fixture(scope="module")
def agent():
    torch.manual_seed(0)
    return VLMAgent(backbone="HuggingFaceTB/SmolVLM-256M-Instruct", device=DEVICE)


def check_schema(res, questions):
    assert res["model"] == "laya-vlm"
    assert set(res["answers"]) == set(questions)
    for qid, qdef in questions.items():
        a = res["answers"][qid]
        assert a["type"] == qdef["type"]
        assert 0.0 <= a["confidence"] <= 1.0
        assert 0.0 <= a["action"]["act_probability"] <= 1.0
        if a["type"] == "choice":
            assert list(a["probabilities"]) == list(qdef["criteria"])
            assert a["choice"] in qdef["criteria"]
            assert math.isclose(sum(a["probabilities"].values()), 1.0, abs_tol=1e-3)
        elif a["type"] == "score":
            assert list(a["probabilities"]) == [str(i) for i in range(len(qdef["criteria"]))]
            assert math.isclose(sum(a["probabilities"].values()), 1.0, abs_tol=1e-3)
            assert 0.0 <= a["score"] <= len(qdef["criteria"]) - 1
        else:
            assert 0.0 <= a["noul"] <= 1.0
    assert res["usage"]["input_tokens"] > 0
    prov = res["provenance"]
    assert prov["prompt_format_version"] == PROMPT_FORMAT_VERSION
    assert len(prov["input_ids_sha256"]) == 64 and prov["n_rows"] >= len(questions)
    assert set(prov["temperatures"]) == set(questions)
    assert prov["backbone"]["id"] and prov["dtype"] and prov["device"] and prov["torch"] and prov["transformers"]
    assert prov["option_attention"] in ("causal", "block") and prov["n_permutations"] >= 1
    json.dumps(prov)


def test_predict_image_and_text(agent):
    for state in ({"image": square((220, 20, 20)), "caption": "a test card"}, {"images": [square((220, 20, 20)), square((20, 40, 220))]}):
        res = agent.predict(state, QUESTIONS)
        check_schema(res, QUESTIONS)
        assert res["usage"]["images"] == len(split_state(state)[0])
    res = agent.predict("Customer: I was billed twice, please refund.", QUESTIONS)
    check_schema(res, QUESTIONS)
    assert res["usage"]["images"] == 0
    res = agent.predict({"image": square((20, 40, 220))}, QUESTIONS, n_permutations=3)
    check_schema(res, QUESTIONS)


def test_block_option_attention(agent):
    agent.model.option_attention = "block"
    try:
        check_schema(agent.predict({"image": square((20, 40, 220))}, QUESTIONS), QUESTIONS)
    finally:
        agent.model.option_attention = "causal"


def test_bidirectional_is_a_deprecated_alias_for_block():
    from laya.vlm import normalize_option_attention

    with pytest.warns(DeprecationWarning):
        assert normalize_option_attention("bidirectional") == "block"
    assert normalize_option_attention("block") == "block"
    assert normalize_option_attention("causal") == "causal"
    with pytest.raises(ValueError):
        normalize_option_attention("full")


def test_marker_positions(agent):
    tok = agent.processor.tokenizer
    (end_id,) = tok(OPTION_END, add_special_tokens=False)["input_ids"]
    for state in ({"image": square((0, 0, 255)), "note": "x"}, "plain text state"):
        for qdef in QUESTIONS.values():
            q = VLMAgent._to_internal(qdef)
            opts = render_options(q)
            order = list(range(len(opts)))
            random.Random(1).shuffle(order)
            it = build_vlm_inputs(agent.processor, state, q, option_order=order)
            ids, markers = it["ids"], it["markers"]
            assert len(markers) == len(opts)
            start = it["option_span"][0]
            for j, m in enumerate(markers):
                assert ids[m] == end_id
                assert tok.decode(ids[start : m + 1]) == OPTION_BULLET + opts[order[j]] + OPTION_END
                start = m + 1
            assert markers[-1] == len(ids) - 1 == it["option_span"][1] - 1


def test_train_head_only(agent):
    model = agent.model
    enc_names = ["vision_model.embeddings.patch_embedding.weight", "connector.modality_projection.proj.weight",
                 "text_model.embed_tokens.weight", "text_model.layers.29.mlp.down_proj.weight", "text_model.norm.weight"]
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


@pytest.mark.parametrize("include_backbone", [True, False])
def test_save_load_roundtrip(agent, tmp_path, include_backbone):
    agent.temperature = [1.3, 0.8, 1.1]
    state = {"image": square((220, 20, 20)), "caption": "a test card"}
    before = [agent.predict(state, QUESTIONS), agent.predict("plain text", QUESTIONS)]
    agent.save(str(tmp_path), include_backbone=include_backbone)
    loaded = VLMAgent(str(tmp_path), device=DEVICE)
    after = [loaded.predict(state, QUESTIONS), loaded.predict("plain text", QUESTIONS)]
    assert [r["answers"] for r in after] == [r["answers"] for r in before]
    assert [r["usage"] for r in after] == [r["usage"] for r in before]
    # same inputs, same backbone commit; only the checkpoint id differs (fresh agent -> the saved directory)
    for b, a in zip(before, after):
        assert a["provenance"]["input_ids_sha256"] == b["provenance"]["input_ids_sha256"]
        assert a["provenance"]["backbone"] == b["provenance"]["backbone"]
        assert a["provenance"]["checkpoint"] == {"id": str(tmp_path), "revision": None}
    assert loaded.cfg["temperature"] == [1.3, 0.8, 1.1]
    del loaded


def test_revisions_are_recorded(agent, tmp_path):
    # the fresh agent resolved the backbone commit it loaded and saves it; a head-only reload pins to it
    rev = agent.cfg["backbone_revision"]
    assert rev and len(rev) == 40
    agent.save(str(tmp_path), include_backbone=False)
    with open(tmp_path / "vlm_agent_config.json") as f:
        assert json.load(f)["backbone_revision"] == rev
    loaded = VLMAgent(str(tmp_path), device=DEVICE)
    assert loaded.cfg["backbone_revision"] == rev
    assert loaded.predict("plain text", QUESTIONS)["provenance"]["backbone"]["revision"] == rev
    assert snapshot_revision("/cache/hub/models--a--b/snapshots/" + rev) == rev
    assert snapshot_revision(str(tmp_path)) is None


def test_provenance_hash_follows_the_input_ids(agent):
    q = {"is_red": QUESTIONS["is_red"]}
    a, b = agent.predict("one state", q), agent.predict("one state", q)
    c = agent.predict("another state", q)
    assert a["provenance"]["input_ids_sha256"] == b["provenance"]["input_ids_sha256"]
    assert a["provenance"]["input_ids_sha256"] != c["provenance"]["input_ids_sha256"]
    it = build_vlm_inputs(agent.processor, "one state", VLMAgent._to_internal(q["is_red"]))
    assert a["provenance"]["input_ids_sha256"] == input_ids_sha256([it["ids"]])
    p4 = agent.predict("one state", q, n_permutations=4)["provenance"]
    assert p4["n_permutations"] == 4 and p4["n_rows"] == 2  # a 2-option question has two distinct orders
    assert p4["temperatures"] == {"is_red": float(agent.temperature[2])}


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
        print("\nlatency %s state, 1 noul question, %s: median %.1f ms (min %.1f, max %.1f)" % (name, DEVICE, ts[2], ts[0], ts[-1]))


def test_jsonl_dataset_and_eval(agent, tmp_path):
    """Prepared-dataset format: <root>/<name>/<split>.jsonl + images/; label indexes the rendered options."""
    base = tmp_path / "toyvqa"
    (base / "images").mkdir(parents=True)
    recs = []
    for i, (color, rgb) in enumerate([("red", (220, 20, 20)), ("blue", (20, 40, 220))] * 3):
        square(rgb).save(base / "images" / ("%d.jpg" % i))
        recs.append({"id": str(i), "image": "images/%d.jpg" % i, "state_text": "photo %d" % i if i % 2 else None,
                     "question": {"type": "choice", "instructions": "What color is the square?", "criteria": ["red", "blue", "green"]},
                     "label": ["red", "blue", "green"].index(color)})
        recs.append({"id": "n%d" % i, "image": "images/%d.jpg" % i, "state_text": None,
                     "question": {"type": "noul", "instructions": "Is the square red?", "criteria": None}, "label": int(color == "red")})
    recs.append({"id": "text-only", "image": None, "state_text": "The sky is blue.",
                 "question": {"type": "choice", "instructions": "What color is the sky?", "criteria": ["red", "blue"]}, "label": 1})
    recs.append({"id": "bad-label", "image": None, "state_text": "x", "question": {"type": "noul", "instructions": "?", "criteria": None}, "label": 5})
    (base / "val.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n")

    examples = load_jsonl_examples(str(tmp_path), "toyvqa", "val")
    assert len(examples) == len(recs) - 1  # out-of-range label dropped
    assert examples[0]["state"]["image"] == str(base / "images" / "0.jpg") and examples[0]["dataset"] == "toyvqa"
    assert examples[1]["target"] == [0.0, 1.0]  # noul label 1 == "true"
    assert examples[-1]["state"] == {"context": "The sky is blue."}

    records = collect_logits(agent.model, agent.processor, examples, batch_size=2)
    assert len(records) == len(examples)
    assert all(len(r["input_ids_sha256"]) == 1 and len(r["input_ids_sha256"][0]) == 64 for r in records)
    m = metrics_from(records, fit_temperatures_from(records))
    assert m["all"]["n"] == m["toyvqa"]["n"] == len(examples)
    assert 0.0 <= m["all"]["acc"] <= 1.0 and 0.0 <= m["all"]["ece"] <= 1.0 and math.isfinite(m["all"]["nll"])


@pytest.mark.parametrize("every_min", [10.0, 1e-6])
def test_state_is_saved_only_on_real_evals(agent, every_min):
    """A cheap eval probe must not trigger a state write, and the write interval has a floor.

    ``train`` is given a small ``eval_every`` so the caller can check progress often; only the calls that report
    a real eval (a truthy return) are worth the seconds a state write costs. ``every_min`` below
    ``MIN_STATE_MINUTES`` is clamped, so even 1e-6 minutes writes nothing extra in a run this short.
    """
    evals, saves = [], []

    def eval_fn(step):
        evals.append(step)
        return step >= 6  # the probe only really evaluates near the end

    train(agent.model, agent.processor, synthetic_examples(4), steps=8, batch_size=2, freeze="head", device=DEVICE,
          eval_fn=eval_fn, eval_every=2, save_state_fn=lambda step, st: saves.append(step),
          save_state_every_min=every_min)

    assert evals == [2, 4, 6, 8]
    assert saves == [6, 8]


# ---------------------------------------------------------------------------------------------------------
# Preprocessing (laya.preprocess): the cheap path must agree with the Hugging Face processor
# ---------------------------------------------------------------------------------------------------------


def frame(seed=0, h=210, w=160):
    """A hard-edged pseudo-Atari frame: resampling differences show up on edges, not on smooth gradients."""
    rng = np.random.default_rng(seed)
    a = np.zeros((h, w, 3), dtype=np.uint8)
    a[:, :, 2] = 40
    for _ in range(12):
        y, x = rng.integers(0, h - 20), rng.integers(0, w - 20)
        a[y:y + rng.integers(3, 20), x:x + rng.integers(3, 20)] = rng.integers(0, 256, 3, dtype=np.uint8)
    return a


@pytest.mark.parametrize("n_in,n_out", [(210, 512), (160, 512), (210, 256), (2048, 512), (1560, 512), (210, 2048)])
def test_axis_weights_match_torchvision_lanczos(n_in, n_out):
    """The analytic resample weights are torch's own: a resize is exactly two matmuls with them."""
    import torchvision.transforms.v2.functional as tvF
    from torchvision.transforms import InterpolationMode

    x = torch.rand(3, n_in, n_in, dtype=torch.float64)
    ref = tvF.resize(x, [n_out, n_out], interpolation=InterpolationMode.LANCZOS, antialias=True)
    w = _axis_weights(n_in, n_out)
    assert torch.allclose(w.sum(1), torch.ones(n_out, dtype=torch.float64))
    assert torch.einsum("ip,cpq,jq->cij", w, x, w).sub(ref).abs().max() < 1e-12


def test_stage1_size_matches_the_processor():
    assert stage1_size(210, 160) == (2048, 1560)  # what Idefics3ImageProcessor does to an Atari frame first
    assert stage1_size(160, 210) == (1560, 2048)
    assert stage1_size(512, 512) == (2048, 2048)


@pytest.mark.parametrize("size,tokens", [(512, 64), (256, 16), (128, 4)])
def test_image_size_sets_token_count(agent, size, tokens):
    """image_size drives both the vision grid and the <image> run in the prompt; they must agree."""
    prep = ImagePrep(image_size=size)
    assert prep.image_seq_len == tokens
    proc = agent.processor
    before = ImagePrep.from_config(agent.cfg, default_backend=agent.prep.backend)
    try:
        prep.apply(proc)
        out = proc(text=[PREFIX_TEXT + proc.image_token], images=[[frame()]], do_image_splitting=False, return_tensors="pt")
        assert tuple(out["pixel_values"].shape[-2:]) == (size, size)
        assert int((out["input_ids"][0] == proc.image_token_id).sum()) == tokens
        assert prefix_ids(proc, PREFIX_TEXT, 1, tokens) == out["input_ids"][0].tolist()
        assert prefix_ids(proc, PREFIX_TEXT, 2, tokens) != prefix_ids(proc, PREFIX_TEXT, 1, tokens)
        pv, pam = prep.pixel_values([frame()], device=agent.device, dtype=agent.model.encoder.dtype)
        assert agent.model.encode_images(pv, pam).shape[:2] == (1, tokens)  # the vision side agrees with the prompt
    finally:
        before.apply(proc)


@pytest.mark.parametrize("size", [512, 256])
def test_gpu_pixels_match_the_processor(agent, size):
    """The device-side path reproduces the processor's pixels bar LANCZOS overshoot clamped at its intermediate."""
    proc, frames = agent.processor, [frame(i) for i in range(4)]
    before = ImagePrep.from_config(agent.cfg, default_backend=agent.prep.backend)
    try:
        ImagePrep(image_size=size, backend="processor").apply(proc)
        ref = torch.cat([proc(text=[PREFIX_TEXT + proc.image_token], images=[[f]], do_image_splitting=False,
                              return_tensors="pt")["pixel_values"][0] for f in frames])
        for interp, mean_tol in (("processor", 0.15), ("lanczos", 0.35), ("bicubic", 0.8)):
            pv, mask = ImagePrep(image_size=size, interpolation=interp).pixel_values(frames)
            assert pv.shape == (len(frames), 3, size, size) and mask.shape == (len(frames), size, size)
            assert mask.all()  # a square resize never pads
            d = (pv - ref.reshape(pv.shape)).abs() * 127.5  # back into 0-255 units
            assert float(d.mean()) < mean_tol, (interp, float(d.mean()), float(d.max()))
            assert float(d.flatten().quantile(0.99)) < 6.0, (interp, float(d.flatten().quantile(0.99)))
    finally:
        before.apply(proc)


def test_gpu_items_defer_the_resize(agent):
    """Items built on the GPU path carry raw uint8 frames; the model turns them into pixels itself."""
    prep = ImagePrep(image_size=256, backend="gpu")
    p = vlm_prefix(agent.processor, [frame(), frame(1)], prep)
    assert p["pixel_values"] is None and p["raw_images"].shape == (2, 3, 210, 160)
    assert p["raw_images"].dtype == torch.uint8 and p["ids"] == prefix_ids(agent.processor, PREFIX_TEXT, 2, 16)
    q = VLMAgent._to_internal(QUESTIONS["color"])
    items = [dict(build_vlm_inputs(agent.processor, {"images": [frame(i), frame(i + 1)]}, q, prep=prep), qtype=0)
             for i in range(3)]
    b = collate_vlm(items, agent.processor.tokenizer.pad_token_id)
    assert b["pixel_values"] is None and b["raw_pixels"].shape == (3, 2, 3, 210, 160) and b["image_mask"].all()
    pv, _ = prep.pixel_values(b["raw_pixels"])
    assert pv.shape == (3, 2, 3, 256, 256)


def test_two_frame_feature_cache_changes_nothing(agent):
    """Reusing last step's encoder output must give the same answer, and halve the encoder's work.

    "Same" is not bit-exact, and cannot be: a cache hit was computed in whatever batch its miss belonged to, and
    the vision tower's reductions are not associative, so batch-of-1 and batch-of-2 already disagree at the same
    1e-4 scale *without* any cache (the second assert pins that down -- if the cache were returning the wrong
    frame's features the gap would be order 1, not order 1e-4).
    """
    frames = [frame(i) for i in range(6)]
    prep = ImagePrep(backend="gpu").apply(agent.processor)
    try:
        cache, plain, cached, alone = FrameFeatureCache(), [], [], []
        for i in range(1, len(frames)):
            pair = [frames[i - 1], frames[i]]
            plain.append(agent.model.encode_raw_images(pair))
            alone.append(torch.cat([agent.model.encode_raw_images([f]) for f in pair]))
            cached.append(torch.stack(cache.features(agent.model.encode_raw_images, pair)))
        scale = max(float(p.abs().max()) for p in plain)
        for a, b in zip(plain, cached):
            assert float((a - b).abs().max()) < 1e-3 * scale
        # the cache's error is the batching error, not an error of its own
        assert (max(float((a - b).abs().max()) for a, b in zip(plain, cached))
                <= 2 * max(float((a - b).abs().max()) for a, b in zip(plain, alone)) + 1e-9)
        # every step but the first reuses the frame it saw as "current" last step
        assert cache.stats == {"hits": len(frames) - 2, "misses": len(frames), "hit_rate": pytest.approx(0.4)}
        # the same frame twice (an episode's first step) is encoded once
        c2 = FrameFeatureCache()
        c2.features(agent.model.encode_raw_images, [frames[0], frames[0]])
        assert c2.stats["misses"] == 1
    finally:
        ImagePrep.from_config(agent.cfg, default_backend=agent.prep.backend).apply(agent.processor)


def test_train_and_predict_at_256(tmp_path):
    """Nothing downstream assumes 64 image tokens: train, predict, save and reload a 256-pixel agent."""
    torch.manual_seed(0)
    a = VLMAgent(backbone="HuggingFaceTB/SmolVLM-256M-Instruct", device=DEVICE, image_size=256, preprocess="gpu")
    assert a.prep.image_seq_len == 16 and a.processor.image_seq_len == 16
    losses = train(a.model, a.processor, synthetic_examples(4), steps=2, batch_size=2, freeze="head", device=DEVICE)
    assert all(math.isfinite(x) for x in losses)
    state = {"images": [square((220, 20, 20)), square((20, 40, 220))]}
    res = a.predict(state, QUESTIONS)
    check_schema(res, QUESTIONS)
    ids = build_vlm_inputs(a.processor, state, VLMAgent._to_internal(QUESTIONS["color"]), prep=a.prep)["ids"]
    assert sum(i == a.processor.image_token_id for i in ids) == 32  # 2 images x 16, not 2 x 64
    a.save(str(tmp_path))
    loaded = VLMAgent(str(tmp_path), device=DEVICE)
    assert loaded.cfg["image_size"] == 256 and loaded.prep.backend == "gpu"
    assert loaded.predict(state, QUESTIONS) == res


def test_config_without_the_new_keys_keeps_the_old_path(tmp_path):
    """A checkpoint saved before image_size existed was trained at 512 through the processor: keep it there."""
    old = json.loads(json.dumps({"backbone": "HuggingFaceTB/SmolVLM-256M-Instruct", "head_layers": 2, "n_act": 2,
                                 "max_len": 1024, "head_max_len": 256, "option_attention": "causal", "dtype": "fp32",
                                 "temperature": [1.0, 1.0, 1.0], "temperature_by_options": {}}))
    prep = ImagePrep.from_config(old, default_backend="processor")
    assert prep.image_size == 512 and prep.backend == "processor" and prep.image_seq_len == 64
    assert ImagePrep.from_config({}, default_backend="gpu").backend == "gpu"
    assert ImagePrep.from_config(ImagePrep(image_size=256, interpolation="bicubic").to_config()) \
        == ImagePrep(image_size=256, interpolation="bicubic")
    with pytest.raises(ValueError):
        ImagePrep(image_size=200)  # not a multiple of patch_size * scale_factor


def test_ragged_image_counts_pad_without_losing_a_frame(agent):
    """A batch mixing one- and two-image rows: the padded slot must vanish, the real frames must not.

    ``Idefics3Model.get_image_features`` drops an image slot whose pixels are *all exactly* 0.0, which is how
    ``VLMDecisionModel.forward`` disposes of padded slots. The risk that buys is a real frame being dropped for
    looking like padding -- impossible here, because uint8 cannot hit the 127.5 that normalises to zero.
    """
    prep = ImagePrep(image_size=256, backend="gpu")
    q = VLMAgent._to_internal(QUESTIONS["color"])
    frames = [frame(i) for i in range(3)]
    items = [dict(build_vlm_inputs(agent.processor, {"image": frames[0]}, q, prep=prep), qtype=0),
             dict(build_vlm_inputs(agent.processor, {"images": frames[1:]}, q, prep=prep), qtype=0)]
    b = collate_vlm(items, agent.processor.tokenizer.pad_token_id)
    assert b["image_mask"].tolist() == [[True, False], [True, True]]

    pv, pam = prep.pixel_values(b["raw_pixels"], device=agent.device, dtype=agent.model.encoder.dtype)
    pv = pv * b["image_mask"].to(agent.device)[..., None, None, None]
    assert int((pv.flatten(2).abs().sum(-1) == 0).sum()) == 1  # exactly the padded slot

    # no real uint8 frame can normalise to all-zero, so none can be mistaken for padding
    for v in (127, 128):
        flat, _ = prep.pixel_values([np.full((210, 160, 3), v, np.uint8)])
        assert float(flat.abs().min()) > 1e-3

    keep = b["image_mask"].to(agent.device)
    feats = agent.model.encode_images(pv[keep], pam[keep])
    assert feats.shape[0] == 3  # three real frames, not four slots
    # encode_raw_images would use the model's own prep (512 here), so resize with this prep explicitly
    pv1, pam1 = prep.pixel_values([frames[0]], device=agent.device, dtype=agent.model.encoder.dtype)
    alone = agent.model.encode_images(pv1, pam1)
    assert float((feats[0] - alone[0]).abs().max()) < 1e-3  # batching noise only
