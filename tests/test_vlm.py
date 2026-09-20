"""Tests for the experimental SmolVLM-backed decision model. Downloads HuggingFaceTB/SmolVLM-256M-Instruct (~0.5 GB)."""
import inspect
import json
import math
import random
import time

import pytest
import torch
from PIL import Image

from laya.common import QTYPES, proper_reward, render_options
from laya.vlm import OPTION_BULLET, OPTION_END, VLMAgent, build_vlm_inputs, split_state
from laya.vlm_train import (collect_logits, fit_temperatures_from, load_jsonl_examples, metrics_from,
                            sigma_at, synthetic_examples, train, vlm_loss)

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


def test_bidirectional_option_attention(agent):
    agent.model.option_attention = "bidirectional"
    try:
        check_schema(agent.predict({"image": square((20, 40, 220))}, QUESTIONS), QUESTIONS)
    finally:
        agent.model.option_attention = "causal"


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
    assert after == before
    assert loaded.cfg["temperature"] == [1.3, 0.8, 1.1]
    del loaded


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
    m = metrics_from(records, fit_temperatures_from(records))
    assert m["all"]["n"] == m["toyvqa"]["n"] == len(examples)
    assert 0.0 <= m["all"]["acc"] <= 1.0 and 0.0 <= m["all"]["ece"] <= 1.0 and math.isfinite(m["all"]["nll"])


# ---------------------------------------------------------------------------------------------------------
# Objective: RLCD (policy gradient on proper scoring rules), no cross-entropy
# ---------------------------------------------------------------------------------------------------------


def _loss_batch(n=8, k=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    mask = torch.ones(n, k, dtype=torch.bool)
    target = torch.softmax(torch.randn(n, k, generator=g), -1)
    qtype = torch.full((n,), QTYPES["choice"])
    return torch.randn(n, k, generator=g), target, qtype, mask


def _grad(logits, target, qtype, mask, noise_seed=1234, **kw):
    x = logits.detach().clone().requires_grad_(True)
    torch.manual_seed(noise_seed)  # the same exploration noise for every call
    loss, _ = vlm_loss(x, target, qtype, mask, **kw)
    loss.backward()
    return x.grad.clone()


def test_objective_defaults_match_the_rlcd_design():
    """Pure policy gradient: no cross-entropy, spherical weight 0.5, group of 8, sigma starting at 1.0."""
    d = inspect.signature(vlm_loss).parameters
    assert d["w_ce"].default == 0.0
    assert d["w_sph"].default == 0.5
    assert d["group_size"].default == 8
    assert d["sigma"].default == 1.0


def test_default_objective_carries_no_cross_entropy_gradient():
    """w_ce is a pure add-on: turning it up adds exactly the soft-cross-entropy gradient and nothing else,
    so at the default w_ce=0 none of it is present."""
    logits, target, qtype, mask = _loss_batch()
    g0 = _grad(logits, target, qtype, mask)                 # defaults -> RLCD alone
    g1 = _grad(logits, target, qtype, mask, w_ce=1.0)
    ce = (torch.softmax(logits, -1) - target) / logits.shape[0]
    assert torch.allclose(g1 - g0, ce, atol=1e-6)
    assert not torch.allclose(g0, ce, atol=1e-3)


def test_reward_uses_the_canonical_spherical_weight():
    """vlm_loss must not silently override proper_reward's w_sph; 0.5 is the designed weight."""
    logits, target, qtype, mask = _loss_batch()
    torch.manual_seed(7)
    _, r_default = vlm_loss(logits, target, qtype, mask)
    torch.manual_seed(7)
    _, r_explicit = vlm_loss(logits, target, qtype, mask, w_sph=0.5)
    torch.manual_seed(7)
    _, r_other = vlm_loss(logits, target, qtype, mask, w_sph=0.75)
    assert torch.allclose(r_default, r_explicit)
    assert not torch.allclose(r_default, r_other)


def test_sigma_anneals_from_one_to_three_tenths():
    """Exploration noise decays over training progress and is clamped at both ends."""
    assert sigma_at(0.0) == pytest.approx(1.0)
    assert sigma_at(0.5) == pytest.approx(0.65)
    assert sigma_at(1.0) == pytest.approx(0.3)
    assert sigma_at(1.7) == pytest.approx(0.3)
    assert sigma_at(-0.2) == pytest.approx(1.0)
    assert sigma_at(0.5, sigma=0.4, sigma_end=0.4) == pytest.approx(0.4)


def test_policy_gradient_climbs_the_reward():
    """The estimator must actually be an ascent direction on proper_reward: averaged over draws, a step
    along -g raises the mean reward."""
    logits, target, qtype, mask = _loss_batch(n=64, seed=3)
    g = torch.zeros_like(logits)
    for s in range(64):
        g += _grad(logits, target, qtype, mask, noise_seed=s)
    g /= 64

    def mean_reward(z):
        q = torch.softmax(z, -1)
        return float(proper_reward(q, target, qtype, mask, w_sph=0.5, w_rps=1.0).mean())

    assert mean_reward(logits - 0.5 * g / g.norm() * logits.numel() ** 0.5) > mean_reward(logits)
