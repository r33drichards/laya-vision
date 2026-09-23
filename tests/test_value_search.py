"""Value head (``VLMDecisionModel.value_head``) and test-time search (``laya.search``). CPU is enough; downloads
HuggingFaceTB/SmolVLM-256M-Instruct like tests/test_vlm.py."""
import copy
import math

import numpy as np
import pytest
import torch
from PIL import Image

from laya import search
from laya.vlm import VLMAgent, collate_vlm
from laya.vlm_train import collect_logits, make_item, synthetic_examples, train, value_loss

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BACKBONE = "HuggingFaceTB/SmolVLM-256M-Instruct"


def fresh(seed=0, **kw):
    torch.manual_seed(seed)
    return VLMAgent(backbone=BACKBONE, device=DEVICE, **kw)


@pytest.fixture(scope="module")
def plain():
    return fresh()


@pytest.fixture(scope="module")
def valued():
    return fresh(value_head=True)


def square(color):
    img = Image.new("RGB", (96, 96), (255, 255, 255))
    img.paste(Image.new("RGB", (48, 48), color), (24, 24))
    return img


Q = {"action": {"type": "choice", "instructions": "Which way should the red square move to reach the right end?",
                "criteria": {"left": "move left", "right": "move right"}}}


class Corridor:
    """1-D corridor of ``n`` cells; the red square starts at ``pos``; reaching the last cell pays 1 and ends."""

    actions = ("left", "right")

    def __init__(self, n=5, pos=2):
        self.n, self.pos, self.done, self.steps = n, pos, False, 0

    def clone(self):
        return copy.deepcopy(self)

    def step(self, action):
        self.pos = max(0, min(self.n - 1, self.pos + (1 if action == "right" else -1)))
        self.steps += 1
        self.done = self.pos == self.n - 1
        return (1.0 if self.done else 0.0), self.done

    def render(self):
        img = Image.new("RGB", (32 * self.n, 32), (255, 255, 255))
        img.paste(Image.new("RGB", (32, 32), (220, 20, 20)), (32 * self.pos, 0))
        return img


# -- value head -----------------------------------------------------------------------------------------------


def test_value_head_off_changes_nothing(plain, valued):
    assert plain.model.value_head is None and "value_head" not in plain.cfg
    assert not any(k.startswith("value_head.") for k in plain.model.state_dict())
    extra = set(valued.model.state_dict()) - set(plain.model.state_dict())
    assert extra and all(k.startswith("value_head.") for k in extra)
    # built from the same seed, everything the two share is the same, so the policy is bit-identical
    sd_v = valued.model.state_dict()
    for k, v in plain.model.state_dict().items():
        assert torch.equal(v, sd_v[k]), k
    state = {"image": square((220, 20, 20))}
    a, b = plain.predict(state, Q), valued.predict(state, Q)
    assert "value" not in a and "value" not in a["answers"]["action"]
    assert plain.model.last_value is None
    assert 0.0 <= b["value"] <= 1.0 and b["answers"]["action"]["value"] == b["value"]
    b_answers = {k: {kk: vv for kk, vv in v.items() if kk != "value"} for k, v in b["answers"].items()}
    assert a["answers"] == b_answers


@pytest.mark.parametrize("include_backbone", [False, True])
def test_value_head_roundtrip(valued, tmp_path, include_backbone):
    with torch.no_grad():  # make the head's output distinctive
        valued.model.value_head[-1].bias.fill_(0.7)
    state = {"image": square((20, 20, 220))}
    before = valued.predict(state, Q)
    valued.save(str(tmp_path), include_backbone=include_backbone)
    loaded = VLMAgent(str(tmp_path), device=DEVICE)
    assert loaded.cfg["value_head"] is True and loaded.model.value_head is not None
    after = loaded.predict(state, Q)
    assert after["value"] == before["value"] and after["answers"] == before["answers"]


def test_value_head_can_be_added_to_a_checkpoint_without_one(plain, tmp_path):
    plain.save(str(tmp_path), include_backbone=False)
    with pytest.warns(UserWarning, match="no value head"):
        a = VLMAgent(str(tmp_path), device=DEVICE, value_head=True)
    assert a.model.value_head is not None
    # and a checkpoint with a value head refuses a model without one
    a.save(str(tmp_path / "v"), include_backbone=False)
    with pytest.raises(ValueError, match="value_head"):
        VLMAgent(str(tmp_path / "v"), device=DEVICE, value_head=False)


def test_value_targets_flow_through_items_and_collate(plain):
    exs = synthetic_examples(2)
    exs[0] = dict(exs[0], value=0.25)
    import random

    items = [make_item(plain.processor, ex, random.Random(0)) for ex in exs[:3]]
    assert items[0]["value"] == 0.25 and "value" not in items[1]
    b = collate_vlm(items, plain.processor.tokenizer.pad_token_id)
    assert b["value"][0].item() == 0.25 and torch.isnan(b["value"][1:]).all()
    assert "value" not in collate_vlm(items[1:], plain.processor.tokenizer.pad_token_id)
    with pytest.raises(ValueError):
        make_item(plain.processor, dict(exs[1], value=1.5), random.Random(0))


def test_value_loss_only_touches_rows_with_a_value():
    logits = torch.zeros(4, requires_grad=True)
    target = torch.tensor([1.0, float("nan"), 0.0, float("nan")])
    loss = value_loss(logits, target, 2.0)
    loss.backward()
    assert math.isclose(loss.item(), 2.0 * math.log(2), rel_tol=1e-6)
    assert logits.grad[1] == 0 and logits.grad[3] == 0 and logits.grad[0] < 0 < logits.grad[2]
    assert value_loss(logits, torch.full((4,), float("nan"))) == 0.0
    assert value_loss(None, target) == 0.0 and value_loss(logits, None) == 0.0 and value_loss(logits, target, 0) == 0.0


def test_train_without_values_is_unchanged_and_with_values_trains_the_head(tmp_path):
    exs = synthetic_examples(2)
    a, b = fresh(1), fresh(1, value_head=True)
    la = train(a.model, a.processor, exs, steps=2, batch_size=2, device=DEVICE, seed=3, log_every=0)
    v0 = {k: v.clone() for k, v in b.model.value_head.state_dict().items()}
    lb = train(b.model, b.processor, exs, steps=2, batch_size=2, device=DEVICE, seed=3, log_every=0)
    assert la == lb  # the head exists but no example has a value: exactly the old objective
    assert all(torch.equal(v, b.model.value_head.state_dict()[k]) for k, v in v0.items())
    valued_exs = [dict(ex, value=float(i % 2)) for i, ex in enumerate(exs)]
    lc = train(b.model, b.processor, valued_exs, steps=2, batch_size=2, device=DEVICE, seed=3, log_every=0)
    assert all(math.isfinite(x) for x in lc)
    assert not all(torch.equal(v, b.model.value_head.state_dict()[k]) for k, v in v0.items())
    recs = collect_logits(b.model, b.processor, valued_exs[:2], batch_size=2)
    assert all("value_logit" in r and "value_target" in r for r in recs)
    del a, b


# -- search -----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("backend", ["gpu", "processor"])
def test_score_states_matches_predict(backend, plain):
    agent = plain if backend == "gpu" else fresh(preprocess="processor")
    assert agent.prep.backend == backend
    imgs = [Corridor(pos=p).render() for p in (0, 2, 2, 4)]
    stats = {}
    out = search.score_states(agent, imgs, Q, stats=stats)
    assert stats == {"forwards": 1, "rows": 3}  # the repeated frame is scored once
    for img, p in zip(imgs, out["probs"]):
        ref = agent.predict({"image": img}, Q)["answers"]["action"]["probabilities"]
        np.testing.assert_allclose(p, [ref["left"], ref["right"]], atol=2e-4)
    assert out["value"] is None


def test_lookahead_takes_the_rewarding_move_in_one_forward(valued):
    envs = [Corridor(pos=3), Corridor(pos=2), Corridor(pos=0)]
    envs[2].done = True
    calls = []
    orig = valued.model.forward
    valued.model.forward = lambda *a, **k: calls.append(1) or orig(*a, **k)
    try:
        stats = {}
        acts = search.plan(valued, envs, Q, {"kind": "lookahead", "depth": 2, "leaf": "zero", "c_prior": 0.0}, stats)
    finally:
        del valued.model.forward
    assert len(calls) == 1 and stats["forwards"] == 1
    assert acts[0] == "right" and acts[1] == "right" and acts[2] == "left"  # a done env is not searched
    assert [e.pos for e in envs] == [3, 2, 0]  # search ran on clones
    assert stats["q"][0][1] == 1.0 and stats["q"][1] == [0.0, 1.0]


def _fake_scores(values):
    """``score_states`` stand-in: uniform policy, value = ``values[pos]`` read off the rendered corridor."""
    def fake(agent, images, question, batch_size=64, stats=None):
        pos = [int(np.asarray(im)[16, :, 1].argmin()) // 32 for im in images]
        if stats is not None:
            stats["forwards"] = stats.get("forwards", 0) + 1
        return {"probs": np.full((len(images), 2), 0.5), "value": np.array([values[p] for p in pos], float)}
    return fake


@pytest.mark.parametrize("settings", [{"kind": "lookahead"}, {"kind": "lookahead", "depth": 3},
                                      {"kind": "puct", "sims": 8}])
def test_search_follows_the_value(monkeypatch, settings):
    # value by position; the goal (cell 5) pays 1. From 1 the value says left; from 3 it says right (and right
    # also leads to the goal within reach of the deeper searches)
    monkeypatch.setattr(search, "score_states", _fake_scores([0.9, 0.1, 0.2, 0.3, 0.6, 0.0]))
    stats = {}
    acts = search.plan(_StubAgent(), [Corridor(n=6, pos=1), Corridor(n=6, pos=3)], Q, dict(settings, c_prior=0.0),
                       stats)
    assert stats["forwards"] == (1 + settings["sims"] if settings["kind"] == "puct" else 1)
    assert acts == ["left", "right"]


class _StubAgent:
    cfg = {}

    @staticmethod
    def _to_internal(qdef):
        return VLMAgent._to_internal(qdef)


def test_puct_with_the_real_model(valued):
    stats = {}
    acts = search.plan(valued, [Corridor(pos=3), Corridor(pos=1)], Q, {"kind": "puct", "sims": 4}, stats)
    assert set(acts) <= {"left", "right"} and stats["forwards"] <= 5
    assert all(sum(v) == 4 for v in stats["visits"])
