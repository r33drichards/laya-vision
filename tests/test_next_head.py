"""Auxiliary next-move head (``VLMDecisionModel.next_head``, KataGo arXiv 1902.10565 section 3.4): off by default and
then invisible, round-trips through checkpoints, takes ``next_target`` through make_item / collate / train, and is
never exposed by ``predict``. CPU is enough; downloads HuggingFaceTB/SmolVLM-256M-Instruct like tests/test_vlm.py."""
import math
import random

import pytest
import torch
from PIL import Image

from laya.vlm import VLMAgent, collate_vlm
from laya.vlm_train import collect_logits, make_item, next_loss, synthetic_examples, train

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BACKBONE = "HuggingFaceTB/SmolVLM-256M-Instruct"


def fresh(seed=0, **kw):
    torch.manual_seed(seed)
    return VLMAgent(backbone=BACKBONE, device=DEVICE, **kw)


@pytest.fixture(scope="module")
def plain():
    return fresh()


@pytest.fixture(scope="module")
def nexted():
    return fresh(next_head=True)


def square(color):
    img = Image.new("RGB", (96, 96), (255, 255, 255))
    img.paste(Image.new("RGB", (48, 48), color), (24, 24))
    return img


Q = {"action": {"type": "choice", "instructions": "Which way should the red square move?",
                "criteria": {"up": "move up", "down": "move down", "left": "move left", "right": "move right"}}}


def _batch(agent, exs, seed=0):
    items = [make_item(agent.processor, ex, random.Random(seed + i)) for i, ex in enumerate(exs)]
    return items, collate_vlm(items, agent.processor.tokenizer.pad_token_id)


def _forward(agent, b):
    with torch.no_grad():
        return agent.model(b["input_ids"].to(agent.device), b["attention_mask"].to(agent.device),
                           b["marker_pos"].to(agent.device), b["marker_mask"].to(agent.device),
                           b["qtype"].to(agent.device), raw_pixels=_dev(b["raw_pixels"], agent),
                           pixel_values=_dev(b["pixel_values"], agent),
                           pixel_attention_mask=_dev(b["pixel_attention_mask"], agent),
                           image_mask=_dev(b["image_mask"], agent), option_span=b["option_span"].to(agent.device))


def _dev(x, agent):
    return None if x is None else x.to(agent.device)


def test_next_head_off_changes_nothing(plain, nexted):
    assert plain.model.next_head is None and "next_head" not in plain.cfg
    assert not any(k.startswith("next_head.") for k in plain.model.state_dict())
    extra = set(nexted.model.state_dict()) - set(plain.model.state_dict())
    assert extra and all(k.startswith("next_head.") for k in extra)
    # built from the same seed, everything the two share is the same, so the policy is bit-identical
    sd_n = nexted.model.state_dict()
    for k, v in plain.model.state_dict().items():
        assert torch.equal(v, sd_n[k]), k
    state = {"image": square((220, 20, 20))}
    a, b = plain.predict(state, Q), nexted.predict(state, Q)
    assert plain.model.last_next is None
    assert a == b  # predict never exposes the next head
    # and the raw forward outputs are bit-identical, with the next logits over the same options on the side
    exs = synthetic_examples(1)
    _, batch = _batch(plain, exs)
    la, aa = _forward(plain, batch)
    assert plain.model.last_next is None
    lb, ab = _forward(nexted, batch)
    assert torch.equal(la, lb) and torch.equal(aa, ab)
    nl = nexted.model.last_next
    assert nl.shape == lb.shape and torch.isfinite(nl).all()
    assert (nl[~batch["marker_mask"].to(nl.device)] == -1e4).all()


def test_value_and_next_heads_together_keep_the_value_head_bit_identical():
    a, b = fresh(2, value_head=True), fresh(2, value_head=True, next_head=True)
    sd_b = b.model.state_dict()
    for k, v in a.model.state_dict().items():
        assert torch.equal(v, sd_b[k]), k
    state = {"image": square((20, 200, 20))}
    assert a.predict(state, Q) == b.predict(state, Q)
    del a, b


@pytest.mark.parametrize("include_backbone", [False, True])
def test_next_head_roundtrip(nexted, tmp_path, include_backbone):
    with torch.no_grad():  # make the head's output distinctive
        nexted.model.next_head[-1].bias.fill_(0.7)
        nexted.model.next_head[1].weight.mul_(3.0)
    _, batch = _batch(nexted, synthetic_examples(1))
    _forward(nexted, batch)
    before = nexted.model.last_next.clone()
    nexted.save(str(tmp_path), include_backbone=include_backbone)
    loaded = VLMAgent(str(tmp_path), device=DEVICE)
    assert loaded.cfg["next_head"] is True and loaded.model.next_head is not None
    _forward(loaded, batch)
    assert torch.equal(loaded.model.last_next, before)


def test_next_head_can_be_added_to_a_checkpoint_without_one(plain, tmp_path):
    plain.save(str(tmp_path), include_backbone=False)
    with pytest.warns(UserWarning, match="no next-move head"):
        a = VLMAgent(str(tmp_path), device=DEVICE, next_head=True)
    assert a.model.next_head is not None
    # and a checkpoint with a next head refuses a model without one
    a.save(str(tmp_path / "n"), include_backbone=False)
    with pytest.raises(ValueError, match="next_head"):
        VLMAgent(str(tmp_path / "n"), device=DEVICE, next_head=False)


def test_next_targets_flow_through_items_and_collate(plain):
    exs = synthetic_examples(2)
    exs[0] = dict(exs[0], next_target=[0.0, 2.0, 2.0])  # normalised by make_item
    items = [make_item(plain.processor, ex, random.Random(0)) for ex in exs[:3]]
    order = items[0]["order"]
    assert items[0]["next_target"] == [[0.0, 0.5, 0.5][i] for i in order] and "next_target" not in items[1]
    b = collate_vlm(items, plain.processor.tokenizer.pad_token_id)
    nt = b["next_target"]
    assert nt.shape == b["marker_mask"].shape
    assert nt[0, :3].tolist() == items[0]["next_target"] and (nt[0, 3:] == 0).all()
    assert torch.isnan(nt[1:]).all()
    assert "next_target" not in collate_vlm(items[1:], plain.processor.tokenizer.pad_token_id)
    for bad in ([0.5, 0.5], [-1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [float("nan"), 1.0, 0.0]):
        with pytest.raises(ValueError):
            make_item(plain.processor, dict(exs[0], next_target=bad), random.Random(0))


def test_next_loss_only_touches_rows_with_a_target():
    logits = torch.tensor([[0.0, 0.0, -1e4], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], requires_grad=True)
    nan = float("nan")
    target = torch.tensor([[1.0, 0.0, 0.0], [nan, nan, nan], [0.25, 0.25, 0.5]])
    loss = next_loss(logits, target, 2.0)
    loss.backward()
    want = 2.0 * (math.log(2) + math.log(3)) / 2  # soft cross-entropy averaged over the two rows with a target
    assert math.isclose(loss.item(), want, rel_tol=1e-6)
    assert (logits.grad[1] == 0).all() and logits.grad[0, 0] < 0 < logits.grad[0, 1]
    assert next_loss(logits, torch.full((3, 3), nan)) == 0.0
    assert next_loss(None, target) == 0.0 and next_loss(logits, None) == 0.0 and next_loss(logits, target, 0) == 0.0


def test_train_without_next_targets_is_unchanged_and_with_them_trains_the_head():
    exs = synthetic_examples(2)
    nexted_exs = [dict(ex, next_target=[1.0] * len(ex["target"])) for ex in exs]
    a, b = fresh(1), fresh(1, next_head=True)
    la = train(a.model, a.processor, exs, steps=2, batch_size=2, device=DEVICE, seed=3, log_every=0)
    n0 = {k: v.clone() for k, v in b.model.next_head.state_dict().items()}
    lb = train(b.model, b.processor, exs, steps=2, batch_size=2, device=DEVICE, seed=3, log_every=0)
    assert la == lb  # the head exists but no example has a next_target: exactly the old objective
    assert all(torch.equal(v, b.model.next_head.state_dict()[k]) for k, v in n0.items())
    c = fresh(1)  # no head: next targets in the data change nothing
    lc = train(c.model, c.processor, nexted_exs, steps=2, batch_size=2, device=DEVICE, seed=3, log_every=0)
    assert lc == la
    del a, c
    ld = train(b.model, b.processor, nexted_exs, steps=2, batch_size=2, device=DEVICE, seed=3, log_every=0,
               w_next=0.15)
    assert all(math.isfinite(x) for x in ld)
    assert not all(torch.equal(v, b.model.next_head.state_dict()[k]) for k, v in n0.items())
    recs = collect_logits(b.model, b.processor, nexted_exs[:2], batch_size=2)
    assert all("next" not in k for r in recs for k in r)  # evaluation records carry no next-head output
    del b
