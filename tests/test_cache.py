"""laya.cache: the cross-call prefix / image-feature arena. The model tests download SmolVLM-256M (~0.5 GB)."""
import numpy as np
import pytest
import torch
from PIL import Image

from laya.cache import BlockPool, DeviceArena, image_digest, parse_budget
from laya.vlm import VLMAgent, split_state, vlm_prefix

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

QUESTIONS = {
    "color": {"type": "choice", "instructions": "What color is the square?",
              "criteria": {"red": "the square is red", "blue": "the square is blue", "green": "the square is green"}},
    "size": {"type": "score", "instructions": "How much of the image does the square fill?",
             "criteria": ["tiny", "about half", "almost all"]},
    "is_red": {"type": "noul", "instructions": "Is the square red?"},
}


def square(color, size=96):
    img = Image.new("RGB", (size, size), (255, 255, 255))
    img.paste(Image.new("RGB", (size // 2, size // 2), color), (size // 4, size // 4))
    return img


def probs(res):
    out = []
    for qid, a in sorted(res["answers"].items()):
        out += list(a["probabilities"].values()) if "probabilities" in a else [a["noul"]]
        out.append(a["action"]["act_probability"])
    return np.array(out)


def close(a, b, atol=2e-4):  # 1e-4 of float rounding plus the 4-decimal rounding of the answers
    assert a["usage"] == b["usage"]
    gap = float(np.abs(probs(a) - probs(b)).max())
    assert gap <= atol, gap


# --------------------------------------------------------------------------------------------- no model needed


def test_parse_budget():
    assert parse_budget("512MiB", 0) == 512 << 20
    assert parse_budget("2GB", 0) == 2 * 10**9
    assert parse_budget(0.5, 1000) == 500
    assert parse_budget(None, 10**6) == 20000  # the default is 2% of the device
    for bad in ("1.5", "12 parsecs", "lots"):
        with pytest.raises(ValueError):
            parse_budget(bad, 1000)


def test_block_pool_is_lru_and_never_grows():
    buf = torch.zeros(4, 2)
    pool = BlockPool([buf])
    assert pool.claim(b"a", 2, 50) == [0, 1]
    assert pool.claim(b"b", 1, 10) == [2]
    pool.get(b"a")                                   # a is now the most recently used
    blocks = pool.claim(b"c", 2, 40)                 # needs 2, 1 free: evicts b (LRU), not a
    assert sorted(blocks) == [2, 3] and set(pool.entries) == {b"a", b"c"} and pool.evictions == 1
    assert pool.claim(b"d", 5, 1) is None and pool.skipped == 1  # bigger than the whole pool: served uncached
    assert pool.claim(b"a", 1, 5) is not None and pool.used_blocks() == 3  # re-storing a key frees its old blocks
    assert pool.buffers[0].data_ptr() == buf.data_ptr() and pool.capacity == 4


def test_image_digest_sees_every_pixel():
    a = np.zeros((8, 8, 3), np.uint8)
    b = a.copy()
    b[7, 7, 2] = 1
    assert image_digest(a) == image_digest(a.copy()) != image_digest(b)
    assert image_digest(Image.fromarray(a)) != image_digest(Image.fromarray(b))
    assert image_digest(a) != image_digest(a.reshape(8, 3, 8))  # same bytes, different shape


# --------------------------------------------------------------------------------------------- with SmolVLM


@pytest.fixture(scope="module")
def agent():
    torch.manual_seed(0)
    return VLMAgent(backbone="HuggingFaceTB/SmolVLM-256M-Instruct", device=DEVICE)


@pytest.fixture
def arena(agent):
    return DeviceArena(agent, budget="64MiB", prefix_share=0.875)


def test_arena_is_claimed_once_and_sized_by_the_budget(agent):
    a = DeviceArena(agent, budget="64MiB", prefix_share=0.875)
    st = a.stats()
    assert 0.9 * 64 * 2**20 / 1e6 <= st["reserved_mb"] <= 64 * 2**20 / 1e6
    assert st["prefix"]["blocks"] > 0 and st["image"]["blocks"] > 0
    assert a.image_tokens == agent.prep.image_seq_len


@pytest.mark.parametrize("backend", ["processor", "gpu"])
def test_prefix_ids_match_vlm_prefix(agent, arena, backend):
    """The arena builds the prefix ids without touching pixels; they must be the ones ``vlm_prefix`` writes."""
    import dataclasses

    old = agent.prep
    agent.prep = dataclasses.replace(old, backend=backend)
    try:
        for images in ([], [square((220, 20, 20))], [square((220, 20, 20)), square((20, 40, 220), 64)]):
            if backend == "gpu" and len({im.size for im in images}) > 1:
                continue
            got, _, digests = arena.prefix_ids(agent, images)
            assert got["ids"] == vlm_prefix(agent.processor, images, agent.prep)["ids"]
            assert len(digests) == len(images)
    finally:
        agent.prep = old


@pytest.mark.parametrize("attention", ["causal", "block"])
def test_agent_loop_hits_and_matches_uncached(agent, arena, attention):
    """Same image + state, different question sets: every call after the first reads the stored prefix, answers
    match the uncached path to rounding, and a repeated call is bit-identical to the one that stored the entry."""
    agent.model.option_attention = attention
    try:
        state = {"image": square((220, 20, 20)), "caption": "a test card from the returns desk"}
        sets = [["color"], ["size", "is_red"], ["is_red"], ["color", "size", "is_red"]]
        first = None
        for i, s in enumerate(sets):
            qs = {q: QUESTIONS[q] for q in s}
            got = agent.predict(state, qs, prefix_cache=True, cache=arena)
            close(got, agent.predict(state, qs))
            first = first or got
        again = agent.predict(state, {"color": QUESTIONS["color"]}, prefix_cache=True, cache=arena)
        assert np.array_equal(probs(again), probs(first))  # bit for bit: the stored tensors, copied back
        st = arena.stats()
        assert st["prefix"]["hits"] == len(sets) and st["prefix"]["misses"] == 1 and st["prefix"]["entries"] == 1
        assert st["image"]["misses"] == 1 and st["image"]["hits"] == 0  # a prefix hit never asks for features
    finally:
        agent.model.option_attention = "causal"


def test_permutations_and_batch_splits_share_the_entry(agent, arena):
    """The prefix ends before the question, so option orders and suffix chunks all reuse one entry."""
    state = {"image": square((20, 40, 220)), "note": "blue one"}
    ref = agent.predict(state, QUESTIONS, n_permutations=4, batch_size=2, prefix_cache=False)
    for perms, bs in ((1, 8), (4, 2), (2, 1)):
        got = agent.predict(state, QUESTIONS, n_permutations=perms, batch_size=bs, prefix_cache=True, cache=arena)
        if perms == 4:
            close(got, ref)
    st = arena.stats()["prefix"]
    assert st["misses"] == 1 and st["hits"] == 2 and st["entries"] == 1


def test_changing_state_text_hits_the_image_tier(agent, arena):
    """A game-like loop: the frame repeats, the state text around it changes, so only the image features hit."""
    frame = np.asarray(square((220, 20, 20), 64))
    qs = {"is_red": QUESTIONS["is_red"]}
    for step in range(3):
        state = {"image": frame, "step": step}
        close(agent.predict(state, qs, cache=arena), agent.predict(state, qs))
    st = arena.stats()
    assert st["image"]["misses"] == 1 and st["image"]["hits"] == 2
    assert st["prefix"]["hits"] == 0


def test_auto_admission_stores_on_the_second_sighting(agent, arena):
    """With ``prefix_cache=None`` a single-row call does not prefill, so the prefix is stored the second time."""
    state = {"image": square((20, 200, 20)), "caption": "green"}
    qs = {"is_red": QUESTIONS["is_red"]}
    calls = []
    prefill = agent.model.encode_prefix
    agent.model.encode_prefix = lambda *a: calls.append(1) or prefill(*a)
    try:
        outs = [agent.predict(state, qs, cache=arena) for _ in range(4)]
    finally:
        del agent.model.encode_prefix
    assert calls == [1]  # first sighting: full path; second: prefill + store; then hits
    st = arena.stats()["prefix"]
    assert st["hits"] == 2 and st["entries"] == 1
    assert np.array_equal(probs(outs[2]), probs(outs[1])) and np.array_equal(probs(outs[3]), probs(outs[1]))
    close(outs[0], outs[1])
    # prefix_cache=False leaves the prefix tier alone even when it holds the entry
    agent.predict(state, qs, prefix_cache=False, cache=arena)
    assert arena.stats()["prefix"]["hits"] == 2


def test_text_only_and_multi_image_states(agent, arena):
    for state in ("Customer: I was billed twice, please refund.",
                  {"images": [square((220, 20, 20)), square((20, 40, 220))], "note": "two views"},
                  {"images": [square((220, 20, 20)), square((220, 20, 20))], "note": "the same view twice"}):
        ref = agent.predict(state, QUESTIONS, prefix_cache=True)
        for _ in range(2):
            close(agent.predict(state, QUESTIONS, prefix_cache=True, cache=arena), ref)
    # the repeated image is encoded once; the first two-image state's features are reused by the third
    assert arena.stats()["image"]["misses"] == 2


def test_a_weight_change_stops_matching(agent, arena):
    """The namespace carries the checkpoint's identity: modifying a weight in place orphans the old entries."""
    state = {"image": square((220, 20, 20)), "caption": "x"}
    qs = {"is_red": QUESTIONS["is_red"]}
    agent.predict(state, qs, prefix_cache=True, cache=arena)
    w = agent.model.encoder.get_input_embeddings().weight
    with torch.no_grad():
        w[0, 0] += 1.0
    try:
        agent.predict(state, qs, prefix_cache=True, cache=arena)
        st = arena.stats()
        assert st["prefix"]["hits"] == 0 and st["prefix"]["misses"] == 2 and st["image"]["misses"] == 2
    finally:
        with torch.no_grad():
            w[0, 0] -= 1.0


def test_small_arena_evicts_and_never_grows(agent):
    small = DeviceArena(agent, budget="16MiB", prefix_share=0.9)
    ptr, n = small.flat.data_ptr(), small.flat.numel()
    colors = [(220, 20, 20), (20, 200, 20), (20, 40, 220), (200, 200, 20), (20, 200, 200)]
    for c in colors:
        agent.predict({"image": square(c), "caption": "a card " * 20}, {"is_red": QUESTIONS["is_red"]},
                      prefix_cache=True, cache=small)
    st = small.stats()
    assert st["prefix"]["evictions"] > 0 and st["prefix"]["used_blocks"] <= st["prefix"]["blocks"]
    assert small.flat.data_ptr() == ptr and small.flat.numel() == n
    # the most recent state is still there
    agent.predict({"image": square(colors[-1]), "caption": "a card " * 20}, {"is_red": QUESTIONS["is_red"]},
                  prefix_cache=True, cache=small)
    assert small.stats()["prefix"]["hits"] == 1


def test_split_state_images_are_what_is_hashed(agent):
    images, _ = split_state({"image": square((1, 2, 3))})
    assert image_digest(images[0]) == image_digest(square((1, 2, 3)).convert("RGB"))


def test_image_only_arena_is_bit_identical_to_no_arena(agent):
    """With the prefix tier off, the call runs as it would without an arena (in-call prefix or full path) on the
    same image features: bit-identical answers, on the storing call and on the hits."""
    only = DeviceArena(agent, budget="16MiB")  # the default: image tier only
    assert only.kv is None and only.images is not None
    state = {"image": square((20, 200, 20)), "caption": "green"}
    for qs in (QUESTIONS, {"is_red": QUESTIONS["is_red"]}):  # several rows (in-call prefix on a CPU), then one
        ref = agent.predict(state, qs)
        for _ in range(2):
            assert np.array_equal(probs(agent.predict(state, qs, cache=only)), probs(ref))
    assert only.stats()["image"]["hits"] == 3
