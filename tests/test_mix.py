"""The training sampler's mix (``laya.vlm_train.group_weights``): no processor, no downloads."""
import math
import random

import pytest

from laya.vlm_train import ItemStream, group_weights, mix_probabilities

GROUPS = {"a": list(range(100)), "b": list(range(400)), "c": list(range(25))}


def test_equal_by_default():
    assert group_weights(GROUPS) == {"a": 1.0, "b": 1.0, "c": 1.0}
    assert mix_probabilities(GROUPS) == pytest.approx({"a": 1 / 3, "b": 1 / 3, "c": 1 / 3})


def test_weights_multiply():
    w = group_weights(GROUPS, {"a": 2.0, "b": 0.5})
    assert w == {"a": 2.0, "b": 0.5, "c": 1.0}
    p = mix_probabilities(GROUPS, {"a": 2.0, "b": 0.5})
    assert p == pytest.approx({"a": 2 / 3.5, "b": 0.5 / 3.5, "c": 1 / 3.5})
    assert group_weights(GROUPS, {"zzz": 9.0}) == {"a": 1.0, "b": 1.0, "c": 1.0}  # unknown names are ignored


def test_alpha_one_is_proportional_to_size():
    assert group_weights(GROUPS, size_alpha=1.0) == {"a": 100.0, "b": 400.0, "c": 25.0}
    assert mix_probabilities(GROUPS, size_alpha=1.0) == pytest.approx({"a": 100 / 525, "b": 400 / 525, "c": 25 / 525})


def test_alpha_half_is_sqrt():
    assert group_weights(GROUPS, size_alpha=0.5) == pytest.approx({"a": 10.0, "b": 20.0, "c": 5.0})
    w = group_weights(GROUPS, {"c": 3.0}, size_alpha=0.5)
    assert w == pytest.approx({"a": 10.0, "b": 20.0, "c": 15.0})
    assert w["b"] == pytest.approx(math.sqrt(400))


def test_item_stream_uses_the_weights():
    """The stream's per-group weights follow ``group_weights``, and the default reproduces equal sampling."""
    examples = [{"dataset": k, "i": i} for k, g in GROUPS.items() for i in g]
    s = ItemStream(None, examples, weights={"a": 2.0}, size_alpha=0.5)
    assert s.keys == ["a", "b", "c"]
    assert s.weights == pytest.approx({"a": 20.0, "b": 20.0, "c": 5.0})
    assert ItemStream(None, examples).weights == {"a": 1.0, "b": 1.0, "c": 1.0}
    # the draw itself: with rng.choices the counts follow the weights (a, b, c = 4:4:1)
    rng = random.Random(0)
    keys = list(s.keys)
    draws = rng.choices(keys, weights=[s.weights[k] for k in keys], k=9000)
    counts = {k: draws.count(k) for k in keys}
    assert abs(counts["a"] - 4000) < 200 and abs(counts["b"] - 4000) < 200 and abs(counts["c"] - 1000) < 150
