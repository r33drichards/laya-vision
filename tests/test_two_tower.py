"""CPU tests for laya/two_tower.py: option-order invariance of both scorers, and the item split (no downloads)."""
from types import SimpleNamespace

import torch
import torch.nn as nn

from laya.two_tower import TwoTowerScorer, collate_two_tower, split_item


class _Vision(nn.Module):
    def forward(self, pixel_values):
        return pixel_values


def _stub_encoder(d=64):
    enc = nn.Module()
    enc.config = SimpleNamespace(text_config=SimpleNamespace(hidden_size=d))
    enc.vision_model = _Vision()
    return enc


def _head(d=64):
    layer = nn.TransformerEncoderLayer(d, 2, 4 * d, 0.0, batch_first=True, norm_first=True)
    return nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)


def _check_equivariant(model):
    torch.manual_seed(0)
    B, P, K, d = 2, 7, 5, 64
    h = torch.randn(B, P, d)
    pmask = torch.ones(B, P, dtype=torch.long)
    pmask[1, 5:] = 0
    opts = model.option_cache(torch.randn(B, K, d))
    mm = torch.ones(B, K, dtype=torch.bool)
    mm[1, 4] = False
    qt = torch.tensor([0, 1])
    perm = torch.tensor([3, 0, 2, 1, 4])  # keeps the padded slot last
    a = model.score(h, pmask, qt, opts, mm)
    b = model.score(h, pmask, qt, opts[:, perm], mm[:, perm])
    assert torch.allclose(a[:, perm], b, atol=1e-5)
    assert (a[1, 4] == -1e4) and (b[1, 4] == -1e4)
    # padded state tokens change nothing
    h2 = h.clone()
    h2[1, 5:] = 100.0
    assert torch.allclose(model.score(h2, pmask, qt, opts, mm), a, atol=1e-5)


def test_tt_order_invariant():
    _check_equivariant(TwoTowerScorer(_stub_encoder(), "tt", proj_dim=32).eval())


def test_li_order_invariant():
    d = 64
    scorer = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
    m = TwoTowerScorer(_stub_encoder(d), "li", head=_head(d), scorer=scorer, type_emb=nn.Embedding(3, d)).eval()
    _check_equivariant(m)


def test_split_item_and_collate():
    # prefix [1 2 3], options "- a" = [9, 10], "- bb" = [9, 11, 11], terminator 0
    it = {"ids": [1, 2, 3, 9, 10, 0, 9, 11, 11, 0], "markers": [5, 9], "option_span": (3, 10), "qtype": 0,
          "target": [0.0, 1.0], "label": 1, "n_images": 0}
    s = split_item(it)
    assert s["prefix"] == [1, 2, 3]
    assert s["options"] == [(9, 10), (9, 11, 11)]
    b = collate_two_tower([it, dict(it, markers=[5], option_span=(3, 6), ids=it["ids"][:6], target=[1.0])],
                          pad_id=99, ctx_ids=[7], end_id=0)
    assert b["opt_ids"].tolist() == [[7, 9, 10, 0, 99], [7, 9, 11, 11, 0]]
    assert b["opt_index"].tolist() == [[0, 1], [0, 0]]
    assert b["prefix_ids"].tolist() == [[1, 2, 3], [1, 2, 3]]
