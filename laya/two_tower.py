"""Experimental two-tower / late-interaction scoring for the SmolVLM decision model (not used by default).

The cross-encoder (``laya.vlm.VLMDecisionModel``) puts image + state + question + every option in one sequence and
reads each option at its terminator. This module scores options without ever putting them in the image's sequence:

* **state tower**: the cross-encoder's own sequence cut before the options,
  ``<|im_start|>User:<image x64><state>\\n<type> question: <ins><end_of_utterance>\\nAssistant: Options:\\n``, through
  the backbone (image included). Causal, so these hidden states do not depend on the options at all.
* **option tower**: each option alone, text only, ``OPTION_CONTEXT + "- <option>\\n"``, through the same backbone's
  language model, pooled at the closing ``\\n`` (the token the cross-encoder reads). It sees no image, no state and
  no question, so an option's vector can be computed once and cached (per option string, for ever).

Two ways to score a (state, option) pair:

* ``mode="tt"`` (pure two-tower, CLM-style): ``exp(logit_scale) * cos(P_s(h_last), P_a(h_opt))`` with MLP
  projections; the state is the last prefix token's hidden state. One dot product per option.
* ``mode="li"`` (late interaction): the cached option vectors become one token each, appended to the state tower's
  hidden states, and the cross-encoder's 2-layer head transformer (initialised from the teacher's weights) runs over
  them with a mask: state tokens attend to state tokens only, option tokens to the state and to every option. The
  head has no positional encoding, so the option logits are permutation-equivariant (order-invariant) by
  construction, while options can still compare themselves with each other. The teacher's scorer reads each option
  token.

Both are order-invariant by construction (up to floating-point summation order) and have no option-count ceiling
beyond memory. Trained by distillation from a cross-encoder checkpoint (``modal_two_tower.py``).
"""
import copy
import functools
import random
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import QTYPES
from .vlm import collate_vlm

OPTION_CONTEXT = "<|im_start|>Assistant: Options:\n"
MODES = ("tt", "li")


def split_item(item: Dict) -> Dict:
    """A cross-encoder item (``laya.vlm_train.make_item``) -> its state prefix ids and each option's token ids.

    The prefix is everything before the option span; option j (marker order) is the tokens between the previous
    terminator and its own, i.e. ``enc("- " + option)`` exactly as the cross-encoder saw it (same truncation)."""
    s = item["option_span"][0]
    ids = item["ids"]
    opts, prev = [], s
    for m in item["markers"]:
        opts.append(tuple(ids[prev:m]))
        prev = m + 1
    return {"prefix": ids[:s], "options": opts}


def option_sequences(options: Sequence[Sequence[int]], ctx_ids: Sequence[int], end_id: int) -> List[List[int]]:
    return [list(ctx_ids) + list(o) + [end_id] for o in options]


def _pad(rows: List[List[int]], pad_id: int):
    L = max(len(r) for r in rows)
    ids = torch.full((len(rows), L), pad_id, dtype=torch.long)
    att = torch.zeros((len(rows), L), dtype=torch.long)
    for i, r in enumerate(rows):
        ids[i, : len(r)] = torch.tensor(r)
        att[i, : len(r)] = 1
    return ids, att


def collate_two_tower(items: List[Dict], pad_id: int, ctx_ids: Sequence[int], end_id: int) -> Dict:
    """``collate_vlm`` (the teacher's batch) plus the two towers' inputs: ``prefix_ids``/``prefix_mask`` [B, P]
    and the batch's unique options ``opt_ids``/``opt_mask`` [U, Lo] with ``opt_index`` [B, kmax] into them."""
    b = collate_vlm(items, pad_id)
    splits = [split_item(it) for it in items]
    b["prefix_ids"], b["prefix_mask"] = _pad([s["prefix"] for s in splits], pad_id)
    uniq: Dict[tuple, int] = {}
    kmax = b["marker_mask"].size(1)
    index = torch.zeros((len(items), kmax), dtype=torch.long)
    for i, s in enumerate(splits):
        for j, o in enumerate(s["options"]):
            index[i, j] = uniq.setdefault(o, len(uniq))
    b["opt_ids"], b["opt_mask"] = _pad(option_sequences(list(uniq), ctx_ids, end_id), pad_id)
    b["opt_index"] = index
    for k in ("dataset", "index", "order"):
        if k in items[0]:
            b[k] = [it[k] for it in items]
    return b


class TwoTowerScorer(nn.Module):
    """Shared SmolVLM backbone + a ``"tt"`` or ``"li"`` scoring head (see the module docstring)."""

    def __init__(self, encoder: nn.Module, mode: str, head: Optional[nn.Module] = None,
                 scorer: Optional[nn.Module] = None, type_emb: Optional[nn.Module] = None, proj_dim: int = 256):
        super().__init__()
        if mode not in MODES:
            raise ValueError("mode must be one of %s" % (MODES,))
        self.encoder, self.mode = encoder, mode
        d = encoder.config.text_config.hidden_size
        import inspect
        takes = inspect.signature(encoder.vision_model.forward).parameters
        self._vision_kw = {"interpolate_pos_encoding": True} if "interpolate_pos_encoding" in takes else {}
        if mode == "tt":
            mlp = lambda: nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, proj_dim))  # noqa: E731
            self.state_proj, self.opt_proj = mlp(), mlp()
            self.type_emb = nn.Embedding(3, d)
            nn.init.zeros_(self.type_emb.weight)
            self.logit_scale = nn.Parameter(torch.tensor(float(torch.log(torch.tensor(20.0)))))
        else:
            if head is None or scorer is None or type_emb is None:
                raise ValueError("mode='li' starts from a cross-encoder's head, scorer and type_emb")
            self.head, self.scorer, self.type_emb = copy.deepcopy(head), copy.deepcopy(scorer), copy.deepcopy(type_emb)
            self.opt_in = nn.Linear(d, d)
            with torch.no_grad():  # start as the identity: the option token is the option tower's terminator state
                self.opt_in.weight.copy_(torch.eye(d))
                self.opt_in.bias.zero_()

    # -- towers --------------------------------------------------------------------------------------------------

    def image_features(self, pixel_values, pixel_attention_mask):
        return self.encoder.get_image_features(pixel_values, pixel_attention_mask, return_dict=True,
                                               **self._vision_kw).pooler_output

    def encode_state(self, prefix_ids, prefix_mask, image_hidden_states=None) -> torch.Tensor:
        """[B, P] (right-padded) -> hidden states [B, P, d]; images merged at their tokens."""
        if image_hidden_states is not None:
            image_hidden_states = image_hidden_states.to(self.encoder.get_input_embeddings().weight.dtype)
        return self.encoder(input_ids=prefix_ids, attention_mask=prefix_mask, image_hidden_states=image_hidden_states,
                            use_cache=False).last_hidden_state

    def encode_options(self, opt_ids, opt_mask) -> torch.Tensor:
        """[U, Lo] option sequences (text only) -> the hidden state at each one's closing terminator [U, d]."""
        h = self.encoder.text_model(input_ids=opt_ids, attention_mask=opt_mask, use_cache=False).last_hidden_state
        last = (opt_mask.sum(-1) - 1).clamp(min=0)
        return h[torch.arange(h.size(0), device=h.device), last]

    def option_cache(self, opt_vec: torch.Tensor) -> torch.Tensor:
        """What gets cached per option: the projected, normalised vector (tt) or the option token (li)."""
        if self.mode == "tt":
            return F.normalize(self.opt_proj(opt_vec.float()), dim=-1)
        return self.opt_in(opt_vec.float())

    # -- scoring -------------------------------------------------------------------------------------------------

    def score(self, h_state, prefix_mask, qtype, opt_cached, marker_mask) -> torch.Tensor:
        """State hidden states [B, P, d] + cached options [B, K, *] -> option logits [B, K] (padding at -1e4)."""
        h_state = h_state.float()
        if self.mode == "tt":
            last = (prefix_mask.sum(-1) - 1).clamp(min=0)
            s = h_state[torch.arange(h_state.size(0), device=h_state.device), last] + self.type_emb(qtype)
            s = F.normalize(self.state_proj(s), dim=-1)
            logits = self.logit_scale.exp().clamp(max=100.0) * torch.einsum("bd,bkd->bk", s, opt_cached.float())
            return logits.masked_fill(~marker_mask, -1e4)
        B, P, _ = h_state.shape
        K = opt_cached.size(1)
        t = self.type_emb(qtype)[:, None, :]
        x = torch.cat([h_state + t, opt_cached.float() + t], 1)
        keys = torch.cat([prefix_mask.bool(), marker_mask], 1)                      # [B, P+K] real tokens
        is_opt_q = torch.zeros(P + K, dtype=torch.bool, device=x.device)
        is_opt_q[P:] = True
        key_is_opt = is_opt_q[None, None, :]
        # state queries see state keys only; option queries see the state and every (real) option
        allowed = keys[:, None, :] & (is_opt_q[None, :, None] | ~key_is_opt)
        eye = torch.eye(P + K, dtype=torch.bool, device=x.device)[None]
        allowed = allowed | eye                                                     # padded queries: no NaN rows
        nhead = self.head.layers[0].self_attn.num_heads
        mask = torch.zeros(allowed.shape, dtype=x.dtype, device=x.device).masked_fill(~allowed, float("-inf"))
        mask = mask.repeat_interleave(nhead, 0)
        for layer in self.head.layers:
            x = layer(x, src_mask=mask)
        logits = self.scorer(x[:, P:]).squeeze(-1).float()
        return logits.masked_fill(~marker_mask, -1e4)

    def forward(self, b: Dict, image_hidden_states=None) -> torch.Tensor:
        """A ``collate_two_tower`` batch -> option logits [B, kmax] in marker order."""
        if image_hidden_states is None and b.get("pixel_values") is not None:
            image_hidden_states = self.image_features(b["pixel_values"].to(self.encoder.dtype),
                                                      b["pixel_attention_mask"])
        h = self.encode_state(b["prefix_ids"], b["prefix_mask"], image_hidden_states)
        opt = self.option_cache(self.encode_options(b["opt_ids"], b["opt_mask"]))
        return self.score(h, b["prefix_mask"], b["qtype"], opt[b["opt_index"]], b["marker_mask"])


def build_from_agent(agent, mode: str, proj_dim: int = 256) -> TwoTowerScorer:
    """A student initialised from a cross-encoder ``VLMAgent``: its backbone (a copy), and for ``"li"`` its head."""
    m = agent.model
    return TwoTowerScorer(copy.deepcopy(m.encoder), mode, head=m.head, scorer=m.scorer, type_emb=m.type_emb,
                          proj_dim=proj_dim)


def option_context_ids(processor) -> tuple:
    tok = processor.tokenizer
    ctx = tok(OPTION_CONTEXT, add_special_tokens=False)["input_ids"]
    end = tok("\n", add_special_tokens=False)["input_ids"]
    assert len(end) == 1
    return list(ctx), end[0]


class _Items(torch.utils.data.Dataset):
    def __init__(self, processor, examples, pairs):
        self.processor, self.examples, self.pairs = processor, examples, pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, j):
        from .vlm_train import make_item

        i, order = self.pairs[j]
        it = make_item(self.processor, self.examples[i], random.Random(0), shuffle=False, order=order)
        it["index"] = i
        return it


@torch.no_grad()
def collect_logits_tt(model: TwoTowerScorer, processor, examples: List[Dict], orders=None, batch_size: int = 32,
                      num_workers: int = 0) -> List[Dict]:
    """``laya.vlm_train.collect_logits`` for a ``TwoTowerScorer`` (same record format, label-order logits)."""
    import numpy as np

    device = next(model.parameters()).device
    model.eval()
    ctx_ids, end_id = option_context_ids(processor)
    pairs = []
    for i, ex in enumerate(examples):
        k = len(ex["target"])
        for order in (orders(k) if orders else [list(range(k))]):
            pairs.append((i, order))
    loader = torch.utils.data.DataLoader(
        _Items(processor, examples, pairs), batch_size=batch_size, num_workers=num_workers,
        collate_fn=functools.partial(collate_two_tower, pad_id=processor.tokenizer.pad_token_id, ctx_ids=ctx_ids,
                                     end_id=end_id))
    per_ex: Dict[int, List[torch.Tensor]] = {}
    for batch in loader:
        b = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(b).float().cpu()
        for r, (i, order) in enumerate(zip(batch["index"], batch["order"])):
            z = torch.empty(len(order))
            z[torch.tensor(order)] = logits[r, : len(order)]
            per_ex.setdefault(i, []).append(z)
    out = []
    for i, ex in enumerate(examples):
        zs = per_ex[i]
        out.append({"logits": zs[0], "logits_per_order": zs, "target": torch.tensor(ex["target"]),
                    "qtype": QTYPES[ex["q"]["t"]], "dataset": ex.get("dataset", "_"),
                    "label": ex.get("label", int(np.argmax(ex["target"])))})
    return out
