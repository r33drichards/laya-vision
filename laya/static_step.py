"""One game decision as a single CUDA graph: what the fixed question buys on a GPU (docs/game-caching.md).

In game play every step asks the same ``choice`` question (the same instructions, the same buttons) about a new
screen. The question cannot be cached as keys/values: in the causal "terminator" layout it comes *after* the
image, so every one of its keys and values depends on the frame. What it does fix is every tensor shape of the
forward pass, and at batch 1 on a GPU the forward is bound by kernel launches, not arithmetic (the 30-layer
language model costs the same ~24 ms on an L4 for 150 or 300 tokens). So ``StaticStep`` captures the whole
decision -- resize, vision tower, connector, merge, language model, heads -- once as a CUDA graph and replays it
for each frame: ~11 ms per decision at batch 1 in bf16 on an L4, 3.4-5.0x fewer (see the doc for the numbers).

The Hugging Face forward has two data-dependent steps a graph cannot hold, and both are constant here, so they
are hoisted out and computed once:

* the vision tower's position ids (bucketized from the patch attention mask, with a boolean-indexed write): a
  full, unpadded tile always gets the same ones, so they are recorded from one eager call;
* the merge of image features into the text embeddings (``masked_scatter``): the ``<image>`` positions are the
  same every step, so it becomes an index write at fixed positions.

Everything else runs the model's own modules in the model's own order; the heads repeat
``VLMDecisionModel.forward``'s tail, and ``tests/test_vlm.py`` checks the result against ``action_probs``. On a
CPU (or with ``capture=False``) the same unrolled step runs eagerly with no speedup, which is what the CPU test
compares; in fp32 on CUDA the replay is bit-identical to the eager model, in bf16 it moves logits by ~1e-2 (the
same size as batch-1 against batch-2 without any graph).

Scope: SmolVLM-family (``readout="terminator"``) checkpoints without image splitting, either preprocessing path,
one option order (as ``action_probs`` and ``predict`` with ``n_permutations=1``). A new frame size or batch size
builds (and captures) a new step; ``model_policy`` keeps one per batch size.
"""
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from .common import QTYPES, confidence_from_probs, render_options, temp_bucket
from .preprocess import prefix_ids
from .vlm import PREFIX_TEXT, VLMAgent, build_vlm_inputs, collate_vlm, option_block_mask, vlm_prefix


class StaticStep:
    """``probs(frames, prev_frames)`` like ``laya.atari_train.action_probs``, for one fixed question and batch size.

    ``question`` is one question definition (``{"type": "choice", "instructions": ..., "criteria": ...}``).
    ``frames`` is 1 or 2 images per decision (2 = ``[previous, current]``). ``capture`` defaults to on for CUDA.
    """

    def __init__(self, agent: VLMAgent, question: Dict, frames: int = 1, batch: int = 1,
                 capture: Optional[bool] = None):
        model, enc = agent.model, agent.model.encoder
        if model.readout != "terminator" or not hasattr(enc, "connector"):
            raise ValueError("StaticStep supports the causal SmolVLM family (readout='terminator') only")
        if agent.prep.split_edge:
            raise ValueError("StaticStep needs one view per image (image_split_edge=0)")
        self.agent, self.model, self.enc, self.prep = agent, model, enc, agent.prep
        self.frames, self.batch, self.dev = frames, batch, agent.device
        self.capture = agent.device.type == "cuda" if capture is None else capture
        q = VLMAgent._to_internal(question)
        self.k = len(render_options(q))
        self.keys = list(q["crit"].keys()) if isinstance(q["crit"], dict) else [str(i) for i in range(self.k)]
        ids = prefix_ids(agent.processor, PREFIX_TEXT, frames, agent.prep.image_seq_len)
        prefix = {"ids": ids, "pixel_values": None, "pixel_attention_mask": None, "n_images": frames}
        max_len, head_max_len = agent.cfg.get("max_len", 1024), agent.cfg.get("head_max_len", 256)
        it = build_vlm_inputs(agent.processor, {}, q, max_len, head_max_len, prefix=prefix, readout="terminator")
        it["qtype"] = QTYPES["choice"]
        b = collate_vlm([it] * batch, agent.processor.tokenizer.pad_token_id, with_pixels=False)
        self.b = {k: v.to(self.dev) for k, v in b.items() if torch.is_tensor(v)}
        self.ids = self.b["input_ids"]
        self.img_pos = (self.ids[0] == enc.config.image_token_id).nonzero()[:, 0]
        self.position_ids = torch.arange(self.ids.shape[1], device=self.dev)[None].expand(batch, -1)
        self.attn = None  # no padding: plain causal, which the language model builds itself
        if model.option_attention == "block":
            self.attn = option_block_mask(self.b["attention_mask"], self.b["option_span"], enc.dtype)
        t = agent.temperature_by_options.get(temp_bucket(QTYPES["choice"], self.k), agent.temperature[QTYPES["choice"]])
        self.t = max(1e-3, float(t))
        self._shape = None
        self.graph = None
        self.calls = 0

    # -- the step ---------------------------------------------------------------------------------------------

    def _setup(self, shape):
        """Static input buffer for frames of ``shape`` (HWC), and the vision position ids for that tile."""
        S, n = self.prep.image_size, self.batch * self.frames
        if self.prep.on_gpu:
            self.inp = torch.zeros((n,) + tuple(shape), dtype=torch.uint8, device=self.dev)
        else:
            self.inp = torch.zeros((n, 3, S, S), dtype=self.enc.dtype, device=self.dev)
        pv = torch.zeros((1, 1, 3, S, S), dtype=self.enc.dtype, device=self.dev) + 1  # not all-zero: a real image
        got = {}
        emb = self.enc.vision_model.embeddings.position_embedding
        hook = emb.register_forward_hook(lambda m, i, o: got.update(pos=i[0][:1].clone()))
        try:
            self.enc.get_image_features(pv, torch.ones((1, 1, S, S), dtype=torch.bool, device=self.dev),
                                        **self.model._vision_kw)
        finally:
            hook.remove()
        self.pos = got["pos"]
        self.text_emb = self.enc.get_input_embeddings()(self.ids)
        self._shape, self.graph = tuple(shape), None

    def _pixels(self):
        if self.prep.on_gpu:
            pv, _ = self.prep.pixel_values(self.inp, dtype=self.enc.dtype)
            return pv
        return self.inp

    def _step(self):
        enc, model, b = self.enc, self.model, self.b
        vm = enc.vision_model
        x = vm.embeddings.patch_embedding(self._pixels()).flatten(2).transpose(1, 2)
        x = vm.encoder(inputs_embeds=x + vm.embeddings.position_embedding(self.pos)).last_hidden_state
        x = vm.post_layernorm(x)
        feats = enc.connector(x).reshape(self.batch, -1, self.text_emb.shape[-1])
        emb = self.text_emb.clone()
        emb[:, self.img_pos] = feats.to(emb.dtype)
        h = enc.text_model(inputs_embeds=emb, attention_mask=self.attn, position_ids=self.position_ids,
                           use_cache=False).last_hidden_state
        # VLMDecisionModel.forward from here on (the "terminator" branch); tests/test_vlm.py pins the two together
        attention_mask, marker_pos, marker_mask = b["attention_mask"], b["marker_pos"], b["marker_mask"]
        h = h.float() + model.type_emb(b["qtype"])[:, None, :]
        if model.head is not None:
            pad = ~attention_mask.bool()
            for layer in model.head.layers:
                h = layer(h, src_key_padding_mask=pad)
        m = torch.gather(h, 1, marker_pos[:, :, None].expand(-1, -1, h.size(-1)))
        logits = model.scorer(m).squeeze(-1).float().masked_fill(~marker_mask, -1e4)
        p = torch.softmax(logits, -1)
        k = marker_mask.sum(-1).clamp(min=2).float()
        ent = -(p * torch.log(p.clamp_min(1e-9))).sum(-1) / torch.log(k)
        top2 = p.topk(2, -1).values
        feats = torch.stack([top2[:, 0], top2[:, 0] - top2[:, 1], ent, k / 255.0], -1)
        last = attention_mask.sum(-1) - 1
        pooled = h[torch.arange(h.size(0), device=h.device), last]
        act = model.act_head(torch.cat([pooled, feats], -1))
        return (torch.softmax(logits[:, : self.k] / self.t, -1), torch.softmax(act.float(), -1)[:, 0])

    def _capture(self):
        s = torch.cuda.Stream(self.dev)
        s.wait_stream(torch.cuda.current_stream(self.dev))
        with torch.cuda.stream(s):
            for _ in range(3):  # warm up: cuBLAS handles, SDPA kernel choice, allocator
                self._step()
        torch.cuda.current_stream(self.dev).wait_stream(s)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.out = self._step()

    # -- public -----------------------------------------------------------------------------------------------

    def _load(self, images: List[np.ndarray]):
        if self.prep.on_gpu:
            src = torch.stack([torch.from_numpy(np.ascontiguousarray(np.asarray(f))) for f in images])
            self.inp.copy_(src, non_blocking=True)
        else:
            pv = torch.cat([vlm_prefix(self.agent.processor, [f], self.prep)["pixel_values"] for f in images])
            self.inp.copy_(pv.to(self.enc.dtype), non_blocking=True)

    @torch.no_grad()
    def probs(self, frames: Sequence[np.ndarray], prev_frames: Optional[Sequence[np.ndarray]] = None,
              return_act: bool = False):
        """Calibrated action probabilities ``[batch, k]`` (and P(act) with ``return_act``), as ``action_probs``."""
        if len(frames) != self.batch:
            raise ValueError("this step was built for batch %d, got %d frames" % (self.batch, len(frames)))
        if (prev_frames is not None) != (self.frames == 2):
            raise ValueError("this step was built for %d frame(s) per decision" % self.frames)
        images = [f for j in range(len(frames)) for f in ((frames[j],) if prev_frames is None
                                                          else (prev_frames[j], frames[j]))]
        shape = np.asarray(images[0]).shape
        if shape != self._shape:
            self._setup(shape)
        self._load(images)
        if self.capture and self.graph is None:
            self._capture()
        if self.graph is not None:
            self.graph.replay()
            p, act = self.out
        else:
            p, act = self._step()
        self.calls += 1
        p = p.cpu().numpy()
        return (p, act.cpu().numpy()) if return_act else p

    def answer(self, frame, prev_frame=None) -> Dict:
        """One decision as ``VLMAgent.predict(...)["answers"][qid]`` shapes a ``choice`` answer (batch 1 only)."""
        p, act = self.probs([frame], None if prev_frame is None else [prev_frame], return_act=True)
        p = p[0].astype(np.float64)
        return {"type": "choice", "choice": self.keys[int(p.argmax())],
                "probabilities": {kk: round(float(v), 4) for kk, v in zip(self.keys, p)},
                "confidence": round(confidence_from_probs(p, self.k), 4),
                "action": {"act_probability": round(float(act[0]), 4)}}


__all__ = ["StaticStep"]
