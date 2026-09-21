"""Experimental image-input decision model on a small vision-language model backbone.

Two backbone families share this module, the API, the training loop and the checkpoint format; the
difference is where each option is read out (``VLMDecisionModel.readout``):

* ``"terminator"``: a causal VLM (SmolVLM, the default), described next.
* ``"mask"``: a bidirectional encoder VLM (ModernVBERT, ``MODERNVBERT_BACKBONE``), described at the end.

The ModernBERT ``DecisionModel`` reads each option at a bidirectional ``[MASK]`` marker. A causal VLM backbone
is a decoder, so the sequence is reordered so that every option's readout token comes AFTER all of
the content it must see:

    <|im_start|>User:<image tokens...><state text>
    <type> question: <ins><end_of_utterance>
    Assistant: Options:
    - opt0\\n
    - opt1\\n
    ...

Readout: the hidden state of the ``\\n`` that terminates each option line (the last token of the option
span) represents that option. It has attended to the image, the state, the question, and the option's own
text. Using a fixed terminator token (instead of the option's last word) gives every readout position the
same token identity, the causal-LM analogue of the ``[MASK]`` marker and of EOS pooling in decoder
embedding models, and needs no new special token (a fresh ``[OPT]`` embedding would be random under
head-only training).

Causal-order caveat: option i cannot attend to option j > i, so the readout for early options is made
without seeing the competitors. This introduces option-order bias. Mitigations:
  * train with random ``option_order`` (``vlm_train.py`` does this),
  * average over ``n_permutations`` orders at inference (``VLMAgent.predict(..., n_permutations=K)``),
  * the optional head transformer (``head_layers > 0``) is bidirectional over the whole sequence,
  * ``option_attention="bidirectional"`` passes a custom 4D mask that lets the option block attend to
    itself in both directions (the pretrained backbone never saw this pattern; it needs fine-tuning).

ModernVBERT (``readout="mask"``) needs none of that. It is ModernBERT-150M plus a SigLIP2 vision tower behind
the same Idefics3 processor SmolVLM uses (same 512-pixel tiles, 64 image tokens, pixel shuffle 4), pretrained
with masked language modelling, so the sequence is Laya's own text format with the image run where the
pretraining chat template puts it:

    [CLS]User:<image tokens...> <type> question: <ins>[SEP][MASK] opt0[MASK] opt1 ...[SEP]<state>[SEP]

Readout: the hidden state of the ``[MASK]`` that opens each option, exactly as in ``laya.common.build_sequence``;
the act head pools ``[CLS]``. Every marker sees the whole sequence, so there is no option-order bias to
mitigate: ``n_permutations`` and ``option_attention`` are accepted and do nothing useful. ``readout`` is chosen
from the backbone's ``model_type`` (``readout_for``), recorded in ``vlm_agent_config.json``, and left on the
processor as ``laya_readout`` so the sequence builders follow it (``ImagePrep`` does the same for the pixels).
"""
import inspect
import json
import os
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from .common import QTYPES, confidence_from_probs, render_options, serialize_state, temp_bucket
from .preprocess import ImagePrep, as_uint8_chw, prefix_ids

DEFAULT_BACKBONE = "HuggingFaceTB/SmolVLM-256M-Instruct"
MODERNVBERT_BACKBONE = "ModernVBERT/modernvbert"
READOUTS = ("terminator", "mask")
#: backbone ``model_type`` values that are bidirectional encoders and take the ``"mask"`` readout
BIDIRECTIONAL_MODEL_TYPES = ("modernvbert",)
CONFIG_NAME = "vlm_agent_config.json"
WEIGHTS_NAME = "model.safetensors"
HEAD_WEIGHTS_NAME = "head.safetensors"

PREFIX_TEXT = "<|im_start|>User:"
QUESTION_TEXT = "\n%s question: %s<end_of_utterance>\nAssistant: Options:\n"
OPTION_BULLET = "- "
OPTION_END = "\n"
# the "mask" readout: ModernVBERT's chat template renders a user turn as ``User:<image>text`` inside the
# tokenizer's ``[CLS] ... [SEP]``; the question and options then follow ``laya.common.build_sequence``
MASK_PREFIX_TEXT = "[CLS]User:"
MASK_QUESTION_TEXT = " %s question: %s"


def readout_for(config) -> str:
    """``"mask"`` for a bidirectional encoder backbone (ModernVBERT), ``"terminator"`` for a causal one."""
    return "mask" if getattr(config, "model_type", None) in BIDIRECTIONAL_MODEL_TYPES else "terminator"


def processor_readout(processor) -> str:
    """The readout a processor's sequences are built for: what the agent left on it, else what its tokenizer
    can do (a ``[MASK]`` token means a masked-LM encoder; SmolVLM's tokenizer has none)."""
    got = getattr(processor, "laya_readout", None)
    if got is not None:
        return got
    return "mask" if processor.tokenizer.mask_token_id is not None else "terminator"


# ---------------------------------------------------------------------------------------------------------
# State handling
# ---------------------------------------------------------------------------------------------------------


def _load_image(img):
    from PIL import Image

    if isinstance(img, Image.Image):
        return img.convert("RGB")
    if isinstance(img, np.ndarray) and img.dtype == np.uint8:
        return img  # a raw HWC frame, e.g. an ALE observation; both preprocessing paths take it as-is
    if isinstance(img, (str, os.PathLike)):
        with Image.open(img) as im:
            return im.convert("RGB")
    raise TypeError("unsupported image type %r (expected PIL.Image, uint8 array or path)" % type(img).__name__)


def split_state(state: Any) -> Tuple[List, str]:
    """Split a state into (images, text).

    Accepts text, a list (conversation turns), a PIL image, or a dict. Dict keys ``"image"`` (PIL image or
    path) and ``"images"`` (list of them) are pulled out as images; the remaining keys are serialized to
    JSON text exactly like the text-only model does.
    """
    try:
        from PIL import Image

        if isinstance(state, Image.Image):
            return [state.convert("RGB")], ""
    except ImportError:
        pass
    if not isinstance(state, dict):
        return [], serialize_state(state)
    images = []
    if state.get("image") is not None:
        images.append(_load_image(state["image"]))
    for img in state.get("images") or []:
        images.append(_load_image(img))
    rest = {k: v for k, v in state.items() if k not in ("image", "images")}
    return images, serialize_state(rest) if rest else ""


# ---------------------------------------------------------------------------------------------------------
# Sequence construction
# ---------------------------------------------------------------------------------------------------------


def vlm_prefix(processor, images: Sequence, prep: Optional[ImagePrep] = None,
               prefix_text: Optional[str] = None) -> Dict[str, Any]:
    """Token ids for ``<|im_start|>User:<image>...`` (``[CLS]User:<image>...`` for the mask readout) plus one
    tile of pixels per image.

    With ``prep.backend == "gpu"`` no pixel is touched here: the ids are built from the image count alone
    (``preprocess.prefix_ids``) and the raw uint8 frames are returned as ``raw_images`` for the device-side
    resize in ``VLMDecisionModel.forward``. This is what keeps the data loader and the play loop cheap.

    ``prep`` defaults to the one ``ImagePrep.apply`` left on the processor, so callers that only ever see a
    processor (the training loader, ``collect_logits``) follow the checkpoint's path without being told.
    """
    if prep is None:
        prep = getattr(processor, "laya_prep", None)
    if prefix_text is None:
        prefix_text = MASK_PREFIX_TEXT if processor_readout(processor) == "mask" else PREFIX_TEXT
    if not images:
        ids = processor.tokenizer(prefix_text, add_special_tokens=False)["input_ids"]
        return {"ids": list(ids), "pixel_values": None, "pixel_attention_mask": None, "raw_images": None, "n_images": 0}
    if prep is not None and prep.on_gpu:
        return {
            "ids": prefix_ids(processor, prefix_text, len(images), prep.image_seq_len),
            "pixel_values": None,
            "pixel_attention_mask": None,
            "raw_images": as_uint8_chw(list(images)),
            "n_images": len(images),
        }
    out = processor(
        text=[prefix_text + processor.image_token * len(images)],
        images=[list(images)],
        do_image_splitting=False,
        return_tensors="pt",
        add_special_tokens=False,  # the framing is spelled out in prefix_text (a no-op for SmolVLM's tokenizer)
    )
    return {
        "ids": out["input_ids"][0].tolist(),
        "pixel_values": out["pixel_values"][0],
        "pixel_attention_mask": out["pixel_attention_mask"][0].bool(),
        "raw_images": None,
        "n_images": len(images),
    }


def build_vlm_inputs(
    processor,
    state: Any,
    q: Dict,
    max_len: int = 1024,
    head_max_len: int = 256,
    option_order: Optional[List[int]] = None,
    truncate_left: bool = False,
    prefix: Optional[Dict[str, Any]] = None,
    prep: Optional[ImagePrep] = None,
    readout: Optional[str] = None,
) -> Dict[str, Any]:
    """Build one VLM sequence for an internal question ``q = {"t", "ins", "crit"}``.

    Returns ``{"ids", "markers", "option_span", "pixel_values", "pixel_attention_mask", "raw_images", "n_images"}``.
    ``markers[j]`` indexes the readout token of the j-th option in ``option_order`` order: the ``\\n``
    terminating its line (``readout="terminator"``, causal backbones) or the ``[MASK]`` opening it
    (``readout="mask"``, ModernVBERT). ``readout`` defaults to what the processor was bound to
    (``processor_readout``). ``prefix`` (from ``vlm_prefix``) may be passed to reuse image preprocessing across
    questions; the state's images are then ignored in favour of it. ``prep`` picks the preprocessing path (see
    ``laya.preprocess``); it is ignored when ``prefix`` is given, which already carries the choice.
    """
    if readout is None:
        readout = processor_readout(processor)
    if readout not in READOUTS:
        raise ValueError("readout must be one of %s, got %r" % (READOUTS, readout))
    tok = processor.tokenizer
    images, text = split_state(state)
    if prefix is None:
        prefix = vlm_prefix(processor, images, prep, MASK_PREFIX_TEXT if readout == "mask" else PREFIX_TEXT)
    if readout == "mask":
        return _mask_inputs(processor, text, q, max_len, head_max_len, option_order, truncate_left, prefix)
    enc = lambda s: tok(s, add_special_tokens=False)["input_ids"]  # noqa: E731
    end_id = enc(OPTION_END)
    assert len(end_id) == 1, "option terminator must be a single token"
    end_id = end_id[0]

    opts = render_options(q)
    order = option_order if option_order is not None else list(range(len(opts)))
    opt_ids = [enc(OPTION_BULLET + opts[i].replace(OPTION_END, " "))[:48] for i in order]
    head_ids = enc(QUESTION_TEXT % (q["t"], str(q["ins"]).replace("<end_of_utterance>", " ")))
    opt_budget = head_max_len - sum(len(o) + 1 for o in opt_ids)
    if opt_budget < 16:
        per = max(4, (head_max_len - 16) // max(1, len(opt_ids)) - 1)
        opt_ids = [o[:per] for o in opt_ids]
        opt_budget = head_max_len - sum(len(o) + 1 for o in opt_ids)
    if len(head_ids) > max(8, opt_budget):
        # keep the tail (it carries the "Assistant: Options:" cue); drop the middle of the instructions
        keep = max(8, opt_budget)
        head_ids = head_ids[: keep // 2] + head_ids[-(keep - keep // 2):]

    tail = list(head_ids)
    markers = []
    span_start = len(tail)
    for o in opt_ids:
        tail.extend(o)
        tail.append(end_id)
        markers.append(len(tail) - 1)

    room = max(0, max_len - len(prefix["ids"]) - len(tail))
    st = enc(text) if text else []
    st = st[-room:] if truncate_left else st[:room]
    off = len(prefix["ids"]) + len(st)
    if len(prefix["ids"]) + len(tail) > max_len:
        raise ValueError("question + options + images exceed max_len=%d" % max_len)
    return {
        "ids": prefix["ids"] + st + tail,
        "markers": [m + off for m in markers],
        "option_span": (span_start + off, len(tail) + off),
        "pixel_values": prefix["pixel_values"],
        "pixel_attention_mask": prefix["pixel_attention_mask"],
        "raw_images": prefix.get("raw_images"),
        "n_images": prefix["n_images"],
    }


def _mask_inputs(processor, text: str, q: Dict, max_len: int, head_max_len: int, option_order: Optional[List[int]],
                 truncate_left: bool, prefix: Dict[str, Any]) -> Dict[str, Any]:
    """The bidirectional sequence (see the module docstring), ``laya.common.build_sequence`` with an image run:

        [CLS]User:<image tokens...> <type> question: <ins>[SEP][MASK] opt0[MASK] opt1 ...[SEP]<state>[SEP]

    Budgets are the text model's: the question plus options fit in ``head_max_len`` (options are cut to 48
    tokens each, then evenly, then the instructions), the state takes what is left of ``max_len``.
    """
    tok = processor.tokenizer
    mask_id, sep_id = tok.mask_token_id, tok.sep_token_id
    if mask_id is None or sep_id is None:
        raise ValueError("readout='mask' needs a tokenizer with [MASK] and [SEP] tokens (a ModernVBERT backbone)")
    clean = lambda s: s.replace(tok.mask_token, " ").replace(processor.image_token, " ")  # noqa: E731
    enc = lambda s: tok(s, add_special_tokens=False)["input_ids"]  # noqa: E731

    opts = render_options(q)
    order = option_order if option_order is not None else list(range(len(opts)))
    opt_ids = [[mask_id] + enc(" " + clean(opts[i]))[:48] for i in order]
    head_ids = enc(MASK_QUESTION_TEXT % (q["t"], clean(str(q["ins"]))))
    opt_budget = head_max_len - sum(len(o) for o in opt_ids)
    if opt_budget < 16:
        per = max(4, (head_max_len - 16) // max(1, len(opt_ids)))
        opt_ids = [o[:per] for o in opt_ids]
        opt_budget = head_max_len - sum(len(o) for o in opt_ids)
    head_ids = head_ids[: max(8, opt_budget)]

    ids = list(prefix["ids"]) + head_ids + [sep_id]
    markers = []
    span_start = len(ids)
    for o in opt_ids:
        markers.append(len(ids))
        ids.extend(o)
    span_end = len(ids)
    ids.append(sep_id)
    if len(ids) > max_len:
        raise ValueError("question + options + images exceed max_len=%d" % max_len)
    st = enc(clean(text)) if text else []
    if st:
        room = max(0, max_len - len(ids) - 1)
        st = st[-room:] if truncate_left else st[:room]
        ids = ids + st + [sep_id]
    return {
        "ids": ids,
        "markers": markers,
        "option_span": (span_start, span_end),
        "pixel_values": prefix["pixel_values"],
        "pixel_attention_mask": prefix["pixel_attention_mask"],
        "raw_images": prefix.get("raw_images"),
        "n_images": prefix["n_images"],
    }


def collate_vlm(items: List[Dict], pad_id: int, with_pixels: bool = True) -> Dict[str, Any]:
    """Right-pad token sequences and stack per-item images (zero images pad ragged image counts).

    Items from the GPU preprocessing path carry ``raw_images`` (uint8, unresized) instead of ``pixel_values``;
    those are stacked into ``raw_pixels`` ``[n, n_img, 3, H, W]`` with an ``image_mask`` ``[n, n_img]`` marking
    the real ones, and ``VLMDecisionModel.forward`` turns them into pixels on the GPU.
    """
    n, L = len(items), max(len(it["ids"]) for it in items)
    kmax = max(len(it["markers"]) for it in items)
    ids = torch.full((n, L), pad_id, dtype=torch.long)
    att = torch.zeros((n, L), dtype=torch.long)
    mpos = torch.zeros((n, kmax), dtype=torch.long)
    mmask = torch.zeros((n, kmax), dtype=torch.bool)
    span = torch.zeros((n, 2), dtype=torch.long)
    has_target = any("target" in it for it in items)
    target = torch.zeros((n, kmax), dtype=torch.float32) if has_target else None
    for i, it in enumerate(items):
        ids[i, : len(it["ids"])] = torch.tensor(it["ids"])
        att[i, : len(it["ids"])] = 1
        k = len(it["markers"])
        mpos[i, :k] = torch.tensor(it["markers"])
        mmask[i, :k] = True
        span[i] = torch.tensor(it["option_span"])
        if has_target and "target" in it:
            target[i, : len(it["target"])] = torch.tensor(it["target"], dtype=torch.float32)
    res = {
        "input_ids": ids,
        "attention_mask": att,
        "marker_pos": mpos,
        "marker_mask": mmask,
        "option_span": span,
        "qtype": torch.tensor([it["qtype"] for it in items]),
        "label": torch.tensor([it.get("label", -1) for it in items]),
        "pixel_values": None,
        "pixel_attention_mask": None,
        "raw_pixels": None,
        "image_mask": None,
    }
    if target is not None:
        res["target"] = target
    n_img = max(it.get("n_images", 0) for it in items)
    if with_pixels and n_img > 0 and any(it.get("raw_images") is not None for it in items):
        ref = next(it["raw_images"] for it in items if it.get("raw_images") is not None)
        shapes = {tuple(it["raw_images"].shape[1:]) for it in items if it.get("raw_images") is not None}
        if len(shapes) != 1:
            raise ValueError("the device-side preprocessing path (preprocess='gpu') stacks raw frames, so they must "
                             "all be the same size, got %s; photos of mixed sizes need preprocess='processor'"
                             % sorted(shapes))
        raw = torch.zeros((n, n_img) + tuple(ref.shape[1:]), dtype=torch.uint8)
        imask = torch.zeros((n, n_img), dtype=torch.bool)
        for i, it in enumerate(items):
            m = it.get("n_images", 0)
            if m:
                raw[i, :m] = it["raw_images"]
                imask[i, :m] = True
        res["raw_pixels"], res["image_mask"] = raw, imask
    elif with_pixels and n_img > 0:
        ref = next(it["pixel_values"] for it in items if it.get("n_images", 0) > 0)
        pv = torch.zeros((n, n_img) + tuple(ref.shape[1:]), dtype=ref.dtype)
        pam = torch.zeros((n, n_img) + tuple(ref.shape[2:]), dtype=torch.bool)
        for i, it in enumerate(items):
            m = it.get("n_images", 0)
            if m:
                pv[i, :m] = it["pixel_values"]
                pam[i, :m] = it["pixel_attention_mask"]
        res["pixel_values"], res["pixel_attention_mask"] = pv, pam
    return res


def option_block_mask(attention_mask: torch.Tensor, option_span: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Additive 4D mask: causal everywhere, bidirectional inside each row's option span, padding keys masked."""
    B, L = attention_mask.shape
    pos = torch.arange(L, device=attention_mask.device)
    allowed = pos[None, :, None] >= pos[None, None, :]
    s, e = option_span[:, 0, None], option_span[:, 1, None]
    in_span = (pos[None] >= s) & (pos[None] < e)
    allowed = allowed | (in_span[:, :, None] & in_span[:, None, :])
    allowed = allowed & attention_mask.bool()[:, None, :]
    mask = torch.zeros((B, 1, L, L), dtype=dtype, device=attention_mask.device)
    return mask.masked_fill(~allowed[:, None], torch.finfo(dtype).min)


# ---------------------------------------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------------------------------------


class VLMDecisionModel(nn.Module):
    """VLM backbone (vision tower + connector + LM, no LM head) + typed decision head.

    The head mirrors ``DecisionModel`` so training code can be shared. The backbone is stored as
    ``encoder`` for that reason (e.g. optimizer groups split on ``"encoder."``). With the ``"terminator"``
    readout (causal backbones) the act head reads the last real token (the final option terminator) instead
    of ``[CLS]``: in a causal model it is the only position that has seen the whole sequence. With the
    ``"mask"`` readout (ModernVBERT) it reads ``[CLS]``, as ``DecisionModel`` does.
    """

    def __init__(
        self,
        backbone: nn.Module,
        head_layers: int = 2,
        n_act: int = 2,
        dropout: float = 0.1,
        option_attention: str = "causal",
        prep: Optional[ImagePrep] = None,
        readout: str = "terminator",
    ):
        super().__init__()
        if option_attention not in ("causal", "bidirectional"):
            raise ValueError("option_attention must be 'causal' or 'bidirectional'")
        if readout not in READOUTS:
            raise ValueError("readout must be one of %s, got %r" % (READOUTS, readout))
        if readout == "mask" and option_attention != "causal":
            raise ValueError("option_attention is for the causal 'terminator' readout; a 'mask' backbone is "
                             "bidirectional everywhere already")
        self.encoder = backbone
        self.option_attention = option_attention
        self.readout = readout
        # SmolVLM's Idefics3 vision tower resizes its position grid to the tile it is given; ModernVBERT's plain
        # SigLIP tower has a fixed 512-pixel grid and must be asked to interpolate (a no-op at 512), so smaller
        # ``image_size`` settings work on both. Decided by what the tower's forward accepts, not by family.
        takes = inspect.signature(backbone.vision_model.forward).parameters
        self._vision_kw = {"interpolate_pos_encoding": True} if "interpolate_pos_encoding" in takes else {}
        self.prep = prep or ImagePrep(backend="processor")
        d = backbone.config.text_config.hidden_size
        nhead = max(1, d // 64)
        layer = nn.TransformerEncoderLayer(d, nhead, 4 * d, dropout, batch_first=True, norm_first=True)
        self.head = nn.TransformerEncoder(layer, head_layers, enable_nested_tensor=False) if head_layers > 0 else None
        self.type_emb = nn.Embedding(3, d)
        self.scorer = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.act_head = nn.Sequential(nn.Linear(d + 4, 256), nn.GELU(), nn.Linear(256, n_act))
        self.register_buffer("temperature", torch.ones(3))

    @property
    def backbone(self) -> nn.Module:
        return self.encoder

    def _image_features(self, pixel_values: torch.Tensor, pixel_attention_mask: Optional[torch.Tensor]):
        """``[B, n_img, 3, H, W]`` -> ``[n_real, image_seq_len, d]``: the backbone's own vision tower + connector,
        with all-zero (padded) image slots dropped, as its forward does with ``pixel_values``."""
        return self.encoder.get_image_features(pixel_values, pixel_attention_mask, return_dict=True,
                                               **self._vision_kw).pooler_output

    def encode_images(self, pixel_values: torch.Tensor, pixel_attention_mask: Optional[torch.Tensor] = None):
        """[n_img, 3, H, W] -> image token features [n_img, image_seq_len, d] (reusable across questions)."""
        pam = pixel_attention_mask[None] if pixel_attention_mask is not None else None
        return self._image_features(pixel_values[None], pam)

    def encode_raw_images(self, raw_images):
        """Raw uint8 frames -> image token features, resizing and normalising on this model's device."""
        pv, pam = self.prep.pixel_values(raw_images, device=self.encoder.device, dtype=self.encoder.dtype)
        return self.encode_images(pv, pam)

    def forward(
        self,
        input_ids,
        attention_mask,
        marker_pos,
        marker_mask,
        qtype,
        pixel_values=None,
        pixel_attention_mask=None,
        image_hidden_states=None,
        option_span=None,
        detach_encoder: bool = False,
        raw_pixels=None,
        image_mask=None,
    ):
        if pixel_values is None and image_hidden_states is None and raw_pixels is not None:
            # GPU preprocessing path: the loader handed us untouched uint8 frames, resize them here
            pixel_values, pixel_attention_mask = self.prep.pixel_values(
                raw_pixels, device=input_ids.device, dtype=self.encoder.dtype)
            if image_mask is not None:
                # padded image slots must stay exactly zero; get_image_features drops them by that test
                pixel_values = pixel_values * image_mask[..., None, None, None]
                pixel_attention_mask = pixel_attention_mask & image_mask[..., None, None]
        if pixel_values is not None and self._vision_kw:
            # the backbone's forward would call the vision tower without the interpolation flag
            image_hidden_states = self._image_features(pixel_values, pixel_attention_mask)
            pixel_values = pixel_attention_mask = None
        attn, enc_kw = attention_mask, {}
        if self.readout == "terminator":
            enc_kw["use_cache"] = False
            if self.option_attention == "bidirectional":
                if option_span is None:
                    raise ValueError("option_attention='bidirectional' requires option_span")
                attn = option_block_mask(attention_mask, option_span, self.encoder.dtype)
        h = self.encoder(
            input_ids=input_ids,
            attention_mask=attn,
            pixel_values=pixel_values,
            pixel_attention_mask=pixel_attention_mask,
            image_hidden_states=image_hidden_states,
            **enc_kw,
        ).last_hidden_state
        if detach_encoder:
            h = h.detach()
        h = h.float()
        h = h + self.type_emb(qtype)[:, None, :]
        if self.head is not None:
            pad = ~attention_mask.bool()
            for layer in self.head.layers:
                h = layer(h, src_key_padding_mask=pad)
        idx = marker_pos.clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
        m = torch.gather(h, 1, idx)
        logits = self.scorer(m).squeeze(-1).float()
        logits = logits.masked_fill(~marker_mask, -1e4)

        p = torch.softmax(logits.detach(), -1)
        k = marker_mask.sum(-1).clamp(min=2).float()
        ent = -(p * torch.log(p.clamp_min(1e-9))).sum(-1) / torch.log(k)
        top2 = p.topk(2, -1).values
        feats = torch.stack([top2[:, 0], top2[:, 0] - top2[:, 1], ent, k / 255.0], -1)
        if self.readout == "mask":
            pooled = h[:, 0].float()  # [CLS]
        else:
            last = (attention_mask.sum(-1) - 1).clamp(min=0)
            pooled = h[torch.arange(h.size(0), device=h.device), last].float()
        act_logits = self.act_head(torch.cat([pooled, feats], -1))
        return logits, act_logits


def _text_tower(enc: nn.Module):
    """``(layers, final norm)`` of the backbone's language model: ``text_model.layers`` in both families, with
    the norm called ``norm`` (SmolVLM's Llama) or ``final_norm`` (ModernVBERT's ModernBERT)."""
    tm = enc.text_model
    return tm.layers, tm.final_norm if hasattr(tm, "final_norm") else tm.norm


def set_trainable(model: VLMDecisionModel, mode: str = "head", n_last: int = 4, train_vision: bool = False) -> int:
    """Freezing stages (no LoRA). Returns the number of trainable parameters.

    * ``"head"``: backbone frozen; type_emb / head / scorer / act_head train.
    * ``"last_n"``: head + the last ``n_last`` LM decoder layers + final norm.
    * ``"full"``: everything except the vision tower (unless ``train_vision``).
    """
    if mode not in ("head", "last_n", "full"):
        raise ValueError("mode must be 'head', 'last_n' or 'full'")
    for p in model.parameters():
        p.requires_grad = True
    enc = model.encoder
    if mode in ("head", "last_n"):
        enc.requires_grad_(False)
    if mode == "last_n":
        layers, norm = _text_tower(enc)
        for layer in layers[max(0, len(layers) - n_last):]:
            layer.requires_grad_(True)
        norm.requires_grad_(True)
    if mode == "full" and not train_vision:
        enc.vision_model.requires_grad_(False)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_vlm_model(cfg: Dict, backbone_dir: Optional[str] = None, dtype: torch.dtype = torch.float32,
                    prep: Optional[ImagePrep] = None) -> VLMDecisionModel:
    """Build from pretrained backbone weights, or (``backbone_dir``) an architecture-only config to load into."""
    from transformers import AutoConfig, AutoModel

    if backbone_dir and os.path.exists(backbone_dir):
        bcfg = AutoConfig.from_pretrained(backbone_dir)
        backbone = AutoModel.from_config(bcfg, attn_implementation="sdpa", dtype=dtype)
    else:
        backbone = AutoModel.from_pretrained(cfg["backbone"], attn_implementation="sdpa", dtype=dtype)
    return VLMDecisionModel(
        backbone,
        cfg.get("head_layers", 2),
        cfg.get("n_act", 2),
        option_attention=cfg.get("option_attention", "causal"),
        prep=prep,
        readout=cfg.get("readout") or readout_for(backbone.config),
    )


# ---------------------------------------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------------------------------------


def _resolve_device(device: Optional[str]) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _permutations(k: int, n: int) -> List[List[int]]:
    """Deterministic option orders: identity, reversed, then seeded random shuffles."""
    perms = [list(range(k))]
    if n > 1 and k > 1:
        perms.append(list(reversed(range(k))))
    rng = random.Random(0)
    tries = 0
    while len(perms) < n and tries < 100:
        p = list(range(k))
        rng.shuffle(p)
        tries += 1
        if p not in perms:
            perms.append(p)
    return perms[:n]


class VLMAgent:
    """Image+text decision runtime with the same ``predict(state, questions)`` API as ``laya.Agent``.

    Build fresh (untrained head) with ``VLMAgent(backbone="HuggingFaceTB/SmolVLM-256M-Instruct")`` (or
    ``backbone="ModernVBERT/modernvbert"`` for the bidirectional family) or load a saved agent with
    ``VLMAgent("path/or/hub-id")``.
    """

    def __init__(
        self,
        model_id_or_path: Optional[str] = None,
        backbone: Optional[str] = None,
        device: Optional[str] = None,
        token: Optional[str] = None,
        dtype: Optional[str] = None,
        **cfg_overrides,
    ):
        from transformers import AutoProcessor

        self.device = _resolve_device(device)
        if model_id_or_path is None:
            self.cfg = {
                "backbone": backbone or DEFAULT_BACKBONE,
                "head_layers": 2,
                "n_act": 2,
                "max_len": 1024,
                "head_max_len": 256,
                "option_attention": "causal",
                "dtype": dtype or "fp32",
                "temperature": [1.0, 1.0, 1.0],
                "temperature_by_options": {},
            }
            self.cfg.update(cfg_overrides)
            # a fresh agent gets the cheap path by default; a saved one keeps whatever it was trained with
            self.prep = ImagePrep.from_config(self.cfg, default_backend="gpu")
            self.cfg.update(self.prep.to_config())
            self.processor = AutoProcessor.from_pretrained(self.cfg["backbone"], token=token)
            self.prep.apply(self.processor)
            self.model = build_vlm_model(self.cfg, dtype=self._torch_dtype(), prep=self.prep)
        else:
            self._load(model_id_or_path, token, dtype, cfg_overrides)
        # the sequence builders only ever see the processor, so it carries the readout like it carries the prep
        self.cfg["readout"] = self.processor.laya_readout = self.model.readout
        self.prep.check(self.processor)
        self.temperature = self.cfg.get("temperature", [1.0, 1.0, 1.0])
        self.temperature_by_options = self.cfg.get("temperature_by_options", {})
        self.model.to(self.device).eval()

    def _torch_dtype(self) -> torch.dtype:
        return torch.bfloat16 if self.cfg.get("dtype") == "bf16" else torch.float32

    def _load(self, model_id_or_path: str, token, dtype, overrides):
        from safetensors.torch import load_file
        from transformers import AutoProcessor

        model_dir = model_id_or_path
        if not os.path.exists(model_dir):
            from huggingface_hub import snapshot_download

            model_dir = snapshot_download(model_id_or_path, token=token or os.environ.get("HF_TOKEN"))
        cfg_path = os.path.join(model_dir, CONFIG_NAME)
        if not os.path.exists(cfg_path):
            raise FileNotFoundError("%r does not contain %s" % (model_id_or_path, CONFIG_NAME))
        with open(cfg_path) as f:
            self.cfg = json.load(f)
        if dtype:
            self.cfg["dtype"] = dtype
        self.cfg.update(overrides)
        proc_dir = os.path.join(model_dir, "processor")
        self.processor = AutoProcessor.from_pretrained(proc_dir if os.path.exists(proc_dir) else self.cfg["backbone"])
        # honour the checkpoint's recorded input resolution and preprocessing path; a config written before
        # those keys existed means 512 through the Hugging Face processor, which is what it was trained with
        self.prep = ImagePrep.from_config(self.cfg, default_backend="processor")
        self.cfg.update(self.prep.to_config())
        self.prep.apply(self.processor)

        full = os.path.join(model_dir, WEIGHTS_NAME)
        if os.path.exists(full):
            self.model = build_vlm_model(self.cfg, os.path.join(model_dir, "backbone"), dtype=self._torch_dtype(),
                                         prep=self.prep)
            self.model.load_state_dict(load_file(full), strict=True)
        else:
            # head-only checkpoint: backbone comes from the pretrained id in the config
            self.model = build_vlm_model(self.cfg, dtype=self._torch_dtype(), prep=self.prep)
            missing, unexpected = self.model.load_state_dict(load_file(os.path.join(model_dir, HEAD_WEIGHTS_NAME)), strict=False)
            bad = [k for k in missing if not k.startswith("encoder.")] + list(unexpected)
            if bad:
                raise ValueError("head checkpoint mismatch: %s" % bad[:5])

    def save(self, path: str, include_backbone: bool = True):
        """Write ``vlm_agent_config.json``, processor, and weights (full, or head-only for frozen backbones)."""
        from safetensors.torch import save_file

        os.makedirs(path, exist_ok=True)
        cfg = dict(
            self.cfg,
            option_attention=self.model.option_attention,
            readout=self.model.readout,
            temperature=list(self.temperature),
            temperature_by_options=dict(self.temperature_by_options),
            **self.prep.to_config(),
        )
        with open(os.path.join(path, CONFIG_NAME), "w") as f:
            json.dump(cfg, f, indent=2)
        self.processor.save_pretrained(os.path.join(path, "processor"))
        sd = {k: v.detach().cpu().contiguous() for k, v in self.model.state_dict().items()}
        if include_backbone:
            self.model.encoder.config.save_pretrained(os.path.join(path, "backbone"))
            save_file(sd, os.path.join(path, WEIGHTS_NAME))
        else:
            save_file({k: v for k, v in sd.items() if not k.startswith("encoder.")}, os.path.join(path, HEAD_WEIGHTS_NAME))

    @staticmethod
    def _to_internal(qdef: Dict) -> Dict:
        t = qdef["type"]
        crit = qdef.get("criteria")
        if t == "choice" and isinstance(crit, list):
            crit = {c: None for c in crit}
        ins = qdef["instructions"]
        if not isinstance(ins, str):
            ins = json.dumps(ins)
        return {"t": t, "ins": ins, "crit": crit}

    @torch.no_grad()
    def predict(
        self,
        state: Any,
        questions: Dict[str, Dict[str, Any]],
        n_permutations: int = 1,
        batch_size: int = 8,
    ) -> Dict[str, Any]:
        """Evaluate typed questions over a text / JSON / image state. Same output schema as ``Agent.predict``.

        Images are encoded once and their features reused for every question row. ``n_permutations > 1``
        scores each question under several option orders and averages the logits (in label order) to
        reduce the causal option-order bias (pointless with the ``"mask"`` readout, which has none).
        """
        images, _ = split_state(state)
        prefix = vlm_prefix(self.processor, images, self.prep)
        img_feats = None
        if images and prefix["raw_images"] is not None:
            img_feats = self.model.encode_raw_images(prefix["raw_images"])
        elif images:
            img_feats = self.model.encode_images(
                prefix["pixel_values"].to(self.device, self._torch_dtype()), prefix["pixel_attention_mask"].to(self.device)
            )

        ids = list(questions.keys())
        internal = {qid: self._to_internal(questions[qid]) for qid in ids}
        rows = []
        for qid in ids:
            q = internal[qid]
            k = len(render_options(q))
            for order in _permutations(k, max(1, n_permutations)):
                it = build_vlm_inputs(
                    self.processor, state, q, self.cfg.get("max_len", 1024), self.cfg.get("head_max_len", 256),
                    option_order=order, prefix=prefix,
                )
                if len(it["markers"]) != k:
                    raise ValueError("question %r options exceed head_max_len" % qid)
                it.update(qtype=QTYPES[q["t"]], qid=qid, order=order)
                rows.append(it)

        row_logits = []
        n_tokens = 0
        for s in range(0, len(rows), batch_size):
            chunk = rows[s : s + batch_size]
            b = collate_vlm(chunk, self.processor.tokenizer.pad_token_id, with_pixels=False)
            n_tokens += int(b["attention_mask"].sum())
            feats = img_feats.repeat(len(chunk), 1, 1) if img_feats is not None else None
            logits, act = self.model(
                b["input_ids"].to(self.device),
                b["attention_mask"].to(self.device),
                b["marker_pos"].to(self.device),
                b["marker_mask"].to(self.device),
                b["qtype"].to(self.device),
                image_hidden_states=feats,
                option_span=b["option_span"].to(self.device),
            )
            act = torch.softmax(act.float(), -1).cpu().numpy()
            logits = logits.float().cpu().numpy()
            for r in range(len(chunk)):
                row_logits.append((logits[r], act[r]))

        answers = {}
        for qid in ids:
            q = internal[qid]
            k = len(render_options(q))
            z_sum, act_sum, n = np.zeros(k), 0.0, 0
            for it, (lg, ac) in zip(rows, row_logits):
                if it["qid"] != qid:
                    continue
                z_sum[it["order"]] += lg[:k]  # marker j scored option order[j]
                act_sum += float(ac[0])
                n += 1
            qt = QTYPES[q["t"]]
            t_scale = self.temperature_by_options.get(temp_bucket(qt, k), self.temperature[qt])
            z = (z_sum / n) / max(1e-3, float(t_scale))
            p = np.exp(z - z.max())
            p = p / p.sum()
            conf_score = round(confidence_from_probs(p, k), 4)
            ext = {"act_probability": round(act_sum / n, 4)}
            if q["t"] == "choice":
                keys = list(q["crit"].keys())
                answers[qid] = {
                    "type": "choice",
                    "choice": keys[int(p.argmax())],
                    "probabilities": {kk: round(float(v), 4) for kk, v in zip(keys, p)},
                    "confidence": conf_score,
                    "action": ext,
                }
            elif q["t"] == "score":
                answers[qid] = {
                    "type": "score",
                    "score": round(float((np.arange(k) * p).sum()), 4),
                    "legend": {str(i): c for i, c in enumerate(q["crit"])},
                    "probabilities": {str(i): round(float(v), 4) for i, v in enumerate(p)},
                    "confidence": conf_score,
                    "action": ext,
                }
            else:
                answers[qid] = {
                    "type": "noul",
                    "noul": round(float(p[1]), 4),
                    "confidence": round(max(float(p[1]), 1.0 - float(p[1])), 4),
                    "action": ext,
                }

        return {
            "model": "laya-vlm",
            "answers": answers,
            "usage": {"input_tokens": n_tokens, "output_tokens": 0, "images": len(images)},
        }

    system_one = predict


def load_vlm(
    model_id_or_path: Optional[str] = None,
    backbone: Optional[str] = None,
    device: Optional[str] = None,
    token: Optional[str] = None,
    **kwargs,
) -> VLMAgent:
    """Load a saved VLM agent, or build a fresh one on ``backbone`` (its head is untrained)."""
    return VLMAgent(model_id_or_path, backbone=backbone, device=device, token=token, **kwargs)
