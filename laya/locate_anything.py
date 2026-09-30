"""Experimental backbone: NVIDIA LocateAnything-3B (MoonViT-SO-400M + Qwen2.5-3B) under the typed decision head.

LocateAnything (https://research.nvidia.com/labs/lpr/locate-anything/) is a grounding VLM: MoonViT at native
resolution, a 2x2 patch merge into a two-layer MLP, and a Qwen2.5-3B language model fine-tuned to emit boxes with
Parallel Box Decoding. Its Hub code (``trust_remote_code``) is written for generation: the language model builds its
own block-diffusion attention mask, defaults to the ``magi`` kernel, and only runs under the transformers it was
written against. This module keeps what matters for a readout and nothing else:

* the language model is transformers' own ``Qwen2Model`` with plain causal SDPA attention, loaded from the
  checkpoint's ``language_model.model.*`` weights (the block-diffusion part is only a mask; the weights are an
  ordinary Qwen2). So the ``"terminator"`` readout, ``option_attention="block"`` and prefix caching in
  ``laya.vlm`` work exactly as they do on SmolVLM;
* the vision tower is the checkpoint's own ``MoonVitPretrainedModel``, imported from the Hub at a pinned commit
  (``modeling_vit.py`` is NVIDIA-licensed, so it is fetched, never vendored here), with SDPA attention;
* the connector is the checkpoint's ``mlp1``.

Images are fixed squares rather than native resolution, so they batch like SmolVLM's tiles: ``image_size`` 448
(a multiple of patch 14 x merge 2) is a 32 x 32 patch grid, 256 language-model tokens after the merge. Pixels are
normalised with mean = std = 0.5, the checkpoint's own values and the same arithmetic ``ImagePrep`` does, so both
preprocessing backends serve it. The sequence framing is ``laya.vlm``'s (the image run is LocateAnything's
``<img><IMG_CONTEXT>...</img>``), not Qwen's chat template: the head is trained on top either way.

Licence: the weights are under the NVIDIA License, non-commercial research use only (Qwen Research License for
the language model). Anything fine-tuned from them inherits that; do not publish such a checkpoint as the
Apache-licensed ``thaitea/laya-vision``.
"""
import json
import os
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

LOCATE_ANYTHING_BACKBONE = "nvidia/LocateAnything-3B"
#: the Hub commit this adapter was written against (its weights, config and ``modeling_vit.py``)
LOCATE_ANYTHING_REVISION = "c32291ca5e996f5a7a485845b4f57a233936bba0"
MODEL_TYPE = "locateanything"
PROCESSOR_CONFIG = "laya_processor.json"

#: config a fresh agent on this backbone starts from (explicit keyword overrides still win)
AGENT_DEFAULTS = {
    "image_size": 448,
    "image_patch_size": 14,
    "image_scale_factor": 2,
    "image_interpolation": "bicubic",
    "dtype": "bf16",
}


def is_locate_anything(backbone: Optional[str]) -> bool:
    """Whether a ``cfg["backbone"]`` id names a LocateAnything checkpoint (``nvidia/LocateAnything-3B`` or a copy)."""
    return bool(backbone) and "locateanything" in backbone.replace("-", "").replace("_", "").lower()


def _moonvit_classes(repo: str, revision: Optional[str], token: Optional[str] = None):
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    kw = dict(revision=revision or LOCATE_ANYTHING_REVISION, token=token)
    return (get_class_from_dynamic_module("modeling_vit.MoonVitPretrainedModel", repo, **kw),
            get_class_from_dynamic_module("modeling_vit.MoonViTConfig", repo, **kw))


class LocateAnythingConfig:
    """The checkpoint's ``config.json`` as a plain dict, with the attributes ``laya.vlm`` reads off a backbone config
    (``model_type``, ``text_config.hidden_size``, ``_commit_hash``) and a ``save_pretrained`` for ``VLMAgent.save``.
    ``code_repo`` / ``code_revision`` say where ``modeling_vit.py`` comes from when the config is reloaded."""

    model_type = MODEL_TYPE

    def __init__(self, raw: Dict[str, Any], commit: Optional[str] = None, code_repo: str = LOCATE_ANYTHING_BACKBONE,
                 code_revision: Optional[str] = None):
        from transformers import Qwen2Config

        self.raw = {k: v for k, v in raw.items() if not k.startswith("laya_")}
        self._commit_hash = commit
        self.code_repo = raw.get("laya_code_repo", code_repo)
        self.code_revision = raw.get("laya_code_revision") or code_revision or commit or LOCATE_ANYTHING_REVISION
        text = dict(self.raw["text_config"])
        text.pop("_attn_implementation_autoset", None)
        self.text_config = Qwen2Config(**text)
        self.text_config._attn_implementation = "sdpa"
        self.vision_config = dict(self.raw["vision_config"])
        self.image_token_index = int(self.raw["image_token_index"])

    @classmethod
    def from_pretrained(cls, path_or_repo: str, revision: Optional[str] = None,
                        token: Optional[str] = None) -> "LocateAnythingConfig":
        if os.path.isdir(path_or_repo):
            with open(os.path.join(path_or_repo, "config.json")) as f:
                return cls(json.load(f))
        from huggingface_hub import hf_hub_download

        revision = revision or LOCATE_ANYTHING_REVISION
        path = hf_hub_download(path_or_repo, "config.json", revision=revision, token=token)
        with open(path) as f:
            raw = json.load(f)
        return cls(raw, commit=_snapshot_commit(path) or revision, code_repo=path_or_repo)

    def save_pretrained(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(dict(self.raw, laya_code_repo=self.code_repo, laya_code_revision=self.code_revision), f, indent=2)


def _snapshot_commit(path: str) -> Optional[str]:
    """``.../snapshots/<commit>/file`` -> ``<commit>``."""
    parts = os.path.normpath(path).split(os.sep)
    if "snapshots" in parts and parts.index("snapshots") + 1 < len(parts):
        c = parts[parts.index("snapshots") + 1]
        return c if len(c) == 40 else None
    return None


class LocateAnythingBackbone(nn.Module):
    """MoonViT + ``mlp1`` + ``Qwen2Model``, presenting the interface ``VLMDecisionModel`` uses on SmolVLM's
    ``Idefics3Model``: ``vision_model``, ``text_model``, ``get_image_features``, ``get_input_embeddings``,
    ``forward(input_ids, attention_mask, pixel_values | image_hidden_states, ...) -> last_hidden_state``."""

    def __init__(self, config: LocateAnythingConfig, token: Optional[str] = None):
        super().__init__()
        from transformers import Qwen2Model

        self.config = config
        vit_cls, vit_cfg_cls = _moonvit_classes(config.code_repo, config.code_revision, token)
        vc = {k: v for k, v in config.vision_config.items()
              if k not in ("auto_map", "_name_or_path", "_attn_implementation_autoset", "model_type", "torch_dtype")}
        vcfg = vit_cfg_cls(**vc)
        vcfg._attn_implementation = "sdpa"
        self.vision_model = vit_cls(vcfg)
        self.patch_size = int(vcfg.patch_size)
        self.merge = tuple(int(m) for m in vcfg.merge_kernel_size)
        v, d = int(vcfg.hidden_size) * self.merge[0] * self.merge[1], config.text_config.hidden_size
        self.mlp1 = nn.Sequential(nn.LayerNorm(v), nn.Linear(v, d), nn.GELU(), nn.Linear(d, d))
        self.text_model = Qwen2Model(config.text_config)
        self.image_token_id = config.image_token_index

    # -- loading ----------------------------------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: LocateAnythingConfig, dtype: torch.dtype = torch.float32,
                    token: Optional[str] = None) -> "LocateAnythingBackbone":
        prev = torch.get_default_dtype()
        torch.set_default_dtype(dtype)  # build the 3.8B parameters in their final dtype, not fp32 then a copy
        try:
            return cls(config, token=token)
        finally:
            torch.set_default_dtype(prev)

    @classmethod
    def from_pretrained(cls, repo: str = LOCATE_ANYTHING_BACKBONE, revision: Optional[str] = None,
                        dtype: torch.dtype = torch.float32, token: Optional[str] = None) -> "LocateAnythingBackbone":
        """Build and load the checkpoint's vision tower, connector and language model (its LM head is dropped),
        one safetensors shard at a time. ``config._commit_hash`` is the commit the files came from."""
        from huggingface_hub import hf_hub_download
        from safetensors import safe_open

        config = LocateAnythingConfig.from_pretrained(repo, revision=revision, token=token)
        model = cls.from_config(config, dtype=dtype, token=token)
        rev = config._commit_hash
        index = hf_hub_download(repo, "model.safetensors.index.json", revision=rev, token=token)
        with open(index) as f:
            shards = sorted(set(json.load(f)["weight_map"].values()))
        own = model.state_dict()  # views of the parameters: copying into them loads in place, one tensor at a time
        seen = set()
        for shard in shards:
            path = hf_hub_download(repo, shard, revision=rev, token=token)
            with safe_open(path, framework="pt") as f, torch.no_grad():
                for key in f.keys():
                    name = checkpoint_key(key)
                    if name is None:
                        continue
                    if name not in own:
                        raise ValueError("unexpected LocateAnything weight %r" % key)
                    own[name].copy_(f.get_tensor(key))
                    seen.add(name)
        missing = sorted(set(own) - seen)
        if missing:
            raise ValueError("LocateAnything checkpoint lacks %d weights, e.g. %s" % (len(missing), missing[:5]))
        return model

    # -- what VLMDecisionModel reads --------------------------------------------------------------------------

    @property
    def device(self) -> torch.device:
        return self.text_model.embed_tokens.weight.device

    @property
    def dtype(self) -> torch.dtype:
        return self.text_model.embed_tokens.weight.dtype

    def get_input_embeddings(self) -> nn.Module:
        return self.text_model.embed_tokens

    def get_image_features(self, pixel_values: torch.Tensor, pixel_attention_mask: Optional[torch.Tensor] = None,
                           return_dict: bool = True):
        """``[B, n_img, 3, S, S]`` -> ``pooler_output`` ``[n_real, (S / 28)^2, d]``. All-zero (padded) image slots are
        dropped, as ``Idefics3Model.get_image_features`` does; a real image never is (normalised pixels are in
        [-1, 1] and a black frame is all -1)."""
        del pixel_attention_mask  # a square resize never pads
        c, h, w = pixel_values.shape[-3:]
        x = pixel_values.reshape(-1, c, h, w)
        x = x[(x != 0).flatten(1).any(-1)]
        p = self.patch_size
        gh, gw = h // p, w // p
        if gh % self.merge[0] or gw % self.merge[1] or gh * p != h or gw * p != w:
            raise ValueError("image side %dx%d must be a multiple of patch %d x merge %s" % (h, w, p, self.merge))
        n = x.size(0)
        # the checkpoint's patchify: row-major patches per image, each [3, p, p], all images packed into one run
        patches = (x.to(self.vision_model.dtype).reshape(n, c, gh, p, gw, p).permute(0, 2, 4, 1, 3, 5)
                   .reshape(n * gh * gw, c, p, p))
        grid = torch.tensor([[gh, gw]] * n, device=x.device, dtype=torch.int64)
        merged = self.vision_model(patches, grid)  # list of [gh * gw / 4, 4 * vit_hidden]
        feats = self.mlp1(torch.cat(merged, 0)).reshape(n, -1, self.config.text_config.hidden_size)
        return SimpleNamespace(pooler_output=feats) if return_dict else (feats,)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                pixel_values: Optional[torch.Tensor] = None, pixel_attention_mask: Optional[torch.Tensor] = None,
                image_hidden_states: Optional[torch.Tensor] = None, position_ids: Optional[torch.Tensor] = None,
                past_key_values=None, use_cache: bool = False):
        emb = self.text_model.embed_tokens(input_ids)
        if pixel_values is not None and image_hidden_states is None:
            image_hidden_states = self.get_image_features(pixel_values, pixel_attention_mask).pooler_output
        if image_hidden_states is not None:
            slots = input_ids == self.image_token_id
            feats = image_hidden_states.reshape(-1, emb.size(-1)).to(emb.dtype)
            if int(slots.sum()) != feats.size(0):
                raise ValueError("%d image tokens in the ids but %d image features" % (int(slots.sum()), feats.size(0)))
            emb = emb.masked_scatter(slots[..., None], feats)
        return self.text_model(inputs_embeds=emb, attention_mask=attention_mask, position_ids=position_ids,
                               past_key_values=past_key_values, use_cache=use_cache)


def checkpoint_key(key: str) -> Optional[str]:
    """A LocateAnything checkpoint key -> this module's name for it; None for what it drops (the LM head)."""
    if key.startswith("language_model.model."):
        return "text_model." + key[len("language_model.model."):]
    if key.startswith(("vision_model.", "mlp1.")):
        return key
    return None


# ---------------------------------------------------------------------------------------------------------
# Processor
# ---------------------------------------------------------------------------------------------------------


class _ImageSettings:
    """The image-processor fields ``ImagePrep.apply`` sets and ``ImagePrep.check`` reads: the checkpoint's
    normalisation (mean = std = 0.5 after rescaling by 1/255), with the resize done by ``ImagePrep``."""

    do_resize = do_rescale = do_normalize = do_convert_rgb = True
    rescale_factor = 1 / 255
    image_mean = [0.5, 0.5, 0.5]
    image_std = [0.5, 0.5, 0.5]

    def __init__(self):
        self.max_image_size: Dict[str, int] = {}
        self.size: Dict[str, int] = {}


class LocateAnythingProcessor:
    """Stands in for the ``Idefics3Processor`` the sequence builders call: a Qwen2 tokenizer plus the image run
    ``<img>`` + ``<IMG_CONTEXT>`` x ``image_seq_len`` + ``</img>``, with pixels from the ``ImagePrep`` that
    ``ImagePrep.apply`` leaves on it (so the processor and the device-side backend give the same tensors)."""

    image_token = "<IMG_CONTEXT>"
    image_start_token = "<img>"
    image_end_token = "</img>"

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.image_processor = _ImageSettings()
        self.image_seq_len = 256
        self.laya_prep = None

    def image_run(self, image_seq_len: Optional[int] = None) -> str:
        n = self.image_seq_len if image_seq_len is None else image_seq_len
        return self.image_start_token + self.image_token * n + self.image_end_token

    def __call__(self, text: List[str], images: List[List[Any]], do_image_splitting: bool = False,
                 return_tensors: str = "pt", add_special_tokens: bool = False, **_):
        if do_image_splitting:
            raise ValueError("LocateAnything images are single fixed-size views; image splitting is not supported")
        if self.laya_prep is None:
            raise ValueError("call ImagePrep.apply(processor) first: it sets the image size")
        if len(text) != 1 or len(images) != 1:
            raise ValueError("one prompt at a time")
        pieces = text[0].split(self.image_token)
        if len(pieces) - 1 != len(images[0]):
            raise ValueError("%d image tokens for %d images" % (len(pieces) - 1, len(images[0])))
        prompt = self.image_run().join(pieces)
        ids = self.tokenizer(prompt, add_special_tokens=add_special_tokens)["input_ids"]
        from .preprocess import as_uint8_chw

        # one image at a time: a state's images may differ in size until they are resized
        views = [self.laya_prep.pixel_values(as_uint8_chw([img])) for img in images[0]]
        pv = torch.cat([v for v, _ in views])
        pam = torch.cat([m for _, m in views])
        return {"input_ids": torch.tensor([ids]), "pixel_values": pv[None], "pixel_attention_mask": pam[None]}

    def save_pretrained(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        self.tokenizer.save_pretrained(path)
        with open(os.path.join(path, PROCESSOR_CONFIG), "w") as f:
            json.dump({"processor_class": type(self).__name__}, f)

    @classmethod
    def from_pretrained(cls, path_or_repo: str, revision: Optional[str] = None,
                        token: Optional[str] = None) -> "LocateAnythingProcessor":
        from transformers import Qwen2Tokenizer

        if os.path.isdir(path_or_repo):
            return cls(Qwen2Tokenizer.from_pretrained(path_or_repo))
        return cls(Qwen2Tokenizer.from_pretrained(path_or_repo, revision=revision or LOCATE_ANYTHING_REVISION,
                                                  token=token))


def is_saved_processor(path: str) -> bool:
    return os.path.exists(os.path.join(path, PROCESSOR_CONFIG))
