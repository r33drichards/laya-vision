"""A fixed-budget device cache that keeps an agent loop's image work across ``VLMAgent.predict`` calls.

``predict``'s own prefix cache (``prefix_cache=``) lives for one call. An agent loop asks about the same screen
again and again -- a new set of questions about the same photo, a game that redraws only every few steps -- and
pays the preprocessing, the vision tower and the prefill each time. ``DeviceArena`` keeps that work on the device
between calls, in two tiers:

* **image tier** (the default) -- the vision tower + connector output per image (``[image_seq_len, d]``: 74 KB
  for SmolVLM-256M in bf16). A hit skips the pixel preprocessing and the vision tower. The rest of the call runs
  exactly as without an arena, on the same feature tensor, so answers are **bit-identical** to ``cache=None``
  (to rounding when a multi-image state mixes hits and misses: the misses are encoded without the hits). It
  hits whenever the image repeats, whatever the state text around it (a step counter, a score) and whatever the
  readout.
* **prefix tier** (opt-in, ``prefix_share > 0``) -- the backbone's key/value cache and last hidden states of the
  image run + state text, the prefix every row shares in the causal ``"terminator"`` layout. A hit also skips the
  prefill: only the question/option suffixes run (``VLMDecisionModel.forward_prefixed``). Pages of ``BLOCK``
  tokens, so an entry costs its length: ~24 KB per token for SmolVLM-256M in bf16, ~2.3 MB for the 64-token image
  run plus a short state (31x an image entry). Answers match the uncached path to float rounding, like
  ``prefix_cache=True``, and a hit reproduces the storing call bit for bit.

Why the prefix tier is off by default (``modal_app.py::bench_arena``, docs/arena-bench-*.json): on an L4 in bf16
a small backbone pass is launch-bound, so a suffix pass on a cached prefix costs about what the full pass does;
the image tier alone took the agent loop from 50 to 37 ms p50 and the game loop from 58 to 35 ms, and the prefix
tier added nothing (37-44 ms) while moving probabilities by up to 0.07 (the prefixed path's bf16 rounding, the same
as ``prefix_cache=True``). On a CPU the prefill is compute-bound and the prefix tier does pay: fp32, 8 cores, agent
loop 567 ms -> 185 (image) -> 99-107 (prefix).

The arena is claimed once, like CLM's ``VectorArena`` (and vLLM's KV cache): one flat allocation sized by a budget
(a fraction of the device's memory, ``"0.02"``, or a size, ``"512MiB"``), carved into the tiers at start-up.
Nothing grows afterwards; least-recently-used entries are evicted. Keys are content hashes (blake2b-128) of
everything the cached tensors depend on:

* a namespace: the checkpoint's identity (``laya.calibration.checkpoint_identity``: config + weight hashes,
  recomputed when a parameter is modified in place, so a hot reload or a training step stops matching old entries),
  ``PROMPT_FORMAT_VERSION``, the dtype, the readout and the image preprocessing;
* the decoded pixels of each image (with mode and size), and for the prefix tier the prefix's token ids (framing,
  image run, state text).

Prefix admission. Storing a prefix for a call that would not otherwise prefill costs an extra pass, so with
``prefix_cache=None`` a prefix is stored on its *second* sighting (a bounded "doorkeeper" set of recently seen
keys, as in TinyLFU), or at once when the call prefills anyway. ``prefix_cache=True`` always stores and
``False`` skips the prefix tier (and prefilling); the image tier applies either way.

Usage::

    arena = DeviceArena(agent, budget="256MiB")                    # image tier
    arena = DeviceArena(agent, budget="1GiB", prefix_share=0.875)  # + prefix tier (CPU)
    for step in loop:
        out = agent.predict(state, questions, cache=arena)
    print(arena.stats())
"""
from __future__ import annotations

import hashlib
import math
import re
import threading
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

DEFAULT_BUDGET = "0.02"  # fraction of total device memory, vLLM-style
BLOCK = 32               # tokens per prefix page
PREFIX_SHARE = 0.0       # of the arena, for the prefix tier: off by default (see the module docstring)
_UNITS = {"": 1, "B": 1, "KB": 10**3, "MB": 10**6, "GB": 10**9, "KIB": 1 << 10, "MIB": 1 << 20, "GIB": 1 << 30}


class CacheDisabled(Exception):
    pass


def parse_budget(spec: Any, total_bytes: int) -> int:
    """``0.02`` -> 2% of the device; ``512MiB`` / ``2GB`` -> that many bytes; ``0`` -> off."""
    if spec is None or spec == "":
        spec = DEFAULT_BUDGET
    if isinstance(spec, (int, float)):
        spec = repr(spec)
    m = re.fullmatch(r"\s*([0-9]*\.?[0-9]+)\s*([A-Za-z]*)\s*", str(spec))
    if not m:
        raise ValueError("cache budget %r is not a fraction (0.02) or a size (512MiB)" % (spec,))
    value, unit = float(m.group(1)), m.group(2).upper()
    if unit not in _UNITS:
        raise ValueError("unknown size unit %r; use B, KB, MB, GB, KiB, MiB or GiB" % m.group(2))
    if not unit:
        if not 0 <= value < 1:
            raise ValueError("a bare number is a fraction of device memory and must be in [0, 1); "
                             "give a unit (512MiB) for an absolute size")
        return int(value * total_bytes)
    return int(value * _UNITS[unit])


def image_digest(img: Any) -> bytes:
    """Content hash of one decoded image (a PIL image or a uint8 array): its pixels, shape and mode."""
    h = hashlib.blake2b(digest_size=16)
    if isinstance(img, np.ndarray):
        a = np.ascontiguousarray(img)
        h.update(repr((a.dtype.str, a.shape)).encode())
        h.update(memoryview(a).cast("B"))
    else:  # PIL
        h.update(repr((img.mode, img.size)).encode())
        h.update(img.tobytes())
    return h.digest()


class BlockPool:
    """Fixed-size blocks in one or more buffers that share the block index; entries own lists of blocks, LRU."""

    def __init__(self, buffers: Sequence[torch.Tensor]):
        self.buffers = list(buffers)
        self.capacity = self.buffers[0].shape[0]
        self.entries: "OrderedDict[bytes, Tuple[List[int], int]]" = OrderedDict()  # key -> (blocks, length)
        self.free: List[int] = list(range(self.capacity - 1, -1, -1))
        self.hits = self.misses = self.evictions = self.skipped = 0

    def get(self, key: bytes) -> Optional[Tuple[List[int], int]]:
        got = self.entries.get(key)
        if got is not None:
            self.entries.move_to_end(key)
        return got

    def claim(self, key: bytes, n_blocks: int, length: int) -> Optional[List[int]]:
        if n_blocks > self.capacity:
            self.skipped += 1
            return None
        old = self.entries.pop(key, None)
        if old is not None:
            self.free.extend(old[0])
        while len(self.free) < n_blocks:
            _, (blocks, _) = self.entries.popitem(last=False)  # least recently used
            self.free.extend(blocks)
            self.evictions += 1
        blocks = [self.free.pop() for _ in range(n_blocks)]
        self.entries[key] = (blocks, length)
        return blocks

    def used_blocks(self) -> int:
        return self.capacity - len(self.free)

    def stats(self, block_bytes: int) -> Dict[str, Any]:
        asked = self.hits + self.misses
        return {"entries": len(self.entries), "blocks": self.capacity, "used_blocks": self.used_blocks(),
                "reserved_mb": round(self.capacity * block_bytes / 10**6, 1),
                "used_mb": round(self.used_blocks() * block_bytes / 10**6, 2),
                "hits": self.hits, "misses": self.misses, "evictions": self.evictions, "too_big": self.skipped,
                "hit_rate": round(self.hits / asked, 4) if asked else None}


class DeviceArena:
    """One device allocation, carved at start-up into an image-feature tier and a prefix (KV) tier; never grown.

    ``agent`` fixes the shapes (layers, KV heads, head size, hidden size, image tokens), the dtype and the device.
    ``budget`` is a fraction of the device's total memory (``0.02``) or a size (``"512MiB"``); on a CPU a fraction
    is taken of 8 GiB. ``prefix_share`` of it goes to the prefix tier and the rest to the image tier: 0 (the
    default) is image features only, 1 prefixes only. ``block`` is the prefix tier's page size in tokens. Several
    agents with the same shapes may share an arena: the namespace in every key keeps their entries apart.
    """

    def __init__(self, agent, budget: Any = None, prefix_share: float = PREFIX_SHARE, block: int = BLOCK,
                 doorkeeper: Optional[int] = None):
        if not 0 <= prefix_share <= 1:
            raise ValueError("prefix_share must be in [0, 1], got %r" % (prefix_share,))
        model = agent.model
        self.device = agent.device
        self.dtype = model.encoder.dtype
        if self.device.type == "cuda":
            free, total = torch.cuda.mem_get_info(self.device)
        else:  # a CPU arena is still bounded, it just cannot ask the driver for a budget
            free = total = 8 << 30
        want = parse_budget(budget, total)
        if want <= 0:
            raise CacheDisabled("arena disabled by budget 0")
        tc = model.encoder.config.text_config
        self.layers = tc.num_hidden_layers
        self.kv_heads = getattr(tc, "num_key_value_heads", None) or tc.num_attention_heads
        self.head_dim = getattr(tc, "head_dim", None) or tc.hidden_size // tc.num_attention_heads
        self.hidden = tc.hidden_size
        self.block = int(block)
        self.image_tokens = agent.prep.image_seq_len
        item = torch.empty((), dtype=self.dtype).element_size()
        kv_block = self.layers * 2 * self.kv_heads * self.block * self.head_dim
        h_block = self.block * self.hidden
        img_block = self.image_tokens * self.hidden
        # never take more than 90% of what is free now: leave the model and its activations room
        n = min(want, int(free * 0.9)) // item
        n_kv = int(n * prefix_share) // (kv_block + h_block)
        n_img = (n - n_kv * (kv_block + h_block)) // img_block if prefix_share < 1 else 0
        if n_kv < 1 and n_img < 1:
            raise CacheDisabled("budget %r holds neither one prefix page nor one image" % (budget,))
        self.flat = torch.zeros(n_img * img_block + n_kv * (kv_block + h_block), device=self.device, dtype=self.dtype)
        cur = 0

        def carve(rows, shape):
            nonlocal cur
            size = rows * int(np.prod(shape))
            view = self.flat[cur:cur + size].view(rows, *shape)
            cur += size
            return view

        self.kv = BlockPool([carve(n_kv, (self.layers, 2, self.kv_heads, self.block, self.head_dim)),
                             carve(n_kv, (self.block, self.hidden))]) if n_kv else None
        self.images = BlockPool([carve(n_img, (self.image_tokens, self.hidden))]) if n_img else None
        self.kv_block_bytes = (kv_block + h_block) * item
        self.img_block_bytes = img_block * item
        self.reserved_mb = round(self.flat.numel() * item / 10**6, 1)
        # host-side, bounded: prefix keys seen once but not stored (admission on the second sighting)
        self._seen: "OrderedDict[bytes, None]" = OrderedDict()
        self._seen_cap = doorkeeper if doorkeeper is not None else max(64, 4 * (n_kv or 1))
        self._lock = threading.Lock()
        self._ns_key = None
        self._ns = b""

    # ------------------------------------------------------------------------------------------------ keys
    def namespace(self, agent) -> bytes:
        """What every cached tensor depends on besides the input: checkpoint generation, prompt format, dtype, prep."""
        from .calibration import checkpoint_identity
        from .vlm import PROMPT_FORMAT_VERSION

        ident = checkpoint_identity(agent)
        key = (id(agent), repr(sorted(ident.items())))
        if key != self._ns_key:
            h = hashlib.blake2b(digest_size=16)
            h.update(repr((key[1], PROMPT_FORMAT_VERSION, str(agent.model.encoder.dtype), agent.model.readout,
                           sorted(agent.prep.to_config().items()), agent.prep.backend,
                           getattr(agent.prep, "interpolation", None))).encode())
            self._ns_key, self._ns = key, h.digest()
        return self._ns

    @staticmethod
    def _key(*parts: bytes) -> bytes:
        h = hashlib.blake2b(digest_size=16)
        for p in parts:
            h.update(len(p).to_bytes(8, "little"))
            h.update(p)
        return h.digest()

    def accepts(self, agent) -> bool:
        """Whether this arena's pools fit ``agent`` (device, dtype, shapes); ``predict`` ignores one that does not."""
        tc = agent.model.encoder.config.text_config
        return (agent.device == self.device and agent.model.encoder.dtype == self.dtype
                and agent.prep.image_seq_len == self.image_tokens and tc.hidden_size == self.hidden
                and tc.num_hidden_layers == self.layers)

    # ------------------------------------------------------------------------------------------------ predict hooks
    def prefix_ids(self, agent, images: Sequence) -> Tuple[Dict[str, Any], bytes, List[bytes]]:
        """``(prefix, namespace, image digests)`` for ``predict``: the ``vlm_prefix`` ids without touching a pixel
        (``laya.preprocess.prefix_ids``, what the processor writes without splitting). With image splitting the
        view count depends on the pixels, so the processor runs as usual."""
        from .preprocess import prefix_ids
        from .vlm import MASK_PREFIX_TEXT, PREFIX_TEXT, processor_readout, vlm_prefix

        ns = self.namespace(agent)
        digests = [image_digest(im) for im in images]
        if not images or agent.prep.split_edge:
            return vlm_prefix(agent.processor, images, agent.prep), ns, digests
        text = MASK_PREFIX_TEXT if processor_readout(agent.processor) == "mask" else PREFIX_TEXT
        ids = prefix_ids(agent.processor, text, len(images), agent.prep.image_seq_len)
        return ({"ids": ids, "pixel_values": None, "pixel_attention_mask": None, "raw_images": None,
                 "n_images": len(images)}, ns, digests)

    def resolve(self, agent, rows: Sequence[Dict], n_image_run: int, images: Sequence, ns: bytes,
                digests: Sequence[bytes], prefix_cache: Optional[bool], prefill: bool
                ) -> Tuple[Optional[Dict[str, Any]], Optional[torch.Tensor]]:
        """``(prefix cache or None, image features or None)`` for ``predict``'s rows.

        ``prefix_cache`` is ``predict``'s argument, ``prefill`` whether ``predict`` would prefill this call without
        an arena. A stored prefix is used whenever it exists (unless ``prefix_cache=False``); a missing one is
        stored when the call prefills anyway or, with ``prefix_cache=None``, on its second sighting. Otherwise the
        call runs as it would without an arena (in-call prefix or full path), with the image tier's features.
        Image features are only produced when something reads them."""
        from .vlm import shared_prefix_len

        causal = agent.model.readout == "terminator"
        block = agent.model.option_attention == "block"
        if prefix_cache is not False and self.kv is not None and causal:
            n = prefix_boundary(rows, n_image_run, block)
            if n:
                ids = rows[0]["ids"][:n]
                key = self.prefix_key(ns, digests, ids)
                cached = self.lookup_prefix(key)
                if cached is not None:
                    return cached, None
                if prefill or self.admit(key):
                    feats = self.image_features(agent, images, ns, digests)
                    cached = agent.model.encode_prefix(torch.tensor([ids], device=agent.device), feats)
                    self.store_prefix(key, cached)
                    return cached, feats
        feats = self.image_features(agent, images, ns, digests) if images else None
        if prefill and causal:  # what predict does without an arena: a prefix shared within this call only
            n = shared_prefix_len(list(rows), n_image_run, block)
            if n:
                return agent.model.encode_prefix(torch.tensor([rows[0]["ids"][:n]], device=agent.device), feats), feats
        return None, feats

    # ------------------------------------------------------------------------------------------------ image tier
    def image_features(self, agent, images: Sequence, ns: bytes, digests: Sequence[bytes]) -> Optional[torch.Tensor]:
        """``[n_views, image_seq_len, d]`` for ``images``: stored rows where present, the vision tower for the rest."""
        from .vlm import vlm_prefix

        if not images:
            return None
        if self.images is None or agent.prep.split_edge:  # a split image is a variable number of views
            return _encode(agent, vlm_prefix(agent.processor, images, agent.prep))
        keys = [self._key(ns, b"img", d) for d in digests]
        pool, rows = self.images, {}
        with self._lock:
            for k in keys:
                got = pool.get(k)
                if got is not None and k not in rows:
                    rows[k] = pool.buffers[0][got[0][0]].clone()  # copied now: the claims below may evict it
            pool.hits += sum(k in rows for k in keys)
            pool.misses += sum(k not in rows for k in keys)
        todo = [k for k in dict.fromkeys(keys) if k not in rows]  # repeats inside one state are encoded once
        if not todo:
            return torch.stack([rows[k] for k in keys])
        if len(todo) == len(images):
            # every image is new and distinct: exactly the uncached path's call, so exactly its tensor
            feats = _encode(agent, vlm_prefix(agent.processor, list(images), agent.prep))
        else:
            feats = _encode(agent, vlm_prefix(agent.processor, [images[keys.index(k)] for k in todo], agent.prep))
        with self._lock:
            for k, f in zip(todo, feats):
                blocks = pool.claim(k, 1, 1)
                if blocks is not None:
                    pool.buffers[0][blocks[0]] = f.to(self.dtype)
        if len(todo) == len(images):
            return feats
        rows.update(zip(todo, feats))
        return torch.stack([rows[k] for k in keys])

    # ------------------------------------------------------------------------------------------------ prefix tier
    def prefix_key(self, ns: bytes, digests: Sequence[bytes], ids: Sequence[int]) -> bytes:
        return self._key(ns, b"kv", *digests, np.asarray(ids, dtype=np.int64).tobytes())

    def lookup_prefix(self, key: bytes) -> Optional[Dict[str, Any]]:
        """``{"h", "kv"}`` like ``encode_prefix`` returns (copies), or None."""
        pool = self.kv
        if pool is None:
            return None
        with self._lock:
            got = pool.get(key)
            if got is None:
                pool.misses += 1
                return None
            pool.hits += 1
            blocks, n = got
            idx = torch.as_tensor(blocks, device=self.device, dtype=torch.long)
            kv = pool.buffers[0].index_select(0, idx)       # [nb, L, 2, H, B, hd], copied under the lock
            h = pool.buffers[1].index_select(0, idx)        # [nb, B, d]
        nb = len(blocks)
        kv = kv.permute(1, 2, 3, 0, 4, 5).reshape(self.layers, 2, self.kv_heads, nb * self.block, self.head_dim)
        kv = kv[:, :, :, :n].contiguous()
        h = h.reshape(1, nb * self.block, self.hidden)[:, :n]
        return {"h": h, "kv": [(kv[i, 0][None], kv[i, 1][None]) for i in range(self.layers)]}

    def admit(self, key: bytes) -> bool:
        """Doorkeeper: True if this prefix was seen (and not stored) recently; records the sighting otherwise."""
        with self._lock:
            if key in self._seen:
                del self._seen[key]
                return True
            self._seen[key] = None
            while len(self._seen) > self._seen_cap:
                self._seen.popitem(last=False)
            return False

    def store_prefix(self, key: bytes, prefix: Dict[str, Any]) -> bool:
        pool = self.kv
        if pool is None:
            return False
        h = prefix["h"]
        n = h.size(1)
        k0 = prefix["kv"][0][0]
        if (h.size(0) != 1 or len(prefix["kv"]) != self.layers or tuple(k0.shape[1:2]) != (self.kv_heads,)
                or k0.size(-1) != self.head_dim or k0.size(2) != n):
            return False  # a layout this arena was not sized for: serve uncached
        nb = math.ceil(n / self.block)
        kv = torch.stack([torch.stack([k[0], v[0]]) for k, v in prefix["kv"]])  # [L, 2, H, n, hd]
        pad = nb * self.block - n
        if pad:
            kv = torch.nn.functional.pad(kv, (0, 0, 0, pad))
            h = torch.nn.functional.pad(h, (0, 0, 0, pad))
        kv = kv.view(self.layers, 2, self.kv_heads, nb, self.block, self.head_dim).permute(3, 0, 1, 2, 4, 5)
        with self._lock:
            blocks = pool.claim(key, nb, n)
            if blocks is None:
                return False
            idx = torch.as_tensor(blocks, device=self.device, dtype=torch.long)
            pool.buffers[0].index_copy_(0, idx, kv.to(self.dtype))
            pool.buffers[1].index_copy_(0, idx, h[0].view(nb, self.block, self.hidden).to(self.dtype))
        return True

    # ------------------------------------------------------------------------------------------------ reporting
    def stats(self) -> Dict[str, Any]:
        with self._lock:
            tiers = {}
            if self.kv is not None:
                tiers["prefix"] = dict(self.kv.stats(self.kv_block_bytes), block_tokens=self.block)
            if self.images is not None:
                tiers["image"] = self.images.stats(self.img_block_bytes)
        return {"device": str(self.device), "dtype": str(self.dtype).replace("torch.", ""),
                "reserved_mb": self.reserved_mb, **tiers}

    def clear(self) -> None:
        """Drop every entry (the allocation stays)."""
        with self._lock:
            for pool in (self.kv, self.images):
                if pool is not None:
                    pool.entries.clear()
                    pool.free = list(range(pool.capacity - 1, -1, -1))
            self._seen.clear()


def _encode(agent, prefix: Dict[str, Any]) -> torch.Tensor:
    """Vision tower + connector for a ``vlm_prefix`` result, as ``VLMAgent.predict`` runs it."""
    if prefix["raw_images"] is not None:
        return agent.model.encode_raw_images(prefix["raw_images"])
    return agent.model.encode_images(prefix["pixel_values"].to(agent.device, agent._torch_dtype()),
                                     prefix["pixel_attention_mask"].to(agent.device))


def prefix_boundary(rows: Sequence[Dict], n_image_run: int, block: bool) -> int:
    """Length of the image run + state text every row shares (the arena's prefix), 0 if not even the image run is."""
    from .vlm import shared_prefix_len

    n = shared_prefix_len(list(rows), n_image_run, block)
    n = min([n] + [r["state_end"] for r in rows])
    # at least one token of every row stays in the suffix; the image run must be inside the prefix
    return n if n >= max(1, n_image_run) else 0


__all__ = ["DeviceArena", "BlockPool", "CacheDisabled", "parse_budget", "image_digest", "prefix_boundary"]
