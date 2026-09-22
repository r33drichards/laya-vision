"""Cheap image preprocessing for the VLM decision model: one step, on the target device.

SmolVLM and ModernVBERT ship the same ``Idefics3Processor`` settings (512-pixel tiles, mean = std = 0.5, patch 16,
pixel shuffle 4, 64 image tokens), so everything here serves both backbones unchanged.

The Hugging Face ``Idefics3ImageProcessor`` costs about 13 ms of CPU per 210x160 Atari frame, and almost none
of it is useful work. Its ``preprocess`` resizes twice with LANCZOS:

    210x160  --(size.longest_edge = 2048)-->  2048x1560  --(max_image_size = 512)-->  512x512

The first hop upscales the frame to 3.2 megapixels only for the second to throw it away again; measured with
``tests/test_vlm.py``'s frames the two hops are 5.6 ms and 6.1 ms of the 12.7 ms total. Both hops are *linear*
in the input, and separable, so the whole chain is one pair of small matrices:

    out[c, i, j] = sum_p sum_q  Wh[i, p] * Ww[j, q] * in[c, p, q]

``ImagePrep`` builds ``Wh`` and ``Ww`` once (``_axis_weights`` reproduces torch's antialiased resample weights
to float64 agreement, see the test) and then each frame costs two matmuls, on whatever device the model lives
on. Nothing else changes: uint8 in, the same rounding back to uint8 that torchvision does, then the processor's
fused rescale+normalize, which for ``image_mean = image_std = 0.5`` and ``rescale_factor = 1/255`` is
``(v - 127.5) / 127.5``. The pixel attention mask is all ones, because a square resize never pads.

Filters (``interpolation``):

* ``"processor"`` (default) is the two-hop chain above, composed into that one matrix pair, so it runs on any
  device. On real Atari frames it lands 0.05 grey levels from the Hugging Face output on average (0.13 at 256).
  It is not bit-exact: the processor rounds *and clamps* the 2048-pixel intermediate back to uint8, and clamping
  is not linear, so LANCZOS overshoot at a hard edge survives here where the processor cut it off. That shows up
  as a handful of pixels per frame up to about 19 levels out, and nowhere else.
* ``"lanczos"`` is a single hop straight to the target size, same machinery. Indistinguishable from
  ``"processor"`` at 512 (the 2048 hop is nearly lossless), twice as far off at 256.
* ``"bicubic"``, ``"bilinear"``, ``"nearest"`` hand a single resize to torchvision. Furthest from the processor
  (bicubic averages 0.41 levels, bilinear 0.79), and the only option if you want to avoid the matmul: note that
  torchvision has no CUDA LANCZOS kernel, so a plain ``tvF.resize`` on a GPU silently drops to BICUBIC anyway.

``image_size`` also sets how many tokens an image costs the language model: ``(image_size / patch_size)^2 /
scale_factor^2``, which is 64 at 512 and 16 at 256 for SmolVLM-256M (patch 16, pixel-shuffle 4). The processor
has to agree, because it is what writes those ``<image>`` tokens into the prompt, so ``apply(processor)`` sets
``image_seq_len`` and ``max_image_size`` on it.

Image splitting (``split_edge``) is the processor's own tiling, which this repo otherwise turns off: the image is
resized so its longest edge is ``split_edge`` (upscaled too, as the processor does), cut into ``image_size``
tiles, and followed by one downscaled global view. That is up to ``(split_edge / image_size)^2 + 1`` tiles of
``image_seq_len`` tokens each: 17 x 64 at the processor's default 2048, 5 x 64 at 1024. It runs only through the
Hugging Face processor (``backend="processor"``); the device-side path above has no tiling.
"""
import functools
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

BACKENDS = ("processor", "gpu")
#: filters run as a precomputed linear operator (exact on any device) vs handed to torchvision
LINEAR_INTERPOLATIONS = ("processor", "lanczos")
TV_INTERPOLATIONS = ("bicubic", "bilinear", "nearest")
INTERPOLATIONS = LINEAR_INTERPOLATIONS + TV_INTERPOLATIONS

# The processor's own values for SmolVLM/Idefics3; ``ImagePrep.check`` asserts the loaded processor still has them.
RESCALE_FACTOR = 1 / 255
IMAGE_MEAN = IMAGE_STD = 0.5
STAGE1 = 2048  # the processor's ``size.longest_edge``


@dataclass(frozen=True)
class ImagePrep:
    """How a frame becomes the model's pixel tensor. Written to ``vlm_agent_config.json``, so a checkpoint
    records what it was trained with and play matches it.

    * ``image_size``: the square side fed to the vision tower (the processor's ``max_image_size.longest_edge``).
    * ``backend``: ``"gpu"`` for the device-side path here, ``"processor"`` for the Hugging Face processor.
    * ``interpolation``: filter for the ``"gpu"`` backend, see the module docstring.
    * ``split_edge``: 0 for one ``image_size`` view per image (the released checkpoints); otherwise the longest
      edge an image is resized to before the processor cuts it into tiles, see the module docstring.
    """

    image_size: int = 512
    backend: str = "gpu"
    interpolation: str = "processor"
    patch_size: int = 16
    scale_factor: int = 4
    split_edge: int = 0

    def __post_init__(self):
        if self.backend not in BACKENDS:
            raise ValueError("backend must be one of %s, got %r" % (BACKENDS, self.backend))
        if self.interpolation not in INTERPOLATIONS:
            raise ValueError("interpolation must be one of %s, got %r" % (INTERPOLATIONS, self.interpolation))
        step = self.patch_size * self.scale_factor
        if self.image_size % step:
            raise ValueError("image_size %d must be a multiple of patch_size * scale_factor = %d"
                             % (self.image_size, step))
        if self.split_edge:
            if self.split_edge < self.image_size:
                raise ValueError("split_edge %d is below image_size %d; use 0 to turn splitting off"
                                 % (self.split_edge, self.image_size))
            if self.backend != "processor":
                raise ValueError("image splitting runs through the Hugging Face processor; pass preprocess='processor'")

    @property
    def image_seq_len(self) -> int:
        """Language-model tokens one image costs: vision patches after the connector's pixel shuffle."""
        return (self.image_size // self.patch_size) ** 2 // self.scale_factor**2

    @property
    def on_gpu(self) -> bool:
        return self.backend == "gpu"

    @property
    def max_tiles(self) -> int:
        """Most vision-tower views one image can become: the tile grid plus the global view when splitting."""
        if not self.split_edge:
            return 1
        return math.ceil(self.split_edge / self.image_size) ** 2 + 1

    # -- config round-trip ------------------------------------------------------------------------------------

    @classmethod
    def from_config(cls, cfg: Dict[str, Any], default_backend: str = "processor") -> "ImagePrep":
        """Read the ``vlm_agent_config.json`` keys. A config written before they existed means 512 through
        the Hugging Face processor, so ``default_backend`` is ``"processor"`` for a load and ``"gpu"`` for a
        fresh agent. Splitting needs the processor, so it is the default backend whenever ``image_split_edge``
        is set."""
        split_edge = int(cfg.get("image_split_edge") or 0)
        return cls(image_size=int(cfg.get("image_size", 512)),
                   backend=cfg.get("preprocess") or ("processor" if split_edge else default_backend),
                   interpolation=cfg.get("image_interpolation", "processor"),
                   split_edge=split_edge)

    def to_config(self) -> Dict[str, Any]:
        return {"image_size": self.image_size, "preprocess": self.backend, "image_interpolation": self.interpolation,
                "image_split_edge": self.split_edge}

    # -- processor agreement ----------------------------------------------------------------------------------

    def apply(self, processor) -> "ImagePrep":
        """Point the processor at ``image_size``: the ``<image>`` run it writes, and its own resize target.

        Both backends need this -- the token ids come from the processor either way. It also leaves the prep on
        the processor as ``laya_prep``, which is how ``vlm_prefix`` finds the path without every caller in
        between having to pass it (the training loader, ``collect_logits``).
        """
        processor.image_processor.max_image_size = {"longest_edge": self.image_size}
        processor.image_processor.size = {"longest_edge": self.split_edge or STAGE1}
        processor.image_seq_len = self.image_seq_len
        processor.laya_prep = self
        return self

    def check(self, processor) -> None:
        """Fail loudly if the processor is not the one this module's normalisation was written for."""
        ip = processor.image_processor
        mean, std = np.atleast_1d(ip.image_mean), np.atleast_1d(ip.image_std)
        flat = lambda v: float(v.min()) if v.min() == v.max() else tuple(v)  # noqa: E731
        expected = [
            ("do_resize", ip.do_resize, True), ("do_rescale", ip.do_rescale, True),
            ("do_normalize", ip.do_normalize, True), ("do_convert_rgb", ip.do_convert_rgb, True),
            ("rescale_factor", float(ip.rescale_factor), RESCALE_FACTOR),
            ("image_mean", flat(mean), IMAGE_MEAN), ("image_std", flat(std), IMAGE_STD),
            ("max_image_size", ip.max_image_size.get("longest_edge"), self.image_size),
            ("size", _longest_edge(ip.size), self.split_edge or STAGE1),
            ("image_seq_len", processor.image_seq_len, self.image_seq_len),
        ]
        wrong = ["%s=%r (expected %r)" % (k, got, want) for k, got, want in expected if got != want]
        if wrong:
            raise ValueError("this preprocessing path assumes the Idefics3 processor settings SmolVLM and ModernVBERT "
                             "ship; " + ", ".join(wrong))

    # -- the actual work --------------------------------------------------------------------------------------

    def pixel_values(
        self,
        images: Any,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """``[..., 3, image_size, image_size]`` pixels and ``[..., image_size, image_size]`` attention mask.

        ``images`` is a uint8 tensor or array shaped ``[..., H, W, 3]`` (channels last, as ALE and PIL give
        them) or ``[..., 3, H, W]``, or anything ``as_uint8_chw`` accepts. Leading dimensions are kept, so a
        ``[B, n_img, H, W, 3]`` training batch comes back as ``[B, n_img, 3, S, S]``.
        """
        x = as_uint8_chw(images, device)
        lead, (c, h, w) = x.shape[:-3], x.shape[-3:]
        x = self._resize(x.reshape(-1, c, h, w))
        x = (x.to(dtype) - IMAGE_MEAN / RESCALE_FACTOR) / (IMAGE_STD / RESCALE_FACTOR)
        s = self.image_size
        mask = torch.ones(x.shape[0], s, s, dtype=torch.bool, device=x.device)
        return x.reshape(*lead, c, s, s), mask.reshape(*lead, s, s)

    def _resize(self, x: torch.Tensor) -> torch.Tensor:
        """uint8 ``[n, 3, H, W]`` -> uint8 ``[n, 3, image_size, image_size]``."""
        if self.interpolation in LINEAR_INTERPOLATIONS:
            wh, ww = resize_operator(x.shape[-2], x.shape[-1], self.image_size, self.interpolation,
                                     device=x.device, dtype=torch.float32)
            y = torch.einsum("ip,ncpq,jq->ncij", wh, x.to(torch.float32), ww)
            return y.round_().clamp_(0, 255).to(torch.uint8)  # as torchvision does for a uint8 input
        import torchvision.transforms.v2.functional as tvF
        from torchvision.transforms import InterpolationMode

        mode = {"bicubic": InterpolationMode.BICUBIC, "bilinear": InterpolationMode.BILINEAR,
                "nearest": InterpolationMode.NEAREST_EXACT}[self.interpolation]
        return tvF.resize(x, [self.image_size, self.image_size], interpolation=mode, antialias=True)


def _longest_edge(size) -> Optional[int]:
    """``size["longest_edge"]`` whether the image processor holds a plain dict or a ``SizeDict``."""
    if isinstance(size, dict):
        return size.get("longest_edge")
    return getattr(size, "longest_edge", None)


# ---------------------------------------------------------------------------------------------------------
# Exact resample operators
# ---------------------------------------------------------------------------------------------------------


def _lanczos(x: torch.Tensor, a: float = 3.0) -> torch.Tensor:
    """Lanczos-3, the filter behind ``PILImageResampling.LANCZOS`` and torch's ``_upsample_lanczos2d_aa``."""
    x = x.abs()
    out = torch.zeros_like(x)
    at_zero = x < 1e-12
    out[at_zero] = 1.0
    inside = (~at_zero) & (x < a)
    xs = x[inside]
    out[inside] = torch.sinc(xs) * torch.sinc(xs / a)
    return out


def _axis_weights(n_in: int, n_out: int, dtype=torch.float64) -> torch.Tensor:
    """``[n_out, n_in]`` weights of torch's antialiased LANCZOS resample along one axis.

    Same construction as ATen's ``upsample`` antialias path: an output pixel's kernel is centred on
    ``(i + 0.5) * scale``, stretched by ``max(1, scale)`` when downsampling, clipped at the borders and
    renormalised. ``tests/test_vlm.py`` checks the result against ``tvF.resize`` itself.
    """
    if n_in == n_out:
        return torch.eye(n_in, dtype=dtype)
    scale = n_in / n_out
    stretch = max(1.0, scale)
    support = 3.0 * stretch
    centre = (torch.arange(n_out, dtype=dtype) + 0.5) * scale
    span = int(math.ceil(support)) * 2 + 2
    lo = torch.floor(centre - support + 0.5).clamp(min=0)
    idx = lo[:, None] + torch.arange(span, dtype=dtype)[None, :]
    w = _lanczos((idx + 0.5 - centre[:, None]) / stretch) * (idx < n_in)
    out = torch.zeros(n_out, n_in, dtype=dtype)
    out.scatter_add_(1, idx.clamp(0, n_in - 1).long(), w)
    return out / out.sum(1, keepdim=True)


def stage1_size(height: int, width: int, longest_edge: int = STAGE1) -> Tuple[int, int]:
    """The processor's first hop: longest edge to ``longest_edge``, the other rounded up to even."""
    if width >= height:
        width, height = longest_edge, int(longest_edge * height / width)
        height += height % 2
    else:
        height, width = longest_edge, int(longest_edge * width / height)
        width += width % 2
    return max(height, 1), max(width, 1)


@functools.lru_cache(maxsize=16)
def resize_operator(in_h: int, in_w: int, out: int, interpolation: str = "processor",
                    device=None, dtype=torch.float32) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(Wh [out, in_h], Ww [out, in_w])``: the whole resize as two matrices, ready on ``device``.

    ``"lanczos"`` is one hop to ``out`` x ``out``; ``"processor"`` composes the two hops the Hugging Face
    processor does, which is possible because each hop is linear. What the composition cannot reproduce is the
    processor's rounding *and clamping* of the intermediate back to uint8 -- see the module docstring.
    """
    if interpolation not in LINEAR_INTERPOLATIONS:
        raise ValueError("resize_operator is for %s, got %r" % (LINEAR_INTERPOLATIONS, interpolation))
    wh, ww = _axis_weights(in_h, out), _axis_weights(in_w, out)
    if interpolation == "processor":
        mid_h, mid_w = stage1_size(in_h, in_w)
        wh = _axis_weights(mid_h, out) @ _axis_weights(in_h, mid_h)
        ww = _axis_weights(mid_w, out) @ _axis_weights(in_w, mid_w)
    return wh.to(device, dtype).contiguous(), ww.to(device, dtype).contiguous()


# ---------------------------------------------------------------------------------------------------------
# Input coercion
# ---------------------------------------------------------------------------------------------------------


def as_uint8_chw(images: Any, device: Optional[torch.device] = None) -> torch.Tensor:
    """Anything image-ish -> a uint8 tensor ``[..., 3, H, W]`` on ``device``.

    Accepts a uint8 torch tensor or numpy array (channels first or last), a PIL image, a path, or a (possibly
    nested) sequence of those, which is stacked. Frames must agree in size to be stacked.
    """
    if isinstance(images, torch.Tensor):
        x = images
    elif isinstance(images, np.ndarray) and images.dtype == np.uint8:
        x = torch.from_numpy(np.ascontiguousarray(images))
    elif isinstance(images, (list, tuple)):
        parts = [as_uint8_chw(im, device) for im in images]
        shapes = {tuple(p.shape) for p in parts}
        if len(shapes) != 1:
            raise ValueError("cannot stack images of different shapes: %s" % sorted(shapes))
        return torch.stack(parts)
    else:
        from PIL import Image

        im = images if isinstance(images, Image.Image) else Image.open(images)
        x = torch.from_numpy(np.array(im.convert("RGB")))  # np.array copies: PIL's buffer is read-only
        if not isinstance(images, Image.Image):
            im.close()
    if x.dtype != torch.uint8:
        raise TypeError("expected uint8 pixels, got %s" % x.dtype)
    if x.ndim < 3:
        raise ValueError("expected at least 3 dims (H, W, C), got %s" % (tuple(x.shape),))
    if x.shape[-1] in (1, 3) and x.shape[-3] not in (1, 3):
        x = x.movedim(-1, -3)  # channels last -> channels first
    if x.shape[-3] == 1:
        x = x.expand(*x.shape[:-3], 3, *x.shape[-2:])
    if x.shape[-3] != 3:
        raise ValueError("expected 3 colour channels, got shape %s" % (tuple(x.shape),))
    if device is not None:
        x = x.to(device, non_blocking=True)
    return x.contiguous()


# ---------------------------------------------------------------------------------------------------------
# Prompt ids
# ---------------------------------------------------------------------------------------------------------


@functools.lru_cache(maxsize=32)
def _expanded_ids(tokenizer, text: str, fake: str, glob: str, image: str, n_images: int,
                  image_seq_len: int) -> Tuple[int, ...]:
    expansion = fake + glob + image * image_seq_len + fake
    return tuple(tokenizer(text + expansion * n_images, add_special_tokens=False)["input_ids"])


def prefix_ids(processor, text: str, n_images: int, image_seq_len: Optional[int] = None) -> List[int]:
    """``text`` with each image replaced by the processor's ``<image>`` run, tokenized (cached).

    This is exactly what ``processor(text=..., images=...)`` produces with ``do_image_splitting=False`` (see
    ``Idefics3Processor.replace_image_token``; SmolVLM2's ``SmolVLMProcessor`` writes the same run and names the
    global tag ``global_image_token``), without touching a single pixel: the ids depend only on the image count
    and ``image_seq_len``. ``tests/test_vlm.py`` asserts the two agree.
    """
    if image_seq_len is None:
        image_seq_len = processor.image_seq_len
    glob = getattr(processor, "global_image_tag", None) or processor.global_image_token
    return list(_expanded_ids(processor.tokenizer, text, processor.fake_image_token, glob,
                              processor.image_token, n_images, image_seq_len))


# ---------------------------------------------------------------------------------------------------------
# Reusing the previous frame's encoder output
# ---------------------------------------------------------------------------------------------------------


class FrameFeatureCache:
    """Vision-tower output per distinct frame, so two-frame play encodes each frame once instead of twice.

    In ``--frames 2`` play the model sees ``[previous, current]`` every step, and this step's *current* frame is
    the next step's *previous* frame, so half of the vision-tower work is a repeat. Keying on frame content
    rather than on episode also catches the repeats inside one batch: an episode's first step, and the step after
    an auto-FIRE, pass the same frame as both images.

    The key is a 64-bit hash of the raw bytes, confirmed by an exact ``np.array_equal`` before a hit counts, so a
    collision costs a re-encode and never a wrong answer. Only the frames seen in the last ``keep`` calls are
    held, which bounds the cache at a couple of frames per episode in flight.
    """

    def __init__(self, keep: int = 2):
        self.keep = keep
        self._gens: List[Dict[int, Tuple[np.ndarray, torch.Tensor]]] = [{}]
        self.hits = self.misses = 0

    def _lookup(self, key: int, frame: np.ndarray) -> Optional[torch.Tensor]:
        for gen in self._gens:
            got = gen.get(key)
            if got is not None and np.array_equal(got[0], frame):
                return got[1]
        return None

    def features(self, encode, frames: Sequence[np.ndarray]) -> List[torch.Tensor]:
        """Features for each frame, encoding only the ones not already known.

        ``encode(list_of_frames) -> [n, image_seq_len, d]`` runs the vision tower and connector. Repeats inside
        ``frames`` are encoded once.
        """
        keys = [hash(np.ascontiguousarray(f).tobytes()) for f in frames]
        out: List[Any] = [None] * len(frames)
        todo: List[int] = []
        for i, (k, f) in enumerate(zip(keys, frames)):
            hit = self._lookup(k, f)
            if hit is not None:
                out[i] = hit
                self.hits += 1
                continue
            same = next((j for j in todo if keys[j] == k and np.array_equal(frames[j], f)), None)
            if same is None:
                todo.append(i)
            else:
                out[i] = ("same", same)
        fresh = encode([frames[i] for i in todo]) if todo else None
        gen: Dict[int, Tuple[np.ndarray, torch.Tensor]] = {}
        for n, i in enumerate(todo):
            out[i] = fresh[n]
            gen[keys[i]] = (np.ascontiguousarray(frames[i]), fresh[n])
            self.misses += 1
        for i, v in enumerate(out):
            if isinstance(v, tuple):
                out[i] = out[v[1]]
        self._gens = ([gen] + self._gens)[: self.keep]
        return out

    @property
    def stats(self) -> Dict[str, float]:
        n = self.hits + self.misses
        return {"hits": self.hits, "misses": self.misses, "hit_rate": self.hits / n if n else 0.0}


__all__ = ["ImagePrep", "FrameFeatureCache", "as_uint8_chw", "prefix_ids", "resize_operator", "stage1_size",
           "BACKENDS", "INTERPOLATIONS"]
