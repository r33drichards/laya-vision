# Try the LocateAnything backbone

[LocateAnything-3B](https://research.nvidia.com/labs/lpr/locate-anything/) is NVIDIA's grounding model: a MoonViT
vision tower at native resolution and a Qwen2.5-3B language model trained on 12M images and 785M boxes. Laya Vision
can use it in place of SmolVLM as an **experimental** backbone. The decision head, `predict` and the training loop
are unchanged. No checkpoint has been trained or evaluated on it yet, so there are no numbers to compare.

**Licence.** The LocateAnything weights are under the
[NVIDIA License](https://huggingface.co/nvidia/LocateAnything-3B/blob/main/LICENSE), for non-commercial research
only, and the language model is under the Qwen Research License. A model fine-tuned from them inherits those terms.
Do not publish one as `thaitea/laya-vision` or under Apache-2.0.

## What changes

The adapter is `laya/locate_anything.py`.

| | SmolVLM-256M | LocateAnything-3B |
|---|---|---|
| Parameters | 237M | 3.8B (400M vision, 3.1B language) |
| Vision tower | SigLIP, 512 px tile, pixel shuffle 4 | MoonViT, patch 14, 2×2 merge |
| Tokens per image | 64 | 256 at the default `image_size=448` |
| Language model | SmolLM2, 30 layers, d=576 | Qwen2.5-3B, 36 layers, d=2048 |
| Default dtype | fp32 | bf16 |
| Readout | option terminator (causal) | the same |

- **The language model runs as a plain causal decoder.** LocateAnything's Hub code builds a block-diffusion mask
  for its parallel box decoding and needs the transformers version it was written for. The adapter loads the
  same weights into transformers' own `Qwen2Model`, so `option_attention="block"`, prefix caching and every
  freezing stage work as they do on SmolVLM. The LM head that emits boxes is not loaded.
- **The vision tower is the checkpoint's own MoonViT.** Its code is fetched from the Hub at the pinned commit
  (`LOCATE_ANYTHING_REVISION`) and never copied into this repository.
- **Images are fixed squares, not native resolution.** Every image is resized to `image_size` (a multiple of 28), so
  images batch like SmolVLM's tiles. This gives up some of what MoonViT is good at, the fine detail of large
  images. `image_size=672` gives 576 tokens per image.
- **The prompt uses Laya's framing, not Qwen's chat template.** The head is trained on top, so this matters less
  than it would for generation.

## Build an agent

```python
import laya

agent = laya.load_vlm(backbone="nvidia/LocateAnything-3B", device="cuda")   # 7.6 GB download, bf16
agent.cfg["backbone_revision"]                                              # the pinned Hub commit
```

The head is untrained, so its answers mean nothing until you fine-tune. Keyword overrides still work, for example
`image_size=672` or `dtype="fp32"`.

## Fine-tune on Modal

Start by training only the head. It compares LocateAnything's features with SmolVLM's at the lowest cost, and
fits on an A10G:

```bash
modal run --detach modal_app.py::finetune --backbone nvidia/LocateAnything-3B \
    --freeze head --minutes 30 --run-name la3b-head-30m
```

Runs are saved under `/ckpt/locateanything-3b/<run>` on the checkpoint volume. See [Run jobs on Modal](run-on-modal.md)
for the setup. For comparison, run the same command with `--backbone HuggingFaceTB/SmolVLM-256M-Instruct` and the
same `--freeze head`.

Unfreezing the language model (`--freeze last_n` or `full`, or `finetune_long`, which always trains the full model)
does not fit the job definitions as they are. The backbone's weights are bf16 by default, and at a backbone learning
rate of 2e-5 most AdamW updates are smaller than bf16 can represent, so such a run needs `--dtype fp32`. In fp32 the
weights, gradients and AdamW state of the language model come to about 55 GB before activations. That needs an 80 GB
GPU, and neither job requests one: change `gpu=` on the function for that run.

## Evaluate

Evaluate it like any other checkpoint ([Evaluate a checkpoint](evaluate.md)). Expect it to be much slower than the
201M checkpoint: it has about 16 times the parameters, and each image costs four times the tokens. A headline number
from such a run follows the same evidence rules as any other, which are listed in
[AGENTS.md](https://github.com/r33drichards/laya-vision/blob/main/AGENTS.md).

## Tests

```bash
python -m pytest -q tests/test_locate_anything.py                               # a tiny random model, seconds
LAYA_TEST_LOCATE_ANYTHING=1 python -m pytest -q tests/test_locate_anything.py   # also loads the real weights
```
