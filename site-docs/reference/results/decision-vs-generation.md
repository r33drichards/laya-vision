# Typed answers vs generated JSON

Fifteen questions (6 `choice`, 5 `noul`, 4 `score`) about one image and a short state text, on one L4 in bf16,
the median of 5 runs after 2 warm-up rounds, with the clock synchronized with CUDA. Every path starts from a decoded
image, so every time includes the CPU preprocessing:

| Path | Time | Output tokens | Result |
|---|---:|---:|---|
| `predict`, thaitea/laya-vision, default `batch_size=8` (2 forward passes) | **0.144 s** | **0** | 15 typed answers with probabilities |
| `predict`, `batch_size=15` (1 forward pass) | **0.098 s** | **0** | the same answers |
| Base SmolVLM-256M-Instruct asked for one compact JSON array, same 512-pixel view | 3.216 s (22x) | 89 | prose, no JSON array; 0/15 usable |
| The same with the base model's shipped image splitting (17 views) | 0.613 s (4.3x) | 9 | "The image does not contain any text."; 0/15 usable |
| The base model asked one question per `generate` call, 15 calls | 4.489 s (31x) | 105 | 1/15 strictly valid; 3/15 agree with `predict` after lenient parsing |

## Reading it

This is a systems comparison, like [SemIf's](https://github.com/r33drichards/SemIf), not a quality one. The two
paths do not share weights: the checkpoint's backbone was fine-tuned with its head, and the base model it started
from cannot follow the compact-array instruction at this size, so its time is the time to say whatever it said.

- A valid array of the checkpoint's own answers is 41 tokens in this tokenizer. At the 36 ms per token the base
  model decoded here it would take about 1.6 s, or 11x `predict`: an estimate, not a measurement.
- Generation also grows with every token of output, where `predict` costs one forward pass per batch of questions
  whatever their answers.
- Agreement is with `predict`'s argmax, not with the truth: `predict` itself says the circle is the largest shape
  on this card. The image is drawn by the script, so the fixture is owned by this repository.

## Reproduce

```bash
modal run modal_app.py::decision_vs_generation --output results/raw/<new>.json
```

The [raw report](https://github.com/r33drichards/laya-vision/blob/main/results/raw/decision-vs-generation-l4.json) has the prompts, every run's timings, the
generated text and token timeline, the pinned revisions of both models, the torch and transformers versions and the
git sha of the code measured. The script is
[`benchmarks/decision_vs_generation.py`](https://github.com/r33drichards/laya-vision/blob/main/benchmarks/decision_vs_generation.py).
