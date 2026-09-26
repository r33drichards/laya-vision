# predict() and the answer schema

The Python API is `laya.load_vlm` and `VLMAgent.predict`, in
[`laya/vlm.py`](https://github.com/r33drichards/laya-vision/blob/main/laya/vlm.py). The output schema is Laya's.

## Loading a checkpoint

```python
agent = laya.load_vlm(model_id_or_path=None, backbone=None, device=None, token=None,
                      revision=None, backbone_revision=None, **kwargs)
```

| Argument | Meaning |
|---|---|
| `model_id_or_path` | A Hub id such as `"thaitea/laya-vision"`, or a local checkpoint directory. `None` builds a fresh agent on `backbone`, whose head is untrained. |
| `backbone` | The backbone for a fresh agent, e.g. `"HuggingFaceTB/SmolVLM-256M-Instruct"` or `"ModernVBERT/modernvbert"`. |
| `device` | A torch device string; defaults to the best available. |
| `token` | A Hugging Face token, for private repos. |
| `revision`, `backbone_revision` | Pin the Hub commit of the checkpoint and of its backbone. The resolved commits are recorded in `predict`'s `provenance`. |

## The state

The first argument of `predict`. It can be:

- a dictionary: the keys `"image"` (a PIL image, a path or encoded bytes) and `"images"` (a list of them) are the
  images; every other key is serialised to JSON text (like `json.dumps`) and read as context;
- a PIL image on its own;
- text, or a list (for example conversation turns), serialised the same way.

Every training record put its text under one key, `"context"`, so the checkpoints have read
`{"context": "..."}` (JSON, newlines escaped) and nothing else. `laya.prompt.make_state(image=img, context=text)`
builds that layout. Other keys work but were never seen in training.

`predict(..., state_format=...)` or `VLMAgent(..., state_format=...)` renders the non-image keys another way:
`"json"` (the default, what every released checkpoint was trained on), `"prose"` (`key: value` lines and
`- item` bullets, as in the CLM repository) or `"text"` (the values alone). Anything other than the checkpoint's own
format changes its input ids. A checkpoint trained with another format records it in `vlm_agent_config.json`
and is served with it by default. `laya/prompt.py` holds this rendering, which data preparation, training, the
evals and `predict` all share.

## Question types

The second argument is a dictionary of questions keyed by an id of your choice. Each question has a `type`, its
`instructions` (text, or any JSON value, which is serialised), and, depending on the type, `criteria`:

| `type` | `criteria` | The model reads |
|---|---|---|
| `choice` | A list of option names, or a dictionary of option name → description | One option per name, rendered as `name` or `name: description` |
| `score` | A list of level descriptions, lowest first | `level 0: …`, `level 1: …`, one per level |
| `noul` | Optional: a dictionary with `"true"` and/or `"false"` descriptions | Always two options, `false: …` and `true: …`, with default descriptions |

## Arguments

```python
agent.predict(state, questions, n_permutations=1, batch_size=8, prefix_cache=None,
              temperature=None, calibration=None, strict_calibration=False, strict=False, state_format=None)
```

| Argument | Meaning |
|---|---|
| `n_permutations` | Score every question under this many option orders (as written, reversed, then seeded shuffles) and average the logits. Reduces the causal backbones' option-order bias; costs one pass per order. |
| `batch_size` | Rows (question × option order) per forward pass. |
| `prefix_cache` | Causal backbones only. Run the shared image-and-state prefix once and only each row's suffix after it. `None` picks automatically, `True` forces it, `False` runs every row in full. The result matches the full path to float rounding. |
| `temperature` | A number for every type, or `{"choice": T, ...}`: replaces the checkpoint's temperatures for this call only. |
| `calibration` | A `laya.Calibration` from [`calibrate`](../how-to/calibrate.md), for this call only. |
| `strict_calibration` | Raise instead of warning when `calibration` was fitted for a different checkpoint. |
| `strict` | Raise `ValueError` instead of truncating a question that does not fit the token budgets (see below). |
| `state_format` | How the state's non-image keys become text for this call: `"json"`, `"prose"` or `"text"` (see [The state](#the-state)). `None` uses the checkpoint's own format. |

## The result

```python
{
  "model": "laya-vlm",
  "answers": {"<question id>": {...}, ...},
  "usage": {"input_tokens": n, "output_tokens": 0, "images": n},
  "provenance": {...},
}
```

Every answer has `type`, `confidence` and `action`; the rest depends on the type:

| Field | Types | Meaning |
|---|---|---|
| `choice` | `choice` | The most probable option name. |
| `score` | `score` | The expected level: the probability-weighted mean of the level indices. |
| `legend` | `score` | Level index → your level description. |
| `noul` | `noul` | P(true). |
| `probabilities` | `choice`, `score` | Option name (or level index) → probability, over the options listed. |
| `confidence` | all | For `choice` and `score`, one minus the normalised entropy of the probabilities (1 is certain, 0 is uniform); for `noul`, `max(P(true), P(false))`. |
| `action.act_probability` | all | The act head's probability of acting on the answer rather than escalating. The head is trained only with `train_act=True` (off by default), which the published checkpoint did not use, so its value there carries no information. |
| `truncated` | all, when a cut happened | See below. |

`provenance` records what produced the numbers: `prompt_format_version`, the `checkpoint` and `backbone` ids and
revisions, `dtype`, `device`, the `torch` and `transformers` versions, `readout` and `option_attention`, a sha256
over every scored row's input ids (`input_ids_sha256`), `n_rows`, `n_permutations`, and the temperature each
question was divided by (`temperatures`).

## Truncation

Inputs are cut to fit the checkpoint's token budgets: each option to 48 tokens (and shorter when many options must
share `head_max_len`, 256), the instructions to what the options leave, the state's text to what `max_len` leaves
after the images. An answer whose question was cut carries a `truncated` field, absent otherwise:

```python
{"options": [labels cut], "indistinguishable": [[label, label], ...], "instructions": bool,
 "instructions_tokens_dropped": n, "state_tokens_dropped": n}
```

`indistinguishable` lists options that are the same tokens once cut, which the model cannot tell apart.
`predict(..., strict=True)` raises `ValueError` instead, naming the question and what would be cut.
