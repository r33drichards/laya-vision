# How it works

A `score` question with 4 levels, a `choice` with 5 options and a `noul` are each rendered as a question followed by
their options, one per line, after the image and the state text. The backbone encodes the whole sequence once per
image, and a small head reads one logit per option from the hidden state at each option's marker. Softmax over the
options, divided by a per-type temperature fitted after training, is the answer. Nothing is generated.

## Causal and bidirectional backbones

- **Causal backbones (SmolVLM)** can only read an option after everything before it, so the options go last and
  each is read at its line terminator. That leaves an option-order bias of about a point of accuracy, which random
  orders in training and permutation averaging at inference (`predict(..., n_permutations=K)`) reduce. The
  recommended checkpoint adds a 4D attention mask that lets the option block attend to itself in both directions
  (`option_attention="block"`; the older spelling `"bidirectional"` still loads but is deprecated), which is worth
  about a point on the reasoning-heavy sets.
- **Bidirectional backbones (ModernVBERT)** use Laya's original format unchanged: a `[MASK]` in front of each
  option, read from a marker that sees the whole sequence. No order bias to fix, and the fastest at inference; it
  trails SmolVLM by about 7 points on the Cauldron holdouts, mostly on reasoning sets like RAVEN and TQA
  ([results](../reference/results/modernvbert-cauldron.md)).

## Training

Training is Laya's RLCD objective. Gaussian noise is added to the option logits, several noisy copies are scored
with a strictly proper scoring rule (log plus spherical, plus a ranked probability score for `score` questions), and
the group-normalised score is the policy-gradient advantage, with a soft cross-entropy term added. The vision tower
stays frozen.

## Budgets

Inputs are cut to fit the checkpoint's token budgets: each option to 48 tokens (and shorter when many options must
share `head_max_len`, 256), the instructions to what the options leave, the state's text to what `max_len` leaves
after the images. An answer whose question was cut says so; see
[predict()](../reference/predict.md#truncation).

## Where to go next

Diagrams of every variant and the shared head: [Architecture](architecture.md). The model code is
[`laya/vlm.py`](https://github.com/r33drichards/laya-vision/blob/main/laya/vlm.py) and the training loop
[`laya/vlm_train.py`](https://github.com/r33drichards/laya-vision/blob/main/laya/vlm_train.py).
