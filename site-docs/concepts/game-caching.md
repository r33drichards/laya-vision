# Caching the fixed question in game play

In game play (`examples/atari_live.py`, `examples/vizdoom_live.py`, `laya.atari_train.play`) every step asks the
same `choice` question: the same instructions and the same buttons. Only the screen changes. Today each frame
re-runs the whole sequence. This note asks whether the fixed question + options part could be cached across
frames instead of recomputed, measures what that could save at most, and compares it with cheaper options.

**Short answer.** Caching the question's keys/values is not worth building. With the current layout it is
impossible. With a retrained question-first layout, the most it could save is 5-18% of the compute, and at
batch 1 on a GPU (the live viewers) it saves nothing, because that forward is limited by kernel launches, not by
token count. The fixed question does make one thing possible: every tensor shape is the same every frame, so the
whole decision can be captured once as a **CUDA graph** and replayed. That is implemented behind `--cuda-graph`
(`laya/static_step.py`). It is **3.4-5.0x faster per decision** at batch 1 in bf16 on an L4 (~11 ms against 37-55 ms)
and gives the same answer. On CPU the vision tower is three quarters of the time, and no caching of the question changes that.

All numbers come from `examples/bench_game_step.py` (the latency breakdown) and `modal_game_cache.py` (the same
script on Modal, plus duplicate-frame rates and the `StaticStep` checks). The raw output is in
`docs/game-caching-measurements.json`. Hardware: **NVIDIA L4** (Modal, torch 2.14 + CUDA 13, 4 torch CPU
threads). The CPU numbers are from **the same Modal container's CPU, 4 threads, fp32**. The local sandbox CPU was
shared with other jobs (load average 13-34 on 4 cores), and its timings were unusable. The model is a fresh
agent on the SmolVLM-256M backbone at 512 px on the device-side preprocessing path. Timing does not depend on the
weights, and the architecture (2 head layers) matches every released SmolVLM checkpoint. Frames are synthetic at
the real sizes: Atari 210x160, ViZDoom 320x240. Each number is the median of 30 calls (5 on CPU) after warm-up.

## Where the tokens are

The sequence today (the SmolVLM "terminator" layout, `laya/vlm.py`):

    <|im_start|>User:  <fake><global-img><image> x64<fake>  \n choice question: <instructions><end_of_utterance>
    \nAssistant: Options:\n  - NOOP: do nothing\n  - FIRE: press fire ...\n  ...

| workload | total | text before image | image run (64 image tokens + 3 tags) | question | options | question + options |
|---|---|---|---|---|---|---|
| Breakout (4 actions) | 152 | 3 | 67 | 46 | 36 | **82 (54%)** |
| Pong (6 actions) | 176 | 3 | 67 | 47 | 59 | **106 (60%)** |
| Boxing (18 actions, full ALE set) | 303 | 3 | 67 | 48 | 185 | **233 (77%)** |
| ViZDoom `basic` (3 buttons) | 152 | 3 | 67 | 56 | 26 | **82 (54%)** |
| ViZDoom `deadly_corridor` (7 buttons) | 181 | 3 | 67 | 45 | 66 | **111 (61%)** |
| Pong, 2 frames (`--frames 2`) | 243 | 3 | 134 | 47 | 59 | **106 (44%)** |

So the fixed part is most of the language model's tokens. That is the case for caching it. The case against:
the language model is a small part of the work. SmolVLM-256M's text tower is about 106M non-embedding
parameters (0.21 GFLOP per token), while its vision tower is a SigLIP-B/16 at 512 px (1024 patches, 12 layers,
about 213 GFLOP per image). At 512 px one image costs as much as ~1000 text tokens.

## Where the time goes today (batch 1, one decision)

L4, **bf16**, causal option attention, ms:

| workload | build inputs (CPU) | resize | vision tower | language model | heads | `predict` end to end | LM: question+options on the image's KV cache |
|---|---|---|---|---|---|---|---|
| Breakout | 0.63 | 0.41 | 6.6 | 23.7 | 1.4 | 36.2 | 28.7 |
| Pong | 0.70 | 0.42 | 6.5 | 24.5 | 1.4 | 36.7 | 29.0 |
| Boxing | 1.66 | 0.42 | 6.6 | 24.7 | 1.5 | 38.6 | 29.9 |
| Doom `basic` | 0.68 | 0.45 | 6.9 | 25.3 | 1.5 | 38.8 | 28.5 |
| Doom `deadly_corridor` | 0.77 | 0.42 | 6.6 | 25.5 | 1.5 | 37.2 | 28.9 |
| Pong, 2 frames | 0.82 | 0.42 | 12.6 | 24.4 | 1.5 | 40.3 | 29.4 |

The language model takes **24-25 ms whether it runs 152 tokens or 303**. Running only the 82-233 question+option
tokens on top of a cached image prefix takes *longer* (28.5-29.9 ms) than the full sequence. At batch 1 on a GPU,
the 30 decoder layers are bound by kernel launches and Python overhead, not arithmetic, so token count barely
matters. The same holds in fp32, where the language model is 23.2-23.8 ms, while the vision tower becomes
compute-bound at 30.7 ms (57 ms for two frames) and `predict` takes 52-54 ms. The `block` option attention of the
released checkpoints costs about 7% more in the language model (26.2 against 24.5 ms: it passes a dense 4D mask).

CPU (the L4 container's CPU, 4 threads, fp32), ms:

| workload | vision tower | language model | heads | `predict` | LM: image prefix only | LM: question+options on cache |
|---|---|---|---|---|---|---|
| Pong | 979 | 286 | 21 | 1332 | 161 | 201 |
| Doom `basic` | 967 | 236 | 18 | 1263 | 148 | 166 |
| Pong, 2 frames | 1915 | 401 | 27 | 2545 | 235 | 214 |

On CPU the work is compute-bound, and the **vision tower is 73-77% of every decision**. The language model's
share of the question+options is at most `LM full - LM image prefix`, or 88-166 ms: **6.5-9.4% of `predict`**. The
split was checked against the unsplit forward (hidden states agree to 1e-4 in fp32), so these are real prefix/tail
costs and not artifacts of the cache.

**Upper bound for any question/option caching**, per decision:

* GPU, batch 1 (the live viewers): **about 0**. The measured question-first simulation below is *slower*.
* CPU: **6.5-9.4%** measured.
* GPU with large batches, where the work becomes compute-bound: the FLOP share of the question+options is 7%
  (Breakout, Doom `basic`), 9% (Pong), 18% (Boxing's 18 options), 4.7% (Pong two-frame).

## The options

### (a) A question-first layout (needs training): not worth it

A layout where the fixed part comes first, so its keys/values never depend on the frame, and where each option is
still read out at a token that sees the image:

    <|im_start|>User: choice question: <instructions>\nOptions:\n(A) NOOP: do nothing\n(B) FIRE: ...\n ...   <- cached once
    <fake><global-img><image> x64<fake><end_of_utterance>\nAssistant:                                    <- per frame
    (A)(B)(C)...                                                                                          <- per frame, readouts

* **Readout.** Option *i* is read at its label token (`(A)`, `(B)`, ...) *after* the image. That token attends
  to the whole cached question, to the image, and to the matching label in the cached option list. The label
  identity lets the model pull in that option's text through attention, the way the current terminator relies on
  sitting at the end of its own option's text. The labels must be distinct tokens (18 for Boxing). A fixed `\n`
  terminator, as used today, would not work here, because it could not tell the readouts apart.
* **Attention.** The readout tokens are all per frame, so they can attend to each other both ways (a small 4D mask
  over the last K positions with the cache prefix fully visible). Then no readout is made without seeing its
  competitors, which removes the option-order bias at the readout. The option text in the cache stays causal.
* **Heads.** The head transformer (2 bidirectional layers over the whole sequence today) would run over the
  per-frame tokens only (image run + readouts). If it also covered the question positions, those positions would
  mix with the image and have to be recomputed every frame.
* **Per frame** that is 70-85 tokens instead of 152-303 (Pong: 73 instead of 176).

What it buys was measured by simulation: the question+options run once into a KV cache, then each frame runs the
image run + K readout tokens on that cache, and the heads run over the per-frame tokens. Per decision, ms:

| | L4 bf16, current | L4 bf16, question-first | CPU fp32, current | CPU fp32, question-first |
|---|---|---|---|---|
| Pong | 32.9 | 37.5 | 1288 | 1154 (-10%) |
| Doom `basic` | 34.0 | 38.2 | 1227 | 1160 (-5%) |
| Pong, 2 frames | 38.9 | 43.9 | 2350 | 2189 (-7%) |

(Sums of the parts: resize + vision + language model + heads, so no input building.) On a GPU it is slower,
because the cached forward pays the cache's overhead and saves no launches. On CPU it saves 5-10%. The cost is a
new layout that invalidates every checkpoint, a new training run of the full language model (the cauldron-score
recipe, then the Atari runs), a new evaluation, and a readout mechanism (a label token recovering its option
through attention) that is untested and weaker than today's. **Recommendation: do not build it.** No training
command is given because none should run. It would be a `layout="question_first"` option in
`build_vlm_inputs` / `VLMDecisionModel` plus the per-frame cache path, and a full-backbone run like the
`atari-8g-2f-512gpu` one.

### (b) ModernVBERT's bidirectional layout: nothing can be cached across images

In ModernVBERT (`readout="mask"`) every token attends to every other token in every layer (ModernBERT alternates
local and global layers, and the global ones mix the whole sequence). The question tokens' hidden states from
layer 1 on are therefore functions of the image tokens. A layer's keys and values are projections of those hidden
states, so every key and value except the first layer's inputs changes with each frame. Reordering does not help:
position in the sequence does not decide what a token sees, the mask does, and it is full. The only
frame-independent quantity is the input token embeddings, a gather that costs 0.045 ms (measured on the L4).
Caching the question here would mean changing the attention pattern itself, and so retraining, which is option (a)
with a model that was pretrained bidirectionally.

### (c) Cheaper things that need no retraining

| idea | measured | verdict |
|---|---|---|
| Cache the text token embeddings | the embedding lookup is **0.045 ms** of a 36 ms decision | nothing to gain |
| Cache the input building (tokenizer) | 0.6-1.7 ms of CPU per decision | small; `StaticStep` builds once as a side effect |
| **Capture the whole decision as a CUDA graph** (the fixed question fixes every shape) | **36-55 -> 11.2 ms** at batch 1, bf16, L4 | **implemented**, `--cuda-graph`, below |
| bf16 instead of fp32 on a GPU | vision tower 30.7 -> 6.6 ms; `predict` 52 -> 36 ms | use it: `--dtype bf16` added to both viewers |
| Batch several env instances per forward | eager bf16: 36.8 / 13.6 / 9.5 ms per frame at 1 / 4 / 16 | already how `play` evaluates; the live viewers run one env |
| Reuse the vision features of an identical frame | exact repeats of the previous frame, random policy: Breakout 7.2%, SpaceInvaders 8.9%, Pong 1.3%, Boxing 1.4%, Freeway 0%; ViZDoom 0.8-1.6% | not worth a flag (see below) |
| `FrameFeatureCache` (`keep=2`) in two-frame play | already on by default on the device-side path: 38.4 -> 53.1 decisions/s (docs/game-training.md) | keep |
| Lower `image_size` (512 -> 256: 64 -> 16 image tokens, 4x fewer patches) | the only big CPU lever | quality collapses (median normalised score 0.255 -> 0.030, docs/game-training.md); not recommended |

**Identical and near-identical frames.** A decision on a frame identical to the last one could return the last
answer exactly. But with a random policy, frames repeat 0-9% of the time on Atari (v5, frameskip 4) and 1-2% on
ViZDoom (4 tics per action). A consecutive pair changes a median of 0.1-1.7% of the pixels on Atari, but 57-73% on
ViZDoom, where the whole view moves. Skipping only exact repeats would save at most the repeat rate. Reusing
features for *nearly* identical frames would change answers: a Pong ball is a few pixels, and it is exactly the
part that matters. So neither was implemented. For two-frame play, `FrameFeatureCache` already removes the one
repeat that is guaranteed (this step's current frame is next step's previous frame). Its `keep` bounds the cache
to the frames of the last `keep` calls; `keep=2` covers that repeat, and a larger `keep` would only catch the
rare exact repeats above.

## Implemented: one CUDA graph per decision (`--cuda-graph`)

`laya.static_step.StaticStep(agent, question, frames=1, batch=1)` runs the decision -- resize, vision tower,
connector, merge, language model and heads -- as one captured CUDA graph and replays it for each frame. It needs
the question to be fixed, which is exactly what game play gives. The Hugging Face forward has two
data-dependent steps a graph cannot hold (the vision tower's position ids come from a bucketize plus a
boolean-indexed write, and the image features are merged with `masked_scatter`). Both are constant for a fixed
question and an unpadded tile, so they are computed once: the position ids are recorded from one eager call, and
the merge becomes an index write at fixed positions. Everything else runs the model's own modules. The heads
repeat `VLMDecisionModel.forward`'s tail, pinned to it by the tests.

* `examples/atari_live.py --cuda-graph` and `examples/vizdoom_live.py --cuda-graph` (with `--dtype bf16` for
  the full effect). Off CUDA the same step runs eagerly with no speedup.
* `laya.atari_train.model_policy(..., cuda_graph=True)` keeps one step per number of episodes still running. It
  replaces the feature cache.
* Scope: SmolVLM-family checkpoints (`readout="terminator"`, causal or `block` option attention), no image
  splitting, either preprocessing path, one option order (as `action_probs`).

L4, `action_probs` (eager) against `StaticStep`, `block` option attention (as in the released checkpoints),
ms per call, Pong and Doom `basic` averaged. The runs were in two containers (`modal_game_cache.py::verify` twice),
and the eager path ran at different speeds in them, so both are shown:

| dtype | frames | batch | eager, run 1 / run 2 | `StaticStep`, run 1 / run 2 | speedup | `StaticStep` ms/frame | max prob diff |
|---|---|---|---|---|---|---|---|
| bf16 | 1 | 1 | 55.0 / 37.7 | **11.3 / 11.0** | **4.9x / 3.4x** | 11.0-11.3 | 0 |
| bf16 | 2 | 1 | 56.0 / 39.9 | 18.6 / 18.0 | 3.0x / 2.2x | 18.0-18.6 | 0 |
| bf16 | 1 | 4 | 57.3 / 41.5 | 37.3 / 36.3 | 1.5x / 1.1x | 9.1-9.3 | 0 |
| bf16 | 1 | 16 | 159.3 / 150.7 | 147.4 / 141.8 | 1.1x | 8.9-9.2 | 0 |
| fp32 | 1 | 1 | 51.8 / 44.1 | 41.7 / 41.2 | 1.2x / 1.1x | 41-42 | 0 |
| fp32 | 2 | 1 | 75.9 / 74.8 | 72.0 / 72.0 | 1.05x | 71-73 | 0 |

The captured step costs the same in both containers, **~11 ms per bf16 decision**. The eager path (36-56 ms) is
bound by launches, so it runs at the host CPU's speed, and that sets the speedup: **3.4-5.0x** at batch 1 in bf16.
`VLMAgent.predict` for the same decision took 39-53 ms, against 11.0-11.6 ms for `StaticStep.answer`. The graph
removes the launch overhead, so it gains the most exactly where question caching gains nothing (batch 1, bf16).
The gain fades as the batch grows or as fp32 makes the vision tower compute-bound. Against the viewers' current
default (fp32 `predict`, 52-67 ms on the L4), `--dtype bf16 --cuda-graph` is 4.7-6x.

**Parity.** `tests/test_vlm.py::test_static_step_matches_action_probs` checks probabilities and P(act) against
`action_probs` on new frames: 1 and 2 frames, batch 1 and 2, causal and `block`, both preprocessing paths. It also
checks `answer` against `predict`. `test_static_step_graph_replay_matches_eager` checks that the captured replay
gives the eager step's answer on each new frame. `test_model_policy_cuda_graph_flag_picks_the_same_actions` checks
that the policy flag picks the same actions as the default while the batch shrinks. All 6 passed on the L4, where
graphs are captured. On CPU the steps run eagerly. The replay is bit-identical in fp32 and with `block` attention
in bf16 (0 difference in the tables above). With causal attention in bf16 the replay moved a probability by up to
0.001, which is bf16 kernel-selection noise: the same size as the batch-1 against batch-2 difference the feature
cache already documents.

## Recommendation

1. Do not build a question-first layout (a). Its ceiling is 5-10% on CPU and 5-18% of the FLOPs in a large
   batch, it is nothing at batch 1 on a GPU, and it costs a new layout, retraining and a weaker readout.
2. On a GPU, play with `--dtype bf16 --cuda-graph`: ~11 ms per decision at batch 1, 4.7-6x fewer than today's fp32
   `predict` default. For evaluation, keep batching environments (`play`). `cuda_graph=True` adds 1.1-1.5x at 4
   episodes and nothing much at 16.
3. On CPU nothing about the question is worth caching: the vision tower is 73-77% of the time. The levers are
   the vision side: the existing two-frame feature cache, or a smaller or faster vision input, which needs
   retraining and has so far cost quality.

Reproduce:

    modal run modal_game_cache.py::main                  # breakdown (bf16, fp32, block, CPU), batching, duplicates
    modal run modal_game_cache.py::main --mode graphs    # CUDA-graph comparison + ViZDoom duplicate rates
    modal run modal_game_cache.py::verify                # StaticStep tests on CUDA + the table above
    python examples/bench_game_step.py --device cpu      # the breakdown locally
