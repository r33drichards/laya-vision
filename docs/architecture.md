# Architecture

Laya Vision is one decision model with three interchangeable backbones. Every branch takes a state (text, and for
the vision branches an image) plus a typed question, runs a single forward pass with no text generation, and
reads one logit per option from a marker token. The decision head, training objective, temperature calibration
and `predict(state, questions)` output schema are shared; what differs is the backbone, the token sequence, and
which token each option is read from.

| Branch | Backbone | Attention | Sequence builder | Option readout | Act-head pooling | Code |
|---|---|---|---|---|---|---|
| BERT (text only) | ModernBERT-large, 28 layers, d=1024 | bidirectional | `laya.common.build_sequence` | `[MASK]` opening each option | `[CLS]` | `laya/common.py`, `laya/agent.py` |
| SmolVLM | SmolVLM-256M-Instruct (SigLIP-B/16 + SmolLM2-135M, 30 layers, d=576) | causal | `laya.vlm.build_vlm_inputs` | `\n` ending each option line | last real token | `laya/vlm.py` |
| SmolVLM, block option attention | same, `option_attention="block"` | causal, except the option block | same | same | same | `laya.vlm.option_block_mask` |
| ModernVBERT | ModernVBERT (SigLIP2 + ModernBERT-150M, 22 layers, d=768) | bidirectional | `laya.vlm._mask_inputs` | `[MASK]` opening each option | `[CLS]` | `laya/vlm.py` (`readout="mask"`) |

All three vision paths use the same Idefics3 image pipeline: one 512-pixel tile per image, patch 16, pixel shuffle
4, so each image costs 64 language-model tokens.

## The whole model

```mermaid
flowchart TB
    subgraph IN["Inputs"]
        direction LR
        ST["state<br/>text / JSON / conversation<br/>+ image (vision branches)"]
        Q["question<br/>type: choice | score | noul<br/>instructions, criteria"]
    end

    subgraph SEQ["Sequence + markers (one row per question)"]
        direction LR
        B1["BERT<br/>[CLS] .. [SEP] [MASK] opt .. [SEP] state [SEP]"]
        B2["SmolVLM<br/>User: img state question<br/>Assistant: Options: - opt#92;n .."]
        B3["ModernVBERT<br/>[CLS]User: img question [SEP] [MASK] opt .. [SEP] state [SEP]"]
    end

    subgraph BB["Backbone (hidden states, no LM head)"]
        direction LR
        E1["ModernBERT-large<br/>bidirectional"]
        E2["SmolVLM-256M<br/>causal, or causal + bidirectional option block"]
        E3["ModernVBERT-250M<br/>bidirectional"]
    end

    subgraph HEAD["Shared decision head (DecisionModel / VLMDecisionModel)"]
        TE["+ type_emb[qtype]  (3 x d)"]
        HT["head: 2 x TransformerEncoderLayer<br/>bidirectional over the whole row, pre-norm, d//64 heads"]
        GA["gather hidden state at marker_pos<br/>(K markers, one per option)"]
        SC["scorer: LayerNorm - Linear(d,d) - GELU - Linear(d,1)"]
        LG["option logits [K], padding masked to -1e4"]
        FE["features: top-1 p, top-1 minus top-2,<br/>normalised entropy, K/255"]
        PO["pooled token<br/>[CLS] (BERT, ModernVBERT) or last real token (SmolVLM)"]
        AH["act_head: Linear(d+4,256) - GELU - Linear(256, n_act)"]
        AL["act logits: decide vs escalate"]
    end

    subgraph OUT["Inference-time output"]
        TS["temperature scaling<br/>per type, or per (type, option-count bucket)"]
        SM["softmax"]
        ANS["choice: argmax + probabilities<br/>score: expected level<br/>noul: p(true)<br/>confidence = 1 - H(p)/log K<br/>act_probability"]
    end

    ST --> SEQ
    Q --> SEQ
    B1 --> E1
    B2 --> E2
    B3 --> E3
    E1 --> TE
    E2 --> TE
    E3 --> TE
    TE --> HT --> GA --> SC --> LG
    LG -. "detached" .-> FE
    HT --> PO
    FE --> AH
    PO --> AH
    AH --> AL
    LG --> TS --> SM --> ANS
    AL --> ANS
```

### Training and calibration (all branches)

```mermaid
flowchart LR
    D["typed examples<br/>(A-OKVQA, ScienceQA, VQAv2 yes/no, The Cauldron, game frames)"]
    RO["random option_order per example<br/>(only matters for the causal SmolVLM readout)"]
    FW["forward pass<br/>option logits + act logits"]
    subgraph LOSS["vlm_loss"]
        PG["proper-scoring-rule policy gradient:<br/>group_size 4 noisy copies of the logits (sigma 0.3),<br/>reward = log score + w_sph x spherical score<br/>(minus ranked probability score for 'score' questions),<br/>group-normalised advantage"]
        CE["w_ce x soft cross-entropy<br/>(constant, or annealed to 0 over 30-80% of training)"]
        EU["- expected utility of the act head<br/>under the decide / escalate cost matrix"]
    end
    FR["freezing: head | last_n LM layers | full<br/>vision tower frozen unless asked"]
    TC["after training: per-type temperatures by LBFGS on held-out logits,<br/>written to the agent config"]

    D --> RO --> FW --> PG
    FW --> CE
    FW --> EU
    FR -.-> FW
    FW --> TC
```

## Branch 1: BERT (Laya's text model)

Upstream Laya. The encoder is ModernBERT-large; the whole sequence is bidirectional, so each option's `[MASK]`
sees the question, every other option and the state. Budgets: question + options within `head_max_len` (192),
the state fills the rest of `max_len` (512).

```mermaid
flowchart TB
    subgraph SEQ["build_sequence  (laya/common.py)"]
        direction LR
        C["[CLS]"] --- H["type question: instructions"] --- S1["[SEP]"]
        S1 --- M0["[MASK]"] --- O0["opt 0"] --- M1["[MASK]"] --- O1["opt 1"] --- MK["..."]
        MK --- S2["[SEP]"] --- STT["state (JSON / text)"] --- S3["[SEP]"]
    end

    ENC["ModernBERT-large encoder<br/>28 layers, d = 1024, 16 heads<br/>bidirectional; global attention every 3rd layer, 128-token sliding window otherwise<br/>sdpa attention"]

    subgraph HEAD["DecisionModel head"]
        TE["+ type_emb[qtype]"]
        HT["2 x TransformerEncoderLayer (d=1024, 16 heads, FFN 4096)<br/>src_key_padding_mask from attention_mask"]
        G["gather at markers = positions of each [MASK]"]
        SC["scorer -> K logits"]
        CLS["h[:, 0]  ([CLS])"]
        AH["act_head(concat(pooled, 4 features))"]
    end

    SEQ --> ENC --> TE --> HT
    HT --> G --> SC
    HT --> CLS --> AH
    SC -. "features" .-> AH
    SC --> OUT1["option logits -> temperature -> probabilities"]
    AH --> OUT2["act logits"]
```

What each `[MASK]` can attend to (bidirectional, so everything):

```mermaid
flowchart LR
    M0["[MASK] opt 0"] <--> M1["[MASK] opt 1"]
    M0 <--> QQ["question"]
    M1 <--> QQ
    M0 <--> SS["state"]
    M1 <--> SS
    M0 <--> CC["[CLS]"]
    M1 <--> CC
```

## Branch 2: SmolVLM (causal readout)

`VLMDecisionModel` with `readout="terminator"`. SmolVLM-256M is a decoder, so the options are moved to the end of
the sequence and each option is read at the `\n` that closes its line: the last token of the option span, and the
only token that has seen the image, the state, the question and the option's own text. Using a fixed terminator
gives every readout position the same token identity, the causal analogue of `[MASK]`. Budgets: `max_len` 1024,
`head_max_len` 256.

```mermaid
flowchart TB
    subgraph IMG["Image path (ImagePrep + Idefics3 vision)"]
        direction TB
        RAW["uint8 frame [3, H, W]"]
        RS["resize to 512 x 512<br/>gpu backend: two LANCZOS hops composed into one matmul pair, on the model's device<br/>processor backend: Hugging Face Idefics3ImageProcessor"]
        NM["normalise (v - 127.5) / 127.5"]
        VT["SigLIP-B/16 vision tower<br/>12 layers, d = 768, patch 16<br/>32 x 32 = 1024 patch tokens<br/>(frozen in training)"]
        PS["connector: pixel shuffle x4<br/>1024 -> 64 tokens, 768 x 16 = 12288 dims"]
        PR["projection -> d = 576"]
        RAW --> RS --> NM --> VT --> PS --> PR
    end

    subgraph SEQ["build_vlm_inputs  (laya/vlm.py)"]
        direction LR
        P["#lt;|im_start|#gt;User:"] --- I["#lt;image#gt; x 64<br/>(replaced by image features)"] --- ST["state text"]
        ST --- QT["#92;ntype question: instructions#lt;end_of_utterance#gt;#92;nAssistant: Options:#92;n"]
        QT --- O0["- opt 0"] --- N0["#92;n"] --- O1["- opt 1"] --- N1["#92;n"] --- MORE["..."]
    end

    PR -. "image_hidden_states replace the #lt;image#gt; run<br/>(encoded once, reused for every question)" .-> SEQ

    LM["SmolLM2-135M language model (Llama)<br/>30 layers, d = 576, 9 heads / 3 KV heads, RoPE<br/>causal mask, use_cache = False, no LM head"]

    subgraph HEAD["VLMDecisionModel head (shared)"]
        TE["+ type_emb[qtype]"]
        HT["2 x bidirectional TransformerEncoderLayer (d=576, 9 heads)"]
        G["gather at markers = position of each #92;n terminator"]
        SC["scorer -> K logits"]
        LAST["h at last real token (final #92;n)"]
        AH["act_head"]
    end

    SEQ --> LM --> TE --> HT
    HT --> G --> SC
    HT --> LAST --> AH
    SC -. "features" .-> AH
```

Causal attention means option 0's readout is made without seeing option 1. That is the option-order bias this
branch has to work around: random `option_order` during training, `n_permutations` averaging of logits at
inference, and the bidirectional head layers on top of the backbone.

```mermaid
flowchart LR
    subgraph ROW["what each readout sees under the causal mask"]
        direction LR
        IMG["image"] --> STA["state"] --> QU["question"] --> A0["- opt 0 #92;n"] --> A1["- opt 1 #92;n"] --> A2["- opt 2 #92;n"]
    end
    A0 -. "readout 0 sees: image, state, question, opt 0" .-> R0["logit 0"]
    A1 -. "readout 1 sees: ... + opt 0, opt 1" .-> R1["logit 1"]
    A2 -. "readout 2 sees: everything" .-> R2["logit 2"]
```

Inference with permutation averaging:

```mermaid
flowchart LR
    Q["question, K options"] --> P1["order 0,1,2"] & P2["order 2,1,0"] & P3["seeded shuffles ..."]
    P1 --> F1["forward"] --> L1["logits (row order)"]
    P2 --> F2["forward"] --> L2["logits (row order)"]
    P3 --> F3["forward"] --> L3["logits (row order)"]
    L1 & L2 & L3 --> UN["un-permute to label order, average"] --> OUT["final logits"]
```

## Branch 3: SmolVLM with block option attention

The same SmolVLM model, sequence and terminator readout, with `option_attention="block"` (formerly `"bidirectional"`, still accepted as a deprecated alias). Instead of the
backbone's default causal mask, `option_block_mask` builds an additive 4D mask that is causal everywhere except
inside the option span, where every option token attends to every other option token in both directions. Each
`\n` readout then sees all competitors, as ModernVBERT's `[MASK]` does, while the image, state and question stay
causal. The pretrained backbone never saw this pattern, so the setting is only meaningful when the language model
is unfrozen (`finetune_long --option-attention block`). It is recorded in `vlm_agent_config.json` and
applied at inference; the `"mask"` readout rejects it.

```mermaid
flowchart TB
    AM["attention_mask [B, L]<br/>option_span [B, 2] = (start, end) of the option block"]
    OBM["option_block_mask<br/>allowed[q, k] = (k <= q)  OR  (q in span AND k in span)<br/>AND key not padding<br/>-> additive mask [B, 1, L, L] in the backbone dtype"]
    LM["SmolLM2-135M with the custom 4D mask<br/>(sdpa attention)"]
    RD["same #92;n terminator readout, same head,<br/>act head still pools the last real token"]
    AM --> OBM --> LM --> RD
```

```mermaid
flowchart LR
    subgraph ROW["what each readout sees with option_attention = block"]
        direction LR
        IMG["image"] --> STA["state"] --> QU["question"] --> OPT["option block:  - opt 0 #92;n  - opt 1 #92;n  - opt 2 #92;n"]
    end
    OPT <-. "bidirectional inside the block" .-> OPT
    OPT -. "every readout sees: image, state, question and all options" .-> R["logits 0, 1, 2"]
```

Attention pattern, query rows against key columns (`x` = may attend):

```
keys:        img  state  q   opt0  opt1  opt2
img            x
state          x    x
question       x    x    x
opt0           x    x    x    x     x     x
opt1           x    x    x    x     x     x
opt2           x    x    x    x     x     x
```

### Inference: the shared-prefix cache (Branches 2 and 3)

Every row `VLMAgent.predict` scores for one state (each question, times each option order with
`n_permutations`) begins with the same tokens: the image run and the state text. Under the causal readout those
tokens' hidden states and keys/values do not depend on anything after them, so `predict` computes them once
(`VLMDecisionModel.encode_prefix`: one batch-1 pass through the full backbone, the only pass the image features
are merged into), replicates the key/value cache across the batch, and runs only each row's suffix through the
text model (`forward_prefixed`), with explicit positions `P..` and a 4D mask `[B, 1, S, P + S]` over the prefix
plus suffix keys (`option_block_mask(..., query_start=P)`: every suffix token sees the whole prefix, causal within
the suffix, the option span fully connected under `"block"`, padding keys masked). The cached prefix hidden
states are put back in front of the suffix ones before the head transformer, which is bidirectional over the
whole row, so the head, scorer and act head see exactly the sequence the full path builds.

The shared prefix is the longest common token prefix of all rows (`shared_prefix_len`), not a fixed boundary:
`build_vlm_inputs` cuts the state to the room each question's tail leaves, so with a long state and questions of
different lengths the rows carry different amounts of state and the cache stops where the shortest cut does.
Under causal attention it may also run past the state (one question under several option orders shares its whole
question text); under `"block"` it stops at the first option span, whose tokens attend forward. The image run
must lie inside it, otherwise `predict` takes the full path. The `"mask"` readout (ModernVBERT) always takes the
full path: a bidirectional prefix depends on what follows it.

`predict(..., prefix_cache=None)` decides per call: on a CPU the cache is used whenever there are two or more
rows (the backbone is compute-bound there), on CUDA only when there are more rows than `batch_size`, because a
backbone pass of this size costs about the same at any batch size up to a few thousand tokens and the cache adds
one. `True` forces the cache, `False` the full path. In fp32 the two paths agree to about 1e-6 in the option
logits (`tests/test_vlm.py::test_prefix_cache_matches_full_path`); in bf16 the probabilities differ by up to
0.004, bf16 rounding of a different summation order.

Latency, `modal run modal_app.py::bench_prefix_cache`: NVIDIA L4, bf16, the published checkpoint
(`cauldron-score-2ep-bidir-full/best`, `option_attention="block"`, 512-pixel processor preprocessing), one
640x480 image, 3 questions (a 4-option choice, a 4-level score and a noul, so `n_permutations` 1 / 4 / 8 gives 3 /
10 / 18 rows), `batch_size=8`, median of 20 calls after 3 warm-ups, end to end including preprocessing. "Short
state" is about 30 state tokens (about 140 tokens per row), "long state" about 740 (about 830 per row). Single
runs on a shared cloud GPU; absolute times move by +-30% between containers, the ratios within a run are stable.

| State | `n_permutations` | Rows | Full path | Cache forced | Speedup | Default (`None`) |
|---|---|---|---|---|---|---|
| image + short state | 1 | 3 | 80 ms | 122 ms | 0.66x | 80 ms (full) |
| image + short state | 4 | 10 | 131 ms | 126 ms | 1.04x | 126 ms (cache) |
| image + short state | 8 | 18 | 195 ms | 135 ms | 1.45x | 135 ms (cache) |
| image + long state | 1 | 3 | 87 ms | 130 ms | 0.67x | 88 ms (full) |
| image + long state | 4 | 10 | 207 ms | 161 ms | 1.28x | 158 ms (cache) |
| image + long state | 8 | 18 | 338 ms | 197 ms | 1.72x | 205 ms (cache) |

On the GPU the saving is in backbone passes (the full path runs `ceil(rows / 8)` of them, the cached path one
prefill plus `ceil(rows / 32)` suffix passes) and in tokens once rows are long; at three rows the extra prefill
pass makes the cache a loss, which is why the default skips it there. On a CPU the time follows the token count,
and the cache pays from two rows on: `tests/test_vlm.py::test_prefix_cache_speed` (SmolVLM-256M, fp32, a Modal
container with 4 vCPUs, one 96x96 image with a 5-token caption, the 3 test questions, median of 3) gave 1316 ms
full vs 1182 ms cached (1.11x) at `n_permutations=1` and 2234 ms vs 1566 ms (1.43x) at 4; the prefix there is
only 79 tokens, so a longer state gains more.

## Branch 4: ModernVBERT (bidirectional readout)

`VLMDecisionModel` with `readout="mask"`, picked automatically from the backbone's `model_type`. ModernVBERT is a
bidirectional masked-language-model encoder (ModernBERT-150M) with a SigLIP2 vision tower behind the same
Idefics3 processor SmolVLM uses, so Laya's original text sequence carries over with the image run where the
pretraining chat template puts it. Every `[MASK]` sees the whole sequence, so there is no option-order bias:
`n_permutations` and `option_attention` are accepted and do nothing.

```mermaid
flowchart TB
    subgraph IMG["Image path (identical settings to SmolVLM)"]
        direction TB
        RAW["uint8 frame"] --> RS["512 x 512, (v - 127.5) / 127.5"]
        RS --> VT["SigLIP2 vision tower<br/>12 layers, d = 768, patch 16, 1024 patches<br/>interpolate_pos_encoding=True (fixed 512 grid)<br/>frozen in training"]
        VT --> PS["connector: pixel shuffle x4 -> 64 tokens, projection to d = 768"]
    end

    subgraph SEQ["_mask_inputs  (laya/vlm.py)"]
        direction LR
        C["[CLS]User:"] --- I["#lt;image#gt; x 64"] --- H[" type question: instructions"] --- S1["[SEP]"]
        S1 --- M0["[MASK]"] --- O0["opt 0"] --- M1["[MASK]"] --- O1["opt 1"] --- MK["..."]
        MK --- S2["[SEP]"] --- STT["state"] --- S3["[SEP]"]
    end

    PS -. "image_hidden_states replace the #lt;image#gt; run" .-> SEQ

    ENC["ModernBERT-150M text encoder<br/>22 layers, d = 768, 12 heads<br/>bidirectional; global attention every 3rd layer, 128-token sliding window otherwise<br/>no LM head"]

    subgraph HEAD["VLMDecisionModel head (shared)"]
        TE["+ type_emb[qtype]"]
        HT["2 x TransformerEncoderLayer (d=768, 12 heads)"]
        G["gather at markers = position of each [MASK]"]
        SC["scorer -> K logits"]
        CLS["h[:, 0]  ([CLS])"]
        AH["act_head"]
    end

    SEQ --> ENC --> TE --> HT
    HT --> G --> SC
    HT --> CLS --> AH
    SC -. "features" .-> AH
```

```mermaid
flowchart LR
    subgraph ROW["what each [MASK] sees (bidirectional everywhere)"]
        direction LR
        CC["[CLS]"] --- IMG["image"] --- QU["question"] --- M0["[MASK] opt 0"] --- M1["[MASK] opt 1"] --- STA["state"]
    end
    M0 <-.-> M1
    M0 <-.-> IMG
    M1 <-.-> STA
    M0 -.-> R0["logit 0"]
    M1 -.-> R1["logit 1"]
```

## Side by side: where an option is read

```mermaid
flowchart LR
    subgraph BERT["BERT / ModernVBERT"]
        direction LR
        b1["[SEP]"] --- b2["[MASK]"] --- b3["electronics"] --- b4["[MASK]"] --- b5["clothing"] --- b6["[SEP]"]
    end
    subgraph SMOL["SmolVLM (both attention modes)"]
        direction LR
        s1["Options:#92;n"] --- s2["- electronics"] --- s3["#92;n"] --- s4["- clothing"] --- s5["#92;n"]
    end
    b2 -.-> L1["logit: electronics"]
    b4 -.-> L2["logit: clothing"]
    s3 -.-> L3["logit: electronics"]
    s5 -.-> L4["logit: clothing"]
```

| | BERT | SmolVLM causal | SmolVLM block option attention | ModernVBERT |
|---|---|---|---|---|
| Marker token | `[MASK]` before the option | `\n` after the option | `\n` after the option | `[MASK]` before the option |
| Marker sees other options | all | earlier ones only | all | all |
| Marker sees the state | yes | yes (state precedes options) | yes | yes |
| Order bias mitigation | none needed | random order in training, `n_permutations` at inference | mask, plus the above | none needed |
| Act-head pooling | `[CLS]` | last real token | last real token | `[CLS]` |
| Backbone attention | bidirectional | causal | causal outside the option block | bidirectional |
| Needs backbone fine-tuning to use | no | no | yes | no |

## Checkpoint layout

```mermaid
flowchart LR
    subgraph TXT["BERT agent (rl_agent_config.json)"]
        t1["model.safetensors: encoder.* type_emb.* head.* scorer.* act_head.*"]
        t2["tokenizer/, encoder/config.json"]
        t3["temperature, temperature_by_options, max_len 512, head_max_len 192"]
    end
    subgraph VLM["VLM agent (vlm_agent_config.json)"]
        v1["model.safetensors (full) or head.safetensors (frozen backbone)"]
        v2["processor/, backbone/config.json"]
        v3["backbone id, readout: terminator | mask,<br/>option_attention: causal | block,<br/>image_size, preprocess, image_interpolation,<br/>temperature, temperature_by_options, max_len 1024, head_max_len 256"]
    end
    LOAD["laya.load / laya.load_vlm"] --> TXT
    LOAD --> VLM
    VLM --> RO["readout picks the sequence builder;<br/>ImagePrep picks the pixel path"]
```
