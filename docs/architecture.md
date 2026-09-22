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
| SmolVLM, bidirectional options | same, `option_attention="bidirectional"` | causal, except the option block | same | same | same | `laya.vlm.option_block_mask` |
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

## Branch 3: SmolVLM with bidirectional options

The same SmolVLM model, sequence and terminator readout, with `option_attention="bidirectional"`. Instead of the
backbone's default causal mask, `option_block_mask` builds an additive 4D mask that is causal everywhere except
inside the option span, where every option token attends to every other option token in both directions. Each
`\n` readout then sees all competitors, as ModernVBERT's `[MASK]` does, while the image, state and question stay
causal. The pretrained backbone never saw this pattern, so the setting is only meaningful when the language model
is unfrozen (`finetune_long --option-attention bidirectional`). It is recorded in `vlm_agent_config.json` and
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
    subgraph ROW["what each readout sees with option_attention = bidirectional"]
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

| | BERT | SmolVLM causal | SmolVLM bidirectional options | ModernVBERT |
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
        v3["backbone id, readout: terminator | mask,<br/>option_attention: causal | bidirectional,<br/>image_size, preprocess, image_interpolation,<br/>temperature, temperature_by_options, max_len 1024, head_max_len 256"]
    end
    LOAD["laya.load / laya.load_vlm"] --> TXT
    LOAD --> VLM
    VLM --> RO["readout picks the sequence builder;<br/>ImagePrep picks the pixel path"]
```
