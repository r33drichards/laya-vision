"""Where one game decision's time goes, and what caching the fixed question + options could save.

Every game step asks the same ``choice`` question (same instructions, same buttons) about a new screen. This
times one decision of the current SmolVLM "terminator" layout piece by piece, for Atari (210x160) and ViZDoom
(240x320) frames and typical button sets, and simulates the two cached layouts that would reuse the question:

    current      <|im_start|>User:<image run><question><options, each ending in \\n>        (nothing reusable)
    split        the same sequence, run as prefix (up to the image) + tail (question + options) on the prefix's
                 KV cache: the tail's time is exactly what caching the question/options could ever save
    qfirst       question-first (docs/game-caching.md, option (a)): <|im_start|>User:<question><options> is run
                 once and kept as a KV cache; each frame runs only <image run> + one readout token per option

    python examples/bench_game_step.py                       # CPU, fp32, a fresh (untrained) agent
    python examples/bench_game_step.py --device cuda --dtype bf16
    modal run modal_game_cache.py::main                      # the same on an L4

Timing does not depend on the weights, so a fresh agent on the SmolVLM-256M backbone stands in for a trained
checkpoint (same architecture: 2 head layers). The frames are synthetic: compute does not depend on pixel values.
Prints one JSON object at the end.
"""
import argparse
import json
import os
import platform
import statistics
import time

import numpy as np
import torch

from laya.atari_train import action_probs
from laya.common import QTYPES
from laya.games import atari_question, doom_question
from laya.vlm import OPTION_END, PREFIX_TEXT, VLMAgent, build_vlm_inputs, collate_vlm, vlm_prefix

ATARI_FRAME, DOOM_FRAME = (210, 160, 3), (240, 320, 3)
WORKLOADS = (  # (label, frame shape, question); minimal ALE action sets and the scenarios' own buttons
    ("Breakout (4 actions)", ATARI_FRAME, atari_question("Breakout", ["NOOP", "FIRE", "RIGHT", "LEFT"])),
    ("Pong (6 actions)", ATARI_FRAME,
     atari_question("Pong", ["NOOP", "FIRE", "RIGHT", "LEFT", "RIGHTFIRE", "LEFTFIRE"])),
    ("Boxing (18 actions)", ATARI_FRAME, atari_question("Boxing", [
        "NOOP", "FIRE", "UP", "RIGHT", "LEFT", "DOWN", "UPRIGHT", "UPLEFT", "DOWNRIGHT", "DOWNLEFT", "UPFIRE",
        "RIGHTFIRE", "LEFTFIRE", "DOWNFIRE", "UPRIGHTFIRE", "UPLEFTFIRE", "DOWNRIGHTFIRE", "DOWNLEFTFIRE"])),
    ("Doom basic (3 buttons)", DOOM_FRAME, doom_question("basic", ["MOVE_LEFT", "MOVE_RIGHT", "ATTACK"])),
    ("Doom deadly_corridor (7 buttons)", DOOM_FRAME, doom_question("deadly_corridor", [
        "MOVE_LEFT", "MOVE_RIGHT", "ATTACK", "MOVE_FORWARD", "MOVE_BACKWARD", "TURN_LEFT", "TURN_RIGHT"])),
)


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def timed(fn, device, reps, warmup=3):
    """Median ms of ``reps`` calls after ``warmup``."""
    for _ in range(warmup):
        fn()
    sync(device)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        sync(device)
        ts.append((time.perf_counter() - t0) * 1000)
    return round(statistics.median(ts), 3)


def head_part(model, h, attention_mask, marker_pos, marker_mask, qtype):
    """``VLMDecisionModel.forward`` after the backbone: type embedding, head transformer, scorer, act head."""
    h = h.float() + model.type_emb(qtype)[:, None, :]
    if model.head is not None:
        pad = ~attention_mask.bool()
        for layer in model.head.layers:
            h = layer(h, src_key_padding_mask=pad)
    m = torch.gather(h, 1, marker_pos[:, :, None].expand(-1, -1, h.size(-1)))
    logits = model.scorer(m).squeeze(-1)
    last = attention_mask.sum(-1) - 1
    feats = torch.zeros(h.size(0), 4, device=h.device)
    return logits, model.act_head(torch.cat([h[torch.arange(h.size(0), device=h.device), last], feats], -1))


def token_counts(agent, it, frames):
    tok = agent.processor.tokenizer
    n_text_prefix = len(tok(PREFIX_TEXT, add_special_tokens=False)["input_ids"])
    n_prefix = len(vlm_prefix(agent.processor, [np.zeros(ATARI_FRAME, np.uint8)] * frames, agent.prep)["ids"])
    s, e = it["option_span"]
    return {"total": len(it["ids"]), "text_before_image": n_text_prefix, "image_run": n_prefix - n_text_prefix,
            "image_tokens": agent.prep.image_seq_len * frames, "question": s - n_prefix, "options": e - s,
            "question_plus_options": e - n_prefix, "n_options": len(it["markers"])}


@torch.no_grad()
def bench_workload(agent, label, shape, question, reps, frames=1):
    dev, model, enc = agent.device, agent.model, agent.model.encoder
    rng = np.random.default_rng(0)
    imgs = [rng.integers(0, 256, shape, dtype=np.uint8) for _ in range(frames)]
    state = {"image": imgs[0]} if frames == 1 else {"images": imgs}
    qs = {"action": question["action"]}
    q = VLMAgent._to_internal(question["action"])
    it = build_vlm_inputs(agent.processor, state, q)
    it["qtype"] = QTYPES["choice"]
    row = {"workload": label, "frame": "%dx%d" % shape[:2], "frames": frames, "tokens": token_counts(agent, it, frames)}
    b = collate_vlm([it], agent.processor.tokenizer.pad_token_id, with_pixels=False)
    b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
    raw = torch.stack([torch.from_numpy(i) for i in imgs])
    pv, pam = agent.prep.pixel_values(raw, device=dev, dtype=enc.dtype)
    feats = model._image_features(pv[None], pam[None]).to(enc.get_input_embeddings().weight.dtype)
    lm_kw = dict(input_ids=b["input_ids"], attention_mask=b["attention_mask"], image_hidden_states=feats)
    attn = b["attention_mask"]
    if model.option_attention == "block":
        from laya.vlm import option_block_mask
        attn = option_block_mask(b["attention_mask"], b["option_span"], enc.dtype)
    lm_kw["attention_mask"] = attn

    t = row["ms"] = {}
    t["build_inputs_cpu"] = timed(lambda: collate_vlm([dict(build_vlm_inputs(agent.processor, state, q), qtype=2)],
                                                      agent.processor.tokenizer.pad_token_id, with_pixels=False),
                                  dev, reps)
    t["resize_on_device"] = timed(lambda: agent.prep.pixel_values(raw, device=dev, dtype=enc.dtype), dev, reps)
    t["vision_tower"] = timed(lambda: model._image_features(pv[None], pam[None]), dev, reps)
    t["lm_full"] = timed(lambda: enc(use_cache=False, **lm_kw), dev, reps)
    h = enc(use_cache=False, **lm_kw).last_hidden_state
    t["head"] = timed(lambda: head_part(model, h, b["attention_mask"], b["marker_pos"], b["marker_mask"], b["qtype"]),
                      dev, reps)
    t["predict_end_to_end"] = timed(lambda: agent.predict(state, qs), dev, reps)
    t["text_embedding_lookup"] = timed(lambda: enc.get_input_embeddings()(b["input_ids"]), dev, reps)

    # split: the image prefix with a KV cache, then the question + options on top of it (causal mask only; the
    # block mask changes which option tokens see each other, not how much there is to compute)
    P = row["tokens"]["text_before_image"] + row["tokens"]["image_run"]
    ids = b["input_ids"]
    pre = enc(input_ids=ids[:, :P], image_hidden_states=feats, use_cache=True)
    past = pre.past_key_values
    ones = torch.ones_like(ids)

    def tail():
        out = enc(input_ids=ids[:, P:], attention_mask=ones, past_key_values=past, use_cache=True)
        past.crop(P)
        return out

    t["lm_image_prefix"] = timed(lambda: enc(input_ids=ids[:, :P], image_hidden_states=feats, use_cache=True),
                                 dev, reps)
    t["lm_question_options_on_cache"] = timed(tail, dev, reps)
    full_causal = enc(input_ids=ids, attention_mask=ones, image_hidden_states=feats, use_cache=False).last_hidden_state
    split = torch.cat([pre.last_hidden_state, tail().last_hidden_state], 1)
    row["split_parity_max_abs"] = float((full_causal.float() - split.float()).abs().max())

    # qfirst: question + options once as a KV cache, then per frame the image run + one readout token per option
    tok = agent.processor.tokenizer
    end_id = tok(OPTION_END, add_special_tokens=False)["input_ids"][0]
    qo = torch.cat([ids[:, : row["tokens"]["text_before_image"]], ids[:, P:]], 1)
    per_frame = torch.cat([ids[:, row["tokens"]["text_before_image"]:P],
                           torch.full((1, len(it["markers"])), end_id, device=dev)], 1)
    Q = qo.shape[1]
    qpast = enc(input_ids=qo, use_cache=True).past_key_values
    qones = torch.ones((1, Q + per_frame.shape[1]), dtype=torch.long, device=dev)

    def qfirst():
        out = enc(input_ids=per_frame, attention_mask=qones, image_hidden_states=feats, past_key_values=qpast,
                  use_cache=True)
        qpast.crop(Q)
        return out

    t["qfirst_lm_per_frame"] = timed(qfirst, dev, reps)
    t["qfirst_lm_question_once"] = timed(lambda: enc(input_ids=qo, use_cache=True), dev, reps)
    hq = qfirst().last_hidden_state
    nq = hq.shape[1]
    mpos = torch.arange(nq - len(it["markers"]), nq, device=dev)[None]
    t["qfirst_head_per_frame"] = timed(lambda: head_part(model, hq, torch.ones((1, nq), dtype=torch.long, device=dev),
                                                         mpos, torch.ones_like(mpos, dtype=torch.bool), b["qtype"]),
                                       dev, reps)
    row["qfirst_tokens_per_frame"] = int(per_frame.shape[1])

    cur = t["resize_on_device"] + t["vision_tower"] + t["lm_full"] + t["head"]
    t["sum_of_parts"] = round(cur, 3)
    t["qfirst_sum"] = round(t["resize_on_device"] + t["vision_tower"] + t["qfirst_lm_per_frame"]
                            + t["qfirst_head_per_frame"], 3)
    row["max_saving_frac_of_model_time"] = round(t["lm_question_options_on_cache"] / cur, 3)
    row["max_saving_frac_of_predict"] = round(t["lm_question_options_on_cache"] / t["predict_end_to_end"], 3)
    return row


@torch.no_grad()
def bench_batching(agent, reps, sizes=(1, 4, 16)):
    """ms per frame with B Atari env instances decided in one forward (``action_probs``, as ``play`` does)."""
    rng = np.random.default_rng(1)
    q = atari_question("Pong", ["NOOP", "FIRE", "RIGHT", "LEFT", "RIGHTFIRE", "LEFTFIRE"])["action"]
    out = {}
    for B in sizes:
        frames = [rng.integers(0, 256, ATARI_FRAME, dtype=np.uint8) for _ in range(B)]
        ms = timed(lambda: action_probs(agent, frames, q), agent.device, max(3, reps // 2), warmup=2)
        out[str(B)] = {"ms_per_step": ms, "ms_per_frame": round(ms / B, 3), "frames_per_s": round(1000 * B / ms, 1)}
    return out


@torch.no_grad()
def bench_graphs(agent, reps, frames=1):
    """Eager ``action_probs`` vs the same decision as one CUDA graph (``laya.static_step.StaticStep``), batch 1."""
    from laya.static_step import StaticStep

    label, shape, question = WORKLOADS[1]
    q = question["action"]
    rng = np.random.default_rng(2)
    cur = [rng.integers(0, 256, shape, dtype=np.uint8)]
    prev = [rng.integers(0, 256, shape, dtype=np.uint8)] if frames == 2 else None
    eager, graph = StaticStep(agent, q, frames, capture=False), StaticStep(agent, q, frames)
    ref = action_probs(agent, cur, q, prev)
    out = {"workload": label, "frames": frames, "eager_model_ms": timed(lambda: action_probs(agent, cur, q, prev),
                                                                        agent.device, reps),
           "unrolled_eager_ms": timed(lambda: eager.probs(cur, prev), agent.device, reps),
           "unrolled_vs_model_max_abs_prob": float(np.abs(eager.probs(cur, prev) - ref).max())}
    try:
        out["graph_ms"] = timed(lambda: graph.probs(cur, prev), agent.device, reps)
        out["graph_vs_model_max_abs_prob"] = float(np.abs(graph.probs(cur, prev) - ref).max())
    except Exception as e:  # noqa: BLE001 - report, do not hide
        out["graph_error"] = "%s: %s" % (type(e).__name__, str(e)[:200])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="fp32", choices=("fp32", "bf16"))
    ap.add_argument("--image-size", type=int, default=512)
    ap.add_argument("--option-attention", default="causal", choices=("causal", "block"))
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = torch's default)")
    ap.add_argument("--workloads", default="", help="comma-separated indexes into WORKLOADS (default: all)")
    ap.add_argument("--no-batching", action="store_true")
    ap.add_argument("--only-graphs", action="store_true")
    args = ap.parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    agent = VLMAgent(backbone="HuggingFaceTB/SmolVLM-256M-Instruct", device=args.device, dtype=args.dtype,
                     image_size=args.image_size, preprocess="gpu", option_attention=args.option_attention)
    hw = torch.cuda.get_device_name(0) if agent.device.type == "cuda" else platform.processor() or platform.machine()
    res = {"device": args.device, "hardware": hw, "cpus": os.cpu_count(), "torch_threads": torch.get_num_threads(),
           "dtype": args.dtype, "image_size": args.image_size, "option_attention": args.option_attention,
           "torch": str(torch.__version__), "workloads": []}
    picks = [int(i) for i in args.workloads.split(",")] if args.workloads else range(len(WORKLOADS))
    if args.only_graphs:
        picks, args.no_batching = [], True
    for i in picks:
        label, shape, question = WORKLOADS[i]
        row = bench_workload(agent, label, shape, question, args.reps)
        res["workloads"].append(row)
        print(json.dumps(row), flush=True)
    if not args.only_graphs:
        two = bench_workload(agent, WORKLOADS[1][0] + ", 2 frames", ATARI_FRAME, WORKLOADS[1][2], args.reps, frames=2)
        res["workloads"].append(two)
        print(json.dumps(two), flush=True)
    if agent.device.type == "cuda":
        res["graphs"] = [bench_graphs(agent, args.reps, f) for f in (1, 2)]
        print(json.dumps(res["graphs"]), flush=True)
    if not args.no_batching:
        res["batching"] = bench_batching(agent, args.reps)
        print(json.dumps(res["batching"]), flush=True)
    print("RESULT " + json.dumps(res))
    return res


if __name__ == "__main__":
    main()
