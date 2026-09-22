"""Training sketch for the VLM-backed decision model (``laya.vlm``), SmolVLM or ModernVBERT.

Fine-tunes on multiple-choice VQA with the same objective as the text model's notebooks: a soft
cross-entropy term plus a proper-scoring-rule policy-gradient term over noisy logits (``proper_reward``).
Options are shuffled per example so a causal backbone cannot learn a position prior (harmless for the
bidirectional one). Nothing here depends on the backbone family: the sequence builder and the model's
readout follow the processor and checkpoint (``laya.vlm.processor_readout``).

Data sources (adapters take one HF ``datasets`` row each):
  * A-OKVQA (``HuggingFaceM4/A-OKVQA``)      -> ``choice``
  * ScienceQA (``derek-thomas/ScienceQA``)   -> ``choice`` (image optional, hint as context)
  * VQAv2 yes/no (``HuggingFaceM4/VQAv2``)   -> ``noul`` with soft target = fraction of "yes" votes

Prepared datasets (``load_jsonl_examples``): ``<root>/<name>/<split>.jsonl`` + ``images/``, one record per line
``{"id", "image", "state_text", "question": {"type", "instructions", "criteria"}, "label"}``. ``modal_app.py``
runs ``finetune`` on these from the ``laya-datasets`` volume.

Smoke run on a tiny synthetic batch (no downloads beyond the backbone):
    python -m laya.vlm_train --synthetic --steps 3 --freeze head
    python -m laya.vlm_train --synthetic --steps 3 --freeze head --backbone ModernVBERT/modernvbert
"""
import argparse
import functools
import json
import math
import os
import random
import time
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch

from .common import QTYPES, ece_score, proper_reward, render_options
from .vlm import VLMAgent, VLMDecisionModel, build_vlm_inputs, collate_vlm, set_trainable

# ---------------------------------------------------------------------------------------------------------
# Examples: {"state": ..., "q": {"t", "ins", "crit"}, "target": [prob per option in label order]}
# ---------------------------------------------------------------------------------------------------------


def _one_hot(i: int, k: int) -> List[float]:
    return [1.0 if j == i else 0.0 for j in range(k)]


def _choice_example(state, question: str, choices: List[str], answer: int) -> Optional[Dict]:
    choices = [str(c) for c in choices]
    if len(set(choices)) != len(choices) or not 0 <= answer < len(choices):
        return None
    return {"state": state, "q": {"t": "choice", "ins": question, "crit": {c: None for c in choices}}, "target": _one_hot(answer, len(choices))}


def aokvqa_example(row: Dict) -> Optional[Dict]:
    return _choice_example({"image": row["image"]}, row["question"], row["choices"], int(row["correct_choice_idx"]))


def scienceqa_example(row: Dict) -> Optional[Dict]:
    state = {"context": row["hint"]} if row.get("hint") else {}
    if row.get("image") is not None:
        state["image"] = row["image"]
    return _choice_example(state or "", row["question"], row["choices"], int(row["answer"]))


def vqav2_yesno_example(row: Dict) -> Optional[Dict]:
    if row.get("answer_type") != "yes/no":
        return None
    votes = [a["answer"] if isinstance(a, dict) else a for a in row.get("answers") or []]
    votes = [v for v in votes if v in ("yes", "no")]
    if votes:
        p_yes = sum(v == "yes" for v in votes) / len(votes)
    elif row.get("multiple_choice_answer") in ("yes", "no"):
        p_yes = float(row["multiple_choice_answer"] == "yes")
    else:
        return None
    return {"state": {"image": row["image"]}, "q": {"t": "noul", "ins": row["question"], "crit": None}, "target": [1.0 - p_yes, p_yes]}


ADAPTERS = {
    "aokvqa": ("HuggingFaceM4/A-OKVQA", aokvqa_example),
    "scienceqa": ("derek-thomas/ScienceQA", scienceqa_example),
    "vqav2_yesno": ("HuggingFaceM4/VQAv2", vqav2_yesno_example),
}


def load_hf_examples(name: str, split: str = "train", limit: int = 1000) -> List[Dict]:
    """Stream ``limit`` usable examples from one of ``ADAPTERS`` (requires the ``datasets`` package)."""
    from datasets import load_dataset

    repo, fn = ADAPTERS[name]
    out = []
    for row in load_dataset(repo, split=split, streaming=True):
        ex = fn(row)
        if ex is not None:
            out.append(ex)
        if len(out) >= limit:
            break
    return out


def synthetic_examples(n: int = 8, seed: int = 0) -> List[Dict]:
    """Coloured squares on white: colour (choice), is-red (noul), size (score)."""
    from PIL import Image, ImageDraw

    rng = random.Random(seed)
    colors = {"red": (220, 30, 30), "blue": (30, 60, 220), "green": (30, 170, 60)}
    sizes = ["small", "medium", "large"]
    out = []
    for _ in range(n):
        c, s = rng.choice(list(colors)), rng.randrange(3)
        img = Image.new("RGB", (96, 96), (255, 255, 255))
        half = (8, 20, 40)[s]
        ImageDraw.Draw(img).rectangle([48 - half, 48 - half, 48 + half, 48 + half], fill=colors[c])
        state = {"image": img}
        out.append({"state": state, "q": {"t": "choice", "ins": "What color is the square?", "crit": {k: None for k in colors}}, "target": _one_hot(list(colors).index(c), 3)})
        out.append({"state": state, "q": {"t": "noul", "ins": "Is the square red?", "crit": None}, "target": _one_hot(int(c == "red"), 2)})
        out.append({"state": state, "q": {"t": "score", "ins": "How large is the square?", "crit": sizes}, "target": _one_hot(s, 3)})
    return out


def make_item(
    processor, ex: Dict, rng: random.Random, shuffle: bool = True, max_len: Optional[int] = None,
    head_max_len: int = 256, order: Optional[List[int]] = None,
) -> Dict:
    """Tokenize one example with a random (or the given) option order; the target is permuted to marker order.
    ``max_len`` defaults to the agent's (``laya.vlm.build_vlm_inputs``)."""
    k = len(render_options(ex["q"]))
    if order is None:
        order = list(range(k))
        if shuffle:
            rng.shuffle(order)
    it = build_vlm_inputs(processor, ex["state"], ex["q"], max_len, head_max_len, option_order=order)
    it["target"] = [ex["target"][i] for i in order]
    it["label"] = max(range(k), key=lambda j: it["target"][j])
    it["qtype"] = QTYPES[ex["q"]["t"]]
    it["order"] = order
    return it


# ---------------------------------------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------------------------------------


ACT_COSTS = (1.0, -3.0, -0.5)  # utility of acting when right / wrong, and of escalating: act when P(right) > 0.625

# Writing a resumable state.pt means a full CPU copy of the weights plus optimizer, a torch.save and a volume
# commit: seconds of GPU time each. These bound the cost however small an interval a caller asks for.
MIN_STATE_MINUTES = 2.0     # never write state.pt more often than this
STATE_SAVE_OVERHEAD = 20.0  # and never more often than 20x the last write's duration (<= 5% of the wall clock)


def vlm_loss(logits, target, qtype, mask, sigma: float = 0.3, group_size: int = 4, w_ce: float = 1.0,
             w_sph: float = 0.75, act_logits=None, act_costs=ACT_COSTS):
    """Proper-scoring-rule policy gradient over noisy logits + soft cross-entropy (as in the text notebooks).

    ``w_sph`` weights the spherical score in ``proper_reward`` (the original design used 0.5). ``w_ce`` can be
    annealed to 0 by the caller so training ends on proper scoring rules alone, which keeps calibration.
    ``act_logits`` (from the model's act head) adds the expected-utility term of the decide-or-escalate cost
    matrix ``act_costs``; without it the head gets no gradient.
    """
    logits = logits.float()
    k = mask.sum(-1, keepdim=True).float()
    eps = torch.randn((group_size,) + logits.shape, device=logits.device) * sigma * mask
    eps = (eps - eps.sum(-1, keepdim=True) / k) * mask
    z = logits.detach().unsqueeze(0) + eps
    q = torch.softmax(z.masked_fill(~mask, -1e4), -1)
    with torch.no_grad():
        r = proper_reward(q, target.unsqueeze(0), qtype, mask, w_sph=w_sph, w_rps=1.0)
        adv = r - r.mean(0, keepdim=True)
        adv = adv / (adv.std() + 1e-6)
    logp = -(((z - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma**2)
    loss_rl = -(adv * logp).mean()
    loss_ce = -(target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1).mean()
    loss = loss_rl + w_ce * loss_ce
    if act_logits is not None:
        c_ok, c_bad, c_esc = act_costs
        p_act = torch.softmax(act_logits.float(), -1)[:, 0]
        with torch.no_grad():  # how right the model's own choice is, as a probability
            y = target.gather(1, logits.argmax(-1, keepdim=True)).squeeze(1)
        eu = p_act * (c_ok * y + c_bad * (1 - y)) + (1 - p_act) * c_esc
        loss = loss - eu.mean()
    return loss, r.mean()


def _to(b: Dict, device, dtype) -> Dict:
    out = {}
    for k, v in b.items():
        if torch.is_tensor(v):
            v = v.to(device, non_blocking=True)
            if k == "pixel_values":
                v = v.to(dtype)
        out[k] = v
    return out


def _forward(model, b):
    return model(
        b["input_ids"], b["attention_mask"], b["marker_pos"], b["marker_mask"], b["qtype"],
        pixel_values=b["pixel_values"], pixel_attention_mask=b["pixel_attention_mask"], option_span=b["option_span"],
        raw_pixels=b.get("raw_pixels"), image_mask=b.get("image_mask"),
    )


def _single_thread_worker(_):
    torch.set_num_threads(1)  # avoid CPU oversubscription across loader workers


def group_weights(groups: Dict[str, Sequence], weights: Optional[Dict[str, float]] = None,
                  size_alpha: float = 0.0) -> Dict[str, float]:
    """Unnormalised sampling weight per group: ``weights.get(k, 1.0) * len(group_k) ** size_alpha``.

    The defaults give every group the same weight (the equal sampling every earlier run used); ``size_alpha=1``
    draws in proportion to size (a plain shuffle of the union), ``0.5`` in proportion to its square root.
    """
    weights = weights or {}
    return {k: float(weights.get(k, 1.0)) * float(len(g)) ** size_alpha for k, g in groups.items()}


def mix_probabilities(groups: Dict[str, Sequence], weights: Optional[Dict[str, float]] = None,
                      size_alpha: float = 0.0) -> Dict[str, float]:
    """``group_weights`` normalised to sum to 1 (the effective mix before any ``max_passes`` cap)."""
    w = group_weights(groups, weights, size_alpha)
    total = sum(w.values()) or 1.0
    return {k: v / total for k, v in w.items()}


class ItemStream(torch.utils.data.IterableDataset):
    """Endless stream of shuffled-option items, sampling each ``balance_key`` group (dataset) by weight.

    Group ``k`` is drawn with probability proportional to ``weights.get(k, 1.0) * len(group_k) ** size_alpha``
    (``group_weights``); the defaults draw every group equally. ``max_passes`` caps how many times a group is
    sampled (in passes over it); a capped group leaves the mix and the others keep their relative shares. The
    stream ends when every group is capped. With several loader workers the cap is split evenly between them.
    ``consumed`` (samples already taken per group in an earlier attempt of a resumed run) counts against those
    caps.
    """

    def __init__(self, processor, examples: List[Dict], seed: int = 0, balance_key: str = "dataset",
                 max_passes: Optional[float] = None, consumed: Optional[Dict[str, int]] = None,
                 weights: Optional[Dict[str, float]] = None, size_alpha: float = 0.0, **item_kw):
        self.processor, self.seed, self.item_kw, self.max_passes = processor, seed, item_kw, max_passes
        self.consumed = consumed or {}
        self.groups: Dict[str, List[Dict]] = {}
        for ex in examples:
            self.groups.setdefault(ex.get(balance_key, "_"), []).append(ex)
        self.keys = sorted(self.groups)
        self.weights = group_weights(self.groups, weights, size_alpha)
        unknown = sorted(set(weights or {}) - set(self.groups))
        if unknown:
            print("mix weights for groups not in the data (ignored): %s" % unknown)

    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        n_workers = wi.num_workers if wi else 1
        rng = random.Random(self.seed * 1000 + (wi.id if wi else 0))
        left = {k: (self.max_passes * len(g) / n_workers if self.max_passes else math.inf) for k, g in self.groups.items()}
        for k in left:  # a resumed run continues the caps where the previous attempt stopped
            left[k] -= self.consumed.get(k, 0) / n_workers
        while True:
            keys = [k for k in self.keys if left[k] >= 1]
            if not keys:
                return
            key = rng.choices(keys, weights=[self.weights[k] for k in keys])[0]
            left[key] -= 1
            ex = rng.choice(self.groups[key])
            try:
                it = make_item(self.processor, ex, rng, **self.item_kw)
            except (OSError, ValueError) as e:  # unreadable image / over-long question
                print("skipping %s: %s" % (ex.get("id"), e))
                continue
            it["dataset"] = key
            yield it


def _collate_train(items, pad_id):
    b = collate_vlm(items, pad_id)
    b["dataset"] = [it["dataset"] for it in items]
    return b


def train(
    model: VLMDecisionModel,
    processor,
    examples: List[Dict],
    steps: int = 100,
    batch_size: int = 2,
    freeze: str = "head",
    n_last: int = 4,
    lr_head: float = 1e-4,
    lr_backbone: float = 2e-5,
    sigma: float = 0.3,
    device: Optional[str] = None,
    seed: int = 0,
    log_every: int = 1,
    max_minutes: Optional[float] = None,
    num_workers: int = 0,
    prefetch_factor: int = 4,
    warmup: int = 0,
    eval_fn: Optional[Callable[[int], object]] = None,
    eval_every: int = 0,
    max_passes: Optional[float] = None,
    stats: Optional[Dict] = None,
    group_size: int = 4,
    w_sph: float = 0.75,
    w_ce: float = 1.0,
    w_ce_schedule: str = "const",
    sigma_end: Optional[float] = None,
    train_act: bool = False,
    balance_key: str = "dataset",
    resume: Optional[Dict] = None,
    save_state_fn: Optional[Callable[[int, Dict], None]] = None,
    save_state_every_min: float = 0.0,
    mix_weights: Optional[Dict[str, float]] = None,
    mix_alpha: float = 0.0,
) -> List[float]:
    """Single-device loop; stops at ``steps``, ``max_minutes``, or when every dataset hits ``max_passes``.

    bf16 autocast on CUDA. Returns per-step losses; ``stats`` (if given) receives samples per dataset, steps/s
    and the fraction of training time spent waiting on the data loader (both excluding eval time).
    The LR follows linear warmup then cosine decay to 10%, on whichever of step or wall-clock progress is further.

    Objective options (all default to the previous behaviour): ``group_size`` and ``sigma`` (annealed to
    ``sigma_end`` over training when given) control the exploration noise, ``w_sph`` the spherical score, and
    ``w_ce_schedule="anneal"`` holds the cross-entropy weight at ``w_ce`` for the first 30% of progress then
    decays it linearly to 0 by 80%. ``train_act`` trains the act/escalate head on its cost matrix.

    Sampling mix: dataset ``k`` is drawn with probability proportional to
    ``mix_weights.get(k, 1.0) * n_k ** mix_alpha`` (``ItemStream``); the defaults draw every dataset equally.

    ``eval_fn(step)`` is called every ``eval_every`` steps and should return a truthy value when it really
    evaluated, so a caller can use it as a cheap progress probe at a small ``eval_every`` without paying for a
    state write on every call.

    Durability: ``save_state_fn(step, state)`` is called every ``save_state_every_min`` minutes and after every
    eval that ran with ``{"step", "opt", "rng", "losses", "seen", "elapsed_s", "eval_s"}``, so the caller can
    write a resumable checkpoint; passing that dict back as ``resume`` continues the run (optimizer, LR schedule,
    loss history, per-dataset sample counts and the time budget, which then counts across attempts). The sampler
    is an endless random stream, so a resumed run re-seeds it rather than replaying the same order.
    ``save_state_every_min`` is clamped to ``MIN_STATE_MINUTES`` and backed off further when writes are slow
    (``STATE_SAVE_OVERHEAD``), so no caller can spend most of its GPU time checkpointing.
    """
    device = torch.device(device or next(model.parameters()).device)
    torch.manual_seed(seed)
    n_train = set_trainable(model, freeze, n_last=n_last)
    enc = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("encoder.")]
    head = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("encoder.")]
    groups = [{"params": head, "lr": lr_head}] + ([{"params": enc, "lr": lr_backbone}] if enc else [])
    base_lrs = [g["lr"] for g in groups]
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    if resume:
        opt.load_state_dict(resume["opt"])
    model.to(device).train()
    dtype = model.encoder.dtype
    amp = device.type == "cuda"
    stream = ItemStream(processor, examples, seed + (resume["step"] if resume else 0), balance_key=balance_key,
                        max_passes=max_passes, consumed=resume["seen"] if resume else None,
                        weights=mix_weights, size_alpha=mix_alpha)
    probs = mix_probabilities(stream.groups, mix_weights, mix_alpha)
    print("sampling mix (alpha=%g, before max_passes): %s"
          % (mix_alpha, ", ".join("%s %.3f" % (k, probs[k]) for k in stream.keys)), flush=True)
    loader = torch.utils.data.DataLoader(
        stream,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=functools.partial(_collate_train, pad_id=processor.tokenizer.pad_token_id),
        pin_memory=amp,
        persistent_workers=num_workers > 0,
        worker_init_fn=_single_thread_worker,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )
    print("training %d params (freeze=%s) on %s, batch %d, amp=%s" % (n_train, freeze, device, batch_size, amp))
    losses, t0, step, wait, t_eval = [], time.time(), 0, 0.0, 0.0
    seen: Dict[str, int] = {}
    if resume:
        losses, step, seen, t_eval = list(resume["losses"]), resume["step"], dict(resume["seen"]), resume["eval_s"]
        t0 = time.time() - resume["elapsed_s"]  # the budget counts training time across attempts
        random.setstate(resume["rng"]["py"])
        np.random.set_state(resume["rng"]["np"])
        torch.set_rng_state(resume["rng"]["torch"])
        print("resuming at step %d (%.1f min already spent, %d samples seen)"
              % (step, resume["elapsed_s"] / 60, sum(seen.values())), flush=True)
    budget = max_minutes * 60 if max_minutes else None
    t_state, save_s = time.time(), 0.0
    min_state_gap = 0.0
    if save_state_every_min:
        min_state_gap = max(save_state_every_min, MIN_STATE_MINUTES) * 60
        if save_state_every_min < MIN_STATE_MINUTES:
            print("save_state_every_min=%.3f min is below the %.1f min floor (a state write costs seconds of GPU "
                  "time); using %.1f min" % (save_state_every_min, MIN_STATE_MINUTES, MIN_STATE_MINUTES), flush=True)

    def state(step_):
        return {"step": step_, "opt": opt.state_dict(), "losses": losses, "seen": seen,
                "elapsed_s": time.time() - t0, "eval_s": t_eval,
                "rng": {"py": random.getstate(), "np": np.random.get_state(), "torch": torch.get_rng_state()}}

    def do_save(step_):
        nonlocal t_state, save_s
        ts = time.time()
        model.eval()
        save_state_fn(step_, state(step_))
        model.train()
        save_s, t_state = time.time() - ts, time.time()

    def save_due():
        # Beyond the floor, back off when writes are slow: state.pt never eats more than ~1/20 of the wall clock,
        # however often it is asked for.
        return time.time() - t_state >= max(min_state_gap, STATE_SAVE_OVERHEAD * save_s)

    t_fetch = time.time()
    for batch in loader:
        wait += time.time() - t_fetch
        for name in batch["dataset"]:
            seen[name] = seen.get(name, 0) + 1
        progress = step / max(1, steps)
        if budget:
            progress = max(progress, (time.time() - t0) / budget)
        if progress >= 1.0:
            break
        f = min(1.0, (step + 1) / warmup) if warmup else 1.0
        f *= 0.1 + 0.45 * (1 + math.cos(math.pi * progress))
        for g, lr in zip(groups, base_lrs):
            g["lr"] = lr * f
        w_ce_now = w_ce
        if w_ce_schedule == "anneal":  # hold, then decay to 0 between 30% and 80% of progress
            w_ce_now = w_ce * min(1.0, max(0.0, (0.8 - progress) / 0.5))
        sigma_now = sigma if sigma_end is None else sigma + (sigma_end - sigma) * min(1.0, progress)
        b = _to(batch, device, dtype)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            logits, act = _forward(model, b)
        loss, reward = vlm_loss(logits, b["target"], b["qtype"], b["marker_mask"], sigma=sigma_now,
                                group_size=group_size, w_ce=w_ce_now, w_sph=w_sph,
                                act_logits=act if train_act else None)
        if not train_act:
            loss = loss + 0.0 * act.float().sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 1.0)
        opt.step()
        losses.append(loss.item())
        if log_every and step % log_every == 0:
            recent = losses[-log_every:]
            print("step %d | %.1f min | loss %.4f (avg %.4f) | reward %.3f | lr %.2e | w_ce %.2f | sigma %.2f | "
                  "data wait %.0f%%" % (step, (time.time() - t0) / 60, losses[-1], sum(recent) / len(recent),
                                        reward.item(), groups[0]["lr"], w_ce_now, sigma_now,
                                        100 * wait / max(1e-6, time.time() - t0)), flush=True)
        step += 1
        if eval_fn is not None and eval_every and step % eval_every == 0:
            te = time.time()
            model.eval()
            ran = eval_fn(step)
            model.train()
            t_eval += time.time() - te
            # Only a real eval earns a state write. ``eval_fn`` is often a cheap probe called every few steps
            # that decides for itself whether to evaluate (it returns falsey when it did not), and writing
            # state.pt costs seconds of GPU time.
            if save_state_fn is not None and ran:
                do_save(step)
        if save_state_fn is not None and min_state_gap and save_due():
            do_save(step)
        t_fetch = time.time()
    model.eval()
    if stats is not None:
        train_s = max(1e-6, time.time() - t0 - t_eval)
        stats.update(steps=step, samples_per_dataset=seen, train_minutes=train_s / 60, eval_minutes=t_eval / 60,
                     steps_per_s=step / train_s, data_wait_frac=wait / train_s)
    return losses


# ---------------------------------------------------------------------------------------------------------
# Prepared VQA datasets: <root>/<name>/{images/, <split>.jsonl, _READY}
# ---------------------------------------------------------------------------------------------------------


def jsonl_example(rec: Dict, root: str, dataset: str = "") -> Optional[Dict]:
    """``{"id", "image", "state_text", "question": {type, instructions, criteria}, "label"}`` -> training example.

    ``"images"`` (a list of paths) replaces ``"image"`` for a multi-image record, e.g. NLVR2's pairs.
    ``label`` indexes the rendered options (choice: criteria order; score: level; noul: 0=false, 1=true).
    An optional ``"target"`` (a probability per option, same order) replaces the one-hot target, e.g. an expert
    policy's action distribution; ``label`` is still used for accuracy.
    """
    qdef = rec["question"]
    q = VLMAgent._to_internal(qdef)
    if q["t"] == "choice" and len(q["crit"]) != len(qdef["criteria"]):
        return None  # duplicate choice strings collapse in the dict form
    k = len(render_options(q))
    label = int(rec["label"])
    if not 0 <= label < k:
        return None
    state = {}
    if rec.get("image"):
        state["image"] = os.path.join(root, rec["image"])
    elif rec.get("images"):
        state["images"] = [os.path.join(root, p) for p in rec["images"]]
    if rec.get("state_text"):
        state["context"] = rec["state_text"]
    target = _one_hot(label, k)
    soft = rec.get("target")
    if soft is not None and len(soft) == k and min(soft) >= 0 and sum(soft) > 0:
        target = [float(p) / sum(soft) for p in soft]
    return {"state": state or "", "q": q, "target": target, "label": label, "dataset": dataset, "id": rec.get("id")}


def load_jsonl_examples(root: str, name: str, split: str, limit: Optional[int] = None) -> List[Dict]:
    """Load ``<root>/<name>/<split>.jsonl``; ``limit`` keeps the first records in file order."""
    base = os.path.join(root, name)
    with open(os.path.join(base, split + ".jsonl")) as f:
        recs = [json.loads(line) for line in f if line.strip()]
    if limit:
        recs = recs[:limit]
    out = [jsonl_example(r, base, name) for r in recs]
    return [ex for ex in out if ex is not None]


# ---------------------------------------------------------------------------------------------------------
# Evaluation and calibration
# ---------------------------------------------------------------------------------------------------------


class _EvalItems(torch.utils.data.Dataset):
    def __init__(self, processor, examples, pairs, transform=None):
        self.processor, self.examples, self.pairs, self.transform = processor, examples, pairs, transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, j):
        i, order = self.pairs[j]
        ex = self.transform(self.examples[i]) if self.transform else self.examples[i]
        it = make_item(self.processor, ex, random.Random(0), shuffle=False, order=order)
        it["index"] = i
        return it


def _collate_eval(items, pad_id):
    b = collate_vlm(items, pad_id)
    b["index"] = [it["index"] for it in items]
    b["order"] = [it["order"] for it in items]
    return b


@torch.no_grad()
def collect_logits(
    model: VLMDecisionModel,
    processor,
    examples: List[Dict],
    batch_size: int = 16,
    num_workers: int = 0,
    device=None,
    orders: Optional[Callable[[int], List[List[int]]]] = None,
    transform: Optional[Callable[[Dict], Dict]] = None,
) -> List[Dict]:
    """Label-order logits per example.

    ``orders(k)`` gives the option orders to score each k-option example under (default: the example's own
    ``"order"`` if it has one, else identity). ``transform(example)`` is applied in the loader just before
    tokenizing, e.g. ``laya.robustness.realize`` to perturb the images without storing them.
    ``"logits"`` is the mean over orders (as in ``VLMAgent.predict(n_permutations=...)``); ``"logits_per_order"``
    keeps each order's logits (label order), aligned with ``orders(k)``.
    """
    device = torch.device(device or next(model.parameters()).device)
    model.eval()
    amp = device.type == "cuda"
    pairs = []
    for i, ex in enumerate(examples):
        k = len(ex["target"])
        for order in (orders(k) if orders else [ex.get("order") or list(range(k))]):
            pairs.append((i, order))
    loader = torch.utils.data.DataLoader(
        _EvalItems(processor, examples, pairs, transform), batch_size=batch_size, num_workers=num_workers,
        collate_fn=functools.partial(_collate_eval, pad_id=processor.tokenizer.pad_token_id),
        worker_init_fn=_single_thread_worker,
    )
    per_ex: Dict[int, List[torch.Tensor]] = {}
    for batch in loader:
        b = _to(batch, device, model.encoder.dtype)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            logits, _ = _forward(model, b)
        logits = logits.float().cpu()
        for r, (i, order) in enumerate(zip(batch["index"], batch["order"])):
            k = len(order)
            z = torch.empty(k)
            z[torch.tensor(order)] = logits[r, :k]  # marker j scored option order[j]
            per_ex.setdefault(i, []).append(z)
    out = []
    for i, ex in enumerate(examples):
        zs = per_ex[i]
        out.append({"logits": torch.stack(zs).mean(0), "logits_per_order": zs, "target": torch.tensor(ex["target"]),
                    "qtype": QTYPES[ex["q"]["t"]], "dataset": ex.get("dataset", "_"),
                    "label": ex.get("label", int(np.argmax(ex["target"])))})
    return out


def cyclic_orders(n: int) -> Callable[[int], List[List[int]]]:
    """The first ``n`` cyclic shifts of the options (fewer if an example has fewer options)."""
    return lambda k: [[(i + s) % k for i in range(k)] for s in range(min(n, k))]


def fit_temperatures_from(records: List[Dict]) -> List[float]:
    """Per-type temperature scaling by LBFGS (same as the text training notebook)."""
    temps = []
    for t in range(3):
        sel = [r for r in records if r["qtype"] == t]
        if len(sel) < 10:
            temps.append(1.0)
            continue
        kmax = max(len(r["logits"]) for r in sel)
        Z = torch.full((len(sel), kmax), -1e4)
        T = torch.zeros((len(sel), kmax))
        for i, r in enumerate(sel):
            Z[i, : len(r["logits"])], T[i, : len(r["target"])] = r["logits"], r["target"]
        log_t = torch.zeros(1, requires_grad=True)
        lbfgs = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

        def closure():
            lbfgs.zero_grad()
            loss = -(T * torch.log_softmax(Z / log_t.exp(), -1)).sum(-1).mean()
            loss.backward()
            return loss

        lbfgs.step(closure)
        temps.append(float(torch.clamp(log_t.exp(), 0.1, 10.0)))
    return temps


def fit_temperatures(model: VLMDecisionModel, processor, examples: List[Dict], **kw) -> List[float]:
    return fit_temperatures_from(collect_logits(model, processor, examples, **kw))


def metrics_from(records: List[Dict], temperatures: Sequence[float] = (1.0, 1.0, 1.0)) -> Dict[str, Dict[str, float]]:
    """Accuracy, ECE (max-prob confidence, 15 bins), and NLL overall and per dataset.

    Groups with ``score`` records also get two ordinal metrics over those records: ``mae``, the absolute
    difference between the expected level under the model and under the target (``|E_p[i] - E_t[i]|``, in
    levels; with a one-hot target that is the distance from the label), and ``xent``, the cross-entropy against
    the (possibly soft) target, which is what a vote-histogram target like AVA's is actually trained on. Argmax
    accuracy and NLL against the argmax label understate such a model: it is trained to spread probability.

    ``choice`` and ``noul`` records with a soft target (human vote shares, e.g. CIFAR-10H or VQAv2 yes/no) add
    ``n_soft`` and ``soft_xent``, the same cross-entropy for those records. Next to each cross-entropy is a
    model-free reference, ``prior_xent`` / ``prior_soft_xent``: the cross-entropy of the same records' targets
    against their mean target (the dataset's average vote histogram), where they all have the same option count.
    """
    groups: Dict[str, List] = {"all": []}
    ordinal: Dict[str, List] = {}
    soft: Dict[str, List] = {}
    targets: Dict[str, Dict[str, List]] = {"xent": {}, "soft_xent": {}}
    for r in records:
        p = torch.softmax(r["logits"] / temperatures[r["qtype"]], -1)
        row = (float(p.max()), float(int(p.argmax()) == r["label"]), -float(torch.log(p[r["label"]].clamp_min(1e-12))))
        groups["all"].append(row)
        groups.setdefault(r["dataset"], []).append(row)
        if r["qtype"] == QTYPES["score"] and r.get("target") is not None:
            t = torch.as_tensor(r["target"], dtype=torch.float32)
            t = t / t.sum().clamp_min(1e-12)
            levels = torch.arange(len(p), dtype=torch.float32)
            orow = (abs(float((p * levels).sum() - (t * levels).sum())), -float((t * torch.log(p.clamp_min(1e-12))).sum()))
            ordinal.setdefault("all", []).append(orow)
            ordinal.setdefault(r["dataset"], []).append(orow)
            for g in ("all", r["dataset"]):
                targets["xent"].setdefault(g, []).append(t)
        elif r["qtype"] != QTYPES["score"] and float(r["target"].max()) < 1.0 - 1e-6:
            t = torch.as_tensor(r["target"], dtype=torch.float32)
            t = t / t.sum().clamp_min(1e-12)
            xent = -float((t * torch.log(p.clamp_min(1e-12))).sum())
            for g in ("all", r["dataset"]):
                soft.setdefault(g, []).append(xent)
                targets["soft_xent"].setdefault(g, []).append(t)
    out = {}
    for name, rows in groups.items():
        a = np.array(rows) if rows else np.zeros((0, 3))
        out[name] = {"n": len(rows), "acc": float(a[:, 1].mean()) if rows else float("nan"),
                     "ece": ece_score(a[:, 0], a[:, 1]), "nll": float(a[:, 2].mean()) if rows else float("nan")}
        if name in ordinal:
            o = np.array(ordinal[name])
            out[name].update(n_score=len(o), mae=float(o[:, 0].mean()), xent=float(o[:, 1].mean()))
        if name in soft:
            out[name].update(n_soft=len(soft[name]), soft_xent=float(np.mean(soft[name])))
        for key, by_group in targets.items():
            ts = by_group.get(name)
            if ts and key in out[name] and len({len(t) for t in ts}) == 1:
                T = torch.stack(ts)
                out[name]["prior_" + key] = -float((T * torch.log(T.mean(0).clamp_min(1e-12))).sum(-1).mean())
    return out


def evaluate(model: VLMDecisionModel, processor, examples: List[Dict], temperatures=(1.0, 1.0, 1.0), **kw) -> Dict:
    return metrics_from(collect_logits(model, processor, examples, **kw), temperatures)


def format_metrics(m: Dict) -> str:
    def one(k, v):
        s = "%s n=%d acc=%.3f ece=%.3f nll=%.3f" % (k, v["n"], v["acc"], v["ece"], v["nll"])
        for key in ("xent", "soft_xent"):
            if key == "xent" and "mae" in v:
                s += " mae=%.3f" % v["mae"]
            if key in v:
                s += " %s=%.3f" % (key, v[key])
                if "prior_" + key in v:
                    s += " (prior %.3f)" % v["prior_" + key]
        return s
    return " | ".join(one(k, v) for k, v in m.items())


def main(argv: Optional[Iterable[str]] = None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backbone", default="HuggingFaceTB/SmolVLM-256M-Instruct")
    ap.add_argument("--init", default=None, help="saved VLM agent dir to continue from")
    ap.add_argument("--synthetic", action="store_true", help="train on coloured-square toy data")
    ap.add_argument("--dataset", action="append", choices=sorted(ADAPTERS), default=[])
    ap.add_argument("--limit", type=int, default=1000, help="examples per dataset")
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--freeze", choices=["head", "last_n", "full"], default="head")
    ap.add_argument("--n-last", type=int, default=4)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    agent = VLMAgent(args.init, backbone=None if args.init else args.backbone, device=args.device)
    examples = synthetic_examples() if args.synthetic else []
    for name in args.dataset:
        examples += load_hf_examples(name, limit=args.limit)
    if not examples:
        ap.error("no training data: pass --synthetic and/or --dataset")
    losses = train(
        agent.model, agent.processor, examples, steps=args.steps, batch_size=args.batch_size,
        freeze=args.freeze, n_last=args.n_last, device=str(agent.device),
    )
    print("final loss %.4f (finite=%s)" % (losses[-1], math.isfinite(losses[-1])))
    if args.out:
        agent.temperature = fit_temperatures(agent.model, agent.processor, examples)
        agent.save(args.out, include_backbone=args.freeze != "head")
        print("saved to", args.out)


if __name__ == "__main__":
    main()
