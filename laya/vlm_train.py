"""Training sketch for the SmolVLM-backed decision model (``laya.vlm``).

Fine-tunes on multiple-choice VQA with the RLCD objective: a policy gradient on strictly proper scoring
rules (``proper_reward``) over noisy logits, and no cross-entropy term. Options are shuffled per example so
the causal backbone cannot learn a position prior.

Data sources (adapters take one HF ``datasets`` row each):
  * A-OKVQA (``HuggingFaceM4/A-OKVQA``)      -> ``choice``
  * ScienceQA (``derek-thomas/ScienceQA``)   -> ``choice`` (image optional, hint as context)
  * VQAv2 yes/no (``HuggingFaceM4/VQAv2``)   -> ``noul`` with soft target = fraction of "yes" votes

Prepared datasets (``load_jsonl_examples``): ``<root>/<name>/<split>.jsonl`` + ``images/``, one record per line
``{"id", "image", "state_text", "question": {"type", "instructions", "criteria"}, "label"}``. ``modal_app.py``
runs ``finetune`` on these from the ``laya-datasets`` volume.

Smoke run on a tiny synthetic batch (no downloads beyond the backbone):
    python -m laya.vlm_train --synthetic --steps 3 --freeze head
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
    processor, ex: Dict, rng: random.Random, shuffle: bool = True, max_len: int = 1024, head_max_len: int = 256,
    order: Optional[List[int]] = None,
) -> Dict:
    """Tokenize one example with a random (or the given) option order; the target is permuted to marker order."""
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


SIGMA_START, SIGMA_END = 1.0, 0.3  # exploration noise, wide early and narrow late


def sigma_at(progress: float, sigma: float = SIGMA_START, sigma_end: float = SIGMA_END) -> float:
    """Exploration noise at ``progress`` in [0, 1]: linear decay from ``sigma`` to ``sigma_end``, clamped.

    Wide noise early samples far enough across the simplex to find the reward's shape; narrow noise late makes
    the same estimator a fine-grained local one.
    """
    return sigma + (sigma_end - sigma) * min(1.0, max(0.0, progress))


def vlm_loss(logits, target, qtype, mask, sigma: float = SIGMA_START, group_size: int = 8, w_ce: float = 0.0,
             w_sph: float = 0.5, w_rps: float = 1.0):
    """RLCD: a policy gradient on strictly proper scoring rules, with no cross-entropy term.

    ``group_size`` noisy copies of the logits are drawn (Gaussian, scale ``sigma``, projected to zero mean
    across the valid options), each is scored by ``proper_reward`` (log score + ``w_sph`` x spherical, minus
    the ranked probability score on ordinal questions), and the Gaussian policy mean -- the model's logits --
    is pushed toward the copies that beat the group's own baseline. The gradient is therefore never computed
    from the reward, only estimated from it, which is what lets the reward be a scoring rule rather than a
    differentiable loss.

    Defaults are the RLCD design: ``w_sph=0.5``, ``group_size=8``, and ``sigma`` annealed 1.0 -> 0.3 by the
    caller (``sigma_at``). ``w_ce`` > 0 adds a soft cross-entropy term on top; it is off by default so the
    objective stays a proper scoring rule end to end. Returns ``(loss, mean reward)``.
    """
    logits = logits.float()
    k = mask.sum(-1, keepdim=True).float()
    eps = torch.randn((group_size,) + logits.shape, device=logits.device) * sigma * mask
    eps = (eps - eps.sum(-1, keepdim=True) / k) * mask
    z = logits.detach().unsqueeze(0) + eps
    q = torch.softmax(z.masked_fill(~mask, -1e4), -1)
    with torch.no_grad():
        r = proper_reward(q, target.unsqueeze(0), qtype, mask, w_sph=w_sph, w_rps=w_rps)
        adv = r - r.mean(0, keepdim=True)
        adv = adv / (adv.std() + 1e-6)
    logp = -(((z - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma**2)
    loss = -(adv * logp).mean()
    if w_ce:
        loss = loss - w_ce * (target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1).mean()
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
    )


def _single_thread_worker(_):
    torch.set_num_threads(1)  # avoid CPU oversubscription across loader workers


class ItemStream(torch.utils.data.IterableDataset):
    """Endless stream of shuffled-option items, sampling each ``balance_key`` group (dataset) equally.

    ``max_passes`` caps how many times a group is sampled (in passes over it); a capped group leaves the mix and
    the others keep equal shares. The stream ends when every group is capped. With several loader workers the
    cap is split evenly between them.
    """

    def __init__(self, processor, examples: List[Dict], seed: int = 0, balance_key: str = "dataset",
                 max_passes: Optional[float] = None, **item_kw):
        self.processor, self.seed, self.item_kw, self.max_passes = processor, seed, item_kw, max_passes
        self.groups: Dict[str, List[Dict]] = {}
        for ex in examples:
            self.groups.setdefault(ex.get(balance_key, "_"), []).append(ex)
        self.keys = sorted(self.groups)

    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        n_workers = wi.num_workers if wi else 1
        rng = random.Random(self.seed * 1000 + (wi.id if wi else 0))
        left = {k: (self.max_passes * len(g) / n_workers if self.max_passes else math.inf) for k, g in self.groups.items()}
        while True:
            keys = [k for k in self.keys if left[k] >= 1]
            if not keys:
                return
            key = rng.choice(keys)
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
    sigma: float = SIGMA_START,
    sigma_end: float = SIGMA_END,
    group_size: int = 8,
    w_sph: float = 0.5,
    w_ce: float = 0.0,
    device: Optional[str] = None,
    seed: int = 0,
    log_every: int = 1,
    max_minutes: Optional[float] = None,
    num_workers: int = 0,
    warmup: int = 0,
    eval_fn: Optional[Callable[[int], None]] = None,
    eval_every: int = 0,
    max_passes: Optional[float] = None,
    stats: Optional[Dict] = None,
) -> List[float]:
    """Single-device loop; stops at ``steps``, ``max_minutes``, or when every dataset hits ``max_passes``.

    bf16 autocast on CUDA. Returns per-step losses; ``stats`` (if given) receives samples per dataset, steps/s
    and the fraction of training time spent waiting on the data loader (both excluding eval time).
    The LR follows linear warmup then cosine decay to 10%, on whichever of step or wall-clock progress is further.

    Objective (``vlm_loss``): RLCD, a policy gradient on proper scoring rules. ``sigma`` decays to ``sigma_end``
    over the same progress measure as the LR, ``group_size`` sets how many noisy copies are scored per step and
    ``w_sph`` the spherical term. ``w_ce`` > 0 re-adds soft cross-entropy and defaults to off.

    The log line carries ``conf``, the mean probability of the model's own top option. A proper scoring rule is
    maximised at the target, so ``conf`` should settle near the targets' own mean top probability; if it climbs
    toward 1.0 while accuracy does not, the run is over-sharpening and calibration is being spent.
    """
    device = torch.device(device or next(model.parameters()).device)
    torch.manual_seed(seed)
    n_train = set_trainable(model, freeze, n_last=n_last)
    enc = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("encoder.")]
    head = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("encoder.")]
    groups = [{"params": head, "lr": lr_head}] + ([{"params": enc, "lr": lr_backbone}] if enc else [])
    base_lrs = [g["lr"] for g in groups]
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    model.to(device).train()
    dtype = model.encoder.dtype
    amp = device.type == "cuda"
    loader = torch.utils.data.DataLoader(
        ItemStream(processor, examples, seed, max_passes=max_passes),
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=functools.partial(_collate_train, pad_id=processor.tokenizer.pad_token_id),
        pin_memory=amp,
        persistent_workers=num_workers > 0,
        worker_init_fn=_single_thread_worker,
        prefetch_factor=4 if num_workers > 0 else None,
    )
    print("training %d params (freeze=%s) on %s, batch %d, amp=%s" % (n_train, freeze, device, batch_size, amp))
    losses, t0, step, wait, t_eval = [], time.time(), 0, 0.0, 0.0
    seen: Dict[str, int] = {}
    budget = max_minutes * 60 if max_minutes else None
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
        b = _to(batch, device, dtype)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            logits, act = _forward(model, b)
        loss, reward = vlm_loss(logits, b["target"], b["qtype"], b["marker_mask"], sigma=sigma_at(progress, sigma, sigma_end),
                                group_size=group_size, w_ce=w_ce, w_sph=w_sph)
        loss = loss + 0.0 * act.float().sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 1.0)
        opt.step()
        losses.append(loss.item())
        if log_every and step % log_every == 0:
            recent = losses[-log_every:]
            with torch.no_grad():
                conf = torch.softmax(logits.float().masked_fill(~b["marker_mask"], -1e4), -1).max(-1).values.mean()
            print("step %d | %.1f min | loss %.4f (avg %.4f) | reward %.3f | sigma %.2f | conf %.3f | lr %.2e | "
                  "data wait %.0f%%"
                  % (step, (time.time() - t0) / 60, losses[-1], sum(recent) / len(recent), reward.item(),
                     sigma_at(progress, sigma, sigma_end), conf.item(), groups[0]["lr"],
                     100 * wait / max(1e-6, time.time() - t0)), flush=True)
        step += 1
        if eval_fn is not None and eval_every and step % eval_every == 0:
            te = time.time()
            model.eval()
            eval_fn(step)
            model.train()
            t_eval += time.time() - te
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
    def __init__(self, processor, examples, pairs):
        self.processor, self.examples, self.pairs = processor, examples, pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, j):
        i, order = self.pairs[j]
        it = make_item(self.processor, self.examples[i], random.Random(0), shuffle=False, order=order)
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
) -> List[Dict]:
    """Label-order logits per example.

    ``orders(k)`` gives the option orders to score each k-option example under (default: identity only).
    ``"logits"`` is the mean over orders (as in ``VLMAgent.predict(n_permutations=...)``); ``"logits_per_order"``
    keeps each order's logits (label order), aligned with ``orders(k)``.
    """
    device = torch.device(device or next(model.parameters()).device)
    model.eval()
    amp = device.type == "cuda"
    pairs = []
    for i, ex in enumerate(examples):
        k = len(ex["target"])
        for order in (orders(k) if orders else [list(range(k))]):
            pairs.append((i, order))
    loader = torch.utils.data.DataLoader(
        _EvalItems(processor, examples, pairs), batch_size=batch_size, num_workers=num_workers,
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
    """Accuracy, ECE (max-prob confidence, 15 bins), and NLL overall and per dataset."""
    groups: Dict[str, List] = {"all": []}
    for r in records:
        p = torch.softmax(r["logits"] / temperatures[r["qtype"]], -1)
        row = (float(p.max()), float(int(p.argmax()) == r["label"]), -float(torch.log(p[r["label"]].clamp_min(1e-12))))
        groups["all"].append(row)
        groups.setdefault(r["dataset"], []).append(row)
    out = {}
    for name, rows in groups.items():
        a = np.array(rows) if rows else np.zeros((0, 3))
        out[name] = {"n": len(rows), "acc": float(a[:, 1].mean()) if rows else float("nan"),
                     "ece": ece_score(a[:, 0], a[:, 1]), "nll": float(a[:, 2].mean()) if rows else float("nan")}
    return out


def evaluate(model: VLMDecisionModel, processor, examples: List[Dict], temperatures=(1.0, 1.0, 1.0), **kw) -> Dict:
    return metrics_from(collect_logits(model, processor, examples, **kw), temperatures)


def format_metrics(m: Dict) -> str:
    return " | ".join("%s n=%d acc=%.3f ece=%.3f nll=%.3f" % (k, v["n"], v["acc"], v["ece"], v["nll"]) for k, v in m.items())


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
    ap.add_argument("--sigma", type=float, default=SIGMA_START, help="exploration noise at the start of training")
    ap.add_argument("--sigma-end", type=float, default=SIGMA_END, help="exploration noise at the end")
    ap.add_argument("--group-size", type=int, default=8, help="noisy logit copies scored per step")
    ap.add_argument("--w-sph", type=float, default=0.5, help="weight of the spherical score in the reward")
    ap.add_argument("--w-ce", type=float, default=0.0, help="weight of the soft cross-entropy add-on (0 = pure RLCD)")
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
        freeze=args.freeze, n_last=args.n_last, device=str(agent.device), sigma=args.sigma,
        sigma_end=args.sigma_end, group_size=args.group_size, w_sph=args.w_sph, w_ce=args.w_ce,
    )
    print("final loss %.4f (finite=%s)" % (losses[-1], math.isfinite(losses[-1])))
    if args.out:
        agent.temperature = fit_temperatures(agent.model, agent.processor, examples)
        agent.save(args.out, include_backbone=args.freeze != "head")
        print("saved to", args.out)


if __name__ == "__main__":
    main()
