"""Train a Laya Vision checkpoint for Ollama's ``/v1/systemone`` and export it as a Hugging Face model for GGUF.

Ollama scores each question by the next-token probabilities of the choice letters after its own prompt
(``laya.ollama``), so the model it serves is a plain SmolVLM causal LM, not Laya's option-scoring head. This module
builds that LM from a Laya Vision checkpoint (its backbone, plus the original SmolVLM's ``lm_head``), trains it on
Laya's data rendered exactly as Ollama renders a request, with the image tokens laid out as llama.cpp lays them
out, and saves it in the layout ``convert_hf_to_gguf.py`` reads.

The loss is the cross-entropy over the question's candidate letters only, which is the softmax Ollama computes.
Ollama always scores at temperature 1, so one temperature, fitted on held-out rows, is folded into ``lm_head``.
"""
import json
import math
import os
import random
import re
import time
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .common import QTYPES
from .ollama import IMAGE_ONLY_STATE, image_views, input_ids, letter_ids, prompts

BACKBONE = "HuggingFaceTB/SmolVLM-256M-Instruct"
IMAGE_MEAN, IMAGE_STD = 0.5, 0.5  # SmolVLM's processor, and the mmproj llama.cpp builds from it
STATES_WITHOUT_TEXT = (IMAGE_ONLY_STATE, "Answer about the attached image.", "Photo attached.")


# ---------------------------------------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------------------------------------


def build_lm(checkpoint: str, revision: Optional[str] = None, backbone_revision: Optional[str] = None,
             token=None, dtype=torch.float32):
    """(Idefics3ForConditionalGeneration, processor): a Laya Vision checkpoint's backbone (vision tower, connector
    and its cut text model) with the ``lm_head`` of the SmolVLM it was cut from. Returns the model on the CPU."""
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from transformers import AutoConfig, AutoProcessor, Idefics3ForConditionalGeneration

    local = checkpoint if os.path.isdir(checkpoint) else snapshot_download(checkpoint, revision=revision, token=token)
    cfg = json.load(open(os.path.join(local, "vlm_agent_config.json")))
    if not cfg["backbone"].startswith("HuggingFaceTB/SmolVLM"):
        raise ValueError("only SmolVLM checkpoints can be served by Ollama (got %s)" % cfg["backbone"])
    config = AutoConfig.from_pretrained(os.path.join(local, "backbone"))
    model = Idefics3ForConditionalGeneration(config).to(dtype)
    weights = {}
    with safe_open(os.path.join(local, "model.safetensors"), "pt") as f:
        for k in f.keys():
            if k.startswith("encoder."):
                weights["model." + k[len("encoder."):]] = f.get_tensor(k)
    src = snapshot_download(cfg["backbone"], revision=backbone_revision or cfg.get("backbone_revision"), token=token,
                            allow_patterns=["*.safetensors", "*.json"])
    for name in sorted(os.listdir(src)):
        if name.endswith(".safetensors"):
            with safe_open(os.path.join(src, name), "pt") as f:
                if "lm_head.weight" in f.keys():
                    weights["lm_head.weight"] = f.get_tensor("lm_head.weight")
    missing, unexpected = model.load_state_dict(weights, strict=False)
    if missing or unexpected:
        raise ValueError("weights do not fit the backbone: missing %s, unexpected %s" % (missing[:5], unexpected[:5]))
    from .vlm import snapshot_revision

    model.laya_sources = {"checkpoint": {"id": checkpoint, "revision": snapshot_revision(local) or revision},
                          "lm_head": {"id": cfg["backbone"], "revision": snapshot_revision(src)}}
    processor = AutoProcessor.from_pretrained(os.path.join(local, "processor"))
    return model, processor


def set_trainable(model, train_vision: bool = False) -> int:
    for name, p in model.named_parameters():
        p.requires_grad = train_vision or "vision_model" not in name
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------------------------------------
# Data: Laya examples as Ollama requests
# ---------------------------------------------------------------------------------------------------------


def public_question(q: Dict, rng: Optional[random.Random] = None) -> Dict:
    """A Laya example's internal question (``{"t", "ins", "crit"}``) as a /v1/systemone question. With ``rng``, a
    ``choice`` question's options are shuffled, so no letter is favoured; ``noul`` and ``score`` keep their order,
    which carries meaning. Also returns the permutation applied (new position -> old index)."""
    t, crit = q["t"], q["crit"]
    if t == "choice":
        keys = list(crit)
        order = list(range(len(keys)))
        if rng is not None:
            rng.shuffle(order)
        return {"type": "choice", "instructions": q["ins"], "criteria": {keys[i]: crit[keys[i]] for i in order}}, order
    if t == "score":
        return {"type": "score", "instructions": q["ins"], "criteria": list(crit)}, list(range(len(crit)))
    out = {"type": "noul", "instructions": q["ins"]}
    if crit:
        out["criteria"] = dict(crit)
    return out, [0, 1]


def _slug(text: str, n: int = 3) -> str:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return "_".join(words[:n]) or "question"


def _state(ex: Dict, rng: random.Random):
    state = ex["state"] if isinstance(ex["state"], dict) else {}
    text = state.get("context")
    if text:
        return {"context": text} if rng.random() < 0.5 else text
    return rng.choice(STATES_WITHOUT_TEXT)


def _images(ex: Dict) -> List[str]:
    state = ex["state"] if isinstance(ex["state"], dict) else {}
    if state.get("image"):
        return [state["image"]]
    return list(state.get("images") or [])


def requests_from(examples: Sequence[Dict], rng: random.Random, max_questions: int = 4,
                  p_single: float = 0.5, shuffle: bool = True) -> List[Dict]:
    """Group examples that share their images and text into one request each (up to ``max_questions`` questions,
    or one with probability ``p_single``), and return one training row per question: the request, which question
    is asked, and its target over that question's letters."""
    groups: Dict[tuple, List[Dict]] = {}
    for ex in examples:
        state = ex["state"] if isinstance(ex["state"], dict) else {}
        groups.setdefault((ex.get("dataset"), tuple(_images(ex)), state.get("context")), []).append(ex)
    rows = []
    for key in sorted(groups, key=str):
        exs = list(groups[key])
        if shuffle:
            rng.shuffle(exs)
        while exs:
            n = 1 if (rng.random() < p_single or max_questions <= 1) else rng.randint(2, max_questions)
            chunk, exs = exs[:n], exs[n:]
            questions, targets, names = {}, {}, []
            for ex in chunk:
                q, order = public_question(ex["q"], rng if shuffle else None)
                name = _slug(q["instructions"] if isinstance(q["instructions"], str) else json.dumps(q["instructions"]))
                if len(chunk) == 1 and rng.random() < 0.3:
                    name = rng.choice(("answer", "q", "decision"))
                while name in questions:
                    name += "_%d" % (len(names) + 1)
                questions[name] = q
                names.append(name)
                targets[name] = ([ex["target"][i] for i in order], ex)
            state = _state(chunk[0], rng)
            for name in names:
                target, ex = targets[name]
                rows.append({"images": _images(chunk[0]), "state": state, "questions": questions, "name": name,
                             "target": target, "label": int(np.argmax(target)), "qtype": QTYPES[ex["q"]["t"]],
                             "dataset": ex.get("dataset"), "id": ex.get("id")})
    return rows


def encode_row(row: Dict, tokenizer, system: Optional[str] = None, max_len: int = 2048) -> Optional[Dict]:
    """Token ids, image views and candidate letters for one row; None when the prompt is longer than ``max_len``."""
    from PIL import Image

    from .ollama import SYSTEM_PROMPT

    names = list(row["questions"])
    rendered = prompts(row["state"], row["questions"], n_images=len(row["images"]),
                       system=SYSTEM_PROMPT if system is None else system)
    prompt, letters = rendered[names.index(row["name"])]
    ids = input_ids(tokenizer, prompt, len(row["images"]))
    if len(ids) > max_len:
        return None
    views = []
    for path in row["images"]:
        with Image.open(path) as im:
            views += image_views(im)
    return {"ids": ids, "views": views, "letters": letter_ids(tokenizer, letters),
            "target": row["target"], "label": row["label"], "qtype": row["qtype"], "dataset": row["dataset"]}


def _pixels(views) -> torch.Tensor:
    a = np.stack([np.asarray(v, dtype=np.float32) / 255.0 for v in views])
    return torch.from_numpy((a - IMAGE_MEAN) / IMAGE_STD).permute(0, 3, 1, 2)


def collate(items: List[Dict], pad_id: int) -> Dict:
    n = max(len(it["ids"]) for it in items)
    ids = torch.full((len(items), n), pad_id, dtype=torch.long)
    mask = torch.zeros((len(items), n), dtype=torch.long)
    for i, it in enumerate(items):
        ids[i, : len(it["ids"])] = torch.tensor(it["ids"])
        mask[i, : len(it["ids"])] = 1
    out = {"input_ids": ids, "attention_mask": mask, "last": mask.sum(1) - 1,
           "letters": [it["letters"] for it in items], "target": [it["target"] for it in items],
           "label": [it["label"] for it in items], "qtype": [it["qtype"] for it in items],
           "dataset": [it["dataset"] for it in items]}
    n_views = max(len(it["views"]) for it in items)
    if n_views:
        pv = torch.zeros((len(items), n_views, 3, 512, 512))
        pm = torch.zeros((len(items), n_views, 512, 512), dtype=torch.bool)
        for i, it in enumerate(items):
            if it["views"]:
                pv[i, : len(it["views"])] = _pixels(it["views"])
                pm[i, : len(it["views"])] = True
        out["pixel_values"], out["pixel_attention_mask"] = pv, pm
    return out


class RowItems(torch.utils.data.Dataset):
    def __init__(self, rows, tokenizer, max_len: int = 2048, system: Optional[str] = None):
        self.rows, self.tokenizer, self.max_len, self.system = rows, tokenizer, max_len, system

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        try:
            return encode_row(self.rows[i], self.tokenizer, self.system, self.max_len)
        except (OSError, ValueError):  # an unreadable image, or a prompt Ollama would refuse
            return None


def _collate_skip(items, pad_id):
    items = [it for it in items if it is not None]
    return collate(items, pad_id) if items else None


def letter_logits(model, batch: Dict, device) -> List[torch.Tensor]:
    """Each row's next-token logits over its candidate letters, read at its last prompt token."""
    kw = {k: batch[k].to(device) for k in ("input_ids", "attention_mask")}
    if "pixel_values" in batch:
        kw["pixel_values"] = batch["pixel_values"].to(device, next(model.parameters()).dtype)
        kw["pixel_attention_mask"] = batch["pixel_attention_mask"].to(device)
    h = model.model(**kw).last_hidden_state
    last = h[torch.arange(h.shape[0], device=device), batch["last"].to(device)]
    logits = model.lm_head(last).float()
    return [logits[i, torch.tensor(c, device=device)] for i, c in enumerate(batch["letters"])]


def letter_loss(model, batch: Dict, device) -> torch.Tensor:
    losses = []
    for z, t in zip(letter_logits(model, batch, device), batch["target"]):
        t = torch.tensor(t, dtype=torch.float32, device=device)
        losses.append(-(t / t.sum() * F.log_softmax(z, -1)).sum())
    return torch.stack(losses).mean()


@torch.no_grad()
def collect(model, rows: Sequence[Dict], tokenizer, batch_size: int = 16, num_workers: int = 4, device=None,
            max_len: int = 2048) -> List[Dict]:
    """Records for ``laya.vlm_train.metrics_from`` (letter logits in option order, label, target, type, dataset)."""
    device = torch.device(device or next(model.parameters()).device)
    model.eval()
    loader = torch.utils.data.DataLoader(RowItems(rows, tokenizer, max_len), batch_size=batch_size,
                                         num_workers=num_workers,
                                         collate_fn=lambda b: _collate_skip(b, tokenizer.pad_token_id))
    out = []
    for batch in loader:
        if batch is None:
            continue
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            zs = letter_logits(model, batch, device)
        for i, z in enumerate(zs):
            out.append({"logits": z.cpu(), "label": batch["label"][i], "qtype": batch["qtype"][i],
                        "target": torch.tensor(batch["target"][i]), "dataset": batch["dataset"][i]})
    return out


def fit_temperature(records: Sequence[Dict]) -> float:
    """One temperature for every question type (Ollama has no per-type scaling), by LBFGS on the NLL."""
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = torch.stack([-(r["target"] / r["target"].sum() * F.log_softmax(r["logits"] / log_t.exp(), -1)).sum()
                            for r in records]).mean()
        loss.backward()
        return loss

    opt.step(closure)
    return float(torch.clamp(log_t.exp(), 0.1, 10.0))


def fold_temperature(model, t: float) -> None:
    """Divide every logit by ``t`` inside the model: ``lm_head`` has no bias, so scaling its weight does it."""
    with torch.no_grad():
        model.lm_head.weight.div_(t)


def train(model, rows: Sequence[Dict], tokenizer, steps: int, batch_size: int = 16, lr: float = 3e-5,
          warmup: int = 200, device=None, num_workers: int = 8, log_every: int = 50, max_minutes: float = 0,
          eval_fn=None, eval_every: int = 0, seed: int = 0, max_len: int = 2048) -> Dict:
    """AdamW with linear warmup and cosine decay on the letter cross-entropy, bf16 autocast on CUDA."""
    device = torch.device(device or next(model.parameters()).device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, steps)))))
    g = torch.Generator().manual_seed(seed)
    sampler = torch.utils.data.RandomSampler(rows, replacement=True, num_samples=steps * batch_size, generator=g)
    loader = torch.utils.data.DataLoader(RowItems(rows, tokenizer, max_len), batch_size=batch_size, sampler=sampler,
                                         num_workers=num_workers, persistent_workers=num_workers > 0,
                                         collate_fn=lambda b: _collate_skip(b, tokenizer.pad_token_id))
    t0, step, losses = time.time(), 0, []
    model.train()
    for batch in loader:
        if batch is None:
            continue
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            loss = letter_loss(model, batch, device)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        step += 1
        losses.append(float(loss))
        if step % log_every == 0:
            print("step %d loss %.4f lr %.2e %.1f min" % (step, np.mean(losses[-log_every:]), sched.get_last_lr()[0],
                                                          (time.time() - t0) / 60), flush=True)
        if eval_fn and eval_every and step % eval_every == 0:
            eval_fn(step)
            model.train()
        if step >= steps or (max_minutes and time.time() - t0 > max_minutes * 60):
            break
    return {"steps": step, "minutes": (time.time() - t0) / 60, "final_loss": float(np.mean(losses[-log_every:]))}


def save_for_gguf(model, processor, out_dir: str, temperature: float, provenance: Dict) -> None:
    """A Hugging Face model directory ``convert_hf_to_gguf.py`` turns into the GGUF text model and its mmproj.
    The image processor's longest edge is set to 512, so llama.cpp encodes one tile plus the overview, as in
    training (``laya.ollama.image_views``)."""
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=True)
    processor.image_processor.size = {"longest_edge": 512}
    processor.image_processor.max_image_size = {"longest_edge": 512}
    processor.save_pretrained(out_dir)
    with open(os.path.join(out_dir, "laya_ollama.json"), "w") as f:
        json.dump({"temperature_folded": temperature, **provenance}, f, indent=1)
