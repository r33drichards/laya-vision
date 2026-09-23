"""Export a laya-vision SmolVLM checkpoint to ONNX for the browser demo in ``web-demo/``.

Three graphs, split where the browser wants to reuse work:

* ``vision.onnx``: ``pixel_values [n, 3, S, S]`` -> ``image_features [n, image_seq_len, d]``, the vision tower plus
  the connector (pixel shuffle + projection). Run once per image, reused for every question.
* ``text.onnx``: ``input_ids [1, L]``, ``image_features [n, image_seq_len, d]``, ``option_span [2]`` ->
  ``last_hidden_state [1, L, d]``. Token embeddings, the image merge (each ``<image>`` token takes the next image
  feature row, as ``Idefics3Model.inputs_merger`` does), and the language model without its LM head. With
  ``option_attention="block"`` the graph builds the same additive 4D mask as ``laya.vlm.option_block_mask`` from
  ``option_span = (start, end)``; for a causal checkpoint the span is ignored. One sequence at a time, no padding.
* ``head.onnx``: ``hidden [1, L, d]``, ``marker_pos [K]``, ``qtype [1]`` -> ``logits [K]``, ``act_logits [n_act]``:
  type embedding, the head transformer layers, the scorer and the act head, exactly ``VLMDecisionModel.forward``
  after the backbone for a single unpadded row.

Only the ``"terminator"`` readout (SmolVLM, SmolVLM2) is supported; ModernVBERT's ``"mask"`` readout would need its
own sequence builder in the page. The vision tower is rebuilt around fixed position ids (``arange``), which is what
Idefics3's ``bucketize`` produces for a full, unpadded ``image_size`` tile; the export asserts that.

Next to the graphs it writes ``laya_web.json`` (everything the page needs to rebuild the token sequence and turn
logits into answers: config, temperatures, token ids, text templates, file sizes and hashes) and copies the
tokenizer files. ``--quantize`` adds variants next to the fp32 graphs:

* ``fp16``: weights and activations in float16 (``keep_io_types``), for WebGPU, with every normalisation kept in
  float32 (``rmsnorm_in_fp32``; LayerNormalization via the converter's block list): they square residuals of a few
  thousand, which overflows float16.
* ``q8`` / ``q4``: weight-only 8-/4-bit ``MatMulNBits`` (block 32, symmetric), activations stay float; for WASM.

Dynamic int8 (``quantize_dynamic``, int8 activations per tensor) was tried and dropped: on the validation inputs it
moved probabilities by up to 0.62 and changed the top answer on 4 of 9 questions. ``--validate`` compares the ONNX
pipeline, run with onnxruntime on CPU, against ``VLMAgent.predict`` on a few fixed inputs, writes
``validation.json``, and exits non-zero if fp32 is past ``--tol``.

    python scripts/export_onnx.py thaitea/laya-vision --out web-demo/models/laya-vision --quantize fp16,q8,q4 --validate
"""
import argparse
import gc
import hashlib
import json
import math
import os
import sys
import time
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from laya.common import QTYPES, render_options  # noqa: E402
from laya.vlm import (OPTION_BULLET, OPTION_END, PREFIX_TEXT, QUESTION_TEXT, VLMAgent,  # noqa: E402
                      build_vlm_inputs, option_block_mask, vlm_prefix)

WEB_CONFIG = "laya_web.json"
QUANT_KINDS = ("fp16", "q8", "q4")
FORMAT_VERSION = 1


class VisionExport(nn.Module):
    """``pixel_values [n, 3, S, S]`` -> ``[n, image_seq_len, d]`` for full ``S``-pixel tiles (no pixel padding)."""

    def __init__(self, enc: nn.Module):
        super().__init__()
        self.vm, self.connector = enc.vision_model, enc.connector
        emb = self.vm.embeddings
        self.register_buffer("pos_ids", torch.arange(emb.num_patches_per_side ** 2)[None], persistent=False)

    def forward(self, pixel_values):
        emb = self.vm.embeddings
        x = emb.patch_embedding(pixel_values).flatten(2).transpose(1, 2) + emb.position_embedding(self.pos_ids)
        x = self.vm.encoder(inputs_embeds=x, attention_mask=None).last_hidden_state
        return self.connector(self.vm.post_layernorm(x))


class TextExport(nn.Module):
    """Embeddings + image merge + language model (no LM head) for one unpadded sequence."""

    def __init__(self, enc: nn.Module, image_token_id: int, block: bool):
        super().__init__()
        self.text_model = enc.text_model
        self.image_token_id = image_token_id
        self.block = block

    def forward(self, input_ids, image_features, option_span):
        emb = self.text_model.embed_tokens(input_ids)  # [1, L, d]
        d = emb.size(-1)
        is_img = input_ids == self.image_token_id
        # the k-th <image> token takes feature row k; a zero row keeps the gather valid when there is no image
        rows = torch.cat([image_features.reshape(-1, d).to(emb.dtype), emb.new_zeros(1, d)], 0)
        idx = (torch.cumsum(is_img.long(), -1) - 1).clamp(min=0)
        idx = torch.where(is_img, idx, torch.full_like(idx, rows.size(0) - 1))
        emb = torch.where(is_img[..., None], rows[idx], emb)
        att = torch.ones_like(input_ids)
        span = option_span[None] if self.block else torch.zeros_like(option_span)[None]
        mask = option_block_mask(att, span, emb.dtype)
        return self.text_model(inputs_embeds=emb, attention_mask=mask, use_cache=False).last_hidden_state


def _encoder_layer(layer: nn.TransformerEncoderLayer, x: torch.Tensor) -> torch.Tensor:
    """``layer(x)`` for a pre-norm ``nn.TransformerEncoderLayer`` in eval mode, one unpadded row, written out so the
    trace keeps the sequence length symbolic (``nn.MultiheadAttention`` bakes the traced length into a reshape)."""
    import torch.nn.functional as F

    sa = layer.self_attn
    h, d = sa.num_heads, x.size(-1)
    q, k, v = F.linear(layer.norm1(x), sa.in_proj_weight, sa.in_proj_bias).chunk(3, -1)
    q, k, v = (t.reshape(1, -1, h, d // h).transpose(1, 2) for t in (q, k, v))
    att = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(d // h), -1) @ v
    x = x + sa.out_proj(att.transpose(1, 2).reshape(1, -1, d))
    return x + layer.linear2(layer.activation(layer.linear1(layer.norm2(x))))


class HeadExport(nn.Module):
    """``VLMDecisionModel.forward`` after the backbone, for a single row with every marker real."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.type_emb, self.head, self.scorer, self.act_head = model.type_emb, model.head, model.scorer, model.act_head

    def forward(self, hidden, marker_pos, qtype):
        h = hidden.float() + self.type_emb(qtype)[:, None, :]
        if self.head is not None:
            for layer in self.head.layers:
                h = _encoder_layer(layer, h)
        m = h[0, marker_pos]  # [K, d]
        logits = self.scorer(m).squeeze(-1)
        p = torch.softmax(logits, -1)
        k = torch.full((), 1.0) * logits.size(0)
        k = k.clamp(min=2.0)
        ent = -(p * torch.log(p.clamp_min(1e-9))).sum(-1) / torch.log(k)
        top2 = p.topk(2, -1).values
        feats = torch.stack([top2[0], top2[0] - top2[1], ent, k / 255.0], -1)
        act = self.act_head(torch.cat([h[0, -1], feats], -1))  # the last token: the final option terminator
        return logits, act


# ---------------------------------------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------------------------------------


def _export(module: nn.Module, args, path: str, input_names, output_names, dynamic_axes, opset: int):
    module.eval()
    with torch.no_grad():
        torch.onnx.export(module, args, path, input_names=input_names, output_names=output_names,
                          dynamic_axes=dynamic_axes, opset_version=opset, dynamo=False, do_constant_folding=True)


def _sample_inputs(agent: VLMAgent):
    """A realistic single-image sequence for tracing."""
    from PIL import Image

    img = Image.fromarray((np.arange(64 * 48 * 3, dtype=np.uint8).reshape(48, 64, 3) * 7) % 255)
    q = VLMAgent._to_internal({"type": "choice", "instructions": "pick", "criteria": ["a", "b", "c"]})
    prefix = vlm_prefix(agent.processor, [img], agent.prep)
    it = build_vlm_inputs(agent.processor, {"image": img, "note": "x"}, q, prefix=prefix)
    return prefix, it


def export_all(agent: VLMAgent, out: str, opset: int = 18, parts=("vision", "text", "head")) -> Dict[str, Any]:
    model = agent.model.float().eval()
    enc = model.encoder
    if model.readout != "terminator":
        raise SystemExit("only the 'terminator' readout (SmolVLM) is exportable; this checkpoint uses %r"
                         % model.readout)
    if agent.prep.split_edge:
        raise SystemExit("image splitting (image_split_edge) is not supported by the browser export")
    os.makedirs(out, exist_ok=True)
    prefix, it = _sample_inputs(agent)
    pv = prefix["pixel_values"].float()

    vis = VisionExport(enc).eval()
    with torch.no_grad():
        ref = model.encode_images(pv, prefix["pixel_attention_mask"])
        got = vis(pv)
    err = float((ref - got).abs().max())
    assert err < 1e-4, "fixed position ids disagree with Idefics3's bucketize path (%.2e)" % err
    if "vision" in parts:
        _export(vis, (pv,), os.path.join(out, "vision.onnx"), ["pixel_values"], ["image_features"],
                {"pixel_values": {0: "n_images"}, "image_features": {0: "n_images"}}, opset)

    image_token_id = int(enc.config.image_token_id)
    text = TextExport(enc, image_token_id, model.option_attention == "block").eval()
    ids = torch.tensor([it["ids"]])
    span = torch.tensor(it["option_span"])
    if "text" in parts:
        _export(text, (ids, got, span), os.path.join(out, "text.onnx"), ["input_ids", "image_features", "option_span"],
                ["last_hidden_state"], {"input_ids": {1: "seq"}, "image_features": {0: "n_images"},
                                        "last_hidden_state": {1: "seq"}}, opset)

    head = HeadExport(model).eval()
    with torch.no_grad():  # the wrappers must reproduce the model's own forward
        hid = text(ids, got, span)
        ref_l, ref_a = head.forward(hid, torch.tensor(it["markers"]), torch.tensor([0]))
        b = {"input_ids": ids, "attention_mask": torch.ones_like(ids), "marker_pos": torch.tensor([it["markers"]]),
             "marker_mask": torch.ones(1, len(it["markers"]), dtype=torch.bool), "qtype": torch.tensor([0])}
        lg, act = model(**b, image_hidden_states=got, option_span=span[None])
    drift = max(float((lg[0] - ref_l).abs().max()), float((act[0] - ref_a).abs().max()))
    assert drift < 1e-4, "text/head wrappers drifted from the model's forward (%.2e)" % drift
    if "head" in parts:
        _export(head, (hid, torch.tensor(it["markers"]), torch.tensor([0])), os.path.join(out, "head.onnx"),
                ["hidden", "marker_pos", "qtype"], ["logits", "act_logits"],
                {"hidden": {1: "seq"}, "marker_pos": {0: "k"}, "logits": {0: "k"}}, opset)
    return {"image_token_id": image_token_id}


def rmsnorm_in_fp32(model):
    """Put every decomposed RMSNorm of a float16-converted graph back on float32 arithmetic, as Hugging Face's
    ``LlamaRMSNorm`` does under half precision: upcast, square, mean, rsqrt, scale, then cast back before the weight.

    SmolVLM's residual stream reaches a few thousand in the last layers (about 2.5e3 before the final norm), so
    ``x**2`` is ~6e6, far past float16's 65504. The converter retargets the norm's upcast ``Cast(to=float)`` to
    float16; the mean is then inf, ``1/sqrt(inf)`` is 0, every hidden state comes out 0 and every option gets the
    same logit (exactly uniform probabilities on WebGPU/Metal). onnxruntime's CPU backend hides this by running
    ``Pow``/``ReduceMean`` in float32, so ``--validate`` passed; ``tests/test_web_demo.py`` checks it with real
    float16 arithmetic. (Blocking the nodes in the converter does not help: it still casts the edges between
    blocked nodes to float16.)
    """
    import onnx
    from onnx import numpy_helper

    nodes = model.graph.node
    prefixes = {n.name.rsplit("/", 1)[0] + "/" for n in nodes if n.op_type == "Pow" and "norm" in n.name}
    upcasts = []
    for n in nodes:
        pre = n.name.rsplit("/", 1)[0] + "/"
        if pre not in prefixes:
            continue
        if n.op_type == "Cast" and n.name == pre + "Cast":  # the upcast; Cast_1 (back to the input dtype) stays
            # The tracer shares this Cast's output with the residual Add (in the float32 graph it is a no-op), so
            # add a float32 twin that only the norm's own nodes read instead of retargeting it.
            f32 = n.output[0] + "_fp32"
            upcasts.append(onnx.helper.make_node("Cast", [n.input[0]], [f32], name=pre + "Cast_fp32",
                                                 to=onnx.TensorProto.FLOAT))
            for c in nodes:
                if c.name.startswith(pre) and c is not n:
                    for i, name in enumerate(c.input):
                        if name == n.output[0]:
                            c.input[i] = f32
        elif n.op_type == "Constant":
            for a in n.attribute:
                if a.name == "value" and a.t.data_type == onnx.TensorProto.FLOAT16:
                    a.t.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(a.t).astype(np.float32), a.t.name))
    assert prefixes or not any(n.op_type == "Pow" for n in nodes), "RMSNorm pattern not found"
    nodes.extend(upcasts)
    del model.graph.value_info[:]  # the converter's float16 annotations on these edges are now wrong; re-inferred
    return model


def quantize(out: str, kinds: List[str]):
    for kind in kinds:
        for name in ("vision", "text", "head"):
            src, dst = os.path.join(out, name + ".onnx"), os.path.join(out, "%s_%s.onnx" % (name, kind))
            if kind == "fp16":
                import onnx
                from onnxruntime.transformers.float16 import DEFAULT_OP_BLOCK_LIST, convert_float_to_float16

                # LayerNormalization (the vision tower's, the head's) computes x**2 over residuals of ~2e3: keep the
                # fused op in float32 (a single op, so only its input and output are cast, and those fit in fp16).
                # In true float16 arithmetic it took the vision features 57 off (of 93); in float32, 0.09.
                m = convert_float_to_float16(onnx.load(src), keep_io_types=True,
                                             op_block_list=list(DEFAULT_OP_BLOCK_LIST) + ["LayerNormalization"])
                m = rmsnorm_in_fp32(m)
                onnx.save_model(m, dst)
            elif kind in ("q4", "q8"):
                import onnx
                from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer

                q = MatMulNBitsQuantizer(onnx.load(src), bits=int(kind[1]), block_size=32, is_symmetric=True)
                q.process()
                q.model.save_model_to_file(dst, use_external_data_format=False)
            else:
                raise SystemExit("unknown --quantize kind %r (%s)" % (kind, ", ".join(QUANT_KINDS)))


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sequence_config(processor, prep, max_len: int = 1024, head_max_len: int = 256) -> Dict[str, Any]:
    """What ``web-demo/laya.js`` needs to rebuild ``build_vlm_inputs``: templates, token ids, budgets, image size.
    Shared by the export and by the committed parity fixture (``tests/test_web_demo.py``)."""
    tok = processor.tokenizer
    enc = lambda s: tok(s, add_special_tokens=False)["input_ids"]  # noqa: E731
    end = enc(OPTION_END)
    assert len(end) == 1, "option terminator must be a single token"
    glob = getattr(processor, "global_image_tag", None) or processor.global_image_token
    return {
        "max_len": int(max_len),
        "head_max_len": int(head_max_len),
        "qtypes": QTYPES,
        "image": {"size": prep.image_size, "seq_len": prep.image_seq_len, "stage1_longest_edge": 2048,
                  "mean": 0.5, "std": 0.5, "resample": "lanczos3-antialias, two hops, uint8 round+clamp after each"},
        "text": {"prefix": PREFIX_TEXT, "question": QUESTION_TEXT, "option_bullet": OPTION_BULLET,
                 "option_end": OPTION_END, "fake_image_token": processor.fake_image_token, "global_image_token": glob,
                 "image_token": processor.image_token, "strip_from_instructions": "<end_of_utterance>"},
        "token_ids": {"image": int(tok.convert_tokens_to_ids(processor.image_token)), "option_end": end[0],
                      "pad": tok.pad_token_id},
    }


def write_config(agent: VLMAgent, out: str, source: str, ids: Dict[str, Any]) -> Dict[str, Any]:
    agent.processor.tokenizer.save_pretrained(os.path.join(out, "tokenizer"))
    files = {}
    for f in sorted(os.listdir(out)):
        if f.endswith(".onnx"):
            files[f] = {"bytes": os.path.getsize(os.path.join(out, f)), "sha256": _sha256(os.path.join(out, f))}
    cfg = agent.cfg
    web = {
        "format_version": FORMAT_VERSION,
        "source": source,
        "backbone": cfg["backbone"],
        "readout": agent.model.readout,
        "option_attention": agent.model.option_attention,
        "temperature": [float(t) for t in agent.temperature],
        "temperature_by_options": {k: float(v) for k, v in agent.temperature_by_options.items()},
        "n_act": int(cfg.get("n_act", 2)),
        "hidden_size": int(agent.model.encoder.config.text_config.hidden_size),
        "files": files,
        "tokenizer": ["tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json"],
    }
    web.update(sequence_config(agent.processor, agent.prep, cfg.get("max_len", 1024), cfg.get("head_max_len", 256)))
    assert web["token_ids"]["image"] == ids["image_token_id"], "tokenizer and model disagree on the <image> id"
    with open(os.path.join(out, WEB_CONFIG), "w") as f:
        json.dump(web, f, indent=2)
    return web


# ---------------------------------------------------------------------------------------------------------
# Validation: the ONNX pipeline in onnxruntime against VLMAgent.predict
# ---------------------------------------------------------------------------------------------------------


VALIDATION_QUESTIONS = {
    "damage": {"type": "score", "instructions": "How much damage does the item show?",
               "criteria": ["none", "cosmetic: scratches or dents", "functional: parts broken or missing",
                            "destroyed"]},
    "category": {"type": "choice", "instructions": "What kind of item is this?",
                 "criteria": ["electronics", "clothing", "furniture", "food", "other"]},
    "outdoors": {"type": "noul", "instructions": "Was the photo taken outdoors?"},
}


def validation_images():
    """Deterministic test images: a smooth gradient with shapes, and a noisy non-square frame."""
    from PIL import Image, ImageDraw

    a = Image.new("RGB", (640, 480))
    px = np.zeros((480, 640, 3), dtype=np.uint8)
    px[..., 0] = np.linspace(0, 255, 640)[None]
    px[..., 1] = np.linspace(255, 0, 480)[:, None]
    px[..., 2] = 90
    a = Image.fromarray(px)
    d = ImageDraw.Draw(a)
    d.rectangle([100, 120, 300, 360], fill=(30, 200, 40))
    d.ellipse([380, 60, 600, 280], fill=(240, 240, 20))
    rng = np.random.RandomState(0)
    b = Image.fromarray(rng.randint(0, 255, (210, 160, 3), dtype=np.uint8))
    return {"shapes": a, "noise": b}


class OnnxPipeline:
    def __init__(self, out: str, suffix: str = ""):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.log_severity_level = 3
        so.enable_cpu_mem_arena = False  # the default arena holds on to peak memory; the host may be small
        mk = lambda n: ort.InferenceSession(os.path.join(out, "%s%s.onnx" % (n, suffix)), so,  # noqa: E731
                                            providers=["CPUExecutionProvider"])
        self.vision, self.text, self.head = mk("vision"), mk("text"), mk("head")

    def predict(self, agent: VLMAgent, state, questions):
        """Mirror of ``VLMAgent.predict`` (one permutation) with the three graphs; sequences from Python."""
        from laya.vlm import split_state

        images, _ = split_state(state)
        prefix = vlm_prefix(agent.processor, images, agent.prep)
        d = agent.cfg["hidden_size"]
        if images:
            feats = self.vision.run(None, {"pixel_values": prefix["pixel_values"].float().numpy()})[0]
        else:
            feats = np.zeros((0, agent.prep.image_seq_len, d), np.float32)
        res = {}
        for qid, qdef in questions.items():
            q = VLMAgent._to_internal(qdef)
            it = build_vlm_inputs(agent.processor, state, q, agent.cfg.get("max_len", 1024),
                                  agent.cfg.get("head_max_len", 256), prefix=prefix)
            hid = self.text.run(None, {"input_ids": np.array([it["ids"]], np.int64), "image_features": feats,
                                       "option_span": np.array(it["option_span"], np.int64)})[0]
            lg, act = self.head.run(None, {"hidden": hid, "marker_pos": np.array(it["markers"], np.int64),
                                           "qtype": np.array([QTYPES[q["t"]]], np.int64)})
            k = len(render_options(q))
            from laya.common import temp_bucket

            t = agent.temperature_by_options.get(temp_bucket(QTYPES[q["t"]], k), agent.temperature[QTYPES[q["t"]]])
            z = lg[:k] / max(1e-3, float(t))
            p = np.exp(z - z.max())
            p = p / p.sum()
            a = np.exp(act - act.max())
            res[qid] = {"probs": p, "act": float((a / a.sum())[0]), "raw_logits": lg}
        return res


def _torch_probs(agent: VLMAgent, state, questions):
    out = agent.predict(state, questions)["answers"]
    res = {}
    for qid, a in out.items():
        if a["type"] == "noul":
            p = np.array([1 - a["noul"], a["noul"]])
        else:
            p = np.array(list(a["probabilities"].values()))
        res[qid] = {"probs": p, "act": a["action"]["act_probability"]}
    return res


def _unrounded_torch(agent: VLMAgent, state, questions):
    """Unrounded probabilities straight from the model (``predict`` rounds to 4 places)."""
    from laya.common import temp_bucket
    from laya.vlm import collate_vlm, split_state

    images, _ = split_state(state)
    prefix = vlm_prefix(agent.processor, images, agent.prep)
    feats = None
    if images:
        feats = agent.model.encode_images(prefix["pixel_values"].float(), prefix["pixel_attention_mask"])
    res = {}
    for qid, qdef in questions.items():
        q = VLMAgent._to_internal(qdef)
        it = build_vlm_inputs(agent.processor, state, q, prefix=prefix)
        it["qtype"] = QTYPES[q["t"]]
        b = collate_vlm([it], agent.processor.tokenizer.pad_token_id, with_pixels=False)
        with torch.no_grad():
            lg, act = agent.model(b["input_ids"], b["attention_mask"], b["marker_pos"], b["marker_mask"], b["qtype"],
                                  image_hidden_states=feats, option_span=b["option_span"])
        k = len(render_options(q))
        t = agent.temperature_by_options.get(temp_bucket(QTYPES[q["t"]], k), agent.temperature[QTYPES[q["t"]]])
        z = lg[0, :k].numpy() / max(1e-3, float(t))
        p = np.exp(z - z.max())
        res[qid] = {"probs": p / p.sum(), "act": float(torch.softmax(act[0], -1)[0]), "raw_logits": lg[0, :k].numpy()}
    return res


def validate(agent: VLMAgent, out: str, variants: List[str], tol: float) -> Dict[str, Any]:
    imgs = validation_images()
    states = {
        "shapes+note": {"image": imgs["shapes"], "note": "customer says it arrived broken"},
        "noise": {"image": imgs["noise"]},
        "text-only": {"note": "a plain text state with no image"},
    }
    report, ok = {}, True
    refs = {}
    for sname, state in states.items():
        refs[sname] = _unrounded_torch(agent, state, VALIDATION_QUESTIONS)
        rounded = _torch_probs(agent, state, VALIDATION_QUESTIONS)
        for qid in VALIDATION_QUESTIONS:  # the unrounded path is predict's, up to its 4-place rounding
            assert np.abs(refs[sname][qid]["probs"] - rounded[qid]["probs"]).max() < 1e-4, (sname, qid)
    # only the processor is needed from here on: drop the PyTorch weights before loading onnxruntime sessions
    agent.cfg["hidden_size"] = agent.model.encoder.config.text_config.hidden_size
    agent.model = None
    gc.collect()
    for v in variants:
        pipe = OnnxPipeline(out, "" if v == "fp32" else "_" + v)
        for sname, state in states.items():
            ref = refs[sname]
            t0 = time.time()
            got = pipe.predict(agent, state, VALIDATION_QUESTIONS)
            report.setdefault("seconds", {})["%s/%s" % (v, sname)] = round(time.time() - t0, 2)
            for qid in VALIDATION_QUESTIONS:
                dp = float(np.abs(ref[qid]["probs"] - got[qid]["probs"]).max())
                dl = float(np.abs(ref[qid]["raw_logits"] - got[qid]["raw_logits"]).max())
                report.setdefault(v, []).append({
                    "state": sname, "q": qid, "max_abs_prob_diff": round(dp, 6), "max_abs_logit_diff": round(dl, 6),
                    "same_argmax": bool(ref[qid]["probs"].argmax() == got[qid]["probs"].argmax()),
                    "act_diff": round(abs(ref[qid]["act"] - got[qid]["act"]), 6),
                    "p_torch": [round(float(x), 5) for x in ref[qid]["probs"]],
                    "p_onnx": [round(float(x), 5) for x in got[qid]["probs"]]})
                if v == "fp32" and dp > tol:
                    ok = False
        del pipe
        gc.collect()
    return {"ok": ok, "tolerance_fp32": tol, "results": report}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("checkpoint", nargs="?", default="thaitea/laya-vision", help="checkpoint dir or Hub id")
    ap.add_argument("--out", default="web-demo/models/laya-vision")
    ap.add_argument("--quantize", default="", help="comma list of fp16, int8")
    ap.add_argument("--opset", type=int, default=18)
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--tol", type=float, default=1e-3, help="max |p_onnx - p_torch| for the fp32 graphs")
    ap.add_argument("--parts", default="vision,text,head", help="graphs to (re)export; the rest are left as they are")
    ap.add_argument("--variants", default="", help="validate only these of fp32, fp16, int8 (default: all present)")
    ap.add_argument("--skip-export", action="store_true", help="only (re)validate existing files")
    a = ap.parse_args(argv)

    agent = VLMAgent(a.checkpoint, device="cpu", dtype="fp32")
    kinds = [k for k in a.quantize.split(",") if k]
    if not a.skip_export:
        ids = export_all(agent, a.out, a.opset, a.parts.split(","))
        quantize(a.out, kinds)
        web = write_config(agent, a.out, a.checkpoint, ids)
        for f, meta in web["files"].items():
            print("%-18s %8.1f MB" % (f, meta["bytes"] / 1e6))
    if a.validate:
        present = ["fp32"] + [k for k in QUANT_KINDS if os.path.exists(os.path.join(a.out, "text_%s.onnx" % k))]
        if a.variants:
            present = [v for v in present if v in a.variants.split(",")]
        rep = validate(agent, a.out, present, a.tol)
        path = os.path.join(a.out, "validation.json")
        if os.path.exists(path):  # validating one variant at a time keeps the others' results
            with open(path) as f:
                old = json.load(f)
            old["results"].update({k: v for k, v in rep["results"].items() if k != "seconds"})
            old["results"].setdefault("seconds", {}).update(rep["results"].get("seconds", {}))
            old["ok"] = rep["ok"] and old.get("ok", True)
            rep = dict(old, tolerance_fp32=a.tol)
        with open(path, "w") as f:
            json.dump(rep, f, indent=2)
        for v in present:
            rows = rep["results"][v]
            print("%-5s max|dp| %.2e  max|dlogit| %.2e  argmax agree %d/%d" % (
                v, max(r["max_abs_prob_diff"] for r in rows), max(r["max_abs_logit_diff"] for r in rows),
                sum(r["same_argmax"] for r in rows), len(rows)))
        if not rep["ok"]:
            raise SystemExit("fp32 ONNX differs from PyTorch by more than %g" % a.tol)


if __name__ == "__main__":
    main()
