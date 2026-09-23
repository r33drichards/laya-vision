"""Test-time search for game play: pick each frame's action by looking ahead in a simulator.

The model plays a game by answering one ``choice`` question over the game's actions from the rendered frame.
With a simulator the next frames can be *looked at* before committing to a move: ``plan`` expands every legal
action (or a PUCT tree of them), scores the resulting frames with the model, and picks the action whose future
looks best. It works on a batch of environments in lockstep and scores every frame of a search step that it can in
one batched model forward (``score_states``).

Environment protocol (adapters for Maze, Snake and classic control implement it)::

    env.clone() -> env              # independent deep copy (its own RNG state included)
    env.step(action: str) -> (reward: float, done: bool)
    env.render() -> PIL.Image       # the frame the model sees
    env.actions: Sequence[str]      # option names, in the question's option order
    env.done: bool

The value of a frame ``V(s)`` in [0, 1]:

* with a value head (agent config ``"value_head": true``): ``sigmoid(value logit)``, the model's P(success from
  here);
* without one: the policy's max probability at ``s`` (how sure the policy is of what to do next: a weak proxy,
  but free);
* a terminal frame (``done`` after the step) is worth 0: nothing follows it, so a won or lost ending must be
  expressed in the step's reward (adapters: positive reward on success, zero or negative on failure).

``settings["leaf"]`` forces one of ``"value"``, ``"policy"`` or ``"zero"`` (``"auto"``, the default, is the first
available of value / policy).

Kinds (``settings["kind"]``):

* ``"lookahead"`` (``depth`` d, default 1): the full tree of ``len(actions) ** d`` action sequences per env. The
  backed-up value of taking ``a`` at node ``s`` is ``Q(s, a) = r + gamma * W(s')`` with ``W(s') = 0`` if ``s'``
  is terminal, ``V(s')`` at depth d, else ``max_a' Q(s', a')``. At the root the choice is
  ``argmax_a Q(root, a) + c_prior * log P(a | root)`` with the policy's (temperature-scaled, as ``predict``)
  probabilities, so the policy decides among actions whose futures look alike and search overrides it only when
  a future is clearly better (e.g. one move dies). One model forward per step scores the roots and every leaf
  of every env (split only when there are more than ``batch_size`` distinct frames).
* ``"puct"`` (``sims`` simulations, default 16): AlphaZero-style tree search per env,
  ``argmax_a Q(a) + c_puct * P(a) * sqrt(N) / (1 + n(a))`` with Q min-max normalised within each tree (rewards
  are not in [0, 1] in every game), unvisited actions valued at their parent's value, and the move with the most
  visits played (prior breaks ties). Each simulation descends every env's tree to one new leaf, and the new
  leaves of all envs are scored together: ``1 + sims`` forwards per step.

Identical frames (a bump into a wall renders the root again) are scored once per forward.
"""
import hashlib
import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

from .common import QTYPES, render_options

DEFAULTS = {
    "kind": "lookahead",
    "depth": 1,
    "gamma": 1.0,
    "c_prior": 0.1,
    "leaf": "auto",
    "batch_size": 64,
    "sims": 16,
    "c_puct": 1.5,
}


def _qdef(question: Dict) -> Dict:
    """A question definition, given either itself (``{"type", ...}``) or a one-entry ``{qid: qdef}`` dict."""
    if "type" in question:
        return question
    if len(question) != 1:
        raise ValueError("plan() takes one question, got %d: %s" % (len(question), list(question)))
    return next(iter(question.values()))


def _frame_key(img) -> bytes:
    a = np.asarray(img)
    return hashlib.blake2b(a.tobytes(), digest_size=16).digest() + repr(a.shape).encode()


@torch.no_grad()
def score_states(agent, images: Sequence, question: Dict, batch_size: int = 64,
                 stats: Optional[Dict] = None) -> Dict[str, Any]:
    """Score many frames under one question: ``{"probs": [N, k], "value": [N] or None}`` (numpy).

    Probabilities are in the question's option order with the checkpoint's temperature, exactly as
    ``VLMAgent.predict(n_permutations=1)`` gives them for ``{"image": img}``; ``value`` is ``sigmoid`` of the value
    head (None without one). Distinct frames are tokenized with the checkpoint's own preprocessing (either
    backend), collated, and run through the model in forwards of up to ``batch_size`` rows: vision tower and
    language model batched across frames. ``stats["forwards"]`` / ``["rows"]`` count what ran.
    """
    from .vlm import build_vlm_inputs, collate_vlm
    from .vlm_train import _to

    q = agent._to_internal(_qdef(question))
    k, qt = len(render_options(q)), QTYPES[q["t"]]
    keys = [_frame_key(im) for im in images]
    uniq: Dict[bytes, int] = {}
    distinct = []
    for key, im in zip(keys, images):
        if key not in uniq:
            uniq[key] = len(distinct)
            distinct.append(im)
    max_len, head_max_len = agent.cfg.get("max_len", 1024), agent.cfg.get("head_max_len", 256)
    model, pad = agent.model, agent.processor.tokenizer.pad_token_id
    t = max(1e-3, float(agent._checkpoint_temperature(qt, k)))
    probs, values = [], []
    for s in range(0, len(distinct), max(1, batch_size)):
        rows = []
        for im in distinct[s: s + batch_size]:
            it = build_vlm_inputs(agent.processor, {"image": im}, q, max_len, head_max_len, prep=agent.prep)
            it["qtype"] = qt
            rows.append(it)
        b = _to(collate_vlm(rows, pad), agent.device, model.encoder.dtype)
        logits, _ = model(b["input_ids"], b["attention_mask"], b["marker_pos"], b["marker_mask"], b["qtype"],
                          pixel_values=b["pixel_values"], pixel_attention_mask=b["pixel_attention_mask"],
                          option_span=b["option_span"], raw_pixels=b["raw_pixels"], image_mask=b["image_mask"])
        probs.append(torch.softmax(logits[:, :k].float() / t, -1).cpu().numpy())
        if model.last_value is not None:
            values.append(torch.sigmoid(model.last_value.float()).cpu().numpy())
        if stats is not None:
            stats["forwards"] = stats.get("forwards", 0) + 1
            stats["rows"] = stats.get("rows", 0) + len(rows)
    idx = [uniq[key] for key in keys]
    P = np.concatenate(probs)[idx] if probs else np.zeros((0, k))
    V = np.concatenate(values)[idx] if values else None
    return {"probs": P, "value": V}


def _leaf_values(scored: Dict[str, Any], leaf: str) -> np.ndarray:
    if leaf == "auto":
        leaf = "value" if scored["value"] is not None else "policy"
    if leaf == "value":
        if scored["value"] is None:
            raise ValueError("leaf='value' needs a checkpoint with a value head (config 'value_head': true)")
        return scored["value"]
    if leaf == "policy":
        return scored["probs"].max(-1)
    if leaf == "zero":
        return np.zeros(len(scored["probs"]))
    raise ValueError("leaf must be auto, value, policy or zero, got %r" % leaf)


class _Node:
    __slots__ = ("env", "reward", "done", "children", "value", "prior", "N", "W", "q_min", "q_max")

    def __init__(self, env, reward: float = 0.0, done: bool = False):
        self.env, self.reward, self.done = env, reward, done
        self.children: Dict[int, "_Node"] = {}
        self.value, self.prior = 0.0, None
        self.N = self.W = None


def _expand(node: _Node, a: int) -> _Node:
    env = node.env.clone()
    r, done = env.step(node.env.actions[a])
    child = _Node(env, float(r), bool(done))
    node.children[a] = child
    return child


def _lookahead(agent, roots: List[_Node], question, cfg, stats) -> List[int]:
    depth, gamma = int(cfg["depth"]), float(cfg["gamma"])
    if depth < 1:
        raise ValueError("lookahead depth must be >= 1")
    leaves: List[_Node] = []
    frontier = roots
    for d in range(depth):
        nxt = []
        for node in frontier:
            for a in range(len(node.env.actions)):
                child = _expand(node, a)
                if not child.done:
                    nxt.append(child)
        frontier = nxt
    leaves = frontier
    stats["children"] = stats.get("children", 0) + sum(len(r.env.actions) ** depth for r in roots)
    scored = score_states(agent, [n.env.render() for n in roots + leaves], question, cfg["batch_size"], stats)
    for n, v in zip(leaves, _leaf_values({"probs": scored["probs"][len(roots):],
                                          "value": None if scored["value"] is None else scored["value"][len(roots):]},
                                         cfg["leaf"])):
        n.value = float(v)

    def worth(n: _Node, d: int) -> float:  # W(s): 0 terminal, V at the horizon, else the best backed-up Q
        if n.done:
            return 0.0
        if d == depth:
            return n.value
        return max(c.reward + gamma * worth(c, d + 1) for c in n.children.values())

    out = []
    for i, root in enumerate(roots):
        prior = scored["probs"][i]
        q = np.array([root.children[a].reward + gamma * worth(root.children[a], 1) for a in range(len(prior))])
        score = q + float(cfg["c_prior"]) * np.log(prior + 1e-6)
        out.append(int(np.argmax(score)))
        stats.setdefault("q", []).append(q.tolist())
    return out


def _puct(agent, roots: List[_Node], question, cfg, stats) -> List[int]:
    gamma, c_puct, leaf = float(cfg["gamma"]), float(cfg["c_puct"]), cfg["leaf"]

    def evaluate(nodes: List[_Node]):
        if not nodes:
            return
        scored = score_states(agent, [n.env.render() for n in nodes], question, cfg["batch_size"], stats)
        for n, p, v in zip(nodes, scored["probs"], _leaf_values(scored, leaf)):
            n.prior, n.value = p.astype(np.float64), float(v)
            n.N, n.W = np.zeros(len(p)), np.zeros(len(p))

    evaluate(roots)
    bounds = [[math.inf, -math.inf] for _ in roots]  # per-tree min / max of the Q values seen

    def norm(i, q):
        lo, hi = bounds[i]
        return (q - lo) / (hi - lo) if hi > lo else 0.5

    for _ in range(int(cfg["sims"])):
        pending, paths = [], []
        for i, root in enumerate(roots):
            node, path = root, []
            while True:
                total = node.N.sum()
                q = np.where(node.N > 0, node.W / np.maximum(node.N, 1), node.value)
                u = np.array([norm(i, x) for x in q]) + c_puct * node.prior * math.sqrt(total + 1) / (1 + node.N)
                a = int(np.argmax(u))
                path.append((node, a))
                if a not in node.children:
                    child = _expand(node, a)
                    if not child.done:
                        pending.append(child)
                    break
                child = node.children[a]
                if child.done:
                    break
                node = child
            paths.append((i, path))
        evaluate(pending)
        for i, path in paths:
            last = path[-1][0].children[path[-1][1]]
            g = 0.0 if last.done else last.value
            for node, a in reversed(path):
                g = node.children[a].reward + gamma * g
                node.N[a] += 1
                node.W[a] += g
                bounds[i][0], bounds[i][1] = min(bounds[i][0], g), max(bounds[i][1], g)
    out = []
    for root in roots:
        out.append(int(np.lexsort((root.prior, root.N))[-1]))  # most visits, then the higher prior
        stats.setdefault("visits", []).append(root.N.tolist())
    return out


def plan(agent, envs: Sequence, question: Dict, settings: Optional[Dict] = None,
         stats: Optional[Dict] = None) -> List[str]:
    """One action per env (the batch is searched in lockstep); see the module docstring for the algorithms.

    ``question`` is the game's question (``{"action": qdef}`` or the qdef itself), whose options are
    ``env.actions`` in order. ``settings`` is ``agent.cfg["search"]``-style: ``{"kind": "lookahead", "depth": 1}``
    or ``{"kind": "puct", "sims": 16}``, plus the optional knobs in ``DEFAULTS``. An env that is already done is
    not searched and gets its first action. ``stats`` (a dict) receives ``forwards`` (model calls), ``rows``
    (distinct frames scored) and per-env ``q`` (lookahead root Q values) or ``visits`` (PUCT root visit counts).
    The envs themselves are never stepped: search runs on clones.
    """
    cfg = dict(DEFAULTS, **(settings or {}))
    stats = stats if stats is not None else {}
    k = len(render_options(agent._to_internal(_qdef(question))))
    live = [i for i, e in enumerate(envs) if not e.done]
    actions: List[str] = [e.actions[0] for e in envs]
    for i in live:
        if len(envs[i].actions) != k:
            raise ValueError("env %d has %d actions but the question has %d options" % (i, len(envs[i].actions), k))
    if not live:
        return actions
    roots = [_Node(envs[i]) for i in live]
    kind = cfg["kind"]
    if kind == "lookahead":
        picks = _lookahead(agent, roots, question, cfg, stats)
    elif kind in ("puct", "mcts"):
        picks = _puct(agent, roots, question, cfg, stats)
    else:
        raise ValueError("unknown search kind %r (lookahead or puct)" % kind)
    for i, a in zip(live, picks):
        actions[i] = envs[i].actions[a]
    return actions


__all__ = ["DEFAULTS", "plan", "score_states"]
