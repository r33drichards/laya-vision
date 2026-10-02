"""Best-of-N with a verifier: exact expected reward of reranking candidates by a score.

The setting: a group (one prompt, one image) has ``M`` candidate answers, each with a verifier score and a reward
(1/0 for right/wrong, or a graded rating). A best-of-N system draws ``N`` of the ``M`` candidates, keeps the one
the verifier scores highest and earns its reward. :func:`best_of_n` returns the **exact** expectation of that
reward over a uniformly random ``N``-subset, with exact score ties broken uniformly at random, instead of a Monte
Carlo estimate over sampled subsets, plus its two reference points on the same subsets:

* ``random``: pick one of the ``N`` at random (= the group's mean reward, whatever ``N`` is);
* ``oracle``: pick the best of the ``N`` by its reward (for 0/1 rewards this is pass@N, the chance that at least
  one of the ``N`` is correct).

How it works: sort the candidates into blocks of equal score, lowest first. The selected candidate comes from the
top block that the subset touches. With ``L`` candidates below a block of size ``m``, the subset's maximum lies
in that block with probability ``(C(L + m, N) - C(L, N)) / C(M, N)``, and given that, every member of the block is
equally likely to be the one picked (uniform tie-breaking is symmetric over the block), so the block contributes
its mean reward. The oracle is the same computation with the reward as the score. Ported from the CLM repo's
``evaluation/bon_eval.py`` (``best_of_n``), extended from 0/1 outcomes to any finite reward.

:func:`bon_curve` averages this over many groups for several ``N``. A group with fewer than ``N`` candidates uses
all of them (budget ``min(N, M)``) and is counted in ``short_groups``, so every group stays in the average.

The second half of the module uses Laya as the verifier: :func:`verifier_scores` asks a checkpoint, for each
candidate response to a question about an image, a ``noul`` ("is this response correct?") and the ``score``
helpfulness rubric the rubric-scored checkpoints were trained on (``laya.rubric``, VLFeedback's state layout
``"Question: ...\\n\\nResponse: ..."``), and returns P(true) and the expected level as two scores to rerank by.
"""
import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

#: the verifier's correctness question (``noul``: P(true) is the score)
CORRECT_QUESTION = {
    "type": "noul",
    "instructions": "Is the response a correct answer to the question about the image?",
    "criteria": {"true": "yes, the response answers the question correctly",
                 "false": "no, the response is wrong or does not answer the question"},
}


def _comb(n: int, k: int) -> int:
    return math.comb(n, k) if n >= k else 0


def _expected_top(pairs: Sequence[Tuple[float, float]], n: int) -> float:
    """E[reward of the top-keyed member of a uniform ``n``-subset], ties in the key broken uniformly.
    ``pairs`` is ``(key, reward)``."""
    blocks: Dict[float, List[float]] = {}
    for key, reward in pairs:
        blocks.setdefault(key, []).append(reward)
    total = _comb(len(pairs), n)
    out, lower = 0.0, 0
    for key in sorted(blocks):
        rewards = blocks[key]
        p = (_comb(lower + len(rewards), n) - _comb(lower, n)) / total
        out += p * sum(rewards) / len(rewards)
        lower += len(rewards)
    return out


def best_of_n(candidates: Sequence[Tuple[float, float]], n: int) -> Dict[str, float]:
    """``candidates`` = ``[(score, reward), ...]`` for one group -> the exact expected reward of best-of-``n``
    selection by score (``selected``), of a random pick (``random``) and of an oracle pick (``oracle``; pass@n for
    0/1 rewards). Scores and rewards must be finite; ``1 <= n <= len(candidates)``."""
    if not 1 <= n <= len(candidates):
        raise ValueError("invalid candidate budget N=%s for %d candidates" % (n, len(candidates)))
    pairs = []
    for score, reward in candidates:
        score, reward = float(score), float(reward)
        if not (math.isfinite(score) and math.isfinite(reward)):
            raise ValueError("finite scores and rewards are required, got (%r, %r)" % (score, reward))
        pairs.append((score, reward))
    rewards = [r for _, r in pairs]
    return {"selected": _expected_top(pairs, n),
            "random": sum(rewards) / len(rewards),
            "oracle": _expected_top([(r, r) for r in rewards], n)}


def bon_curve(groups: Iterable[Sequence[Tuple[float, float]]], ns: Sequence[int]) -> Dict[int, Dict[str, float]]:
    """Mean of :func:`best_of_n` over ``groups`` for each ``N`` in ``ns``, with ``gap_closed`` =
    ``(selected - random) / (oracle - random)`` (the share of the oracle's gain over random picking that the
    verifier recovers; ``None`` when the oracle gains nothing). Groups with fewer than ``N`` candidates are scored
    at budget ``min(N, M)`` and counted in ``short_groups``."""
    groups = [list(g) for g in groups]
    if not groups or any(not g for g in groups):
        raise ValueError("need at least one group, and no empty group")
    out = {}
    for n in ns:
        if n < 1:
            raise ValueError("N must be positive, got %r" % n)
        acc = {"selected": 0.0, "random": 0.0, "oracle": 0.0}
        short = 0
        for g in groups:
            short += len(g) < n
            r = best_of_n(g, min(n, len(g)))
            for k in acc:
                acc[k] += r[k]
        row = {k: v / len(groups) for k, v in acc.items()}
        gain = row["oracle"] - row["random"]
        row["gap_closed"] = (row["selected"] - row["random"]) / gain if gain > 1e-12 else None
        row.update(n_groups=len(groups), short_groups=short)
        out[n] = row
    return out


# -- Laya as the verifier -----------------------------------------------------------------------------------------

def candidate_state(image, question: str, response: str) -> Dict:
    """The state a candidate is judged in: the image, and the question and response as text, laid out as in the
    VLFeedback rubric records the score head was trained on."""
    return {"image": image, "context": "Question: %s\n\nResponse: %s" % (question, response)}


def verifier_questions() -> Dict[str, Dict]:
    """``{"correct": noul, "helpful": score}``: the two questions :func:`verifier_scores` asks per candidate."""
    from .rubric import rubric_question

    return {"correct": dict(CORRECT_QUESTION), "helpful": rubric_question("helpfulness")}


def verifier_scores(agent, image, question: str, responses: Sequence[str], **predict_kwargs) -> List[Dict[str, float]]:
    """One ``{"correct": P(true), "helpful": expected level 0-4}`` per response, from ``agent.predict`` on
    :func:`candidate_state` (one call per response, both questions in it). ``predict_kwargs`` go to ``predict``."""
    qs = verifier_questions()
    out = []
    for response in responses:
        ans = agent.predict(candidate_state(image, question, response), qs, **predict_kwargs)["answers"]
        out.append({"correct": float(ans["correct"]["noul"]), "helpful": float(ans["helpful"]["score"])})
    return out


def choice_candidates(q: Dict, label: int) -> Tuple[str, List[str], List[int]]:
    """A multiple-choice question in ``VLMAgent``'s internal form (``{"t": "choice", "ins", "crit"}``) and its
    label -> ``(question text, candidate responses, 0/1 rewards)``: every option becomes a candidate answer, the
    labelled one the only correct one. The response is the option as the model reads it (``name: description``)."""
    from .common import render_options

    if q["t"] != "choice":
        raise ValueError("choice questions only, got %r" % q["t"])
    options = render_options(q)
    if not 0 <= label < len(options):
        raise ValueError("label %d out of range for %d options" % (label, len(options)))
    return q["ins"], options, [int(i == label) for i in range(len(options))]


def vlfeedback_candidates(row: Dict, aspect: str = "Helpfulness", max_chars: int = 1200,
                          max_prompt_chars: int = 400) -> Tuple[str, List[str], List[int]]:
    """One raw VLFeedback row (``{"prompt", "completions": [{"response", "annotations"}, ...]}``) -> ``(prompt,
    responses, ratings 1-5 on aspect)``, clipped exactly as ``laya.rubric.vlfeedback_records`` clips its training
    records. Responses that are empty or have no valid rating are left out."""
    from .rubric import _completions, _rating, clip_text

    prompt = clip_text(str(row.get("prompt") or ""), max_prompt_chars)
    responses, ratings = [], []
    for c in _completions(row.get("completions")):
        response = clip_text(str(c.get("response") or ""), max_chars)
        r = _rating((c.get("annotations") or {}).get(aspect))
        if response and r is not None:
            responses.append(response)
            ratings.append(r)
    return prompt, responses, ratings


__all__ = ["best_of_n", "bon_curve", "candidate_state", "verifier_questions", "verifier_scores", "choice_candidates",
           "vlfeedback_candidates", "CORRECT_QUESTION"]
