"""The one place a (state, question) becomes model text, shared by data preparation, training, evaluation and
``predict`` so the text a checkpoint is trained on and the text it is served cannot drift apart.

The token-level framing (``laya.vlm.build_vlm_inputs``, ``laya.common.build_sequence``) takes its text from here:

* ``question_to_internal``: an API question ``{"type", "instructions", "criteria"}`` to the internal
  ``{"t", "ins", "crit"}`` both ``predict`` and ``laya.vlm_train.jsonl_example`` use (``VLMAgent._to_internal``
  and ``Agent._to_internal`` are this function).
* ``render_options`` / ``option_labels``: option texts and names in label order.
* ``serialize_state``: the non-image part of a state as text, in one of ``STATE_FORMATS``:
    - ``"json"`` (the default, and what every released checkpoint was trained on): ``json.dumps`` with
      ``ensure_ascii=False``; a string is kept as it is.
    - ``"prose"``: ``to_text``, the rendering of the sibling CLM repository (``clm.schema.to_text``): an object
      becomes ``key: value`` fields (top-level fields separated by a blank line, nested ones indented), an array
      one ``- item`` line per element. No quotes, braces or ``\\n`` escapes.
    - ``"text"``: the values only (``to_text`` of each top-level value, separated by a blank line), for a state
      whose keys are just wrappers such as ``{"context": ...}``.
  Only ``"json"`` reproduces the released checkpoints' input ids; the others are opt-in (``VLMAgent(...,
  state_format=...)``, recorded in ``vlm_agent_config.json`` so a checkpoint trained with one is served with it).
* ``record_state``: the state a prepared-dataset record (``{"image" | "images", "state_text"}``) becomes, the
  layout every checkpoint was trained on (``{"image": ..., "context": state_text}``); ``make_state`` builds the
  same layout for a caller of ``predict``.
* ``normalize_option_text``: the option normalisation data preparation applies (A-OKVQA options are lower-cased by
  ``laya.cauldron.parse_options`` so the answer matches); ``predict`` does not apply it unless asked.

Nothing here imports torch or a tokenizer, so it is cheap to import from the data-preparation jobs.
"""
import json
import os
from typing import Any, Dict, List, Optional, Sequence, Union

STATE_FORMATS = ("json", "prose", "text")
DEFAULT_STATE_FORMAT = "json"
#: the state key the text of a prepared record (``state_text``) is stored under, in training and in evaluation
CONTEXT_KEY = "context"
OPTION_CASES = ("lower", "title")


def check_state_format(state_format: Optional[str]) -> str:
    """``state_format`` validated (``None`` means ``DEFAULT_STATE_FORMAT``)."""
    fmt = state_format or DEFAULT_STATE_FORMAT
    if fmt not in STATE_FORMATS:
        raise ValueError("state_format must be one of %s, got %r" % (STATE_FORMATS, state_format))
    return fmt


def to_text(x: Any, indent: int = 0) -> str:
    """A string, number, object or array as plain text (``clm.schema.to_text``): an object becomes ``key: value``
    fields (top-level fields separated by a blank line, nested ones indented by two spaces), an array one
    ``- item`` line per element. Key order is preserved."""
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, bool):
        return "true" if x else "false"
    if isinstance(x, (int, float)):
        return str(x)
    pad = " " * indent
    if isinstance(x, dict):
        parts = []
        for k, v in x.items():
            if isinstance(v, (dict, list, tuple)) and v:
                parts.append("%s%s:\n%s" % (pad, k, to_text(v, indent + 2)))
            else:
                parts.append("%s%s: %s" % (pad, k, to_text(v)))
        return ("\n\n" if indent == 0 else "\n").join(parts)
    if isinstance(x, (list, tuple)):
        parts = []
        for v in x:
            if isinstance(v, (dict, list, tuple)) and v:
                parts.append("%s-\n%s" % (pad, to_text(v, indent + 2)))
            else:
                parts.append("%s- %s" % (pad, to_text(v)))
        return "\n".join(parts)
    return json.dumps(x, ensure_ascii=False)


def serialize_state(state: Union[str, dict, list], state_format: Optional[str] = None) -> str:
    """The text of a (non-image) state in ``state_format`` (see the module docstring). A string is returned as it
    is in every format."""
    fmt = check_state_format(state_format)
    if isinstance(state, str):
        return state
    if fmt == "json":
        return json.dumps(state, ensure_ascii=False)
    if fmt == "text" and isinstance(state, dict):
        return "\n\n".join(to_text(v) for v in state.values() if to_text(v) != "")
    return to_text(state)


def question_to_internal(qdef: Dict) -> Dict:
    """``{"type", "instructions", "criteria"}`` -> ``{"t", "ins", "crit"}``: a ``choice`` list becomes
    ``{name: None}`` (so duplicate names collapse), non-string instructions are ``json.dumps``-ed."""
    t = qdef["type"]
    crit = qdef.get("criteria")
    if t == "choice" and isinstance(crit, list):
        crit = {c: None for c in crit}
    ins = qdef["instructions"]
    if not isinstance(ins, str):
        ins = json.dumps(ins)
    return {"t": t, "ins": ins, "crit": crit}


def render_options(q: Dict) -> List[str]:
    """Render option texts in label-index order. Noul is always [false, true]."""
    t, crit = q["t"], q.get("crit")
    if t == "choice":
        return [k if not v else "%s: %s" % (k, v) for k, v in crit.items()]
    if t == "score":
        return ["level %d: %s" % (i, c) for i, c in enumerate(crit)]
    crit = crit or {}
    return [
        "false: " + (crit.get("false") or "no, the statement does not hold"),
        "true: " + (crit.get("true") or "yes, the statement holds"),
    ]


def option_labels(q: Dict) -> List[str]:
    """Option labels in label-index order, as the answers name them (choice keys, score levels, false/true)."""
    if q["t"] == "choice":
        return list(q["crit"].keys())
    if q["t"] == "score":
        return [str(i) for i in range(len(q["crit"]))]
    return ["false", "true"]


def normalize_option_text(text: str, case: Optional[str] = None) -> str:
    """One option name stripped and, with ``case``, re-cased: ``"lower"`` (what ``laya.cauldron.parse_options``
    does to A-OKVQA options) or ``"title"`` (first letter upper case, the rest kept)."""
    s = str(text).strip()
    if case is None:
        return s
    if case == "lower":
        return s.lower()
    if case == "title":
        return s[:1].upper() + s[1:]
    raise ValueError("case must be None or one of %s, got %r" % (OPTION_CASES, case))


def normalize_choice_options(q: Dict, case: Optional[str]) -> Dict:
    """An internal ``choice`` question with every option name re-cased (``normalize_option_text``); unchanged for
    other types, for ``case=None``, or when re-casing would make two options the same."""
    if case is None or q["t"] != "choice" or not isinstance(q.get("crit"), dict):
        return q
    new = {normalize_option_text(k, case): v for k, v in q["crit"].items()}
    if len(new) != len(q["crit"]):
        return q
    return dict(q, crit=new)


def record_state(rec: Dict, root: str = "") -> Union[str, Dict]:
    """The state of a prepared-dataset record: ``{"image": <root>/<path>}`` or ``{"images": [...]}`` plus
    ``{"context": state_text}`` when it has text, ``""`` when it has neither."""
    state: Dict[str, Any] = {}
    if rec.get("image"):
        state["image"] = os.path.join(root, rec["image"])
    elif rec.get("images"):
        state["images"] = [os.path.join(root, p) for p in rec["images"]]
    if rec.get("state_text"):
        state[CONTEXT_KEY] = rec["state_text"]
    return state or ""


def make_state(image: Any = None, images: Optional[Sequence[Any]] = None, context: Optional[str] = None) -> Union[str, Dict]:
    """A ``predict`` state laid out the way the checkpoints were trained: the image(s) plus the text under
    ``"context"`` (the key every training record used). Other keys are read, but were never seen in training."""
    state: Dict[str, Any] = {}
    if image is not None:
        state["image"] = image
    elif images:
        state["images"] = list(images)
    if context:
        state[CONTEXT_KEY] = context
    return state or ""


__all__ = ["STATE_FORMATS", "DEFAULT_STATE_FORMAT", "CONTEXT_KEY", "OPTION_CASES", "check_state_format", "to_text",
           "serialize_state", "question_to_internal", "render_options", "option_labels", "normalize_option_text",
           "normalize_choice_options", "record_state", "make_state"]
