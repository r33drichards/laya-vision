"""Laya: Fast, non-autoregressive System 1 decision engine with calibrated probabilities.

The public names below are imported on first use (PEP 562), so ``import laya.client`` (the HTTP client, which needs
only ``requests``) and ``import laya.mock`` load no torch. ``from laya import VLMAgent`` works as before.
"""
import importlib

_EXPORTS = {
    "Agent": "agent",
    "RLAgent": "agent",
    "load": "agent",
    "Calibration": "calibration",
    "QTYPES": "common",
    "QTYPE_NAMES": "common",
    "confidence_from_probs": "common",
    "ece_score": "common",
    "proper_reward": "common",
    "render_options": "common",
    "td_lambda_targets": "common",
    "clean_email_body": "email",
    "email_questions": "email",
    "email_state": "email",
    "guard_questions": "presets",
    "moderation_questions": "presets",
    "router_questions": "presets",
    "triage_questions": "presets",
    "VLMAgent": "vlm",
    "VLMDecisionModel": "vlm",
    "load_vlm": "vlm",
    "LayaClient": "client",
    "LayaError": "client",
}

__version__ = "0.2.0.dev0"  # the one version: pyproject.toml reads it (dynamic = ["version"])
__all__ = [
    "Agent",
    "RLAgent",
    "load",
    "VLMAgent",
    "VLMDecisionModel",
    "load_vlm",
    "Calibration",
    "clean_email_body",
    "email_questions",
    "email_state",
    "guard_questions",
    "moderation_questions",
    "router_questions",
    "triage_questions",
    "proper_reward",
    "td_lambda_targets",
    "ece_score",
    "confidence_from_probs",
    "render_options",
    "QTYPES",
    "QTYPE_NAMES",
    "LayaClient",
    "LayaError",
    "__version__",
]


def __getattr__(name):
    if name in _EXPORTS:
        value = getattr(importlib.import_module("." + _EXPORTS[name], __name__), name)
        globals()[name] = value
        return value
    raise AttributeError("module %r has no attribute %r" % (__name__, name))


def __dir__():
    return sorted(set(globals()) | set(_EXPORTS))
