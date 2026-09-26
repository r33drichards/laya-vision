"""Laya: Fast, non-autoregressive System 1 decision engine with calibrated probabilities."""

from .agent import Agent, RLAgent, load
from .calibration import Calibration
from .common import (
    QTYPES,
    QTYPE_NAMES,
    confidence_from_probs,
    ece_score,
    proper_reward,
    render_options,
    td_lambda_targets,
)
from .email import clean_email_body, email_questions, email_state
from .presets import guard_questions, moderation_questions, router_questions, triage_questions
from .vlm import VLMAgent, VLMDecisionModel, load_vlm

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
    "__version__",
]
