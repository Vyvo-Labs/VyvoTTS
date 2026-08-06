"""Shared raw-token post-training utilities for VyvoTTS."""

from importlib import import_module
from typing import Any

from .common import (
    IGNORE_INDEX,
    CompletionLogProbs,
    RawPreferenceCollator,
    RawTokenPreferenceCollator,
    completion_log_probs,
    get_completion_logps,
    pack_completion_values,
)

_PREFERENCE_EXPORTS = {
    "PreferenceLossOutput",
    "PreferenceObjective",
    "PreferenceTrainer",
    "PreferenceTrainingConfig",
    "RawTokenPreferenceTrainer",
    "build_cli_parser",
    "direct_preference_loss",
    "dpo_loss",
    "fine_grained_preference_loss",
    "fpo_loss",
    "self_preference_loss",
    "spo_loss",
    "train_from_config",
}

_LAZY_EXPORTS = {
    **{name: ".preference" for name in _PREFERENCE_EXPORTS},
    "AudioSFTTrainer": ".sft",
    "CompletionOnlyCollator": ".sft",
    "completion_only_labels": ".sft",
    "weighted_audio_causal_loss": ".sft",
    "OnlinePolicyTrainer": ".online",
    "OnlineTrainingConfig": ".online",
    "clipped_policy_loss": ".online",
    "group_relative_advantages": ".online",
    "CompositeReward": ".rewards",
    "basic_text_normalize": ".rewards",
    "build_composite_reward": ".rewards",
    "fpo_mask_from_error_spans": ".data",
    "select_preference_pair": ".data",
    "resolve_stage_config": ".pipeline",
    "run_pipeline": ".pipeline",
}


def __getattr__(name: str) -> Any:
    """Lazily import Trainer symbols, keeping ``python -m ...preference`` clean."""

    if name in _LAZY_EXPORTS:
        return getattr(import_module(_LAZY_EXPORTS[name], __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "IGNORE_INDEX",
    "AudioSFTTrainer",
    "CompletionLogProbs",
    "CompletionOnlyCollator",
    "CompositeReward",
    "OnlinePolicyTrainer",
    "OnlineTrainingConfig",
    "PreferenceLossOutput",
    "PreferenceObjective",
    "PreferenceTrainer",
    "PreferenceTrainingConfig",
    "RawPreferenceCollator",
    "RawTokenPreferenceCollator",
    "RawTokenPreferenceTrainer",
    "basic_text_normalize",
    "build_cli_parser",
    "build_composite_reward",
    "clipped_policy_loss",
    "completion_log_probs",
    "completion_only_labels",
    "direct_preference_loss",
    "dpo_loss",
    "fine_grained_preference_loss",
    "fpo_loss",
    "fpo_mask_from_error_spans",
    "get_completion_logps",
    "group_relative_advantages",
    "pack_completion_values",
    "resolve_stage_config",
    "run_pipeline",
    "select_preference_pair",
    "self_preference_loss",
    "spo_loss",
    "train_from_config",
    "weighted_audio_causal_loss",
]
