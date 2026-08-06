"""Build raw-token preference pairs from scored TTS candidates.

Candidate generation and expensive waveform scoring can happen on a separate
GPU fleet.  This module performs the deterministic final step: apply reward
and Pareto gates, retain exact codec token IDs, and construct DPO/SPO/FPO rows.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from .rewards import CompositeReward, build_composite_reward

TAIL_ERROR_TYPES = frozenset(
    {"repetition", "truncation", "early_stop", "omission", "global"}
)


def fpo_mask_from_error_spans(
    completion_length: int,
    error_spans: Sequence[Mapping[str, Any]],
    *,
    codes_per_group: int,
    audio_start_index: int = 2,
    tail_error_types: frozenset[str] = TAIL_ERROR_TYPES,
) -> list[bool]:
    """Convert frame-level ASR/error spans into a completion-relative FPO mask.

    Spans use an inclusive ``start_frame`` and exclusive ``end_frame``.
    Local pronunciation/substitution errors select only the corresponding codec
    frames.  Repetition and truncation-style failures select from the first bad
    frame through the end, following FPO's causal credit-assignment strategy.
    The first two completion tokens are normally ``START_OF_AI`` and
    ``START_OF_SPEECH`` and therefore remain unselected.
    """

    if completion_length <= 0:
        raise ValueError("completion_length must be positive")
    if codes_per_group <= 0:
        raise ValueError("codes_per_group must be positive")
    if not 0 <= audio_start_index < completion_length:
        raise ValueError("audio_start_index must be inside the completion")

    mask = [False] * completion_length
    for index, span in enumerate(error_spans):
        if not isinstance(span, Mapping):
            raise TypeError(f"error_spans[{index}] must be a mapping")
        try:
            start_frame = int(span["start_frame"])
            end_frame = int(span.get("end_frame", start_frame + 1))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"error_spans[{index}] has invalid frame bounds") from exc
        if start_frame < 0 or end_frame <= start_frame:
            raise ValueError(f"error_spans[{index}] must have 0 <= start < end")

        start = audio_start_index + start_frame * codes_per_group
        error_type = str(span.get("type", "local")).strip().lower()
        if error_type in tail_error_types:
            end = completion_length
        else:
            end = audio_start_index + end_frame * codes_per_group
        start = min(completion_length, start)
        end = min(completion_length, end)
        for token_index in range(start, end):
            mask[token_index] = True

    if not any(mask):
        raise ValueError("error spans do not overlap any completion token")
    return mask


def _tokens(candidate: Mapping[str, Any], name: str) -> list[int]:
    value = candidate.get("completion_input_ids")
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name}.completion_input_ids must be a token sequence")
    if not value:
        raise ValueError(f"{name}.completion_input_ids must be a non-empty token sequence")
    tokens = [int(token) for token in value]
    if any(token < 0 for token in tokens):
        raise ValueError(f"{name}.completion_input_ids contains a negative token")
    return tokens


def _metrics(candidate: Mapping[str, Any], name: str) -> dict[str, float]:
    value = candidate.get("metrics")
    if not isinstance(value, Mapping):
        raise TypeError(f"{name}.metrics must be a mapping")
    return {str(key): float(metric) for key, metric in value.items() if key != "transcript"}


def select_preference_pair(
    prompt_input_ids: Sequence[int],
    candidates: Sequence[Mapping[str, Any]],
    reward: CompositeReward,
    *,
    minimum_reward_gap: float = 0.3,
    codes_per_group: int | None = None,
    audio_start_index: int = 2,
    weight_by_gap: bool = False,
) -> dict[str, Any]:
    """Choose a Pareto-safe winner/loser pair for raw-token preference training."""

    prompt = [int(token) for token in prompt_input_ids]
    if not prompt or any(token < 0 for token in prompt):
        raise ValueError("prompt_input_ids must be a non-empty non-negative sequence")
    if len(candidates) < 2:
        raise ValueError("at least two candidates are required")
    if minimum_reward_gap < 0:
        raise ValueError("minimum_reward_gap must be non-negative")

    prepared: list[tuple[float, Mapping[str, Any], list[int], dict[str, float]]] = []
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, Mapping):
            raise TypeError(f"candidates[{index}] must be a mapping")
        metrics = _metrics(candidate, f"candidates[{index}]")
        score = reward.aggregate(metrics)
        if not reward.passes_protected_thresholds(metrics):
            continue
        prepared.append(
            (score, candidate, _tokens(candidate, f"candidates[{index}]"), metrics)
        )
    if len(prepared) < 2:
        raise ValueError("fewer than two candidates pass protected reward thresholds")

    prepared.sort(key=lambda item: item[0], reverse=True)
    chosen_score, chosen, chosen_tokens, chosen_metrics = prepared[0]
    rejected_entry = None
    for entry in reversed(prepared[1:]):
        rejected_score, _, _, rejected_metrics = entry
        if chosen_score - rejected_score < minimum_reward_gap:
            continue
        if reward.is_preferred(chosen_metrics, rejected_metrics):
            rejected_entry = entry
            break
    if rejected_entry is None:
        raise ValueError("no Pareto-safe preference pair meets minimum_reward_gap")

    rejected_score, rejected, rejected_tokens, _ = rejected_entry

    def selection_mask(candidate: Mapping[str, Any], tokens: list[int]) -> list[bool]:
        supplied = candidate.get("selection_mask")
        if supplied is not None:
            mask = [bool(value) for value in supplied]
            if len(mask) != len(tokens):
                raise ValueError("candidate selection_mask length must match completion tokens")
            return mask
        spans = candidate.get("error_spans")
        if spans is not None:
            if codes_per_group is None:
                raise ValueError("codes_per_group is required when using error_spans")
            return fpo_mask_from_error_spans(
                len(tokens),
                spans,
                codes_per_group=codes_per_group,
                audio_start_index=audio_start_index,
            )
        # A clean winner normally has no error span.  Leaving its branch empty
        # lets the rejected branch's timestamped errors define the shared FPO
        # positions; selecting the whole clean completion would collapse FPO
        # back into utterance-level DPO when masks are combined with ``union``.
        return [False] * len(tokens)

    gap = chosen_score - rejected_score
    return {
        "prompt_input_ids": prompt,
        "chosen_input_ids": chosen_tokens,
        "rejected_input_ids": rejected_tokens,
        "chosen_selection_mask": selection_mask(chosen, chosen_tokens),
        "rejected_selection_mask": selection_mask(rejected, rejected_tokens),
        "sample_weight": gap if weight_by_gap else 1.0,
        "reward_gap": gap,
        "chosen_metrics": chosen_metrics,
        "rejected_metrics": rejected_entry[3],
    }


def build_dataset_from_jsonl(config: Mapping[str, Any]) -> tuple[int, int]:
    """Read candidate groups and write accepted raw-token pairs as JSONL."""

    reward = build_composite_reward(config["reward"])
    input_path = Path(config["input_jsonl"]).expanduser()
    output_path = Path(config["output_jsonl"]).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    accepted = 0
    rejected = 0
    with input_path.open("r", encoding="utf-8") as source, output_path.open(
        "w", encoding="utf-8"
    ) as destination:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            try:
                pair = select_preference_pair(
                    row["prompt_input_ids"],
                    row["candidates"],
                    reward,
                    minimum_reward_gap=float(config.get("minimum_reward_gap", 0.3)),
                    codes_per_group=config.get("codes_per_group"),
                    audio_start_index=int(config.get("audio_start_index", 2)),
                    weight_by_gap=bool(config.get("weight_by_gap", False)),
                )
            except ValueError:
                rejected += 1
                continue
            pair["source_line"] = line_number
            destination.write(json.dumps(pair, ensure_ascii=False) + "\n")
            accepted += 1
    return accepted, rejected


def main() -> None:
    parser = argparse.ArgumentParser(description="Build VyvoTTS raw-token preference pairs")
    parser.add_argument("--config", required=True, help="Pair-building YAML")
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    accepted, rejected = build_dataset_from_jsonl(config)
    print(json.dumps({"accepted": accepted, "rejected": rejected}))


if __name__ == "__main__":
    main()


__all__ = [
    "TAIL_ERROR_TYPES",
    "build_dataset_from_jsonl",
    "fpo_mask_from_error_spans",
    "select_preference_pair",
]
