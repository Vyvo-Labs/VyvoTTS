"""Shared raw-token utilities for preference post-training.

The utilities in this module deliberately avoid decoding token IDs back to text.
Speech completions contain codec IDs which are not safely round-trippable through a
text tokenizer, so post-training keeps prompts and completions as integer sequences.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

import torch

IGNORE_INDEX = -100


@dataclass(frozen=True)
class CompletionLogProbs:
    """Causal token log-probabilities and their sequence reductions.

    All token-level tensors are aligned with ``labels[:, 1:]`` because a causal
    language model predicts token *t* from logits at position *t - 1*.
    ``sequence_logps`` and ``mean_logps`` cover the full completion.  Their
    ``selected_*`` counterparts additionally apply the optional FPO mask.
    """

    token_logps: torch.Tensor
    completion_mask: torch.Tensor
    selection_mask: torch.Tensor
    sequence_logps: torch.Tensor
    mean_logps: torch.Tensor
    token_counts: torch.Tensor
    selected_sequence_logps: torch.Tensor
    selected_mean_logps: torch.Tensor
    selected_token_counts: torch.Tensor


def _as_token_list(value: Any, *, name: str, allow_empty: bool = False) -> list[int]:
    if isinstance(value, torch.Tensor):
        if value.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional, got shape {tuple(value.shape)}")
        value = value.detach().cpu().tolist()

    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a one-dimensional sequence of integer token IDs")

    result: list[int] = []
    for index, token_id in enumerate(value):
        if isinstance(token_id, bool) or not isinstance(token_id, Integral):
            raise TypeError(f"{name}[{index}] must be an integer token ID, got {token_id!r}")
        token_id = int(token_id)
        if token_id < 0:
            raise ValueError(f"{name}[{index}] must be non-negative, got {token_id}")
        result.append(token_id)

    if not result and not allow_empty:
        raise ValueError(f"{name} must contain at least one token")
    return result


def _as_binary_mask(value: Any, *, name: str, expected_length: int) -> list[bool]:
    if isinstance(value, torch.Tensor):
        if value.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional, got shape {tuple(value.shape)}")
        value = value.detach().cpu().tolist()

    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a one-dimensional binary sequence")
    if len(value) != expected_length:
        raise ValueError(
            f"{name} has length {len(value)}, but its completion has length {expected_length}"
        )

    result: list[bool] = []
    for index, item in enumerate(value):
        if isinstance(item, bool):
            result.append(item)
        elif isinstance(item, Integral) and int(item) in (0, 1):
            result.append(bool(item))
        else:
            raise ValueError(f"{name}[{index}] must be 0, 1, or bool, got {item!r}")
    return result


def _as_weight(value: Any, *, name: str) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"{name} must be scalar")
        value = value.detach().cpu().item()
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a positive finite number")
    weight = float(value)
    if not torch.isfinite(torch.tensor(weight)) or weight <= 0:
        raise ValueError(f"{name} must be positive and finite, got {weight}")
    return weight


class RawTokenPreferenceCollator:
    """Collate raw prompt/chosen/rejected token IDs without text conversion.

    Each feature must contain ``prompt_input_ids``, ``chosen_input_ids``, and
    ``rejected_input_ids``.  Completion IDs must *not* repeat the prompt.  The
    optional ``chosen_selection_mask`` and ``rejected_selection_mask`` fields are
    binary, completion-relative FPO masks.  A shared ``selection_mask`` is also
    accepted when the chosen and rejected completions have equal length.

    The resulting labels mask every prompt and padding position with
    ``ignore_index``.  Both branches are padded to one common length so a Trainer
    can evaluate chosen and rejected sequences in a single concatenated forward.
    """

    model_input_names = (
        "prompt_input_ids",
        "chosen_input_ids",
        "rejected_input_ids",
        "chosen_selection_mask",
        "rejected_selection_mask",
        "selection_mask",
        "sample_weight",
    )

    def __init__(
        self,
        pad_token_id: int,
        *,
        ignore_index: int = IGNORE_INDEX,
        max_length: int | None = None,
        padding_side: str = "right",
        require_selection: bool = False,
    ) -> None:
        if isinstance(pad_token_id, bool) or not isinstance(pad_token_id, Integral):
            raise TypeError("pad_token_id must be a non-negative integer")
        if int(pad_token_id) < 0:
            raise ValueError("pad_token_id must be non-negative")
        if max_length is not None and max_length < 2:
            raise ValueError("max_length must be at least 2 when provided")
        if padding_side not in {"left", "right"}:
            raise ValueError("padding_side must be 'left' or 'right'")

        self.pad_token_id = int(pad_token_id)
        self.ignore_index = int(ignore_index)
        self.max_length = max_length
        self.padding_side = padding_side
        self.require_selection = require_selection

    def __call__(self, features: Sequence[Mapping[str, Any]]) -> dict[str, torch.Tensor]:
        if not features:
            raise ValueError("Cannot collate an empty preference batch")

        rows: list[dict[str, Any]] = []
        max_batch_length = 0
        for row_index, feature in enumerate(features):
            if not isinstance(feature, Mapping):
                raise TypeError(f"features[{row_index}] must be a mapping")
            required = {"prompt_input_ids", "chosen_input_ids", "rejected_input_ids"}
            missing = required.difference(feature)
            if missing:
                raise KeyError(f"features[{row_index}] is missing fields: {sorted(missing)}")

            prompt = _as_token_list(
                feature["prompt_input_ids"],
                name=f"features[{row_index}].prompt_input_ids",
            )
            chosen = _as_token_list(
                feature["chosen_input_ids"], name=f"features[{row_index}].chosen_input_ids"
            )
            rejected = _as_token_list(
                feature["rejected_input_ids"], name=f"features[{row_index}].rejected_input_ids"
            )

            shared_mask = feature.get("selection_mask")
            if shared_mask is not None:
                if len(chosen) != len(rejected):
                    raise ValueError(
                        f"features[{row_index}].selection_mask requires equal chosen and "
                        "rejected completion lengths"
                    )
                shared = _as_binary_mask(
                    shared_mask,
                    name=f"features[{row_index}].selection_mask",
                    expected_length=len(chosen),
                )
            else:
                shared = None

            chosen_selection = feature.get("chosen_selection_mask", shared)
            rejected_selection = feature.get("rejected_selection_mask", shared)
            chosen_selection = (
                [True] * len(chosen)
                if chosen_selection is None
                else _as_binary_mask(
                    chosen_selection,
                    name=f"features[{row_index}].chosen_selection_mask",
                    expected_length=len(chosen),
                )
            )
            rejected_selection = (
                [True] * len(rejected)
                if rejected_selection is None
                else _as_binary_mask(
                    rejected_selection,
                    name=f"features[{row_index}].rejected_selection_mask",
                    expected_length=len(rejected),
                )
            )

            if self.require_selection and (not any(chosen_selection) and not any(rejected_selection)):
                raise ValueError(f"features[{row_index}] does not select any FPO completion token")

            chosen_length = len(prompt) + len(chosen)
            rejected_length = len(prompt) + len(rejected)
            row_max = max(chosen_length, rejected_length)
            if self.max_length is not None and row_max > self.max_length:
                raise ValueError(
                    f"features[{row_index}] needs {row_max} tokens, exceeding max_length="
                    f"{self.max_length}; pre-truncate data explicitly to avoid silent audio loss"
                )
            max_batch_length = max(max_batch_length, row_max)
            rows.append(
                {
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected,
                    "chosen_selection": chosen_selection,
                    "rejected_selection": rejected_selection,
                    "sample_weight": _as_weight(
                        feature.get("sample_weight", 1.0),
                        name=f"features[{row_index}].sample_weight",
                    ),
                }
            )

        output: dict[str, list[torch.Tensor]] = {
            "chosen_input_ids": [],
            "chosen_attention_mask": [],
            "chosen_labels": [],
            "chosen_completion_mask": [],
            "chosen_selection_mask": [],
            "rejected_input_ids": [],
            "rejected_attention_mask": [],
            "rejected_labels": [],
            "rejected_completion_mask": [],
            "rejected_selection_mask": [],
        }
        prompt_lengths: list[int] = []
        chosen_lengths: list[int] = []
        rejected_lengths: list[int] = []
        sample_weights: list[float] = []

        for row in rows:
            prompt = row["prompt"]
            for branch in ("chosen", "rejected"):
                completion = row[branch]
                selection = row[f"{branch}_selection"]
                ids = prompt + completion
                attention = [True] * len(ids)
                labels = [self.ignore_index] * len(prompt) + completion
                completion_mask = [False] * len(prompt) + [True] * len(completion)
                selection_mask = [False] * len(prompt) + selection
                pad_length = max_batch_length - len(ids)
                if self.padding_side == "right":
                    ids += [self.pad_token_id] * pad_length
                    attention += [False] * pad_length
                    labels += [self.ignore_index] * pad_length
                    completion_mask += [False] * pad_length
                    selection_mask += [False] * pad_length
                else:
                    ids = [self.pad_token_id] * pad_length + ids
                    attention = [False] * pad_length + attention
                    labels = [self.ignore_index] * pad_length + labels
                    completion_mask = [False] * pad_length + completion_mask
                    selection_mask = [False] * pad_length + selection_mask

                output[f"{branch}_input_ids"].append(torch.tensor(ids, dtype=torch.long))
                output[f"{branch}_attention_mask"].append(torch.tensor(attention, dtype=torch.bool))
                output[f"{branch}_labels"].append(torch.tensor(labels, dtype=torch.long))
                output[f"{branch}_completion_mask"].append(
                    torch.tensor(completion_mask, dtype=torch.bool)
                )
                output[f"{branch}_selection_mask"].append(
                    torch.tensor(selection_mask, dtype=torch.bool)
                )

            prompt_lengths.append(len(prompt))
            chosen_lengths.append(len(row["chosen"]))
            rejected_lengths.append(len(row["rejected"]))
            sample_weights.append(row["sample_weight"])

        batch = {key: torch.stack(value) for key, value in output.items()}
        batch["prompt_lengths"] = torch.tensor(prompt_lengths, dtype=torch.long)
        batch["chosen_completion_lengths"] = torch.tensor(chosen_lengths, dtype=torch.long)
        batch["rejected_completion_lengths"] = torch.tensor(rejected_lengths, dtype=torch.long)
        batch["sample_weight"] = torch.tensor(sample_weights, dtype=torch.float32)
        return batch


def completion_log_probs(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    completion_mask: torch.Tensor | None = None,
    selection_mask: torch.Tensor | None = None,
    ignore_index: int = IGNORE_INDEX,
) -> CompletionLogProbs:
    """Return shifted causal log-probabilities for completion tokens only.

    Args:
        logits: Model logits of shape ``[batch, sequence, vocabulary]``.
        labels: Token IDs of shape ``[batch, sequence]``. Prompt/pad positions
            should normally be ``ignore_index``.
        completion_mask: Optional boolean mask in unshifted label coordinates.
        selection_mask: Optional FPO mask in unshifted label coordinates.
        ignore_index: Label sentinel excluded from every reduction.
    """

    if logits.ndim != 3:
        raise ValueError(f"logits must have shape [B, L, V], got {tuple(logits.shape)}")
    if labels.ndim != 2:
        raise ValueError(f"labels must have shape [B, L], got {tuple(labels.shape)}")
    if logits.shape[:2] != labels.shape:
        raise ValueError(
            f"logits/labels batch and sequence dimensions differ: "
            f"{tuple(logits.shape[:2])} vs {tuple(labels.shape)}"
        )
    if logits.shape[1] < 2:
        raise ValueError("At least two sequence positions are required for causal log-probabilities")
    if logits.shape[-1] < 1:
        raise ValueError("Vocabulary dimension must be positive")

    def validate_mask(mask: torch.Tensor | None, name: str) -> torch.Tensor | None:
        if mask is None:
            return None
        if mask.shape != labels.shape:
            raise ValueError(f"{name} shape {tuple(mask.shape)} must equal labels {tuple(labels.shape)}")
        return mask.to(device=labels.device, dtype=torch.bool)

    completion_mask = validate_mask(completion_mask, "completion_mask")
    selection_mask = validate_mask(selection_mask, "selection_mask")

    shifted_labels = labels[:, 1:]
    valid = shifted_labels.ne(ignore_index)
    if completion_mask is not None:
        valid &= completion_mask[:, 1:]

    if valid.any():
        selected_labels = shifted_labels[valid]
        if selected_labels.min().item() < 0:
            raise ValueError("Non-ignored labels must be non-negative token IDs")
        if selected_labels.max().item() >= logits.shape[-1]:
            raise ValueError(
                f"Label token ID {selected_labels.max().item()} is outside model vocabulary "
                f"size {logits.shape[-1]}"
            )
    else:
        raise ValueError("Batch contains no causally predictable completion tokens")

    safe_labels = shifted_labels.masked_fill(~valid, 0)
    # Selective log-softmax avoids materializing an additional [B, L, V]
    # float32 tensor, which is prohibitively large for speech-token vocabularies.
    shifted_logits = logits[:, :-1, :]
    target_logits = shifted_logits.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    token_logps = target_logits.float() - torch.logsumexp(shifted_logits, dim=-1).float()
    token_logps = token_logps.masked_fill(~valid, 0.0)

    selected = valid if selection_mask is None else valid & selection_mask[:, 1:]
    token_counts = valid.sum(dim=-1)
    selected_counts = selected.sum(dim=-1)
    sequence_logps = token_logps.sum(dim=-1)
    selected_sequence_logps = token_logps.masked_fill(~selected, 0.0).sum(dim=-1)
    selected_mean_logps = selected_sequence_logps / selected_counts.clamp_min(1)
    selected_mean_logps = selected_mean_logps.masked_fill(selected_counts == 0, 0.0)
    return CompletionLogProbs(
        token_logps=token_logps,
        completion_mask=valid,
        selection_mask=selected,
        sequence_logps=sequence_logps,
        mean_logps=sequence_logps / token_counts,
        token_counts=token_counts,
        selected_sequence_logps=selected_sequence_logps,
        selected_mean_logps=selected_mean_logps,
        selected_token_counts=selected_counts,
    )


def pack_completion_values(
    values: torch.Tensor,
    completion_mask: torch.Tensor,
    selection_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack shifted sequence tensors into completion-relative coordinates.

    Returns ``(packed_values, packed_completion_mask, packed_selection_mask)``.
    This makes chosen/rejected token positions comparable for FPO even when their
    prompts have different padded offsets.
    """

    if values.ndim != 2 or completion_mask.shape != values.shape:
        raise ValueError("values and completion_mask must have the same [B, L] shape")
    completion_mask = completion_mask.to(dtype=torch.bool, device=values.device)
    if selection_mask is None:
        selection_mask = completion_mask
    elif selection_mask.shape != values.shape:
        raise ValueError("selection_mask must have the same [B, L] shape as values")
    else:
        selection_mask = selection_mask.to(dtype=torch.bool, device=values.device)

    lengths = completion_mask.sum(dim=-1)
    if (lengths == 0).any():
        raise ValueError("Every row must contain at least one completion token")
    max_length = int(lengths.max().item())
    packed_values = values.new_zeros((values.shape[0], max_length))
    packed_valid = torch.zeros(
        (values.shape[0], max_length), dtype=torch.bool, device=values.device
    )
    packed_selected = torch.zeros_like(packed_valid)
    for row in range(values.shape[0]):
        row_mask = completion_mask[row]
        length = int(lengths[row].item())
        packed_values[row, :length] = values[row, row_mask]
        packed_valid[row, :length] = True
        packed_selected[row, :length] = selection_mask[row, row_mask]
    return packed_values, packed_valid, packed_selected


# Descriptive and backward-friendly aliases for callers.
RawPreferenceCollator = RawTokenPreferenceCollator
get_completion_logps = completion_log_probs


__all__ = [
    "IGNORE_INDEX",
    "CompletionLogProbs",
    "RawPreferenceCollator",
    "RawTokenPreferenceCollator",
    "completion_log_probs",
    "get_completion_logps",
    "pack_completion_values",
]
