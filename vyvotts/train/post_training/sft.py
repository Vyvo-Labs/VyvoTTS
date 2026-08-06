"""Completion-only supervised fine-tuning for VyvoTTS.

The stock Hugging Face causal-LM loss treats the text prompt and every audio
codebook equally.  This module keeps loss on the assistant/audio completion
only and can emphasize the first (semantic/coarse) codec streams, which have
the largest effect on pronunciation and word error rate.
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from datasets import DatasetDict, load_dataset, load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


def completion_only_labels(
    input_ids: Sequence[int],
    *,
    end_of_human: int,
    ignore_index: int = -100,
) -> list[int]:
    """Mask the user/text prompt and supervise tokens after ``END_OF_HUMAN``.

    ``START_OF_AI`` and ``START_OF_SPEECH`` remain supervised so the model
    learns the complete response grammar.  A malformed example is rejected
    instead of silently training on prompt tokens.
    """

    ids = [int(token) for token in input_ids]
    try:
        boundary = ids.index(int(end_of_human))
    except ValueError as exc:
        raise ValueError("input_ids does not contain END_OF_HUMAN") from exc
    if boundary + 1 >= len(ids):
        raise ValueError("input_ids has no assistant completion")
    return [ignore_index] * (boundary + 1) + ids[boundary + 1 :]


@dataclass
class CompletionOnlyCollator:
    """Pad raw token rows and enforce completion-only labels."""

    pad_token_id: int
    end_of_human: int
    ignore_index: int = -100
    trust_dataset_labels: bool = False

    def __call__(self, features: list[Mapping[str, Any]]) -> dict[str, torch.Tensor]:
        if not features:
            raise ValueError("cannot collate an empty batch")

        ids: list[torch.Tensor] = []
        labels: list[torch.Tensor] = []
        for feature in features:
            row_ids = [int(token) for token in feature["input_ids"]]
            if self.trust_dataset_labels and "labels" in feature:
                row_labels = [int(token) for token in feature["labels"]]
                if len(row_labels) != len(row_ids):
                    raise ValueError("input_ids and labels must have equal lengths")
            else:
                row_labels = completion_only_labels(
                    row_ids,
                    end_of_human=self.end_of_human,
                    ignore_index=self.ignore_index,
                )
            ids.append(torch.tensor(row_ids, dtype=torch.long))
            labels.append(torch.tensor(row_labels, dtype=torch.long))

        padded_ids = torch.nn.utils.rnn.pad_sequence(
            ids, batch_first=True, padding_value=self.pad_token_id
        )
        padded_labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=self.ignore_index
        )
        return {
            "input_ids": padded_ids,
            "attention_mask": padded_ids.ne(self.pad_token_id).long(),
            "labels": padded_labels,
        }


def weighted_audio_causal_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    audio_tokens_start: int,
    codebook_size: int,
    codebook_weights: Sequence[float],
    boundary_token_ids: Sequence[int] = (),
    boundary_weight: float = 1.0,
    label_smoothing: float = 0.0,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Causal cross entropy with configurable codec-stream weighting."""

    if logits.ndim != 3 or labels.ndim != 2 or logits.shape[:2] != labels.shape:
        raise ValueError("logits [B,T,V] and labels [B,T] must align")
    if codebook_size <= 0:
        raise ValueError("codebook_size must be positive")
    if not codebook_weights or any(
        not math.isfinite(float(weight)) or weight <= 0 for weight in codebook_weights
    ):
        raise ValueError("codebook_weights must contain positive finite values")
    if not math.isfinite(boundary_weight) or boundary_weight <= 0:
        raise ValueError("boundary_weight must be positive and finite")
    if not 0.0 <= label_smoothing < 1.0:
        raise ValueError("label_smoothing must be in [0, 1)")

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    valid = shift_labels.ne(ignore_index)
    if not valid.any():
        raise ValueError("batch contains no supervised completion tokens")

    token_losses = F.cross_entropy(
        shift_logits.transpose(1, 2),
        shift_labels,
        reduction="none",
        ignore_index=ignore_index,
        label_smoothing=float(label_smoothing),
    )
    weights = torch.ones_like(token_losses)
    safe_labels = shift_labels.masked_fill(~valid, 0)
    audio_upper = audio_tokens_start + len(codebook_weights) * codebook_size
    is_audio = valid & safe_labels.ge(audio_tokens_start) & safe_labels.lt(audio_upper)
    phases = torch.div(
        (safe_labels - audio_tokens_start).clamp_min(0),
        codebook_size,
        rounding_mode="floor",
    )
    phase_weights = torch.as_tensor(
        codebook_weights, dtype=token_losses.dtype, device=token_losses.device
    )
    weights = torch.where(is_audio, phase_weights[phases.clamp_max(len(codebook_weights) - 1)], weights)

    if boundary_token_ids:
        boundary = torch.zeros_like(valid)
        for token_id in boundary_token_ids:
            boundary |= safe_labels.eq(int(token_id))
        weights = torch.where(boundary & valid, weights.new_tensor(boundary_weight), weights)

    weighted_mask = weights * valid
    return (token_losses * weighted_mask).sum() / weighted_mask.sum().clamp_min(1.0)


class AudioSFTTrainer(Trainer):
    """Trainer using completion-only, codebook-aware causal loss."""

    def __init__(
        self,
        *args,
        audio_tokens_start: int,
        codebook_size: int,
        codebook_weights: Sequence[float],
        boundary_token_ids: Sequence[int] = (),
        boundary_weight: float = 1.0,
        label_smoothing: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.audio_tokens_start = int(audio_tokens_start)
        self.codebook_size = int(codebook_size)
        self.codebook_weights = tuple(float(value) for value in codebook_weights)
        self.boundary_token_ids = tuple(int(value) for value in boundary_token_ids)
        self.boundary_weight = float(boundary_weight)
        self.audio_label_smoothing = float(label_smoothing)

    def compute_loss(
        self,
        model,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch=None,
    ):
        labels = inputs.pop("labels")
        outputs = model(**inputs, use_cache=False)
        loss = weighted_audio_causal_loss(
            outputs.logits,
            labels,
            audio_tokens_start=self.audio_tokens_start,
            codebook_size=self.codebook_size,
            codebook_weights=self.codebook_weights,
            boundary_token_ids=self.boundary_token_ids,
            boundary_weight=self.boundary_weight,
            label_smoothing=self.audio_label_smoothing,
        )
        return (loss, outputs) if return_outputs else loss


def _load_data(path: str, split: str):
    candidate = Path(path).expanduser()
    if candidate.is_file():
        builders = {
            ".json": "json",
            ".jsonl": "json",
            ".parquet": "parquet",
            ".csv": "csv",
        }
        try:
            builder = builders[candidate.suffix.lower()]
        except KeyError as exc:
            raise ValueError(f"unsupported local dataset file: {candidate}") from exc
        return load_dataset(builder, data_files=str(candidate), split=split)
    if candidate.exists():
        data = load_from_disk(str(candidate))
        if isinstance(data, DatasetDict):
            if split not in data:
                raise KeyError(f"local DatasetDict has no split {split!r}")
            return data[split]
        return data
    return load_dataset(path, split=split)


def _dtype(name: str | None):
    if name is None or name == "auto":
        return "auto"
    values = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if name not in values:
        raise ValueError(f"unsupported dtype {name!r}")
    return values[name]


def train_from_config(raw: Mapping[str, Any]) -> Path:
    """Run completion-only SFT from a parsed YAML mapping."""

    with open(raw["inference_config"], "r", encoding="utf-8") as handle:
        tokens = yaml.safe_load(handle)
    model_name = str(raw["model_name"])
    tokenizer_name = str(raw.get("tokenizer_name", model_name))
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=_dtype(raw.get("dtype", "bfloat16")),
        attn_implementation=raw.get("attn_implementation", "sdpa"),
    )

    codes_per_group = int(raw["codes_per_group"])
    codebook_size = int(raw["codebook_size"])
    if codes_per_group <= 0 or codebook_size <= 0:
        raise ValueError("codes_per_group and codebook_size must be positive")
    codebook_weights = tuple(
        float(value)
        for value in raw.get(
            "codebook_weights", [1.5] + [1.0] * (codes_per_group - 1)
        )
    )
    if len(codebook_weights) != codes_per_group:
        raise ValueError(
            "codebook_weights length must exactly match codes_per_group "
            f"({len(codebook_weights)} != {codes_per_group})"
        )

    highest_required_id = int(tokens["AUDIO_TOKENS_START"]) + codes_per_group * codebook_size
    if model.get_input_embeddings().num_embeddings < highest_required_id:
        raise ValueError(
            "model vocabulary is smaller than the configured audio-token range; "
            "load a checkpoint whose embeddings were resized during pretraining"
        )

    train_data = _load_data(str(raw["dataset"]), str(raw.get("split", "train")))
    eval_data = None
    if raw.get("eval_dataset"):
        eval_data = _load_data(
            str(raw["eval_dataset"]), str(raw.get("eval_split", "validation"))
        )

    training_values = dict(raw.get("training_args", {}))
    output_dir = str(raw.get("output_dir", training_values.get("output_dir", "outputs/sft")))
    training_values["output_dir"] = output_dir
    training_values.setdefault("remove_unused_columns", False)
    training_values.setdefault("report_to", "none")
    arguments = TrainingArguments(**training_values)

    trainer = AudioSFTTrainer(
        model=model,
        args=arguments,
        train_dataset=train_data,
        eval_dataset=eval_data,
        data_collator=CompletionOnlyCollator(
            pad_token_id=int(tokens["PAD_TOKEN"]),
            end_of_human=int(tokens["END_OF_HUMAN"]),
            trust_dataset_labels=bool(raw.get("trust_dataset_labels", False)),
        ),
        audio_tokens_start=int(tokens["AUDIO_TOKENS_START"]),
        codebook_size=codebook_size,
        codebook_weights=codebook_weights,
        boundary_token_ids=(
            int(tokens["START_OF_AI"]),
            int(tokens["START_OF_SPEECH"]),
            int(tokens["END_OF_SPEECH"]),
            int(tokens["END_OF_AI"]),
        ),
        boundary_weight=float(raw.get("boundary_weight", 1.25)),
        label_smoothing=float(raw.get("label_smoothing", 0.0)),
    )
    trainer.train(resume_from_checkpoint=raw.get("resume_from_checkpoint"))
    final_dir = Path(output_dir) / "final"
    trainer.save_model(str(final_dir))
    trainer.accelerator.wait_for_everyone()
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(str(final_dir))
        with (final_dir / "sft_config.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(dict(raw), handle, sort_keys=False)
    trainer.accelerator.wait_for_everyone()
    return final_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Completion-only VyvoTTS SFT")
    parser.add_argument("--config", required=True, help="SFT YAML configuration")
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    print(train_from_config(raw))


if __name__ == "__main__":
    main()


__all__ = [
    "AudioSFTTrainer",
    "CompletionOnlyCollator",
    "completion_only_labels",
    "train_from_config",
    "weighted_audio_causal_loss",
]
