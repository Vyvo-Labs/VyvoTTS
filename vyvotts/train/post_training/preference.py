"""Raw-token DPO, Self Preference Optimization, and fine-grained FPO.

The Trainer in this module is intentionally model-agnostic: any Hugging Face
causal language model can be used as long as the policy and reference model share
the vocabulary used by the raw speech-token dataset.

Dataset rows must contain ``prompt_input_ids``, ``chosen_input_ids``, and
``rejected_input_ids`` as one-dimensional integer arrays. Completions must not
repeat the prompt. Optional fields are ``sample_weight`` and completion-relative
binary ``selection_mask`` (shared, equal-length pairs) or
``chosen_selection_mask`` / ``rejected_selection_mask`` (FPO). Run a YAML config
with ``python -m vyvotts.train.post_training.preference --config CONFIG.yaml``.
The public ``train_from_config`` function provides the same entry point to Python.
"""

from __future__ import annotations

import argparse
import warnings
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, ClassVar, Literal

import torch
import torch.nn.functional as F
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from .common import (
    CompletionLogProbs,
    RawTokenPreferenceCollator,
    completion_log_probs,
    pack_completion_values,
)

PreferenceObjective = Literal["dpo", "spo", "fpo"]


@dataclass
class PreferenceTrainingConfig:
    """Objective settings accepted directly by ``RawTokenPreferenceTrainer``.

    This is a plain dataclass so it can also be passed to
    ``transformers.HfArgumentParser`` in a command-line entry point.
    ``spo`` means the 2025 *Self Preference Optimization with Self
    Regularization* objective, not another algorithm sharing the SPO acronym.
    """

    objective: PreferenceObjective = "dpo"
    beta: float | None = None
    gamma: float = 0.8
    label_smoothing: float = 0.0
    length_normalize_dpo: bool = False
    fpo_mask_combine: Literal["union", "intersection", "chosen", "rejected"] = "union"
    fpo_token_reduction: Literal["sum", "mean"] = "sum"

    def __post_init__(self) -> None:
        self.objective = str(self.objective).lower()  # type: ignore[assignment]
        if self.objective not in {"dpo", "spo", "fpo"}:
            raise ValueError("objective must be one of: dpo, spo, fpo")
        if self.beta is None:
            self.beta = 2.5 if self.objective == "spo" else 0.1
        if not torch.isfinite(torch.tensor(self.beta)) or self.beta <= 0:
            raise ValueError("beta must be positive and finite")
        if not torch.isfinite(torch.tensor(self.gamma)):
            raise ValueError("gamma must be finite")
        if not 0.0 <= self.label_smoothing < 0.5:
            raise ValueError("label_smoothing must be in [0, 0.5)")
        if self.fpo_mask_combine not in {"union", "intersection", "chosen", "rejected"}:
            raise ValueError(
                "fpo_mask_combine must be union, intersection, chosen, or rejected"
            )
        if self.fpo_token_reduction not in {"sum", "mean"}:
            raise ValueError("fpo_token_reduction must be sum or mean")


@dataclass(frozen=True)
class PreferenceLossOutput:
    """A differentiable preference loss plus detached logging signals."""

    loss: torch.Tensor
    per_example_loss: torch.Tensor
    logits: torch.Tensor
    metrics: Mapping[str, torch.Tensor]


def _validate_vector(tensor: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor) or tensor.ndim != 1:
        shape = getattr(tensor, "shape", None)
        raise ValueError(f"{name} must be a [batch] tensor, got {shape}")
    if tensor.numel() == 0:
        raise ValueError(f"{name} cannot be empty")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains non-finite values")
    return tensor.float()


def _validate_same_shape(named_tensors: Mapping[str, torch.Tensor]) -> None:
    shapes = {name: tuple(tensor.shape) for name, tensor in named_tensors.items()}
    if len(set(shapes.values())) != 1:
        raise ValueError(f"Preference tensors must have identical shapes, got {shapes}")


def _weighted_mean(values: torch.Tensor, sample_weight: torch.Tensor | None) -> torch.Tensor:
    if sample_weight is None:
        return values.mean()
    sample_weight = _validate_vector(sample_weight, "sample_weight").to(values.device)
    if sample_weight.shape != values.shape:
        raise ValueError(
            f"sample_weight shape {tuple(sample_weight.shape)} must equal losses "
            f"{tuple(values.shape)}"
        )
    if (sample_weight <= 0).any():
        raise ValueError("sample_weight values must be strictly positive")
    return (values * sample_weight).sum() / sample_weight.sum()


def direct_preference_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    *,
    beta: float = 0.1,
    label_smoothing: float = 0.0,
    sample_weight: torch.Tensor | None = None,
) -> PreferenceLossOutput:
    """Compute the canonical reference-anchored DPO objective."""

    tensors = {
        "policy_chosen_logps": _validate_vector(policy_chosen_logps, "policy_chosen_logps"),
        "policy_rejected_logps": _validate_vector(
            policy_rejected_logps, "policy_rejected_logps"
        ),
        "reference_chosen_logps": _validate_vector(
            reference_chosen_logps, "reference_chosen_logps"
        ),
        "reference_rejected_logps": _validate_vector(
            reference_rejected_logps, "reference_rejected_logps"
        ),
    }
    _validate_same_shape(tensors)
    if not torch.isfinite(torch.tensor(beta)) or beta <= 0:
        raise ValueError("beta must be positive and finite")
    if not 0.0 <= label_smoothing < 0.5:
        raise ValueError("label_smoothing must be in [0, 0.5)")

    pc = tensors["policy_chosen_logps"]
    pr = tensors["policy_rejected_logps"].to(pc.device)
    rc = tensors["reference_chosen_logps"].to(pc.device)
    rr = tensors["reference_rejected_logps"].to(pc.device)
    preference_logits = (pc - pr) - (rc - rr)
    scaled_logits = beta * preference_logits
    losses = -(
        (1.0 - label_smoothing) * F.logsigmoid(scaled_logits)
        + label_smoothing * F.logsigmoid(-scaled_logits)
    )
    chosen_rewards = beta * (pc - rc).detach()
    rejected_rewards = beta * (pr - rr).detach()
    reward_margin = chosen_rewards - rejected_rewards
    loss = _weighted_mean(losses, sample_weight)
    return PreferenceLossOutput(
        loss=loss,
        per_example_loss=losses,
        logits=scaled_logits,
        metrics={
            "objective_loss": loss.detach(),
            "preference_accuracy": (reward_margin > 0).float().mean(),
            "preference_logit": scaled_logits.detach().mean(),
            "chosen_reward": chosen_rewards.mean(),
            "rejected_reward": rejected_rewards.mean(),
            "reward_margin": reward_margin.mean(),
        },
    )


def self_preference_loss(
    chosen_mean_logps: torch.Tensor,
    rejected_mean_logps: torch.Tensor,
    *,
    beta: float = 2.5,
    gamma: float = 0.8,
    sample_weight: torch.Tensor | None = None,
) -> PreferenceLossOutput:
    """Compute 2025 Self Preference Optimization with length normalization.

    The exact per-example objective is ``SiLU(z)`` where
    ``z = -(beta * (mean_logp_chosen - mean_logp_rejected) - gamma)``.
    Mean completion log-probabilities are required to remove sequence-length bias.
    """

    chosen = _validate_vector(chosen_mean_logps, "chosen_mean_logps")
    rejected = _validate_vector(rejected_mean_logps, "rejected_mean_logps").to(chosen.device)
    _validate_same_shape({"chosen_mean_logps": chosen, "rejected_mean_logps": rejected})
    if not torch.isfinite(torch.tensor(beta)) or beta <= 0:
        raise ValueError("beta must be positive and finite")
    if not torch.isfinite(torch.tensor(gamma)):
        raise ValueError("gamma must be finite")

    logp_margin = chosen - rejected
    z = -(beta * logp_margin - gamma)
    losses = F.silu(z)
    loss = _weighted_mean(losses, sample_weight)
    return PreferenceLossOutput(
        loss=loss,
        per_example_loss=losses,
        logits=-z,
        metrics={
            "objective_loss": loss.detach(),
            "preference_accuracy": (logp_margin > 0).float().mean().detach(),
            "preference_logit": (-z).mean().detach(),
            "spo_z": z.mean().detach(),
            "logp_margin": logp_margin.mean().detach(),
        },
    )


def fine_grained_preference_loss(
    policy_chosen_token_logps: torch.Tensor,
    policy_rejected_token_logps: torch.Tensor,
    reference_chosen_token_logps: torch.Tensor,
    reference_rejected_token_logps: torch.Tensor,
    selection_mask: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
    beta: float = 0.1,
    label_smoothing: float = 0.0,
    token_reduction: Literal["sum", "mean"] = "sum",
    sample_weight: torch.Tensor | None = None,
) -> PreferenceLossOutput:
    """Compute token-level selective DPO from the 2025 TTS FPO objective.

    Chosen and rejected tensors must already be aligned in completion-relative
    coordinates.  Only positions enabled by ``selection_mask`` contribute.  The
    paper's exact sum-over-selected-tokens reduction is the default; ``mean`` is
    available when mask lengths vary widely and a length-invariant scale is desired.
    """

    tensors = {
        "policy_chosen_token_logps": policy_chosen_token_logps,
        "policy_rejected_token_logps": policy_rejected_token_logps,
        "reference_chosen_token_logps": reference_chosen_token_logps,
        "reference_rejected_token_logps": reference_rejected_token_logps,
    }
    if any(not isinstance(tensor, torch.Tensor) or tensor.ndim != 2 for tensor in tensors.values()):
        raise ValueError("All FPO log-probability inputs must have shape [batch, token]")
    _validate_same_shape(tensors)
    shape = policy_chosen_token_logps.shape
    if selection_mask.shape != shape:
        raise ValueError(f"selection_mask shape {tuple(selection_mask.shape)} must equal {shape}")
    if valid_mask is None:
        valid_mask = torch.ones_like(selection_mask, dtype=torch.bool)
    elif valid_mask.shape != shape:
        raise ValueError(f"valid_mask shape {tuple(valid_mask.shape)} must equal {shape}")
    selected = selection_mask.to(dtype=torch.bool) & valid_mask.to(dtype=torch.bool)
    selected_counts = selected.sum(dim=-1)
    if (selected_counts == 0).any():
        rows = (selected_counts == 0).nonzero(as_tuple=False).flatten().tolist()
        raise ValueError(f"FPO selection_mask selects no aligned token for rows {rows}")
    if not torch.isfinite(torch.tensor(beta)) or beta <= 0:
        raise ValueError("beta must be positive and finite")
    if not 0.0 <= label_smoothing < 0.5:
        raise ValueError("label_smoothing must be in [0, 0.5)")
    if token_reduction not in {"sum", "mean"}:
        raise ValueError("token_reduction must be sum or mean")

    pc = policy_chosen_token_logps.float()
    pr = policy_rejected_token_logps.float().to(pc.device)
    rc = reference_chosen_token_logps.float().to(pc.device)
    rr = reference_rejected_token_logps.float().to(pc.device)
    selected = selected.to(pc.device)
    valid_mask = valid_mask.to(pc.device)
    token_logits = beta * ((pc - rc) - (pr - rr))
    token_losses = -(
        (1.0 - label_smoothing) * F.logsigmoid(token_logits)
        + label_smoothing * F.logsigmoid(-token_logits)
    )
    masked_losses = token_losses.masked_fill(~selected, 0.0)
    per_example = masked_losses.sum(dim=-1)
    if token_reduction == "mean":
        per_example = per_example / selected_counts.to(pc.device)

    selected_logits = token_logits.masked_select(selected)
    chosen_rewards = beta * (pc - rc).detach()
    rejected_rewards = beta * (pr - rr).detach()
    selected_reward_margin = (chosen_rewards - rejected_rewards).masked_select(selected)
    loss = _weighted_mean(per_example, sample_weight)
    return PreferenceLossOutput(
        loss=loss,
        per_example_loss=per_example,
        logits=token_logits,
        metrics={
            "objective_loss": loss.detach(),
            "preference_accuracy": (selected_reward_margin > 0).float().mean(),
            "preference_logit": selected_logits.detach().mean(),
            "reward_margin": selected_reward_margin.mean(),
            "selected_tokens": selected_counts.float().mean().detach(),
            "selected_fraction": (
                selected.sum().float() / valid_mask.sum().clamp_min(1).float()
            ).detach(),
        },
    )


def _extract_logits(outputs: Any) -> torch.Tensor:
    logits = outputs.get("logits") if isinstance(outputs, Mapping) else getattr(outputs, "logits", None)
    if logits is None and isinstance(outputs, (tuple, list)) and outputs:
        logits = outputs[0]
    if not isinstance(logits, torch.Tensor):
        raise TypeError("Causal language model output must expose a logits tensor")
    return logits


class RawTokenPreferenceTrainer(Trainer):
    """Hugging Face Trainer for raw speech-token DPO, SPO, or FPO.

    Args:
        reference_model: Frozen SFT/reference policy. Required for DPO and FPO;
            omitted for reference-free SPO.
        preference_config: Objective settings, also compatible with
            ``HfArgumentParser``.

    Dataset rows should be collated by ``RawTokenPreferenceCollator``.  Metrics
    are buffered during gradient accumulation and emitted alongside Trainer logs
    as ``preference/*`` (or ``eval_preference/*``) means.
    """

    _raw_signature_columns: ClassVar[list[str]] = [
        "prompt_input_ids",
        "chosen_input_ids",
        "rejected_input_ids",
        "chosen_selection_mask",
        "rejected_selection_mask",
        "selection_mask",
        "sample_weight",
    ]

    def __init__(
        self,
        *args: Any,
        reference_model: torch.nn.Module | None = None,
        preference_config: PreferenceTrainingConfig | None = None,
        **kwargs: Any,
    ) -> None:
        self.preference_config = preference_config or PreferenceTrainingConfig()
        if self.preference_config.objective in {"dpo", "fpo"} and reference_model is None:
            raise ValueError(f"{self.preference_config.objective.upper()} requires reference_model")
        if reference_model is not None and kwargs.get("model") is reference_model:
            raise ValueError("reference_model must be a distinct frozen model, not the policy object")

        super().__init__(*args, **kwargs)
        self.reference_model = reference_model
        if self.reference_model is self.model:
            raise ValueError("reference_model must be a distinct frozen model, not the policy object")
        if self.reference_model is not None:
            self.reference_model.requires_grad_(False)
            self.reference_model.eval()
        self._preference_metrics: dict[str, defaultdict[str, list[float]]] = {
            "train": defaultdict(list),
            "eval": defaultdict(list),
        }

    def _set_signature_columns_if_needed(self) -> None:
        super()._set_signature_columns_if_needed()
        assert self._signature_columns is not None
        self._signature_columns = list(
            dict.fromkeys([*self._signature_columns, *self._raw_signature_columns])
        )

    @staticmethod
    def _pair_forward(
        model: torch.nn.Module,
        inputs: Mapping[str, torch.Tensor],
    ) -> tuple[Any, CompletionLogProbs, CompletionLogProbs]:
        chosen_ids = inputs["chosen_input_ids"]
        rejected_ids = inputs["rejected_input_ids"]
        if chosen_ids.shape != rejected_ids.shape:
            raise ValueError("Collator must pad chosen and rejected branches to one common shape")
        batch_size = chosen_ids.shape[0]
        input_ids = torch.cat([chosen_ids, rejected_ids], dim=0)
        attention_mask = torch.cat(
            [inputs["chosen_attention_mask"], inputs["rejected_attention_mask"]], dim=0
        )
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        logits = _extract_logits(outputs)
        if logits.shape[0] != 2 * batch_size:
            raise ValueError("Model returned an unexpected batch dimension")
        chosen = completion_log_probs(
            logits[:batch_size],
            inputs["chosen_labels"],
            completion_mask=inputs["chosen_completion_mask"],
            selection_mask=inputs["chosen_selection_mask"],
        )
        rejected = completion_log_probs(
            logits[batch_size:],
            inputs["rejected_labels"],
            completion_mask=inputs["rejected_completion_mask"],
            selection_mask=inputs["rejected_selection_mask"],
        )
        return outputs, chosen, rejected

    def _reference_forward(
        self,
        policy_model: torch.nn.Module,
        inputs: Mapping[str, torch.Tensor],
    ) -> tuple[CompletionLogProbs, CompletionLogProbs]:
        if self.reference_model is None:
            raise RuntimeError("Reference forward requested without a reference model")
        policy_vocab = getattr(getattr(policy_model, "config", None), "vocab_size", None)
        reference_vocab = getattr(
            getattr(self.reference_model, "config", None), "vocab_size", None
        )
        if (
            policy_vocab is not None
            and reference_vocab is not None
            and policy_vocab != reference_vocab
        ):
            raise ValueError(
                f"Policy/reference vocabularies differ: {policy_vocab} vs {reference_vocab}"
            )
        try:
            policy_device = next(policy_model.parameters()).device
            reference_device = next(self.reference_model.parameters()).device
        except StopIteration as error:
            raise ValueError("Policy and reference models must have parameters") from error
        if policy_device != reference_device:
            self.reference_model.to(policy_device)
        self.reference_model.eval()
        with torch.no_grad():
            _, chosen, rejected = self._pair_forward(self.reference_model, inputs)
        return chosen, rejected

    def _store_metrics(self, metrics: Mapping[str, torch.Tensor], *, train: bool) -> None:
        split = "train" if train else "eval"
        for name, value in metrics.items():
            if not isinstance(value, torch.Tensor) or value.numel() != 1:
                raise ValueError(f"Metric {name!r} must be a scalar tensor")
            self._preference_metrics[split][name].append(float(value.detach().float().cpu()))

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: Mapping[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, Mapping[str, torch.Tensor]]:
        del num_items_in_batch  # This Trainer performs its own sample-weighted reduction.
        required = {
            f"{branch}_{field}"
            for branch in ("chosen", "rejected")
            for field in ("input_ids", "attention_mask", "labels", "completion_mask", "selection_mask")
        }
        missing = required.difference(inputs)
        if missing:
            raise KeyError(f"Preference batch is missing fields: {sorted(missing)}")

        _, policy_chosen, policy_rejected = self._pair_forward(model, inputs)
        sample_weight = inputs.get("sample_weight")
        cfg = self.preference_config

        if cfg.objective == "spo":
            result = self_preference_loss(
                policy_chosen.mean_logps,
                policy_rejected.mean_logps,
                beta=cfg.beta,
                gamma=cfg.gamma,
                sample_weight=sample_weight,
            )
        else:
            reference_chosen, reference_rejected = self._reference_forward(model, inputs)
            if cfg.objective == "dpo":
                policy_chosen_logps = (
                    policy_chosen.mean_logps
                    if cfg.length_normalize_dpo
                    else policy_chosen.sequence_logps
                )
                policy_rejected_logps = (
                    policy_rejected.mean_logps
                    if cfg.length_normalize_dpo
                    else policy_rejected.sequence_logps
                )
                reference_chosen_logps = (
                    reference_chosen.mean_logps
                    if cfg.length_normalize_dpo
                    else reference_chosen.sequence_logps
                )
                reference_rejected_logps = (
                    reference_rejected.mean_logps
                    if cfg.length_normalize_dpo
                    else reference_rejected.sequence_logps
                )
                result = direct_preference_loss(
                    policy_chosen_logps,
                    policy_rejected_logps,
                    reference_chosen_logps,
                    reference_rejected_logps,
                    beta=cfg.beta,
                    label_smoothing=cfg.label_smoothing,
                    sample_weight=sample_weight,
                )
            else:
                pc, pc_valid, pc_selected = pack_completion_values(
                    policy_chosen.token_logps,
                    policy_chosen.completion_mask,
                    policy_chosen.selection_mask,
                )
                pr, pr_valid, pr_selected = pack_completion_values(
                    policy_rejected.token_logps,
                    policy_rejected.completion_mask,
                    policy_rejected.selection_mask,
                )
                rc, _, _ = pack_completion_values(
                    reference_chosen.token_logps, reference_chosen.completion_mask
                )
                rr, _, _ = pack_completion_values(
                    reference_rejected.token_logps, reference_rejected.completion_mask
                )
                common_length = min(pc.shape[1], pr.shape[1])
                pc, pc_valid, pc_selected = (
                    pc[:, :common_length], pc_valid[:, :common_length], pc_selected[:, :common_length]
                )
                pr, pr_valid, pr_selected = (
                    pr[:, :common_length], pr_valid[:, :common_length], pr_selected[:, :common_length]
                )
                rc, rr = rc[:, :common_length], rr[:, :common_length]
                valid = pc_valid & pr_valid
                if cfg.fpo_mask_combine == "union":
                    selected = pc_selected | pr_selected
                elif cfg.fpo_mask_combine == "intersection":
                    selected = pc_selected & pr_selected
                elif cfg.fpo_mask_combine == "chosen":
                    selected = pc_selected
                else:
                    selected = pr_selected
                result = fine_grained_preference_loss(
                    pc,
                    pr,
                    rc,
                    rr,
                    selected,
                    valid_mask=valid,
                    beta=cfg.beta,
                    label_smoothing=cfg.label_smoothing,
                    token_reduction=cfg.fpo_token_reduction,
                    sample_weight=sample_weight,
                )

        common_metrics = {
            **result.metrics,
            "chosen_logp": policy_chosen.mean_logps.detach().mean(),
            "rejected_logp": policy_rejected.mean_logps.detach().mean(),
            "completion_logp_margin": (
                policy_chosen.mean_logps - policy_rejected.mean_logps
            ).detach().mean(),
            "chosen_completion_tokens": policy_chosen.token_counts.float().mean().detach(),
            "rejected_completion_tokens": policy_rejected.token_counts.float().mean().detach(),
        }
        self._store_metrics(common_metrics, train=model.training)

        if not return_outputs:
            return result.loss
        example_logits = result.logits
        if example_logits.ndim != 1:
            example_logits = -result.per_example_loss
        return result.loss, {
            "loss": result.loss.detach(),
            "logits": example_logits.detach().unsqueeze(-1),
        }

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        split = "eval" if any(key.startswith("eval_") for key in logs) else "train"
        prefix = "eval_" if split == "eval" else ""
        buffered = self._preference_metrics[split]
        for name, values in buffered.items():
            if values:
                logs[f"{prefix}preference/{name}"] = sum(values) / len(values)
        buffered.clear()
        super().log(logs, start_time)


def _config_section(config: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name, {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"Config section {name!r} must be a mapping")
    return dict(value)


def _first_config_value(
    root: Mapping[str, Any],
    section: Mapping[str, Any],
    *names: str,
    default: Any = None,
) -> Any:
    for name in names:
        if name in section:
            return section[name]
        if name in root:
            return root[name]
    return default


def _load_yaml_mapping(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser()
    if not config_path.is_file():
        raise FileNotFoundError(f"Preference config does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, Mapping):
        raise TypeError(f"Preference config must contain a YAML mapping: {config_path}")
    return dict(loaded)


def _preference_config_from_mapping(config: Mapping[str, Any]) -> PreferenceTrainingConfig:
    section = _config_section(config, "preference")
    valid_names = {field.name for field in fields(PreferenceTrainingConfig)}
    values = {
        name: section[name] if name in section else config[name]
        for name in valid_names
        if name in section or name in config
    }
    unknown = set(section).difference(valid_names)
    if unknown:
        raise ValueError(f"Unknown preference config fields: {sorted(unknown)}")
    return PreferenceTrainingConfig(**values)


def _training_arguments_from_mapping(
    config: Mapping[str, Any],
    *,
    output_dir: str,
) -> tuple[TrainingArguments, str | bool | None]:
    section = _config_section(config, "training")
    valid_names = {field.name for field in fields(TrainingArguments)}
    for name in valid_names:
        if name in config and name not in section:
            section[name] = config[name]

    aliases = {
        "epochs": "num_train_epochs",
        "batch_size": "per_device_train_batch_size",
        "lr": "learning_rate",
    }
    for alias, canonical in aliases.items():
        alias_in_root = alias in config and alias not in section
        if alias_in_root:
            section[alias] = config[alias]
        if alias in section:
            if canonical in section:
                raise ValueError(f"Specify only one of training.{alias} and training.{canonical}")
            section[canonical] = section.pop(alias)

    resume = section.pop("resume_from_checkpoint", config.get("resume_from_checkpoint"))
    unknown = set(section).difference(valid_names)
    if unknown:
        raise ValueError(f"Unknown TrainingArguments fields: {sorted(unknown)}")

    defaults: dict[str, Any] = {
        "output_dir": output_dir,
        "num_train_epochs": 1.0,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "learning_rate": 5e-7,
        "logging_steps": 10,
        "save_steps": 500,
        "save_total_limit": 2,
        "report_to": "none",
        "remove_unused_columns": True,
    }
    defaults.update(section)
    defaults["output_dir"] = output_dir
    return TrainingArguments(**defaults), resume


def _load_raw_pair_dataset(config: Mapping[str, Any], objective: str):
    from datasets import DatasetDict, load_dataset, load_from_disk

    section = _config_section(config, "data")
    dataset_name = _first_config_value(
        config,
        section,
        "dataset",
        "dataset_name_or_path",
        "path",
    )
    if not isinstance(dataset_name, str) or not dataset_name.strip():
        raise ValueError(
            "Config requires data.dataset (or dataset_name_or_path) pointing to a "
            "load_from_disk directory or Hugging Face dataset"
        )
    split = str(_first_config_value(config, section, "split", "dataset_split", default="train"))
    local_path = Path(dataset_name).expanduser()
    if local_path.is_file():
        suffix = local_path.suffix.lower()
        builders = {
            ".json": "json",
            ".jsonl": "json",
            ".parquet": "parquet",
            ".csv": "csv",
        }
        try:
            builder = builders[suffix]
        except KeyError as error:
            raise ValueError(
                f"Unsupported local preference file {local_path}; use JSONL, JSON, "
                "Parquet, CSV, or a datasets.save_to_disk directory"
            ) from error
        dataset = load_dataset(builder, data_files=str(local_path), split=split)
    elif local_path.exists():
        dataset = load_from_disk(str(local_path))
        if isinstance(dataset, DatasetDict):
            if split not in dataset:
                raise KeyError(f"Local DatasetDict has no split {split!r}; available: {list(dataset)}")
            dataset = dataset[split]
    else:
        dataset_config_name = section.get("config_name")
        dataset = load_dataset(dataset_name, dataset_config_name, split=split)

    if not hasattr(dataset, "__len__") or len(dataset) == 0:
        raise ValueError("Preference dataset must be non-empty and map-style")
    first = dataset[0]
    if not isinstance(first, Mapping):
        raise TypeError("Preference dataset rows must be mappings")
    required = {"prompt_input_ids", "chosen_input_ids", "rejected_input_ids"}
    missing = required.difference(first)
    if missing:
        raise KeyError(f"Preference dataset is missing required columns: {sorted(missing)}")
    if objective == "fpo" and not {
        "selection_mask",
        "chosen_selection_mask",
        "rejected_selection_mask",
    }.intersection(first):
        raise KeyError(
            "FPO requires selection_mask or branch-specific chosen/rejected_selection_mask"
        )
    return dataset


def _model_load_kwargs(config: Mapping[str, Any]) -> dict[str, Any]:
    section = _config_section(config, "model")
    kwargs: dict[str, Any] = {}
    for name in ("revision", "trust_remote_code", "attn_implementation", "low_cpu_mem_usage"):
        if name in section:
            kwargs[name] = section[name]

    dtype_name = _first_config_value(config, section, "dtype", "torch_dtype")
    if dtype_name is not None:
        if isinstance(dtype_name, torch.dtype):
            dtype = dtype_name
        else:
            dtype_map = {
                "auto": "auto",
                "float32": torch.float32,
                "fp32": torch.float32,
                "float16": torch.float16,
                "fp16": torch.float16,
                "bfloat16": torch.bfloat16,
                "bf16": torch.bfloat16,
            }
            try:
                dtype = dtype_map[str(dtype_name).lower()]
            except KeyError as error:
                raise ValueError(
                    "model.dtype must be auto, float32/fp32, float16/fp16, or bfloat16/bf16"
                ) from error
        kwargs["torch_dtype"] = dtype
    return kwargs


def _sample_max_token_id(row: Mapping[str, Any]) -> int:
    maximum = -1
    for name in ("prompt_input_ids", "chosen_input_ids", "rejected_input_ids"):
        value = row[name]
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().tolist()
        if not value:
            raise ValueError(f"First dataset row has an empty {name}")
        try:
            maximum = max(maximum, max(int(token) for token in value))
        except (TypeError, ValueError) as error:
            raise TypeError(f"First dataset row has non-integer values in {name}") from error
    return maximum


def train_from_config(config: Mapping[str, Any] | str | Path) -> str:
    """Train from a YAML path or equivalent mapping and return the final path.

    Recommended YAML structure::

        model:
          name_or_path: /path/to/sft-checkpoint
          reference_name_or_path: /path/to/sft-checkpoint  # DPO/FPO
          tokenizer_name_or_path: LiquidAI/LFM2-350M
          dtype: bfloat16
        data:
          dataset: /path/to/raw-preference-dataset
          split: train
          pad_token_id: 64407
          max_length: 8192
        preference:
          objective: dpo  # dpo, spo, or fpo
          beta: 0.1
        training:
          output_dir: output/preference
          num_train_epochs: 1
          per_device_train_batch_size: 1

    Flat aliases such as ``model_name_or_path``, ``dataset``, ``output_dir``,
    ``epochs``, ``batch_size``, and ``lr`` are accepted for small configs.
    """

    root = _load_yaml_mapping(config) if isinstance(config, (str, Path)) else dict(config)
    if not isinstance(root, Mapping):
        raise TypeError("config must be a mapping or YAML path")
    model_section = _config_section(root, "model")
    data_section = _config_section(root, "data")
    training_section = _config_section(root, "training")
    preference_config = _preference_config_from_mapping(root)

    model_name = _first_config_value(
        root, model_section, "name_or_path", "model_name_or_path", "model_name"
    )
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("Config requires model.name_or_path (or model_name_or_path)")
    tokenizer_name = _first_config_value(
        root,
        model_section,
        "tokenizer_name_or_path",
        "tokenizer_name",
        default=model_name,
    )
    reference_name = _first_config_value(
        root,
        model_section,
        "reference_name_or_path",
        "reference_model_name_or_path",
        "reference_model_name",
        default=model_name,
    )
    output_dir = _first_config_value(
        root, training_section, "output_dir", default=root.get("output_dir")
    )
    if not isinstance(output_dir, str) or not output_dir.strip():
        raise ValueError("Config requires training.output_dir (or output_dir)")

    dataset = _load_raw_pair_dataset(root, preference_config.objective)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        revision=model_section.get("revision", "main"),
        trust_remote_code=bool(model_section.get("trust_remote_code", False)),
    )
    model_kwargs = _model_load_kwargs(root)
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    reference_model = None
    if preference_config.objective in {"dpo", "fpo"}:
        if not isinstance(reference_name, str) or not reference_name.strip():
            raise ValueError("DPO/FPO requires model.reference_name_or_path")
        reference_model = AutoModelForCausalLM.from_pretrained(reference_name, **model_kwargs)

    embeddings = model.get_input_embeddings()
    if embeddings is None:
        raise ValueError("Policy model must expose input embeddings")
    vocabulary_size = int(embeddings.num_embeddings)
    first_row = dataset[0]
    first_max_id = _sample_max_token_id(first_row)
    if first_max_id >= vocabulary_size:
        raise ValueError(
            f"Dataset token ID {first_max_id} exceeds policy vocabulary size {vocabulary_size}"
        )
    if reference_model is not None:
        reference_embeddings = reference_model.get_input_embeddings()
        if reference_embeddings is None or reference_embeddings.num_embeddings != vocabulary_size:
            raise ValueError("Policy and reference models must have identical vocabulary sizes")

    pad_token_id = _first_config_value(root, data_section, "pad_token_id")
    if pad_token_id is None:
        pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        raise ValueError("Config must set data.pad_token_id when tokenizer has no pad token")
    if isinstance(pad_token_id, bool) or not isinstance(pad_token_id, int):
        raise TypeError("data.pad_token_id must be an integer")
    if not 0 <= pad_token_id < vocabulary_size:
        raise ValueError(
            f"pad_token_id {pad_token_id} is outside policy vocabulary size {vocabulary_size}"
        )
    model.config.pad_token_id = pad_token_id
    if reference_model is not None:
        reference_model.config.pad_token_id = pad_token_id
    if len(tokenizer) != vocabulary_size:
        warnings.warn(
            f"Tokenizer length {len(tokenizer)} differs from model vocabulary {vocabulary_size}; "
            "provide the expanded training tokenizer so final checkpoints are self-contained.",
            stacklevel=2,
        )

    max_length = _first_config_value(root, data_section, "max_length")
    if max_length is not None and (
        isinstance(max_length, bool) or not isinstance(max_length, int)
    ):
        raise TypeError("data.max_length must be an integer")
    collator = RawTokenPreferenceCollator(
        pad_token_id=pad_token_id,
        max_length=max_length,
        padding_side=str(data_section.get("padding_side", "right")),
        require_selection=preference_config.objective == "fpo",
    )
    training_args, resume_from_checkpoint = _training_arguments_from_mapping(
        root, output_dir=output_dir
    )
    trainer = RawTokenPreferenceTrainer(
        model=model,
        reference_model=reference_model,
        preference_config=preference_config,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    final_dir = Path(output_dir) / "final"
    trainer.save_model(str(final_dir))
    trainer.accelerator.wait_for_everyone()
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(final_dir)
        with (final_dir / "post_training_config.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(root, handle, sort_keys=False)
    trainer.accelerator.wait_for_everyone()
    return str(final_dir)


def build_cli_parser() -> argparse.ArgumentParser:
    """Build the command-line parser without starting a training job."""

    parser = argparse.ArgumentParser(
        description="Train raw speech-token pairs with DPO, 2025 Self-PO, or FPO.",
        epilog=(
            "Dataset keys: prompt_input_ids, chosen_input_ids, rejected_input_ids; "
            "FPO also needs selection_mask or branch-specific selection masks."
        ),
    )
    parser.add_argument("--config", required=True, help="Path to the YAML training config")
    return parser


def main(argv: list[str] | None = None) -> str:
    """CLI entry point for ``python -m ...post_training.preference``."""

    args = build_cli_parser().parse_args(argv)
    final_dir = train_from_config(args.config)
    print(f"Preference-trained checkpoint saved to {final_dir}")
    return final_dir


# Concise aliases matching common preference-optimization terminology.
dpo_loss = direct_preference_loss
spo_loss = self_preference_loss
fpo_loss = fine_grained_preference_loss
PreferenceTrainer = RawTokenPreferenceTrainer


__all__ = [
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
]


if __name__ == "__main__":
    main()
