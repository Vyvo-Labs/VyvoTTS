"""On-policy REINFORCE and GRPO for discrete-token VyvoTTS models.

The rollout path intentionally operates on raw token IDs. Generated codec
tokens are grammar constrained, decoded to waveforms, and scored by frozen ASR,
speaker, and quality models before their rewards are converted to advantages.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from accelerate import Accelerator
from datasets import DatasetDict, load_dataset, load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList

from vyvotts.audio_utils import decode_audio_value
from vyvotts.codec import load_codec
from vyvotts.inference.constraints import (
    AudioTokenLogitsProcessor,
    AudioTokenSequenceError,
    extract_audio_codes,
)
from vyvotts.train.post_training.common import completion_log_probs


@dataclass
class OnlineTrainingConfig:
    """Optimization settings shared by REINFORCE and GRPO."""

    method: str = "grpo"
    group_size: int = 8
    max_steps: int = 1000
    max_new_tokens: int = 1200
    min_audio_frames: int = 4
    temperature: float = 0.7
    top_p: float = 0.95
    learning_rate: float = 1e-5
    weight_decay: float = 0.0
    clip_epsilon: float = 0.2
    kl_beta: float = 0.0
    policy_updates: int = 1
    max_grad_norm: float = 1.0
    normalize_advantages: bool = False
    advantage_epsilon: float = 1e-4
    reinforce_baseline_momentum: float = 0.9
    save_steps: int = 100
    seed: int = 42

    def __post_init__(self) -> None:
        self.method = self.method.lower()
        if self.method not in {"reinforce", "rl", "grpo"}:
            raise ValueError("method must be 'reinforce', 'rl', or 'grpo'")
        if self.group_size < (2 if self.method == "grpo" else 1):
            raise ValueError("GRPO requires group_size >= 2")
        if self.max_steps <= 0 or self.max_new_tokens <= 0:
            raise ValueError("max_steps and max_new_tokens must be positive")
        if self.policy_updates <= 0:
            raise ValueError("policy_updates must be positive")
        if not 0.0 <= self.clip_epsilon < 1.0:
            raise ValueError("clip_epsilon must be in [0, 1)")
        if self.kl_beta < 0:
            raise ValueError("kl_beta must be non-negative")


def group_relative_advantages(
    rewards: torch.Tensor,
    *,
    normalize: bool = False,
    epsilon: float = 1e-4,
) -> torch.Tensor:
    """Center rewards within each prompt group, optionally scaling by std."""

    if rewards.ndim != 1 or rewards.numel() < 2:
        raise ValueError("group rewards must be a one-dimensional tensor with at least two values")
    advantages = rewards - rewards.mean()
    if normalize:
        std = rewards.std(unbiased=False)
        if std > epsilon:
            advantages = advantages / std
    return advantages


def reinforce_advantages(rewards: torch.Tensor, baseline: float) -> torch.Tensor:
    """Return sequence advantages relative to a running scalar baseline."""

    if rewards.ndim != 1 or rewards.numel() == 0:
        raise ValueError("rewards must be a non-empty one-dimensional tensor")
    return rewards - float(baseline)


def validate_online_distributed_type(distributed_type: Any) -> None:
    """Reject parameter-sharded engines that cannot safely run ``generate``.

    Online rollout generation unwraps ordinary DDP so each rank can sample
    from its complete local model. FSDP, DeepSpeed ZeRO-3, and Megatron-LM do
    not guarantee complete parameters after unwrapping.
    """

    name = getattr(distributed_type, "name", distributed_type)
    normalized = str(name).upper().rsplit(".", 1)[-1]
    unsupported = {"FSDP", "DEEPSPEED", "MEGATRON_LM"}
    if normalized in unsupported:
        raise RuntimeError(
            "online waveform-scored RL/GRPO currently supports a single process "
            "or unsharded data parallelism (DDP), not "
            f"{normalized}; disable FSDP/DeepSpeed/Megatron for this stage"
        )


def sampled_token_kl(policy_logps: torch.Tensor, reference_logps: torch.Tensor) -> torch.Tensor:
    """Positive sampled-token KL estimator used by common GRPO recipes."""

    if policy_logps.shape != reference_logps.shape:
        raise ValueError("policy and reference log-probabilities must have equal shapes")
    # Extreme probabilities can overflow exp() before a padding mask is
    # applied.  Clipping only the estimator's log-ratio keeps it finite without
    # changing the policy log-probabilities used by the gradient objective.
    log_ratio = (reference_logps - policy_logps).clamp(-20.0, 20.0)
    return torch.exp(log_ratio) - log_ratio - 1.0


def clipped_policy_loss(
    policy_logps: torch.Tensor,
    old_logps: torch.Tensor,
    token_mask: torch.Tensor,
    advantages: torch.Tensor,
    *,
    clip_epsilon: float = 0.2,
    reference_logps: torch.Tensor | None = None,
    kl_beta: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute a length-normalized clipped policy-gradient objective."""

    if policy_logps.shape != old_logps.shape or token_mask.shape != policy_logps.shape:
        raise ValueError("log-probability and token-mask tensors must have equal [B, T] shapes")
    if advantages.ndim != 1 or advantages.shape[0] != policy_logps.shape[0]:
        raise ValueError("advantages must have one value per sequence")

    mask = token_mask.to(dtype=torch.bool, device=policy_logps.device)
    if (mask.sum(dim=-1) == 0).any():
        raise ValueError("every rollout must contain at least one completion token")

    log_ratio = (policy_logps - old_logps).clamp(-20.0, 20.0)
    ratio = torch.exp(log_ratio)
    advantage = advantages.to(policy_logps.device).unsqueeze(-1)
    surrogate = ratio * advantage
    if clip_epsilon > 0:
        clipped_ratio = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon)
        surrogate = torch.minimum(surrogate, clipped_ratio * advantage)

    per_sequence = -(surrogate * mask).sum(dim=-1) / mask.sum(dim=-1)
    policy_loss = per_sequence.mean()

    if reference_logps is None or kl_beta == 0:
        kl = policy_logps.new_zeros(())
    else:
        token_kl = sampled_token_kl(policy_logps, reference_logps).masked_fill(~mask, 0.0)
        kl = (token_kl.sum(dim=-1) / mask.sum(dim=-1)).mean()

    total = policy_loss + kl_beta * kl
    clip_fraction = (
        (((ratio - 1.0).abs() > clip_epsilon) & mask).float().sum() / mask.sum()
        if clip_epsilon > 0
        else ratio.new_zeros(())
    )
    return total, {
        "policy_loss": policy_loss.detach(),
        "kl": kl.detach(),
        "clip_fraction": clip_fraction.detach(),
        "ratio_mean": ratio.masked_select(mask).mean().detach(),
    }


def build_tts_prompt(
    tokenizer,
    text: str,
    token_config: Mapping[str, int],
    speaker: str | None = None,
) -> list[int]:
    """Build the prompt format used by VyvoTTS preprocessing and inference."""

    text = text.strip()
    if not text:
        raise ValueError("target text must not be empty")
    prompt_text = f"{speaker}: {text}" if speaker else text
    text_ids = tokenizer.encode(prompt_text, add_special_tokens=True)
    text_ids.append(int(token_config["END_OF_TEXT"]))
    return [int(token_config["START_OF_HUMAN"])] + text_ids + [
        int(token_config["END_OF_HUMAN"])
    ]


def _load_dataset(path: str, split: str = "train"):
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


def _reference_waveform(value: Any, target_rate: int) -> torch.Tensor | None:
    if value is None:
        return None
    waveform, sample_rate = decode_audio_value(
        value, default_sample_rate=target_rate
    )

    if waveform.ndim == 2:
        waveform = waveform.mean(dim=-1 if waveform.shape[-1] <= 8 else 0)
    waveform = waveform.flatten()
    if sample_rate != target_rate:
        import torchaudio.functional as AF

        waveform = AF.resample(waveform, sample_rate, target_rate)
    return waveform


class OnlinePolicyTrainer:
    """Small, explicit on-policy trainer for waveform-scored TTS rollouts."""

    def __init__(
        self,
        *,
        model,
        tokenizer,
        dataset,
        codec,
        token_config: Mapping[str, int],
        reward_fn: Any,
        output_dir: str,
        config: OnlineTrainingConfig,
        text_column: str = "reference_text",
        speaker_column: str = "speaker_id",
        reference_audio_column: str | None = None,
        reference_model=None,
        accelerator: Accelerator | None = None,
    ) -> None:
        self.accelerator = accelerator or Accelerator()
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.codec = codec
        self.tokens = {key: int(value) for key, value in token_config.items()}
        self.reward_fn = reward_fn
        self.output_dir = Path(output_dir)
        self.config = config
        self.text_column = text_column
        self.speaker_column = speaker_column
        self.reference_audio_column = reference_audio_column
        self.reference_model = reference_model
        self._reinforce_baseline: float | None = None

        if text_column not in dataset.column_names:
            raise ValueError(
                f"online dataset needs a {text_column!r} column; retain reference text "
                "during tokenization"
            )
        if config.kl_beta > 0 and reference_model is None:
            raise ValueError("kl_beta > 0 requires a frozen reference_model")
        if reference_model is not None:
            reference_model.eval()
            reference_model.requires_grad_(False)

    def _score(self, audio: torch.Tensor, row: Mapping[str, Any]) -> dict[str, float]:
        reference = None
        if self.reference_audio_column:
            reference = _reference_waveform(
                row.get(self.reference_audio_column), self.codec.sample_rate
            )
        result = self.reward_fn.score(
            audio=audio.detach().float().cpu().flatten(),
            target_text=str(row[self.text_column]),
            sample_rate=self.codec.sample_rate,
            reference_audio=reference,
        )
        if isinstance(result, Mapping):
            metrics = {str(key): float(value) for key, value in result.items()}
            if "total" not in metrics:
                raise ValueError("reward mapping must contain a 'total' value")
            return metrics
        return {"total": float(result)}

    def _rollout(self, row: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, float]]]:
        speaker = row.get(self.speaker_column) if self.speaker_column else None
        speaker = str(speaker).strip() if speaker else None
        prompt = build_tts_prompt(
            self.tokenizer, str(row[self.text_column]), self.tokens, speaker
        )
        prompt_ids = torch.tensor([prompt], dtype=torch.long, device=self.accelerator.device)
        prompt_ids = prompt_ids.repeat(self.config.group_size, 1)
        prompt_mask = torch.ones_like(prompt_ids)

        processor = AudioTokenLogitsProcessor(
            prompt_length=len(prompt),
            start_of_ai=self.tokens["START_OF_AI"],
            start_of_speech=self.tokens["START_OF_SPEECH"],
            end_of_speech=self.tokens["END_OF_SPEECH"],
            audio_tokens_start=self.tokens["AUDIO_TOKENS_START"],
            codes_per_group=self.codec.codes_per_group,
            codebook_size=self.codec.codebook_size,
            pad_token_id=self.tokens["PAD_TOKEN"],
            min_audio_frames=self.config.min_audio_frames,
        )

        unwrapped = self.accelerator.unwrap_model(self.model)
        unwrapped.eval()
        with torch.inference_mode():
            generated = unwrapped.generate(
                input_ids=prompt_ids,
                attention_mask=prompt_mask,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=True,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                repetition_penalty=1.0,
                eos_token_id=self.tokens["END_OF_SPEECH"],
                pad_token_id=self.tokens["PAD_TOKEN"],
                logits_processor=LogitsProcessorList([processor]),
                use_cache=True,
            )

        completion_mask = torch.zeros_like(generated, dtype=torch.bool)
        completion_mask[:, len(prompt) :] = generated[:, len(prompt) :].ne(
            self.tokens["PAD_TOKEN"]
        )

        metrics: list[dict[str, float]] = []
        for sequence in generated:
            try:
                codes = extract_audio_codes(
                    sequence,
                    self.tokens["START_OF_SPEECH"],
                    self.tokens["END_OF_SPEECH"],
                    self.tokens["AUDIO_TOKENS_START"],
                    self.codec.codes_per_group,
                    self.codec.codebook_size,
                    strict=True,
                )
                audio = self.codec.decode(codes, device="cpu")
                if audio is None:
                    raise AudioTokenSequenceError("codec produced no waveform")
            except AudioTokenSequenceError as exc:
                score = {"total": -1.0, "codec_valid": 0.0, "error": str(exc)}
            else:
                # Reward backend/configuration failures are systemic.  Let them
                # stop training rather than silently assigning -1 to every
                # rollout and producing a zero-advantage GRPO group.
                score = self._score(audio, row)
                score["codec_valid"] = 1.0
            metrics.append(score)

        return generated, completion_mask, metrics

    def _forward_logps(self, model, input_ids: torch.Tensor, completion_mask: torch.Tensor):
        attention_mask = input_ids.ne(self.tokens["PAD_TOKEN"])
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        labels = input_ids.masked_fill(~completion_mask, -100)
        return completion_log_probs(
            outputs.logits,
            labels,
            completion_mask=completion_mask,
        ).token_logps

    def _advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        if self.config.method == "grpo":
            return group_relative_advantages(
                rewards,
                normalize=self.config.normalize_advantages,
                epsilon=self.config.advantage_epsilon,
            )

        mean_reward = float(rewards.mean().item())
        if self._reinforce_baseline is None:
            self._reinforce_baseline = mean_reward
        advantages = reinforce_advantages(rewards, self._reinforce_baseline)
        momentum = self.config.reinforce_baseline_momentum
        self._reinforce_baseline = momentum * self._reinforce_baseline + (1 - momentum) * mean_reward
        return advantages

    def _save(self, name: str) -> None:
        self.accelerator.wait_for_everyone()
        target = self.output_dir / name
        if self.accelerator.is_main_process:
            target.mkdir(parents=True, exist_ok=True)
        self.accelerator.wait_for_everyone()
        # FSDP/DeepSpeed state gathering is collective and must run on every
        # rank even though only the main process writes checkpoint files.
        state_dict = self.accelerator.get_state_dict(self.model)
        model = self.accelerator.unwrap_model(self.model)
        model.save_pretrained(
            target,
            is_main_process=self.accelerator.is_main_process,
            save_function=self.accelerator.save,
            state_dict=state_dict,
        )
        if self.accelerator.is_main_process:
            self.tokenizer.save_pretrained(target)
        self.accelerator.wait_for_everyone()

    def train(self) -> Path:
        validate_online_distributed_type(self.accelerator.distributed_type)
        torch.manual_seed(self.config.seed + self.accelerator.process_index)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        loader = DataLoader(
            self.dataset, batch_size=1, shuffle=True, collate_fn=lambda rows: rows[0]
        )
        self.model, optimizer, loader = self.accelerator.prepare(
            self.model, optimizer, loader
        )
        if self.reference_model is not None:
            self.reference_model.to(self.accelerator.device)

        data_iterator = iter(loader)
        for step in range(1, self.config.max_steps + 1):
            try:
                row = next(data_iterator)
            except StopIteration:
                data_iterator = iter(loader)
                row = next(data_iterator)

            generated, completion_mask, reward_metrics = self._rollout(row)
            generated = generated.to(self.accelerator.device)
            completion_mask = completion_mask.to(self.accelerator.device)
            rewards = torch.tensor(
                [item["total"] for item in reward_metrics],
                dtype=torch.float32,
                device=self.accelerator.device,
            )
            if not any(item.get("codec_valid") == 1.0 for item in reward_metrics):
                raise RuntimeError(
                    "all rollouts are codec-invalid; increase max_new_tokens or inspect "
                    "the constrained generation/token configuration"
                )
            advantages = self._advantages(rewards).detach()

            self.model.eval()
            with torch.no_grad():
                old_logps = self._forward_logps(
                    self.model, generated, completion_mask
                ).detach()
                reference_logps = None
                if self.reference_model is not None:
                    reference_logps = self._forward_logps(
                        self.reference_model, generated, completion_mask
                    ).detach()

            self.model.train()
            last_stats: dict[str, torch.Tensor] = {}
            for _ in range(self.config.policy_updates):
                optimizer.zero_grad(set_to_none=True)
                policy_logps = self._forward_logps(
                    self.model, generated, completion_mask
                )
                loss, last_stats = clipped_policy_loss(
                    policy_logps,
                    old_logps,
                    completion_mask[:, 1:],
                    advantages,
                    clip_epsilon=self.config.clip_epsilon,
                    reference_logps=reference_logps,
                    kl_beta=self.config.kl_beta,
                )
                self.accelerator.backward(loss)
                self.accelerator.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm
                )
                optimizer.step()

            if self.accelerator.is_main_process:
                components: dict[str, float] = {}
                for key in reward_metrics[0]:
                    if key == "error":
                        continue
                    values = [item.get(key) for item in reward_metrics]
                    values = [value for value in values if isinstance(value, (int, float))]
                    if values:
                        components[f"reward/{key}"] = sum(values) / len(values)
                log = {
                    "step": step,
                    "reward/mean": float(rewards.mean()),
                    "reward/min": float(rewards.min()),
                    "reward/max": float(rewards.max()),
                    "advantage/std": float(advantages.std(unbiased=False)),
                    **{key: float(value) for key, value in last_stats.items()},
                    **components,
                }
                print(json.dumps(log, sort_keys=True))

            if self.config.save_steps and step % self.config.save_steps == 0:
                self._save(f"checkpoint-{step}")

        self._save("final")
        return self.output_dir / "final"


def _load_yaml(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description="ASR/quality-guided RL or GRPO for VyvoTTS")
    parser.add_argument("--config", required=True, help="Online post-training YAML")
    args = parser.parse_args()
    raw = _load_yaml(args.config)

    training = OnlineTrainingConfig(**raw.get("training", {}))
    token_config = _load_yaml(raw["inference_config"])
    accelerator = Accelerator()
    dtype = torch.bfloat16 if accelerator.device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(raw["model_name"], torch_dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(raw.get("tokenizer_name", raw["model_name"]))
    codec = load_codec(
        codec_type=raw.get("codec_type", token_config.get("CODEC_TYPE", "mimi")),
        model_name=raw.get("codec_model_name", token_config.get("CODEC_MODEL")),
        device=raw.get("codec_device", "cpu"),
        **(
            {"num_codebooks": token_config.get("NUM_CODEBOOKS", 8)}
            if raw.get("codec_type", token_config.get("CODEC_TYPE", "mimi")) == "mimi"
            else {}
        ),
    )

    reference_model = None
    if training.kl_beta > 0:
        reference_name = raw.get("reference_model_name", raw["model_name"])
        reference_model = AutoModelForCausalLM.from_pretrained(
            reference_name, torch_dtype=dtype
        )

    from vyvotts.train.post_training.rewards import build_composite_reward

    reward_fn = build_composite_reward(raw.get("reward", {}))
    dataset = _load_dataset(raw["dataset"], raw.get("split", "train"))
    trainer = OnlinePolicyTrainer(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        codec=codec,
        token_config=token_config,
        reward_fn=reward_fn,
        output_dir=raw["output_dir"],
        config=training,
        text_column=raw.get("text_column", "reference_text"),
        speaker_column=raw.get("speaker_column", "speaker_id"),
        reference_audio_column=raw.get("reference_audio_column"),
        reference_model=reference_model,
        accelerator=accelerator,
    )
    trainer.train()


if __name__ == "__main__":
    main()


__all__ = [
    "OnlinePolicyTrainer",
    "OnlineTrainingConfig",
    "build_tts_prompt",
    "clipped_policy_loss",
    "group_relative_advantages",
    "reinforce_advantages",
    "sampled_token_kl",
    "validate_online_distributed_type",
]
