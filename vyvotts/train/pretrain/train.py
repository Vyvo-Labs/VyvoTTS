"""Pretrain a causal language model on a scheduled QA/TTS mixture."""

from __future__ import annotations

import argparse
import inspect
import math
import os
import warnings
from collections.abc import Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import Any

import torch
import yaml
from datasets import load_dataset, load_from_disk
from torch.utils.data import Dataset, Sampler
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint

DEFAULT_CONFIG_FILE = (
    Path(__file__).resolve().parents[2] / "configs" / "train" / "lfm2_5_pretrain.yaml"
)


# ---------------------------------------------------------------------------
# Config and token helpers
# ---------------------------------------------------------------------------
def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a pretraining YAML configuration."""
    with Path(config_path).expanduser().open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise TypeError(f"Expected a YAML mapping in {config_path}")
    return config


def parse_ratio(value: str | float) -> float:
    """Parse a positive QA:TTS ratio into QA batches per TTS batch."""
    if isinstance(value, str):
        parts = value.split(":")
        if len(parts) == 1:
            ratio = float(parts[0])
        elif len(parts) == 2:
            numerator, denominator = (float(part) for part in parts)
            if denominator <= 0:
                raise ValueError("The TTS side of ratio must be greater than zero")
            ratio = numerator / denominator
        else:
            raise ValueError(f"Invalid ratio {value!r}; expected a value such as '2:1'")
    else:
        ratio = float(value)

    if not math.isfinite(ratio) or ratio <= 0:
        raise ValueError("QA:TTS ratio must be a finite value greater than zero")
    return ratio


def interpolate_ratio(initial: float, final: float, step: int, total_steps: int) -> float:
    """Linearly interpolate a ratio, reaching the target on the last step."""
    if total_steps <= 1:
        return final if step > 0 else initial
    progress = min(max(step, 0) / (total_steps - 1), 1.0)
    return initial + (final - initial) * progress


def audio_token_capacity(codec_type: str, num_codebooks: int | None = None) -> int:
    """Return the number of codec token values used by the configured codec."""
    codec_type = codec_type.lower()
    if codec_type == "mimi":
        codebooks = 8 if num_codebooks is None else int(num_codebooks)
        if not 1 <= codebooks <= 32:
            raise ValueError("Mimi num_codebooks must be between 1 and 32")
        return codebooks * 2048
    if codec_type == "snac":
        return 7 * 4096
    raise ValueError(f"Unknown codec_type: {codec_type!r}")


def extend_tokenizer(tokenizer, audio_capacity: int, pad_token_id: int) -> int:
    """Install the repository's stable custom-token range and configure padding.

    Custom indices 0--9 are reserved for control tokens; codec values start at
    custom index 10. The inclusive upper token is retained for compatibility
    with existing VyvoTTS checkpoints.

    Returns:
        The number of tokens newly added to ``tokenizer``.
    """
    custom_tokens = [f"<custom_token_{index}>" for index in range(audio_capacity + 11)]
    added = int(tokenizer.add_tokens(custom_tokens))

    custom_start = int(tokenizer.convert_tokens_to_ids("<custom_token_0>"))
    pad_token = tokenizer.convert_ids_to_tokens(int(pad_token_id))
    if int(pad_token_id) != custom_start + 7 or pad_token != "<custom_token_7>":
        raise ValueError(
            f"pad_token={pad_token_id} resolves to {pad_token!r}, not a VyvoTTS "
            "PAD token at custom index 7. Check that the training and inference "
            "configs use the same base tokenizer."
        )
    if int(tokenizer.convert_tokens_to_ids("<custom_token_10>")) != custom_start + 10:
        raise ValueError("VyvoTTS custom tokens are not stored in a contiguous ID range")
    tokenizer.pad_token = pad_token
    return added


def resolve_resume_checkpoint(
    resume_from_checkpoint: str | bool | None,
    output_dir: str | Path,
) -> str | None:
    """Resolve explicit or automatic Trainer checkpoint resumption."""
    if resume_from_checkpoint in (None, False, "", "false", "False"):
        return None
    if resume_from_checkpoint is True or str(resume_from_checkpoint).lower() in {
        "true",
        "auto",
        "latest",
    }:
        checkpoint = get_last_checkpoint(str(output_dir))
        if checkpoint is None:
            raise ValueError(f"No Trainer checkpoint found under {output_dir}")
        return checkpoint

    checkpoint = Path(str(resume_from_checkpoint)).expanduser()
    if not checkpoint.is_dir():
        raise ValueError(f"Checkpoint directory does not exist: {checkpoint}")
    return str(checkpoint)


def calculate_total_steps(
    dataset_size: int,
    per_device_batch_size: int,
    world_size: int,
    gradient_accumulation_steps: int,
    epochs: float,
    max_steps: int = -1,
    drop_last: bool = False,
) -> int:
    """Match Trainer's ceil-based optimizer-step calculation."""
    if max_steps > 0:
        return int(max_steps)
    global_micro_batch = per_device_batch_size * world_size
    if drop_last:
        micro_batches = dataset_size // global_micro_batch
    else:
        micro_batches = math.ceil(dataset_size / global_micro_batch)
    if micro_batches == 0:
        raise ValueError("Dataset is smaller than one global batch with drop_last enabled")
    updates_per_epoch = max(
        1, math.ceil(micro_batches / max(1, gradient_accumulation_steps))
    )
    return max(1, math.ceil(updates_per_epoch * epochs))


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------
def _load_ds(path: str):
    local_path = Path(path).expanduser()
    if local_path.exists():
        return load_from_disk(str(local_path))
    return load_dataset(path, split="train")


def _ratio_batch_counts(ratio: float) -> tuple[int, int]:
    """Approximate a ratio with a compact, deterministic batch cycle."""
    from fractions import Fraction

    fraction = Fraction(ratio).limit_denominator(16)
    return fraction.numerator, fraction.denominator


class GradualRatioDataset(Dataset):
    """Mix QA and TTS examples in homogeneous global batches.

    ``batch_total`` must be the launched world size multiplied by the
    per-device batch size. Trainer/Accelerate then shards each global block
    across workers without mixing QA and TTS samples in one synchronized step.
    """

    def __init__(
        self,
        dataset1,
        dataset2,
        batch_total: int,
        initial_ratio: float = 2,
        final_ratio: float = 1,
        total_steps: int | None = None,
    ):
        if len(dataset1) == 0 or len(dataset2) == 0:
            raise ValueError("Both QA and TTS datasets must contain at least one example")
        if batch_total <= 0:
            raise ValueError("batch_total must be greater than zero")

        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.batch_total = int(batch_total)
        self.initial_ratio = parse_ratio(initial_ratio)
        self.final_ratio = parse_ratio(final_ratio)
        self.total_steps = total_steps
        self.current_step = 0

        max_qa, max_tts = _ratio_batch_counts(max(self.initial_ratio, self.final_ratio))
        qa_cycles = len(dataset1) // (self.batch_total * max_qa)
        tts_cycles = len(dataset2) // (self.batch_total * max_tts)
        self.num_cycles = min(qa_cycles, tts_cycles)
        if self.num_cycles == 0:
            minimum_qa = self.batch_total * max_qa
            minimum_tts = self.batch_total * max_tts
            raise ValueError(
                "Datasets are too small for one complete global mixing cycle: "
                f"need at least {minimum_qa} QA and {minimum_tts} TTS examples"
            )

        initial_qa, initial_tts = _ratio_batch_counts(self.initial_ratio)
        self.length = self.num_cycles * (initial_qa + initial_tts) * self.batch_total

    def set_current_step(self, step: int) -> None:
        self.current_step = max(0, int(step))

    def get_current_ratio(self) -> float:
        if not self.total_steps:
            return self.initial_ratio
        return interpolate_ratio(
            self.initial_ratio,
            self.final_ratio,
            self.current_step,
            self.total_steps,
        )

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        if index < 0:
            index += self.length
        if index < 0 or index >= self.length:
            raise IndexError(index)

        qa_batches, tts_batches = _ratio_batch_counts(self.get_current_ratio())
        batches_per_cycle = qa_batches + tts_batches
        global_batch = index // self.batch_total
        sample_in_batch = index % self.batch_total
        cycle, batch_in_cycle = divmod(global_batch, batches_per_cycle)

        if batch_in_cycle < qa_batches:
            source_batch = cycle * qa_batches + batch_in_cycle
            source_index = source_batch * self.batch_total + sample_in_batch
            return self.dataset1[source_index % len(self.dataset1)]

        source_batch = cycle * tts_batches + (batch_in_cycle - qa_batches)
        source_index = source_batch * self.batch_total + sample_in_batch
        return self.dataset2[source_index % len(self.dataset2)]


# ---------------------------------------------------------------------------
# Trainer and collator
# ---------------------------------------------------------------------------
class GlobalBatchShuffleSampler(Sampler[int]):
    """Shuffle global batches while keeping their examples contiguous."""

    def __init__(self, dataset, global_batch_size: int, seed: int = 0):
        if global_batch_size <= 0:
            raise ValueError("global_batch_size must be greater than zero")
        self.dataset = dataset
        self.global_batch_size = int(global_batch_size)
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.dataset)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        num_blocks = math.ceil(len(self.dataset) / self.global_batch_size)
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        block_order = torch.randperm(num_blocks, generator=generator).tolist()
        for block in block_order:
            start = block * self.global_batch_size
            stop = min(start + self.global_batch_size, len(self.dataset))
            yield from range(start, stop)


class FSDPTrainer(Trainer):
    """Trainer that preserves the ordered global-batch mixing schedule.

    Actual process sharding and FSDP wrapping are delegated to Trainer's
    Accelerator. This is also valid for a single CPU/GPU process.
    """

    def __init__(self, *args, initial_ratio: float = 2, final_ratio: float = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_ratio = parse_ratio(initial_ratio)
        self.final_ratio = parse_ratio(final_ratio)
        self.total_steps = calculate_total_steps(
            dataset_size=len(self.train_dataset),
            per_device_batch_size=self.args.per_device_train_batch_size,
            world_size=self.args.world_size,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            epochs=self.args.num_train_epochs,
            max_steps=self.args.max_steps,
            drop_last=self.args.dataloader_drop_last,
        )
        if hasattr(self.train_dataset, "total_steps"):
            self.train_dataset.total_steps = self.total_steps
            self.add_callback(MixingRatioCallback(self.train_dataset))

    def _get_train_sampler(self, train_dataset=None):
        # Accelerate shards this block-ordered sampler across processes. A
        # custom DistributedSampler here would shard it a second time.
        dataset = self.train_dataset if train_dataset is None else train_dataset
        global_batch_size = self.args.per_device_train_batch_size * self.args.world_size
        data_seed = self.args.data_seed if self.args.data_seed is not None else self.args.seed
        return GlobalBatchShuffleSampler(dataset, global_batch_size, seed=data_seed)

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        if "loss" in logs:
            logs["mix_qa_to_tts_ratio"] = self.get_current_ratio()
        super().log(logs, start_time)

    def save_model(self, output_dir=None, _internal_call=False):
        """Save model state plus tokenizer for both full and sharded FSDP."""
        output_dir = output_dir or self.args.output_dir
        super().save_model(output_dir, _internal_call=_internal_call)

        processor = getattr(self, "processing_class", None)
        if processor is None:
            processor = getattr(self, "tokenizer", None)
        if self.args.should_save and processor is not None:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            processor.save_pretrained(output_dir)

    def get_current_ratio(self) -> float:
        return interpolate_ratio(
            self.initial_ratio,
            self.final_ratio,
            self.state.global_step,
            self.total_steps,
        )


class MixingRatioCallback(TrainerCallback):
    """Synchronize mutable mixture state before Trainer fetches each update."""

    def __init__(self, dataset: GradualRatioDataset):
        self.dataset = dataset

    def on_train_begin(self, args, state, control, **kwargs):
        self.dataset.set_current_step(state.global_step)
        return control

    def on_step_end(self, args, state, control, **kwargs):
        self.dataset.set_current_step(state.global_step)
        return control


def use_full_state_dict_for_final_save(trainer: Trainer) -> bool:
    """Switch an active FSDP trainer to a portable full-state final save.

    Returns:
        ``True`` when an FSDP plugin was changed, otherwise ``False``.
    """
    if not getattr(trainer, "is_fsdp_enabled", False):
        return False
    plugin = trainer.accelerator.state.fsdp_plugin
    if "FULL_STATE_DICT" in str(plugin.state_dict_type):
        return False

    # Accelerate retains an existing sharded config when only the enum changes.
    # Clearing both configs makes set_state_dict_type install matching full-state
    # policies (CPU offload and rank-0-only writing).
    plugin.state_dict_config = None
    plugin.optim_state_dict_config = None
    plugin.set_state_dict_type("FULL_STATE_DICT")
    return True


def data_collator(features: Sequence[Mapping[str, Sequence[int]]], pad_token_id: int):
    """Pad causal-LM inputs while excluding padded labels from loss."""
    if not features:
        raise ValueError("Cannot collate an empty feature list")

    input_ids = [list(feature["input_ids"]) for feature in features]
    attention_mask = [
        list(feature.get("attention_mask", [1] * len(ids)))
        for feature, ids in zip(features, input_ids)
    ]
    labels = [
        list(feature.get("labels", ids)) for feature, ids in zip(features, input_ids)
    ]
    for ids, mask, target in zip(input_ids, attention_mask, labels):
        if not (len(ids) == len(mask) == len(target)):
            raise ValueError("input_ids, attention_mask, and labels must have equal lengths")

    return {
        "input_ids": torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(ids, dtype=torch.long) for ids in input_ids],
            batch_first=True,
            padding_value=int(pad_token_id),
        ),
        "attention_mask": torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(mask, dtype=torch.long) for mask in attention_mask],
            batch_first=True,
            padding_value=0,
        ),
        "labels": torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(target, dtype=torch.long) for target in labels],
            batch_first=True,
            padding_value=-100,
        ),
    }


def _trainer_tokenizer_argument(tokenizer) -> dict[str, Any]:
    """Support both pre- and post-4.46 Trainer tokenizer argument names."""
    if "processing_class" in inspect.signature(Trainer.__init__).parameters:
        return {"processing_class": tokenizer}
    return {"tokenizer": tokenizer}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(config_path: str | Path = DEFAULT_CONFIG_FILE) -> None:
    config = load_config(config_path)

    model_name = config["model_name"]
    tokenizer_name = config.get("tokenizer_name", model_name)
    tts_dataset_name = config["TTS_dataset"]
    # Missing QA data denotes a TTS-only run, which keeps older configs such as
    # qwen3_pt.yaml usable without duplicating every example in a self-mixture.
    qa_dataset_name = config.get("text_QA_dataset")
    initial_ratio = parse_ratio(config.get("ratio", "1:1"))
    final_ratio = parse_ratio(config.get("final_ratio", "1:1"))
    codec_type = str(config.get("codec_type", "snac")).lower()
    num_codebooks = config.get("num_codebooks")
    audio_capacity = audio_token_capacity(codec_type, num_codebooks)

    output_dir = Path(config["save_folder"]).expanduser()
    resume_checkpoint = resolve_resume_checkpoint(
        config.get("resume_from_checkpoint"), output_dir
    )
    report_to = config.get("report_to", "wandb")
    if report_to == "wandb" or (isinstance(report_to, list) and "wandb" in report_to):
        os.environ.setdefault("WANDB_PROJECT", config.get("project_name", "vyvotts"))

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=bool(
            config.get("overwrite_output_dir", resume_checkpoint is None)
        ),
        num_train_epochs=float(config["epochs"]),
        per_device_train_batch_size=int(config["batch_size"]),
        gradient_accumulation_steps=int(config.get("gradient_accumulation_steps", 1)),
        logging_steps=int(config.get("logging_steps", 1)),
        bf16=bool(config.get("bf16", True)),
        fp16=bool(config.get("fp16", False)),
        run_name=config.get("run_name"),
        report_to=report_to,
        save_steps=int(config["save_steps"]),
        save_total_limit=config.get("save_total_limit"),
        remove_unused_columns=True,
        learning_rate=float(config["learning_rate"]),
        lr_scheduler_type=config.get("lr_scheduler_type", "cosine"),
        warmup_ratio=float(config.get("warmup_ratio", 0.0)),
        average_tokens_across_devices=bool(
            config.get("average_tokens_across_devices", False)
        ),
        dataloader_num_workers=0,
        dataloader_drop_last=bool(config.get("dataloader_drop_last", False)),
        seed=int(config.get("seed", 42)),
        data_seed=int(config.get("data_seed", config.get("seed", 42))),
        max_steps=int(config.get("max_steps", -1)),
    )

    configured_processes = config.get("number_processes")
    if configured_processes is not None and int(configured_processes) != training_args.world_size:
        warnings.warn(
            "number_processes in the model YAML does not match the launched world "
            f"size ({configured_processes} != {training_args.world_size}); using the "
            "launcher value.",
            stacklevel=2,
        )

    tokenizer_source = tokenizer_name
    if resume_checkpoint and (Path(resume_checkpoint) / "tokenizer_config.json").is_file():
        tokenizer_source = resume_checkpoint
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    added_tokens = extend_tokenizer(tokenizer, audio_capacity, int(config["pad_token"]))

    tts_dataset = _load_ds(tts_dataset_name)
    global_micro_batch = training_args.per_device_train_batch_size * training_args.world_size
    if qa_dataset_name is None:
        train_dataset = tts_dataset
    else:
        qa_dataset = (
            tts_dataset if qa_dataset_name == tts_dataset_name else _load_ds(qa_dataset_name)
        )
        train_dataset = GradualRatioDataset(
            qa_dataset,
            tts_dataset,
            global_micro_batch,
            initial_ratio=initial_ratio,
            final_ratio=final_ratio,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        attn_implementation=config.get("attn_implementation", "sdpa"),
        torch_dtype=torch.bfloat16 if training_args.bf16 else None,
    )
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = int(config["pad_token"])

    trainer = FSDPTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=partial(data_collator, pad_token_id=int(config["pad_token"])),
        initial_ratio=initial_ratio,
        final_ratio=final_ratio,
        **_trainer_tokenizer_argument(tokenizer),
    )

    if training_args.should_log:
        print(f"Model: {model_name} | codec: {codec_type} | added tokens: {added_tokens}")
        mixture = (
            f"QA:TTS ratio {initial_ratio:g}:1 -> {final_ratio:g}:1"
            if qa_dataset_name is not None
            else "TTS-only dataset"
        )
        print(
            f"{mixture} | world size: {training_args.world_size} | "
            f"steps: {trainer.total_steps}"
        )

    trainer.train(resume_from_checkpoint=resume_checkpoint)

    final_output_dir = Path(config.get("final_output_dir", output_dir / "final"))
    use_full_state_dict_for_final_save(trainer)
    trainer.save_model(str(final_output_dir))
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(str(final_output_dir))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config_path",
        nargs="?",
        type=Path,
        help="Optional YAML config path (the historical no-argument command still works)",
    )
    parser.add_argument("--config", dest="named_config_path", type=Path)
    args = parser.parse_args()
    if args.config_path is not None and args.named_config_path is not None:
        parser.error("pass the config either positionally or with --config, not both")
    return args


if __name__ == "__main__":
    cli_args = _parse_args()
    main(cli_args.named_config_path or cli_args.config_path or DEFAULT_CONFIG_FILE)
