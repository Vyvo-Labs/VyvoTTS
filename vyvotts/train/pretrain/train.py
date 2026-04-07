import torch
import yaml
import wandb
from pathlib import Path
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, AutoTokenizer
from torch.distributed.fsdp.fully_sharded_data_parallel import FullStateDictConfig
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP, FullStateDictConfig, StateDictType)
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from huggingface_hub import HfApi
from accelerate import Accelerator

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CONFIG_FILE = Path(__file__).parent.parent.parent / "configs" / "train" / "lfm2_5_pretrain.yaml"

with open(CONFIG_FILE, "r") as file:
    config = yaml.safe_load(file)

model_name = config["model_name"]
tokenizer_name = config.get("tokenizer_name", model_name)
codec_type = config.get("codec_type", "snac")

dsn1 = config["text_QA_dataset"]
dsn2 = config["TTS_dataset"]

run_name = config["run_name"]
project_name = config["project_name"]
base_repo_id = config["save_folder"]

epochs = config["epochs"]
batch_size = config["batch_size"]
save_steps = config["save_steps"]
pad_token = config["pad_token"]
number_processes = config["number_processes"]
learning_rate = config["learning_rate"]

ratio_str = config["ratio"]
initial_ratio = int(ratio_str.split(":")[0])
final_ratio = 1

# Audio token count based on codec
num_codebooks = config.get("num_codebooks", 8 if codec_type == "mimi" else 3)
if codec_type == "mimi":
    number_add_tokens = num_codebooks * 2048 + 10
elif codec_type == "snac":
    number_add_tokens = 7 * 4096 + 10  # SNAC is fixed: 3 codebooks → 7 codes/group
else:
    raise ValueError(f"Unknown codec_type: {codec_type}")


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------
def _load_ds(path):
    if path.startswith("/") or path.startswith("./"):
        return load_from_disk(path)
    return load_dataset(path, split="train")


class GradualRatioDataset(Dataset):
    """Mix QA + TTS data with a ratio that decays from initial_ratio:1 → 1:1."""

    def __init__(self, dataset1, dataset2, batch_total,
                 initial_ratio=2, final_ratio=1, total_steps=None):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.batch_total = batch_total
        self.initial_ratio = initial_ratio
        self.final_ratio = final_ratio
        self.total_steps = total_steps
        self.current_step = 0

        max_ratio = max(initial_ratio, final_ratio)
        num_cycles_ds1 = len(dataset1) // (batch_total * max_ratio)
        num_cycles_ds2 = len(dataset2) // batch_total
        self.num_cycles = min(num_cycles_ds1, num_cycles_ds2)
        self.length = self.num_cycles * (initial_ratio + 1) * batch_total

    def set_current_step(self, step):
        self.current_step = step

    def get_current_ratio(self):
        if self.total_steps is None or self.total_steps == 0:
            return self.initial_ratio
        progress = min(self.current_step / self.total_steps, 1.0)
        current_ratio = self.initial_ratio - (self.initial_ratio - self.final_ratio) * progress
        return max(int(round(current_ratio)), self.final_ratio)

    def __len__(self):
        return int(self.length)

    def __getitem__(self, index):
        current_ratio = self.get_current_ratio()
        cycle_length = (current_ratio + 1) * self.batch_total
        cycle = index // cycle_length
        pos_in_cycle = index % cycle_length

        if pos_in_cycle < current_ratio * self.batch_total:
            batch_in_cycle = pos_in_cycle // self.batch_total
            sample_in_batch = pos_in_cycle % self.batch_total
            idx = (cycle * current_ratio * self.batch_total
                   + batch_in_cycle * self.batch_total + sample_in_batch)
            return self.dataset1[idx % len(self.dataset1)]
        else:
            sample_in_batch = pos_in_cycle - current_ratio * self.batch_total
            idx = cycle * self.batch_total + sample_in_batch
            return self.dataset2[idx % len(self.dataset2)]


# ---------------------------------------------------------------------------
# FSDP Trainer
# ---------------------------------------------------------------------------
class AlternatingDistributedSampler(DistributedSampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=False):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.shuffle = shuffle

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        indices = indices[self.rank:self.total_size:self.num_replicas]
        return iter(indices)


class FSDPTrainer(Trainer):
    def __init__(self, *args, initial_ratio=2, final_ratio=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.repo_id = base_repo_id
        self.initial_ratio = initial_ratio
        self.final_ratio = final_ratio
        self.text_step = 0
        self.audio_step = 0
        self.total_steps = self.calculate_total_steps()

    def calculate_total_steps(self):
        num_update_steps_per_epoch = len(self.train_dataset) // (
            self.args.per_device_train_batch_size
            * self.args.gradient_accumulation_steps
            * self.args.world_size
        )
        return int(num_update_steps_per_epoch * self.args.num_train_epochs)

    def get_current_ratio(self):
        if self.total_steps == 0:
            return self.initial_ratio
        progress = min(self.state.global_step / self.total_steps, 1.0)
        current_ratio = self.initial_ratio - (self.initial_ratio - self.final_ratio) * progress
        return max(int(round(current_ratio)), self.final_ratio)

    def get_train_dataloader(self):
        if hasattr(self.train_dataset, 'total_steps'):
            self.train_dataset.total_steps = self.total_steps

        sampler = AlternatingDistributedSampler(
            self.train_dataset,
            num_replicas=torch.distributed.get_world_size(),
            rank=torch.distributed.get_rank(),
            shuffle=False,
        )
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=0,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def training_step(self, model, inputs, num_items_in_batch=None):
        if hasattr(self.train_dataset, 'set_current_step'):
            self.train_dataset.set_current_step(self.state.global_step)
        return super().training_step(model, inputs, num_items_in_batch)

    def log(self, logs, start_time=None):
        super().log(logs, start_time)
        if self.is_world_process_zero() and "loss" in logs:
            current_ratio = self.get_current_ratio()
            wandb.log({"current_ratio": current_ratio, "global_step": self.state.global_step})

            cycle_length = current_ratio + 1
            step_in_cycle = self.state.global_step % cycle_length
            if step_in_cycle < current_ratio:
                wandb.log({"text_loss": logs["loss"], "text_step": self.text_step})
                self.text_step += 1
            else:
                wandb.log({"audio_loss": logs["loss"], "audio_step": self.audio_step})
                self.audio_step += 1

    def save_model(self, output_dir=None, _internal_call=False):
        output_dir = output_dir or self.args.output_dir
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, save_policy):
            cpu_state_dict = self.model.state_dict()
        self.model.save_pretrained(output_dir, state_dict=cpu_state_dict)


# ---------------------------------------------------------------------------
# Collator
# ---------------------------------------------------------------------------
def data_collator(features):
    input_ids = [f["input_ids"] for f in features]
    attention_mask = [f.get("attention_mask", [1] * len(ids)) for f, ids in zip(features, input_ids)]
    labels = [f.get("labels", ids) for f, ids in zip(features, input_ids)]

    input_ids = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(i, dtype=torch.long) for i in input_ids],
        batch_first=True, padding_value=pad_token)
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(m, dtype=torch.long) for m in attention_mask],
        batch_first=True, padding_value=0)
    labels = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(l, dtype=torch.long) for l in labels],
        batch_first=True, padding_value=-100)

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
wandb.init(project=project_name, name=run_name)

accelerator = Accelerator()
device = accelerator.device

tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    attn_implementation="sdpa",
    torch_dtype=torch.bfloat16,
)
model = model.to(device)

if accelerator.is_local_main_process:
    print(f"Model: {model_name} on {device}")
    print(f"Codec: {codec_type}, adding {number_add_tokens} audio tokens")

new_tokens = [f"<custom_token_{i}>" for i in range(number_add_tokens + 1)]
tokenizer.add_tokens(new_tokens)
model.resize_token_embeddings(len(tokenizer))

ds1 = _load_ds(dsn1)
ds2 = _load_ds(dsn2)

batch_total = batch_size * number_processes
num_update_steps_per_epoch = len(ds1) // (batch_size * number_processes)
total_steps = int(num_update_steps_per_epoch * epochs)

train_dataset = GradualRatioDataset(
    ds1, ds2, batch_total,
    initial_ratio=initial_ratio,
    final_ratio=final_ratio,
    total_steps=total_steps,
)

training_args = TrainingArguments(
    overwrite_output_dir=True,
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    logging_steps=1,
    bf16=True,
    output_dir=base_repo_id,
    report_to="wandb",
    save_steps=save_steps,
    remove_unused_columns=True,
    learning_rate=learning_rate,
    lr_scheduler_type="cosine",
    average_tokens_across_devices=False,
)

trainer = FSDPTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
    initial_ratio=initial_ratio,
    final_ratio=final_ratio,
)

if accelerator.is_local_main_process:
    print(f"Ratio: {initial_ratio}:1 → {final_ratio}:1 | Steps: {total_steps}")

trainer.train()
