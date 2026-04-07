import torch
import yaml
import wandb
from pathlib import Path
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, AutoTokenizer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CONFIG_FILE = Path(__file__).parent.parent.parent / "configs" / "train" / "lfm2_5_ft.yaml"

with open(CONFIG_FILE, "r") as file:
    config = yaml.safe_load(file)

model_name = config["model_name"]
tokenizer_name = config.get("tokenizer_name", model_name)
codec_type = config.get("codec_type", "snac")

dsn = config["TTS_dataset"]

run_name = config["run_name"]
project_name = config["project_name"]
base_repo_id = config["save_folder"]
epochs = config["epochs"]
batch_size = config["batch_size"]
save_steps = config["save_steps"]
pad_token = config["pad_token"]
learning_rate = config["learning_rate"]

# Audio token count based on codec
num_codebooks = config.get("num_codebooks", 8 if codec_type == "mimi" else 3)
if codec_type == "mimi":
    number_add_tokens = num_codebooks * 2048 + 10
elif codec_type == "snac":
    number_add_tokens = 7 * 4096 + 10
else:
    raise ValueError(f"Unknown codec_type: {codec_type}")


# ---------------------------------------------------------------------------
# Dataset helper
# ---------------------------------------------------------------------------
def _load_ds(path):
    if path.startswith("/") or path.startswith("./"):
        return load_from_disk(path)
    return load_dataset(path, split="train")


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
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    attn_implementation="sdpa",
    torch_dtype=torch.bfloat16,
)

new_tokens = [f"<custom_token_{i}>" for i in range(number_add_tokens + 1)]
tokenizer.add_tokens(new_tokens)
model.resize_token_embeddings(len(tokenizer))

print(f"Model: {model_name} | Codec: {codec_type} | Audio tokens: {number_add_tokens}")

ds = _load_ds(dsn)

wandb.init(project=project_name, name=run_name)

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
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds,
    data_collator=data_collator,
)

trainer.train()
