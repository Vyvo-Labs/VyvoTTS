"""Standalone finetune script — launched by vyvotts.finetune via accelerate."""
import json
import sys
import torch
from pathlib import Path
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from accelerate import Accelerator


def data_collator_fn(pad_token):
    def collator(features):
        ids = [f["input_ids"] for f in features]
        masks = [f.get("attention_mask", [1] * len(x)) for f, x in zip(features, ids)]
        labels = [f.get("labels", x) for f, x in zip(features, ids)]

        ids = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(x, dtype=torch.long) for x in ids], batch_first=True, padding_value=pad_token)
        masks = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(x, dtype=torch.long) for x in masks], batch_first=True, padding_value=0)
        labels = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(x, dtype=torch.long) for x in labels], batch_first=True, padding_value=-100)

        return {"input_ids": ids, "attention_mask": masks, "labels": labels}
    return collator


def main():
    # Read training config from JSON file passed as argument
    config_path = sys.argv[1]
    with open(config_path) as f:
        cfg = json.load(f)

    accelerator = Accelerator()

    ds = load_from_disk(cfg["tokenized_path"])
    tokenizer = AutoTokenizer.from_pretrained(cfg["tokenizer_name"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["checkpoint"], attn_implementation="sdpa", torch_dtype=torch.bfloat16,
    )
    model = model.to(accelerator.device)

    if accelerator.is_local_main_process:
        print(f"  Dataset: {len(ds)} samples, {accelerator.num_processes} GPUs")

    args = TrainingArguments(
        output_dir=cfg["output_dir"],
        overwrite_output_dir=True,
        num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=cfg["batch_size"],
        learning_rate=cfg["lr"],
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=10,
        save_steps=cfg["save_steps"],
        save_total_limit=2,
        bf16=True,
        report_to="none",
        remove_unused_columns=True,
        fsdp="auto_wrap",
    )

    trainer = Trainer(
        model=model, args=args, train_dataset=ds,
        data_collator=data_collator_fn(cfg["pad_token"]),
    )

    if accelerator.is_local_main_process:
        print(f"  Training {cfg['epochs']} epochs (bs={cfg['batch_size']}, lr={cfg['lr']})")

    trainer.train()

    if accelerator.is_local_main_process:
        final = f"{cfg['output_dir']}/final"
        trainer.save_model(final)
        tokenizer.save_pretrained(final)
        print(f"  Saved to {final}")


if __name__ == "__main__":
    main()
