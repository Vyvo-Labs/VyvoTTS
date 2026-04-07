"""
Single-speaker finetune pipeline: tokenize → train → inference.

Usage:
    python -m vyvotts.finetune \
        --dataset Vyvo/ElevenLabs-EN \
        --speaker ElevenLabs \
        --output_dir output/ElevenLabs

    python -m vyvotts.finetune \
        --dataset Vyvo/ElevenLabs-EN Vyvo/ElevenLabs-EN-Elise2-Lpq0RJl4hRqNiDLfiBMr \
        --speaker ElevenLabs Elise2 \
        --output_dir output/ElevenLabs output/Elise2 \
        --num_gpus 8
"""
from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.transforms as T
import yaml
from datasets import Dataset, load_from_disk
from huggingface_hub import snapshot_download

from vyvotts.codec import load_codec
from vyvotts.audio_tokenizer import remove_duplicate_frames


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_CHECKPOINT = "/scratch/kadirnar/checkpoints/lfm2_5-pretrain/checkpoint-39699"
DEFAULT_TOKENIZER = "LiquidAI/LFM2-350M"
DEFAULT_CONFIG = "vyvotts/configs/inference/lfm2_5.yaml"
DEFAULT_CODEC = "mimi"
DEFAULT_EPOCHS = 3
DEFAULT_BATCH_SIZE = 4
DEFAULT_LR = 2e-5
DEFAULT_SAVE_STEPS = 500

# Path to the standalone training script
_TRAIN_SCRIPT = Path(__file__).parent / "train" / "finetune" / "run.py"


def _load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Step 1 — Tokenize
# ---------------------------------------------------------------------------
def _tokenize_worker(rank, parquet_files, speaker, config, codec_type, tokenizer_name, return_dict):
    """Worker: tokenize a shard of parquet files on a specific GPU."""
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    device = f"cuda:{rank}"
    num_codebooks = config.get("NUM_CODEBOOKS")
    codec_kwargs = {"num_codebooks": num_codebooks} if num_codebooks else {}
    print(f"  [GPU {rank}] Loading {codec_type} codec on {device}...", flush=True)
    codec = load_codec(codec_type=codec_type, device=device, **codec_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    cpg = codec.codes_per_group

    ATS = config["AUDIO_TOKENS_START"]
    EOT = config["END_OF_TEXT"]

    all_ids, all_labels, all_masks = [], [], []
    total, skipped = 0, 0

    for pf in parquet_files:
        table = pq.read_table(pf)
        audio_col, text_col = table.column("audio"), table.column("text")
        has_speaker_col = "speaker" in table.column_names
        speaker_col = table.column("speaker") if has_speaker_col else None

        for i in range(len(table)):
            try:
                text = text_col[i].as_py()
                if not text or not text.strip():
                    skipped += 1
                    continue

                entry = audio_col[i].as_py()
                audio_bytes = entry.get("bytes") if isinstance(entry, dict) else None
                if not audio_bytes:
                    skipped += 1
                    continue

                arr, sr = sf.read(io.BytesIO(audio_bytes))
                if arr.ndim > 1:
                    arr = arr[:, 0]

                wav = torch.from_numpy(arr.astype(np.float32)).unsqueeze(0)
                if sr != 24000:
                    wav = T.Resample(sr, 24000)(wav)
                wav = wav.unsqueeze(0).to(device)

                codes = codec.encode(wav)
                codes = [c + ATS for c in codes]
                codes = remove_duplicate_frames(codes, cpg)

                row_speaker = speaker_col[i].as_py() if has_speaker_col else speaker
                text_ids = tokenizer.encode(f"{row_speaker}: {text.strip()}", add_special_tokens=True)
                text_ids.append(EOT)

                seq = (
                    [config["START_OF_HUMAN"]] + text_ids + [config["END_OF_HUMAN"]]
                    + [config["START_OF_AI"], config["START_OF_SPEECH"]]
                    + codes
                    + [config["END_OF_SPEECH"], config["END_OF_AI"]]
                )

                all_ids.append(seq)
                all_labels.append(seq)
                all_masks.append([1] * len(seq))
                total += 1

                if total % 500 == 0:
                    print(f"  [GPU {rank}] {total} samples...", flush=True)

            except Exception as e:
                skipped += 1

    print(f"  [GPU {rank}] Done: {total} samples (skipped {skipped})", flush=True)
    return_dict[rank] = (all_ids, all_labels, all_masks)

    del codec
    torch.cuda.empty_cache()


def tokenize(
    dataset_repo: str,
    speaker: str,
    tokenized_path: str,
    config: dict,
    codec_type: str = DEFAULT_CODEC,
    tokenizer_name: str = DEFAULT_TOKENIZER,
    num_gpus: int = None,
) -> str:
    """Encode audio with codec and build training sequences using multiple GPUs."""
    import torch.multiprocessing as mp

    if Path(tokenized_path).exists():
        print(f"  [skip] Already tokenized at {tokenized_path}")
        return tokenized_path

    num_gpus = num_gpus or torch.cuda.device_count()
    num_gpus = min(num_gpus, torch.cuda.device_count())
    print(f"  Using {num_gpus} GPU(s) for tokenization")

    print(f"  Downloading {dataset_repo}...")
    dataset_dir = snapshot_download(repo_id=dataset_repo, repo_type="dataset", revision="main")
    parquet_files = sorted(str(p) for p in Path(dataset_dir).rglob("*.parquet"))
    print(f"  Found {len(parquet_files)} parquet files")

    if num_gpus == 1:
        # Single GPU — run in-process
        manager = mp.Manager()
        return_dict = manager.dict()
        _tokenize_worker(0, parquet_files, speaker, config, codec_type, tokenizer_name, return_dict)
        all_ids, all_labels, all_masks = return_dict[0]
    else:
        # Multi-GPU — shard parquet files across GPUs
        shards = [[] for _ in range(num_gpus)]
        for i, pf in enumerate(parquet_files):
            shards[i % num_gpus].append(pf)

        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass

        manager = mp.Manager()
        return_dict = manager.dict()
        processes = []

        for rank in range(num_gpus):
            p = mp.Process(
                target=_tokenize_worker,
                args=(rank, shards[rank], speaker, config, codec_type, tokenizer_name, return_dict),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        # Merge results from all GPUs
        all_ids, all_labels, all_masks = [], [], []
        for rank in range(num_gpus):
            ids, labels, masks = return_dict[rank]
            all_ids.extend(ids)
            all_labels.extend(labels)
            all_masks.extend(masks)

    print(f"  Total tokenized: {len(all_ids)} samples")

    ds = Dataset.from_dict({"input_ids": all_ids, "labels": all_labels, "attention_mask": all_masks})
    ds.save_to_disk(tokenized_path)

    return tokenized_path


# ---------------------------------------------------------------------------
# Step 2 — Finetune
# ---------------------------------------------------------------------------
def _write_accelerate_config(num_gpus: int, output_path: str):
    """Write accelerate FSDP config."""
    cfg = {
        "compute_environment": "LOCAL_MACHINE",
        "debug": False,
        "distributed_type": "FSDP",
        "downcast_bf16": False,
        "enable_cpu_affinity": False,
        "machine_rank": 0,
        "main_training_function": "main",
        "mixed_precision": "bf16",
        "num_machines": 1,
        "num_processes": num_gpus,
        "rdzv_backend": "static",
        "same_network": False,
        "tpu_use_cluster": False,
        "tpu_use_sudo": False,
        "use_cpu": False,
        "fsdp_config": {
            "fsdp_auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
            "fsdp_backward_prefetch": "BACKWARD_PRE",
            "fsdp_forward_prefetch": True,
            "fsdp_offload_params": False,
            "fsdp_sharding_strategy": "FULL_SHARD",
            "fsdp_state_dict_type": "SHARDED_STATE_DICT",
            "fsdp_transformer_layer_cls_to_wrap": "Lfm2DecoderLayer",
            "fsdp_sync_module_states": True,
            "fsdp_use_orig_params": True,
            "fsdp_cpu_ram_efficient_loading": True,
        },
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(cfg, f, indent=2)


def finetune(
    tokenized_path: str,
    output_dir: str,
    checkpoint: str = DEFAULT_CHECKPOINT,
    tokenizer_name: str = DEFAULT_TOKENIZER,
    config: dict = None,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr: float = DEFAULT_LR,
    save_steps: int = DEFAULT_SAVE_STEPS,
    num_gpus: int = None,
) -> str:
    """Finetune model. Auto-selects single or multi-GPU."""
    num_gpus = num_gpus or torch.cuda.device_count()
    num_gpus = min(num_gpus, torch.cuda.device_count())

    # Write training params to JSON for the training script
    train_config = f"{output_dir}/.train_config.json"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(train_config, "w") as f:
        json.dump({
            "tokenized_path": tokenized_path,
            "checkpoint": checkpoint,
            "tokenizer_name": tokenizer_name,
            "output_dir": output_dir,
            "pad_token": config["PAD_TOKEN"],
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "save_steps": save_steps,
        }, f)

    if num_gpus > 1:
        # Multi-GPU: accelerate launch with FSDP
        accel_config = f"{output_dir}/.accelerate_config.json"
        _write_accelerate_config(num_gpus, accel_config)

        print(f"  {num_gpus} GPUs (FSDP)")
        cmd = [
            sys.executable, "-m", "accelerate.commands.launch",
            "--config_file", accel_config,
            str(_TRAIN_SCRIPT), train_config,
        ]
        result = subprocess.run(cmd, cwd=str(Path.cwd()))
        Path(accel_config).unlink(missing_ok=True)
    else:
        # Single GPU: direct launch
        print(f"  1 GPU")
        cmd = [sys.executable, str(_TRAIN_SCRIPT), train_config]
        result = subprocess.run(cmd, cwd=str(Path.cwd()))

    # Cleanup
    Path(train_config).unlink(missing_ok=True)

    if result.returncode != 0:
        raise RuntimeError(f"Training failed with exit code {result.returncode}")

    return f"{output_dir}/final"


# ---------------------------------------------------------------------------
# Step 3 — Test inference
# ---------------------------------------------------------------------------
def test_inference(
    model_path: str,
    speaker: str,
    output_dir: str,
    config_path: str = DEFAULT_CONFIG,
    tokenizer_name: str = DEFAULT_TOKENIZER,
    codec_type: str = DEFAULT_CODEC,
):
    """Generate test wav files from finetuned model."""
    from vyvotts.inference.transformers_inference import VyvoTTSTransformersInference

    engine = VyvoTTSTransformersInference(
        config_path=config_path,
        model_name=model_path,
        tokenizer_name=tokenizer_name,
        codec_type=codec_type,
        device="cuda:0",
        attn_implementation="sdpa",
    )

    texts = [
        "Hello, this is a test of the finetuned voice model.",
        "The quick brown fox jumps over the lazy dog.",
        "I can generate speech in any voice I was trained on.",
    ]

    for i, text in enumerate(texts):
        path = f"{output_dir}/test_{i}.wav"
        audio, _ = engine.generate(
            text, voice=speaker,
            max_new_tokens=1200, temperature=0.6,
            top_p=0.95, repetition_penalty=1.1,
            output_path=path,
        )
        dur = f"{audio.shape[-1]/24000:.2f}s" if audio is not None else "FAIL"
        print(f"    [{i}] {dur} — {path}")

    del engine
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Single-speaker finetune: tokenize → train → test")
    parser.add_argument("--dataset", nargs="+", required=True)
    parser.add_argument("--speaker", nargs="+", required=True)
    parser.add_argument("--output_dir", nargs="+", default=None)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--codec", default=DEFAULT_CODEC, choices=["mimi", "snac"])
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--save_steps", type=int, default=DEFAULT_SAVE_STEPS)
    parser.add_argument("--num_gpus", type=int, default=None)
    parser.add_argument("--work_dir", default="/scratch/kadirnar")
    parser.add_argument("--skip_tokenize", action="store_true")
    parser.add_argument("--skip_train", action="store_true")
    args = parser.parse_args()

    assert len(args.dataset) == len(args.speaker), "--dataset and --speaker must have same count"
    if args.output_dir is None:
        args.output_dir = [f"output/{s}" for s in args.speaker]
    assert len(args.output_dir) == len(args.dataset), "--output_dir must match --dataset count"

    config = _load_yaml(args.config)

    for dataset_repo, speaker, output_dir in zip(args.dataset, args.speaker, args.output_dir):
        name = dataset_repo.split("/")[-1]
        tokenized_path = f"{args.work_dir}/{name}-{args.codec}-tokenized"

        print(f"\n{'='*60}")
        print(f"  {speaker} — {dataset_repo}")
        print(f"{'='*60}")

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        if not args.skip_tokenize:
            print(f"\n[1/3] Tokenize")
            tokenize(dataset_repo, speaker, tokenized_path, config, args.codec, args.tokenizer, args.num_gpus)

        if not args.skip_train:
            print(f"\n[2/3] Finetune")
            model_path = finetune(
                tokenized_path, output_dir, args.checkpoint, args.tokenizer,
                config, args.epochs, args.batch_size, args.lr, args.save_steps,
                args.num_gpus,
            )
        else:
            model_path = f"{output_dir}/final"

        print(f"\n[3/3] Test inference")
        test_inference(model_path, speaker, output_dir, args.config, args.tokenizer, args.codec)

    print(f"\n{'='*60}")
    print("DONE!")
    print(f"{'='*60}")
    for d in args.output_dir:
        wavs = sorted(Path(d).glob("*.wav"))
        if wavs:
            print(f"\n{d}/")
            for w in wavs:
                print(f"  {w.name}")


if __name__ == "__main__":
    main()
