"""
Tokenize Emilia English data using Mimi (or SNAC) codec on multi-GPU.

Supports two dataset sources:
  - ylacombe/emilia-subset  (parquet-based, 3.39M EN samples, ~245 GB)
  - amphion/Emilia-Dataset  (tar-based WebDataset, Emilia + Emilia-YODAS EN)

Usage:
    # ylacombe/emilia-subset (recommended — single repo, parquet, easy):
    python -m vyvotts.tokenize_emilia \
        --dataset ylacombe/emilia-subset \
        --output_dataset "your-user/emilia-en-mimi-tokens" \
        --model_type qwen3

    # amphion/Emilia-Dataset (tar-based, Emilia + YODAS):
    python -m vyvotts.tokenize_emilia \
        --dataset amphion/Emilia-Dataset \
        --subsets emilia yodas \
        --output_dataset "your-user/emilia-full-en-mimi-tokens"

    # Resume after interruption (skips already-processed shards):
    python -m vyvotts.tokenize_emilia \
        --dataset ylacombe/emilia-subset \
        --output_dataset "..." \
        --work_dir /scratch/emilia_tokens
"""
from __future__ import annotations

import argparse
import io
import json
import os
import tarfile
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import soundfile as sf
import torch
import torch.multiprocessing as mp
import torchaudio.transforms as T
import yaml
from huggingface_hub import snapshot_download

from vyvotts.codec import load_codec


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Parquet-based processing (ylacombe/emilia-subset)
# ---------------------------------------------------------------------------

def process_single_parquet(
    parquet_path: str,
    codec,
    audio_tokens_start: int,
    target_sample_rate: int = 24000,
) -> list[dict]:
    """Encode audio from a single parquet shard.

    Returns a list of dicts with keys: text, codes_list, duration, speaker, dnsmos.
    """
    import pyarrow.parquet as pq

    table = pq.read_table(parquet_path)
    results = []

    # Identify columns
    columns = table.column_names
    audio_col_name = "mp3" if "mp3" in columns else "audio"
    text_col_name = "text"

    audio_col = table.column(audio_col_name)
    text_col = table.column(text_col_name)
    duration_col = table.column("duration") if "duration" in columns else None
    speaker_col = table.column("speaker") if "speaker" in columns else None
    dnsmos_col = table.column("dnsmos") if "dnsmos" in columns else None

    for i in range(len(table)):
        try:
            # Extract text
            text = text_col[i].as_py()
            if not text or not text.strip():
                continue

            # Extract and decode audio
            audio_entry = audio_col[i].as_py()
            if isinstance(audio_entry, dict):
                audio_bytes = audio_entry.get("bytes")
                if audio_bytes is None:
                    # Audio stored as path reference — skip
                    continue
            elif isinstance(audio_entry, bytes):
                audio_bytes = audio_entry
            else:
                continue

            audio_array, sr = sf.read(io.BytesIO(audio_bytes))
            if len(audio_array.shape) > 1:
                audio_array = audio_array[:, 0]  # mono

            # Resample if needed
            waveform = torch.from_numpy(audio_array.astype(np.float32)).unsqueeze(0)
            if sr != target_sample_rate:
                waveform = T.Resample(sr, target_sample_rate)(waveform)
            waveform = waveform.unsqueeze(0)  # [1, 1, T]

            # Encode with codec
            codes = codec.encode(waveform)
            codes_with_offset = [c + audio_tokens_start for c in codes]

            results.append({
                "text": text.strip(),
                "codes_list": codes_with_offset,
                "duration": duration_col[i].as_py() if duration_col else len(audio_array) / sr,
                "speaker": speaker_col[i].as_py() if speaker_col else "",
                "dnsmos": dnsmos_col[i].as_py() if dnsmos_col else 0.0,
            })

        except Exception as e:
            continue

    return results


# ---------------------------------------------------------------------------
# Tar-based processing (amphion/Emilia-Dataset)
# ---------------------------------------------------------------------------

def process_single_tar(
    tar_path: str,
    codec,
    audio_tokens_start: int,
    target_sample_rate: int = 24000,
) -> list[dict]:
    """Extract audio/text from a single tar file and encode audio with codec.

    Returns a list of dicts with keys: text, codes_list, duration, speaker, dnsmos.
    """
    results = []

    with tarfile.open(tar_path, "r") as tar:
        members = {m.name: m for m in tar.getmembers()}

        json_members = {
            m.name.rsplit(".", 1)[0]: m
            for m in members.values()
            if m.name.endswith(".json")
        }

        for key, json_member in json_members.items():
            audio_member = None
            for ext in (".mp3", ".wav", ".flac"):
                candidate = key + ext
                if candidate in members:
                    audio_member = members[candidate]
                    break

            if audio_member is None:
                continue

            try:
                json_bytes = tar.extractfile(json_member).read()
                meta = json.loads(json_bytes)

                text = meta.get("text", "")
                if not text or not text.strip():
                    continue

                audio_bytes = tar.extractfile(audio_member).read()
                audio_array, sr = sf.read(io.BytesIO(audio_bytes))

                if len(audio_array.shape) > 1:
                    audio_array = audio_array[:, 0]

                waveform = torch.from_numpy(audio_array.astype(np.float32)).unsqueeze(0)
                if sr != target_sample_rate:
                    waveform = T.Resample(sr, target_sample_rate)(waveform)
                waveform = waveform.unsqueeze(0)  # [1, 1, T]

                codes = codec.encode(waveform)
                codes_with_offset = [c + audio_tokens_start for c in codes]

                results.append({
                    "text": text.strip(),
                    "codes_list": codes_with_offset,
                    "duration": meta.get("duration", len(audio_array) / sr),
                    "speaker": meta.get("speaker", ""),
                    "dnsmos": meta.get("dnsmos", 0.0),
                })

            except Exception as e:
                continue

    return results


# ---------------------------------------------------------------------------
# GPU worker (handles both parquet and tar shards)
# ---------------------------------------------------------------------------

def _gpu_worker(
    rank: int,
    shard_files: list[str],
    shard_type: str,
    codec_type: str,
    codec_model_name: str | None,
    audio_tokens_start: int,
    target_sample_rate: int,
    work_dir: str,
):
    """Worker process: process assigned shard files on a specific GPU.

    Args:
        shard_type: "parquet" or "tar".
    """
    device = f"cuda:{rank}"
    print(f"[GPU {rank}] Loading {codec_type} codec on {device}...")
    codec = load_codec(codec_type=codec_type, model_name=codec_model_name, device=device)

    out_dir = Path(work_dir) / f"gpu_{rank}"
    out_dir.mkdir(parents=True, exist_ok=True)

    process_fn = process_single_parquet if shard_type == "parquet" else process_single_tar
    total_samples = 0

    for idx, shard_path in enumerate(shard_files):
        shard_name = Path(shard_path).stem
        out_file = out_dir / f"{shard_name}.pt"

        # Resume support: skip already-processed shards
        if out_file.exists():
            data = torch.load(out_file, weights_only=False)
            total_samples += len(data)
            print(f"[GPU {rank}] Skipping {shard_name} (already done, {len(data)} samples)")
            continue

        t0 = time.time()
        results = process_fn(shard_path, codec, audio_tokens_start, target_sample_rate)
        elapsed = time.time() - t0

        torch.save(results, out_file)
        total_samples += len(results)

        print(
            f"[GPU {rank}] [{idx + 1}/{len(shard_files)}] {shard_name}: "
            f"{len(results)} samples in {elapsed:.1f}s "
            f"(total: {total_samples})"
        )

    print(f"[GPU {rank}] Done. Total: {total_samples} samples.")


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def download_emilia_subset(hf_token: str | None = None) -> tuple[list[str], str]:
    """Download ylacombe/emilia-subset parquet files.

    Returns:
        Tuple of (list of parquet file paths, shard_type).
    """
    print("Downloading ylacombe/emilia-subset...")
    dataset_dir = snapshot_download(
        repo_id="ylacombe/emilia-subset",
        repo_type="dataset",
        revision="main",
        allow_patterns=["data/*.parquet"],
        max_workers=16,
        token=hf_token,
    )

    parquet_files = sorted(str(p) for p in Path(dataset_dir).rglob("*.parquet"))
    print(f"Downloaded {len(parquet_files)} parquet shards to {dataset_dir}")
    return parquet_files, "parquet"


def download_emilia_tars(
    subsets: list[str],
    hf_token: str | None = None,
) -> tuple[list[str], str]:
    """Download English tar files from amphion/Emilia-Dataset.

    Returns:
        Tuple of (list of tar file paths, shard_type).
    """
    allow_patterns = []
    for subset in subsets:
        if subset == "emilia":
            allow_patterns.append("Emilia/EN/*.tar")
        elif subset == "yodas":
            allow_patterns.append("Emilia-YODAS/EN/*.tar")
        else:
            raise ValueError(f"Unknown subset: {subset}. Use 'emilia' or 'yodas'.")

    print(f"Downloading English data from amphion/Emilia-Dataset...")
    print(f"  Subsets: {subsets}")
    print(f"  Patterns: {allow_patterns}")

    dataset_dir = snapshot_download(
        repo_id="amphion/Emilia-Dataset",
        repo_type="dataset",
        revision="main",
        allow_patterns=allow_patterns,
        max_workers=16,
        token=hf_token,
    )

    tar_files = sorted(str(p) for p in Path(dataset_dir).rglob("EN/*.tar"))
    print(f"Downloaded {len(tar_files)} tar files to {dataset_dir}")
    return tar_files, "tar"


# ---------------------------------------------------------------------------
# Build training sequences and push to hub
# ---------------------------------------------------------------------------

def build_training_sequences(
    work_dir: str,
    model_type: str,
    codec_type: str,
    codec_model_name: str | None,
    output_dataset: str,
):
    """Collect tokenized results, build training sequences, push to hub."""
    from datasets import Dataset
    from transformers import AutoTokenizer
    from vyvotts.audio_tokenizer import remove_duplicate_frames

    if model_type == "qwen3":
        tokenizer_model = "Qwen/Qwen3-0.6B"
        config_path = "vyvotts/configs/inference/qwen3.yaml"
    elif model_type == "lfm2":
        tokenizer_model = "LiquidAI/LFM2-350M"
        config_path = "vyvotts/configs/inference/lfm2.yaml"
    elif model_type == "lfm2_5":
        tokenizer_model = "LiquidAI/LFM2-350M"  # same tokenizer as LFM2
        config_path = "vyvotts/configs/inference/lfm2_5.yaml"
    else:
        raise ValueError(f"Invalid model_type: {model_type}")

    config = load_config(config_path)
    END_OF_TEXT = config["END_OF_TEXT"]
    START_OF_SPEECH = config["START_OF_SPEECH"]
    END_OF_SPEECH = config["END_OF_SPEECH"]
    START_OF_HUMAN = config["START_OF_HUMAN"]
    END_OF_HUMAN = config["END_OF_HUMAN"]
    START_OF_AI = config["START_OF_AI"]
    END_OF_AI = config["END_OF_AI"]

    _codec = load_codec(codec_type=codec_type, model_name=codec_model_name, device="cpu")
    codes_per_group = _codec.codes_per_group
    del _codec

    print(f"Loading tokenizer: {tokenizer_model}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)

    pt_files = sorted(Path(work_dir).rglob("*.pt"))
    print(f"Found {len(pt_files)} intermediate files")

    all_input_ids = []
    all_labels = []
    all_attention_masks = []
    total_loaded = 0

    for pt_file in pt_files:
        results = torch.load(pt_file, weights_only=False)
        total_loaded += len(results)

        for sample in results:
            text = sample["text"]
            codes_list = sample["codes_list"]

            if not codes_list:
                continue

            try:
                codes_list = remove_duplicate_frames(codes_list, codes_per_group)
            except ValueError:
                continue

            text_prompt = text
            if sample.get("speaker"):
                text_prompt = f"{sample['speaker']}: {text}"

            text_ids = tokenizer.encode(text_prompt, add_special_tokens=True)
            text_ids.append(END_OF_TEXT)

            input_ids = (
                [START_OF_HUMAN]
                + text_ids
                + [END_OF_HUMAN]
                + [START_OF_AI]
                + [START_OF_SPEECH]
                + codes_list
                + [END_OF_SPEECH]
                + [END_OF_AI]
            )

            all_input_ids.append(input_ids)
            all_labels.append(input_ids)
            all_attention_masks.append([1] * len(input_ids))

        if total_loaded % 100_000 == 0 and total_loaded > 0:
            print(f"  Loaded {total_loaded} samples...")

    print(f"Total valid sequences: {len(all_input_ids)}")

    ds = Dataset.from_dict({
        "input_ids": all_input_ids,
        "labels": all_labels,
        "attention_mask": all_attention_masks,
    })

    # Save locally or push to hub
    if output_dataset.startswith("/") or output_dataset.startswith("./"):
        print(f"Saving dataset to: {output_dataset}")
        Path(output_dataset).mkdir(parents=True, exist_ok=True)
        ds.save_to_disk(output_dataset)
    else:
        print(f"Pushing dataset to hub: {output_dataset}")
        ds.push_to_hub(output_dataset, max_shard_size="2GB")
    print("Done!")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Tokenize Emilia English data with Mimi/SNAC on multi-GPU"
    )
    parser.add_argument(
        "--dataset", type=str, default="ylacombe/emilia-subset",
        help="Dataset source: 'ylacombe/emilia-subset' (parquet) or 'amphion/Emilia-Dataset' (tar)"
    )
    parser.add_argument("--output_dataset", type=str, required=True,
                        help="HuggingFace dataset repo to push results")
    parser.add_argument("--subsets", nargs="+", default=["emilia", "yodas"],
                        choices=["emilia", "yodas"],
                        help="(amphion only) Which subsets to process")
    parser.add_argument("--model_type", type=str, default="qwen3",
                        choices=["qwen3", "lfm2", "lfm2_5"])
    parser.add_argument("--codec_type", type=str, default="mimi",
                        choices=["snac", "mimi"])
    parser.add_argument("--codec_model_name", type=str, default=None)
    parser.add_argument("--num_gpus", type=int, default=None,
                        help="Number of GPUs (default: all available)")
    parser.add_argument("--work_dir", type=str, default="/scratch/kadirnar/emilia_tokens",
                        help="Working directory for intermediate results")
    args = parser.parse_args()

    hf_token = os.environ.get("HF_TOKEN")

    # Step 1: Download
    if args.dataset == "ylacombe/emilia-subset":
        shard_files, shard_type = download_emilia_subset(hf_token=hf_token)
    elif args.dataset == "amphion/Emilia-Dataset":
        shard_files, shard_type = download_emilia_tars(args.subsets, hf_token=hf_token)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    if not shard_files:
        print("No shard files found. Check dataset access and HF_TOKEN.")
        return

    # Step 2: Determine GPUs
    num_gpus = args.num_gpus or torch.cuda.device_count()
    num_gpus = min(num_gpus, torch.cuda.device_count())
    assert num_gpus > 0, "No CUDA GPUs available"
    print(f"\nUsing {num_gpus} GPU(s) for encoding ({shard_type} shards)")

    # Load config
    config_map = {
        "qwen3": "vyvotts/configs/inference/qwen3.yaml",
        "lfm2": "vyvotts/configs/inference/lfm2.yaml",
        "lfm2_5": "vyvotts/configs/inference/lfm2_5.yaml",
    }
    config_path = config_map[args.model_type]
    config = load_config(config_path)
    audio_tokens_start = config["AUDIO_TOKENS_START"]

    # Step 3: Shard files across GPUs (round-robin)
    tar_shards = [[] for _ in range(num_gpus)]
    for i, f in enumerate(shard_files):
        tar_shards[i % num_gpus].append(f)

    for rank in range(num_gpus):
        print(f"  GPU {rank}: {len(tar_shards[rank])} shards")

    # Step 4: Multi-GPU encoding
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    mp.set_start_method("spawn", force=True)
    processes = []
    for rank in range(num_gpus):
        p = mp.Process(
            target=_gpu_worker,
            args=(
                rank, tar_shards[rank], shard_type,
                args.codec_type, args.codec_model_name,
                audio_tokens_start, 24000, str(work_dir),
            ),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
        if p.exitcode != 0:
            print(f"WARNING: Process exited with code {p.exitcode}")

    # Step 5: Build training sequences and push to hub
    print("\n--- Building training sequences ---")
    build_training_sequences(
        str(work_dir), args.model_type,
        args.codec_type, args.codec_model_name,
        args.output_dataset,
    )


if __name__ == "__main__":
    main()
