import os
import yaml
import torch
import torch.multiprocessing as mp
import torchaudio.transforms as T
from datasets import load_dataset, concatenate_datasets
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from vyvotts.codec import load_codec


def load_config(config_path):
    """
    Load tokenizer configuration from YAML file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Dictionary with configuration values
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def tokenise_audio(waveform, codec, ds_sample_rate, target_sample_rate, audio_tokens_start):
    """
    Tokenize audio waveform using the audio codec.

    Args:
        waveform: Audio array from dataset
        codec: Codec instance (SNACCodec or MimiCodec)
        ds_sample_rate: Original dataset sample rate
        target_sample_rate: Target sample rate (24000)
        audio_tokens_start: Offset for audio tokens

    Returns:
        List of audio token IDs with proper offsets applied
    """
    # Convert to tensor and prepare for processing
    import numpy as np
    if not isinstance(waveform, np.ndarray):
        waveform = np.array(waveform, dtype=np.float32)
    waveform = torch.from_numpy(waveform).unsqueeze(0)
    waveform = waveform.to(dtype=torch.float32)

    # Resample to target sample rate if needed
    resample_transform = T.Resample(orig_freq=ds_sample_rate, new_freq=target_sample_rate)
    waveform = resample_transform(waveform)
    waveform = waveform.unsqueeze(0)  # [1, 1, T]

    # Encode with codec — returns interleaved codes with per-codebook offsets
    codes = codec.encode(waveform)

    # Add audio_tokens_start offset
    return [c + audio_tokens_start for c in codes]


def remove_duplicate_frames(codes_list, codes_per_group):
    """
    Remove consecutive duplicate audio frames to reduce redundancy.

    Each frame consists of codes_per_group codes.
    Frames with identical first codes are considered duplicates.

    Args:
        codes_list: List of audio codes
        codes_per_group: Number of codes per frame group

    Returns:
        Deduplicated codes list
    """
    if len(codes_list) % codes_per_group != 0:
        raise ValueError(f"Input list length must be divisible by {codes_per_group}")

    # Keep first frame
    result = codes_list[:codes_per_group]
    removed_frames = 0

    # Check each subsequent frame
    for i in range(codes_per_group, len(codes_list), codes_per_group):
        current_first_code = codes_list[i]
        previous_first_code = result[-codes_per_group]

        if current_first_code != previous_first_code:
            result.extend(codes_list[i:i + codes_per_group])
        else:
            removed_frames += 1

    return result


def _encode_shard(rank, num_gpus, dataset_shard, codec_type, codec_model_name,
                  ds_sample_rate, target_sample_rate, audio_tokens_start, return_dict):
    """Worker function: encode a dataset shard on a specific GPU.

    Args:
        rank: GPU index (0, 1, 2, ...).
        num_gpus: Total number of GPUs.
        dataset_shard: HuggingFace Dataset shard to process.
        codec_type: "snac" or "mimi".
        codec_model_name: HuggingFace model ID (None for default).
        ds_sample_rate: Original audio sample rate.
        target_sample_rate: Target sample rate (24000).
        audio_tokens_start: Token offset for audio tokens.
        return_dict: Shared dict to store results.
    """
    device = f"cuda:{rank}"
    print(f"[GPU {rank}] Loading {codec_type} codec on {device}...")
    codec = load_codec(codec_type=codec_type, model_name=codec_model_name, device=device)

    def add_codes(example):
        codes_list = None
        try:
            audio_data = example.get("audio")
            if audio_data and "array" in audio_data:
                codes_list = tokenise_audio(
                    audio_data["array"], codec,
                    ds_sample_rate, target_sample_rate, audio_tokens_start,
                )
        except Exception as e:
            print(f"[GPU {rank}] Skipping row: {e}")
        example["codes_list"] = codes_list
        return example

    print(f"[GPU {rank}] Encoding {len(dataset_shard)} examples...")
    encoded = dataset_shard.map(add_codes, remove_columns=["audio"])
    return_dict[rank] = encoded
    print(f"[GPU {rank}] Done.")


def process_dataset(
    original_dataset,
    output_dataset,
    model_type="qwen3",
    codec_type="snac",
    codec_model_name=None,
    text_field="text_scribe",
    target_sample_rate=24000,
    num_gpus=None,
):
    """
    Process dataset: tokenize audio and text, create training sequences.

    Automatically shards audio encoding across all available GPUs.

    Args:
        original_dataset: HuggingFace dataset path to process
        output_dataset: HuggingFace dataset path for output
        model_type: Model type - either "qwen3" or "lfm2" (default: "qwen3")
        codec_type: Audio codec type - "snac" or "mimi" (default: "snac")
        codec_model_name: HuggingFace model ID for codec (None for default)
        text_field: Name of text field in dataset (default: "text_scribe")
        target_sample_rate: Target audio sample rate (default: 24000)
        num_gpus: Number of GPUs to use (default: all available)
    """
    # Set tokenizer and config based on model type
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
        raise ValueError(f"Invalid model_type: {model_type}. Must be 'qwen3', 'lfm2', or 'lfm2_5'")

    # Load configuration
    print(f"Loading config from: {config_path}")
    config = load_config(config_path)

    TOKENIZER_LENGTH = config['TOKENIZER_LENGTH']
    START_OF_TEXT = config['START_OF_TEXT']
    END_OF_TEXT = config['END_OF_TEXT']
    START_OF_SPEECH = config['START_OF_SPEECH']
    END_OF_SPEECH = config['END_OF_SPEECH']
    START_OF_HUMAN = config['START_OF_HUMAN']
    END_OF_HUMAN = config['END_OF_HUMAN']
    START_OF_AI = config['START_OF_AI']
    END_OF_AI = config['END_OF_AI']
    PAD_TOKEN = config['PAD_TOKEN']
    AUDIO_TOKENS_START = config['AUDIO_TOKENS_START']

    # Download dataset
    print(f"Downloading dataset: {original_dataset}")
    snapshot_download(
        repo_id=original_dataset,
        repo_type="dataset",
        revision="main",
        max_workers=64,
    )

    # Load dataset
    print("Loading dataset...")
    ds = load_dataset(original_dataset, split="train")
    ds_sample_rate = ds[0]["audio"]["sampling_rate"]

    # Determine number of GPUs
    available_gpus = torch.cuda.device_count()
    if num_gpus is None:
        num_gpus = available_gpus
    num_gpus = min(num_gpus, available_gpus)
    assert num_gpus > 0, "No CUDA GPUs available"

    print(f"Using {num_gpus} GPU(s) for audio encoding")

    if num_gpus == 1:
        # Single-GPU path (no multiprocessing overhead)
        print(f"Loading {codec_type} codec on cuda:0...")
        codec = load_codec(codec_type=codec_type, model_name=codec_model_name, device="cuda:0")
        codes_per_group = codec.codes_per_group

        def add_codes(example):
            codes_list = None
            try:
                audio_data = example.get("audio")
                if audio_data and "array" in audio_data:
                    codes_list = tokenise_audio(
                        audio_data["array"], codec,
                        ds_sample_rate, target_sample_rate, AUDIO_TOKENS_START,
                    )
            except Exception as e:
                print(f"Skipping row due to error: {e}")
            example["codes_list"] = codes_list
            return example

        print("Tokenizing audio...")
        ds = ds.map(add_codes, remove_columns=["audio"])
    else:
        # Multi-GPU path: shard dataset across GPUs
        shards = [ds.shard(num_shards=num_gpus, index=i) for i in range(num_gpus)]

        manager = mp.Manager()
        return_dict = manager.dict()

        mp.set_start_method("spawn", force=True)
        processes = []
        for rank in range(num_gpus):
            p = mp.Process(
                target=_encode_shard,
                args=(rank, num_gpus, shards[rank], codec_type, codec_model_name,
                      ds_sample_rate, target_sample_rate, AUDIO_TOKENS_START, return_dict),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        # Merge shards back (in order)
        encoded_shards = [return_dict[rank] for rank in range(num_gpus)]
        ds = concatenate_datasets(encoded_shards)

        # Get codes_per_group from a temporary codec (CPU, lightweight)
        _codec = load_codec(codec_type=codec_type, model_name=codec_model_name, device="cpu")
        codes_per_group = _codec.codes_per_group
        del _codec

    # Load text tokenizer
    print(f"Loading tokenizer: {tokenizer_model}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
    num_proc = os.cpu_count() - 2

    # Filter out failed tokenizations
    print("Filtering invalid examples...")
    ds = ds.filter(lambda x: x["codes_list"] is not None)
    ds = ds.filter(lambda x: len(x["codes_list"]) > 0)

    # Remove duplicate frames
    def remove_duplicate_frames_wrapper(example):
        """Wrapper for remove_duplicate_frames."""
        example["codes_list"] = remove_duplicate_frames(example["codes_list"], codes_per_group)
        return example

    print("Removing duplicate frames...")
    ds = ds.map(remove_duplicate_frames_wrapper, num_proc=num_proc)

    print(f"""
NOTE: Text prompt customization
You can modify the text prompt in create_input_ids() below.
For multispeaker models, ensure your dataset has a "source" field.
- Single-speaker: uses example['{text_field}']
- Multi-speaker: uses example['source']: example['{text_field}']
""")

    def create_input_ids(example):
        """
        Create training input sequence with proper formatting.

        Format: [HUMAN] text [/HUMAN] [AI] [SPEECH] audio_codes [/SPEECH] [/AI]
        """
        # Determine whether to include the source field
        if "source" in example:
            text_prompt = f"{example['source']}: {example[text_field]}"
        else:
            text_prompt = example[text_field]

        # Tokenize text input
        text_ids = tokenizer.encode(text_prompt, add_special_tokens=True)
        text_ids.append(END_OF_TEXT)
        example["text_tokens"] = text_ids

        # Construct full sequence with special tokens
        input_ids = (
            [START_OF_HUMAN]
            + example["text_tokens"]
            + [END_OF_HUMAN]
            + [START_OF_AI]
            + [START_OF_SPEECH]
            + example["codes_list"]
            + [END_OF_SPEECH]
            + [END_OF_AI]
        )

        example["input_ids"] = input_ids
        example["labels"] = input_ids
        example["attention_mask"] = [1] * len(input_ids)

        return example

    # Create final training sequences
    print("Creating input sequences...")
    ds = ds.map(
        create_input_ids,
        num_proc=num_proc,
        remove_columns=[text_field, "codes_list"]
    )

    # Keep only training columns
    columns_to_keep = ["input_ids", "labels", "attention_mask"]
    columns_to_remove = [col for col in ds.column_names if col not in columns_to_keep]
    ds = ds.remove_columns(columns_to_remove)

    # Upload processed dataset
    print(f"Pushing dataset to: {output_dataset}")
    ds.push_to_hub(output_dataset)
    print("Done!")
