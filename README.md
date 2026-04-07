<div align="center">
<h2>
    VyvoTTS: LLM-Based Text-to-Speech Training Framework
</h2>
<div>
    <div align="center">
    <img width="400" alt="VyvoTTS Logo" src="assets/logo.png" style="max-width: 100%; height: auto;">
</div>
</div>
<div>
    <a href="https://github.com/Vyvo-Labs/VyvoTTS" target="_blank">
        <img src="https://img.shields.io/github/stars/Vyvo-Labs/VyvoTTS?style=for-the-badge&color=FF6B6B&labelColor=2D3748" alt="GitHub stars">
    </a>
    <a href="https://github.com/Vyvo-Labs/VyvoTTS/blob/main/LICENSE" target="_blank">
        <img src="https://img.shields.io/badge/License-MIT-4ECDC4?style=for-the-badge&labelColor=2D3748" alt="MIT License">
    </a>
    <a href="https://python.org" target="_blank">
        <img src="https://img.shields.io/badge/Python-3.10+-45B7D1?style=for-the-badge&logo=python&logoColor=white&labelColor=2D3748" alt="Python 3.10+">
    </a>
    <a href="https://huggingface.co/spaces/Vyvo/VyvoTTS-LFM2" target="_blank">
        <img src="https://img.shields.io/badge/🤗_Hugging_Face-Spaces-FFD93D?style=for-the-badge&labelColor=2D3748" alt="HuggingFace Spaces">
    </a>
</div>
</div>

## Overview

VyvoTTS converts text to speech by having an LLM generate interleaved audio codec tokens, which are then decoded to audio waveforms. It supports **SNAC** and **Mimi** audio codecs.

## Installation

```bash
git clone https://github.com/Vyvo-Labs/VyvoTTS.git
cd VyvoTTS
uv venv --python 3.12 && source .venv/bin/activate

# Base install (includes both SNAC and Mimi support)
uv pip install -e "."

# With inference backends
uv pip install -e ".[vllm]"    # vLLM
uv pip install -e ".[sglang]"  # SGLang

# With training dependencies
uv pip install -e ".[train]"
```

## Inference

```python
from vyvotts.inference.transformers_inference import VyvoTTSTransformersInference

engine = VyvoTTSTransformersInference(
    config_path="vyvotts/configs/inference/lfm2_5.yaml",
    model_name="LiquidAI/LFM2.5-350M",
    tokenizer_name="LiquidAI/LFM2-350M",
    codec_type="mimi",       # or "snac"
)

audio, timing = engine.generate("Hello world", output_path="output.wav")
```

All four backends share the same interface — swap by changing the import:

```python
from vyvotts.inference.vllm_inference import VyvoTTSInference              # vLLM (fastest TTFT)
from vyvotts.inference.sglang_inference import VyvoTTSSGLangInference      # SGLang (highest tok/s)
from vyvotts.inference.transformers_inference import VyvoTTSTransformersInference  # HuggingFace
from vyvotts.inference.unsloth_inference import VyvoTTSUnslothInference    # 4/8-bit quantized
```

### Benchmark (Qwen3-1.7B, H100 PCIe)

| Engine | TTFT | TTFA | Tokens/s |
|--------|------|------|----------|
| SGLang | 10ms | 32ms | **308** |
| vLLM | **6ms** | **30ms** | 292 |
| Unsloth | 25ms | 55ms | 54 |
| Transformers | 22ms | 50ms | 50 |

## Dataset Preparation

### Standard dataset tokenization

```python
from vyvotts.audio_tokenizer import process_dataset

process_dataset(
    original_dataset="MrDragonFox/Elise",
    output_dataset="username/dataset-name",
    model_type="lfm2_5",     # or "qwen3", "lfm2"
    codec_type="mimi",       # or "snac"
    num_gpus=8,              # multi-GPU support
)
```

### Large-scale Emilia dataset tokenization

```bash
python -m vyvotts.tokenize_emilia \
    --dataset ylacombe/emilia-subset \
    --output_dataset /scratch/output \
    --model_type lfm2_5 \
    --codec_type mimi \
    --num_gpus 8
```

Supports two Emilia sources:
- `ylacombe/emilia-subset` — 3.39M EN samples, parquet-based
- `amphion/Emilia-Dataset` — Emilia + Emilia-YODAS EN, tar-based

## Training

### Pre-training (multi-GPU FSDP)

```bash
python -m accelerate.commands.launch \
    --config_file vyvotts/configs/train/accelerate_pretrain.yaml \
    vyvotts/train/pretrain/train.py
```

Configure in `vyvotts/configs/train/lfm2_5_pretrain.yaml`:
- Model, tokenizer, codec type
- Dataset paths (local or HuggingFace)
- QA:TTS ratio scheduling (2:1 → 1:1)

### Fine-tuning

Single-speaker fine-tuning with automatic tokenization:

```bash
# Single speaker
python -m vyvotts.finetune \
    --dataset Vyvo/ElevenLabs-EN \
    --speaker ElevenLabs \
    --output_dir output/ElevenLabs

# Multiple speakers at once
python -m vyvotts.finetune \
    --dataset Vyvo/ElevenLabs-EN Vyvo/ElevenLabs-EN-Elise2-Lpq0RJl4hRqNiDLfiBMr \
    --speaker ElevenLabs Elise2 \
    --output_dir output/ElevenLabs output/Elise2 \
    --epochs 3 --batch_size 4 --lr 2e-5
```

The pipeline handles everything: download → tokenize with codec → train → generate test wav files.

See [FINETUNE.md](FINETUNE.md) for the full guide.

### Full training (accelerate)

```bash
# Full fine-tuning
python -m accelerate.commands.launch \
    --config_file vyvotts/configs/train/accelerate_finetune.yaml \
    vyvotts/train/finetune/train.py

# LoRA fine-tuning
python -m accelerate.commands.launch \
    --config_file vyvotts/configs/train/accelerate_finetune.yaml \
    vyvotts/train/finetune/lora.py
```

## Supported Models

| Model | Type | Config |
|-------|------|--------|
| LiquidAI/LFM2.5-350M | Hybrid conv+attention | `lfm2_5.yaml` |
| LiquidAI/LFM2-350M | Hybrid conv+attention | `lfm2.yaml` |
| Qwen/Qwen3-0.6B | Transformer | `qwen3.yaml` |
| Llama3 | Transformer | `llama3.yaml` |

## Acknowledgements

- [Orpheus TTS](https://github.com/canopyai/orpheus-tts)
- [LiquidAI](https://huggingface.co/LiquidAI)
- [Kyutai Mimi](https://huggingface.co/kyutai/mimi)
- [SNAC](https://github.com/hubertsiuzdak/snac)

## License

MIT
