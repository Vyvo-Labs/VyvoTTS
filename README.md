<div align="center">
<h2>
    VyvoTTS: LLM-Based Text-to-Speech Training Framework 🚀
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

## Installation

> **Note:** vLLM and SGLang need different torch versions, so use separate environments.

```bash
# vLLM backend (recommended)
pip install vyvotts[vllm]

# SGLang backend
pip install vyvotts[sglang]

# Training
pip install vyvotts[train]
```

Or install from source:

```bash
git clone https://github.com/Vyvo-Labs/VyvoTTS.git
cd VyvoTTS

# vLLM
pip install -e ".[vllm]"

# SGLang
pip install -e ".[sglang]"
```

## Inference

```python
from vyvotts.inference.vllm_inference import VyvoTTSInference

engine = VyvoTTSInference(model_name="Vyvo/VyvoTTS-LFM2-Neuvillette")
audio = engine.generate("Hello world", output_path="output.wav")
```

Switch backend by changing the import:

```python
from vyvotts.inference.vllm_inference import VyvoTTSInference              # vLLM
from vyvotts.inference.sglang_inference import VyvoTTSSGLangInference      # SGLang
from vyvotts.inference.transformers_inference import VyvoTTSTransformersInference  # Transformers
from vyvotts.inference.unsloth_inference import VyvoTTSUnslothInference    # Unsloth
```

### Benchmark (Qwen3-1.7B, H100 PCIe)

| Engine | TTFT | TTFA | Tokens/s |
|--------|------|------|----------|
| SGLang | 10ms | 32ms | **308** |
| vLLM | **6ms** | **30ms** | 292 |
| Unsloth | 25ms | 55ms | 54 |
| Transformers | 22ms | 50ms | 50 |

## Training

```bash
# Fine-tuning (30GB+ VRAM)
accelerate launch --config_file vyvotts/configs/train/accelerate_finetune.yaml vyvotts/train/finetune/train.py

# Pre-training
accelerate launch --config_file vyvotts/configs/train/accelerate_pretrain.yaml vyvotts/train/pretrain/train.py
```

## Dataset Preparation

```python
from vyvotts.audio_tokenizer import process_dataset

process_dataset(
    original_dataset="MrDragonFox/Elise",
    output_dataset="username/dataset-name",
    model_type="lfm2",  # or "qwen3"
)
```

## Acknowledgements

- [Orpheus TTS](https://github.com/canopyai/orpheus-tts)
- [LiquidAI](https://huggingface.co/LiquidAI)

## License

MIT
