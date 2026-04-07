import torch
from typing import Optional, Dict, Any
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from vyvotts.inference.base import BaseVyvoTTSInference


class VyvoTTSInference(BaseVyvoTTSInference):
    """High-performance TTS inference engine using vLLM backend."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        config_path: Optional[str] = None,
        model_name: str = "Vyvo/VyvoTTS-LFM2-Neuvillette",
        tokenizer_name: Optional[str] = None,
        codec_type: str = "snac",
        codec_model_name: str = None,
        enforce_eager: bool = False,
        max_model_len: int = 2048,
        gpu_memory_utilization: float = 0.95,
        **llm_kwargs,
    ):
        super().__init__(config, config_path)

        self.engine = LLM(
            model=model_name,
            enforce_eager=enforce_eager,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            **llm_kwargs,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name or model_name)
        self.codec = self._load_codec(codec_type, codec_model_name)

    def generate(
        self,
        text: str,
        voice: Optional[str] = None,
        output_path: Optional[str] = None,
        temperature: float = 0.6,
        top_p: float = 0.8,
        max_tokens: int = 1200,
        repetition_penalty: float = 1.3,
    ) -> Optional[torch.Tensor]:
        """Generate speech from text input."""
        prompt_ids = self._build_prompt_tokens(text, voice)

        params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop_token_ids=[self.END_OF_SPEECH],
            repetition_penalty=repetition_penalty,
        )

        outputs = self.engine.generate(
            [{"prompt_token_ids": prompt_ids[0].tolist()}],
            [params],
        )
        token_ids = outputs[0].outputs[0].token_ids
        generated_ids = torch.tensor([token_ids], dtype=torch.long)

        audio_samples = self._extract_audio_from_tokens(generated_ids)
        audio = audio_samples[0] if audio_samples else None

        if output_path and audio is not None:
            self.save_audio(audio, output_path)

        return audio


if __name__ == "__main__":
    engine = VyvoTTSInference(model_name="Vyvo/VyvoTTS-EN-Beta")
    audio = engine.generate("Hello world", output_path="output.wav")
    if audio is not None:
        print(f"Audio: {audio.shape[-1]/24000:.2f}s")
