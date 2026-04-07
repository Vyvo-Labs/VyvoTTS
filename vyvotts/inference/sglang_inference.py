import torch
import sglang as sgl
from typing import Optional, Dict, Any
from transformers import AutoTokenizer

from vyvotts.inference.base import BaseVyvoTTSInference


class VyvoTTSSGLangInference(BaseVyvoTTSInference):
    """High-performance TTS inference engine using SGLang backend."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        config_path: Optional[str] = None,
        model_name: str = "Vyvo/VyvoTTS-LFM2-Neuvillette",
        tokenizer_name: Optional[str] = None,
        codec_type: str = "snac",
        codec_model_name: str = None,
        context_length: int = 2048,
        mem_fraction_static: float = 0.90,
        **engine_kwargs,
    ):
        super().__init__(config, config_path)

        self.engine = sgl.Engine(
            model_path=model_name,
            context_length=context_length,
            mem_fraction_static=mem_fraction_static,
            attention_backend="triton",
            **engine_kwargs,
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
        """Generate speech from text input.

        Args:
            text: Input text to convert to speech.
            voice: Optional voice identifier.
            output_path: Optional path to save audio file.
            temperature: Sampling temperature.
            top_p: Top-p sampling parameter.
            max_tokens: Maximum tokens to generate.
            repetition_penalty: Penalty for token repetition.

        Returns:
            Audio tensor containing the generated speech, or None on failure.
        """
        prompt_ids = self._build_prompt_tokens(text, voice)
        input_ids_list = prompt_ids[0].tolist()

        sampling_params = {
            "temperature": temperature,
            "top_p": top_p,
            "max_new_tokens": max_tokens,
            "stop_token_ids": [self.END_OF_SPEECH],
            "repetition_penalty": repetition_penalty,
            "skip_special_tokens": False,
        }

        outputs = self.engine.generate(
            input_ids=[input_ids_list],
            sampling_params=sampling_params,
        )
        token_ids = outputs[0]["output_ids"]
        generated_ids = torch.tensor([token_ids], dtype=torch.long)

        audio_samples = self._extract_audio_from_tokens(generated_ids)
        audio = audio_samples[0] if audio_samples else None

        if output_path and audio is not None:
            self.save_audio(audio, output_path)

        return audio

    def shutdown(self):
        """Release GPU resources and kill subprocesses."""
        self.engine.shutdown()


def text_to_speech(
    prompt: str,
    voice: Optional[str] = None,
    config_path: Optional[str] = None,
) -> Optional[torch.Tensor]:
    """Generate speech from text using SGLang engine."""
    engine = VyvoTTSSGLangInference(config_path=config_path)
    audio = engine.generate(prompt, voice)
    engine.shutdown()
    return audio


if __name__ == "__main__":
    audio_output = text_to_speech("Hello world", voice="zoe")
    print("Decoded audio (tensors):", audio_output)
