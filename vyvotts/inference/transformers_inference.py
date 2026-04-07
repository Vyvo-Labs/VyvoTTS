import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Tuple, Optional, Dict, Any

from vyvotts.inference.base import BaseVyvoTTSInference


class VyvoTTSTransformersInference(BaseVyvoTTSInference):
    """TTS inference engine using HuggingFace Transformers."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        config_path: Optional[str] = None,
        model_name: str = "Vyvo/VyvoTTS-LFM2-Neuvillette",
        tokenizer_name: Optional[str] = None,
        codec_type: str = "snac",
        codec_model_name: str = None,
        device: str = "cuda",
        attn_implementation: str = "sdpa",
    ):
        super().__init__(config, config_path)
        self.device = device

        self.codec = self._load_codec(codec_type, codec_model_name, device=device)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_implementation,
        ).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name or model_name)

    def generate(
        self,
        text: str,
        voice: Optional[str] = None,
        max_new_tokens: int = 1200,
        temperature: float = 0.6,
        top_p: float = 0.95,
        repetition_penalty: float = 1.1,
        output_path: Optional[str] = None,
    ) -> Tuple[Optional[torch.Tensor], Dict[str, float]]:
        """Generate speech from text input.

        Args:
            text: Input text to convert to speech.
            voice: Optional voice identifier.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Top-p sampling parameter.
            repetition_penalty: Penalty for token repetition.
            output_path: Optional path to save audio file.

        Returns:
            Tuple of (audio tensor, timing info dict).
        """
        torch.cuda.synchronize()
        total_start = time.time()

        # Preprocess
        torch.cuda.synchronize()
        t0 = time.time()
        tokens = self._build_prompt_tokens(text, voice)
        input_ids, attention_mask = self._pad_and_batch([tokens], device=self.device)
        torch.cuda.synchronize()
        preprocess_time = time.time() - t0

        # Generate
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                num_return_sequences=1,
                eos_token_id=self.END_OF_SPEECH,
            )
        torch.cuda.synchronize()
        generation_time = time.time() - t0

        # Decode audio
        torch.cuda.synchronize()
        t0 = time.time()
        audio_samples = self._extract_audio_from_tokens(generated_ids, device=self.device)
        torch.cuda.synchronize()
        audio_time = time.time() - t0

        total_time = time.time() - total_start

        timing_info = {
            'preprocessing_time': preprocess_time,
            'generation_time': generation_time,
            'audio_processing_time': audio_time,
            'total_time': total_time,
        }

        audio = audio_samples[0] if audio_samples else None
        if output_path and audio is not None:
            self.save_audio(audio, output_path)

        return audio, timing_info


def main():
    """Example usage of VyvoTTSTransformersInference."""
    engine = VyvoTTSTransformersInference()

    test_text = "Hey there my name is Elise, and I'm a speech generation model that can sound like a person."
    audio, timing_info = engine.generate(test_text)

    if audio is not None:
        print(f"Audio generated successfully with shape: {audio.shape}")
        print(f"Timing info: {timing_info}")
    else:
        print("Failed to generate audio")

    return audio


if __name__ == "__main__":
    main()
