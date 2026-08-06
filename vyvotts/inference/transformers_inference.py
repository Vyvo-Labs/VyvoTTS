import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList
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
        codec_type: Optional[str] = None,
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
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name or model_name)

    def _synchronize(self) -> None:
        """Synchronize CUDA timing without breaking CPU inference."""
        if str(self.device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)

    def generate(
        self,
        text: str,
        voice: Optional[str] = None,
        max_new_tokens: int = 1200,
        temperature: float = 0.6,
        top_p: float = 0.95,
        repetition_penalty: float = 1.1,
        output_path: Optional[str] = None,
        constrained_decoding: bool = True,
        min_audio_frames: int = 1,
        use_random_voice: bool = False,
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
            constrained_decoding: Enforce codec token ranges and frame boundaries.
            min_audio_frames: Minimum complete frames before speech may end.
            use_random_voice: Select a random known speaker when voice is omitted.

        Returns:
            Tuple of (audio tensor, timing info dict).
        """
        self._synchronize()
        total_start = time.time()

        # Preprocess
        self._synchronize()
        t0 = time.time()
        tokens = self._build_prompt_tokens(text, voice, use_random_voice=use_random_voice)
        input_ids, attention_mask = self._pad_and_batch([tokens], device=self.device)
        self._synchronize()
        preprocess_time = time.time() - t0

        logits_processor = LogitsProcessorList()
        if constrained_decoding:
            logits_processor.append(
                self._audio_logits_processor(input_ids.shape[1], min_audio_frames)
            )

        # Generate
        self._synchronize()
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
                pad_token_id=self.PAD_TOKEN,
                logits_processor=logits_processor,
            )
        self._synchronize()
        generation_time = time.time() - t0

        # Decode audio
        self._synchronize()
        t0 = time.time()
        audio_samples = self._extract_audio_from_tokens(generated_ids, device=self.device)
        self._synchronize()
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
