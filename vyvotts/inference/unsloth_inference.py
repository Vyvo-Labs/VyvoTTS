import torch
from typing import Optional, Dict, Any
from unsloth import FastLanguageModel

from vyvotts.inference.base import BaseVyvoTTSInference


class VyvoTTSUnslothInference(BaseVyvoTTSInference):
    """Memory-efficient TTS inference engine using Unsloth backend.

    Supports 4-bit and 8-bit quantization. Keeps codec model on CPU
    to minimize GPU memory usage.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        config_path: Optional[str] = None,
        model_name: str = "Vyvo/VyvoTTS-v2-Neuvillette",
        tokenizer_name: Optional[str] = None,
        codec_type: str = "snac",
        codec_model_name: str = None,
        max_seq_length: int = 8192,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
    ):
        super().__init__(config, config_path)

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=None,  # Auto detection
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
        )
        FastLanguageModel.for_inference(self.model)

        # Codec on CPU to save GPU memory
        self.codec = self._load_codec(codec_type, codec_model_name, device="cpu")

    def generate(
        self,
        text: str,
        voice: Optional[str] = None,
        max_new_tokens: int = 1200,
        temperature: float = 0.6,
        top_p: float = 0.95,
        repetition_penalty: float = 1.1,
        do_sample: bool = True,
        output_path: Optional[str] = None,
    ) -> Optional[torch.Tensor]:
        """Generate speech from text input.

        Args:
            text: Input text to convert to speech.
            voice: Optional voice identifier for voice cloning.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Top-p sampling parameter.
            repetition_penalty: Penalty for token repetition.
            do_sample: Whether to use sampling.
            output_path: Optional path to save audio file.

        Returns:
            Audio tensor containing the generated speech, or None on failure.
        """
        input_ids = self._build_prompt_tokens(text, voice).to("cuda")
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                num_return_sequences=1,
                eos_token_id=self.END_OF_SPEECH,
                use_cache=True,
            )

        # Decode on CPU (where codec model lives)
        audio_samples = self._extract_audio_from_tokens(generated_ids, device="cpu")
        audio = audio_samples[0] if audio_samples else None

        if output_path and audio is not None:
            self.save_audio(audio, output_path)

        return audio


def text_to_speech(
    text: str,
    voice: Optional[str] = None,
    output_path: Optional[str] = None,
    **kwargs,
) -> Optional[torch.Tensor]:
    """Generate speech from text using Unsloth model."""
    engine = VyvoTTSUnslothInference()
    audio = engine.generate(text, voice, output_path=output_path, **kwargs)
    return audio


if __name__ == "__main__":
    engine = VyvoTTSUnslothInference(load_in_4bit=True)

    test_text = "Hey there my name is Elise, and I'm a speech generation model that can sound like a person."
    audio = engine.generate(test_text)

    if audio is not None:
        engine.save_audio(audio, "output.wav")
        print(f"Audio saved with shape: {audio.shape}")
    else:
        print("Failed to generate audio")
