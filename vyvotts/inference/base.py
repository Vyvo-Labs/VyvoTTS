import yaml
import torch
import soundfile as sf
from pathlib import Path
from typing import List, Optional, Dict, Any

from vyvotts.codec import load_codec, BaseCodec
from vyvotts.inference.constraints import (
    AudioTokenLogitsProcessor,
    AudioTokenSequenceError,
    extract_audio_codes,
)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


class BaseVyvoTTSInference:
    """Base class for all VyvoTTS inference engines.

    Handles config loading, token constants, audio codec decoding,
    and audio file saving. Subclasses implement model loading and generation.
    """

    SAMPLE_RATE = 24000
    DEFAULT_CONFIG_PATH = "vyvotts/configs/inference/lfm2.yaml"

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        config_path: Optional[str] = None,
    ):
        if config is not None:
            self.config = config
        elif config_path is not None:
            self.config = load_config(config_path)
        else:
            self.config = load_config(self.DEFAULT_CONFIG_PATH)

        self._setup_token_constants()

    def _setup_token_constants(self):
        """Set special token IDs from config."""
        c = self.config
        self.TOKENIZER_LENGTH = c['TOKENIZER_LENGTH']
        self.START_OF_TEXT = c['START_OF_TEXT']
        self.END_OF_TEXT = c['END_OF_TEXT']
        self.START_OF_SPEECH = c['START_OF_SPEECH']
        self.END_OF_SPEECH = c['END_OF_SPEECH']
        self.START_OF_HUMAN = c['START_OF_HUMAN']
        self.END_OF_HUMAN = c['END_OF_HUMAN']
        self.START_OF_AI = c['START_OF_AI']
        self.END_OF_AI = c['END_OF_AI']
        self.PAD_TOKEN = c['PAD_TOKEN']
        self.AUDIO_TOKENS_START = c['AUDIO_TOKENS_START']

    def _load_codec(
        self,
        codec_type: Optional[str] = None,
        codec_model_name: str = None,
        device: str = "cpu",
        optimize: bool = False,
        **kwargs,
    ) -> BaseCodec:
        """Load audio codec model.

        Args:
            codec_type: Codec type — "snac" or "mimi".
            codec_model_name: HuggingFace model ID or local path.
            device: Target device.
            optimize: (SNAC only) Apply fast-snac FP16 Triton optimizations.
        """
        codec_type = codec_type or self.config.get("CODEC_TYPE", "snac")
        codec_model_name = codec_model_name or self.config.get("CODEC_MODEL")
        num_codebooks = self.config.get("NUM_CODEBOOKS")
        if num_codebooks is not None:
            kwargs["num_codebooks"] = num_codebooks

        return load_codec(
            codec_type=codec_type,
            model_name=codec_model_name,
            device=device,
            optimize=optimize,
            **kwargs,
        )

    # Speaker IDs seen during training. Random selection is opt-in because a
    # speaker prefix is not present in single-speaker training examples.
    DEFAULT_SPEAKERS = [
        "EN_B00000_S00000", "EN_B00000_S00010", "EN_B00000_S00020",
        "EN_B00000_S00030", "EN_B00000_S00040", "EN_B00000_S00050",
        "EN_B00000_S00060", "EN_B00000_S00070", "EN_B00000_S00080",
        "EN_B00000_S00090", "EN_B00000_S00100", "EN_B00000_S00110",
    ]

    def _build_prompt_tokens(
        self,
        text: str,
        voice: Optional[str] = None,
        use_random_voice: bool = False,
    ) -> torch.Tensor:
        """Tokenize text and wrap with special tokens.

        A speaker prefix is included only when ``voice`` is supplied or random
        speaker selection is explicitly requested.

        Returns:
            Token IDs tensor of shape (1, seq_len).
        """
        if voice is None and use_random_voice:
            import random

            voice = random.choice(self.DEFAULT_SPEAKERS)
        prompt = f"{voice}: {text}" if voice is not None else text
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids

        start = torch.tensor([[self.START_OF_HUMAN]], dtype=torch.int64)
        end = torch.tensor([[self.END_OF_TEXT, self.END_OF_HUMAN]], dtype=torch.int64)

        return torch.cat([start, input_ids, end], dim=1)

    def _pad_and_batch(
        self,
        token_sequences: List[torch.Tensor],
        device: str = "cpu",
    ):
        """Left-pad token sequences to equal length and create attention masks.

        Returns:
            Tuple of (input_ids, attention_mask) tensors on the target device.
        """
        max_len = max(seq.shape[1] for seq in token_sequences)

        padded = []
        masks = []
        for seq in token_sequences:
            pad_len = max_len - seq.shape[1]
            padded.append(torch.cat([
                torch.full((1, pad_len), self.PAD_TOKEN, dtype=torch.int64),
                seq,
            ], dim=1))
            masks.append(torch.cat([
                torch.zeros(1, pad_len, dtype=torch.int64),
                torch.ones(1, seq.shape[1], dtype=torch.int64),
            ], dim=1))

        return (
            torch.cat(padded, dim=0).to(device),
            torch.cat(masks, dim=0).to(device),
        )

    def _audio_logits_processor(
        self,
        prompt_length: int,
        min_audio_frames: int = 1,
    ) -> AudioTokenLogitsProcessor:
        """Build the codec-aware generation grammar for this engine."""
        return AudioTokenLogitsProcessor(
            prompt_length=prompt_length,
            start_of_ai=self.START_OF_AI,
            start_of_speech=self.START_OF_SPEECH,
            end_of_speech=self.END_OF_SPEECH,
            audio_tokens_start=self.AUDIO_TOKENS_START,
            codes_per_group=self.codec.codes_per_group,
            codebook_size=self.codec.codebook_size,
            pad_token_id=self.PAD_TOKEN,
            min_audio_frames=min_audio_frames,
        )

    def _extract_audio_from_tokens(
        self,
        generated_ids: torch.Tensor,
        device: str = "cpu",
    ) -> List[Optional[torch.Tensor]]:
        """Extract audio tokens from generated IDs and decode to waveforms.

        Each row is parsed independently. A malformed row yields ``None`` at
        the same batch index; tokens are never clamped or frame-trimmed here.
        """
        if generated_ids.ndim == 1:
            generated_ids = generated_ids.unsqueeze(0)
        if generated_ids.ndim != 2:
            raise ValueError("generated_ids must be a rank-1 or rank-2 tensor")

        audio_samples: List[Optional[torch.Tensor]] = []
        for row in generated_ids:
            try:
                code_list = extract_audio_codes(
                    row,
                    start_of_speech=self.START_OF_SPEECH,
                    end_of_speech=self.END_OF_SPEECH,
                    audio_tokens_start=self.AUDIO_TOKENS_START,
                    codes_per_group=self.codec.codes_per_group,
                    codebook_size=self.codec.codebook_size,
                )
            except AudioTokenSequenceError:
                audio_samples.append(None)
                continue

            audio = self.codec.decode(code_list, device=device)
            if audio is not None:
                # Apply fade-out to avoid click/artifact at the end
                audio = audio.clone()
                fade_samples = min(int(0.05 * self.codec.sample_rate), audio.shape[-1])
                if fade_samples > 0:
                    fade = torch.linspace(1.0, 0.0, fade_samples, device=audio.device)
                    audio[..., -fade_samples:] *= fade
            audio_samples.append(audio)

        return audio_samples

    def save_audio(
        self,
        audio_tensor: torch.Tensor,
        output_path: str,
        sample_rate: int = None,
    ) -> None:
        """Save audio tensor to WAV file."""
        if audio_tensor is None:
            raise ValueError("No audio tensor provided")

        sample_rate = sample_rate or self.SAMPLE_RATE
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        audio_numpy = audio_tensor.detach().squeeze().cpu().numpy()
        sf.write(output_path, audio_numpy, sample_rate)
