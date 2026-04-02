import yaml
import torch
import soundfile as sf
from pathlib import Path
from typing import List, Optional, Dict, Any
from snac import SNAC


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


class BaseVyvoTTSInference:
    """Base class for all VyvoTTS inference engines.

    Handles config loading, token constants, SNAC audio decoding,
    and audio file saving. Subclasses implement model loading and generation.
    """

    CODES_PER_GROUP = 7
    SAMPLE_RATE = 24000
    DEFAULT_CONFIG_PATH = "vyvotts/configs/inference/qwen3.yaml"

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

    def _load_snac_model(
        self,
        snac_model_name: str = "hubertsiuzdak/snac_24khz",
        device: str = "cpu",
        optimize: bool = False,
    ) -> SNAC:
        """Load SNAC audio codec model.

        Args:
            snac_model_name: HuggingFace model ID or local path.
            device: Target device.
            optimize: If True, apply fast-snac FP16 Triton optimizations.
        """
        model = SNAC.from_pretrained(snac_model_name)
        model = model.to(device)

        if optimize and device != "cpu":
            from snac.optimize import optimize_snac_triton
            # Generate sample codes for warmup
            sample_audio = torch.randn(1, 1, self.SAMPLE_RATE, device=device)
            with torch.no_grad():
                sample_codes = model.encode(sample_audio)
            self._optimized_decode = optimize_snac_triton(
                model, sample_codes, dtype="fp16", use_compile=True,
            )

        return model

    def _build_prompt_tokens(
        self,
        text: str,
        voice: Optional[str] = None,
    ) -> torch.Tensor:
        """Tokenize text and wrap with special tokens.

        Returns:
            Token IDs tensor of shape (1, seq_len).
        """
        prompt = f"{voice}: {text}" if voice else text
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

    def _redistribute_codes(
        self,
        code_list: List[int],
        device: str = "cpu",
    ) -> Optional[torch.Tensor]:
        """De-interleave flat audio codes into 3 SNAC layers and decode to audio.

        SNAC uses 3 hierarchical codebook levels. Each audio frame produces 7
        interleaved codes with per-level offsets of n*4096:
            [L0, L1a+4096, L2a+2*4096, L2b+3*4096, L1b+4*4096, L2c+5*4096, L2d+6*4096]

        This method reverses the interleaving using vectorized tensor operations.
        """
        num_groups = len(code_list) // self.CODES_PER_GROUP
        if num_groups == 0:
            return None

        codes = torch.tensor(
            code_list[:num_groups * self.CODES_PER_GROUP]
        ).view(num_groups, self.CODES_PER_GROUP)

        layer_0 = codes[:, 0]
        layer_1 = torch.stack([
            codes[:, 1] - 4096,
            codes[:, 4] - 4 * 4096,
        ], dim=1).reshape(-1)
        layer_2 = torch.stack([
            codes[:, 2] - 2 * 4096,
            codes[:, 3] - 3 * 4096,
            codes[:, 5] - 5 * 4096,
            codes[:, 6] - 6 * 4096,
        ], dim=1).reshape(-1)

        # Clamp to valid codebook range (0-4095)
        layer_0 = layer_0.clamp(0, 4095)
        layer_1 = layer_1.clamp(0, 4095)
        layer_2 = layer_2.clamp(0, 4095)

        snac_codes = [layer.unsqueeze(0).to(device) for layer in (layer_0, layer_1, layer_2)]
        if hasattr(self, '_optimized_decode'):
            return self._optimized_decode(snac_codes)
        return self.snac_model.decode(snac_codes)

    def _extract_audio_from_tokens(
        self,
        generated_ids: torch.Tensor,
        device: str = "cpu",
    ) -> List[torch.Tensor]:
        """Extract audio tokens from generated IDs and decode to waveforms.

        Finds the last START_OF_SPEECH marker, extracts everything after it,
        removes END_OF_SPEECH markers, and decodes via SNAC.
        """
        # Crop to content after last START_OF_SPEECH
        indices = (generated_ids == self.START_OF_SPEECH).nonzero(as_tuple=True)
        if len(indices[1]) > 0:
            last_idx = indices[1][-1].item()
            cropped = generated_ids[:, last_idx + 1:]
        else:
            cropped = generated_ids

        audio_samples = []
        for row in cropped:
            row = row[row != self.END_OF_SPEECH]

            # Trim to complete frames (multiple of CODES_PER_GROUP)
            n = (row.size(0) // self.CODES_PER_GROUP) * self.CODES_PER_GROUP
            if n == 0:
                continue

            # Vectorized offset subtraction
            code_list = (row[:n] - self.AUDIO_TOKENS_START).tolist()

            audio = self._redistribute_codes(code_list, device=device)
            if audio is not None:
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
