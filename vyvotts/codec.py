import torch
from abc import ABC, abstractmethod
from typing import List, Optional


class BaseCodec(ABC):
    """Abstract base class for audio codecs."""

    @property
    @abstractmethod
    def codes_per_group(self) -> int:
        """Number of interleaved codes per audio frame group."""
        ...

    @property
    @abstractmethod
    def codebook_size(self) -> int:
        """Size of each codebook."""
        ...

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """Audio sample rate in Hz."""
        ...

    @abstractmethod
    def encode(self, waveform: torch.Tensor) -> List[int]:
        """Encode waveform to interleaved code list with per-codebook offsets.

        Args:
            waveform: Audio tensor of shape [1, 1, T] at self.sample_rate Hz.

        Returns:
            Flat list of interleaved token IDs (without AUDIO_TOKENS_START offset).
        """
        ...

    @abstractmethod
    def decode(self, code_list: List[int], device: str = "cpu") -> Optional[torch.Tensor]:
        """Decode interleaved code list back to audio waveform.

        Args:
            code_list: Flat list of codes (with per-codebook offsets, without AUDIO_TOKENS_START).
            device: Target device for output tensor.

        Returns:
            Audio tensor or None if code_list is too short.
        """
        ...


class SNACCodec(BaseCodec):
    """SNAC audio codec (3 hierarchical codebook levels, 7 codes per group).

    Interleaving pattern per frame:
        [L0, L1a+4096, L2a+2*4096, L2b+3*4096, L1b+4*4096, L2c+5*4096, L2d+6*4096]
    """

    def __init__(self, model_name: str = "hubertsiuzdak/snac_24khz", device: str = "cpu", optimize: bool = False):
        from snac import SNAC

        self.model = SNAC.from_pretrained(model_name).to(device)
        self.model.eval()
        self.device = device
        self._optimized_decode = None

        if optimize and device != "cpu":
            from snac.optimize import optimize_snac_triton

            sample_audio = torch.randn(1, 1, 24000, device=device)
            with torch.no_grad():
                sample_codes = self.model.encode(sample_audio)
            self._optimized_decode = optimize_snac_triton(
                self.model, sample_codes, dtype="fp16", use_compile=True,
            )

    @property
    def codes_per_group(self) -> int:
        return 7

    @property
    def codebook_size(self) -> int:
        return 4096

    @property
    def sample_rate(self) -> int:
        return 24000

    def encode(self, waveform: torch.Tensor) -> List[int]:
        with torch.inference_mode():
            codes = self.model.encode(waveform.to(self.device))

        all_codes = []
        for i in range(codes[0].shape[1]):
            all_codes.extend([
                codes[0][0][i].item(),
                codes[1][0][2 * i].item() + 4096,
                codes[2][0][4 * i].item() + 2 * 4096,
                codes[2][0][4 * i + 1].item() + 3 * 4096,
                codes[1][0][2 * i + 1].item() + 4 * 4096,
                codes[2][0][4 * i + 2].item() + 5 * 4096,
                codes[2][0][4 * i + 3].item() + 6 * 4096,
            ])
        return all_codes

    def decode(self, code_list: List[int], device: str = "cpu") -> Optional[torch.Tensor]:
        num_groups = len(code_list) // self.codes_per_group
        if num_groups == 0:
            return None

        codes = torch.tensor(
            code_list[: num_groups * self.codes_per_group]
        ).view(num_groups, self.codes_per_group)

        layer_0 = codes[:, 0].clamp(0, 4095)
        layer_1 = torch.stack([
            codes[:, 1] - 4096,
            codes[:, 4] - 4 * 4096,
        ], dim=1).reshape(-1).clamp(0, 4095)
        layer_2 = torch.stack([
            codes[:, 2] - 2 * 4096,
            codes[:, 3] - 3 * 4096,
            codes[:, 5] - 5 * 4096,
            codes[:, 6] - 6 * 4096,
        ], dim=1).reshape(-1).clamp(0, 4095)

        snac_codes = [layer.unsqueeze(0).to(device) for layer in (layer_0, layer_1, layer_2)]
        if self._optimized_decode:
            return self._optimized_decode(snac_codes)
        return self.model.decode(snac_codes)


class MimiCodec(BaseCodec):
    """Mimi (Kyutai) audio codec — 8 codebooks at uniform 12.5 Hz frame rate.

    All codebooks share the same temporal resolution, so interleaving is simple:
        [CB0, CB1+2048, CB2+2*2048, ..., CB7+7*2048]
    """

    def __init__(self, model_name: str = "kyutai/mimi", device: str = "cpu", num_codebooks: int = 8):
        from transformers import MimiModel

        self.model = MimiModel.from_pretrained(model_name).to(device)
        self.model.eval()
        self.device = device
        self._num_codebooks = num_codebooks

    @property
    def codes_per_group(self) -> int:
        return self._num_codebooks

    @property
    def codebook_size(self) -> int:
        return 2048

    @property
    def sample_rate(self) -> int:
        return 24000

    def encode(self, waveform: torch.Tensor) -> List[int]:
        waveform = waveform.to(self.device)
        with torch.inference_mode():
            encoder_outputs = self.model.encode(waveform, num_quantizers=self._num_codebooks)
            audio_codes = encoder_outputs.audio_codes  # [B, K, T]

        audio_codes = audio_codes[0, :self._num_codebooks, :]  # [K, T]
        num_frames = audio_codes.shape[1]

        all_codes = []
        for t in range(num_frames):
            for k in range(self._num_codebooks):
                all_codes.append(audio_codes[k, t].item() + k * self.codebook_size)
        return all_codes

    def decode(self, code_list: List[int], device: str = "cpu") -> Optional[torch.Tensor]:
        num_groups = len(code_list) // self.codes_per_group
        if num_groups == 0:
            return None

        codes = torch.tensor(
            code_list[: num_groups * self.codes_per_group]
        ).view(num_groups, self.codes_per_group)

        # De-interleave: each column k has offset k * codebook_size
        layers = []
        for k in range(self._num_codebooks):
            layer = (codes[:, k] - k * self.codebook_size).clamp(0, self.codebook_size - 1)
            layers.append(layer)

        # Stack to [1, K, T] — each layer is [T], stack gives [K, T]
        audio_codes = torch.stack(layers, dim=0).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            decoder_outputs = self.model.decode(audio_codes)
            audio_values = decoder_outputs.audio_values  # [B, C, T]

        return audio_values.to(device)


def load_codec(
    codec_type: str = "snac",
    model_name: str = None,
    device: str = "cpu",
    optimize: bool = False,
    **kwargs,
) -> BaseCodec:
    """Factory function to create a codec instance.

    Args:
        codec_type: "snac" or "mimi".
        model_name: HuggingFace model ID. Defaults to canonical model per codec type.
        device: Target device.
        optimize: (SNAC only) Enable Triton FP16 optimizations.
        **kwargs: Extra args forwarded to codec constructor.
    """
    if codec_type == "snac":
        model_name = model_name or "hubertsiuzdak/snac_24khz"
        return SNACCodec(model_name=model_name, device=device, optimize=optimize)
    elif codec_type == "mimi":
        model_name = model_name or "kyutai/mimi"
        return MimiCodec(model_name=model_name, device=device, **kwargs)
    else:
        raise ValueError(f"Unknown codec type: {codec_type!r}. Supported: 'snac', 'mimi'")
