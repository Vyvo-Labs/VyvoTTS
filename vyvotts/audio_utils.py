"""Compatibility helpers for Hugging Face and torchcodec audio values."""

from __future__ import annotations

import io
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import soundfile as sf
import torch


def decode_audio_value(
    value: Any,
    *,
    default_sample_rate: int | None = None,
) -> tuple[torch.Tensor, int]:
    """Return waveform and sample rate from common dataset audio containers.

    Datasets 4.x returns a torchcodec ``AudioDecoder`` for decoded ``Audio``
    columns, while older releases returned an ``array``/``sampling_rate``
    mapping. Paths, embedded bytes, tensors, and both representations are
    accepted without importing torchcodec eagerly.
    """

    if value is None:
        raise ValueError("audio value is missing")

    if hasattr(value, "get_all_samples"):
        samples = value.get_all_samples()
        data = getattr(samples, "data", None)
        sample_rate = getattr(samples, "sample_rate", None)
        if data is None or sample_rate is None:
            raise TypeError("AudioDecoder samples must expose data and sample_rate")
        return torch.as_tensor(data).detach().float().cpu(), int(sample_rate)

    if isinstance(value, (str, Path)):
        array, sample_rate = sf.read(str(value), dtype="float32")
        return torch.as_tensor(array), int(sample_rate)

    if isinstance(value, Mapping):
        if value.get("array") is not None:
            sample_rate = value.get("sampling_rate", default_sample_rate)
            if sample_rate is None:
                raise ValueError("array audio requires sampling_rate")
            return torch.as_tensor(value["array"]).detach().float().cpu(), int(sample_rate)
        if value.get("bytes") is not None:
            array, sample_rate = sf.read(io.BytesIO(value["bytes"]), dtype="float32")
            return torch.as_tensor(array), int(sample_rate)
        if value.get("path") is not None:
            array, sample_rate = sf.read(str(value["path"]), dtype="float32")
            return torch.as_tensor(array), int(sample_rate)
        raise ValueError("audio mapping needs array, bytes, or path")

    if isinstance(value, torch.Tensor):
        if default_sample_rate is None:
            raise ValueError("tensor audio requires default_sample_rate")
        return value.detach().float().cpu(), int(default_sample_rate)

    try:
        tensor = torch.as_tensor(value).detach().float().cpu()
    except (TypeError, ValueError) as exc:
        raise TypeError(f"unsupported audio value type: {type(value).__name__}") from exc
    if default_sample_rate is None:
        raise ValueError("array-like audio requires default_sample_rate")
    return tensor, int(default_sample_rate)


__all__ = ["decode_audio_value"]
