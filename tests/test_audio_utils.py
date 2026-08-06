from types import SimpleNamespace

import torch

from vyvotts.audio_utils import decode_audio_value
from vyvotts.train.post_training.online import _reference_waveform


class _FakeAudioDecoder:
    def get_all_samples(self):
        return SimpleNamespace(
            data=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            sample_rate=16000,
        )


def test_torchcodec_audio_decoder_is_supported_without_eager_import():
    waveform, sample_rate = decode_audio_value(_FakeAudioDecoder())
    assert waveform.shape == (2, 2)
    assert sample_rate == 16000


def test_online_reference_audio_decoder_is_mono_and_resampled():
    waveform = _reference_waveform(_FakeAudioDecoder(), 24000)
    assert waveform.ndim == 1
    assert waveform.numel() == 3
