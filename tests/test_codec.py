import torch

from vyvotts.codec import SNACCodec


class _RecordingSNACModel:
    def __init__(self):
        self.code_devices = None

    def decode(self, codes):
        self.code_devices = [code.device.type for code in codes]
        return torch.ones(1, 1, 4)


def test_snac_decodes_on_model_device_before_moving_output():
    codec = SNACCodec.__new__(SNACCodec)
    codec.model = _RecordingSNACModel()
    codec.device = "cpu"
    codec._optimized_decode = None

    audio = codec.decode(
        [0, 4096, 8192, 12288, 16384, 20480, 24576],
        device="meta",
    )

    assert codec.model.code_devices == ["cpu", "cpu", "cpu"]
    assert audio.device.type == "meta"
