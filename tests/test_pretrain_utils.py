from pathlib import Path

import pytest

from vyvotts.train.pretrain.train import (
    GlobalBatchShuffleSampler,
    GradualRatioDataset,
    audio_token_capacity,
    calculate_total_steps,
    data_collator,
    extend_tokenizer,
    interpolate_ratio,
    parse_ratio,
    resolve_resume_checkpoint,
    use_full_state_dict_for_final_save,
)


class FakeTokenizer:
    def __init__(self, base_size=100):
        self.tokens = {f"base_{index}": index for index in range(base_size)}
        self.ids = {index: token for token, index in self.tokens.items()}
        self.pad_token = None

    def add_tokens(self, tokens):
        added = 0
        for token in tokens:
            if token not in self.tokens:
                token_id = len(self.tokens)
                self.tokens[token] = token_id
                self.ids[token_id] = token
                added += 1
        return added

    def convert_tokens_to_ids(self, token):
        return self.tokens[token]

    def convert_ids_to_tokens(self, token_id):
        return self.ids.get(token_id)


def test_ratio_parsing_and_interpolation_reach_configured_targets():
    assert parse_ratio("3:2") == pytest.approx(1.5)
    assert parse_ratio(2) == pytest.approx(2.0)
    assert interpolate_ratio(2.0, 1.0, step=4, total_steps=5) == 1.0
    with pytest.raises(ValueError, match="greater than zero"):
        parse_ratio("2:0")


def test_gradual_dataset_keeps_each_global_batch_on_one_source():
    qa = [{"source": "qa", "id": index} for index in range(12)]
    tts = [{"source": "tts", "id": index} for index in range(8)]
    dataset = GradualRatioDataset(
        qa,
        tts,
        batch_total=2,
        initial_ratio=2,
        final_ratio=1,
        total_steps=3,
    )

    assert [dataset[index]["source"] for index in range(6)] == [
        "qa",
        "qa",
        "qa",
        "qa",
        "tts",
        "tts",
    ]
    dataset.set_current_step(2)
    assert dataset.get_current_ratio() == 1.0
    assert [dataset[index]["source"] for index in range(8)] == [
        "qa",
        "qa",
        "tts",
        "tts",
        "qa",
        "qa",
        "tts",
        "tts",
    ]


def test_sampler_shuffles_whole_global_blocks_deterministically():
    sampler = GlobalBatchShuffleSampler(list(range(24)), global_batch_size=4, seed=11)
    first_epoch = list(sampler)
    repeated_epoch = list(sampler)
    blocks = [first_epoch[index : index + 4] for index in range(0, 24, 4)]

    assert first_epoch == repeated_epoch
    assert sorted(first_epoch) == list(range(24))
    assert all(block == list(range(block[0], block[0] + 4)) for block in blocks)

    sampler.set_epoch(1)
    assert list(sampler) != first_epoch


def test_collator_pads_inputs_masks_and_labels_independently():
    batch = data_collator(
        [
            {"input_ids": [1, 2], "labels": [-100, 2]},
            {"input_ids": [3]},
        ],
        pad_token_id=9,
    )
    assert batch["input_ids"].tolist() == [[1, 2], [3, 9]]
    assert batch["attention_mask"].tolist() == [[1, 1], [1, 0]]
    assert batch["labels"].tolist() == [[-100, 2], [3, -100]]


def test_tokenizer_extension_is_stable_and_assigns_pad_token():
    tokenizer = FakeTokenizer(base_size=100)
    assert extend_tokenizer(tokenizer, audio_capacity=8, pad_token_id=107) == 19
    assert tokenizer.pad_token == "<custom_token_7>"
    assert extend_tokenizer(tokenizer, audio_capacity=8, pad_token_id=107) == 0

    with pytest.raises(ValueError, match="pad_token"):
        extend_tokenizer(FakeTokenizer(base_size=100), audio_capacity=8, pad_token_id=99)
    shifted = FakeTokenizer(base_size=100)
    shifted.add_tokens(["unrelated"])
    with pytest.raises(ValueError, match="custom index 7"):
        extend_tokenizer(shifted, audio_capacity=8, pad_token_id=107)


def test_codec_capacity_and_step_count_validation():
    assert audio_token_capacity("snac") == 7 * 4096
    assert audio_token_capacity("mimi", 4) == 4 * 2048
    assert calculate_total_steps(101, 4, 2, 2, epochs=3) == 21
    assert calculate_total_steps(101, 4, 2, 2, epochs=3, max_steps=9) == 9
    assert calculate_total_steps(16, 4, 2, 1, epochs=1, drop_last=True) == 2
    with pytest.raises(ValueError, match="smaller than one global batch"):
        calculate_total_steps(3, 4, 2, 1, epochs=1, drop_last=True)
    with pytest.raises(ValueError, match="num_codebooks"):
        audio_token_capacity("mimi", 0)


def test_resume_resolution_uses_latest_or_explicit_checkpoint(tmp_path: Path):
    (tmp_path / "checkpoint-2").mkdir()
    latest = tmp_path / "checkpoint-10"
    latest.mkdir()

    assert resolve_resume_checkpoint(True, tmp_path) == str(latest)
    assert resolve_resume_checkpoint("latest", tmp_path) == str(latest)
    assert resolve_resume_checkpoint(str(tmp_path / "checkpoint-2"), tmp_path) == str(
        tmp_path / "checkpoint-2"
    )
    assert resolve_resume_checkpoint(False, tmp_path) is None

    with pytest.raises(ValueError, match="does not exist"):
        resolve_resume_checkpoint(str(tmp_path / "missing"), tmp_path)


def test_final_save_switches_sharded_fsdp_plugin_to_full_state():
    class Plugin:
        state_dict_type = "SHARDED_STATE_DICT"
        state_dict_config = object()
        optim_state_dict_config = object()

        def set_state_dict_type(self, value):
            assert self.state_dict_config is None
            assert self.optim_state_dict_config is None
            self.state_dict_type = value

    plugin = Plugin()
    trainer = type(
        "FakeTrainer",
        (),
        {
            "is_fsdp_enabled": True,
            "accelerator": type(
                "FakeAccelerator", (), {"state": type("State", (), {"fsdp_plugin": plugin})()}
            )(),
        },
    )()

    assert use_full_state_dict_for_final_save(trainer) is True
    assert plugin.state_dict_type == "FULL_STATE_DICT"
    assert use_full_state_dict_for_final_save(trainer) is False
