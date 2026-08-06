import json

import pytest

from vyvotts.train.post_training.data import (
    fpo_mask_from_error_spans,
    select_preference_pair,
)
from vyvotts.train.post_training.preference import _load_raw_pair_dataset
from vyvotts.train.post_training.rewards import CompositeReward


def test_local_fpo_span_selects_complete_codec_frames():
    mask = fpo_mask_from_error_spans(
        12,
        [{"start_frame": 1, "end_frame": 3, "type": "pronunciation"}],
        codes_per_group=2,
        audio_start_index=2,
    )
    assert mask == [False, False, False, False, True, True, True, True, False, False, False, False]


def test_repetition_fpo_span_selects_causal_tail():
    mask = fpo_mask_from_error_spans(
        10,
        [{"start_frame": 2, "type": "repetition"}],
        codes_per_group=2,
    )
    assert mask == [False] * 6 + [True] * 4


def test_pair_selection_applies_reward_gap_and_pareto_guard():
    reward = CompositeReward(
        {"wer_reward": 0.7, "quality_reward": 0.3},
        pareto_metrics=["wer_reward", "quality_reward"],
    )
    candidates = [
        {
            "completion_input_ids": [5, 6, 10],
            "metrics": {"wer_reward": 0.9, "quality_reward": 0.9},
        },
        {
            "completion_input_ids": [5, 6, 11],
            "metrics": {"wer_reward": 0.2, "quality_reward": 0.2},
            "error_spans": [{"start_frame": 0, "end_frame": 1, "type": "pronunciation"}],
        },
        {
            "completion_input_ids": [5, 6, 12],
            "metrics": {"wer_reward": 0.95, "quality_reward": 0.1},
        },
    ]

    pair = select_preference_pair(
        [1, 2],
        candidates,
        reward,
        minimum_reward_gap=0.3,
        codes_per_group=1,
        weight_by_gap=True,
    )
    assert pair["chosen_input_ids"] == [5, 6, 10]
    assert pair["rejected_input_ids"] == [5, 6, 11]
    assert pair["sample_weight"] == pytest.approx(0.7)
    assert pair["chosen_selection_mask"] == [False, False, False]
    assert pair["rejected_selection_mask"] == [False, False, True]


def test_pair_selection_rejects_ambiguous_small_gap():
    reward = CompositeReward({"score": 1.0})
    candidates = [
        {"completion_input_ids": [1], "metrics": {"score": 0.6}},
        {"completion_input_ids": [2], "metrics": {"score": 0.5}},
    ]
    with pytest.raises(ValueError, match="minimum_reward_gap"):
        select_preference_pair([9], candidates, reward, minimum_reward_gap=0.3)


def test_preference_training_loads_pair_builder_jsonl(tmp_path):
    path = tmp_path / "pairs.jsonl"
    path.write_text(
        json.dumps(
            {
                "prompt_input_ids": [1, 2],
                "chosen_input_ids": [3, 4],
                "rejected_input_ids": [5, 6],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    dataset = _load_raw_pair_dataset({"data": {"dataset": str(path)}}, "dpo")
    assert dataset[0]["chosen_input_ids"] == [3, 4]
