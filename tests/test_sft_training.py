import pytest
import torch
import torch.nn.functional as F

from vyvotts.train.post_training.sft import (
    CompletionOnlyCollator,
    completion_only_labels,
    weighted_audio_causal_loss,
)


def test_completion_labels_mask_prompt_but_keep_response_headers():
    ids = [1, 2, 3, 4, 5, 10, 11]
    assert completion_only_labels(ids, end_of_human=4) == [
        -100,
        -100,
        -100,
        -100,
        5,
        10,
        11,
    ]


def test_completion_collator_remasks_legacy_full_sequence_labels():
    collator = CompletionOnlyCollator(pad_token_id=0, end_of_human=4)
    batch = collator(
        [
            {"input_ids": [1, 4, 5, 6], "labels": [1, 4, 5, 6]},
            {"input_ids": [2, 3, 4, 7, 8]},
        ]
    )

    assert batch["input_ids"].tolist() == [[1, 4, 5, 6, 0], [2, 3, 4, 7, 8]]
    assert batch["labels"].tolist() == [
        [-100, -100, 5, 6, -100],
        [-100, -100, -100, 7, 8],
    ]
    assert batch["attention_mask"].tolist() == [[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]]


def test_audio_loss_applies_codebook_and_boundary_weights():
    torch.manual_seed(0)
    logits = torch.randn(1, 5, 20)
    labels = torch.tensor([[-100, 10, 13, 5, 6]])

    actual = weighted_audio_causal_loss(
        logits,
        labels,
        audio_tokens_start=10,
        codebook_size=3,
        codebook_weights=[2.0, 0.5],
        boundary_token_ids=[5],
        boundary_weight=3.0,
    )
    raw = F.cross_entropy(
        logits[:, :-1].transpose(1, 2), labels[:, 1:], reduction="none", ignore_index=-100
    )[0]
    expected = (raw * torch.tensor([2.0, 0.5, 3.0, 1.0])).sum() / 6.5
    assert torch.allclose(actual, expected)


def test_audio_loss_rejects_invalid_codec_configuration():
    logits = torch.zeros(1, 2, 4)
    labels = torch.tensor([[-100, 1]])
    with pytest.raises(ValueError, match="finite"):
        weighted_audio_causal_loss(
            logits,
            labels,
            audio_tokens_start=1,
            codebook_size=1,
            codebook_weights=[float("nan")],
        )
