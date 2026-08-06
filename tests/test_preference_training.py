from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from vyvotts.train.post_training import (
    PreferenceTrainingConfig,
    RawTokenPreferenceCollator,
    RawTokenPreferenceTrainer,
    completion_log_probs,
    direct_preference_loss,
    fine_grained_preference_loss,
    pack_completion_values,
    self_preference_loss,
)
from vyvotts.train.post_training import preference as preference_module


def test_raw_token_collator_masks_prompt_padding_and_fpo_tokens() -> None:
    collator = RawTokenPreferenceCollator(pad_token_id=99)
    batch = collator(
        [
            {
                "prompt_input_ids": [1, 2],
                "chosen_input_ids": [3, 4],
                "rejected_input_ids": [5],
                "chosen_selection_mask": [0, 1],
                "rejected_selection_mask": [1],
                "sample_weight": 2.0,
            }
        ]
    )

    assert batch["chosen_input_ids"].tolist() == [[1, 2, 3, 4]]
    assert batch["rejected_input_ids"].tolist() == [[1, 2, 5, 99]]
    assert batch["chosen_labels"].tolist() == [[-100, -100, 3, 4]]
    assert batch["rejected_labels"].tolist() == [[-100, -100, 5, -100]]
    assert batch["chosen_completion_mask"].tolist() == [[False, False, True, True]]
    assert batch["chosen_selection_mask"].tolist() == [[False, False, False, True]]
    assert batch["rejected_attention_mask"].tolist() == [[True, True, True, False]]
    assert batch["prompt_lengths"].tolist() == [2]
    assert batch["chosen_completion_lengths"].tolist() == [2]
    assert batch["sample_weight"].tolist() == [2.0]


def test_raw_token_collator_rejects_silent_truncation_and_bad_masks() -> None:
    feature = {
        "prompt_input_ids": [1, 2],
        "chosen_input_ids": [3, 4],
        "rejected_input_ids": [5],
    }
    with pytest.raises(ValueError, match="exceeding max_length"):
        RawTokenPreferenceCollator(0, max_length=3)([feature])

    bad_mask = {**feature, "chosen_selection_mask": [1]}
    with pytest.raises(ValueError, match="has length"):
        RawTokenPreferenceCollator(0)([bad_mask])

    bad_id = {**feature, "chosen_input_ids": [3, -1]}
    with pytest.raises(ValueError, match="non-negative"):
        RawTokenPreferenceCollator(0)([bad_id])


def test_completion_log_probs_shift_and_reduce_only_completion() -> None:
    logits = torch.zeros(1, 4, 6)
    labels = torch.tensor([[-100, -100, 3, 4]])
    completion_mask = torch.tensor([[False, False, True, True]])
    selection_mask = torch.tensor([[False, False, False, True]])

    result = completion_log_probs(
        logits,
        labels,
        completion_mask=completion_mask,
        selection_mask=selection_mask,
    )
    expected = -math.log(6)
    assert result.token_counts.tolist() == [2]
    assert result.selected_token_counts.tolist() == [1]
    assert result.sequence_logps.item() == pytest.approx(2 * expected)
    assert result.mean_logps.item() == pytest.approx(expected)
    assert result.selected_sequence_logps.item() == pytest.approx(expected)

    with pytest.raises(ValueError, match="outside model vocabulary"):
        completion_log_probs(logits, torch.tensor([[-100, -100, 3, 9]]))


def test_pack_completion_values_removes_prompt_offsets() -> None:
    values = torch.tensor([[0.0, 0.0, 1.0, 2.0], [0.0, 3.0, 4.0, 0.0]])
    valid = torch.tensor(
        [[False, False, True, True], [False, True, True, False]], dtype=torch.bool
    )
    selected = torch.tensor(
        [[False, False, False, True], [False, True, False, False]], dtype=torch.bool
    )
    packed, packed_valid, packed_selected = pack_completion_values(values, valid, selected)
    assert packed.tolist() == [[1.0, 2.0], [3.0, 4.0]]
    assert packed_valid.all()
    assert packed_selected.tolist() == [[False, True], [True, False]]


def test_dpo_matches_reference_anchored_closed_form() -> None:
    result = direct_preference_loss(
        torch.tensor([-1.0]),
        torch.tensor([-2.0]),
        torch.tensor([-1.5]),
        torch.tensor([-1.5]),
        beta=0.2,
    )
    assert result.loss.item() == pytest.approx(torch.nn.functional.softplus(torch.tensor(-0.2)).item())
    assert result.logits.item() == pytest.approx(0.2)
    assert result.metrics["preference_accuracy"].item() == 1.0
    assert result.metrics["reward_margin"].item() == pytest.approx(0.2)


def test_spo_uses_2025_silu_objective_and_mean_logps() -> None:
    assert PreferenceTrainingConfig(objective="spo").beta == 2.5
    result = self_preference_loss(
        torch.tensor([-1.0]),
        torch.tensor([-2.0]),
        beta=2.5,
        gamma=0.8,
    )
    expected_z = -(2.5 * 1.0 - 0.8)
    assert result.loss.item() == pytest.approx(
        torch.nn.functional.silu(torch.tensor(expected_z)).item()
    )
    assert result.logits.item() == pytest.approx(-expected_z)
    assert result.metrics["spo_z"].item() == pytest.approx(expected_z)


def test_fpo_ignores_unselected_token_rewards() -> None:
    result = fine_grained_preference_loss(
        policy_chosen_token_logps=torch.tensor([[100.0, 2.0, -100.0]]),
        policy_rejected_token_logps=torch.zeros(1, 3),
        reference_chosen_token_logps=torch.zeros(1, 3),
        reference_rejected_token_logps=torch.zeros(1, 3),
        selection_mask=torch.tensor([[False, True, False]]),
        beta=1.0,
    )
    expected = torch.nn.functional.softplus(torch.tensor(-2.0)).item()
    assert result.loss.item() == pytest.approx(expected)
    assert result.metrics["selected_tokens"].item() == 1.0
    assert result.metrics["selected_fraction"].item() == pytest.approx(1 / 3)


class _TinyCausalLM(torch.nn.Module):
    def __init__(self, vocab_size: int = 16) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, 8)
        self.head = torch.nn.Linear(8, vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        del attention_mask, use_cache
        return SimpleNamespace(logits=self.head(self.embedding(input_ids)))


def test_trainer_computes_raw_token_dpo_and_buffers_metrics(tmp_path) -> None:
    transformers = pytest.importorskip("transformers")
    policy = _TinyCausalLM()
    reference = _TinyCausalLM()
    reference.load_state_dict(policy.state_dict())
    collator = RawTokenPreferenceCollator(pad_token_id=0)
    batch = collator(
        [
            {
                "prompt_input_ids": [1, 2],
                "chosen_input_ids": [3, 4],
                "rejected_input_ids": [5, 6],
            }
        ]
    )
    args = transformers.TrainingArguments(
        output_dir=str(tmp_path),
        per_device_train_batch_size=1,
        report_to="none",
        remove_unused_columns=True,
    )
    trainer = RawTokenPreferenceTrainer(
        model=policy,
        reference_model=reference,
        preference_config=PreferenceTrainingConfig(objective="dpo", beta=0.1),
        args=args,
        data_collator=collator,
    )
    device = next(policy.parameters()).device
    batch = {name: value.to(device) for name, value in batch.items()}
    loss = trainer.compute_loss(policy, batch)
    assert loss.item() == pytest.approx(math.log(2), rel=1e-5)
    loss.backward()
    assert policy.embedding.weight.grad is not None
    assert all(parameter.grad is None for parameter in reference.parameters())
    assert trainer._preference_metrics["train"]["preference_accuracy"] == [0.0]


@pytest.mark.parametrize("objective", ["dpo", "fpo"])
def test_reference_based_trainers_require_distinct_reference(objective: str, tmp_path) -> None:
    transformers = pytest.importorskip("transformers")
    model = _TinyCausalLM()
    args = transformers.TrainingArguments(output_dir=str(tmp_path), report_to="none")
    with pytest.raises(ValueError, match="requires reference_model"):
        RawTokenPreferenceTrainer(
            model=model,
            args=args,
            preference_config=PreferenceTrainingConfig(objective=objective),
        )


def test_train_from_config_runs_and_saves_final_artifacts(monkeypatch, tmp_path) -> None:
    class FakeTokenizer:
        pad_token_id = 0

        def __len__(self):
            return 16

        def save_pretrained(self, path):
            path.mkdir(parents=True, exist_ok=True)
            (path / "tokenizer.json").write_text("{}", encoding="utf-8")

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(16, 2)
            self.config = SimpleNamespace(pad_token_id=None)

        def get_input_embeddings(self):
            return self.embedding

    trainer_instances = []

    class FakeTrainer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.accelerator = SimpleNamespace(wait_for_everyone=lambda: None)
            self.resume = None
            trainer_instances.append(self)

        def train(self, resume_from_checkpoint=None):
            self.resume = resume_from_checkpoint

        def save_model(self, path):
            torch_path = Path(path)
            torch_path.mkdir(parents=True, exist_ok=True)
            (torch_path / "model.safetensors").write_bytes(b"test")

        def is_world_process_zero(self):
            return True

    dataset = [
        {
            "prompt_input_ids": [1, 2],
            "chosen_input_ids": [3, 4],
            "rejected_input_ids": [5, 6],
        }
    ]
    monkeypatch.setattr(preference_module, "_load_raw_pair_dataset", lambda *_: dataset)
    monkeypatch.setattr(
        preference_module.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: FakeTokenizer(),
    )
    monkeypatch.setattr(
        preference_module.AutoModelForCausalLM,
        "from_pretrained",
        lambda *args, **kwargs: FakeModel(),
    )
    monkeypatch.setattr(preference_module, "RawTokenPreferenceTrainer", FakeTrainer)

    output_dir = tmp_path / "preference"
    final_dir = preference_module.train_from_config(
        {
            "model": {
                "name_or_path": "policy",
                "reference_name_or_path": "reference",
                "tokenizer_name_or_path": "tokenizer",
            },
            "data": {"dataset": "pairs", "pad_token_id": 0, "max_length": 32},
            "preference": {"objective": "dpo", "beta": 0.2},
            "training": {
                "output_dir": str(output_dir),
                "batch_size": 2,
                "epochs": 1,
                "resume_from_checkpoint": "checkpoint-10",
            },
        }
    )

    assert final_dir == str(output_dir / "final")
    assert (output_dir / "final" / "model.safetensors").is_file()
    assert (output_dir / "final" / "tokenizer.json").is_file()
    assert (output_dir / "final" / "post_training_config.yaml").is_file()
    trainer = trainer_instances[0]
    assert trainer.resume == "checkpoint-10"
    assert trainer.kwargs["args"].per_device_train_batch_size == 2
    assert trainer.kwargs["preference_config"].objective == "dpo"
    assert trainer.kwargs["reference_model"] is not None
