import pytest
import torch

from vyvotts.train.post_training.online import (
    OnlineTrainingConfig,
    clipped_policy_loss,
    group_relative_advantages,
    sampled_token_kl,
    validate_online_distributed_type,
)


def test_grpo_advantages_are_group_centered_and_optionally_scaled():
    rewards = torch.tensor([1.0, 2.0, 3.0])
    centered = group_relative_advantages(rewards)
    normalized = group_relative_advantages(rewards, normalize=True)

    assert centered.tolist() == [-1.0, 0.0, 1.0]
    assert centered.mean().item() == pytest.approx(0.0)
    assert normalized.std(unbiased=False).item() == pytest.approx(1.0)


def test_sampled_kl_is_nonnegative_and_zero_for_equal_policies():
    policy = torch.tensor([[-1.0, -2.0]])
    assert torch.equal(sampled_token_kl(policy, policy), torch.zeros_like(policy))
    estimate = sampled_token_kl(policy, torch.tensor([[-2.0, -1.0]]))
    assert (estimate >= 0).all()


def test_clipped_policy_loss_masks_prompt_and_adds_kl():
    policy = torch.tensor([[0.2, 100.0], [-0.2, -100.0]], requires_grad=True)
    old = torch.zeros_like(policy)
    mask = torch.tensor([[True, False], [True, False]])
    advantages = torch.tensor([1.0, -1.0])
    reference = torch.zeros_like(policy)

    loss, stats = clipped_policy_loss(
        policy,
        old,
        mask,
        advantages,
        clip_epsilon=0.1,
        reference_logps=reference,
        kl_beta=0.2,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert stats["kl"] >= 0
    assert policy.grad[:, 1].abs().sum().item() == 0.0


def test_online_config_rejects_single_sample_grpo_group():
    with pytest.raises(ValueError, match="group_size"):
        OnlineTrainingConfig(method="grpo", group_size=1)


@pytest.mark.parametrize("distributed_type", ["FSDP", "DEEPSPEED", "MEGATRON_LM"])
def test_online_training_rejects_parameter_sharded_engines(distributed_type):
    with pytest.raises(RuntimeError, match="unsharded data parallelism"):
        validate_online_distributed_type(distributed_type)


@pytest.mark.parametrize("distributed_type", ["NO", "MULTI_GPU", "MULTI_CPU"])
def test_online_training_accepts_single_process_and_ddp(distributed_type):
    validate_online_distributed_type(distributed_type)
