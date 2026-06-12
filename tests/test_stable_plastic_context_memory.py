import torch

from src.stable_plastic_context_memory import (
    ContextMemoryConfig,
    StablePlasticContextMemory,
)


def random_probs(batch: int, contexts: int) -> torch.Tensor:
    x = torch.rand(batch, contexts)
    return x / x.sum(dim=-1, keepdim=True)


def test_legacy_exact_equivalence():
    q_single = random_probs(7, 10)
    q_proto = random_probs(7, 10)
    weight = 0.05
    module = StablePlasticContextMemory(
        ContextMemoryConfig(mode="legacy", legacy_single_weight=weight)
    )
    actual, audit = module(q_single=q_single, q_legacy_prototype=q_proto)
    expected = weight * q_single + (1.0 - weight) * q_proto
    assert torch.allclose(actual, expected, atol=1e-7, rtol=1e-6)
    assert audit["mode"] == "legacy"


def test_arbitrary_context_counts_and_normalization():
    for contexts in (3, 5, 10, 17):
        module = StablePlasticContextMemory(ContextMemoryConfig(mode="stable_plastic"))
        posterior, _ = module(
            q_single=random_probs(4, contexts),
            q_stable=random_probs(4, contexts),
            q_plastic=random_probs(4, contexts),
        )
        assert posterior.shape == (4, contexts)
        assert torch.allclose(posterior.sum(dim=-1), torch.ones(4), atol=1e-6)


def test_adaptive_mode_returns_audit_weights():
    module = StablePlasticContextMemory(ContextMemoryConfig(mode="adaptive"))
    posterior, audit = module(
        q_single=random_probs(6, 8),
        q_stable=random_probs(6, 8),
        q_plastic=random_probs(6, 8),
    )
    weights = audit["fusion_weights_single_plastic_stable"]
    assert posterior.shape == (6, 8)
    assert weights.shape == (6, 3)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(6), atol=1e-6)


def test_no_dataset_specific_configuration_field():
    fields = ContextMemoryConfig.__dataclass_fields__
    assert "dataset" not in fields
    assert "dataset_name" not in fields
