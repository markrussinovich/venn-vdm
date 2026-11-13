"""CPU-only structural checks for optional hybrid Muon optimization."""

import pytest
import torch

from venn.heads.optimizers import build_outcome_optimizers


class TinyBilinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.left = torch.nn.Linear(4, 3, bias=False)
        self.right = torch.nn.Linear(2, 3, bias=False)
        self.bias = torch.nn.Parameter(torch.zeros(()))

    def forward(self, x, options):
        return self.left(x) @ self.right(options).T + self.bias


def test_default_is_exactly_one_adamw_over_all_parameters():
    head = TinyBilinear()
    opts = build_outcome_optimizers(head)
    assert len(opts) == 1 and isinstance(opts[0], torch.optim.AdamW)
    assert {id(p) for group in opts[0].param_groups for p in group["params"]} == {
        id(p) for p in head.parameters()}


def test_muon_selects_only_matrices_and_adamw_handles_bias():
    head = TinyBilinear()
    opts = build_outcome_optimizers(head, optimizer="muon", muon_lr=.02)
    assert len(opts) == 2 and isinstance(opts[0], torch.optim.Muon)
    assert isinstance(opts[1], torch.optim.AdamW)
    assert {id(p) for group in opts[0].param_groups for p in group["params"]} == {
        id(head.left.weight), id(head.right.weight)}
    assert [p for group in opts[1].param_groups for p in group["params"]] == [head.bias]
    assert opts[0].param_groups[0]["lr"] == .02
    assert opts[1].param_groups[0]["lr"] == .001


def test_hybrid_updates_and_finite_parameters_cpu():
    torch.manual_seed(12)
    head = TinyBilinear()
    original = {name: p.detach().clone() for name, p in head.named_parameters()}
    optimizers = build_outcome_optimizers(head, optimizer="muon", lr=.001, muon_lr=.01)
    x = torch.randn(5, 4)
    options = torch.randn(3, 2)
    for _ in range(2):
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        objective = head(x, options).square().mean() + head.bias
        objective.backward()
        for opt in optimizers:
            opt.step()
    assert all(torch.isfinite(p).all() for p in head.parameters())
    assert all(not torch.equal(original[name], p) for name, p in head.named_parameters())


@pytest.mark.parametrize("kwargs", [
    {"optimizer": "fake"},
    {"optimizer": "adamw", "muon_lr": .01},
    {"optimizer": "muon", "muon_lr": 0.},
    {"optimizer": "muon", "muon_lr": float("nan")},
    {"optimizer": "muon", "muon_lr": True},
])
def test_invalid_optimizer_configs_fail_closed(kwargs):
    with pytest.raises(ValueError):
        build_outcome_optimizers(TinyBilinear(), **kwargs)


def test_adamw_fallback_for_biasless_matrix_only_module():
    # A matrix-only module has no need to construct an empty AdamW optimizer.
    head = torch.nn.Linear(2, 3, bias=False)
    opts = build_outcome_optimizers(head, optimizer="muon", muon_lr=.01)
    assert len(opts) == 1 and isinstance(opts[0], torch.optim.Muon)


def test_opt_in_muon_trainer_keeps_serializable_head_and_hyperparameters():
    import numpy as np
    from venn.heads.outcome import fit_outcome_head

    rng = np.random.default_rng(31)
    x = rng.normal(size=(30, 4)).astype(np.float32)
    labels = (np.arange(30) % 2).astype(np.int64)
    split = np.repeat([0, 1, 2], 10)
    records = {"x": x, "action": labels, "split": split}
    options = np.eye(2, dtype=np.float32)
    muon, report = fit_outcome_head(records, labels, options, rank=2, seed=3,
                                    max_epochs=2, batch=5, optimizer="muon", muon_lr=.01)
    assert torch.isfinite(muon.left.weight).all() and torch.isfinite(muon.right.weight).all()
    assert set(muon.state_dict()) == {"left.weight", "right.weight", "bias"}
    assert report["hyperparameters"]["optimizer"] == "muon"
    assert report["hyperparameters"]["muon_lr"] == .01
    adam, report_adam = fit_outcome_head(records, labels, options, rank=2, seed=3,
                                         max_epochs=2, batch=5)
    assert "optimizer" not in report_adam["hyperparameters"]
    assert set(adam.state_dict()) == set(muon.state_dict())
