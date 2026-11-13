"""Optional Muon+AdamW experiments; preserve AdamW as the default."""

import math

import torch


def build_outcome_optimizers(head, *, optimizer="adamw", lr=1e-3, weight_decay=.01, muon_lr=None):
    """Return optimizers for a small bilinear outcome head.

    Muon is assigned only matrix parameters; other parameters retain AdamW.
    This is an experimental optimizer choice, not a tested performance gain.
    """
    if optimizer == "adamw":
        if muon_lr is not None:
            raise ValueError("muon_lr is only valid with optimizer='muon'")
        return (torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay),)
    if optimizer != "muon":
        raise ValueError("optimizer must be 'adamw' or 'muon'")
    selected_lr = lr if muon_lr is None else muon_lr
    if (isinstance(selected_lr, bool) or not isinstance(selected_lr, (int, float))
            or not math.isfinite(selected_lr) or selected_lr <= 0):
        raise ValueError("muon_lr must be finite and positive")
    muon_type = getattr(torch.optim, "Muon", None)
    if muon_type is None:
        raise ValueError("Muon requires a PyTorch build exposing torch.optim.Muon")
    matrices = [parameter for parameter in head.parameters() if parameter.ndim == 2]
    other = [parameter for parameter in head.parameters() if parameter.ndim != 2]
    if not matrices:
        raise ValueError("Muon needs at least one matrix parameter")
    opts = [muon_type(matrices, lr=selected_lr, weight_decay=weight_decay)]
    if other:
        opts.append(torch.optim.AdamW(other, lr=lr, weight_decay=weight_decay))
    return tuple(opts)
