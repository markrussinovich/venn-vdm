"""Temperature scaling for post-hoc probability calibration."""

import torch
from torch.nn import functional as F


def fit_temperature(logits, targets):
    """Fit a softmax temperature to ``logits`` and integer ``targets`` by minimizing cross-entropy with L-BFGS.

    Returns the temperature as a float clamped to ``[e^-5, e^5]``.
    """
    values = torch.as_tensor(logits, dtype=torch.float64)
    labels = torch.as_tensor(targets, dtype=torch.long)
    log_t = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], max_iter=100, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(values / log_t.clamp(-5, 5).exp(), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_t.detach().clamp(-5, 5).exp())


def probabilities(logits, temperature):
    """Return softmax probabilities of ``logits / temperature`` as a NumPy array."""
    return F.softmax(torch.as_tensor(logits) / temperature, dim=-1).numpy()

