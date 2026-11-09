"""Cosine decision head with a learned abstention ("none") logit."""

import torch
from torch import nn

from venn.index.prewarm import prewarm


class CosineAbstention(nn.Module):
    """Scores options by temperature-scaled cosine similarity and appends a learned "none" logit.

    ``temperature`` divides the option cosines; ``none_intercept`` is the abstention bias, stored divided
    by ``temperature`` so it lives on the same scale as the option logits.
    """
    def __init__(self, temperature, none_intercept, dimensions=768, hidden=512):
        super().__init__()
        self.register_buffer("temperature", torch.tensor(float(temperature)))
        self.register_buffer("none_intercept", torch.tensor(float(none_intercept / temperature)))
        self.none = nn.Sequential(
            nn.Linear(2 * dimensions, hidden), nn.GELU(), nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.none[-1].weight)
        nn.init.zeros_(self.none[-1].bias)

    def forward(self, state, question, options, mask):
        """Return logits of shape ``(batch, options + 1)``; the last column is "none".

        Options are centered on their masked centroid before scoring and masked-out positions get ``-1e4``.
        """
        q, options, _ = prewarm(question, options, mask)
        logits = ((state[:, None] * options).sum(-1) / self.temperature).masked_fill(~mask, -1e4)
        none = self.none(torch.cat([state, q], dim=-1)) + self.none_intercept
        return torch.cat([logits, none], dim=-1)
