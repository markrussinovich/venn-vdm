"""Center a question and its options on the option centroid."""

from torch.nn import functional as F


def prewarm(question, options, mask):
    """Subtract the masked option centroid from ``question`` and ``options`` and unit-normalize both.

    Returns ``(question, options, centroid)``; masked-out options are zeroed.
    """
    weights = mask.to(options.dtype).unsqueeze(-1)
    count = weights.sum(1).clamp_min(1)
    centroid = (options * weights).sum(1) / count
    centered = F.normalize(options - centroid[:, None], dim=-1)
    centered = centered.masked_fill(~mask[:, :, None], 0)
    q = F.normalize(question - centroid, dim=-1)
    return q, centered, centroid
