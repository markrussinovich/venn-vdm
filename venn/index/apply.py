"""Prepare a Choice question once and score many state vectors against it."""

import torch
from torch.nn import functional as F

from venn.index.prewarm import prewarm


class PreparedQuestion:
    """A question whose centered options and abstention-head terms are precomputed for fast streaming.

    ``model`` is a ``CosineAbstention`` or ``ContextAbstention`` head; ``question`` (1-D) and ``options``
    (2-D) are tensors on the same device. ``temperature`` is the default softmax temperature for ``stream``.
    """
    def __init__(self, model, question, options, temperature=1.0):
        mask = torch.ones(1, len(options), dtype=torch.bool, device=options.device)
        q, o, _ = prewarm(question[None], options[None], mask)
        self.options = o[0]
        self.model = model
        self.calibration_temperature = temperature
        width = question.shape[0]
        first = model.none[0]
        self.state_weight = first.weight[:, :width]
        self.question_bias = F.linear(q[0], first.weight[:, width:2 * width], first.bias)
        self.context_weight = first.weight[:, 2 * width:]

    @torch.inference_mode()
    def logits(self, states):
        """Return ``(rows, options + 1)`` logits for unit-norm ``states``; the last column is "none"."""
        cosine = states @ self.options.T
        option_logits = cosine / self.model.temperature
        hidden = F.linear(states, self.state_weight) + self.question_bias
        if self.context_weight.shape[1]:
            context = torch.stack([cosine.max(1).values,
                                   torch.full_like(cosine[:, 0], float(torch.tensor(len(self.options)).log()))], 1)
            hidden = hidden + F.linear(context, self.context_weight)
        hidden = self.model.none[1](hidden)
        none = self.model.none[2](hidden) + self.model.none_intercept
        if self.context_weight.shape[1]:
            none = none + context[:, 1:]
        return torch.cat([option_logits, none], dim=-1)

    @torch.inference_mode()
    def stream(self, vectors, batch_size=65536, temperature=None):
        """Yield ``(offset, probabilities)`` for each batch of ``vectors``.

        Rows are unit-normalized and softmaxed at ``temperature`` (default: the calibration temperature).
        """
        device = self.options.device
        temperature = self.calibration_temperature if temperature is None else temperature
        for offset in range(0, len(vectors), batch_size):
            states = torch.as_tensor(vectors[offset:offset + batch_size]).to(device).float()
            states = F.normalize(states, dim=-1)
            probabilities = (self.logits(states) / temperature).softmax(-1)
            yield offset, probabilities
