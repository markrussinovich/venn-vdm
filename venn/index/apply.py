"""Prepare a Choice question once and score many state vectors against it."""

import math
from numbers import Real

import torch
from torch.nn import functional as F

from venn.index.prewarm import prewarm


def _temperature(value):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
        raise ValueError("Calibration temperature must be finite and positive")
    return float(value)


class PreparedQuestion:
    """A question whose centered options and abstention-head terms are precomputed for fast streaming.

    ``model`` is a ``CosineAbstention`` or ``ContextAbstention`` head; ``question`` (1-D) and ``options``
    (2-D) are tensors on the same device. ``temperature`` is the default softmax temperature for ``stream``.
    """
    def __init__(self, model, question, options, temperature=1.0):
        temperature = _temperature(temperature)
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
        temperature = _temperature(self.calibration_temperature if temperature is None else temperature)
        for offset in range(0, len(vectors), batch_size):
            states = torch.as_tensor(vectors[offset:offset + batch_size]).to(device).float()
            states = F.normalize(states, dim=-1)
            probabilities = (self.logits(states) / temperature).softmax(-1)
            yield offset, probabilities


class PreparedQuestionBatch:
    """Optionally score several Choice questions sharing one frozen abstention head.

    Returns one logit matrix per question, retaining its option order and trailing
    ``none`` column. Ragged option sets are bucketed and the computation is tiled
    over rows, avoiding a dense (rows, all questions, largest option count) tensor.

    ``max_workspace_bytes`` is a *target* for temporary tensor sizes, not a peak
    RSS guarantee (backend kernels, caching, and returned logits also use memory).
    The input head and prewarmed questions must remain unchanged during use.
    """

    def __init__(self, questions, max_workspace_bytes=8 * 1024 * 1024):
        questions = tuple(questions)
        if not questions or not all(isinstance(q, PreparedQuestion) for q in questions):
            raise ValueError("Provide one or more PreparedQuestion instances")
        if type(max_workspace_bytes) is not int or max_workspace_bytes <= 0:
            raise ValueError("max_workspace_bytes must be a positive integer")
        first = questions[0]
        model = first.model
        device, dtype = first.options.device, first.options.dtype
        width = first.options.shape[-1]
        if not all(q.model is model and q.options.ndim == 2 and q.options.device == device
                   and q.options.dtype == dtype and q.options.shape[-1] == width
                   and len(q.options) > 0 and q.question_bias.device == device
                   and q.question_bias.dtype == dtype for q in questions):
            raise ValueError("Questions must have a shared model, device, dtype and vector width, with nonempty options")
        if first.context_weight.shape[1] not in (0, 2):
            raise ValueError("Unsupported abstention context width")

        self.model = model
        self.width = width
        self.lengths = tuple(len(q.options) for q in questions)
        self.calibration_temperatures = tuple(_temperature(q.calibration_temperature) for q in questions)
        self.max_workspace_bytes = max_workspace_bytes
        self.state_weight = first.state_weight
        self.context_weight = first.context_weight
        self._device = device
        self._dtype = dtype

        # Group similarly sized option sets: at most a factor-two padding within
        # each bucket. Bound the size of each precomputed group and preserve an
        # explicit map to the caller's original question order.
        buckets = {}
        for i, k in enumerate(self.lengths):
            bucket = 1 << (k - 1).bit_length()
            buckets.setdefault(bucket, []).append(i)
        groups = []
        element_bytes = first.options.element_size()
        for bucket, indices in buckets.items():
            max_group = max(1, min(32, max_workspace_bytes // (4 * element_bytes * width * bucket)))
            for offset in range(0, len(indices), max_group):
                ids = tuple(indices[offset:offset + max_group])
                count = max(self.lengths[i] for i in ids)
                options = torch.stack([F.pad(questions[i].options.detach(), (0, 0, 0, count - self.lengths[i]))
                                       for i in ids])
                mask = torch.arange(count, device=device)[None, :] < torch.tensor(
                    [self.lengths[i] for i in ids], device=device)[:, None]
                biases = torch.stack([questions[i].question_bias.detach() for i in ids])
                # Match PreparedQuestion's float32 count logarithm construction.
                logs = torch.tensor([float(torch.tensor(self.lengths[i]).log()) for i in ids],
                                    dtype=dtype, device=device)
                groups.append((ids, options, mask, biases, logs))
        self._groups = tuple(groups)

        # Approximate upper bound on temporary activation elements per row.
        # The final result itself consumes rows * sum(k_i + 1) elements.
        hidden = self.state_weight.shape[0]
        output_elements = sum(k + 1 for k in self.lengths)
        largest = max(len(ids) * (4 * hidden + 3 * opts.shape[1]) + hidden + width
                      for ids, opts, _, _, _ in self._groups)
        elements_per_row = largest + output_elements
        self._row_cap = max(1, max_workspace_bytes // (element_bytes * elements_per_row))

    @torch.inference_mode()
    def logits(self, states):
        """Score unit-normalized ``(rows, width)`` states; retain per-question widths.

        This returns every question's logits for all rows; output storage is
        necessarily proportional to ``rows * sum(k_i + 1)`` even when internal
        calculations are chunked.
        """
        if (not torch.is_tensor(states) or states.ndim != 2 or states.shape[1] != self.width or states.device != self._device
                or states.dtype != self._dtype):
            raise ValueError("States must have the shared vector width, device and dtype")
        if len(states) == 0:
            return tuple(states.new_empty((0, k + 1)) for k in self.lengths)
        parts = [[] for _ in self.lengths]
        for row in states.split(self._row_cap):
            state_projection = F.linear(row, self.state_weight)
            for ids, options, mask, biases, logs in self._groups:
                cosine = torch.einsum("bd,qkd->bqk", row, options)
                option_logits = cosine / self.model.temperature
                hidden = state_projection[:, None, :] + biases[None, :, :]
                if self.context_weight.shape[1]:
                    context = torch.stack((cosine.masked_fill(~mask[None], float("-inf")).max(-1).values,
                                           logs[None, :].expand(len(row), -1)), dim=-1)
                    hidden = hidden + F.linear(context, self.context_weight)
                none = self.model.none[2](self.model.none[1](hidden)) + self.model.none_intercept
                if self.context_weight.shape[1]:
                    none = none + context[:, :, 1:2]
                for j, i in enumerate(ids):
                    parts[i].append(torch.cat((option_logits[:, j, :self.lengths[i]],
                                               none[:, j, :]), dim=-1))
        return tuple(torch.cat(chunks, 0) for chunks in parts)

    @torch.inference_mode()
    def stream(self, vectors, batch_size=65536, temperature=None):
        """Yield ``(offset, tuple_of_probabilities)`` in the original question order.

        Each normalized input chunk is bounded by the workspace-aware row cap.
        ``temperature`` may override all per-question calibration temperatures.
        """
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        device = self._device
        cap = min(batch_size, self._row_cap)
        temperatures = self.calibration_temperatures if temperature is None else (_temperature(temperature),) * len(self.lengths)
        for offset in range(0, len(vectors), cap):
            states = torch.as_tensor(vectors[offset:offset + cap]).to(device=device, dtype=self._dtype)
            states = F.normalize(states, dim=-1)
            yield offset, tuple((logits / temp).softmax(-1)
                                for logits, temp in zip(self.logits(states), temperatures))
