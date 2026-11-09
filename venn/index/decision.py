"""Nearest-reference labels and leave-one-out centroid distances."""

import numpy as np


def nearest_decision(scores, reference_labels):
    """Pick each query's label from its best-scoring reference.

    ``scores`` is ``(queries, references)``; returns ``(predicted_labels, margin)`` where ``margin`` is the gap
    between the best and second-best per-label scores. Needs at least two distinct reference labels.
    """
    labels = np.unique(reference_labels)
    by_label = np.column_stack([scores[:, reference_labels == label].max(1) for label in labels])
    order = np.argsort(by_label, axis=1)
    predicted = labels[order[:, -1]]
    margin = by_label[np.arange(len(scores)), order[:, -1]] - by_label[np.arange(len(scores)), order[:, -2]]
    return predicted, margin


def leave_one_out_distances(values):
    """Return each row's squared distance to the centroid of the other rows along the second-to-last axis."""
    centroid = (values.sum(axis=-2, keepdims=True) - values) / (values.shape[-2] - 1)
    return np.sum((values - centroid) ** 2, axis=-1)

