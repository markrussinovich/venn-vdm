"""Linear choice families: multinomial logistic regression from 768-d vectors to the registered labels.

The penalty C is chosen by K-fold cross-validation inside the train rows (grouped when groups are given) on the
mean out-of-fold log loss. The objective matches scikit-learn's ``LogisticRegression``: ``0.5 * ||W||^2 + C *
sum_i w_i * cross_entropy_i`` with an unpenalised intercept, minimised in float64 with L-BFGS. A temperature is
fitted on out-of-fold logits (or the caller's selection slice); report metrics come from the caller's report slice.
"""
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from venn.heads.calibration import fit_temperature, probabilities
from venn.heads.family import cv_folds, ece_interval, row_weights

CS = (0.01, 0.1, 1.0, 10.0, 100.0)


def fit_multinomial(X, y, classes, C, weights=None, max_iter=500, device="cpu"):
    """Return ``(W, b)`` minimising ``0.5 ||W||^2 + C * sum_i w_i CE(x_i W' + b, y_i)`` (float64, L-BFGS, on ``device``)."""
    X = torch.as_tensor(np.asarray(X, dtype=np.float64), device=device)
    y = torch.as_tensor(np.asarray(y, dtype=np.int64), device=device)
    w = torch.as_tensor(np.ones(len(y)) if weights is None else np.asarray(weights, dtype=np.float64), device=device)
    W = torch.zeros((classes, X.shape[1]), dtype=torch.float64, requires_grad=True, device=device)
    b = torch.zeros(classes, dtype=torch.float64, requires_grad=True, device=device)
    optimizer = torch.optim.LBFGS([W, b], lr=1, max_iter=max_iter, tolerance_grad=1e-9, tolerance_change=1e-12,
                                  history_size=20, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = 0.5 * (W * W).sum() + C * (F.cross_entropy(X @ W.T + b, y, reduction="none") * w).sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    return W.detach().cpu().numpy(), b.detach().cpu().numpy()


def logits(X, W, b):
    return np.asarray(X, dtype=np.float64) @ W.T + b


class LinearChoiceHead(nn.Module):
    """Serving module with the family-head call signature: maps each requested option vector to its registered label
    and returns that label's linear logit, plus a "none" column that never wins.

    Full-table requests preserve row order, even with identical vectors; the server binds subsets by label.
    """

    def __init__(self, W, b, option_vectors):
        super().__init__()
        self.register_buffer("W", torch.as_tensor(np.asarray(W, dtype=np.float32)))
        self.register_buffer("b", torch.as_tensor(np.asarray(b, dtype=np.float32)))
        self.register_buffer("O", torch.as_tensor(np.asarray(option_vectors, dtype=np.float32)))

    def forward(self, state, question, options, mask, universal=None):
        if state.ndim != 2:
            raise ValueError("Linear families take single-vector states")
        if options.shape[1:] == self.O.shape and torch.equal(options, self.O.expand_as(options)):
            index = torch.arange(len(self.O), device=options.device).expand(options.shape[:2])
        else:
            index = torch.cdist(options.reshape(-1, options.shape[-1]).float(), self.O).argmin(1).reshape(options.shape[:2])
        out = (state.float() @ self.W.T + self.b).gather(1, index)
        out = out.masked_fill(~mask, -1e4)
        return torch.cat([out, torch.full((len(state), 1), -1e4, dtype=out.dtype, device=out.device)], 1)


def fit_linear_family(vectors, labels, option_vectors, *, seed=0, partitions=None, groups=None, folds=5, class_weight=None, Cs=CS, device="cpu"):
    """Fit a linear choice family (on ``device``; the result is a CPU module). Returns ``(module, info, parts, arrays)``;
    ``arrays`` holds W, b for persistence."""
    start = time.perf_counter()
    vectors = np.asarray(vectors, dtype=np.float64)
    if vectors.ndim != 2:
        raise ValueError("Linear families take single-vector states")
    labels = np.asarray(labels)
    classes = len(option_vectors)
    if partitions is None:
        parts = [np.arange(len(labels), dtype=np.int64), np.array([], dtype=np.int64), np.array([], dtype=np.int64)]
    else:
        parts = [np.asarray(part, dtype=np.int64) for part in partitions]
    train, selection, report_rows = parts
    if np.bincount(labels[train], minlength=classes).min() < 2:
        raise ValueError("Every label needs at least two train rows")
    weights = row_weights(labels, class_weight)
    train_groups = None if groups is None else np.asarray(groups)[train]
    k = int(min(folds, len(np.unique(train_groups)) if train_groups is not None else folds))
    if k < 2:
        raise ValueError("Cross-validation needs at least two folds")
    fold_rows = [train[f] for f in cv_folds(labels[train], k, seed, train_groups)]
    oof = {C: np.zeros((len(labels), classes)) for C in Cs}
    for rows in fold_rows:
        fit_rows = np.setdiff1d(train, rows)
        for C in Cs:
            W, b = fit_multinomial(vectors[fit_rows], labels[fit_rows], classes, C, weights[fit_rows], device=device)
            oof[C][rows] = logits(vectors[rows], W, b)
    def cv_loss(C):
        per = F.cross_entropy(torch.as_tensor(oof[C][train]), torch.as_tensor(labels[train]), reduction="none").numpy()
        return float((per * weights[train]).sum() / weights[train].sum())
    table = {C: cv_loss(C) for C in Cs}
    best = min(Cs, key=lambda C: (table[C], C))
    W, b = fit_multinomial(vectors[train], labels[train], classes, best, weights[train], device=device)
    if len(selection):
        t = fit_temperature(logits(vectors[selection], W, b).astype(np.float32), labels[selection])
        temperature_source = "selection slice"
    else:
        t = fit_temperature(oof[best][train].astype(np.float32), labels[train])
        temperature_source = "out-of-fold logits"
    cv_p = probabilities(oof[best][train].astype(np.float32), t)
    cv_ece, cv_interval = ece_interval(cv_p, labels[train], seed, groups=train_groups)
    info = {"examples": len(labels), "split_counts": [len(p) for p in parts], "form": "linear", "temperature": t,
            "temperature_source": temperature_source,
            "settings": {"C": best, "chosen_by": f"{k}-fold cross-validation log loss" + (" grouped by group_id" if groups is not None else "")},
            "cv_log_loss_by_C": {str(C): v for C, v in table.items()}, "cv_accuracy": float((cv_p.argmax(1) == labels[train]).mean()),
            "cv_ece": cv_ece, "cv_ece_ci95": cv_interval,
            "class_weight": class_weight if class_weight in (None, "balanced") else {str(c): w for c, w in class_weight.items()},
            "scorer": "linear_multinomial", "k2_fallback": False, "epochs": None, "early_stopped": False, "seed": seed}
    if len(report_rows):
        p = probabilities(logits(vectors[report_rows], W, b).astype(np.float32), t)
        ece, interval = ece_interval(p, labels[report_rows], seed, groups=None if groups is None else np.asarray(groups)[report_rows])
        required = max(0, 500 - len(report_rows))
        info.update({"report_accuracy": float((p.argmax(1) == labels[report_rows]).mean()), "report_ece": ece, "report_ece_ci95": interval,
                     "calibrated": not required and interval[1] <= .05, "additional_report_items_needed": required})
    else:
        info.update({"report_accuracy": None, "report_ece": None, "report_ece_ci95": None, "calibrated": False, "additional_report_items_needed": 500})
    info["wall_seconds"] = time.perf_counter() - start
    return LinearChoiceHead(W, b, option_vectors).eval(), info, parts, {"W": W, "b": b}
