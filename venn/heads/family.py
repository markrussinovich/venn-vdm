"""Per-family decision heads: episode sampling, training, calibration and reporting."""

import threading
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from venn.heads.calibration import fit_temperature, probabilities
from venn.index.prewarm import prewarm


def debiased_ece(p, targets, bins=15):
    """Return the debiased expected calibration error of probabilities ``p`` against ``targets``.

    Confidence is split into ``bins`` equal-width bins and each bin's sampling variance is subtracted.
    """
    confidence = p.max(1)
    correct = (p.argmax(1) == targets).astype(float)
    groups = np.minimum((confidence * bins).astype(int), bins - 1)
    result = 0.0
    for group in range(bins):
        selected = groups == group
        count = int(selected.sum())
        if count:
            accuracy = correct[selected].mean()
            variance = accuracy * (1 - accuracy) / max(count - 1, 1)
            result += count / len(p) * np.sqrt(max(
                (accuracy - confidence[selected].mean()) ** 2 - variance, 0
            ))
    return float(result)


def ece_interval(p, targets, seed=0, repeats=500, groups=None):
    """Return ``(ece, [low, high])``: the debiased ECE and its 95% bootstrap interval.

    Resamples rows ``repeats`` times, or whole groups when ``groups`` labels each row.
    """
    rng = np.random.default_rng(seed)
    values = [debiased_ece(p, targets)]
    positions = None
    if groups is not None:
        positions = [np.flatnonzero(groups == group) for group in np.unique(groups)]
    for _ in range(repeats):
        selected = (rng.integers(len(p), size=len(p)) if positions is None else
                    np.concatenate([positions[i] for i in rng.integers(len(positions), size=len(positions))]))
        values.append(debiased_ece(p[selected], targets[selected]))
    return values[0], np.quantile(values[1:], [.025, .975]).tolist()


def split_labels(labels, seed):
    """Split row indices into train, validation and report parts (about 60/20/20) within each label.

    Returns three ``int64`` index arrays. Raises ``ValueError`` if any label has fewer than five rows.
    """
    rng = np.random.default_rng(seed)
    parts = [[], [], []]
    for label in np.unique(labels):
        indices = rng.permutation(np.flatnonzero(labels == label))
        if len(indices) < 5:
            raise ValueError("Each registered label needs at least five distinct examples")
        train = max(1, int(len(indices) * .6))
        stop = max(1, int(len(indices) * .2))
        for part, values in zip(parts, (indices[:train], indices[train:train + stop],
                                       indices[train + stop:])):
            part.extend(values.tolist())
    return [np.array(part, dtype=np.int64) for part in parts]


def split_groups(labels, groups, seed):
    """Split like ``split_labels`` but keep every row of a group in the same part.

    Raises ``ValueError`` if a group carries more than one label or a label has fewer than five groups.
    """
    unique, inverse = np.unique(groups, return_inverse=True)
    group_labels = []
    for index in range(len(unique)):
        values = np.unique(labels[inverse == index])
        if len(values) != 1:
            raise ValueError("Each example group must have one label")
        group_labels.append(values[0])
    parts = split_labels(np.asarray(group_labels), seed)
    return [np.flatnonzero(np.isin(inverse, part)) for part in parts]


def episodes(labels, indices, classes, count, rng, k=None, none_rate=.3, max_k=None):
    """Sample ``count`` multiple-choice episodes from the rows in ``indices``.

    Each episode draws ``k`` distinct option classes (or 2..``max_k`` at random) and, with probability
    ``none_rate`` when at least three classes exist, omits the true class so the target is "none".
    Returns ``rows``, padded ``options``, ``mask`` and ``target`` arrays; raises ``ValueError`` for an
    infeasible option count.
    """
    rows = rng.choice(indices, count, replace=count > len(indices))
    maximum = (max_k or len(classes)) if k is None else k
    if not 2 <= maximum <= len(classes):
        raise ValueError("Infeasible distinct-option count")
    options = np.empty((count, maximum), dtype=np.int64)
    mask = np.zeros((count, maximum), dtype=bool)
    targets = np.empty(count, dtype=np.int64)
    for i, row in enumerate(rows):
        size = int(rng.integers(2, maximum + 1)) if k is None else k
        true = int(labels[row])
        others = classes[classes != true]
        absent = len(classes) >= 3 and rng.random() < none_rate
        if absent and size == len(classes):
            size -= 1
        selected = rng.choice(others, size if absent else size - 1, replace=False)
        if not absent:
            selected = np.append(selected, true)
        rng.shuffle(selected)
        options[i, :size] = selected
        options[i, size:] = selected[0]
        mask[i, :size] = True
        targets[i] = maximum if absent else int(np.flatnonzero(selected == true)[0])
    return {"rows": rows, "options": options, "mask": mask, "target": targets}


class ContextAbstention(nn.Module):
    """Cosine decision head whose "none" logit also sees the best option cosine and the option count.

    ``temperature`` and ``none_intercept`` are learnable; options are centered on their centroid while
    ``center_options`` is set.
    """
    def __init__(self, temperature, none_intercept, dimensions=768, hidden=128):
        super().__init__()
        self.log_temperature = nn.Parameter(torch.tensor(float(np.log(temperature))))
        self.none_intercept = nn.Parameter(torch.tensor(float(none_intercept)))
        self.center_options = True
        self.none = nn.Sequential(nn.Linear(2 * dimensions + 2, hidden),
                                  nn.GELU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.none[-1].weight)
        nn.init.zeros_(self.none[-1].bias)

    @property
    def temperature(self):
        """Learned softmax temperature, clamped to ``[e^-7, e^3]``."""
        return self.log_temperature.clamp(-7, 3).exp()

    def forward(self, state, question, options, mask):
        """Return logits of shape ``(batch, options + 1)``; the last column is "none"."""
        if self.center_options:
            question, options, _ = prewarm(question, options, mask)
        cosine = (state[:, None] * options).sum(-1)
        logits = (cosine / self.temperature).masked_fill(~mask, -1e4)
        context = torch.stack([cosine.masked_fill(~mask, -1).max(1).values,
                               mask.sum(1).float().log()], 1)
        none = self.none(torch.cat([state, question, context], 1))
        none = none + self.none_intercept + mask.sum(1).float().log()[:, None]
        return torch.cat([logits, none], 1)


class FamilyScorer(nn.Module):
    """Residual scorer trained per family on top of a universal abstention head.

    Uses a rank-16 bilinear residual below 1000 ``examples`` and a small MLP otherwise; ``fields=True``
    adds attention pooling over multi-vector states.
    """
    def __init__(self, examples, dimensions=768, rank=16, hidden=64, fields=False):
        super().__init__()
        self.low_rank = examples < 1000
        self.fields = fields
        if self.low_rank:
            self.left = nn.Linear(dimensions, rank, bias=False)
            self.right = nn.Linear(dimensions, rank, bias=False)
            nn.init.zeros_(self.right.weight)
        else:
            self.scorer = nn.Sequential(nn.Linear(4 * dimensions, hidden), nn.GELU(),
                                        nn.Linear(hidden, 1))
            nn.init.zeros_(self.scorer[-1].weight)
            nn.init.zeros_(self.scorer[-1].bias)
        self.none = nn.Sequential(nn.Linear(2 * dimensions + 2, hidden), nn.GELU(),
                                  nn.Linear(hidden, 1))
        nn.init.zeros_(self.none[-1].weight)
        nn.init.zeros_(self.none[-1].bias)
        if fields:
            self.query = nn.Linear(dimensions, dimensions, bias=False)
            nn.init.zeros_(self.query.weight)

    def pool(self, state, question):
        """Attention-pool ``(batch, fields, dim)`` states using the question as query, then unit-normalize.

        2-D states are only normalized.
        """
        if state.ndim == 3:
            attention = (state * self.query(question)[:, None]).sum(-1)
            state = (state * attention.softmax(1)[:, :, None]).sum(1)
        return F.normalize(state, dim=-1)

    def forward(self, state, question, options, mask, universal):
        """Return the ``universal`` head logits plus learned option residuals and a "none" correction.

        Output shape is ``(batch, options + 1)``.
        """
        state = self.pool(state, question)
        base = universal(state, question, options, mask)
        if self.low_rank:
            residual = (self.left(state)[:, None] * self.right(options)).sum(-1)
        else:
            s, q = state[:, None], question[:, None]
            features = torch.cat([s * options, (s - options).abs(),
                                  (s * q).expand_as(options), options * q], -1)
            residual = self.scorer(features).squeeze(-1)
        context = torch.stack([(state[:, None] * options).sum(-1).masked_fill(~mask, -1).max(1).values,
                               mask.sum(1).float().log()], 1)
        none = self.none(torch.cat([state, question, context], 1))
        return torch.cat([(base[:, :-1] + residual).masked_fill(~mask, -1e4),
                          base[:, -1:] + none], 1)


class FamilyQuestion:
    """A registered family's head bound to one question and option set, for scoring many states.

    ``entry`` holds the family's ``head`` (``FamilyScorer``) and ``universal`` abstention head;
    ``question`` and ``options`` are array-likes.
    """
    def __init__(self, entry, question, options):
        self.entry = entry
        self.question, self.options = torch.tensor(question), torch.tensor(options)

    @torch.inference_mode()
    def logits(self, states):
        """Return ``(rows, options + 1)`` logits for a tensor of ``states``."""
        question = self.question.expand(len(states), -1)
        options = self.options.expand(len(states), -1, -1)
        mask = torch.ones(options.shape[:2], dtype=torch.bool)
        return self.entry["head"](states, question, options, mask, self.entry["universal"])

    @torch.inference_mode()
    def stream(self, vectors, batch_size=65536, temperature=1):
        """Yield ``(offset, probabilities)`` per batch of ``vectors``, scoring 128 rows at a time at ``temperature``."""
        for offset in range(0, len(vectors), batch_size):
            outputs = []
            for start in range(offset, min(offset + batch_size, len(vectors)), 128):
                states = torch.as_tensor(vectors[start:min(start + 128, offset + batch_size)]).float()
                outputs.append((self.logits(states) / temperature).softmax(-1))
            yield offset, torch.cat(outputs)


def predict(model, universal, vectors, option_vectors, question, e, device="cpu", batch=128):
    """Return logits for every episode in ``e`` as a NumPy array; uses ``universal`` alone when ``model`` is ``None``."""
    values = []
    with torch.inference_mode():
        for offset in range(0, len(e["rows"]), batch):
            selected = slice(offset, offset + batch)
            state = torch.as_tensor(vectors[e["rows"][selected]], device=device)
            q = torch.as_tensor(question, device=device).expand(len(state), -1)
            options = torch.as_tensor(option_vectors[e["options"][selected]], device=device)
            mask = torch.as_tensor(e["mask"][selected], device=device)
            output = universal(state, q, options, mask) if model is None else model(
                state, q, options, mask, universal)
            values.append(output.cpu().numpy())
    return np.concatenate(values)


WEIGHT_DECAYS = (0.001, 0.01, 0.1)
_INIT_LOCK = threading.Lock()  # model initialisation draws from torch's global RNG; nothing after it does (safe under concurrent registrations)


def cv_folds(labels, k, seed, groups=None):
    """Label-stratified fold assignment: rows (or whole groups) sorted by label with a seeded tiebreak, dealt round robin.

    Returns ``k`` index arrays. With ``groups`` every row of a group lands in the same fold.
    """
    rng = np.random.default_rng(seed)
    if groups is None:
        order = np.lexsort((rng.random(len(labels)), labels))
        return [np.sort(order[i::k]) for i in range(k)]
    unique, inverse = np.unique(groups, return_inverse=True)
    first_label = np.array([labels[np.flatnonzero(inverse == g)[0]] for g in range(len(unique))])
    order = np.lexsort((rng.random(len(unique)), first_label))
    return [np.sort(np.flatnonzero(np.isin(inverse, order[i::k]))) for i in range(k)]


def row_weights(labels, class_weight):
    """Per-row loss weights: ``None`` (all 1), ``"balanced"`` (n / (classes * count)) or a ``{class index: weight}`` dict."""
    if class_weight is None:
        return np.ones(len(labels))
    if class_weight == "balanced":
        classes, counts = np.unique(labels, return_counts=True)
        lookup = dict(zip(classes.tolist(), (len(labels) / (len(classes) * counts)).tolist()))
        return np.array([lookup[int(c)] for c in labels])
    return np.array([float(class_weight.get(int(c), 1.0)) for c in labels])


def _train(vectors, labels, weights, rows, option_vectors, question, universal, classes, seed, device, weight_decay, epochs,
           validation=None, val_weights=None, patience=None, k2=None):
    """Train a ``FamilyScorer`` on ``rows`` for up to ``epochs`` epochs.

    Returns ``(model, losses, logits, k2_correct)``: per-epoch validation loss and logits, and per-epoch correctness on
    the two-option ``k2`` episodes. With ``patience`` training stops once the validation loss has not improved for that
    many epochs; the remaining epochs repeat the last values.
    """
    with _INIT_LOCK:
        torch.manual_seed(seed)
        model = FamilyScorer(len(labels), vectors.shape[-1], fields=vectors.ndim == 3).to(device)
    rng = np.random.default_rng(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=weight_decay)
    losses, logits, k2_correct, best, bad = [], [], [], float("inf"), 0
    for _ in range(epochs):
        if patience is not None and bad >= patience:
            losses.append(losses[-1])
            logits.append(logits[-1])
            k2_correct.append(k2_correct[-1])
            continue
        training = episodes(labels, rows, classes, min(8192, max(2048, len(rows) * 2)), rng)
        model.train()
        for offset in range(0, len(training["rows"]), 128):
            selected = slice(offset, offset + 128)
            state = torch.as_tensor(vectors[training["rows"][selected]], device=device)
            q = torch.as_tensor(question, device=device).expand(len(state), -1)
            options = torch.as_tensor(option_vectors[training["options"][selected]], device=device)
            mask = torch.as_tensor(training["mask"][selected], device=device)
            target = torch.as_tensor(training["target"][selected], device=device)
            w = torch.as_tensor(weights[training["rows"][selected]], dtype=torch.float32, device=device)
            optimizer.zero_grad()
            loss = (F.cross_entropy(model(state, q, options, mask, universal), target, reduction="none") * w).sum() / w.sum()
            loss.backward()
            optimizer.step()
        if validation is not None:
            model.eval()
            out = predict(model, universal, vectors, option_vectors, question, validation, device)
            per = F.cross_entropy(torch.tensor(out), torch.tensor(validation["target"]), reduction="none").numpy()
            losses.append(float((per * val_weights).sum() / val_weights.sum()))
            logits.append(out)
            if k2 is not None:
                k2_correct.append(predict(model, universal, vectors, option_vectors, question, k2, device).argmax(1) == k2["target"])
            if losses[-1] < best - 1e-5:
                best, bad = losses[-1], 0
            else:
                bad += 1
    return model, losses, logits, k2_correct


def fit_family(vectors, labels, option_vectors, question, universal, seed=0,
               device="cpu", partitions=None, epochs=24, patience=4, groups=None, folds=5,
               class_weight=None, weight_decays=WEIGHT_DECAYS):
    """Train a ``FamilyScorer`` on every train row, with its settings chosen by cross-validation; calibrate and report.

    ``partitions`` is an optional caller split ``(train, selection, report)`` of row indices (selection and report may
    be empty); by default every row is a train row. Weight decay and the number of epochs are chosen by ``folds``-fold
    cross-validation inside the train rows (grouped by ``groups`` when given; each fold stops after ``patience``
    epochs without improvement): the setting with the lowest mean validation loss wins. The temperature is fitted on
    the selection slice if one is given, otherwise on out-of-fold logits at the chosen setting. Report metrics come
    from the report slice when given; otherwise they are cross-validated estimates and the family is not marked
    calibrated. ``class_weight`` is ``None``, ``"balanced"`` or ``{class index: weight}``.

    Returns ``(model, info, parts)``: the CPU model, the report dict and ``(train, selection, report)`` index arrays.
    """
    start = time.perf_counter()
    labels = np.asarray(labels)
    if partitions is None:
        parts = [np.arange(len(labels), dtype=np.int64), np.array([], dtype=np.int64), np.array([], dtype=np.int64)]
    else:
        parts = [np.asarray(part, dtype=np.int64) for part in partitions]
    train, selection, report_rows = parts
    classes = np.unique(labels)
    if np.bincount(labels[train], minlength=len(classes)).min() < 2:
        raise ValueError("Every label needs at least two train rows")
    weights = row_weights(labels, class_weight)
    universal = universal.to(device).eval()
    for parameter in universal.parameters():
        parameter.requires_grad_(False)
    train_groups = None if groups is None else np.asarray(groups)[train]
    k = int(min(folds, len(np.unique(train_groups)) if train_groups is not None else folds))
    if k < 2:
        raise ValueError("Cross-validation needs at least two folds")
    fold_rows = [train[f] for f in cv_folds(labels[train], k, seed, train_groups)]
    rng = np.random.default_rng(seed)
    fold_validation = [episodes(labels, rows, classes, len(rows), rng) for rows in fold_rows]
    fold_k2 = [episodes(labels, rows, classes, len(rows), rng, k=2) for rows in fold_rows]
    tasks = [(wd, i) for wd in weight_decays for i in range(len(fold_rows))]

    def run(task):
        wd, i = task
        return _train(vectors, labels, weights, np.setdiff1d(train, fold_rows[i]), option_vectors, question, universal, classes,
                      seed + i, device, wd, epochs, fold_validation[i], weights[fold_validation[i]["rows"]], patience, fold_k2[i])

    results = [run(task) for task in tasks]
    runs = {wd: [results[tasks.index((wd, i))] for i in range(len(fold_rows))] for wd in weight_decays}
    curves = {wd: np.mean([run[1] for run in fold_runs], 0) for wd, fold_runs in runs.items()}
    chosen_wd, chosen_epochs = min(((wd, int(np.argmin(c)) + 1) for wd, c in curves.items()), key=lambda x: (curves[x[0]][x[1] - 1], x[0]))
    chosen = runs[chosen_wd]
    oof_logits = np.concatenate([run[2][chosen_epochs - 1] for run in chosen])
    oof_targets = np.concatenate([v["target"] for v in fold_validation])
    oof_rows = np.concatenate([v["rows"] for v in fold_validation])
    model = _train(vectors, labels, weights, train, option_vectors, question, universal, classes, seed, device, chosen_wd, chosen_epochs)[0]
    model.eval()
    if len(selection):
        sel = episodes(labels, selection, classes, len(selection), rng)
        t = fit_temperature(predict(model, universal, vectors, option_vectors, question, sel, device), sel["target"])
        temperature_source = "selection slice"
    else:
        t = fit_temperature(oof_logits, oof_targets)
        temperature_source = "out-of-fold logits"
    cv_p = probabilities(oof_logits, t)
    cv_ece, cv_interval = ece_interval(cv_p, oof_targets, seed, groups=None if groups is None else np.asarray(groups)[oof_rows])
    info = {"examples": len(labels), "split_counts": [len(part) for part in parts], "temperature": t, "temperature_source": temperature_source,
            "settings": {"weight_decay": chosen_wd, "epochs": chosen_epochs,
                         "chosen_by": f"{k}-fold cross-validation" + (" grouped by group_id" if groups is not None else "")},
            "cv_loss_by_weight_decay": {str(wd): [float(x) for x in c] for wd, c in curves.items()},
            "cv_accuracy": float((cv_p.argmax(1) == oof_targets).mean()), "cv_ece": cv_ece, "cv_ece_ci95": cv_interval,
            "class_weight": class_weight if class_weight in (None, "balanced") else {str(c): w for c, w in class_weight.items()},
            "epochs": chosen_epochs, "early_stopped": False,
            "scorer": "rank16_bilinear" if model.low_rank else "residual_mlp_hidden64", "seed": seed}
    baseline_vectors = vectors.mean(1) if vectors.ndim == 3 else vectors
    if len(report_rows):
        report = episodes(labels, report_rows, classes, len(report_rows), rng)
        p = probabilities(predict(model, universal, vectors, option_vectors, question, report, device), t)
        ece, interval = ece_interval(p, report["target"], seed, groups=None if groups is None else np.asarray(groups)[report["rows"]])
        required = max(0, 500 - len(report_rows))
        info.update({"report_accuracy": float((p.argmax(1) == report["target"]).mean()), "report_ece": ece, "report_ece_ci95": interval,
                     "calibrated": not required and interval[1] <= .05, "additional_report_items_needed": required})
        k2 = episodes(labels, report_rows, classes, len(report_rows), rng, k=2)
        learned = predict(model, universal, vectors, option_vectors, question, k2, device).argmax(1) == k2["target"]
        baseline = predict(None, universal, baseline_vectors, option_vectors, question, k2, device).argmax(1) == k2["target"]
    else:
        info.update({"report_accuracy": None, "report_ece": None, "report_ece_ci95": None, "calibrated": False,
                     "additional_report_items_needed": 500})
        learned = np.concatenate([run[3][chosen_epochs - 1] for run in chosen])
        baseline = np.concatenate([predict(None, universal, baseline_vectors, option_vectors, question, e, device).argmax(1) == e["target"] for e in fold_k2])
    difference = learned.astype(float) - baseline
    draws = [difference[rng.integers(len(difference), size=len(difference))].mean() for _ in range(500)]
    info["k2_gain_ci95"] = np.quantile(draws, [.025, .975]).tolist()
    info["k2_fallback"] = info["k2_gain_ci95"][0] <= 0
    info["wall_seconds"] = time.perf_counter() - start
    return model.cpu().eval(), info, parts
