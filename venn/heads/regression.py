"""Regression family heads: a closed-form ridge from 768-d vectors to a numeric target.

Rows are split into train / selection / report slices. The ridge penalty is chosen by repeated K-fold
cross-validation inside the train slice only; folds are label-stratified (rows sorted by target with a
seeded tiebreak, then dealt round-robin), one seed per repeat. The fitted head reports RMSE, MAE and
Spearman on the report slice and an 80% prediction interval from report-slice residual quantiles.
"""
import hashlib
import json
import math

import numpy as np

ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
LEVEL = 0.80
MIN_CALIBRATION_ITEMS = 500


def stratified_folds(target, k, seed):
    rng = np.random.default_rng(seed)
    order = np.lexsort((rng.random(len(target)), target))
    return [np.sort(order[i::k]) for i in range(k)]


def default_split(target, seed):
    """60/20/20 by target: rows sorted by target with a seeded tiebreak, positions dealt 3:1:1."""
    rng = np.random.default_rng(seed)
    order = np.lexsort((rng.random(len(target)), target))
    slot = np.arange(len(order)) % 5
    return np.sort(order[slot < 3]), np.sort(order[slot == 3]), np.sort(order[slot == 4])


def fit_ridge(X, y, alpha):
    """Closed form with an unpenalised intercept: centre X and y, solve (Xc'Xc + alpha I) w = Xc'yc."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x_mean, y_mean = X.mean(0), y.mean()
    Xc = X - x_mean
    gram = Xc.T @ Xc
    gram[np.diag_indices_from(gram)] += alpha
    coef = np.linalg.solve(gram, Xc.T @ (y - y_mean))
    return coef, float(y_mean - x_mean @ coef)


def predict(coef, intercept, X):
    return np.asarray(X, dtype=np.float64) @ coef + intercept


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def spearman(a, b):
    """Rank correlation with average ranks for ties."""
    def ranks(v):
        v = np.asarray(v, dtype=np.float64)
        order = np.argsort(v, kind="stable")
        r = np.empty(len(v))
        r[order] = np.arange(len(v), dtype=np.float64)
        _, inverse, counts = np.unique(v, return_inverse=True, return_counts=True)
        sums = np.bincount(inverse, weights=r)
        return (sums / counts)[inverse]
    if len(a) < 3:
        return None
    ra, rb = ranks(a), ranks(b)
    if ra.std() == 0 or rb.std() == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def choose_alpha(X, y, folds=5, repeats=3, alphas=ALPHAS):
    """Mean over repeats of out-of-fold RMSE, per alpha; returns (best alpha, table, out-of-fold predictions).

    One SVD of each training fold's centred matrix serves every alpha: w = V diag(s / (s^2 + alpha)) U' yc.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    oof = {alpha: np.zeros((repeats, len(y))) for alpha in alphas}
    for repeat in range(repeats):
        for fold in stratified_folds(y, folds, seed=repeat):
            train = np.setdiff1d(np.arange(len(y)), fold)
            x_mean, y_mean = X[train].mean(0), y[train].mean()
            u, s, vt = np.linalg.svd(X[train] - x_mean, full_matrices=False)
            uty = u.T @ (y[train] - y_mean)
            projected = (X[fold] - x_mean) @ vt.T
            for alpha in alphas:
                oof[alpha][repeat, fold] = projected @ (s / (s * s + alpha) * uty) + y_mean
    table = {alpha: float(np.mean([rmse(y, oof[alpha][r]) for r in range(repeats)])) for alpha in alphas}
    best = min(alphas, key=lambda a: (table[a], a))
    return best, table, oof[best].mean(0)


def limited_threads(limit=8):
    """Cap BLAS threads while fitting: many-core hosts otherwise oversubscribe these small solves.

    On Windows this does nothing: resizing OpenBLAS's thread pool repeatedly deadlocked a later small LAPACK solve on
    Windows ARM64 in CI, whatever the new thread count."""
    import contextlib
    import sys

    if sys.platform == "win32":
        return contextlib.nullcontext()
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:  # optional; without it the fit is correct, only slower on very wide hosts
        return contextlib.nullcontext()
    return threadpool_limits(limits=limit, user_api="blas")


def wilson(successes, n, z=1.959963984540054):
    if n == 0:
        return [None, None]
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [max(0.0, centre - half), min(1.0, centre + half)]


def interval(residuals, seed):
    """80% interval offsets from residual quantiles, and its coverage estimated by a seeded two-half swap."""
    residuals = np.asarray(residuals, dtype=np.float64)
    lo_q, hi_q = (1 - LEVEL) / 2, 1 - (1 - LEVEL) / 2
    offsets = [float(np.quantile(residuals, lo_q)), float(np.quantile(residuals, hi_q))]
    if len(residuals) < 4:
        return offsets, None, [None, None]
    half = np.random.default_rng(seed).permutation(len(residuals))
    a, b = half[: len(half) // 2], half[len(half) // 2:]
    hits = 0
    for fit_rows, test_rows in ((a, b), (b, a)):
        lo, hi = np.quantile(residuals[fit_rows], [lo_q, hi_q])
        hits += int(((residuals[test_rows] >= lo) & (residuals[test_rows] <= hi)).sum())
    return offsets, hits / len(residuals), wilson(hits, len(residuals))


def checksum(coef, intercept):
    return hashlib.sha256(np.asarray(coef, dtype=np.float64).tobytes() + np.float64(intercept).tobytes()).hexdigest()


def register(X, y, *, seed=0, split=None, folds=5, repeats=3, alphas=ALPHAS):
    """Fit and report a ridge regression family. Returns (head dict, report dict, parts)."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    parts = split if split is not None else default_split(y, seed)
    train, selection, report_rows = (np.asarray(p, dtype=np.int64) for p in parts)
    if len(train) < 10:
        raise ValueError("The train slice needs at least 10 rows")
    if len(set(train.tolist()) | set(selection.tolist()) | set(report_rows.tolist())) != len(train) + len(selection) + len(report_rows):
        raise ValueError("Split slices must not share rows")
    with limited_threads():
        alpha, table, oof = choose_alpha(X[train], y[train], folds, repeats, alphas)
        coef, intercept = fit_ridge(X[train], y[train], alpha)
    report = {"objective": "regress", "form": "ridge", "alpha": alpha, "cv_rmse_by_alpha": {str(a): v for a, v in table.items()},
              "cv": {"folds": folds, "repeats": repeats, "scheme": "label-stratified, seeded tiebreak, seed = repeat"},
              "slices": {"train": int(len(train)), "selection": int(len(selection)), "report": int(len(report_rows))}}
    if len(selection):
        report["selection"] = {"rmse": rmse(y[selection], predict(coef, intercept, X[selection]))}
    if len(report_rows):
        pred = predict(coef, intercept, X[report_rows])
        residuals = y[report_rows] - pred
        report["report"] = {"rmse": rmse(y[report_rows], pred), "mae": float(np.mean(np.abs(residuals))), "spearman": spearman(y[report_rows], pred)}
        source = "report-slice residuals"
    else:
        residuals = y[train] - oof
        report["report"] = None
        source = "cross-validated train residuals (no report slice was given)"
    offsets, coverage, coverage_ci = interval(residuals, seed)
    calibrated = bool(len(report_rows) >= MIN_CALIBRATION_ITEMS and coverage_ci[0] is not None
                      and coverage_ci[0] >= LEVEL - 0.05 and coverage_ci[1] <= LEVEL + 0.05)
    report["interval"] = {"level": LEVEL, "offsets": offsets, "source": source, "coverage": coverage, "coverage_ci95": coverage_ci,
                          "coverage_method": "quantiles from one seeded half, coverage on the other, both ways"}
    report["calibrated"] = calibrated
    report["checksum"] = checksum(coef, intercept)
    head = {"coef": coef, "intercept": intercept, "offsets": offsets, "calibrated": calibrated}
    return head, report, (train, selection, report_rows)


# ----------------------------------------------------------------------------- kernel ridge and extra numeric inputs
KERNEL_ALPHAS = (0.01, 0.1, 1.0, 10.0)
GAMMAS = (0.5, 1.0, 2.0, 4.0)
INPUT_SCALES = (0.03, 0.1, 0.3)
INPUTS = ("log_words", "log_chars")


def input_features(texts, names):
    """Extra numeric inputs computed from the state text: log word count and log character count."""
    columns = []
    for name in names:
        if name == "log_words":
            columns.append([math.log(max(len(t.split()), 1)) for t in texts])
        elif name == "log_chars":
            columns.append([math.log(max(len(t), 1)) for t in texts])
        else:
            raise ValueError(f"Unknown input {name!r}; use {', '.join(INPUTS)}")
    return np.array(columns, dtype=np.float64).T


def design(vectors, extra, stats, scale):
    """The head's inputs: the vector, plus standardised extra inputs multiplied by ``scale``."""
    if extra is None:
        return np.asarray(vectors, dtype=np.float64)
    mean, sd = stats
    return np.hstack([np.asarray(vectors, dtype=np.float64), scale * (np.asarray(extra, dtype=np.float64) - mean) / sd])


def rbf(A, B, gamma):
    sq = (A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2 * A @ B.T
    return np.exp(-gamma * np.maximum(sq, 0))


def spd_solve(A, b, method=None):
    """Solve A x = b for a symmetric positive-definite A.

    ``np.linalg.solve`` everywhere except Windows, where conjugate gradients built from matrix products are used: in CI,
    numpy's LAPACK solve hung on Windows ARM64 for small, well-conditioned kernel systems (24 x 24, condition 24)."""
    import sys

    method = method or ("cg" if sys.platform == "win32" else "lapack")
    if method == "lapack":
        return np.linalg.solve(A, b)
    x = np.zeros_like(b, dtype=np.float64)
    r = np.asarray(b, dtype=np.float64).copy()
    p = r.copy()
    rs = float(r @ r)
    tol = 1e-28 * max(float(b @ b), 1e-300)
    for _ in range(20 * len(b) + 100):
        if rs <= tol:
            break
        Ap = A @ p
        step = rs / float(p @ Ap)
        x += step * p
        r -= step * Ap
        rs_new = float(r @ r)
        p = r + (rs_new / rs) * p
        rs = rs_new
    return x


def fit_kernel(X, y, alpha, gamma):
    """RBF kernel ridge on a centred target: dual weights solve (K + alpha I) a = y - mean(y)."""
    mu = float(np.mean(y))
    K = rbf(X, X, gamma)
    K[np.diag_indices_from(K)] += alpha
    return spd_solve(K, np.asarray(y, dtype=np.float64) - mu), mu


def predict_head(head, vectors, extra=None):
    X = design(vectors, extra, (head["extra_mean"], head["extra_sd"]) if head.get("extra_mean") is not None else None, head.get("scale"))
    if head["form"] == "kernel_ridge":
        return rbf(X, head["X"], head["gamma"]) @ head["dual"] + head["mu"]
    return X @ head["coef"] + head["intercept"]


def _fit(form, X, y, alpha, gamma):
    if form == "kernel_ridge":
        dual, mu = fit_kernel(X, y, alpha, gamma)
        return {"form": form, "X": X, "dual": dual, "mu": mu, "gamma": gamma}
    coef, intercept = fit_ridge(X, y, alpha)
    return {"form": form, "coef": coef, "intercept": intercept}


def choose_extended(vectors, extra, y, form, folds=5, repeats=3, alphas=None, gammas=None, scales=None):
    """Joint choice of (input scale, alpha, gamma) by mean out-of-fold RMSE over repeats, the folds ``choose_alpha`` uses."""
    alphas = tuple(alphas or (KERNEL_ALPHAS if form == "kernel_ridge" else ALPHAS))
    gammas = tuple(gammas or GAMMAS) if form == "kernel_ridge" else (None,)
    scales = tuple(scales or INPUT_SCALES) if extra is not None else (None,)
    oof = {(s, a, g): np.zeros((repeats, len(y))) for s in scales for a in alphas for g in gammas}
    for repeat in range(repeats):
        for fold in stratified_folds(y, folds, seed=repeat):
            train = np.setdiff1d(np.arange(len(y)), fold)
            stats = None if extra is None else (extra[train].mean(0), extra[train].std(0) + 1e-9)
            for s in scales:
                Xtr = design(vectors[train], None if extra is None else extra[train], stats, s)
                Xte = design(vectors[fold], None if extra is None else extra[fold], stats, s)
                for g in gammas:
                    for a in alphas:
                        head = _fit(form, Xtr, y[train], a, g)
                        head.update(extra_mean=None)
                        oof[(s, a, g)][repeat, fold] = predict_head(head, Xte)
    table = {key: float(np.mean([rmse(y, v[r]) for r in range(repeats)])) for key, v in oof.items()}
    best = min(table, key=lambda k: (table[k], [x if x is not None else 0 for x in k]))
    return best, table, oof[best].mean(0)


def register_extended(vectors, y, *, form="kernel_ridge", extra=None, input_names=(), seed=0, split=None, folds=5, repeats=3,
                      alphas=None, gammas=None, scales=None):
    """Fit and report a kernel-ridge family, or a ridge family with extra numeric inputs. Same report as ``register``."""
    vectors = np.asarray(vectors, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    extra = None if extra is None else np.asarray(extra, dtype=np.float64)
    parts = split if split is not None else default_split(y, seed)
    train, selection, report_rows = (np.asarray(p, dtype=np.int64) for p in parts)
    if len(train) < 10:
        raise ValueError("The train slice needs at least 10 rows")
    if len(set(train.tolist()) | set(selection.tolist()) | set(report_rows.tolist())) != len(train) + len(selection) + len(report_rows):
        raise ValueError("Split slices must not share rows")
    with limited_threads():
        (scale, alpha, gamma), table, oof = choose_extended(vectors[train], None if extra is None else extra[train], y[train], form,
                                                           folds, repeats, alphas, gammas, scales)
        stats = None if extra is None else (extra[train].mean(0), extra[train].std(0) + 1e-9)
        X = design(vectors[train], None if extra is None else extra[train], stats, scale)
        head = _fit(form, X, y[train], alpha, gamma)
    head.update(extra_mean=None if stats is None else stats[0], extra_sd=None if stats is None else stats[1], scale=scale, inputs=list(input_names))
    report = {"objective": "regress", "form": form, "alpha": alpha, "gamma": gamma, "input_scale": scale, "inputs": list(input_names),
              "cv_rmse_by_setting": {json.dumps([s, a, g]): v for (s, a, g), v in table.items()},
              "cv": {"folds": folds, "repeats": repeats, "scheme": "label-stratified, seeded tiebreak, seed = repeat"},
              "slices": {"train": int(len(train)), "selection": int(len(selection)), "report": int(len(report_rows))}}
    def pred(rows):
        return predict_head(head, vectors[rows], None if extra is None else extra[rows])
    if len(selection):
        report["selection"] = {"rmse": rmse(y[selection], pred(selection))}
    if len(report_rows):
        p = pred(report_rows)
        residuals = y[report_rows] - p
        report["report"] = {"rmse": rmse(y[report_rows], p), "mae": float(np.mean(np.abs(residuals))), "spearman": spearman(y[report_rows], p)}
        source = "report-slice residuals"
    else:
        residuals = y[train] - oof
        report["report"] = None
        source = "cross-validated train residuals (no report slice was given)"
    offsets, coverage, coverage_ci = interval(residuals, seed)
    calibrated = bool(len(report_rows) >= MIN_CALIBRATION_ITEMS and coverage_ci[0] is not None
                      and coverage_ci[0] >= LEVEL - 0.05 and coverage_ci[1] <= LEVEL + 0.05)
    report["interval"] = {"level": LEVEL, "offsets": offsets, "source": source, "coverage": coverage, "coverage_ci95": coverage_ci,
                          "coverage_method": "quantiles from one seeded half, coverage on the other, both ways"}
    report["calibrated"] = calibrated
    head.update(offsets=offsets, calibrated=calibrated)
    report["checksum"] = extended_checksum(head)
    return head, report, (train, selection, report_rows)


HEAD_ARRAYS = ("X", "dual", "coef", "extra_mean", "extra_sd")
HEAD_SCALARS = ("form", "mu", "gamma", "intercept", "scale", "inputs", "offsets", "calibrated")


def extended_checksum(head):
    digest = hashlib.sha256()
    for key in HEAD_ARRAYS:
        if head.get(key) is not None:
            digest.update(np.asarray(head[key], dtype=np.float64).tobytes())
    digest.update(json.dumps({k: head.get(k) for k in HEAD_SCALARS}, default=float, sort_keys=True).encode())
    return digest.hexdigest()
