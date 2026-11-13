"""A small head compiled from an outcome store.

The head learns the store's own policy: for each stored case it is trained to pick the option that a
``nearest_outcome`` question over the store returns for that state. Once registered it answers the same
question from the head alone, with no lookup. Two objectives are offered. ``bc`` (behaviour cloning)
imitates the store's greedy option at every stored state. ``awr`` (advantage-weighted regression) imitates
the option that was actually taken, weighting each case by ``exp(sign * (outcome - baseline) / tau)`` where
``sign`` is +1 for maximization and -1 for minimization, and the
baseline is the mean outcome of the state's nearest stored cases across all options and ``tau`` is the
standard deviation of training outcomes; weights are clipped at 20.

The head is a rank-16 bilinear form ``x^T W o`` over the state vector and an option vector, plus a bias:
49,153 parameters for 2,304-d states and 768-d option vectors, 36,913 with one-hot option vectors.
"""
import hashlib
import math
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from venn.heads.optimizers import build_outcome_optimizers

CHUNK = 64  # insertion-order cases per split unit, so neighbouring cases share a slice
OBJECTIVES = ("bc", "awr")
MINIMUM_CASES = 5 * CHUNK


class BilinearHead(nn.Module):
    """Rank-``rank`` bilinear score per option: ``left(x) . right(o)`` plus a shared bias."""

    def __init__(self, state_dimensions, option_dimensions, rank=16):
        super().__init__()
        self.left = nn.Linear(state_dimensions, rank, bias=False)
        self.right = nn.Linear(option_dimensions, rank, bias=False)
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, x, options):
        return self.left(x) @ self.right(options).T + self.bias


def count_parameters(module):
    return sum(parameter.numel() for parameter in module.parameters())


def chunk_split(count, seed, chunk=CHUNK, shares=(.6, .2, .2)):
    """Assign insertion-order chunks of cases to slices 0/1/2 (train, early stop, report)."""
    chunks = int(math.ceil(count / chunk))
    order = np.random.default_rng(seed).permutation(chunks)
    bounds = np.cumsum([int(round(chunks * share)) for share in shares[:-1]])
    assignment = np.empty(chunks, dtype=np.int8)
    assignment[order[:bounds[0]]] = 0
    assignment[order[bounds[0]:bounds[1]]] = 1
    assignment[order[bounds[1]:]] = 2
    return np.repeat(assignment, chunk)[:count]


def records_from_store(adapter, options, seed=0):
    """Export state, option, outcome and split per case, keeping option-local insertion chunks intact.

    If per-option rounding omits a slice, allocate those chunks together to retain all three fitting slices.
    """
    export = getattr(adapter, "export", None)
    if not callable(export):
        raise ValueError("Registering a head needs a store adapter that can export its cases")
    exported = export()
    xs, actions, outcomes, splits = [], [], [], []
    for index, option in enumerate(options):
        keys, values = exported.get(option, ((), ()))
        keys = np.asarray(keys, dtype=np.float16)
        values = np.asarray(values, dtype=np.float32)
        if keys.ndim != 2 or len(keys) != len(values) or not np.isfinite(values).all() or not np.isfinite(keys.astype(np.float32)).all():
            raise ValueError("Exported cases must be finite vectors with one outcome each")
        xs.append(keys)
        actions.append(np.full(len(keys), index, dtype=np.int64))
        outcomes.append(values)
        splits.append(chunk_split(len(keys), seed * 7 + index))
    if sum(len(x) for x in xs) < MINIMUM_CASES:
        raise ValueError(f"Registering a head needs at least {MINIMUM_CASES} stored cases")
    split = np.concatenate(splits)
    if len(np.unique(split)) < 3:
        # Small option partitions can round away a slice; allocate their intact chunks together.
        sizes = [min(CHUNK, len(keys) - begin) for keys in xs for begin in range(0, len(keys), CHUNK)]
        assignment = chunk_split(len(sizes) * CHUNK, seed)[::CHUNK]
        split = np.repeat(assignment, sizes)
    return {"x": np.concatenate(xs), "action": np.concatenate(actions), "outcome": np.concatenate(outcomes),
            "split": split}


def memory_policy(adapter, x, options, *, k=11, exact_match=False, unsupported_value=None, goal="maximize", batch=1024):
    """The store's own answer at each state, through the adapter contract.

    Returns ``(greedy, utilities, baseline)``: the index of the option a ``nearest_outcome`` question would
    answer (first maximum; 0 when nothing is supported), the per-option utilities (``nan`` when unsupported
    and no ``unsupported_value`` is given), and the mean outcome of the ``k`` nearest returned cases across
    all options, which is the AWR baseline.
    """
    sign = 1. if goal == "maximize" else -1.
    greedy = np.zeros(len(x), dtype=np.int64)
    utilities = np.full((len(x), len(options)), np.nan, dtype=np.float32)
    baseline = np.zeros(len(x), dtype=np.float32)
    for begin in range(0, len(x), batch):
        block = np.asarray(x[begin:begin + batch])
        found = adapter.search(block, list(options), k, filters={}, exact_match=exact_match)
        for row in range(len(block)):
            pool, scores = [], []
            for index, option in enumerate(options):
                evidence = found[option][row]
                if evidence:
                    mean = float(np.mean([case["outcome"] for case in evidence]))
                    pool.extend((float(case["distance"]), float(case["outcome"])) for case in evidence)
                else:
                    mean = unsupported_value
                if mean is not None:
                    utilities[begin + row, index] = mean
                scores.append(sign * mean if mean is not None else -np.inf)
            greedy[begin + row] = int(np.argmax(scores)) if np.isfinite(scores).any() else 0
            if pool:
                pool.sort(key=lambda case: case[0])
                baseline[begin + row] = float(np.mean([outcome for _, outcome in pool[:k]]))
    return greedy, utilities, baseline


def advantage_weights(outcomes, baseline, train_mask, clip=20., *, goal="maximize"):
    """AWR weights favoring higher rewards or lower costs, with training-slice outcome spread ``tau``."""
    outcomes = np.asarray(outcomes, dtype=np.float64)
    tau = float(np.std(outcomes[train_mask])) or 1.
    sign = 1. if goal == "maximize" else -1.
    weights = np.exp(sign * (outcomes - np.asarray(baseline, dtype=np.float64)) / tau)
    return np.minimum(weights, clip).astype(np.float32), tau


def fit_outcome_head(records, labels, option_vectors, *, weights=None, device="cpu", seed=0, rank=16,
                     max_epochs=20, patience=3, batch=4096, lr=1e-3, weight_decay=0.01, warm_start=None,
                     optimizer="adamw", muon_lr=None):
    """Fit a bilinear head by (weighted) cross-entropy on ``labels``; early stopping on the held-out slice.

    Returns ``(head, report)``. The head keeps the weights of the best early-stop epoch.
    """
    if max_epochs < 1 or any(not np.any(np.asarray(records["split"]) == s) for s in range(3)):
        raise ValueError("Outcome fitting needs at least one epoch and nonempty train, early-stop and report slices")
    torch.manual_seed(seed)
    started = time.perf_counter()
    x = torch.from_numpy(np.asarray(records["x"], dtype=np.float16)).to(device)
    action = torch.from_numpy(np.asarray(records["action"], dtype=np.int64)).to(device)
    split = torch.from_numpy(np.asarray(records["split"], dtype=np.int64)).to(device)
    y = torch.from_numpy(np.asarray(labels, dtype=np.int64)).to(device)
    options = torch.from_numpy(np.asarray(option_vectors, dtype=np.float32)).to(device)
    w = None if weights is None else torch.from_numpy(np.asarray(weights, dtype=np.float32)).to(device)
    head = BilinearHead(x.shape[1], options.shape[1], rank).to(device)
    if warm_start is not None:
        head.load_state_dict(warm_start)
    optimizers = build_outcome_optimizers(head, optimizer=optimizer, lr=lr,
                                          weight_decay=weight_decay, muon_lr=muon_lr)
    slices = {s: torch.nonzero(split == s).squeeze(1) for s in range(3)}

    def batch_loss(rows, reduce):
        logits = head(x[rows].float(), options)
        if w is None:
            return F.cross_entropy(logits, y[rows], reduction=reduce), float(len(rows))
        per_row = F.cross_entropy(logits, y[rows], reduction="none") * w[rows]
        return (per_row.sum() if reduce == "sum" else per_row.sum() / w[rows].sum().clamp_min(1e-9)), float(w[rows].sum())

    def mean_loss(index):
        total = count = 0.
        for begin in range(0, len(index), batch):
            loss, weight = batch_loss(index[begin:begin + batch], "sum")
            total, count = total + float(loss), count + weight
        return total / max(count, 1e-9)

    generator = torch.Generator(device="cpu").manual_seed(seed)
    best, best_state, since, history, epochs = float("inf"), None, 0, [], 0
    for epoch in range(max_epochs):
        head.train()
        order = slices[0][torch.randperm(len(slices[0]), generator=generator).to(device)]
        for begin in range(0, len(order), batch):
            loss, _ = batch_loss(order[begin:begin + batch], "mean")
            for opt in optimizers:
                opt.zero_grad(set_to_none=True)
            loss.backward()
            for opt in optimizers:
                opt.step()
        head.eval()
        with torch.inference_mode():
            early = mean_loss(slices[1])
        history.append(early)
        epochs = epoch + 1
        if early < best - 1e-6:
            best, since = early, 0
            best_state = {key: value.detach().clone() for key, value in head.state_dict().items()}
        else:
            since += 1
            if since >= patience:
                break
    head.load_state_dict(best_state)
    head.eval()
    with torch.inference_mode():
        losses = {name: mean_loss(slices[s]) for s, name in enumerate(("train", "early_stop", "report"))}
        report_rows = slices[2]
        agree_label = agree_action = 0
        for begin in range(0, len(report_rows), batch):
            rows = report_rows[begin:begin + batch]
            guess = head(x[rows].float(), options).argmax(1)
            agree_label += int((guess == y[rows]).sum())
            agree_action += int((guess == action[rows]).sum())
        losses["report_agreement_with_labels"] = agree_label / max(1, len(report_rows))
        losses["report_agreement_with_stored_option"] = agree_action / max(1, len(report_rows))
    hyperparameters = {"max_epochs": max_epochs, "patience": patience, "batch": batch, "lr": lr,
                       "weight_decay": weight_decay}
    if optimizer == "muon":
        hyperparameters.update({"optimizer": "muon", "muon_lr": lr if muon_lr is None else muon_lr})
    return head, {"parameters": count_parameters(head), "rank": rank, "epochs": epochs, "early_stop_history": history,
                  "losses": losses, "weighted": weights is not None, "warm_start": warm_start is not None,
                  "records": {"total": int(len(x)), "train": int(len(slices[0])), "early_stop": int(len(slices[1])),
                              "report": int(len(slices[2]))},
                  "hyperparameters": hyperparameters,
                  "seconds": time.perf_counter() - started}


@torch.inference_mode()
def predict(head, x, option_vectors, device="cpu", batch=4096):
    """Head scores per option, shape ``(len(x), options)``; higher is preferred."""
    head.eval()
    options = torch.as_tensor(np.asarray(option_vectors, dtype=np.float32), device=device)
    out = [head(torch.as_tensor(np.asarray(x[begin:begin + batch], dtype=np.float32), device=device), options).float().cpu().numpy()
           for begin in range(0, len(x), batch)]
    return np.concatenate(out) if out else np.zeros((0, len(option_vectors)), dtype=np.float32)


def digest(head):
    return hashlib.sha256(b"".join(value.detach().cpu().numpy().tobytes() for value in head.state_dict().values())).hexdigest()


def save_family(path, family):
    """Persist a registered family (weights, options, option vectors, tag and report) as one ``.pt`` file."""
    head = family["head"]
    torch.save({"state_dict": {key: value.detach().cpu() for key, value in head.state_dict().items()},
                "options": list(family["options"]), "option_vectors": torch.as_tensor(np.asarray(family["option_vectors"], np.float32)),
                "tag": dict(family["tag"]), "objective": family["objective"], "goal": family["goal"], "rank": int(family["rank"]),
                "memory_question": dict(family["memory_question"]), "report": family["report"]}, path)


def load_family(path, device="cpu"):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state = payload["state_dict"]
    head = BilinearHead(state["left.weight"].shape[1], state["right.weight"].shape[1], payload["rank"])
    head.load_state_dict(state)
    head = head.to(device).eval()
    return {"head": head, "options": list(payload["options"]), "option_vectors": payload["option_vectors"].numpy(),
            "tag": payload["tag"], "objective": payload["objective"], "goal": payload["goal"], "rank": payload["rank"],
            "memory_question": payload["memory_question"], "report": payload["report"], "sha256": digest(head)}
