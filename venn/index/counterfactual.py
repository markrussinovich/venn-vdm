"""Paired Choice-logit diagnostics without semantic equivalence claims.

The caller supplies corresponding rows and stable option identities. This
module compares outputs; it does not establish that an intervention preserved
meaning, that a label is correct, or that logits are calibrated probabilities.
"""

import numpy as np


def compare_choice_logits(base_options, base_logits, variant_options, variant_logits, *, atol=1e-6):
    """Compare Choice logits across two candidate frames by *option identity*.

    Inputs are 2-D finite NumPy-compatible arrays with a final ``none`` column.
    Both arrays must have identical state-row order. Return non-probabilistic,
    per-row argmax changes, tie sets and logit deltas on the common options.
    A changed candidate set is reported as a changed intervention, not an error.

    The two calls must use compatible representation, head and score units;
    the function cannot infer or certify those from the numeric arrays.
    """
    def validate(options, logits):
        if (not isinstance(options, (tuple, list)) or not options
                or any(not isinstance(x, str) or not x or x == "none" for x in options)
                or len(set(options)) != len(options)):
            raise ValueError("Options must be unique nonempty identities other than 'none'")
        values = np.asarray(logits, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] != len(options) + 1:
            raise ValueError("Logits must be (states, options + one none column)")
        if not np.isfinite(values).all():
            raise ValueError("Logits must be finite; compare only unmasked Choice columns")
        return values

    if isinstance(atol, bool) or not isinstance(atol, (float, int)) or not np.isfinite(atol) or atol < 0:
        raise ValueError("atol must be finite and nonnegative")
    before = validate(base_options, base_logits)
    after = validate(variant_options, variant_logits)
    if before.shape[0] != after.shape[0]:
        raise ValueError("Both frames must have corresponding state rows")
    original = tuple(base_options) + ("none",)
    changed = tuple(variant_options) + ("none",)
    before_position = {name: i for i, name in enumerate(original)}
    after_position = {name: i for i, name in enumerate(changed)}
    common = tuple(name for name in original if name in after_position)

    def winner(row, names):
        maximum = row.max()
        ties = [name for name, value in zip(names, row) if maximum - value <= atol]
        return names[int(row.argmax())], ties

    rows = []
    for left, right in zip(before, after):
        left_choice, left_ties = winner(left, original)
        right_choice, right_ties = winner(right, changed)
        deltas = [abs(left[before_position[name]] - right[after_position[name]]) for name in common]
        rows.append({
            "base_choice": left_choice,
            "variant_choice": right_choice,
            "choice_changed": left_choice != right_choice,
            "abstention_changed": (left_choice == "none") != (right_choice == "none"),
            "base_ties": left_ties,
            "variant_ties": right_ties,
            "max_shared_logit_delta": float(max(deltas)),
            "shared_scores_unchanged": all(delta <= atol for delta in deltas),
        })
    return {
        "same_option_set": set(base_options) == set(variant_options),
        "added": [name for name in variant_options if name not in before_position],
        "removed": [name for name in base_options if name not in after_position],
        "compared_options": list(common),
        "rows": rows,
    }
