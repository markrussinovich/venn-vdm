"""Metamorphic checks for candidate-local Choice geometry and its limitations.

These are algebraic representation checks on synthetic embeddings, not tests of
semantic paraphrase invariance or calibration of the pretrained reader/head.
"""

import pytest
import torch
from torch.nn import functional as F

from venn.heads.abstention import CosineAbstention
from venn.index.prewarm import prewarm


def _case(seed, *, width=8, options=5):
    generator = torch.Generator().manual_seed(seed)
    state = F.normalize(torch.randn(1, width, generator=generator), dim=-1)
    question = torch.randn(1, width, generator=generator)
    choices = torch.randn(1, options, width, generator=generator)
    mask = torch.ones(1, options, dtype=torch.bool)
    mask[0, -1] = False  # Include a genuinely masked padding row.
    return state, question, choices, mask


@pytest.mark.parametrize("seed", range(10))
def test_candidate_order_permutation_changes_only_option_axis(seed):
    """Reordering identical candidate *identities* permutes logits, not none."""
    state, question, choices, mask = _case(seed)
    model = CosineAbstention(.7, -.23, dimensions=8, hidden=5).eval()
    with torch.no_grad():
        model.none[-1].weight.fill_(.11)  # Exercise the nontrivial learned none path.
        original = model(state, question, choices, mask)
        order = torch.tensor([4, 2, 0, 3, 1])
        changed = model(state, question, choices[:, order], mask[:, order])
    torch.testing.assert_close(changed[:, :-1], original[:, order], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(changed[:, -1:], original[:, -1:], atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("seed", range(5))
def test_joint_question_and_candidate_translation_is_invariant(seed):
    """Common translation cancels in option-centering; state is unchanged."""
    state, question, choices, mask = _case(seed)
    offset = torch.arange(8, dtype=state.dtype).reshape(1, 8) * .17
    model = CosineAbstention(.6, .2, dimensions=8, hidden=4).eval()
    with torch.no_grad():
        model.none[-1].weight.fill_(.13)
        before = model(state, question, choices, mask)
        after = model(state, question + offset, choices + offset[:, None], mask)
    torch.testing.assert_close(after, before, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("seed", range(5))
def test_centered_cosine_is_orthogonally_covariant(seed):
    """Rotate *both* the state and candidates; only geometric similarity is claimed."""
    state, question, choices, mask = _case(seed)
    generator = torch.Generator().manual_seed(1000 + seed)
    orthogonal, _ = torch.linalg.qr(torch.randn(8, 8, generator=generator))
    q, o, _ = prewarm(question, choices, mask)
    qr, rotated, _ = prewarm(question @ orthogonal, choices @ orthogonal, mask)
    torch.testing.assert_close(qr, q @ orthogonal, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(rotated, o @ orthogonal, atol=2e-6, rtol=2e-6)
    before = (state[:, None] * o).sum(-1)
    after = ((state @ orthogonal)[:, None] * rotated).sum(-1)
    torch.testing.assert_close(after, before, atol=2e-6, rtol=2e-6)


def test_fixed_learned_head_is_not_generally_rotation_invariant():
    """Negative control: rotated data without transported weights may change none."""
    model = CosineAbstention(1., 0., dimensions=2, hidden=1).eval()
    with torch.no_grad():
        model.none[0].weight.zero_()
        model.none[0].bias.zero_()
        model.none[0].weight[0, 0] = 1.
        model.none[-1].weight.fill_(1.)
        model.none[-1].bias.zero_()
    state = torch.tensor([[1., 0.]])
    question = torch.tensor([[0., .8]])
    choices = torch.tensor([[[1., 0.], [0., 1.], [-1., 0.]]])
    mask = torch.ones(1, 3, dtype=torch.bool)
    rotation = torch.tensor([[0., 1.], [-1., 0.]])
    with torch.no_grad():
        before = model(state, question, choices, mask)
        rotated = model(state @ rotation, question @ rotation, choices @ rotation, mask)
    torch.testing.assert_close(rotated[:, :-1], before[:, :-1], atol=1e-6, rtol=1e-6)
    assert abs((before[0, -1] - rotated[0, -1]).item()) > .1


def test_duplicate_option_is_not_a_frame_symmetry():
    """Adding a candidate can change the centroid and therefore existing scores."""
    state = torch.tensor([[1., 0.]])
    question = torch.tensor([[0., .8]])
    choices = torch.tensor([[[1., 0.], [0., 1.], [-1., 0.]]])
    repeated = torch.cat([choices, choices[:, :1]], dim=1)
    _, a, _ = prewarm(question, choices, torch.ones(1, 3, dtype=torch.bool))
    _, b, _ = prewarm(question, repeated, torch.ones(1, 4, dtype=torch.bool))
    assert not torch.allclose((state[:, None] * a).sum(-1),
                              (state[:, None] * b[:, :3]).sum(-1), atol=1e-4)


def test_identical_option_vectors_are_finite_but_uninformative():
    """Degenerate geometry must not yield NaN or invented directional evidence."""
    model = CosineAbstention(.5, .1, dimensions=2, hidden=4).eval()
    state = torch.tensor([[1., 0.]])
    question = torch.tensor([[0., 1.]])
    choices = torch.tensor([[[1., 1.], [1., 1.], [1., 1.]]])
    mask = torch.ones(1, 3, dtype=torch.bool)
    result = model(state, question, choices, mask)
    assert torch.isfinite(result).all()
    torch.testing.assert_close(result[:, :-1], torch.zeros(1, 3))
