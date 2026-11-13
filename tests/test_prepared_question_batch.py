"""CPU-only, synthetic-vector parity tests for shared-model multi-question scoring."""

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from venn.heads.abstention import CosineAbstention
from venn.heads.family import ContextAbstention
from venn.index.apply import PreparedQuestion, PreparedQuestionBatch


def _fixture(head, count=4, width=12):
    torch.manual_seed(137)
    questions = []
    for k, temperature in zip([1, 2, 5, 9][:count], [0.7, 1.1, 2.5, 0.8]):
        q = torch.randn(width)
        options = F.normalize(torch.randn(k, width), dim=-1)
        questions.append(PreparedQuestion(head, q, options, temperature=temperature))
    states = F.normalize(torch.randn(17, width), dim=-1)
    return questions, states


@pytest.mark.parametrize("context", [False, True])
def test_batched_logits_match_individual_questions(context):
    torch.manual_seed(17)
    head = (ContextAbstention(temperature=0.37, none_intercept=-0.41, dimensions=12, hidden=15) if context else CosineAbstention(
        temperature=0.37, none_intercept=-0.41, dimensions=12, hidden=15)).eval()
    # Exercise nontrivial learned none head in both cases.
    with torch.no_grad():
        nn.init.normal_(head.none[-1].weight, std=0.2)
        nn.init.normal_(head.none[-1].bias, std=0.2)
    questions, states = _fixture(head)
    batch = PreparedQuestionBatch(questions)
    actual = batch.logits(states)
    expected = tuple(q.logits(states) for q in questions)
    assert len(actual) == 4
    for a, e, k in zip(actual, expected, batch.lengths):
        assert a.shape == (len(states), k + 1)
        torch.testing.assert_close(a, e, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("context", [False, True])
def test_stream_preserves_per_question_calibration_and_order(context):
    head = ContextAbstention(temperature=0.37, none_intercept=-0.41, dimensions=12, hidden=15) if context else CosineAbstention(
        temperature=0.37, none_intercept=0.2, dimensions=12, hidden=15)
    questions, states = _fixture(head)
    actual = list(PreparedQuestionBatch(questions).stream(states.numpy(), batch_size=6))
    assert [offset for offset, _ in actual] == [0, 6, 12]
    for index, q in enumerate(questions):
        want = torch.cat([prob for _, prob in q.stream(states.numpy(), batch_size=6)])
        got = torch.cat([prob[index] for _, prob in actual])
        torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-5)


def test_mixed_models_rejected():
    first = CosineAbstention(0.4, 0, dimensions=12)
    second = CosineAbstention(0.4, 0, dimensions=12)
    q = torch.ones(12)
    o = torch.eye(12)[:3]
    with pytest.raises(ValueError, match="shared model"):
        PreparedQuestionBatch([PreparedQuestion(first, q, o), PreparedQuestion(second, q, o)])


def test_empty_questions_rejected():
    with pytest.raises(ValueError, match="one or more"):
        PreparedQuestionBatch([])


def test_one_question_and_zero_row_batch():
    head = CosineAbstention(0.9, 0.1, dimensions=12)
    questions, states = _fixture(head, count=1)
    for rows in [states, states[:0]]:
        got, = PreparedQuestionBatch(questions).logits(rows)
        torch.testing.assert_close(got, questions[0].logits(rows), atol=1e-5, rtol=1e-5)


def test_question_permutation_is_equivariant():
    head = CosineAbstention(0.4, -0.1, dimensions=12, hidden=16)
    questions, states = _fixture(head)
    baseline = PreparedQuestionBatch(questions).logits(states)
    order = [3, 0, 2, 1]
    reordered = PreparedQuestionBatch([questions[i] for i in order]).logits(states)
    for j, original_index in enumerate(order):
        torch.testing.assert_close(reordered[j], baseline[original_index], atol=1e-5, rtol=1e-5)


def test_candidate_permutation_moves_only_candidate_columns():
    head = ContextAbstention(0.7, -0.5, dimensions=12, hidden=16).eval()
    torch.manual_seed(51)
    _, states = _fixture(head, count=1)
    original = torch.randn(5, 12)
    perm = torch.tensor([2, 4, 0, 1, 3])
    # Same question is necessary: use the identical question for both prewarm operations.
    instructions = torch.randn(12)
    base = PreparedQuestion(head, instructions, original)
    moved = PreparedQuestion(head, instructions, original[perm])
    expected = base.logits(states)
    actual = PreparedQuestionBatch([moved]).logits(states)[0]
    torch.testing.assert_close(actual[:, :-1], expected[:, :-1][:, perm], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(actual[:, -1], expected[:, -1], atol=1e-5, rtol=1e-5)


def test_stream_rejects_nonpositive_batch_size():
    head = CosineAbstention(0.9, 0.1, dimensions=12)
    questions, states = _fixture(head, count=1)
    with pytest.raises(ValueError, match="batch_size"):
        list(PreparedQuestionBatch(questions).stream(states.numpy(), batch_size=0))


@pytest.mark.parametrize("temperature", [0, -1, float("nan"), float("inf"), -float("inf"), True, "1"])
def test_invalid_calibration_temperatures_fail_before_scoring(temperature):
    head = CosineAbstention(0.9, 0.1, dimensions=12)
    questions, states = _fixture(head, count=1)
    with pytest.raises(ValueError, match="temperature"):
        PreparedQuestion(head, torch.ones(12), torch.eye(12)[:2], temperature=temperature)
    with pytest.raises(ValueError, match="temperature"):
        list(questions[0].stream(states.numpy(), temperature=temperature))
    with pytest.raises(ValueError, match="temperature"):
        list(PreparedQuestionBatch(questions).stream(states.numpy(), temperature=temperature))
    questions[0].calibration_temperature = temperature
    with pytest.raises(ValueError, match="temperature"):
        PreparedQuestionBatch(questions)


@pytest.mark.parametrize("context", [False, True])
def test_frozen_head_forward_is_an_independent_parity_control(context):
    """Compare with the original batched forward, not only PreparedQuestion."""
    torch.manual_seed(902)
    cls = ContextAbstention if context else CosineAbstention
    head = cls(0.75, -0.3, dimensions=12, hidden=19).eval()
    with torch.no_grad():
        nn.init.normal_(head.none[-1].weight, std=0.3)
        nn.init.normal_(head.none[-1].bias, std=0.1)
    states = F.normalize(torch.randn(7, 12), dim=-1)
    original = []
    prepared = []
    for k in (1, 3, 8, 2, 4):
        question = torch.randn(12)
        options = torch.randn(k, 12)
        original.append((question, options))
        prepared.append(PreparedQuestion(head, question, options))
    result = PreparedQuestionBatch(prepared, max_workspace_bytes=2048).logits(states)
    for actual, (question, options) in zip(result, original):
        expected = head(states, question.expand(len(states), -1),
                        options.expand(len(states), -1, -1),
                        torch.ones(len(states), len(options), dtype=torch.bool))
        torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)


def test_hand_calculated_cosine_and_none_logits():
    head = CosineAbstention(temperature=0.5, none_intercept=-0.2, dimensions=2, hidden=4)
    question = torch.tensor([1.0, 1.0])
    options = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    state = torch.tensor([[1.0, 0.0]])
    actual = PreparedQuestionBatch([PreparedQuestion(head, question, options)]).logits(state)[0]
    expected = torch.tensor([[2 ** 0.5, -(2 ** 0.5), -0.4]])
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_large_ragged_set_is_bucketed_and_order_preserved():
    head = CosineAbstention(0.6, -0.1, dimensions=12, hidden=8).eval()
    torch.manual_seed(144)
    question = torch.randn(12)
    sizes = [1] * 31 + [2] * 31 + [3] * 31 + [20] + [1] * 31
    prepared = [PreparedQuestion(head, question, torch.randn(k, 12)) for k in sizes]
    batch = PreparedQuestionBatch(prepared, max_workspace_bytes=4096)
    # Grouping never pads 1-option sets out to the 20-option maximum.
    storage = sum(o.numel() for _, o, _, _, _ in batch._groups)
    original = sum(q.options.numel() for q in prepared)
    assert storage < 2 * original
    assert len(batch._groups) > 1
    states = F.normalize(torch.randn(17, 12), dim=-1)
    actual = batch.logits(states)
    assert len(actual) == len(prepared)
    for i, original_question in enumerate(prepared):
        torch.testing.assert_close(actual[i], original_question.logits(states), atol=1e-5, rtol=1e-5)


def test_stream_workspace_splits_rows_and_calibration_override():
    head = ContextAbstention(0.85, -0.1, dimensions=12, hidden=19).eval()
    questions, states = _fixture(head)
    batch = PreparedQuestionBatch(questions, max_workspace_bytes=1024)
    assert batch._row_cap < len(states)
    chunks = list(batch.stream(states.numpy(), batch_size=100, temperature=1.25))
    assert chunks[0][0] == 0 and chunks[-1][0] > 0
    assert all(b - a <= batch._row_cap for (a, _), (b, _) in zip(chunks, chunks[1:]))
    for i, q in enumerate(questions):
        got = torch.cat([chunk[i] for _, chunk in chunks])
        expected = (q.logits(states) / 1.25).softmax(-1)
        torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-5)


def test_invalid_budget_and_states_rejected():
    head = CosineAbstention(0.6, 0.1, dimensions=12)
    questions, states = _fixture(head)
    for budget in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match="max_workspace_bytes"):
            PreparedQuestionBatch(questions, max_workspace_bytes=budget)
    batch = PreparedQuestionBatch(questions)
    for bad in (states[:, :-1], states.to(torch.float64), states[0]):
        with pytest.raises(ValueError, match="States must"):
            batch.logits(bad)

@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("context", [False, True])
def test_seeded_randomized_ragged_parity(seed, context):
    torch.manual_seed(seed + 2222)
    cls = ContextAbstention if context else CosineAbstention
    head = cls(0.45, -0.2, dimensions=12, hidden=14).eval()
    with torch.no_grad():
        nn.init.normal_(head.none[-1].weight, std=0.2)
    questions = [PreparedQuestion(head, torch.randn(12), torch.randn(k, 12))
                 for k in torch.randint(1, 20, (13,)).tolist()]
    batch = PreparedQuestionBatch(questions, max_workspace_bytes=512)
    states = F.normalize(torch.randn(11, 12), dim=-1)
    actual = batch.logits(states)
    for got, prepared in zip(actual, questions):
        torch.testing.assert_close(got, prepared.logits(states), atol=3e-5, rtol=3e-5)
