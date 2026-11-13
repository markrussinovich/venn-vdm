"""CPU-only conformance checks for caller-supplied OutcomeService adapters."""

from types import SimpleNamespace

import numpy as np
import pytest

from venn.outcomes import OutcomeService
from venn.profiles import VectorTag

TAG = VectorTag("experimental:contract-test", "a" * 40, "test-vector", 2)
STATES = [{"vector": [1., 0.], "tag": TAG.to_dict()}]


def _question(handle, **changes):
    return {"q": {"type": "nearest_outcome", "store": handle,
                  "options": ["a", "b"], "k": 2, **changes}}


class MemoryAdapter:
    tag = TAG
    distance = "squared_l2"
    read_after_write = "immediate for this synthetic test"

    def __init__(self, response=None):
        self.response = response

    def search(self, queries, options, k, *, filters, exact_match=False):
        if self.response is not None:
            return self.response
        assert queries.shape == (1, 2) and options == ["a", "b"]
        assert k == 2 and filters == {} and not exact_match
        return {"a": [[{"id": "a0", "outcome": 3., "distance": .04}]],
                "b": [[]]}


def test_complete_adapter_contract_preserves_evidence_and_abstention():
    service = OutcomeService(head_device="cpu")
    handle = service.register_store(MemoryAdapter())
    response = service.decide_batch(STATES, _question(handle), rng=np.random.default_rng(2))
    answer = response["results"][0]["answers"]["q"]
    assert answer["answer"] == "a"
    assert answer["options"]["a"]["mean"] == 3.
    assert answer["options"]["a"]["evidence_ids"] == ["a0"]
    assert answer["options"]["b"]["count"] == 0
    assert answer["options"]["b"]["mean"] is None
    assert answer["read_after_write"] == "immediate for this synthetic test"
    assert answer["distance"] == "squared_l2"
    assert answer["calibrated"] is False


@pytest.mark.parametrize("missing", ["tag", "search", "read_after_write", "distance"])
def test_registration_rejects_missing_contract_metadata(missing):
    adapter = MemoryAdapter()
    incomplete = SimpleNamespace(**{
        name: getattr(adapter, name) for name in ("tag", "search", "read_after_write", "distance")
        if name != missing
    })
    with pytest.raises(ValueError, match=missing):
        OutcomeService().register_store(incomplete)


@pytest.mark.parametrize("attribute", ["read_after_write", "distance"])
@pytest.mark.parametrize("value", [None, 3, "", "  "])
def test_registration_rejects_invalid_contract_metadata(attribute, value):
    adapter = MemoryAdapter()
    setattr(adapter, attribute, value)
    with pytest.raises(ValueError, match=attribute):
        OutcomeService().register_store(adapter)


@pytest.mark.parametrize("bad_response", [
    None,
    {"a": [[]]},
    {"a": [[]], "b": []},
    {"a": [[], []], "b": [[]]},
    {"a": [None], "b": [[]]},
    {"a": [[{"id": "a", "outcome": 1., "distance": 0.}]*3], "b": [[]]},
    {"a": [[{"id": "a", "outcome": 1.}]], "b": [[]]},
])
def test_malformed_neighbor_shape_fails_as_a_contract_error(bad_response):
    class Bad(MemoryAdapter):
        def search(self, *args, **kwargs):
            return bad_response
    service = OutcomeService()
    handle = service.register_store(Bad())
    with pytest.raises(ValueError, match="adapter returned"):
        service.decide_batch(STATES, _question(handle), rng=np.random.default_rng(1))


@pytest.mark.parametrize("field", ["outcome", "distance"])
@pytest.mark.parametrize("value", [[], [1., 9.], {"nested": 1.}, 10 ** 400])
def test_malformed_evidence_values_fail_as_a_contract_error(field, value):
    case = {"id": "a0", "outcome": 3., "distance": .04, field: value}
    service = OutcomeService()
    handle = service.register_store(MemoryAdapter({"a": [[case]], "b": [[]]}))
    with pytest.raises(ValueError, match="adapter returned"):
        service.decide_batch(STATES, _question(handle))


def test_multi_state_tuple_rows_preserve_evidence_and_support():
    response = {
        "a": ([{"id": "a0", "outcome": np.float32(3.), "distance": np.float64(.04)}], ()),
        "b": ((), ({"id": "b1", "outcome": 5., "distance": .09},)),
    }
    service = OutcomeService()
    handle = service.register_store(MemoryAdapter(response))
    states = [*STATES, {"vector": [0., 1.], "tag": TAG.to_dict()}]
    results = service.decide_batch(states, _question(handle))["results"]
    for result, winner, identifier, mean in zip(results, ["a", "b"], ["a0", "b1"], [3., 5.]):
        answer = result["answers"]["q"]
        assert answer["answer"] == winner
        assert answer["options"][winner]["evidence_ids"] == [identifier]
        assert answer["options"][winner]["mean"] == mean
        other = "b" if winner == "a" else "a"
        assert answer["options"][other]["count"] == 0
        assert answer["options"][other]["mean"] is None
