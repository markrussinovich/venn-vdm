import json

import numpy as np
import pytest

from venn import OutcomeService
from venn.adapters import AzureSearchAdapter, AzureSearchOutcomeAdapter
from venn.profiles import VENN_1, VectorTag


TAG = VectorTag(VENN_1.id, VENN_1.revision, "test-projection", 2)


@pytest.fixture
def azure_search():
    search = pytest.importorskip("azure.search.documents")
    from azure.core.credentials import AzureKeyCredential
    from azure.core.pipeline.transport import HttpResponse, HttpTransport

    class Response(HttpResponse):
        def __init__(self, request, rows):
            super().__init__(request, None)
            self.status_code = 200
            self.headers = {"Content-Type": "application/json"}
            self.content_type = "application/json"
            self.data = json.dumps({"value": rows}).encode()

        def body(self):
            return self.data

        def json(self):
            return json.loads(self.data)

    class Transport(HttpTransport):
        def __init__(self):
            self.requests = []
            self.responses = []

        def open(self):
            pass

        def close(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

        def send(self, request, **kwargs):
            self.requests.append(json.loads(request.body))
            return Response(request, self.responses.pop(0))

    transport = Transport()
    with search.SearchClient("https://example.search.windows.net", "cases",
                             AzureKeyCredential("test-key"), transport=transport,
                             retry_total=0) as client:
        yield client, transport


def test_azure_scan_keyset_pagination_and_pinned_batches(azure_search):
    client, transport = azure_search
    vectors = np.eye(5, VENN_1.dimensions, dtype=np.float32)
    rows = [{"key": str(i), "vector": vector.tolist(), "label": i % 2}
            for i, vector in enumerate(vectors)]
    transport.responses.extend([rows[:3], rows[3:], []])
    adapter = AzureSearchAdapter(client, profile=VENN_1.id, revision=VENN_1.revision,
                                 id_field="key", vector_field="vector", outcome_field="label")
    batches = list(adapter.batches(3))
    assert [row["ids"] for row in batches] == [["0", "1", "2"], ["3", "4"]]
    assert [value for row in batches for value in row["outcomes"]] == [0, 1, 0, 1, 0]
    np.testing.assert_array_equal(np.concatenate([row["vectors"] for row in batches]), vectors)
    assert all(row["vectors"].dtype == np.float32 and row["profile"] == VENN_1.id
               and row["revision"] == VENN_1.revision for row in batches)
    assert [request.get("filter") for request in transport.requests] == [
        None, "key gt '2'", "key gt '4'"]
    assert all(request["top"] == 3 and request["orderby"] == "key asc"
               and request["select"] == "key,vector,label" and "skip" not in request
               for request in transport.requests)


def test_azure_scan_empty_page_limit_and_validation(azure_search):
    client, transport = azure_search
    transport.responses.append([])
    pin = {"profile": VENN_1.id, "revision": VENN_1.revision}
    adapter = AzureSearchAdapter(client, **pin)
    assert list(adapter.batches(65536)) == []
    assert transport.requests[0]["top"] == 1000
    assert transport.requests[0]["select"] == "id,embedding"
    with pytest.raises(ValueError, match="Batch size"):
        list(adapter.batches(0))
    with pytest.raises(ValueError, match="Profile mismatch"):
        AzureSearchAdapter(client, profile="other", revision=VENN_1.revision)
    with pytest.raises(ValueError, match="field names"):
        AzureSearchAdapter(client, id_field="id desc", **pin)
    assert len(transport.requests) == 1


def test_azure_scan_without_outcomes_and_stalled_cursor(azure_search):
    client, transport = azure_search
    row = {"id": "a", "embedding": np.ones(VENN_1.dimensions).tolist()}
    transport.responses.extend([[row], [row]])
    batches = AzureSearchAdapter(client, profile=VENN_1.id, revision=VENN_1.revision).batches()
    assert next(batches)["outcomes"] == [None]
    with pytest.raises(ValueError, match="cursor did not advance"):
        next(batches)


@pytest.mark.parametrize("vector", [[1., 0.], [0.] * 768, [float("nan")] * 768])
def test_azure_scan_rejects_incompatible_vectors(azure_search, vector):
    client, transport = azure_search
    transport.responses.append([{"id": "a", "embedding": vector}])
    with pytest.raises(ValueError, match="vectors"):
        list(AzureSearchAdapter(client, profile=VENN_1.id, revision=VENN_1.revision).batches())


@pytest.mark.parametrize("exhaustive", [False, True])
def test_azure_outcome_real_sdk_prefilters_in_indexed_and_exhaustive_modes(azure_search, exhaustive):
    client, transport = azure_search
    transport.responses.extend([
        [{"key": "case", "utility": 4., "@search.score": .8}], [], [], []])
    settings = {"exhaustive": True} if exhaustive else {}
    adapter = AzureSearchOutcomeAdapter(client, tag=TAG, id_field="key", vector_field="vector",
                                        option_field="action", outcome_field="utility", **settings)
    queries = np.asarray([[1., 0.], [0., 1.]])
    result = adapter.search(queries, ["a", "b"], 2, filters={
        "tenant": "O'Brien' or true", "active": True, "missing": None, "count": 3, "rating": 1.5})
    assert result["a"][0][0]["id"] == "case"
    assert result["a"][0][0]["outcome"] == 4.
    assert result["a"][0][0]["distance"] == pytest.approx(.25)
    assert result["a"][1] == [] and result["b"] == [[], []]
    assert len(transport.requests) == 4
    for i, request in enumerate(transport.requests):
        assert not request.get("search")
        assert request["vectorFilterMode"] == "preFilter" and request["top"] == 2
        assert request["select"] == "key,utility"
        assert request["filter"] == (
            f"action eq '{'a' if i < 2 else 'b'}' and tenant eq 'O''Brien'' or true'"
            " and active eq true and missing eq null and count eq 3 and rating eq 1.5")
        vector = request["vectorQueries"][0]
        assert vector == {"kind": "vector", "vector": queries[i % 2].tolist(),
                          "fields": "vector", "k": 2, "exhaustive": exhaustive}


def test_azure_outcome_converts_scores_to_distances(azure_search):
    client, transport = azure_search
    transport.responses.append([{"id": "case", "outcome": 3., "@search.score": 2 / 3}])
    adapter = AzureSearchOutcomeAdapter(client, tag=TAG)
    rows = adapter.search(np.asarray([[1., 0.]]), ["a"], 1, filters={})["a"][0]
    assert rows[0]["distance"] == pytest.approx(.5)


def test_azure_outcome_service_uses_fresh_queries_for_newly_visible_cases(azure_search):
    client, transport = azure_search
    transport.responses.extend([
        [{"id": "first", "outcome": 2., "@search.score": 1.}], [],
        [{"id": "first", "outcome": 2., "@search.score": 1.}],
        [{"id": "visible-after-indexing", "outcome": 5., "@search.score": .5}]])
    service = OutcomeService()
    handle = service.register_store(AzureSearchOutcomeAdapter(client, tag=TAG))
    questions = {"q": {"type": "nearest_outcome", "store": handle, "options": ["a", "b"],
                       "k": 1, "filters": {"tenant": "wanted"}}}
    state = {"vector": [1., 0.], "tag": TAG.to_dict()}
    first = service.decide_batch([state], questions)["results"][0]["answers"]["q"]
    following = service.decide_batch([state], questions)["results"][0]["answers"]["q"]
    assert first["answer"] == "a" and following["answer"] == "b"
    assert following["options"]["b"]["evidence_ids"] == ["visible-after-indexing"]
    assert all("tenant eq 'wanted'" in request["filter"] for request in transport.requests)


@pytest.mark.parametrize("score", [0., -.1, 1.1, float("nan")])
def test_azure_outcome_rejects_invalid_scores(azure_search, score):
    client, transport = azure_search
    transport.responses.append([{"id": "case", "outcome": 3., "@search.score": score}])
    with pytest.raises(ValueError, match="invalid vector score"):
        AzureSearchOutcomeAdapter(client, tag=TAG).search(
            np.asarray([[1., 0.]]), ["a"], 1, filters={})


def test_azure_outcome_rejects_unsupported_or_unsafe_queries(azure_search):
    client, transport = azure_search
    adapter = AzureSearchOutcomeAdapter(client, tag=TAG)
    queries = np.asarray([[1., 0.]])
    with pytest.raises(ValueError, match="exact-key"):
        adapter.search(queries, ["a"], 1, filters={}, exact_match=True)
    with pytest.raises(ValueError, match="field names"):
        adapter.search(queries, ["a"], 1, filters={"tenant eq 'x' or true": "a"})
    with pytest.raises(ValueError, match="finite scalar"):
        adapter.search(queries, ["a"], 1, filters={"tenant": float("inf")})
    with pytest.raises(ValueError, match="field names"):
        AzureSearchOutcomeAdapter(client, tag=TAG, vector_field="a,b")
    with pytest.raises(ValueError, match="boolean"):
        AzureSearchOutcomeAdapter(client, tag=TAG, exhaustive="false")
    assert transport.requests == []
