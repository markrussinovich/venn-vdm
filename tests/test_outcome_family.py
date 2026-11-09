import faiss
import numpy as np
import pytest

from venn import OutcomeService
from venn.adapters import FaissOutcomeAdapter
from venn.heads.outcome import BilinearHead, advantage_weights, chunk_split, fit_outcome_head, memory_policy, records_from_store
from venn.profiles import VectorTag

TAG = VectorTag("experimental:family-test", "0" * 40, "test-8", 8, ("image",))
OPTIONS = ["left", "right", "fire"]


def toy_store(rng, cases=600):
    """Three clusters; the best option differs per cluster, with some stored cases taking the wrong option."""
    centres = rng.normal(size=(3, 8)).astype(np.float32)
    partitions = {option: {"index": faiss.IndexFlatL2(8), "ids": [], "outcomes": [], "exact_positions": {}} for option in OPTIONS}
    truth = {}
    for i in range(cases):
        cluster = i % 3
        vector = (centres[cluster] + .15 * rng.normal(size=8)).astype(np.float16).astype(np.float32)
        option = OPTIONS[cluster] if rng.random() < .7 else OPTIONS[rng.integers(3)]
        outcome = 1. if option == OPTIONS[cluster] else -1.
        part = partitions[option]
        key = np.asarray(vector, np.float16).tobytes()
        import hashlib
        part["exact_positions"][hashlib.sha256(key).digest()] = len(part["ids"])
        part["ids"].append(i)
        part["outcomes"].append(outcome)
        part["index"].add(vector[None])
        truth[i] = cluster
    return FaissOutcomeAdapter(partitions, tag=TAG, query_dtype="float16"), centres


def test_chunk_split_shares_and_records_need_export():
    split = chunk_split(6400, seed=3)
    counts = np.bincount(split, minlength=3) / 6400
    assert abs(counts[0] - .6) < .05 and abs(counts[1] - .2) < .05 and abs(counts[2] - .2) < .05
    assert all(len(set(split[i:i + 64])) == 1 for i in range(0, 6400, 64))

    class NoExport:
        tag = TAG

        def search(self, *args, **kwargs):
            return {}

    with pytest.raises(ValueError, match="export"):
        records_from_store(NoExport(), OPTIONS)


def test_memory_policy_matches_the_api_answer():
    rng = np.random.default_rng(0)
    adapter, centres = toy_store(rng)
    service = OutcomeService(head_device="cpu")
    handle = service.register_store(adapter)
    queries = np.concatenate([centres, rng.normal(size=(5, 8))]).astype(np.float16)
    greedy, utilities, baseline = memory_policy(adapter, queries, OPTIONS, k=5, exact_match=True, unsupported_value=0.)
    states = [{"vector": q.astype(np.float32), "tag": TAG.to_dict()} for q in queries]
    question = {"q": {"type": "nearest_outcome", "store": handle, "options": OPTIONS, "k": 5, "exact_match": True, "unsupported_value": 0.}}
    api = service.decide_batch(states, question, rng=np.random.default_rng(0))
    for row, result in enumerate(api["results"]):
        answer = result["answers"]["q"]
        assert OPTIONS[greedy[row]] == answer["answer"]
        np.testing.assert_allclose(utilities[row], [answer["options"][o]["mean"] for o in OPTIONS], rtol=1e-5)
    assert np.isfinite(baseline).all()


def test_minimizing_awr_weights_favor_lower_costs_and_reach_registration(monkeypatch):
    rewards = np.array([-1., 1.])
    maximize, tau = advantage_weights(rewards, np.zeros(2), np.ones(2, bool))
    minimize, cost_tau = advantage_weights(rewards, np.zeros(2), np.ones(2, bool), goal="minimize")
    assert maximize[1] > maximize[0] and minimize[0] > minimize[1]
    np.testing.assert_array_equal(maximize, minimize[::-1])
    assert tau == cost_tau

    from venn.heads import outcome

    adapter, _ = toy_store(np.random.default_rng(1))
    service = OutcomeService(head_device="cpu")
    store = service.register_store(adapter)
    real_weights = outcome.advantage_weights
    calls = []

    def weights(*args, **kwargs):
        calls.append(kwargs["goal"])
        return real_weights(*args, **kwargs)

    monkeypatch.setattr(outcome, "advantage_weights", weights)
    service.register_family("cost", from_store=store, options=OPTIONS, objective="awr",
                            goal="minimize", fit={"max_epochs": 1})
    assert calls == ["minimize"]


def test_small_option_chunks_still_have_all_fitting_slices():
    class Export:
        def export(self):
            return {str(i): (np.random.default_rng(i).normal(size=(80, 8)), np.ones(80))
                    for i in range(4)}

    records = records_from_store(Export(), [str(i) for i in range(4)], seed=0)
    assert len(records["x"]) == 320
    assert set(records["split"]) == {0, 1, 2}
    for start in range(0, 320, 80):
        assert len(set(records["split"][start:start + 64])) == 1
        assert len(set(records["split"][start + 64:start + 80])) == 1
    head, report = fit_outcome_head(records, records["action"], np.eye(4), rank=2, max_epochs=2)
    assert all(report["records"][name] > 0 for name in ("train", "early_stop", "report"))
    assert report["early_stop_history"][0] > 0
    assert np.isfinite(report["losses"]["early_stop"])
    for split, epochs in ((np.zeros(320), 1), (records["split"], 0)):
        with pytest.raises(ValueError, match="nonempty"):
            fit_outcome_head({**records, "split": split}, records["action"], np.eye(4), max_epochs=epochs)


@pytest.mark.parametrize("objective", ["bc", "awr"])
def test_register_family_answers_without_a_lookup_and_round_trips(tmp_path, objective):
    rng = np.random.default_rng(1)
    adapter, centres = toy_store(rng)
    service = OutcomeService(head_device="cpu")
    handle = service.register_store(adapter)
    summary = service.register_family("policy", from_store=handle, options=OPTIONS, objective=objective, k=5,
                                      exact_match=True, unsupported_value=0., seed=0, fit={"max_epochs": 40, "batch": 128, "lr": 1e-2})
    assert summary["parameters"] == 8 * 16 + 3 * 16 + 1 and summary["cases"] == 600 and summary["objective"] == objective
    report = summary["report"]
    assert report["records"]["train"] + report["records"]["early_stop"] + report["records"]["report"] == 600
    assert report["labels"]["objective"] == objective and (report["labels"]["tau"] is None) == (objective == "bc")
    # The head recovers the cluster structure: at the cluster centres it prefers the cluster's option.
    states = [{"vector": c, "tag": TAG.to_dict()} for c in centres]
    answers = service.decide_batch(states, {"move": {"type": "nearest_outcome", "family": "policy"}}, rng=np.random.default_rng(0))
    picks = [r["answers"]["move"]["answer"] for r in answers["results"]]
    assert picks == OPTIONS
    first = answers["results"][0]["answers"]["move"]
    assert first["options"] is None and first["family"] == "policy" and first["objective"] == objective
    assert len(first["head"]) == 3 and first["exploratory"] is False and first["goal"] == "maximize"
    assert adapter.export()["left"][0].shape[1] == 8  # the store is untouched
    # Exploration still applies and is flagged.
    explored = service.decide_batch(states, {"move": {"type": "nearest_outcome", "family": "policy"}},
                                    rng=np.random.default_rng(0), epsilon=[1., 1., 1.])
    assert all(r["answers"]["move"]["exploratory"] for r in explored["results"])
    # Save, load into a fresh service, identical scores.
    path = tmp_path / "policy.pt"
    sha = service.save_family("policy", path)
    fresh = OutcomeService(head_device="cpu")
    assert fresh.load_family("policy", path) == sha == summary["sha256"]
    np.testing.assert_allclose(fresh.head_values("policy", centres), service.head_values("policy", centres), rtol=1e-6)
    assert fresh.families["policy"]["memory_question"] == {"k": 5, "exact_match": True, "unsupported_value": 0.}
    # Warm start continues from the registered weights.
    again = service.register_family("policy2", from_store=handle, options=OPTIONS, objective=objective, k=5, exact_match=True,
                                    unsupported_value=0., warm_start="policy", fit={"max_epochs": 1, "batch": 128})
    assert again["report"]["warm_start"] is True


def test_family_validation_fails_closed():
    rng = np.random.default_rng(2)
    adapter, centres = toy_store(rng)
    service = OutcomeService(head_device="cpu")
    handle = service.register_store(adapter)
    with pytest.raises(ValueError, match="Unknown app-owned outcome store"):
        service.register_family("p", from_store="missing", options=OPTIONS)
    with pytest.raises(ValueError, match="Invalid family"):
        service.register_family("p", from_store=handle, options=OPTIONS, objective="huber")
    with pytest.raises(ValueError, match="option vector"):
        service.register_family("p", from_store=handle, options=OPTIONS, option_vectors=np.ones((2, 4)))
    with pytest.raises(ValueError, match="warm_start"):
        service.register_family("p", from_store=handle, options=OPTIONS, warm_start="nothing")
    states = [{"vector": centres[0], "tag": TAG.to_dict()}]
    with pytest.raises(ValueError, match="Unknown registered family"):
        service.decide_batch(states, {"q": {"type": "nearest_outcome", "family": "p"}})
    service.register_family("p", from_store=handle, options=OPTIONS, fit={"max_epochs": 1, "batch": 128})
    with pytest.raises(ValueError, match="options must match"):
        service.decide_batch(states, {"q": {"type": "nearest_outcome", "family": "p", "options": ["left", "right"]}})
    with pytest.raises(ValueError, match="goal"):
        service.decide_batch(states, {"q": {"type": "nearest_outcome", "family": "p", "goal": "minimize"}})
    other = VectorTag("experimental:family-test", "1" * 40, "test-8", 8, ("image",))
    with pytest.raises(ValueError, match="mismatch"):
        service.decide_batch([{"vector": centres[0], "tag": other.to_dict()}], {"q": {"type": "nearest_outcome", "family": "p"}})
    small = {o: {"index": faiss.IndexFlatL2(8), "ids": [], "outcomes": [], "exact_positions": {}} for o in OPTIONS}
    tiny = service.register_store(FaissOutcomeAdapter(small, tag=TAG, query_dtype="float16"))
    with pytest.raises(ValueError, match="at least"):
        service.register_family("tiny", from_store=tiny, options=OPTIONS)


def test_bilinear_head_state_dict_layout_is_stable():
    head = BilinearHead(2304, 768, 16)
    assert sum(p.numel() for p in head.parameters()) == 49153
    assert set(head.state_dict()) == {"left.weight", "right.weight", "bias"}
    assert sum(p.numel() for p in BilinearHead(2304, 3, 16).parameters()) == 36913
