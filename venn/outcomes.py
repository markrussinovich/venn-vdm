"""Decisions from fresh app-owned cases, and small heads compiled from them."""

import time
import uuid

import numpy as np

from venn.profiles import IMAGE_TAGS, VectorTag

OBJECTIVES = ("bc", "awr")


class OutcomeService:
    """Answers ``nearest_outcome`` questions by comparing option outcomes among a state's nearest stored cases."""
    def __init__(self, image_reader=None, device="cuda:0", head_device=None):
        """``device`` is used only to construct the image reader lazily for raw uint8 image batches; tagged
        vector states run on CPU without it. Pass ``image_reader`` to reuse an existing reader. ``head_device``
        is where registered families are fitted and served (CUDA when available, else CPU).
        """
        self.stores, self.families = {}, {}
        self.image_reader, self.device = image_reader, device
        self._head_device = head_device

    @property
    def head_device(self):
        if self._head_device is None:
            import torch

            self._head_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        return self._head_device

    @head_device.setter
    def head_device(self, value):
        self._head_device = value

    def register_store(self, adapter):
        """Register an outcome store adapter and return its handle.

        The adapter needs a ``tag``, a ``search`` method, ``read_after_write`` and ``distance``; raises
        ``ValueError`` if it cannot search.
        """
        VectorTag.parse(adapter.tag)
        if not callable(getattr(adapter, "search", None)):
            raise ValueError("An outcome adapter must implement filtered fresh searches")
        handle = uuid.uuid4().hex
        self.stores[handle] = adapter
        return handle

    def register_family(self, name, *, from_store, options, option_vectors=None, objective="awr", k=11,
                        exact_match=False, unsupported_value=None, goal="maximize", seed=0, warm_start=None,
                        rank=16, fit=None):
        """Compile a store into a small head that answers ``nearest_outcome`` questions without a lookup.

        The head is trained from every case the store exports (the adapter needs ``export()``): ``bc``
        imitates the option the store itself returns for each stored state under ``k``, ``exact_match``,
        ``unsupported_value`` and ``goal``; ``awr`` imitates the stored option with advantage weights.
        ``option_vectors`` (one row per option, for example text embeddings of the option names) default to
        one-hot vectors. ``warm_start`` names a registered family to continue from. Returns a summary with
        the family's ``sha256``, parameter count, case count and fitting report. Raises ``ValueError`` for an
        unknown store, an adapter without ``export``, too few cases, or invalid options and settings.
        """
        from venn.heads import outcome as heads

        started = time.perf_counter()
        adapter = self.stores.get(from_store) if isinstance(from_store, str) else None
        if adapter is None:
            raise ValueError("Unknown app-owned outcome store")
        if (not isinstance(name, str) or not name or not isinstance(options, (list, tuple)) or not 1 <= len(options) <= 256
                or not all(isinstance(option, str) and option and option != "none" for option in options)
                or len(set(options)) != len(options) or objective not in OBJECTIVES or type(k) is not int
                or not 1 <= k <= 1024 or type(exact_match) is not bool or goal not in ("maximize", "minimize")
                or type(rank) is not int or not 1 <= rank <= 256):
            raise ValueError("Invalid family name, options, objective, k, exact_match, goal or rank")
        if unsupported_value is not None and (isinstance(unsupported_value, bool)
                                              or not isinstance(unsupported_value, (int, float)) or not np.isfinite(unsupported_value)):
            raise ValueError("Unsupported utility prior must be finite")
        tag = VectorTag.parse(adapter.tag)
        options = list(options)
        if option_vectors is None:
            option_vectors = np.eye(len(options), dtype=np.float32)
        option_vectors = np.asarray(option_vectors, dtype=np.float32)
        if option_vectors.shape[0] != len(options) or option_vectors.ndim != 2 or not np.isfinite(option_vectors).all():
            raise ValueError("Provide one finite option vector per option")
        warm = None
        if warm_start is not None:
            previous = self.families.get(warm_start)
            if previous is None or previous["options"] != options or previous["option_vectors"].shape != option_vectors.shape \
                    or previous["rank"] != rank:
                raise ValueError("warm_start must name a registered family with the same options, option vectors and rank")
            warm = previous["head"].state_dict()
        records = heads.records_from_store(adapter, options, seed)
        if records["x"].shape[1] != tag.dimensions:
            raise ValueError("Exported cases do not match the store's representation tag")
        greedy, _, baseline = heads.memory_policy(adapter, records["x"], options, k=k, exact_match=exact_match,
                                                  unsupported_value=unsupported_value, goal=goal)
        labels_seconds = time.perf_counter() - started
        tau = None
        if objective == "bc":
            labels, weights = greedy, None
        else:
            labels = records["action"]
            weights, tau = heads.advantage_weights(records["outcome"], baseline, records["split"] == 0, goal=goal)
        head, report = heads.fit_outcome_head(records, labels, option_vectors, weights=weights, device=self.head_device,
                                              seed=seed, rank=rank, warm_start=warm, **(fit or {}))
        report["labels"] = {"objective": objective, "seconds": labels_seconds, "tau": tau,
                            "store_greedy_agreement_with_stored_option": float((greedy == records["action"]).mean()),
                            "store_greedy_counts": {option: int((greedy == i).sum()) for i, option in enumerate(options)}}
        family = {"head": head, "options": options, "option_vectors": option_vectors, "tag": tag.to_dict(),
                  "objective": objective, "goal": goal, "rank": rank, "sha256": heads.digest(head), "report": report,
                  "memory_question": {"k": k, "exact_match": exact_match, "unsupported_value": unsupported_value}}
        self.families[name] = family
        return {"name": name, "sha256": family["sha256"], "objective": objective, "parameters": report["parameters"],
                "cases": report["records"]["total"], "seconds": time.perf_counter() - started, "report": report}

    def save_family(self, name, path):
        """Write a registered family to ``path`` (a ``.pt`` file) and return its ``sha256``."""
        from venn.heads import outcome as heads

        if name not in self.families:
            raise ValueError("Unknown registered family")
        heads.save_family(path, self.families[name])
        return self.families[name]["sha256"]

    def load_family(self, name, path):
        """Register a family saved with ``save_family`` under ``name``; returns its ``sha256``."""
        from venn.heads import outcome as heads

        if not isinstance(name, str) or not name:
            raise ValueError("Invalid family name")
        self.families[name] = heads.load_family(path, self.head_device)
        return self.families[name]["sha256"]

    def head_values(self, name, vectors):
        """Head scores per option for tagged vectors already in the family's representation."""
        from venn.heads import outcome as heads

        family = self.families.get(name)
        if family is None:
            raise ValueError("Unknown registered family")
        return heads.predict(family["head"], np.asarray(vectors, dtype=np.float32), family["option_vectors"], self.head_device)

    def decide_batch(self, states, questions, *, representation="image-thirds-70",
                     rng=None, epsilon=None):
        """Answer 1..32 named ``nearest_outcome`` questions for 1..128 states.

        ``states`` is a uint8 RGB image batch or a list of ``{"vector", "tag"}`` dicts; ``representation`` applies
        only to uint8 image batches, tagged vector states carry their own tag. Each answer reports per-option
        outcome statistics, the best supported option (or "none") and an ``action`` that explores with
        probability ``epsilon[row]``. Raises ``ValueError`` for invalid input.
        """
        if not isinstance(questions, dict) or not 1 <= len(questions) <= 32:
            raise ValueError("Provide 1..32 named nearest_outcome questions")
        if isinstance(states, np.ndarray) and states.dtype == np.uint8:
            if representation not in IMAGE_TAGS:
                raise ValueError("Unknown image representation")
            if (states.ndim != 4 or states.shape[-1] != 3 or not 1 <= len(states) <= 128
                    or min(states.shape[1:3]) < 3 or states.shape[1] * states.shape[2] > 4000000
                    or states.nbytes > 64000000):
                raise ValueError("Invalid uint8 RGB image batch")
            if self.image_reader is None:
                from venn.images import ImageBatchReader

                self.image_reader = ImageBatchReader(self.device)
            values = self.image_reader.encode_frames(states, representation)
            tag = IMAGE_TAGS[representation]
        else:
            if not isinstance(states, (list, tuple)) or not 1 <= len(states) <= 128:
                raise ValueError("Provide 1..128 tagged states or a uint8 image batch")
            try:
                tag = VectorTag.parse(states[0]["tag"])
                for state in states:
                    tag.check(state["tag"])
                values = np.asarray([state["vector"] for state in states], dtype=np.float32)
            except (KeyError, TypeError) as error:
                raise ValueError("Every vector needs its complete representation tag") from error
        values = np.asarray(values, dtype=np.float32)
        if (values.ndim != 2 or not 1 <= len(values) <= 128
                or values.shape[1] != tag.dimensions or not np.isfinite(values).all()
                or np.any(np.linalg.norm(values, axis=1) <= 1e-12)):
            raise ValueError("Invalid vectors for the representation tag")
        epsilon = np.zeros(len(values)) if epsilon is None else np.asarray(epsilon, dtype=float)
        if epsilon.shape != (len(values),) or not np.isfinite(epsilon).all() or np.any((epsilon < 0) | (epsilon > 1)):
            raise ValueError("Provide one exploration probability in [0,1] per state")
        rng = np.random.default_rng() if rng is None else rng
        results = [{"answers": {}, "vector": vector, "vector_tag": tag.to_dict()} for vector in values]
        for name, question in questions.items():
            if not isinstance(name, str) or not name or not isinstance(question, dict) or question.get("type") != "nearest_outcome":
                raise ValueError("Use named nearest_outcome questions")
            if question.get("family") is not None:
                self._answer_from_family(name, question, values, tag, results, rng, epsilon)
                continue
            handle = question.get("store")
            adapter = self.stores.get(handle) if isinstance(handle, str) else None
            if adapter is None:
                raise ValueError("Unknown app-owned outcome store")
            tag.check(adapter.tag)
            options = question.get("options")
            k, minimum = question.get("k", 11), question.get("min_support", 1)
            goal = question.get("goal", "maximize")
            if (not isinstance(options, list) or not 1 <= len(options) <= 256
                    or not all(isinstance(option, str) and option and option != "none" for option in options)
                    or len(set(options)) != len(options) or type(k) is not int or not 1 <= k <= 1024
                    or type(minimum) is not int or not 1 <= minimum <= k
                    or goal not in ("maximize", "minimize")):
                raise ValueError("Invalid options, k, min_support or goal")
            filters = question.get("filters", {})
            if (not isinstance(filters, dict) or len(filters) > 32
                    or not all(isinstance(key, str) and key and isinstance(value, (str, int, float, bool, type(None)))
                               and (not isinstance(value, float) or np.isfinite(value)) for key, value in filters.items())):
                raise ValueError("Filters must be finite scalar equality predicates")
            exact = question.get("exact_match", False)
            if type(exact) is not bool:
                raise ValueError("exact_match must be boolean")
            neighbors = adapter.search(values, options, k, filters=filters, exact_match=exact)
            for row, result in enumerate(results):
                stats, utilities = {}, []
                prior = question.get("unsupported_value")
                if prior is not None and (isinstance(prior, bool) or not isinstance(prior, (int, float)) or not np.isfinite(prior)):
                    raise ValueError("Unsupported utility prior must be finite")
                for option in options:
                    evidence = neighbors[option][row]
                    outcomes = np.asarray([case["outcome"] for case in evidence], dtype=float)
                    distances = np.asarray([case["distance"] for case in evidence], dtype=float)
                    if not np.isfinite(outcomes).all() or not np.isfinite(distances).all():
                        raise ValueError("Store returned nonfinite evidence")
                    supported = len(outcomes) >= minimum
                    mean = float(outcomes.mean()) if len(outcomes) else None
                    stats[option] = {
                        "mean": mean, "count": len(outcomes),
                        "spread": float(outcomes.std()) if len(outcomes) else None,
                        "mean_distance": float(distances.mean()) if len(distances) else None,
                        "evidence_ids": [case["id"] for case in evidence], "supported": supported,
                    }
                    utilities.append(mean if supported else prior)
                eligible = [i for i, option in enumerate(options) if stats[option]["supported"]]
                sign = 1 if goal == "maximize" else -1
                best = max((sign * stats[options[i]]["mean"] for i in eligible), default=None)
                winners = [i for i in eligible if sign * stats[options[i]]["mean"] == best]
                answer = options[winners[0]] if winners else "none"
                # The optional app policy preserves a utility prior for unseen options,
                # without presenting that prior as a measured outcome.
                scores = np.asarray([sign * utility if utility is not None else -np.inf for utility in utilities])
                explore = bool(rng.random() < epsilon[row])
                if explore:
                    action = int(rng.integers(len(options)))
                elif np.isfinite(scores).any():
                    action = int(rng.choice(np.flatnonzero(scores == scores.max())))
                else:
                    action = int(rng.choice(len(options)))
                result["answers"][name] = {
                    "answer": answer, "action": options[action], "options": stats,
                    "probabilities": None, "confidence": None, "calibrated": False,
                    "goal": goal, "exploratory": explore or not stats[options[action]]["supported"],
                    "read_after_write": adapter.read_after_write,
                    "distance": adapter.distance,
                }
        return {"results": results, "vector_tag": tag.to_dict()}

    def _answer_from_family(self, name, question, values, tag, results, rng, epsilon):
        """Answer from a registered head alone; the store is not consulted."""
        family = self.families.get(question["family"]) if isinstance(question["family"], str) else None
        if family is None:
            raise ValueError("Unknown registered family")
        tag.check(family["tag"])
        options = question.get("options", family["options"])
        if list(options) != family["options"]:
            raise ValueError("Family options must match the registered options")
        if question.get("goal", family["goal"]) != family["goal"]:
            raise ValueError("A family answers under the goal it was registered with")
        scores = self.head_values(question["family"], values)
        for row, result in enumerate(results):
            preference = scores[row]
            best = int(np.argmax(preference))
            explore = bool(rng.random() < epsilon[row])
            action = int(rng.integers(len(options))) if explore else int(rng.choice(np.flatnonzero(preference == preference.max())))
            result["answers"][name] = {
                "answer": options[best], "action": options[action], "options": None,
                "family": question["family"], "objective": family["objective"], "head": [float(v) for v in preference],
                "probabilities": None, "confidence": None, "calibrated": False,
                "goal": family["goal"], "exploratory": explore, "read_after_write": None, "distance": None,
            }
