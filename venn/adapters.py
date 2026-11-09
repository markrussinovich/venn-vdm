"""Adapters that read vectors and nearest cases from FAISS, pgvector, Qdrant, Weaviate and LanceDB."""

import re

import numpy as np

from venn.profiles import VENN_1


def batch(rows):
    """Build a profile-tagged batch dict from ``(id, vector, outcome)`` rows.

    Returns ``ids``, float32 ``vectors``, ``outcomes``, ``profile`` and ``revision``. Raises ``ValueError``
    for wrong dimensions, nonfinite or zero vectors.
    """
    ids, vectors, outcomes = zip(*rows)
    values = np.asarray(vectors, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != VENN_1.dimensions or not np.isfinite(values).all():
        raise ValueError("Adapter rows must contain finite profile-compatible vectors")
    if np.any(np.linalg.norm(values, axis=1) <= 1e-12):
        raise ValueError("Adapter vectors must be nonzero")
    return {"ids": list(ids), "vectors": values, "outcomes": list(outcomes),
            "profile": VENN_1.id, "revision": VENN_1.revision}


def check(profile, revision, batch_size):
    """Raise ``ValueError`` unless the profile pin matches and ``batch_size`` is in 1..65536."""
    VENN_1.check(profile, revision)
    if not isinstance(batch_size, int) or not 1 <= batch_size <= 65536:
        raise ValueError("Batch size must be in 1..65536")


class FaissAdapter:
    """Reads rows from a FAISS index with parallel ``ids`` (and optional ``outcomes``) lists.

    Raises ``ValueError`` if the profile, dimension or row counts disagree.
    """
    def __init__(self, index, ids, outcomes=None, *, profile, revision):
        VENN_1.check(profile, revision)
        if index.d != VENN_1.dimensions or index.ntotal != len(ids):
            raise ValueError("FAISS rows and dimensions must match the pinned profile")
        if outcomes is not None and len(outcomes) != len(ids):
            raise ValueError("Outcomes must match FAISS rows")
        self.index, self.ids, self.outcomes = index, ids, outcomes

    def batches(self, batch_size=1024):
        """Yield profile-tagged batches reconstructed from the index in storage order."""
        check(VENN_1.id, VENN_1.revision, batch_size)
        for start in range(0, self.index.ntotal, batch_size):
            values = self.index.reconstruct_n(start, min(batch_size, self.index.ntotal - start))
            yield batch([(self.ids[start + i], vector,
                          None if self.outcomes is None else self.outcomes[start + i])
                         for i, vector in enumerate(values)])


class PgVectorAdapter:
    """Reads ID, vector and optional outcome columns from a pgvector table through a DB-API cursor.

    Table and column names must be simple SQL identifiers; raises ``ValueError`` otherwise.
    """
    def __init__(self, cursor, table, *, profile, revision, id_column="id",
                 vector_column="embedding", outcome_column=None):
        VENN_1.check(profile, revision)
        def identifier(value):
            if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
                raise ValueError("Use simple SQL identifiers")
            return '"' + value + '"'
        columns = [identifier(id_column), identifier(vector_column)]
        if outcome_column:
            columns.append(identifier(outcome_column))
        self.query = f"SELECT {', '.join(columns)} FROM {identifier(table)} ORDER BY {columns[0]}"
        self.cursor, self.has_outcomes = cursor, bool(outcome_column)

    def batches(self, batch_size=1024):
        """Yield profile-tagged batches ordered by the ID column."""
        check(VENN_1.id, VENN_1.revision, batch_size)
        self.cursor.execute(self.query)
        while rows := self.cursor.fetchmany(batch_size):
            yield batch([(row[0], row[1], row[2] if self.has_outcomes else None) for row in rows])


class QdrantAdapter:
    """Reads points from a Qdrant collection, optionally selecting a named vector and a payload outcome key."""
    def __init__(self, client, collection, *, profile, revision, vector_name=None,
                 outcome_key=None):
        VENN_1.check(profile, revision)
        self.client, self.collection = client, collection
        self.vector_name, self.outcome_key = vector_name, outcome_key

    def batches(self, batch_size=1024):
        """Yield profile-tagged batches by scrolling the collection.

        Raises ``ValueError`` if the named vector is missing or the scroll cursor does not advance.
        """
        check(VENN_1.id, VENN_1.revision, batch_size)
        offset = None
        while True:
            points, following = self.client.scroll(
                collection_name=self.collection, limit=batch_size, offset=offset,
                with_payload=True, with_vectors=True)
            rows = []
            for point in points:
                vector = point.vector
                if isinstance(vector, dict):
                    if self.vector_name not in vector:
                        raise ValueError("Select the pinned named vector")
                    vector = vector[self.vector_name]
                rows.append((point.id, vector, (point.payload or {}).get(self.outcome_key)))
            if rows:
                yield batch(rows)
            if following is None:
                break
            if following == offset:
                raise ValueError("Qdrant cursor did not advance")
            offset = following


class WeaviateAdapter:
    """Reads objects from a Weaviate collection, optionally selecting a named vector and a property outcome key."""
    def __init__(self, collection, *, profile, revision, vector_name="default",
                 outcome_key=None):
        VENN_1.check(profile, revision)
        self.collection, self.vector_name = collection, vector_name
        self.outcome_key = outcome_key

    def batches(self, batch_size=1024):
        """Yield profile-tagged batches of up to ``batch_size`` objects; raises ``ValueError`` if the named vector is missing."""
        check(VENN_1.id, VENN_1.revision, batch_size)
        rows = []
        for item in self.collection.iterator(include_vector=True, cache_size=batch_size):
            vector = item.vector
            if isinstance(vector, dict):
                if self.vector_name not in vector:
                    raise ValueError("Select the pinned named vector")
                vector = vector[self.vector_name]
            rows.append((str(item.uuid), vector, item.properties.get(self.outcome_key)))
            if len(rows) == batch_size:
                yield batch(rows)
                rows = []
        if rows:
            yield batch(rows)


class LanceDBAdapter:
    """Reads ID, vector and optional outcome columns from a LanceDB table."""
    def __init__(self, table, *, profile, revision, id_column="id", vector_column="vector",
                 outcome_column=None):
        VENN_1.check(profile, revision)
        self.table, self.id_column, self.vector_column = table, id_column, vector_column
        self.outcome_column = outcome_column

    def batches(self, batch_size=1024):
        """Yield profile-tagged batches scanned from the table."""
        check(VENN_1.id, VENN_1.revision, batch_size)
        columns = [self.id_column, self.vector_column]
        if self.outcome_column:
            columns.append(self.outcome_column)
        scanner = self.table.search().select(columns).limit(None)
        for record in scanner.to_batches(batch_size=batch_size):
            rows = record.to_pylist()
            if rows:
                yield batch([(row[self.id_column], row[self.vector_column],
                              row.get(self.outcome_column)) for row in rows])


class FaissOutcomeAdapter:
    """Live per-option indexes and parallel app-owned ID/outcome/metadata lists."""

    read_after_write = "immediate in-process; caller serializes reads and writes"
    distance = "squared_l2"

    def __init__(self, partitions, *, tag, query_dtype="float32"):
        from venn.profiles import VectorTag

        self.tag = VectorTag.parse(tag)
        if query_dtype not in ("float32", "float16"):
            raise ValueError("Use float32 or float16 query keys")
        self.partitions, self.query_dtype = partitions, query_dtype

    def export(self):
        """Every stored case per option as ``(vectors, outcomes)`` arrays, for ``OutcomeService.register_family``."""
        exported = {}
        for option, partition in self.partitions.items():
            index, outcomes = partition["index"], partition["outcomes"]
            if index.ntotal != len(partition["ids"]) or len(partition["ids"]) != len(outcomes):
                raise ValueError("Live FAISS rows, IDs and outcomes must agree")
            vectors = index.reconstruct_n(0, index.ntotal) if index.ntotal else np.zeros((0, self.tag.dimensions), np.float32)
            exported[option] = (np.asarray(vectors, dtype=np.float32), np.asarray(outcomes, dtype=np.float32))
        return exported

    def search(self, queries, options, k, *, filters, exact_match=False):
        """Return, per option, one list per query of up to ``k`` nearest cases as ``{"id", "outcome", "distance"}``.

        ``filters`` are equality predicates on app-owned row metadata; ``exact_match`` returns a stored case with
        an identical float16 key at distance 0. Raises ``ValueError`` for inconsistent partitions.
        """
        import faiss
        import hashlib

        if exact_match and self.query_dtype != "float16":
            raise ValueError("Exact-key utility overrides require float16 query keys")
        queries = np.ascontiguousarray(np.asarray(queries, dtype=self.query_dtype), dtype=np.float32)
        result = {}
        for option in options:
            partition = self.partitions.get(option)
            rows = [[] for _ in queries]
            result[option] = rows
            if partition is None:
                continue
            index, ids, outcomes = partition["index"], partition["ids"], partition["outcomes"]
            if (index.d != self.tag.dimensions or index.metric_type != faiss.METRIC_L2
                    or index.ntotal != len(ids)
                    or len(ids) != len(outcomes)):
                raise ValueError("Live FAISS rows, IDs, outcomes and tag dimensions must agree")
            if not index.ntotal:
                continue
            positions = None
            search_index = index
            if filters:
                metadata = partition.get("metadata")
                if metadata is None or len(metadata) != len(ids):
                    raise ValueError("Filtered FAISS reads require app-owned row metadata")
                positions = [i for i, record in enumerate(metadata)
                             if all(record.get(key) == value for key, value in filters.items())]
                if not positions:
                    continue
                search_index = faiss.IndexFlatL2(self.tag.dimensions)
                # Build the eligible exact subset before k-NN, not a post-filtered top-k.
                search_index.add(np.asarray([index.reconstruct(i) for i in positions], dtype=np.float32))
            count = min(k, search_index.ntotal)
            distances, neighbors = search_index.search(queries, count)
            exact_positions = partition.get("exact_positions", {})
            for row, (distance_row, neighbor_row) in enumerate(zip(distances, neighbors)):
                fingerprint = hashlib.sha256(np.asarray(queries[row], dtype=np.float16).tobytes()).digest()
                exact_position = exact_positions.get(fingerprint) if exact_match else None
                if exact_position is not None and (positions is None or exact_position in positions):
                    rows[row] = [{"id": ids[exact_position], "outcome": float(outcomes[exact_position]),
                                  "distance": 0.}]
                    continue
                for distance, position in zip(distance_row, neighbor_row):
                    if position < 0:
                        continue
                    position = int(position) if positions is None else positions[int(position)]
                    rows[row].append({"id": ids[position], "outcome": float(outcomes[position]),
                                      "distance": max(0., float(distance))})
        return result


class QdrantOutcomeAdapter:
    """Per-option nearest-case search over a Qdrant collection with an unnamed Euclidean vector.

    Points carry the option under ``option_key`` and the outcome under ``outcome_key``. Raises
    ``ValueError`` if the collection's vector config does not match ``tag``.
    """
    read_after_write = "visible after the app's completed wait=True upsert"
    distance = "squared_l2"

    def __init__(self, client, collection, *, tag, option_key="option", outcome_key="outcome"):
        from venn.profiles import VectorTag

        self.tag = VectorTag.parse(tag)
        self.client, self.collection = client, collection
        self.option_key, self.outcome_key = option_key, outcome_key
        from qdrant_client import models

        config = client.get_collection(collection).config.params.vectors
        if (not isinstance(config, models.VectorParams) or config.size != self.tag.dimensions
                or config.distance != models.Distance.EUCLID):
            raise ValueError("Outcome collection must use an unnamed tag-sized Euclidean vector")

    def search(self, queries, options, k, *, filters, exact_match=False):
        """Return per-option, per-query lists of ``{"id", "outcome", "distance"}`` from exact filtered search.

        ``exact_match`` is not supported and raises ``ValueError``.
        """
        from qdrant_client import models

        if exact_match:
            raise ValueError("Qdrant does not implement the optional exact-key utility override")
        result = {}
        for option in options:
            conditions = [models.FieldCondition(key=self.option_key, match=models.MatchValue(value=option))]
            for key, value in filters.items():
                if value is None:
                    conditions.append(models.IsNullCondition(is_null=models.PayloadField(key=key)))
                elif isinstance(value, float):
                    conditions.append(models.FieldCondition(key=key, range=models.Range(gte=value, lte=value)))
                else:
                    conditions.append(models.FieldCondition(key=key, match=models.MatchValue(value=value)))
            result[option] = []
            for query in queries:
                response = self.client.query_points(
                    collection_name=self.collection, query=query.tolist(), limit=k,
                    query_filter=models.Filter(must=conditions), with_payload=True,
                    search_params=models.SearchParams(exact=True))
                result[option].append([
                    {"id": point.id, "outcome": float(point.payload[self.outcome_key]),
                     "distance": float(point.score) ** 2} for point in response.points])
        return result


def sql_identifier(value):
    """Return ``value`` double-quoted for SQL; raises ``ValueError`` unless it is a simple identifier."""
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
        raise ValueError("Use simple SQL identifiers")
    return '"' + value + '"'


class PgVectorOutcomeAdapter:
    """Per-option nearest-case search over a pgvector table through a DB-API cursor.

    Table and column names must be simple SQL identifiers.
    """
    read_after_write = "app cursor transaction visibility; snapshot isolation may hide external writes"
    distance = "squared_l2"

    def __init__(self, cursor, table, *, tag, id_column="id", vector_column="embedding",
                 option_column="option", outcome_column="outcome"):
        from venn.profiles import VectorTag

        self.tag = VectorTag.parse(tag)
        self.cursor = cursor
        self.table, self.id, self.vector, self.option, self.outcome = map(
            sql_identifier, (table, id_column, vector_column, option_column, outcome_column))

    def search(self, queries, options, k, *, filters, exact_match=False):
        """Return per-option, per-query lists of ``{"id", "outcome", "distance"}`` (squared L2) via SQL.

        ``exact_match`` is not supported and raises ``ValueError``.
        """
        if exact_match:
            raise ValueError("pgvector does not implement the optional exact-key utility override")
        result = {}
        predicates = "".join(f" AND {sql_identifier(key)} IS NOT DISTINCT FROM %s" for key in filters)
        for option in options:
            result[option] = []
            for query in queries:
                vector = str(query.tolist())
                self.cursor.execute(
                    f"SELECT {self.id}, {self.outcome}, ({self.vector} <-> %s::vector)^2 "
                    f"FROM {self.table} WHERE {self.option} = %s{predicates} "
                    f"ORDER BY {self.vector} <-> %s::vector LIMIT %s",
                    (vector, option, *filters.values(), vector, k))
                result[option].append([
                    {"id": row[0], "outcome": float(row[1]), "distance": float(row[2])}
                    for row in self.cursor.fetchall()])
        return result


class LanceDBOutcomeAdapter:
    """Per-option nearest-case search over a LanceDB table with option and outcome columns."""
    read_after_write = "fresh query on the app's table after completed writes; checkout_latest before reads"
    distance = "squared_l2"

    def __init__(self, table, *, tag, id_column="id", vector_column="vector",
                 option_column="option", outcome_column="outcome"):
        from venn.profiles import VectorTag

        self.tag = VectorTag.parse(tag)
        self.table = table
        for name in (id_column, vector_column, option_column, outcome_column):
            sql_identifier(name)
        self.id, self.vector, self.option, self.outcome = id_column, vector_column, option_column, outcome_column

    def search(self, queries, options, k, *, filters, exact_match=False):
        """Return per-option, per-query lists of ``{"id", "outcome", "distance"}`` from prefiltered L2 search.

        Checks out the latest table version first. ``exact_match`` is not supported and raises ``ValueError``.
        """
        if exact_match:
            raise ValueError("LanceDB does not implement the optional exact-key utility override")

        def predicate(key, value):
            column = sql_identifier(key)
            if value is None:
                return column + " IS NULL"
            literal = "'" + value.replace("'", "''") + "'" if isinstance(value, str) else str(value).lower()
            return column + " = " + literal

        self.table.checkout_latest()
        result = {}
        for option in options:
            where = " AND ".join([predicate(self.option, option)]
                                 + [predicate(key, value) for key, value in filters.items()])
            result[option] = []
            for query in queries:
                rows = (self.table.search(query, vector_column_name=self.vector)
                        .distance_type("l2").where(where, prefilter=True).limit(k).to_list())
                result[option].append([
                    {"id": row[self.id], "outcome": float(row[self.outcome]), "distance": float(row["_distance"])}
                    for row in rows])
        return result


class WeaviateOutcomeAdapter:
    """Per-option nearest-case search over a Weaviate collection; distances use the backend's own metric."""
    read_after_write = "backend replication consistency; no immediate external-write guarantee"
    distance = "backend_distance"

    def __init__(self, collection, *, tag, option_key="option", outcome_key="outcome"):
        from venn.profiles import VectorTag

        self.tag = VectorTag.parse(tag)
        self.collection, self.option_key, self.outcome_key = collection, option_key, outcome_key

    def search(self, queries, options, k, *, filters, exact_match=False):
        """Return per-option, per-query lists of ``{"id", "outcome", "distance"}`` from filtered ``near_vector`` queries.

        ``exact_match`` is not supported and raises ``ValueError``.
        """
        from weaviate.classes.query import Filter, MetadataQuery

        if exact_match:
            raise ValueError("Weaviate does not implement the optional exact-key utility override")
        result = {}
        for option in options:
            condition = Filter.by_property(self.option_key).equal(option)
            for key, value in filters.items():
                condition = condition & (Filter.by_property(key).is_none(True) if value is None
                                         else Filter.by_property(key).equal(value))
            result[option] = []
            for query in queries:
                response = self.collection.query.near_vector(
                    near_vector=query.tolist(), filters=condition, limit=k,
                    return_properties=[self.outcome_key], return_metadata=MetadataQuery(distance=True))
                result[option].append([
                    {"id": str(row.uuid), "outcome": float(row.properties[self.outcome_key]),
                     "distance": float(row.metadata.distance)} for row in response.objects])
        return result
