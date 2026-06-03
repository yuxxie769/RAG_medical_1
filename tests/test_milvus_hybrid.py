import pytest
from pymilvus import FunctionType

from app.retrieval.hybrid import HybridRetriever
from app.retrieval.milvus_store import MilvusCollectionSchema, MilvusOperationError, MilvusRecord, MilvusStore
from scripts.reset_milvus_collection import reset_collection


class FakeMilvusClient:
    def __init__(self):
        self.inserted_rows = []
        self.persisted_ids = set()
        self.search_calls = []
        self.hybrid_search_calls = []
        self.dropped = []
        self.flushed = []

    def query(self, **kwargs):
        return [{"doc_id": doc_id} for doc_id in self.persisted_ids]

    def insert(self, **kwargs):
        self.inserted_rows.extend(kwargs["data"])
        self.persisted_ids.update(row["doc_id"] for row in kwargs["data"])

    def flush(self, **kwargs):
        self.flushed.append(kwargs)

    def search(self, **kwargs):
        self.search_calls.append(kwargs)
        return [[{"id": "doc_1", "distance": 0.8, "entity": {"content": "answer"}}]]

    def hybrid_search(self, **kwargs):
        self.hybrid_search_calls.append(kwargs)
        return [[{"id": "doc_1", "distance": 0.5, "entity": {"content": "answer"}}]]


class FakeEmbedder:
    def __init__(self):
        self.calls = []

    def embed_texts(self, texts):
        self.calls.append(texts)
        return [[0.1, 0.2]]


def make_store():
    store = MilvusStore(MilvusCollectionSchema(collection_name="test_documents", dimension=2))
    store.client = FakeMilvusClient()
    store.create_collection = lambda: None
    return store


def test_schema_contains_cosine_dense_index_and_bm25_function():
    store = make_store()
    schema = store._build_schema().to_dict()
    indexes = [index.to_dict() for index in store._build_index_params()]

    fields = {field["name"]: field for field in schema["fields"]}
    assert fields["sparse_embedding"]["type"].name == "SPARSE_FLOAT_VECTOR"
    assert fields["content"]["params"]["enable_analyzer"] is True
    assert '"type":"chinese"' in fields["content"]["params"]["analyzer_params"]
    assert schema["functions"][0]["type"] == FunctionType.BM25
    assert schema["functions"][0]["output_field_names"] == ["sparse_embedding"]
    assert indexes[0]["metric_type"] == "COSINE"
    assert indexes[1]["metric_type"] == "BM25"


def test_upsert_leaves_sparse_vector_generation_to_milvus():
    store = make_store()
    record = MilvusRecord(
        doc_id="doc_1",
        vector=[0.1, 0.2],
        content="口干怎么办",
        question="口干怎么办",
        answer="补充水分",
        metadata={},
    )

    store.upsert([record])

    assert len(store.client.inserted_rows) == 1
    assert "embedding" in store.client.inserted_rows[0]
    assert "sparse_embedding" not in store.client.inserted_rows[0]
    assert store.client.flushed == []


def test_upsert_skips_existing_documents_and_inserts_only_missing_documents():
    store = make_store()
    store.client.persisted_ids.add("doc_1")
    records = [
        MilvusRecord(doc_id="doc_1", vector=[0.1, 0.2], content="A", question="Q", answer="A", metadata={}),
        MilvusRecord(doc_id="doc_2", vector=[0.2, 0.3], content="B", question="Q", answer="B", metadata={}),
    ]

    result = store.upsert(records)

    assert result.existing_count == 1
    assert result.inserted_count == 1
    assert [row["doc_id"] for row in store.client.inserted_rows] == ["doc_2"]


def test_hybrid_search_uses_two_requests_and_rrf_ranker():
    store = make_store()

    hits = store.search(
        query_vector=[0.1, 0.2],
        query_text="口干怎么办",
        search_method="hybrid",
    )

    call = store.client.hybrid_search_calls[0]
    assert len(call["reqs"]) == 2
    assert call["reqs"][0].anns_field == "embedding"
    assert call["reqs"][0].param["metric_type"] == "COSINE"
    assert call["reqs"][1].anns_field == "sparse_embedding"
    assert call["reqs"][1].param["metric_type"] == "BM25"
    assert call["ranker"].dict()["params"]["k"] == 60
    assert call["timeout"] == store.request_timeout_seconds
    assert hits[0].doc_id == "doc_1"


@pytest.mark.parametrize(
    ("search_method", "query_vector", "query_text", "expected_field", "expected_metric"),
    [
        ("dense", [0.1, 0.2], None, "embedding", "COSINE"),
        ("sparse", None, "口干怎么办", "sparse_embedding", "BM25"),
    ],
)
def test_single_search_modes_use_the_expected_field(
    search_method,
    query_vector,
    query_text,
    expected_field,
    expected_metric,
):
    store = make_store()

    store.search(
        query_vector=query_vector,
        query_text=query_text,
        search_method=search_method,
    )

    call = store.client.search_calls[0]
    assert call["anns_field"] == expected_field
    assert call["search_params"]["metric_type"] == expected_metric
    assert call["timeout"] == store.request_timeout_seconds


def test_count_uses_timeout_when_fetching_collection_stats():
    class StatsClient(FakeMilvusClient):
        def get_collection_stats(self, **kwargs):
            self.stats_call = kwargs
            return {"row_count": 7}

    store = MilvusStore(MilvusCollectionSchema(collection_name="test_documents", dimension=2))
    store.client = StatsClient()
    store.create_collection = lambda: None

    assert store.count() == 7
    assert store.client.stats_call["timeout"] == store.request_timeout_seconds


def test_flush_uses_management_timeout():
    store = make_store()

    store.flush()

    assert store.client.flushed[0]["collection_name"] == "test_documents"
    assert store.client.flushed[0]["timeout"] == store.management_timeout_seconds


def test_milvus_errors_are_wrapped_with_operation_context():
    class BrokenClient(FakeMilvusClient):
        def search(self, **kwargs):
            raise TimeoutError("search timed out")

    store = MilvusStore(MilvusCollectionSchema(collection_name="test_documents", dimension=2))
    store.client = BrokenClient()
    store.create_collection = lambda: None

    with pytest.raises(MilvusOperationError, match="dense_search"):
        store.search(query_vector=[0.1, 0.2], search_method="dense")


def test_sparse_retrieval_skips_dense_embedding():
    store = make_store()
    embedder = FakeEmbedder()
    retriever = HybridRetriever(embedder=embedder, store=store)

    retriever.retrieve(query="口干怎么办", search_method="sparse")

    assert embedder.calls == []
    assert store.client.search_calls[0]["anns_field"] == "sparse_embedding"


def test_existing_dense_only_schema_is_rejected():
    class OldSchemaClient:
        def describe_collection(self, **kwargs):
            return {
                "fields": [
                    {"name": "doc_id"},
                    {"name": "embedding"},
                    {"name": "content"},
                ]
            }

    store = MilvusStore(MilvusCollectionSchema(collection_name="old_documents"))
    store.client = OldSchemaClient()

    with pytest.raises(RuntimeError, match="old dense-only schema"):
        store._validate_existing_schema()


def test_reset_collection_refuses_to_drop_without_confirmation():
    class ExistingCollectionClient:
        def __init__(self):
            self.dropped = []

        def has_collection(self, **kwargs):
            return True

        def drop_collection(self, **kwargs):
            self.dropped.append(kwargs["collection_name"])

    class FakeResetStore:
        schema = MilvusCollectionSchema(collection_name="existing")

        def __init__(self):
            self.client = ExistingCollectionClient()
            self.created = False

        def connect(self):
            return self.client

        def create_collection(self):
            self.created = True

    store = FakeResetStore()

    with pytest.raises(RuntimeError, match="--confirm-drop"):
        reset_collection(store)

    assert store.client.dropped == []
    assert store.created is False
